//! Gemma 3 12B as ComfyUI runs it for LTX-2.3's text conditioning
//! (comfy/text_encoders/lt.py + llama.py), starting with its tokenizer.
//!
//! Tokenizer: the SentencePiece model stored inside the text-encoder file ("spiece_model"),
//! called the way ComfyUI's SPieceTokenizer + SDTokenizer call it: the raw prompt (no chat
//! template, no weight syntax), BOS prepended, no EOS, then left-padded with 0 to at least
//! 1024 tokens.

use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;

pub const BOS: u32 = 2;
pub const PAD: u32 = 0;
pub const MIN_LENGTH: usize = 1024;

pub struct GemmaTokenizer {
    sp: SentencePieceProcessor,
}

impl GemmaTokenizer {
    pub fn from_proto(bytes: &[u8]) -> Result<Self> {
        let sp = SentencePieceProcessor::from_serialized_proto(bytes).context("Gemma SentencePiece model")?;
        Ok(GemmaTokenizer { sp })
    }

    /// The real tokens: BOS then the prompt's pieces.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // ComfyUI splits on its two special tokens first; a plain prompt has neither
        for special in ["<image_soft_token>", "<end_of_turn>"] {
            anyhow::ensure!(!text.contains(special), "prompts containing {special} are not handled yet");
        }
        let pieces = self.sp.encode(text).context("SentencePiece encode")?;
        let mut ids = vec![BOS];
        ids.extend(pieces.into_iter().map(|p| p.id));
        Ok(ids)
    }

    /// What ComfyUI hands the model: left padding with 0 up to 1024 tokens.
    pub fn encode_padded(&self, text: &str) -> Result<(Vec<u32>, usize)> {
        let real = self.encode(text)?;
        let n = real.len();
        let mut ids = vec![PAD; MIN_LENGTH.saturating_sub(n)];
        ids.extend(real);
        Ok((ids, n))
    }
}

// ── the model ────────────────────────────────────────────────────────────────
//
// ComfyUI runs Gemma for LTX-AV in f32 on the whole left-padded sequence (1024 tokens for any
// shorter prompt), every op on torch's own kernel: F.linear as cublasSgemm, F.rms_norm as torch's
// fused kernel, RoPE as comfy_kitchen's eager formula (its CUDA kernels take f16 and bf16 only),
// SDPA on the memory-efficient CUTLASS kernel (no fused backend takes grouped K/V with a mask, so
// ComfyUI repeats them to 16 heads first), torch's tanh GELU. The padding is not cosmetic: the
// GEMMs' shapes and the attention's rows are the padded ones, and their sums round accordingly,
// so the real tokens' states come out as ComfyUI's only when the pad tokens run too.
use crate::kernels;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

const HIDDEN: usize = 3840;
const LAYERS: usize = 48;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const EPS: f32 = 1e-6;

pub struct Gemma {
    st: MmapedSafetensors,
    pub tok: GemmaTokenizer,
}

/// A tensor's stored bytes, untouched, as a u8 tensor on `dev`.
fn raw_bytes(st: &MmapedSafetensors, name: &str, dev: &Device) -> Result<Tensor> {
    let view = st.get(name).with_context(|| format!("gemma: no {name}"))?;
    let data = view.data();
    Ok(Tensor::from_raw_buffer(data, DType::U8, &[data.len()], &Device::Cpu)?.to_device(dev)?)
}

/// ComfyUI's attention mask for a left-padded sequence of `t` tokens whose last `real` are the
/// prompt: finfo(f32).min / 4 on the pad keys, plus the same above the diagonal (causal), so a pad
/// key above the diagonal holds min / 2. (1, 1, t, t). Gemma's sliding window (1024) never cuts
/// inside 1024 tokens, which is all ComfyUI's mask then is.
fn attention_mask(t: usize, real: usize, dev: &Device) -> Result<Tensor> {
    let min4 = f32::MIN / 4.0;
    let pad = t - real;
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let a = if j < pad { min4 } else { 0.0 };
            let c = if j > i { min4 } else { 0.0 };
            m[i * t + j] = a + c;
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, t, t), dev)?)
}

/// (B, KV, T, D) -> (B, 2 KV, T, D), each key/value head serving two query heads (repeat_interleave)
fn repeat_kv(x: &Tensor) -> Result<Tensor> {
    let (b, kv, t, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.broadcast_as((b, kv, HEADS / KV_HEADS, t, d))?.contiguous()?.reshape((b, HEADS, t, d))?)
}

/// (T, heads * D) -> (1, heads, T, D), contiguous (view(B, T, H, D).transpose(1, 2), made
/// contiguous by the norm or the repeat that follows in ComfyUI)
fn heads_first(x: &Tensor, heads: usize) -> Result<Tensor> {
    let t = x.dim(0)?;
    Ok(x.reshape((t, heads, HEAD_DIM))?.permute((1, 0, 2))?.contiguous()?.unsqueeze(0)?)
}

impl Gemma {
    pub fn open(path: &Path) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }.with_context(|| format!("open {}", path.display()))?;
        let proto = st.load("spiece_model", &Device::Cpu)?.to_vec1::<u8>()?;
        Ok(Gemma { tok: GemmaTokenizer::from_proto(&proto)?, st })
    }

    /// A linear's weight as ComfyUI's text-encoder forward multiplies by it: dequantised straight
    /// to f32 (no bf16 step), (out, in). The file mixes formats per layer ("fp4_mixed": most layers NVFP4,
    /// some fp8), each named by the layer's `comfy_quant` entry.
    fn linear(&self, p: &str, dev: &Device) -> Result<Tensor> {
        let name = format!("{p}.weight");
        let shape = self.st.get(&name)?.shape().to_vec();
        let format = match self.st.get(&format!("{p}.comfy_quant")) {
            Ok(v) => serde_json::from_slice::<serde_json::Value>(v.data())?["format"].as_str().unwrap_or("").to_string(),
            Err(_) => String::new(),
        };
        match format.as_str() {
            "nvfp4" => {
                let q = raw_bytes(&self.st, &name, dev)?.reshape((shape[0], shape[1]))?;
                let s = raw_bytes(&self.st, &format!("{p}.weight_scale"), dev)?;
                let g = self.st.load(&format!("{p}.weight_scale_2"), dev)?.to_dtype(DType::F32)?.reshape(1)?;
                Ok(crate::kernels::dequant_nvfp4(&q, &s, &g)?)
            }
            "float8_e4m3fn" => {
                let q = raw_bytes(&self.st, &name, dev)?.reshape((shape[0], shape[1]))?;
                let s = self.st.load(&format!("{p}.weight_scale"), dev)?.to_dtype(DType::F32)?.reshape(1)?;
                Ok(crate::kernels::dequant_fp8(&q, &s)?)
            }
            // not quantised: bf16 in the file, widened
            "" => Ok(self.st.load(&name, dev)?.to_dtype(DType::F32)?),
            other => anyhow::bail!("{p}: quantisation {other:?} is not implemented"),
        }
    }

    /// One linear's weight as `linear` makes it, for checking against ComfyUI's.
    pub fn debug_linear(&self, p: &str, dev: &Device) -> Result<Tensor> {
        self.linear(p, dev)
    }

    /// x (T, in) f32 through linear `p`: F.linear as torch runs it in f32
    fn apply(&self, p: &str, x: &Tensor, dev: &Device) -> Result<Tensor> {
        Ok(kernels::torch_linear_f32(x, &self.linear(p, dev)?)?)
    }

    /// A Gemma norm weight as ComfyUI uses it: (w + 1) rounded to bf16, then f32.
    fn norm(&self, name: &str, dev: &Device) -> Result<Tensor> {
        let w = self.st.load(name, dev)?.to_dtype(DType::F32)?;
        Ok((w + 1.0)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)
    }

    fn rms(&self, x: &Tensor, name: &str, dev: &Device) -> Result<Tensor> {
        Ok(kernels::torch_rms_norm_f32(x, &self.norm(name, dev)?, EPS)?)
    }

    /// The embedding rows of `ids` straight from the file, f32, times sqrt(hidden): (T, 3840)
    fn embed(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let view = self.st.get("model.embed_tokens.weight")?;
        let data = view.data();
        let row = HIDDEN * 2;
        let mut bytes = Vec::with_capacity(ids.len() * row);
        for &id in ids {
            bytes.extend_from_slice(&data[id as usize * row..(id as usize + 1) * row]);
        }
        let scale = Tensor::new((HIDDEN as f64).sqrt() as f32, dev)?;
        Ok(Tensor::from_raw_buffer(&bytes, DType::BF16, &[ids.len(), HIDDEN], &Device::Cpu)?
            .to_device(dev)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&scale)?)
    }

    /// Layer `l`'s self-attention on the normed input h (T, 3840): its q, k (normed, rotated), v and
    /// output, before o_proj, (T, 4096)
    fn attention(&self, l: usize, h: &Tensor, mask: &Tensor, (cos, sin): &(Tensor, Tensor), dev: &Device) -> Result<Tensor> {
        let p = format!("model.layers.{l}.self_attn");
        let q = heads_first(&self.apply(&format!("{p}.q_proj"), h, dev)?, HEADS)?;
        let k = heads_first(&self.apply(&format!("{p}.k_proj"), h, dev)?, KV_HEADS)?;
        let v = heads_first(&self.apply(&format!("{p}.v_proj"), h, dev)?, KV_HEADS)?;
        let q = kernels::gemma_rope_f32(&self.rms(&q, &format!("{p}.q_norm.weight"), dev)?, cos, sin)?;
        let k = kernels::gemma_rope_f32(&self.rms(&k, &format!("{p}.k_norm.weight"), dev)?, cos, sin)?;
        let a = kernels::torch_mem_eff_attention_f32(&q, &repeat_kv(&k)?, &repeat_kv(&v)?, Some(mask))?;
        Ok(a.squeeze(0)?)
    }

    /// TransformerBlockGemma2.forward: x (T, 3840) -> (T, 3840)
    fn layer(&self, l: usize, x: &Tensor, mask: &Tensor, rope: &(Tensor, Tensor), dev: &Device) -> Result<Tensor> {
        let p = format!("model.layers.{l}");
        let a = self.attention(l, &self.rms(x, &format!("{p}.input_layernorm.weight"), dev)?, mask, rope, dev)?;
        let o = self.rms(&self.apply(&format!("{p}.self_attn.o_proj"), &a, dev)?, &format!("{p}.post_attention_layernorm.weight"), dev)?;
        let x = (x + o)?;
        let h = self.rms(&x, &format!("{p}.pre_feedforward_layernorm.weight"), dev)?;
        let m = kernels::torch_gelu_tanh_mul_f32(
            &self.apply(&format!("{p}.mlp.gate_proj"), &h, dev)?,
            &self.apply(&format!("{p}.mlp.up_proj"), &h, dev)?,
        )?;
        let f = self.rms(&self.apply(&format!("{p}.mlp.down_proj"), &m, dev)?, &format!("{p}.post_feedforward_layernorm.weight"), dev)?;
        Ok((x + f)?)
    }

    /// Both RoPE tables over the padded positions: global layers (every sixth) take theta 1e6
    /// scaled by 8, sliding ones theta 1e4
    fn ropes(&self, t: usize, dev: &Device) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
        Ok((kernels::gemma_rope_tables(t, HEAD_DIM, 1_000_000.0, 8.0, dev)?, kernels::gemma_rope_tables(t, HEAD_DIM, 10_000.0, 1.0, dev)?))
    }

    /// The 49 hidden states ComfyUI's "all" layer output holds for the real tokens: the input to
    /// each of the 48 layers, then the final-normed output. f32, (49, real, 3840). `ids` is the
    /// whole left-padded sequence, of which the last `real` are the prompt.
    pub fn hidden_states(&self, ids: &[u32], real: usize, dev: &Device) -> Result<Tensor> {
        let t = ids.len();
        anyhow::ensure!(real <= t, "gemma: {real} real tokens of {t}");
        let mut x = self.embed(ids, dev)?;
        let mask = attention_mask(t, real, dev)?;
        let (global, local) = self.ropes(t, dev)?;
        let mut states = vec![];
        for l in 0..LAYERS {
            states.push(x.narrow(0, t - real, real)?);
            x = self.layer(l, &x, &mask, if l % 6 == 5 { &global } else { &local }, dev)?;
        }
        states.push(self.rms(&x, "model.norm.weight", dev)?.narrow(0, t - real, real)?);
        Ok(Tensor::stack(&states, 0)?)
    }

    /// Layers 0 and 5 op by op against refs/gemma_layer_ref.py's capture: each op once on
    /// ComfyUI's input for it, so one op's difference shows alone, then each layer whole
    pub fn debug_layers(&self, refs: &Path, dev: &Device) -> Result<()> {
        let r = unsafe { MmapedSafetensors::new(refs)? };
        let get = |n: &str| -> Result<Tensor> { Ok(r.load(n, dev)?.to_dtype(DType::F32)?) };
        let report = |name: &str, ours: &Tensor, theirs: &Tensor| -> Result<()> {
            let ours = ours.reshape(theirs.dims())?;
            let d = (&ours - theirs)?.abs()?;
            let differ = d.ne(0f32)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            let max = d.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mark = if differ == 0.0 { "exact" } else { "" };
            println!("{name:34} {differ:>9} of {:>9} differ, max |d| {max:.3e} {mark}", theirs.elem_count());
            Ok(())
        };
        let ids: Vec<u32> = r.load("ids", &Device::Cpu)?.to_dtype(DType::I64)?.to_vec1::<i64>()?.iter().map(|&i| i as u32).collect();
        let t = ids.len();
        let real = ids.iter().filter(|&&i| i != PAD).count();
        let mask = attention_mask(t, real, dev)?;
        let (global, local) = self.ropes(t, dev)?;
        for (j, (c, s)) in [&global, &local].into_iter().enumerate() {
            report(&format!("rope{j} cos"), c, &get(&format!("rope{j}.cos"))?.narrow(3, 0, HEAD_DIM / 2)?.squeeze(0)?.squeeze(0)?)?;
            report(&format!("rope{j} sin"), s, &get(&format!("rope{j}.sin"))?.squeeze(0)?.squeeze(0)?)?;
        }
        report("embeddings (layer 0 input)", &self.embed(&ids, dev)?, &get("l0.x")?.squeeze(0)?)?;
        for l in [0usize, 5] {
            let p = format!("model.layers.{l}");
            let c = |n: &str| -> Result<Tensor> { get(&format!("l{l}.{n}")) };
            let rows = |x: Tensor| -> Result<Tensor> { Ok(x.reshape((t, x.elem_count() / t))?) };
            let rope = if l % 6 == 5 { &global } else { &local };
            println!("── layer {l} ({})", if l % 6 == 5 { "global" } else { "sliding" });
            report("mask", &mask, &c("sdpa.mask")?)?;
            let x = rows(c("x")?)?;
            report("input_layernorm", &self.rms(&x, &format!("{p}.input_layernorm.weight"), dev)?, &c("input_norm")?)?;
            let h = rows(c("input_norm")?)?;
            for n in ["q_proj", "k_proj", "v_proj"] {
                report(n, &self.apply(&format!("{p}.self_attn.{n}"), &h, dev)?, &c(n)?)?;
            }
            let qn = self.rms(&heads_first(&rows(c("q_proj")?)?, HEADS)?, &format!("{p}.self_attn.q_norm.weight"), dev)?;
            report("q_norm", &qn, &c("q_norm")?)?;
            let kn = self.rms(&heads_first(&rows(c("k_proj")?)?, KV_HEADS)?, &format!("{p}.self_attn.k_norm.weight"), dev)?;
            report("k_norm", &kn, &c("k_norm")?)?;
            report("rope q", &kernels::gemma_rope_f32(&c("q_norm")?, &rope.0, &rope.1)?, &c("sdpa.q")?)?;
            report("rope k", &kernels::gemma_rope_f32(&c("k_norm")?, &rope.0, &rope.1)?, &c("sdpa.k")?)?;
            let a = kernels::torch_mem_eff_attention_f32(
                &c("sdpa.q")?,
                &repeat_kv(&c("sdpa.k")?)?,
                &repeat_kv(&c("sdpa.v")?)?,
                Some(&c("sdpa.mask")?),
            )?;
            report("attention (mem-efficient)", &a, &c("sdpa.out")?)?;
            report("o_proj", &self.apply(&format!("{p}.self_attn.o_proj"), &rows(c("sdpa.out")?)?, dev)?, &c("o_proj")?)?;
            report(
                "post_attention_layernorm",
                &self.rms(&rows(c("o_proj")?)?, &format!("{p}.post_attention_layernorm.weight"), dev)?,
                &c("post_attn_norm")?,
            )?;
            let x1 = (&x + rows(c("post_attn_norm")?)?)?;
            report(
                "pre_feedforward_layernorm",
                &self.rms(&x1, &format!("{p}.pre_feedforward_layernorm.weight"), dev)?,
                &c("pre_ff_norm")?,
            )?;
            let h2 = rows(c("pre_ff_norm")?)?;
            report("gate_proj", &self.apply(&format!("{p}.mlp.gate_proj"), &h2, dev)?, &c("gate")?)?;
            report("up_proj", &self.apply(&format!("{p}.mlp.up_proj"), &h2, dev)?, &c("up")?)?;
            report("gelu(gate) * up", &kernels::torch_gelu_tanh_mul_f32(&rows(c("gate")?)?, &rows(c("up")?)?)?, &c("down.in")?)?;
            report("down_proj", &self.apply(&format!("{p}.mlp.down_proj"), &rows(c("down.in")?)?, dev)?, &c("down")?)?;
            report(
                "post_feedforward_layernorm",
                &self.rms(&rows(c("down")?)?, &format!("{p}.post_feedforward_layernorm.weight"), dev)?,
                &c("post_ff_norm")?,
            )?;
            report("layer output (their parts)", &(&x1 + rows(c("post_ff_norm")?)?)?, &c("out")?)?;
            report("whole layer, our chain", &self.layer(l, &x, &mask, rope, dev)?, &c("out")?)?;
        }
        let hidden = self.hidden_states(&ids, real, dev)?;
        let want = get("hidden")?.squeeze(0)?;
        for l in [0usize, 1, 2, 6, 12, 24, 36, 47, 48] {
            report(&format!("hidden[{l}]"), &hidden.get(l)?, &want.get(l)?)?;
        }
        Ok(())
    }
}

/// LTX-2.3's DualLinearProjection (the LTX-2.3 text-projection file), in bf16 with PyTorch's rounding
/// after every op: each of the 49 states RMS-normalised per token, laid out hidden-major, then
/// one linear to the video context (4096) and one to the audio context (2048).
pub fn ltx_project(hidden: &Tensor, proj: &MmapedSafetensors, dev: &Device) -> Result<Tensor> {
    ltx_project_as(hidden, proj, dev, true)
}

/// The same projection all in f32 (weights widened), for when ComfyUI does not run it in bf16.
pub fn ltx_project_f32(hidden: &Tensor, proj: &MmapedSafetensors, dev: &Device) -> Result<Tensor> {
    ltx_project_as(hidden, proj, dev, false)
}

/// The two linears' inputs in the bf16 projection, (T, 188160) each as bf16 values in f32
/// (video, audio), for checking the steps before the GEMM on their own.
pub fn ltx_projection_inputs(hidden: &Tensor) -> Result<(Tensor, Tensor)> {
    let bf = |t: Tensor| -> Result<Tensor> { Ok(t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?) };
    let (layers, t, h) = hidden.dims3()?;
    let x = bf(hidden.clone())?.permute((1, 2, 0))?.contiguous()?;
    let mean = bf(bf(x.sqr()?)?.mean_keepdim(1)?)?;
    let r = bf(bf((mean + 1e-6)?)?.sqrt()?.recip()?)?;
    let xn = bf(x.broadcast_mul(&r)?)?.reshape((t, h * layers))?;
    let scaled =
        |dim: usize| -> Result<Tensor> { bf(xn.broadcast_mul(&Tensor::new((dim as f64 / h as f64).sqrt() as f32, hidden.device())?)?) };
    Ok((scaled(4096)?, scaled(2048)?))
}

fn ltx_project_as(hidden: &Tensor, proj: &MmapedSafetensors, dev: &Device, half: bool) -> Result<Tensor> {
    if !half {
        let (layers, t, h) = hidden.dims3()?;
        let x = hidden.permute((1, 2, 0))?.contiguous()?;
        let r = (x.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?.recip()?;
        let xn = x.broadcast_mul(&r)?.reshape((t, h * layers))?;
        let mut outs = vec![];
        for (which, dim) in [("video", 4096usize), ("audio", 2048usize)] {
            let w = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.weight"), dev)?.to_dtype(DType::F32)?;
            let b = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.bias"), dev)?.to_dtype(DType::F32)?;
            let s = Tensor::new((dim as f64 / h as f64).sqrt() as f32, dev)?;
            outs.push(xn.broadcast_mul(&s)?.matmul(&w.t()?)?.broadcast_add(&b)?);
        }
        return Ok(Tensor::cat(&outs, 1)?.unsqueeze(0)?);
    }
    let bf = |t: Tensor| -> Result<Tensor> { Ok(t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?) };
    let (layers, t, h) = hidden.dims3()?;
    // (49, T, H) -> bf16 -> (T, H, 49)
    let x = bf(hidden.clone())?.permute((1, 2, 0))?.contiguous()?;
    let sq = bf(x.sqr()?)?;
    let mean = bf(sq.mean_keepdim(1)?)?; // over H, f32 accumulation then rounded
    let r = bf(bf((mean + 1e-6)?)?.sqrt()?.recip()?)?;
    let xn = bf(x.broadcast_mul(&r)?)?.reshape((t, h * layers))?;
    let mut outs = vec![];
    for (which, dim) in [("video", 4096usize), ("audio", 2048usize)] {
        let w = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.weight"), dev)?;
        let b = proj.load(&format!("text_embedding_projection.{which}_aggregate_embed.bias"), dev)?;
        anyhow::ensure!(w.dim(0)? == dim, "{which} projection is {} wide, not {dim}", w.dim(0)?);
        let s = Tensor::new((dim as f64 / h as f64).sqrt() as f32, dev)?;
        let xs = xn.broadcast_mul(&s)?.to_dtype(DType::BF16)?;
        outs.push(crate::kernels::torch_linear(&xs, &w.to_dtype(DType::BF16)?, &b.to_dtype(DType::BF16)?)?);
    }
    Ok(Tensor::cat(&outs, 1)?.to_dtype(DType::F32)?.unsqueeze(0)?)
}
