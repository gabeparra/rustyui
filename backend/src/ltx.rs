//! LTX-2.3's audio-video DiT, after ComfyUI's comfy/ldm/lightricks/av_model.py and
//! model.py. Activations stay bf16 and every eager PyTorch step rounds where PyTorch's does:
//! linears run as torch runs them (cuBLASLt bf16, or scaled_mm for the fp8 layers), attention is
//! FlashAttention-2 built and split as torch builds and splits it, and the elementwise sequences
//! are ops.cu's ltx_* kernels.

use crate::kernels;
use anyhow::{Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor, D};

const DIT: &str = "model.diffusion_model.";

/// A linear layer as the checkpoint stores it: bf16, or e4m3 bytes with a per-tensor scale
/// (layers that carry a `comfy_quant` entry).
pub enum Linear {
    Bf16 { w: Tensor, b: Tensor },
    Fp8 { w: Tensor, scale: Tensor, one: Tensor, b: Tensor },
}

fn raw_u8(st: &MmapedSafetensors, name: &str, dev: &Device) -> Result<Tensor> {
    let view = st.get(name).with_context(|| format!("ltx: no {name}"))?;
    Ok(kernels::upload(view.data(), DType::U8, view.shape(), dev)?)
}

/// A checkpoint tensor onto the card in its own dtype, through the pinned uploader (the same
/// bytes `st.load` would give, several times faster); other dtypes go the ordinary way
fn fetch(st: &MmapedSafetensors, name: &str, dev: &Device) -> Result<Tensor> {
    let view = st.get(name).with_context(|| format!("ltx: no {name}"))?;
    let dtype = match format!("{:?}", view.dtype()).as_str() {
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "U8" => DType::U8,
        _ => return Ok(st.load(name, dev)?),
    };
    Ok(kernels::upload(view.data(), dtype, view.shape(), dev)?)
}

/// LoRAs over the DiT as LoraLoaderModelOnly stacks them: each file (PEFT `lora_A`/`lora_B` pairs
/// keyed `diffusion_model.<layer>`, no alpha) with its strength, in order
pub struct Loras {
    files: Vec<(MmapedSafetensors, f64, std::path::PathBuf)>,
    /// the card's multiprocessors, which lay out torch's random numbers (rng::CudaRandn)
    sms: u32,
    /// every layer folded so far, in host memory: a block that has to load again (it did not stay
    /// on the card, or the next graph uses the same LoRAs) uploads these instead of folding again.
    /// Folding is deterministic (its random bytes are seeded by the weight's name), so the bytes
    /// are the same either way.
    folded: std::cell::RefCell<std::collections::HashMap<String, Folded>>,
}

/// A layer's weight with its LoRAs folded in: e4m3 bytes and the new scale for an fp8 layer, bf16
/// bytes for the rest
struct Folded {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    scale: Option<f32>,
}

/// comfy.utils.string_to_seed: CRC-32 (0xEDB88320, reflected) of the key's characters
fn string_to_seed(key: &str) -> u64 {
    let mut crc = 0xFFFF_FFFFu32;
    for ch in key.chars() {
        crc ^= ch as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    (crc ^ 0xFFFF_FFFF) as u64
}

impl Loras {
    pub fn open(list: &[(std::path::PathBuf, f64)]) -> Result<Self> {
        let mut files = vec![];
        for (p, s) in list {
            files.push((unsafe { MmapedSafetensors::new(p)? }, *s, p.clone()));
        }
        use candle_core::cuda_backend::cudarc::driver::{sys::CUdevice_attribute, CudaContext};
        let sms = CudaContext::new(0)?.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        Ok(Self { files, sms, folded: Default::default() })
    }

    /// Keeps a folded layer's weight in host memory (`folded`)
    fn keep(&self, layer: &str, lin: &Linear) -> Result<()> {
        let f = match lin {
            Linear::Fp8 { w, scale, .. } => Folded {
                bytes: w.flatten_all()?.to_vec1::<u8>()?,
                shape: w.dims().to_vec(),
                scale: Some(scale.flatten_all()?.to_vec1::<f32>()?[0]),
            },
            Linear::Bf16 { w, .. } => {
                let v = w.flatten_all()?.to_vec1::<half::bf16>()?;
                // the bf16 values' own bytes
                let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }.to_vec();
                Folded { bytes, shape: w.dims().to_vec(), scale: None }
            }
        };
        self.folded.borrow_mut().insert(layer.to_string(), f);
        Ok(())
    }

    /// How many layers have been folded (and kept), and their bytes
    pub fn kept(&self) -> (usize, usize) {
        let f = self.folded.borrow();
        (f.len(), f.values().map(|v| v.bytes.len()).sum())
    }

    /// Each file's layers that name no weight of the checkpoint (a LoRA for another model)
    pub fn unmatched(&self, st: &MmapedSafetensors) -> Vec<(std::path::PathBuf, usize)> {
        let mut out = vec![];
        for (f, _, p) in &self.files {
            let missing = f
                .tensors()
                .iter()
                .filter_map(|(k, _)| k.strip_suffix(".lora_A.weight").map(str::to_string))
                .filter(|layer| st.get(&format!("model.{layer}.weight")).is_err())
                .count();
            if missing > 0 {
                out.push((p.clone(), missing));
            }
        }
        out
    }
}

impl Linear {
    /// The layer `p` with the LoRAs that touch it folded in, as ComfyUI's DynamicVRAM loader
    /// (comfy-aimdo's ModelPatcherDynamic, on by default on NVIDIA and what the reference ComfyUI server runs)
    /// applies them the first time a layer is cast (comfy/ops.py resolve_cast_module_with_vbar): the
    /// weight in the compute dtype, bf16 (an fp8 one dequantized by comfy_kitchen: e4m3 times its
    /// scale, rounded once), each LoRA's mm(B, A) as a bf16 GEMM on cuBLAS added in bf16, in the order
    /// the graph stacks the LoRAs (calculate_weight with the weight's dtype as the intermediate), and
    /// an fp8 layer quantized again with a new scale by ComfyUI's own fp8 layout (requantize_from_float
    /// with scale="recalculate" and the layer's seed: stochastic rounding). The legacy ModelPatcher
    /// (--disable-dynamic-vram) adds the LoRAs in fp16 instead and gets other numbers; the server's
    /// are these.
    pub fn load_lora(st: &MmapedSafetensors, p: &str, dev: &Device, loras: Option<&Loras>) -> Result<Self> {
        let Some(loras) = loras else { return Self::load(st, p, dev) };
        let layer = p.strip_prefix("model.").unwrap_or(p).to_string();
        // folded before: the kept bytes go back onto the card, nothing is folded again
        if let Some(f) = loras.folded.borrow().get(&layer) {
            let b = fetch(st, &format!("{p}.bias"), dev)?;
            return Ok(match f.scale {
                Some(scale) => Linear::Fp8 {
                    w: kernels::upload(&f.bytes, DType::U8, &f.shape, dev)?,
                    scale: Tensor::new(&[scale], dev)?,
                    one: Tensor::new(&[1f32], dev)?,
                    b,
                },
                None => Linear::Bf16 { w: kernels::upload(&f.bytes, DType::BF16, &f.shape, dev)?, b },
            });
        }
        let lin = Self::load(st, p, dev)?;
        let Some(w) = lin.lora_bf16(&layer, dev, loras)? else { return Ok(lin) };
        let out = match lin {
            Linear::Bf16 { b, .. } => Linear::Bf16 { w, b },
            Linear::Fp8 { one, b, .. } => {
                // ComfyUI's TensorCoreFP8E4M3Layout.quantize(scale="recalculate",
                // stochastic_rounding=seed): the bf16 weight's amax in f32 over 448 (a CPU scalar
                // divisor, so times its f32 reciprocal), the weight times bf16(1 / scale) in bf16,
                // then comfy_kitchen's stochastic rounding on torch.randint bytes from a generator
                // seeded with the CRC-32 of the layer's key (its seed_key, without ".weight"); a
                // seed of 0 would mean comfy_kitchen's plain rounding instead
                let amax = w.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                let scale = amax * (1.0f32 / 448.0f32);
                let seed = string_to_seed(&layer);
                let q = if seed == 0 {
                    kernels::ck_quantize_e4m3_bf16(&w, &Tensor::new(&[scale], dev)?)?
                } else {
                    let inv = Tensor::new(&[half::bf16::from_f32(1.0f32 / scale)], dev)?;
                    let scaled = w.broadcast_mul(&inv)?;
                    let rng = crate::rng::CudaRandn::new(seed, loras.sms, 1536).randint_u8_gpu(scaled.elem_count(), dev)?;
                    kernels::ck_stochastic_round_e4m3(&scaled, &rng)?.reshape(scaled.dims())?
                };
                Linear::Fp8 { w: q, scale: Tensor::new(&[scale], dev)?, one, b }
            }
        };
        loras.keep(&layer, &out)?;
        Ok(out)
    }

    /// This layer's weight in bf16 with every LoRA that touches it added, as calculate_weight does
    /// it with a bf16 intermediate, or None when none does; `layer` is its key without "model."
    fn lora_bf16(&self, layer: &str, dev: &Device, loras: &Loras) -> Result<Option<Tensor>> {
        let (a_key, b_key) = (format!("{layer}.lora_A.weight"), format!("{layer}.lora_B.weight"));
        let pairs: Vec<(&MmapedSafetensors, f64)> =
            loras.files.iter().filter(|(f, _, _)| f.get(&a_key).is_ok()).map(|(f, s, _)| (f, *s)).collect();
        if pairs.is_empty() {
            return Ok(None);
        }
        let mut w = match self {
            Linear::Bf16 { w, .. } => w.clone(),
            Linear::Fp8 { w, scale, .. } => kernels::ck_dequant_e4m3_bf16(w, scale)?,
        };
        for (f, strength) in pairs {
            let down = fetch(f, &a_key, dev)?.to_dtype(DType::BF16)?;
            let up = fetch(f, &b_key, dev)?.to_dtype(DType::BF16)?;
            let mut diff = kernels::torch_mm_bf16(&up, &down)?;
            if strength != 1.0 {
                // (strength * alpha) * lora_diff: the bf16 product times a python float, in f32
                diff = (diff.to_dtype(DType::F32)? * strength)?.to_dtype(DType::BF16)?;
            }
            let diff = diff.reshape(w.dims())?;
            w = (w + diff)?;
        }
        Ok(Some(w))
    }

    pub fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        let b = fetch(st, &format!("{p}.bias"), dev)?;
        if st.get(&format!("{p}.comfy_quant")).is_ok() {
            let w = raw_u8(st, &format!("{p}.weight"), dev)?;
            let scale = fetch(st, &format!("{p}.weight_scale"), dev)?.to_dtype(DType::F32)?.reshape(1)?;
            let one = Tensor::new(&[1f32], dev)?;
            Ok(Linear::Fp8 { w, scale, one, b })
        } else {
            Ok(Linear::Bf16 { w: fetch(st, &format!("{p}.weight"), dev)?, b })
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Linear::Bf16 { w, b } => kernels::torch_linear(x, w, b)?,
            Linear::Fp8 { w, scale, one, b } => kernels::fp8_linear(x, w, scale, one, Some(b))?,
        })
    }
}

/// CrossAttention: q/k RMSNorm over all heads (eps 1e-5), split-half RoPE, per-head gates.
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Option<Linear>,
    heads: usize,
}

/// What an attention call did on the way, for checking against ComfyUI's
#[derive(Default)]
pub struct AttnTrace {
    pub q: Option<Tensor>,
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
    pub sdpa: Option<Tensor>,
    pub gate_logits: Option<Tensor>,
}

impl Attention {
    fn load(st: &MmapedSafetensors, p: &str, heads: usize, dev: &Device) -> Result<Self> {
        Self::load_lora(st, p, heads, dev, None)
    }

    fn load_lora(st: &MmapedSafetensors, p: &str, heads: usize, dev: &Device, loras: Option<&Loras>) -> Result<Self> {
        let gate = if st.get(&format!("{p}.to_gate_logits.weight")).is_ok() {
            Some(Linear::load_lora(st, &format!("{p}.to_gate_logits"), dev, loras)?)
        } else {
            None
        };
        Ok(Self {
            to_q: Linear::load_lora(st, &format!("{p}.to_q"), dev, loras)?,
            to_k: Linear::load_lora(st, &format!("{p}.to_k"), dev, loras)?,
            to_v: Linear::load_lora(st, &format!("{p}.to_v"), dev, loras)?,
            to_out: Linear::load_lora(st, &format!("{p}.to_out.0"), dev, loras)?,
            q_norm: fetch(st, &format!("{p}.q_norm.weight"), dev)?,
            k_norm: fetch(st, &format!("{p}.k_norm.weight"), dev)?,
            gate,
            heads,
        })
    }

    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>, pe: Option<&Tensor>, k_pe: Option<&Tensor>) -> Result<Tensor> {
        self.forward_traced(x, context, pe, k_pe, None, None)
    }

    /// With `mask`, a self-attention runs as _attention_with_guide_mask (cuDNN, as ComfyUI's
    /// masked attention does) instead of FlashAttention
    pub fn forward_traced(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        pe: Option<&Tensor>,
        k_pe: Option<&Tensor>,
        mask: Option<&GuideMask>,
        mut trace: Option<&mut AttnTrace>,
    ) -> Result<Tensor> {
        // q and k are rebound, not shadowed, so each step's input goes as the next exists (a
        // shadowed tensor lives to the end of the call)
        let mut q = self.to_q.forward(x)?;
        let ctx = context.unwrap_or(x);
        let mut k = self.to_k.forward(ctx)?;
        let v = self.to_v.forward(ctx)?;
        q = kernels::ltx_rms_norm(&q, Some(&self.q_norm), 1e-5)?;
        k = kernels::ltx_rms_norm(&k, Some(&self.k_norm), 1e-5)?;
        if let Some(pe) = pe {
            q = kernels::ltx_rope(&q, pe)?;
            k = kernels::ltx_rope(&k, k_pe.unwrap_or(pe))?;
        }
        let mut out = match mask {
            Some(m) => kernels::guide_attention(&q, &k, &v, self.heads, m.guide_start, &m.noisy, &m.tracked)?,
            None => sdpa(&q, &k, &v, self.heads)?,
        };
        if let Some(t) = trace.as_deref_mut() {
            t.q = Some(q.clone());
            t.k = Some(k.clone());
            t.v = Some(v.clone());
            t.sdpa = Some(out.clone());
        }
        drop((q, k, v));
        if let Some(g) = &self.gate {
            let logits = g.forward(x)?;
            if let Some(t) = trace {
                t.gate_logits = Some(logits.clone());
            }
            out = kernels::ltx_head_gate(&out, &logits, self.heads)?;
        }
        self.to_out.forward(&out)
    }
}

/// torch's scaled_dot_product_attention on (B, T, heads * dim) inputs, default scale
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, tq, inner) = q.dims3()?;
    let tk = k.dim(1)?;
    let d = inner / heads;
    let scale = (1.0f64 / (d as f64).sqrt()) as f32;
    let out = candle_flash_attn::flash_attn(
        &q.reshape((b, tq, heads, d))?,
        &k.reshape((b, tk, heads, d))?,
        &v.reshape((b, tk, heads, d))?,
        scale,
        false,
    )?;
    Ok(out.reshape((b, tq, inner))?)
}

/// FeedForward: proj, tanh GELU, out (comfy.ops.linear_input_act's eager path)
pub struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Self::load_lora(st, p, dev, None)
    }

    fn load_lora(st: &MmapedSafetensors, p: &str, dev: &Device, loras: Option<&Loras>) -> Result<Self> {
        Ok(Self {
            proj: Linear::load_lora(st, &format!("{p}.net.0.proj"), dev, loras)?,
            out: Linear::load_lora(st, &format!("{p}.net.2"), dev, loras)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_owned(x.clone())
    }

    /// The same, letting go of each input as soon as the next step has its output (the 16384-wide
    /// hidden state is 0.4 GB at 13k tokens, held to the end of one expression it was the step's peak)
    pub fn forward_owned(&self, x: Tensor) -> Result<Tensor> {
        let h = self.proj.forward(&x)?;
        drop(x);
        let g = kernels::ltx_gelu_tanh(&h)?;
        drop(h);
        self.out.forward(&g)
    }
}

/// One BasicAVTransformerBlock
pub struct Block {
    pub attn1: Attention,
    pub attn2: Attention,
    pub audio_attn1: Attention,
    pub audio_attn2: Attention,
    pub audio_to_video_attn: Attention,
    pub video_to_audio_attn: Attention,
    pub ff: FeedForward,
    pub audio_ff: FeedForward,
    sst: Tensor,
    audio_sst: Tensor,
    prompt_sst: Tensor,
    audio_prompt_sst: Tensor,
    a2v_audio: Tensor,
    a2v_video: Tensor,
}

/// A block's inputs other than the two streams (ComfyUI's keyword arguments). Timesteps are ada
/// tensors of one row, one row per latent frame (CompressedTimestep, which the ltx_* kernels
/// repeat over each frame's tokens) or one row per token.
pub struct BlockArgs<'a> {
    pub v_context: &'a Tensor,
    pub a_context: &'a Tensor,
    pub v_timestep: &'a Tensor,
    pub a_timestep: &'a Tensor,
    pub v_pe: &'a Tensor,
    pub a_pe: &'a Tensor,
    pub v_cross_pe: &'a Tensor,
    pub a_cross_pe: &'a Tensor,
    pub v_cross_ss: &'a Tensor,
    pub a_cross_ss: &'a Tensor,
    pub v_cross_gate: &'a Tensor,
    pub a_cross_gate: &'a Tensor,
    pub v_prompt: &'a Tensor,
    pub a_prompt: &'a Tensor,
    /// the video self-attention's guide mask (LTXVAddGuide at a strength other than 1)
    pub v_mask: Option<&'a GuideMask>,
}

/// get_ada_values for one index: table[i] + the timestep's i-th slice, rounded to bf16
fn ada(table: &Tensor, ts: &Tensor, i: usize) -> Result<Tensor> {
    let d = table.dim(1)?;
    let t = ts.narrow(D::Minus1, i * d, d)?;
    Ok(kernels::ltx_add(&t, &table.get(i)?.reshape((1, 1, d))?)?)
}

/// comfy_kitchen's fused rms_norm(x) * (1 + scale) + shift (eps 1e-6), which ComfyUI runs for the
/// video stream and for every text cross-attention's query
fn rms_adaln(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    Ok(kernels::ck_rms_adaln(x, scale, shift, 1e-6)?)
}

/// The same as three PyTorch kernels, as ComfyUI writes it for the audio stream's self-attention
/// and feed-forward
fn rms_modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    Ok(kernels::ltx_modulate(&kernels::ltx_rms_norm(x, None, 1e-6)?, scale, shift)?)
}

impl Block {
    pub fn load(st: &MmapedSafetensors, index: usize, dev: &Device) -> Result<Self> {
        Self::load_lora(st, index, dev, None)
    }

    pub fn load_lora(st: &MmapedSafetensors, index: usize, dev: &Device, loras: Option<&Loras>) -> Result<Self> {
        let p = format!("{DIT}transformer_blocks.{index}.");
        let t = |n: &str| -> Result<Tensor> {
            let x = fetch(st, &format!("{p}{n}"), dev)?;
            Ok(if n.contains("scale_shift_table") { x.to_dtype(DType::BF16)? } else { x })
        };
        let attn = |n: &str| Attention::load_lora(st, &format!("{p}{n}"), 32, dev, loras);
        Ok(Self {
            attn1: attn("attn1")?,
            attn2: attn("attn2")?,
            audio_attn1: attn("audio_attn1")?,
            audio_attn2: attn("audio_attn2")?,
            audio_to_video_attn: attn("audio_to_video_attn")?,
            video_to_audio_attn: attn("video_to_audio_attn")?,
            ff: FeedForward::load_lora(st, &format!("{p}ff"), dev, loras)?,
            audio_ff: FeedForward::load_lora(st, &format!("{p}audio_ff"), dev, loras)?,
            sst: t("scale_shift_table")?,
            audio_sst: t("audio_scale_shift_table")?,
            prompt_sst: t("prompt_scale_shift_table")?,
            audio_prompt_sst: t("audio_prompt_scale_shift_table")?,
            a2v_audio: t("scale_shift_table_a2v_ca_audio")?,
            a2v_video: t("scale_shift_table_a2v_ca_video")?,
        })
    }

    /// apply_cross_attention_adaln: q from the 6..9 ada rows, the context modulated by the
    /// prompt table, the result gated
    fn text_cross(
        x: &Tensor,
        context: &Tensor,
        attn: &Attention,
        sst: &Tensor,
        prompt_sst: &Tensor,
        ts: &Tensor,
        prompt_ts: &Tensor,
    ) -> Result<Tensor> {
        let (shift_q, scale_q, gate) = (ada(sst, ts, 6)?, ada(sst, ts, 7)?, ada(sst, ts, 8)?);
        let (shift_kv, scale_kv) = (ada(prompt_sst, prompt_ts, 0)?, ada(prompt_sst, prompt_ts, 1)?);
        let attn_input = rms_adaln(x, &scale_q, &shift_q)?;
        let ehs = kernels::ltx_modulate(context, &scale_kv, &shift_kv)?;
        Ok(kernels::ltx_mul(&attn.forward(&attn_input, Some(&ehs), None, None)?, &gate)?)
    }

    pub fn forward(&self, vx: &Tensor, ax: &Tensor, a: &BlockArgs) -> Result<(Tensor, Tensor)> {
        // Each stream is rebound, never shadowed, and every intermediate lives in its own scope:
        // a shadowed tensor stays alive to the end of the call, and at 13k video tokens the dead
        // ones came to about 1 GB at the feed-forward. The arithmetic is unchanged.
        // video self-attention and text cross-attention
        let mut vx = {
            let (vshift, vscale) = (ada(&self.sst, a.v_timestep, 0)?, ada(&self.sst, a.v_timestep, 1)?);
            let attn1_out = self.attn1.forward_traced(&rms_adaln(vx, &vscale, &vshift)?, None, Some(a.v_pe), None, a.v_mask, None)?;
            kernels::ltx_addcmul(vx, &attn1_out, &ada(&self.sst, a.v_timestep, 2)?)?
        };
        let cross = Self::text_cross(&vx, a.v_context, &self.attn2, &self.sst, &self.prompt_sst, a.v_timestep, a.v_prompt)?;
        vx = kernels::ltx_add(&vx, &cross)?;
        drop(cross);

        // audio self-attention and text cross-attention
        let mut ax = {
            let (ashift, ascale) = (ada(&self.audio_sst, a.a_timestep, 0)?, ada(&self.audio_sst, a.a_timestep, 1)?);
            let attn1_out = self.audio_attn1.forward(&rms_modulate(ax, &ascale, &ashift)?, None, Some(a.a_pe), None)?;
            kernels::ltx_addcmul(ax, &attn1_out, &ada(&self.audio_sst, a.a_timestep, 2)?)?
        };
        let cross =
            Self::text_cross(&ax, a.a_context, &self.audio_attn2, &self.audio_sst, &self.audio_prompt_sst, a.a_timestep, a.a_prompt)?;
        ax = kernels::ltx_add(&ax, &cross)?;
        drop(cross);

        // audio to video, then video to audio (on the updated video)
        let ax_norm3 = kernels::ltx_rms_norm(&ax, None, 1e-6)?;
        let a_ss = self.a2v_audio.narrow(0, 0, 4)?;
        let v_ss = self.a2v_video.narrow(0, 0, 4)?;
        let a2v = {
            let (scale_a, shift_a) = (ada(&a_ss, a.a_cross_ss, 0)?, ada(&a_ss, a.a_cross_ss, 1)?);
            let (scale_v, shift_v) = (ada(&v_ss, a.v_cross_ss, 0)?, ada(&v_ss, a.v_cross_ss, 1)?);
            let vx_scaled = rms_adaln(&vx, &scale_v, &shift_v)?;
            let ax_scaled = kernels::ltx_modulate(&ax_norm3, &scale_a, &shift_a)?;
            self.audio_to_video_attn.forward(&vx_scaled, Some(&ax_scaled), Some(a.v_cross_pe), Some(a.a_cross_pe))?
        };
        vx = kernels::ltx_addcmul(&vx, &a2v, &ada(&self.a2v_video.narrow(0, 4, 1)?, a.v_cross_gate, 0)?)?;
        drop(a2v);

        let v2a = {
            let (scale_a, shift_a) = (ada(&a_ss, a.a_cross_ss, 2)?, ada(&a_ss, a.a_cross_ss, 3)?);
            let (scale_v, shift_v) = (ada(&v_ss, a.v_cross_ss, 2)?, ada(&v_ss, a.v_cross_ss, 3)?);
            let ax_scaled = kernels::ltx_modulate(&ax_norm3, &scale_a, &shift_a)?;
            let vx_scaled = rms_adaln(&vx, &scale_v, &shift_v)?;
            self.video_to_audio_attn.forward(&ax_scaled, Some(&vx_scaled), Some(a.a_cross_pe), Some(a.v_cross_pe))?
        };
        drop(ax_norm3);
        ax = kernels::ltx_addcmul(&ax, &v2a, &ada(&self.a2v_audio.narrow(0, 4, 1)?, a.a_cross_gate, 0)?)?;
        drop(v2a);

        // feed-forwards
        let ff_out = {
            let (vshift, vscale) = (ada(&self.sst, a.v_timestep, 3)?, ada(&self.sst, a.v_timestep, 4)?);
            self.ff.forward_owned(rms_adaln(&vx, &vscale, &vshift)?)?
        };
        vx = kernels::ltx_addcmul(&vx, &ff_out, &ada(&self.sst, a.v_timestep, 5)?)?;
        drop(ff_out);
        let ff_out = {
            let (ashift, ascale) = (ada(&self.audio_sst, a.a_timestep, 3)?, ada(&self.audio_sst, a.a_timestep, 4)?);
            self.audio_ff.forward_owned(rms_modulate(&ax, &ascale, &ashift)?)?
        };
        ax = kernels::ltx_addcmul(&ax, &ff_out, &ada(&self.audio_sst, a.a_timestep, 5)?)?;
        Ok((vx, ax))
    }
}

// ── checking against ComfyUI (dit_ref2.py's capture) ────────────────────

pub(crate) fn compare(what: &str, ours: &Tensor, theirs: &Tensor) -> Result<()> {
    if ours.dims() != theirs.dims() {
        println!("{what:44} shape {:?} vs ComfyUI's {:?}", ours.dims(), theirs.dims());
        return Ok(());
    }
    let (o, t) = (ours.to_dtype(DType::F32)?, theirs.to_dtype(DType::F32)?);
    let differ = o.ne(&t)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    let maxd = (&o - &t)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let scale = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mark = if differ == 0.0 { "exact" } else { "" };
    println!("{what:44} {differ:>9} of {:>9} differ, max |d| {maxd:.3e} (max |x| {scale:.3e}) {mark}", o.elem_count());
    Ok(())
}

/// GuideAttentionMask's two biases for `total` tokens whose last ones, from `guide_start`, are
/// guide tokens weighted `weights` (each guide token's strength): log(w) in bf16 (finfo.min for
/// w <= 0) on the guide keys of every noisy query (1, 1, 1, total), and on the noisy keys of each
/// guide query (1, 1, tracked, total). Zero everywhere else.
pub fn guide_masks(total: usize, guide_start: usize, weights: &[f64], dev: &Device) -> Result<(Tensor, Tensor)> {
    let g = weights.len();
    anyhow::ensure!(guide_start + g <= total, "guide_masks: {g} guide tokens from {guide_start} of {total}");
    // torch.full(..., strength, dtype=bf16) (through a float), then torch.log in f32, rounded to bf16
    let log_w: Vec<half::bf16> = weights
        .iter()
        .map(|&w| {
            let w = half::bf16::from_f32(w as f32);
            if w.to_f32() > 0.0 {
                half::bf16::from_f32(w.to_f32().max(f32::from_bits(0x0080_0000)).ln())
            } else {
                half::bf16::MIN
            }
        })
        .collect();
    let zero = half::bf16::ZERO;
    let mut noisy = vec![zero; total];
    noisy[guide_start..guide_start + g].copy_from_slice(&log_w);
    let mut tracked = vec![zero; g * total];
    for (i, w) in log_w.iter().enumerate() {
        tracked[i * total..i * total + guide_start].fill(*w);
    }
    Ok((Tensor::from_vec(noisy, (1, 1, 1, total), dev)?, Tensor::from_vec(tracked, (1, 1, g, total), dev)?))
}

/// One LTXVAddGuide guide on the latent being sampled: its latent frames, appended at the end of
/// the video latent, the positions they take (keyframe_idxs) and how strongly they attend
/// (guide_attention_entries). Conditioning carries these, in the order they were added.
#[derive(Clone, Debug, PartialEq)]
pub struct Guide {
    /// latent frames, rows and columns
    pub frames: usize,
    pub h: usize,
    pub w: usize,
    /// the pixel frame its first latent frame stands for
    pub frame_idx: i64,
    /// latent_to_pixel_coords' causal_fix for its frames (a lone first frame covers one pixel frame)
    pub causal_fix: bool,
    pub strength: f64,
}

impl Guide {
    pub fn tokens(&self) -> usize {
        self.frames * self.h * self.w
    }

    /// keyframe_idxs: each token's (start, end) pixel coordinates in time, rows and columns
    pub fn coords(&self) -> impl Iterator<Item = [[i64; 2]; 3]> + '_ {
        let t = move |g: i64| -> i64 {
            if self.causal_fix {
                (8 * g + 1 - 8).max(0)
            } else {
                8 * g
            }
        };
        (0..self.frames as i64).flat_map(move |g| {
            (0..self.h as i64).flat_map(move |hi| {
                (0..self.w as i64)
                    .map(move |wi| [[t(g) + self.frame_idx, t(g + 1) + self.frame_idx], [32 * hi, 32 * (hi + 1)], [32 * wi, 32 * (wi + 1)]])
            })
        })
    }
}

/// GuideAttentionMask: where the guide tokens start, and the two biases (guide_masks)
pub struct GuideMask {
    pub guide_start: usize,
    pub noisy: Tensor,
    pub tracked: Tensor,
}

/// _build_guide_self_attention_mask for `total` video tokens ending in the guides' tokens: none
/// when every guide's strength is 1 (as bf16 weights too)
pub fn guide_mask(total: usize, guides: &[Guide], dev: &Device) -> Result<Option<GuideMask>> {
    let n: usize = guides.iter().map(Guide::tokens).sum();
    if n == 0 || !guides.iter().any(|g| g.strength != 1.0) {
        return Ok(None);
    }
    let weights: Vec<f64> = guides.iter().flat_map(|g| std::iter::repeat_n(g.strength, g.tokens())).collect();
    if weights.iter().all(|&w| half::bf16::from_f32(w as f32).to_f32() == 1.0) {
        return Ok(None);
    }
    anyhow::ensure!(n <= total, "{n} guide tokens in a latent of {total}");
    let (noisy, tracked) = guide_masks(total, total - n, &weights, dev)?;
    Ok(Some(GuideMask { guide_start: total - n, noisy, tracked }))
}

/// debug-guide-attn REF: the guide-masked self-attention (kernels::guide_attention on cuDNN)
/// against ComfyUI's _attention_with_guide_mask on the same q, k and v (refs/guide_attn_ref.py)
pub fn debug_guide_attn(reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    for case in ["a", "b", "c"] {
        let Ok(q) = get(&format!("{case}.q")) else { continue };
        let (k, v) = (get(&format!("{case}.k"))?, get(&format!("{case}.v"))?);
        let (noisy, tracked) = (get(&format!("{case}.noisy_mask"))?, get(&format!("{case}.tracked_mask"))?);
        let strength = get(&format!("{case}.strength"))?.to_vec1::<f64>()?[0];
        let (t, g) = (q.dim(1)?, tracked.dim(2)?);
        let (ours_n, ours_t) = guide_masks(t, t - g, &vec![strength; g], &dev)?;
        compare(&format!("{case}: noisy mask"), &ours_n, &noisy)?;
        compare(&format!("{case}: tracked mask"), &ours_t, &tracked)?;
        let t0 = std::time::Instant::now();
        let out = kernels::guide_attention(&q, &k, &v, 32, t - g, &ours_n, &ours_t)?;
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        compare(&format!("{case}: attention, {t} tokens, {g} guides, {strength}"), &out, &get(&format!("{case}.out"))?)?;
        println!("{case}: {:.1} ms (the first call builds the cuDNN plans)", secs * 1e3);
    }
    Ok(())
}

/// debug-ltx-block CHECKPOINT REF2: each block piece fed ComfyUI's own inputs, then whole blocks
pub fn debug_block(ckpt: &str, reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    for bi in [0usize, 2] {
        let blk = Block::load(&st, bi, &dev)?;
        let b = format!("b{bi}");
        println!("── block {bi} ({})", if matches!(blk.attn1.to_q, Linear::Fp8 { .. }) { "fp8" } else { "bf16" });
        for (sub, attn) in [
            ("attn1", &blk.attn1),
            ("attn2", &blk.attn2),
            ("audio_attn1", &blk.audio_attn1),
            ("audio_attn2", &blk.audio_attn2),
            ("audio_to_video_attn", &blk.audio_to_video_attn),
            ("video_to_audio_attn", &blk.video_to_audio_attn),
        ] {
            let x = get(&format!("{b}.{sub}.in"))?;
            let ctx = get(&format!("{b}.{sub}.in_context")).ok();
            let pe = get(&format!("{b}.{sub}.in_pe.0")).ok();
            let k_pe = get(&format!("{b}.{sub}.in_k_pe.0")).ok();
            let mut tr = AttnTrace::default();
            let out = attn.forward_traced(&x, ctx.as_ref(), pe.as_ref(), k_pe.as_ref(), None, Some(&mut tr))?;
            for (name, t) in [("q", &tr.q), ("k", &tr.k), ("v", &tr.v), ("out", &tr.sdpa)] {
                compare(&format!("{b}.{sub} sdpa {name}"), t.as_ref().unwrap(), &get(&format!("{b}.{sub}.sdpa0.{name}"))?)?;
            }
            if let Some(gl) = &tr.gate_logits {
                compare(&format!("{b}.{sub} gate logits"), gl, &get(&format!("{b}.{sub}.gate_logits"))?)?;
            }
            compare(&format!("{b}.{sub} output"), &out, &get(&format!("{b}.{sub}.out"))?)?;
            // the attention alone, on ComfyUI's own q, k, v
            let alone =
                sdpa(&get(&format!("{b}.{sub}.sdpa0.q"))?, &get(&format!("{b}.{sub}.sdpa0.k"))?, &get(&format!("{b}.{sub}.sdpa0.v"))?, 32)?;
            compare(&format!("{b}.{sub} sdpa on their q,k,v"), &alone, &get(&format!("{b}.{sub}.sdpa0.out"))?)?;
        }
        for (sub, ff) in [("ff", &blk.ff), ("audio_ff", &blk.audio_ff)] {
            compare(&format!("{b}.{sub} output"), &ff.forward(&get(&format!("{b}.{sub}.in"))?)?, &get(&format!("{b}.{sub}.out"))?)?;
        }
        let (vx, ax) = (get(&format!("{b}.in.x.0"))?, get(&format!("{b}.in.x.1"))?);
        let n = |k: &str| get(&format!("{b}.in.{k}"));
        let (v_ctx, a_ctx, v_ts, a_ts) = (n("v_context")?, n("a_context")?, n("v_timestep.data")?, n("a_timestep")?);
        let (v_pe, a_pe, v_cpe, a_cpe) = (n("v_pe.0")?, n("a_pe.0")?, n("v_cross_pe.0")?, n("a_cross_pe.0")?);
        let (v_css, a_css) = (n("v_cross_scale_shift_timestep.data")?, n("a_cross_scale_shift_timestep")?);
        let (v_cg, a_cg) = (n("v_cross_gate_timestep.data")?, n("a_cross_gate_timestep")?);
        let (v_pr, a_pr) = (n("v_prompt_timestep")?, n("a_prompt_timestep")?);
        let args = BlockArgs {
            v_context: &v_ctx,
            a_context: &a_ctx,
            v_timestep: &v_ts,
            a_timestep: &a_ts,
            v_pe: &v_pe,
            a_pe: &a_pe,
            v_cross_pe: &v_cpe,
            a_cross_pe: &a_cpe,
            v_cross_ss: &v_css,
            a_cross_ss: &a_css,
            v_cross_gate: &v_cg,
            a_cross_gate: &a_cg,
            v_prompt: &v_pr,
            a_prompt: &a_pr,
            v_mask: None,
        };
        let (ov, oa) = blk.forward(&vx, &ax, &args)?;
        compare(&format!("{b} whole block, video"), &ov, &get(&format!("{b}.out.0"))?)?;
        compare(&format!("{b} whole block, audio"), &oa, &get(&format!("{b}.out.1"))?)?;
    }
    Ok(())
}

// ── the DiT around its blocks: timestep embeddings, patchify, positions, output head ──────────

/// AdaLayerNormSingle: the 256-wide sinusoid of the (already x1000) timestep, linear_1, SiLU,
/// linear_2 (the embedded timestep), then SiLU and the k-row modulation linear.
pub struct AdaLnSingle {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

/// get_timestep_embedding's exponents: -ln(10000) * k, then / 128, as torch's f32 kernels round them
fn sinusoid_exponents(dev: &Device) -> Result<Tensor> {
    let neg_log = (-(10000f64.ln())) as f32;
    let inv = 1.0f32 / 128.0;
    let e: Vec<f32> = (0..128).map(|k| (neg_log * k as f32) * inv).collect();
    Ok(Tensor::new(e.as_slice(), dev)?)
}

impl AdaLnSingle {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            l1: Linear::load(st, &format!("{p}.emb.timestep_embedder.linear_1"), dev)?,
            l2: Linear::load(st, &format!("{p}.emb.timestep_embedder.linear_2"), dev)?,
            out: Linear::load(st, &format!("{p}.linear"), dev)?,
        })
    }

    /// (modulation (n, k * D), embedded timestep (n, D)) for f32 timesteps (n,)
    pub fn forward(&self, t: &Tensor, expo: &Tensor) -> Result<(Tensor, Tensor)> {
        let sinus = kernels::ltx_timestep_sinusoid(t, expo)?;
        let emb = self.l2.forward(&kernels::ltx_silu(&self.l1.forward(&sinus)?)?)?;
        Ok((self.out.forward(&kernels::ltx_silu(&emb)?)?, emb))
    }
}

/// The timestep tensors one DiT call hands its blocks and output head
pub struct Timesteps {
    pub v: Tensor,
    pub a: Tensor,
    pub a_cross_ss: Tensor,
    pub v_cross_ss: Tensor,
    pub v_cross_gate: Tensor,
    pub a_cross_gate: Tensor,
    pub v_prompt: Tensor,
    pub a_prompt: Tensor,
    pub v_embedded: Tensor,
    pub a_embedded: Tensor,
}

/// The rotation matrices: self-attention for video and audio, and the time-only pair the
/// audio/video cross attentions use
pub struct Positions {
    pub v: Tensor,
    pub a: Tensor,
    pub v_cross: Tensor,
    pub a_cross: Tensor,
}

/// generate_freq_grid_np: theta ** linspace(0, 1, inner_dim // (2 * dims)) * pi / 2 in f64, then f32
fn freq_grid(dims: usize, inner_dim: usize) -> Vec<f32> {
    let n = inner_dim / (2 * dims);
    let start = 1f64.ln() / 10000f64.ln();
    let stop = 10000f64.ln() / 10000f64.ln();
    let step = (stop - start) / (n - 1) as f64;
    (0..n)
        .map(|k| {
            let y = if k == n - 1 { stop } else { k as f64 * step + start };
            (10000f64.powf(y) * std::f64::consts::PI / 2.0) as f32
        })
        .collect()
}

/// _precompute_freqs_cis (split mode, middle of each token's span): coords (tokens, dims, 2) f32
/// start/end positions, already in seconds where torch scales them; max_pos per dim.
fn rope(coords: &[[[f32; 2]; 3]], dims: usize, max_pos: &[f32], inner_dim: usize, heads: usize, dev: &Device) -> Result<Tensor> {
    let idx = freq_grid(dims, inner_dim);
    let inv_max: Vec<f32> = max_pos.iter().map(|m| 1.0f32 / m).collect();
    let f = idx.len() * dims;
    let mut freqs = Vec::with_capacity(coords.len() * f);
    for c in coords {
        let frac: Vec<f32> = (0..dims).map(|d| ((c[d][0] + c[d][1]) * 0.5) * inv_max[d]).collect();
        for &ix in &idx {
            for fr in &frac {
                freqs.push(ix * (fr * 2.0 - 1.0));
            }
        }
    }
    let pad = inner_dim / 2 - f;
    let m = kernels::ltx_rope_matrix(&Tensor::from_vec(freqs, (coords.len(), f), dev)?, pad)?;
    Ok(m.reshape((1, coords.len(), heads, inner_dim / 2 / heads, 2, 2))?)
}

/// The bytes of the text connectors' weights in a checkpoint (what loading them takes on the card)
pub fn connector_bytes(st: &MmapedSafetensors) -> usize {
    st.tensors().iter().filter(|(n, _)| n.contains("embeddings_connector")).map(|(_, v)| v.data().len()).sum()
}

/// The DiT's weights outside the blocks
pub struct Dit {
    patchify_proj: Linear,
    audio_patchify_proj: Linear,
    adaln_single: AdaLnSingle,
    prompt_adaln_single: AdaLnSingle,
    audio_adaln_single: AdaLnSingle,
    audio_prompt_adaln_single: AdaLnSingle,
    av_ca_video_ss: AdaLnSingle,
    av_ca_a2v_gate: AdaLnSingle,
    av_ca_audio_ss: AdaLnSingle,
    av_ca_v2a_gate: AdaLnSingle,
    scale_shift_table: Tensor,
    audio_scale_shift_table: Tensor,
    proj_out: Linear,
    audio_proj_out: Linear,
    expo: Tensor,
}

impl Dit {
    pub fn load(st: &MmapedSafetensors, dev: &Device) -> Result<Self> {
        let a = |n: &str| AdaLnSingle::load(st, &format!("{DIT}{n}"), dev);
        Ok(Self {
            patchify_proj: Linear::load(st, &format!("{DIT}patchify_proj"), dev)?,
            audio_patchify_proj: Linear::load(st, &format!("{DIT}audio_patchify_proj"), dev)?,
            adaln_single: a("adaln_single")?,
            prompt_adaln_single: a("prompt_adaln_single")?,
            audio_adaln_single: a("audio_adaln_single")?,
            audio_prompt_adaln_single: a("audio_prompt_adaln_single")?,
            av_ca_video_ss: a("av_ca_video_scale_shift_adaln_single")?,
            av_ca_a2v_gate: a("av_ca_a2v_gate_adaln_single")?,
            av_ca_audio_ss: a("av_ca_audio_scale_shift_adaln_single")?,
            av_ca_v2a_gate: a("av_ca_v2a_gate_adaln_single")?,
            scale_shift_table: st.load(&format!("{DIT}scale_shift_table"), dev)?.to_dtype(DType::BF16)?,
            audio_scale_shift_table: st.load(&format!("{DIT}audio_scale_shift_table"), dev)?.to_dtype(DType::BF16)?,
            proj_out: Linear::load(st, &format!("{DIT}proj_out"), dev)?,
            audio_proj_out: Linear::load(st, &format!("{DIT}audio_proj_out"), dev)?,
            expo: sinusoid_exponents(dev)?,
        })
    }

    /// _prepare_timestep. `v_t` and `a_t` are what process_timestep hands over, f32: a sigma (1,)
    /// when nothing is masked, else sigma times the denoise mask, one per token (video F*H*W in
    /// frame-major order, audio T). `ppf` is H*W; `spatial` is has_spatial_mask (a frame whose
    /// mask varies), which turns the per-frame rows off. Each AdaLN runs on as many rows as
    /// ComfyUI's does (one per frame, one per token or one), since cuBLAS may pick another kernel
    /// for another row count; per-token video rows are then kept one per frame as
    /// CompressedTimestep keeps them. `batch` samples share these timesteps (CFG's negative and
    /// prompt in one call): every AdaLN then runs on all their rows at once, sample by sample, and
    /// each result is (batch, rows, width).
    pub fn timesteps(&self, v_t: &Tensor, a_t: &Tensor, ppf: usize, spatial: bool, batch: usize) -> Result<Timesteps> {
        // timestep * 1000 (a CPU-scalar multiply, one f32 rounding)
        let v = (v_t.flatten_all()? * 1000.0)?;
        let a = (a_t.flatten_all()? * 1000.0)?;
        let (n_v, n_a) = (v.elem_count(), a.elem_count());
        let per_frame = !spatial && n_v % ppf == 0;
        // one sample's rows, then the next's
        let tile = |t: &Tensor| -> Result<Tensor> { Ok(if batch == 1 { t.clone() } else { t.repeat(batch)? }) };
        let row = |t: Tensor| -> Result<Tensor> {
            let d = t.dim(D::Minus1)?;
            Ok(t.reshape((batch, (), d))?)
        };
        // CompressedTimestep from per-token rows: the first token's row of each frame
        let compress = |t: Tensor| -> Result<Tensor> {
            let (n, d) = t.dims2()?;
            let per = n / batch;
            if !spatial && ppf > 1 && per >= ppf && per.is_multiple_of(ppf) {
                Ok(t.reshape((n / ppf, ppf, d))?.narrow(1, 0, 1)?.contiguous()?.reshape((batch, per / ppf, d))?)
            } else {
                row(t)
            }
        };
        // the per-frame path projects one timestep per frame (the frame's first token, * 1000)
        let v_in = if per_frame && n_v > 1 {
            (v_t.flatten_all()?.reshape((n_v / ppf, ppf))?.narrow(1, 0, 1)?.contiguous()?.flatten_all()? * 1000.0)?
        } else {
            v.clone()
        };
        let (v_ada, v_emb) = self.adaln_single.forward(&tile(&v_in)?, &self.expo)?;
        let (v_ada, v_emb) = if per_frame { (row(v_ada)?, row(v_emb)?) } else { (compress(v_ada)?, compress(v_emb)?) };
        // compute_prompt_timestep: each sample's largest timestep, one row each
        let (v_max, a_max) = (v.max(0)?.reshape(1)?, a.max(0)?.reshape(1)?);
        let (v_prompt, _) = self.prompt_adaln_single.forward(&tile(&v_max)?, &self.expo)?;
        let (a_ss, _) = self.av_ca_audio_ss.forward(&tile(&a)?, &self.expo)?;
        let (v_ss, _) = self.av_ca_video_ss.forward(&tile(&v)?, &self.expo)?;
        // the gates take the other stream's max timestep (over the whole batch, which shares
        // them), one per token of this one, times av_ca_factor (1000 / 1000)
        let (v_gate, _) = self.av_ca_a2v_gate.forward(&(a_max.broadcast_as(batch * n_v)?.contiguous()? * 1.0)?, &self.expo)?;
        let (a_gate, _) = self.av_ca_v2a_gate.forward(&(v_max.broadcast_as(batch * n_a)?.contiguous()? * 1.0)?, &self.expo)?;
        let (a_ada, a_emb) = self.audio_adaln_single.forward(&tile(&a)?, &self.expo)?;
        let (a_prompt, _) = self.audio_prompt_adaln_single.forward(&tile(&a_max)?, &self.expo)?;
        Ok(Timesteps {
            v: v_ada,
            a: row(a_ada)?,
            a_cross_ss: row(a_ss)?,
            v_cross_ss: compress(v_ss)?,
            v_cross_gate: compress(v_gate)?,
            a_cross_gate: row(a_gate)?,
            v_prompt: row(v_prompt)?,
            a_prompt: row(a_prompt)?,
            v_embedded: v_emb,
            a_embedded: row(a_emb)?,
        })
    }

    /// Patchify (patch size 1) and project: video (1, C, F, H, W) -> (1, F*H*W, 4096), audio
    /// (1, C, T, F) -> (1, T, 2048). einops leaves the video a permuted view, which F.linear
    /// multiplies on plain cuBLAS and then adds the bias to; the audio rearrange copies, so its
    /// linear fuses the bias like the rest. With guides (`keyframes`), _process_input picks the
    /// tokens through its grid mask first, a copy, so the video's linear fuses its bias too.
    pub fn patchify(&self, v: &Tensor, a: &Tensor, keyframes: bool) -> Result<(Tensor, Tensor)> {
        let (b, c, f, h, w) = v.dims5()?;
        anyhow::ensure!(b == 1 || !keyframes, "ltx: guides with a batch of {b} (CFG above 1 with LTXVAddGuide) are not implemented yet");
        let vx = match &self.patchify_proj {
            Linear::Bf16 { w: wt, b: bias } if keyframes => {
                kernels::torch_linear(&v.reshape((c, f * h * w))?.t()?.contiguous()?, wt, bias)?
            }
            Linear::Bf16 { w: wt, b: bias } if b == 1 => kernels::torch_linear_channels_first(&v.reshape((c, b * f * h * w))?, wt, bias)?,
            // a batch of permuted views: matmul's reshape copies them into (b * tokens, c), mm on
            // plain cuBLAS, then linear's separate add_ of the bias
            Linear::Bf16 { w: wt, b: bias } => {
                let x = v.reshape((b, c, f * h * w))?.transpose(1, 2)?.contiguous()?.reshape((b * f * h * w, c))?;
                kernels::ltx_add(&kernels::torch_linear_nobias(&x, wt)?, bias)?
            }
            Linear::Fp8 { .. } => anyhow::bail!("ltx: an fp8 patchify_proj is not handled"),
        };
        let vx = vx.reshape((b, f * h * w, ()))?;
        let (ab, ac, at, af) = a.dims4()?;
        let ax = a.permute((0, 2, 1, 3))?.reshape((ab, at, ac * af))?;
        Ok((vx, self.audio_patchify_proj.forward(&ax)?))
    }

    /// Positions for a (F, H, W) latent video at `fps` and `audio_len` audio latents; the last
    /// frames, the guides', take the guides' own positions (keyframe_idxs)
    pub fn positions(&self, f: usize, h: usize, w: usize, fps: f64, audio_len: usize, guides: &[Guide], dev: &Device) -> Result<Positions> {
        // latent corners scaled by the VAE's (8, 32, 32), the first frame's causal fix, time / fps
        let inv_fps = (1.0 / fps) as f32;
        let mut v = Vec::with_capacity(f * h * w);
        for fi in 0..f {
            let t0 = ((8 * fi) as i64 + 1 - 8).max(0) as f32 * inv_fps;
            let t1 = ((8 * (fi + 1)) as i64 + 1 - 8).max(0) as f32 * inv_fps;
            for hi in 0..h {
                for wi in 0..w {
                    v.push([[t0, t1], [(32 * hi) as f32, (32 * (hi + 1)) as f32], [(32 * wi) as f32, (32 * (wi + 1)) as f32]]);
                }
            }
        }
        let n: usize = guides.iter().map(Guide::tokens).sum();
        if n > 0 {
            if guides.iter().any(|g| g.h != h || g.w != w) || n > v.len() {
                anyhow::bail!("ltx: guides {guides:?} do not fit a {f}x{h}x{w} latent (crop the guides before resizing it)");
            }
            let start = v.len() - n;
            for (slot, c) in v[start..].iter_mut().zip(guides.iter().flat_map(Guide::coords)) {
                *slot = [
                    [c[0][0] as f32 * inv_fps, c[0][1] as f32 * inv_fps],
                    [c[1][0] as f32, c[1][1] as f32],
                    [c[2][0] as f32, c[2][1] as f32],
                ];
            }
        }
        // audio latents in seconds: mel frame 4k, causal (+1 - 4, at least 0), * 160 / 16000
        let inv_sr = 1.0f32 / 16000.0;
        let secs = |k: usize| -> f32 { (((4 * k) as i64 + 1 - 4).max(0) as f32 * 160.0) * inv_sr };
        let a: Vec<[[f32; 2]; 3]> = (0..audio_len).map(|k| [[secs(k), secs(k + 1)], [0.0; 2], [0.0; 2]]).collect();
        Ok(Positions {
            v: rope(&v, 3, &[20.0, 2048.0, 2048.0], 4096, 32, dev)?,
            a: rope(&a, 1, &[20.0], 2048, 32, dev)?,
            v_cross: rope(&v, 1, &[20.0], 2048, 32, dev)?,
            a_cross: rope(&a, 1, &[20.0], 2048, 32, dev)?,
        })
    }

    /// _process_output: LayerNorm (no affine, eps 1e-6), modulate by the 2-row table plus the
    /// embedded timestep, project, unpatchify to (1, 128, F, H, W) and (1, 8, T, 16)
    pub fn output(&self, vx: &Tensor, ax: &Tensor, ts: &Timesteps, f: usize, h: usize, w: usize) -> Result<(Tensor, Tensor)> {
        let head = |x: &Tensor, table: &Tensor, emb: &Tensor, proj: &Linear| -> Result<Tensor> {
            // emb first: it may hold a row per frame or per token, which the table row repeats over
            let shift = kernels::ltx_add(emb, &table.get(0)?.reshape((1, 1, ()))?)?;
            let scale = kernels::ltx_add(emb, &table.get(1)?.reshape((1, 1, ()))?)?;
            proj.forward(&kernels::ltx_modulate(&kernels::ltx_layer_norm(x, 1e-6)?, &scale, &shift)?)
        };
        let v = head(vx, &self.scale_shift_table, &ts.v_embedded, &self.proj_out)?;
        let a = head(ax, &self.audio_scale_shift_table, &ts.a_embedded, &self.audio_proj_out)?;
        let (b, c) = (v.dim(0)?, v.dim(2)?);
        let v = v.reshape((b, f, h, w, c))?.permute((0, 4, 1, 2, 3))?.contiguous()?;
        let t = a.dim(1)?;
        let a = a.reshape((b, t, 8, 16))?.permute((0, 2, 1, 3))?.contiguous()?;
        Ok((v, a))
    }
}

/// debug-ltx-io CHECKPOINT REF2: everything around the blocks against ComfyUI's capture
pub fn debug_io(ckpt: &str, reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let dit = Dit::load(&st, &dev)?;
    let vl = get("in.x.0")?;
    let ppf = vl.dim(3)? * vl.dim(4)?;
    let ts = dit.timesteps(&get("in.timestep.0")?, &get("in.timestep.1")?, ppf, false, 1)?;
    for (name, ours, key) in [
        ("v timestep", &ts.v, "ts.out.0.0.data"),
        ("a timestep", &ts.a, "ts.out.0.1"),
        ("a cross scale/shift", &ts.a_cross_ss, "ts.out.0.2.0"),
        ("v cross scale/shift", &ts.v_cross_ss, "ts.out.0.2.1.data"),
        ("a2v gate", &ts.v_cross_gate, "ts.out.0.2.2.data"),
        ("v2a gate", &ts.a_cross_gate, "ts.out.0.2.3"),
        ("v prompt", &ts.v_prompt, "ts.out.0.3"),
        ("a prompt", &ts.a_prompt, "ts.out.0.4"),
        ("v embedded", &ts.v_embedded, "ts.out.1.0.data"),
        ("a embedded", &ts.a_embedded, "ts.out.1.1"),
    ] {
        compare(name, ours, &get(key)?)?;
    }
    let al = get("in.x.1")?;
    let (vx, ax) = dit.patchify(&vl, &al, false)?;
    compare("patchified video", &vx, &get("b0.in.x.0")?)?;
    compare("patchified audio", &ax, &get("b0.in.x.1")?)?;
    let (_, _, f, h, w) = vl.dims5()?;
    let pos = dit.positions(f, h, w, 30.0, al.dim(2)?, &[], &dev)?;
    compare("video rope", &pos.v, &get("b0.in.v_pe.0")?)?;
    compare("audio rope", &pos.a, &get("b0.in.a_pe.0")?)?;
    compare("video cross rope", &pos.v_cross, &get("b0.in.v_cross_pe.0")?)?;
    compare("audio cross rope", &pos.a_cross, &get("b0.in.a_cross_pe.0")?)?;
    let ctx = get("in.context")?;
    compare("video context", &ctx.narrow(2, 0, 4096)?.contiguous()?, &get("b0.in.v_context")?)?;
    compare("audio context", &ctx.narrow(2, 4096, 2048)?.contiguous()?, &get("b0.in.a_context")?)?;
    let (ov, oa) = dit.output(&get("final.in.0")?, &get("final.in.1")?, &ts, f, h, w)?;
    compare("output video (from the last block's)", &ov, &get("out.0")?)?;
    compare("output audio (from the last block's)", &oa, &get("out.1")?)?;
    Ok(())
}

/// The DiT's blocks, kept on the GPU between calls while more than `reserve` bytes stay free;
/// the rest are read from the checkpoint each time they are needed.
pub struct BlockCache {
    resident: Vec<Option<std::rc::Rc<Block>>>,
    reserve: usize,
    /// the LoRAs folded into every block this cache loads
    loras: Option<std::rc::Rc<Loras>>,
    /// whether every block's LoRAs are folded and kept (`fold_all`)
    all_folded: bool,
}

/// Free VRAM after a synchronize: memory freed on the stream stays in CUDA's pool, counted as used,
/// until the stream syncs and the pool gives it back (LoRA folding frees a lot of it)
fn free_vram(dev: &Device) -> Result<usize> {
    dev.synchronize()?;
    match dev {
        Device::Cuda(d) => Ok(d.cuda_stream().context().mem_get_info().map_err(|e| anyhow::anyhow!("free VRAM: {e:?}"))?.0),
        _ => anyhow::bail!("ltx: not on the GPU"),
    }
}

impl BlockCache {
    pub fn new(reserve: usize) -> Self {
        Self { resident: vec![], reserve, loras: None, all_folded: false }
    }

    /// The same cache with LoRAs folded into each block as it loads
    pub fn with_loras(mut self, loras: Option<std::rc::Rc<Loras>>) -> Self {
        self.loras = loras;
        self
    }

    /// A cache that keeps VRAM free for one call's activations at `tokens` video tokens: 1 GiB,
    /// plus 128 KiB a token (the feed-forward's 16384-wide hidden state, q/k/v and the per-token
    /// cross-attention AdaLN rows, with room over). 17 frames at 640x448 (840 tokens) keep about
    /// 1.1 GiB back, so all 48 blocks (18.8 GiB) stay on a 24 GB card, as in ComfyUI.
    pub fn for_tokens(tokens: usize) -> Self {
        Self::new((1usize << 30) + tokens * (128 << 10))
    }

    /// What loading one block that is not resident takes on top of the activations: the block
    /// (about 0.39 GiB), and with LoRAs not folded yet the room to fold them in (an fp16 copy and
    /// an f32 product of its largest layer, about 0.8 GiB at the peak)
    fn streaming_need(&self) -> usize {
        (1usize << 29) + if self.loras.is_some() && !self.all_folded { 1usize << 30 } else { 0 }
    }

    /// With LoRAs, folds every block once, a block at a time, keeping each in host memory and
    /// letting its copy on the card go, so the blocks then load as plain uploads: no folding's
    /// scratch space left between them on the card, and none of it at any later step or graph.
    fn fold_all(&mut self, st: &MmapedSafetensors, dev: &Device) -> Result<()> {
        let Some(loras) = self.loras.clone() else { return Ok(()) };
        if self.all_folded {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let mut i = 0;
        while st.get(&format!("{DIT}transformer_blocks.{i}.scale_shift_table")).is_ok() {
            if self.resident.get(i).is_none_or(|b| b.is_none()) {
                drop(Block::load_lora(st, i, dev, Some(&loras))?);
                dev.synchronize()?;
            }
            i += 1;
        }
        self.all_folded = true;
        let (layers, bytes) = loras.kept();
        eprintln!(
            "[ltx] LoRAs folded into {layers} layers of {i} blocks in {:.1}s, {:.1} GB kept in RAM",
            t.elapsed().as_secs_f64(),
            bytes as f64 / 1e9
        );
        Ok(())
    }

    /// Sets the VRAM to keep free for `tokens` video tokens' activations (as `for_tokens`) and
    /// lets resident blocks go, last first, until that much is free: a bigger graph than the last
    /// one needs more room.
    pub fn fit(&mut self, tokens: usize, dev: &Device) -> Result<()> {
        self.reserve = Self::for_tokens(tokens).reserve;
        self.evict_to(self.reserve, dev)
    }

    /// The same with `extra` bytes more free for a while (the connectors, loaded beside the blocks)
    pub fn fit_with(&mut self, tokens: usize, extra: usize, dev: &Device) -> Result<()> {
        self.reserve = Self::for_tokens(tokens).reserve;
        self.evict_to(self.reserve + extra, dev)
    }

    fn evict_to(&mut self, free_wanted: usize, dev: &Device) -> Result<()> {
        while free_vram(dev)? < free_wanted {
            match self.resident.iter().rposition(|b| b.is_some()) {
                Some(i) => {
                    self.resident[i] = None;
                    dev.synchronize()?;
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Loads every block that fits before sampling starts (ComfyUI loads the model before its
    /// first step too), so the steps run with the weights already on the card. Returns how many
    /// are resident.
    pub fn preload(&mut self, st: &MmapedSafetensors, dev: &Device) -> Result<usize> {
        self.fold_all(st, dev)?;
        let mut i = 0;
        let mut all = true;
        while st.get(&format!("{DIT}transformer_blocks.{i}.scale_shift_table")).is_ok() {
            // room to load this block (and fold its LoRAs in) beyond the activations' share
            if std::env::var_os("RUSTYUI_TRACE_VRAM").is_some() {
                eprintln!(
                    "[ltx] before block {i}: {} MiB free, {} MiB wanted",
                    free_vram(dev)? >> 20,
                    (self.reserve + self.streaming_need()) >> 20
                );
            }
            if free_vram(dev)? <= self.reserve + self.streaming_need() {
                all = false;
                break;
            }
            self.get(st, i, dev)?;
            i += 1;
        }
        // blocks left over load during every step: keep room for one beside the activations
        if !all {
            self.evict_to(self.reserve + self.streaming_need(), dev)?;
        }
        Ok(self.resident())
    }

    fn get(&mut self, st: &MmapedSafetensors, i: usize, dev: &Device) -> Result<std::rc::Rc<Block>> {
        if let Some(Some(b)) = self.resident.get(i) {
            return Ok(b.clone());
        }
        let b = std::rc::Rc::new(Block::load_lora(st, i, dev, self.loras.as_deref())?);
        if free_vram(dev)? > self.reserve + self.streaming_need() {
            if self.resident.len() <= i {
                self.resident.resize(i + 1, None);
            }
            self.resident[i] = Some(b.clone());
        }
        Ok(b)
    }

    /// How many blocks stay on the GPU
    pub fn resident(&self) -> usize {
        self.resident.iter().filter(|b| b.is_some()).count()
    }
}

impl Dit {
    /// One full DiT call, its blocks from `cache` (resident) or the checkpoint: latents video
    /// (1, 128, F, H, W) and audio (1, 8, T, 16), timesteps as `timesteps` takes them (a sigma, or
    /// one per token), context (1, L, 6144). `each_block` sees the streams after every block.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_streamed(
        &self,
        st: &MmapedSafetensors,
        cache: &mut BlockCache,
        v_lat: &Tensor,
        a_lat: &Tensor,
        v_t: &Tensor,
        a_t: &Tensor,
        spatial: bool,
        context: &Tensor,
        fps: f64,
        guides: &[Guide],
        mut each_block: impl FnMut(usize, &Tensor, &Tensor) -> Result<()>,
    ) -> Result<(Tensor, Tensor)> {
        let dev = v_lat.device().clone();
        let (_, _, f, h, w) = v_lat.dims5()?;
        let ts = self.timesteps(v_t, a_t, h * w, spatial, v_lat.dim(0)?)?;
        let (mut vx, mut ax) = self.patchify(v_lat, a_lat, !guides.is_empty())?;
        let pos = self.positions(f, h, w, fps, a_lat.dim(2)?, guides, &dev)?;
        let mask = guide_mask(f * h * w, guides, &dev)?;
        let v_ctx = context.narrow(2, 0, 4096)?.contiguous()?;
        let a_ctx = context.narrow(2, 4096, 2048)?.contiguous()?;
        let args = BlockArgs {
            v_context: &v_ctx,
            a_context: &a_ctx,
            v_timestep: &ts.v,
            a_timestep: &ts.a,
            v_pe: &pos.v,
            a_pe: &pos.a,
            v_cross_pe: &pos.v_cross,
            a_cross_pe: &pos.a_cross,
            v_cross_ss: &ts.v_cross_ss,
            a_cross_ss: &ts.a_cross_ss,
            v_cross_gate: &ts.v_cross_gate,
            a_cross_gate: &ts.a_cross_gate,
            v_prompt: &ts.v_prompt,
            a_prompt: &ts.a_prompt,
            v_mask: mask.as_ref(),
        };
        let mut i = 0;
        while st.get(&format!("{DIT}transformer_blocks.{i}.scale_shift_table")).is_ok() {
            let blk = cache.get(st, i, &dev)?;
            let (v2, a2) = blk.forward(&vx, &ax, &args)?;
            (vx, ax) = (v2, a2);
            each_block(i, &vx, &ax)?;
            i += 1;
        }
        self.output(&vx, &ax, &ts, f, h, w)
    }
}

/// debug-ltx-forward CHECKPOINT REF2: one whole DiT call from ComfyUI's own inputs
pub fn debug_forward(ckpt: &str, reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let dit = Dit::load(&st, &dev)?;
    let t = std::time::Instant::now();
    let mut cache = BlockCache::for_tokens(get("in.x.0")?.dims()[2..].iter().product());
    let (ov, oa) = dit.forward_streamed(
        &st,
        &mut cache,
        &get("in.x.0")?,
        &get("in.x.1")?,
        &get("in.timestep.0")?,
        &get("in.timestep.1")?,
        false,
        &get("in.context")?,
        30.0,
        &[],
        |i, vx, ax| {
            if let (Ok(v), Ok(a)) = (get(&format!("b{i}.out.0")), get(&format!("b{i}.out.1"))) {
                compare(&format!("after block {i}, video"), vx, &v)?;
                compare(&format!("after block {i}, audio"), ax, &a)?;
            }
            Ok(())
        },
    )?;
    dev.synchronize()?;
    println!("forward with streamed blocks: {:.1} s", t.elapsed().as_secs_f64());
    compare("DiT output, video", &ov, &get("out.0")?)?;
    compare("DiT output, audio", &oa, &get("out.1")?)?;
    Ok(())
}

/// debug-ltx-batch CHECKPOINT REF (refs/dit_batch_ref.py): the server's batched DiT call
/// (CFG's negative and prompt together) from its own inputs: the prepared timesteps, patchify,
/// every block and the output. The samples share their timesteps, which is checked.
pub fn debug_batch(ckpt: &str, reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let dit = Dit::load(&st, &dev)?;
    let (vl, al) = (get("in.x.0")?, get("in.x.1")?);
    let (b, _, f, h, w) = vl.dims5()?;
    let (vt, at) = (get("in.timestep.0")?, get("in.timestep.1")?);
    for (name, t) in [("video", &vt), ("audio", &at)] {
        let first = t.narrow(0, 0, 1)?;
        for i in 1..b {
            anyhow::ensure!(
                t.narrow(0, i, 1)?.eq(&first)?.flatten_all()?.min(0)?.to_scalar::<u8>()? == 1,
                "the batch's {name} timesteps differ"
            );
        }
    }
    let (vt, at) = (vt.narrow(0, 0, 1)?, at.narrow(0, 0, 1)?);
    let fps = 24.0;
    let ts = dit.timesteps(&vt, &at, h * w, false, b)?;
    compare("timestep: video ada", &ts.v, &get("ts.out.0.0.data")?)?;
    compare("timestep: audio ada", &ts.a, &get("ts.out.0.1")?)?;
    compare("timestep: audio cross scale/shift", &ts.a_cross_ss, &get("ts.out.0.2.0")?)?;
    compare("timestep: video cross scale/shift", &ts.v_cross_ss, &get("ts.out.0.2.1.data")?)?;
    compare("timestep: a2v gate", &ts.v_cross_gate, &get("ts.out.0.2.2.data")?)?;
    compare("timestep: v2a gate", &ts.a_cross_gate, &get("ts.out.0.2.3")?)?;
    compare("timestep: video prompt", &ts.v_prompt, &get("ts.out.0.3")?)?;
    compare("timestep: audio prompt", &ts.a_prompt, &get("ts.out.0.4")?)?;
    let (vx, ax) = dit.patchify(&vl, &al, false)?;
    compare("patchify, video", &vx, &get("b0.in.x.0")?)?;
    compare("patchify, audio", &ax, &get("b0.in.x.1")?)?;
    let t = std::time::Instant::now();
    let mut cache = BlockCache::for_tokens(b * f * h * w);
    let (ov, oa) = dit.forward_streamed(&st, &mut cache, &vl, &al, &vt, &at, false, &get("in.context")?, fps, &[], |i, vx, ax| {
        compare(&format!("after block {i}, video"), vx, &get(&format!("b{i}.out.0"))?)?;
        compare(&format!("after block {i}, audio"), ax, &get(&format!("b{i}.out.1"))?)?;
        Ok(())
    })?;
    dev.synchronize()?;
    println!("batched forward: {:.1} s", t.elapsed().as_secs_f64());
    compare("DiT output, video", &ov, &get("out.0")?)?;
    compare("DiT output, audio", &oa, &get("out.1")?)?;
    Ok(())
}

// ── the text side: Embeddings1DConnector ───────────────────────────────────────

/// Embeddings1DConnector: the projected text padded with learnable registers to at least 1024
/// tokens, 8 pre-norm blocks (gated attention with split RoPE over the token index / 4096, then
/// the feed-forward), a final RMSNorm. One per stream.
pub struct Connector {
    blocks: Vec<(Attention, FeedForward)>,
    registers: Tensor,
    heads: usize,
}

impl Connector {
    pub fn load(st: &MmapedSafetensors, which: &str, dev: &Device) -> Result<Self> {
        let p = format!("{DIT}{which}_embeddings_connector");
        let mut blocks = vec![];
        let mut i = 0;
        while st.get(&format!("{p}.transformer_1d_blocks.{i}.attn1.to_q.weight")).is_ok() {
            let b = format!("{p}.transformer_1d_blocks.{i}");
            blocks.push((Attention::load(st, &format!("{b}.attn1"), 32, dev)?, FeedForward::load(st, &format!("{b}.ff"), dev)?));
            i += 1;
        }
        Ok(Self { blocks, registers: st.load(&format!("{p}.learnable_registers"), dev)?, heads: 32 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, t, inner) = x.dims3()?;
        let n_reg = self.registers.dim(0)?;
        let dup = t.max(1024).div_ceil(n_reg);
        let tiled = self.registers.unsqueeze(0)?.repeat((dup, 1, 1))?.reshape((dup * n_reg, inner))?;
        let mut h = Tensor::cat(&[x, &tiled.narrow(0, t, dup * n_reg - t)?.unsqueeze(0)?], 1)?;
        let len = h.dim(1)?;
        let coords: Vec<[[f32; 2]; 3]> = (0..len).map(|i| [[i as f32, i as f32], [0.0; 2], [0.0; 2]]).collect();
        let pe = rope(&coords, 1, &[4096.0], inner, self.heads, x.device())?;
        for (attn, ff) in &self.blocks {
            let a = attn.forward(&kernels::ltx_rms_norm(&h, None, 1e-6)?, None, Some(&pe), None)?;
            h = kernels::ltx_add(&a, &h)?;
            let f = ff.forward(&kernels::ltx_rms_norm(&h, None, 1e-6)?)?;
            h = kernels::ltx_add(&f, &h)?;
        }
        Ok(kernels::ltx_rms_norm(&h, None, 1e-6)?)
    }
}

/// preprocess_text_embeds: the projection's (1, T, 4096 + 2048) through the two connectors
pub fn text_context(video: &Connector, audio: &Connector, projected: &Tensor) -> Result<Tensor> {
    let v = video.forward(&projected.narrow(2, 0, 4096)?.contiguous()?)?;
    let a = audio.forward(&projected.narrow(2, 4096, 2048)?.contiguous()?)?;
    Ok(Tensor::cat(&[&v, &a], 2)?)
}

/// debug-ltx-connector CHECKPOINT REF2: the connectors on ComfyUI's projected text
pub fn debug_connector(ckpt: &str, reference: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let (video, audio) = (Connector::load(&st, "video", &dev)?, Connector::load(&st, "audio", &dev)?);
    let ours = text_context(&video, &audio, &r.load("connector_in", &dev)?)?;
    let theirs = r.load("connector_out", &dev)?;
    compare("connector output, video part", &ours.narrow(2, 0, 4096)?.contiguous()?, &theirs.narrow(2, 0, 4096)?.contiguous()?)?;
    compare("connector output, audio part", &ours.narrow(2, 4096, 2048)?.contiguous()?, &theirs.narrow(2, 4096, 2048)?.contiguous()?)?;
    compare("DiT context (connector output)", &ours, &r.load("in.context", &dev)?)?;
    Ok(())
}

// ── sampling: LTXVScheduler, the noise, euler ──────────────────────────────────

/// LTXVScheduler's sigmas, in torch's CPU float32 arithmetic: linspace(1, 0, steps + 1), the
/// token-count shift (exp(s) / (exp(s) + 1/sigma - 1)), then stretched so the last nonzero
/// sigma is `terminal`. `tokens` is the video latent's F * H * W.
pub fn scheduler_sigmas(steps: usize, max_shift: f64, base_shift: f64, stretch: bool, terminal: f64, tokens: usize) -> Vec<f32> {
    let step = (0.0f32 - 1.0f32) / steps as f32;
    let half = steps.div_ceil(2);
    let lin: Vec<f32> = (0..=steps).map(|i| if i < half { 1.0f32 + step * i as f32 } else { 0.0f32 - step * (steps - i) as f32 }).collect();
    let mm = (max_shift - base_shift) / (4096.0 - 1024.0);
    let b = base_shift - mm * 1024.0;
    let shift = (tokens as f64 * mm + b).exp();
    let shift32 = shift as f32;
    let mut sigmas: Vec<f32> = lin
        .iter()
        .map(|&s| {
            if s == 0.0 {
                return 0.0;
            }
            // exp(s) / (exp(s) + (1 / sigma - 1) ** 1): __rtruediv__ is reciprocal() * other
            let inner = (1.0f32 / s) * 1.0 - 1.0;
            let denom = shift32 + inner;
            (1.0f32 / denom) * shift32
        })
        .collect();
    if stretch {
        let nonzero: Vec<usize> = (0..sigmas.len()).filter(|&i| sigmas[i] != 0.0).collect();
        let one_minus: Vec<f32> = nonzero.iter().map(|&i| 1.0f32 - sigmas[i]).collect();
        let scale = one_minus[one_minus.len() - 1] / (1.0 - terminal) as f32;
        for (k, &i) in nonzero.iter().enumerate() {
            sigmas[i] = 1.0f32 - one_minus[k] / scale;
        }
    }
    sigmas
}

/// The seeded noise for one audio-video latent: video first, then audio, one generator
pub fn noise(seed: u64, video_shape: &[usize], audio_shape: &[usize], dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut g = crate::rng::TorchCpuGenerator::new(seed);
    let nv: usize = video_shape.iter().product();
    let na: usize = audio_shape.iter().product();
    let v = Tensor::from_vec(g.randn(nv), video_shape, dev)?;
    let a = Tensor::from_vec(g.randn(na), audio_shape, dev)?;
    Ok((v, a))
}

/// A latent's kept part, as the sampler's denoise mask gives it (LTXVImgToVideoInplace's noise_mask,
/// through LTXVConcatAVLatent): `*_latent` the clean latent the kept elements are held to, f32;
/// `*_mask` 1 where the sampler generates and 0 where it keeps, f32, each its latent's shape as
/// prepare_mask made it (the video's (1, 128, F, H, W), the audio's (1, 8, T, 16)).
pub struct Inpaint {
    pub v_latent: Tensor,
    pub v_mask: Tensor,
    pub a_latent: Tensor,
    pub a_mask: Tensor,
}

impl Inpaint {
    /// has_spatial_mask: any latent frame whose mask (channel 0, in the model's dtype as the DiT
    /// gets it) is not one value
    fn spatial(&self) -> Result<bool> {
        let (_, _, f, h, w) = self.v_mask.dims5()?;
        let m: Vec<f32> = self.v_mask.narrow(1, 0, 1)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok((0..f).any(|i| {
            let frame = &m[i * h * w..(i + 1) * h * w];
            frame.iter().any(|&v| v != frame[0])
        }))
    }

    /// process_timestep's masks, in the model's dtype: the video's channel 0 (F * H * W tokens),
    /// the audio's channel 0 and bin 0 (T tokens)
    pub fn token_masks(&self) -> Result<(Tensor, Tensor)> {
        let v = self.v_mask.narrow(1, 0, 1)?.flatten_all()?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        let a = self.a_mask.narrow(1, 0, 1)?.narrow(3, 0, 1)?.flatten_all()?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        Ok((v, a))
    }
}

/// x * mask + latent * (1 - mask) in f32, as KSamplerX0Inpaint blends (LTXV's scale_latent_inpaint
/// is the clean latent itself, so the kept part enters the model unnoised)
fn blend(x: &Tensor, latent: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let keep = mask.affine(-1.0, 1.0)?;
    Ok((x.broadcast_mul(mask)? + latent.broadcast_mul(&keep)?)?)
}

/// CONST.noise_scaling: sigma * noise + (1 - sigma) * latent_image, f32
pub fn noise_scaling(sigma: f32, noise: &Tensor, latent: &Tensor) -> Result<Tensor> {
    let s = Tensor::new(&[sigma], noise.device())?;
    let one_minus = Tensor::new(&[1.0f32 - sigma], noise.device())?;
    Ok((noise.broadcast_mul(&s)? + latent.broadcast_mul(&one_minus)?)?)
}

/// What each step's model call is guided by: the prompt's text context and, for CFG above one,
/// the negative's with the scales. ComfyUI's CFGGuider combines the two predictions as
/// uncond + (cond - uncond) * cfg; Guider_LTXAVDualCFG, when its video and audio scales differ,
/// does it in noise space (x - prediction), each modality with its own scale.
pub struct Guidance<'a> {
    pub cond: &'a Tensor,
    pub uncond: Option<&'a Tensor>,
    pub video_cfg: f64,
    pub audio_cfg: f64,
    pub dual: bool,
}

impl<'a> Guidance<'a> {
    /// the prompt alone (CFG 1, no negative pass)
    pub fn plain(cond: &'a Tensor) -> Self {
        Guidance { cond, uncond: None, video_cfg: 1.0, audio_cfg: 1.0, dual: false }
    }
}

/// One euler step's model call: the f32 latents in bf16 through the DiT, the prediction made a
/// denoised latent the CONST way (x - out * sigma, in f32), and for CFG the negative's prediction
/// combined with it as cfg_function does. With a mask, KSamplerX0Inpaint around it: the kept part
/// reset to the clean latent going in and coming out, and the timesteps process_timestep makes
/// (sigma times the mask, per token).
#[allow(clippy::too_many_arguments)]
fn denoise(
    dit: &Dit,
    st: &MmapedSafetensors,
    cache: &mut BlockCache,
    v: &Tensor,
    a: &Tensor,
    sigma: f32,
    g: &Guidance,
    fps: f64,
    guides: &[Guide],
    inpaint: Option<&Inpaint>,
) -> Result<(Tensor, Tensor)> {
    let dev = v.device().clone();
    let s = Tensor::new(&[sigma], &dev)?;
    let (v_in, a_in, v_t, a_t, spatial) = match inpaint {
        None => (v.clone(), a.clone(), s.clone(), s.clone(), false),
        Some(m) => {
            // the model's extra conds reach process_timestep in its dtype: the mask is bf16 there
            let (tv, ta) = m.token_masks()?;
            (blend(v, &m.v_latent, &m.v_mask)?, blend(a, &m.a_latent, &m.a_mask)?, (tv * sigma as f64)?, (ta * sigma as f64)?, m.spatial()?)
        }
    };
    let (vb, ab) = (v_in.to_dtype(DType::BF16)?, a_in.to_dtype(DType::BF16)?);
    // the prediction made a denoised latent (CONST: x - out * sigma), sample `i` of the call's output
    let denoised =
        |o: &Tensor, i: usize, x: &Tensor| -> Result<Tensor> { Ok((x - o.narrow(0, i, 1)?.to_dtype(DType::F32)?.broadcast_mul(&s)?)?) };
    let (dv, da) = match g.uncond {
        None => {
            let (ov, oa) = dit.forward_streamed(st, cache, &vb, &ab, &v_t, &a_t, spatial, g.cond, fps, guides, |_, _, _| Ok(()))?;
            (denoised(&ov, 0, &v_in)?, denoised(&oa, 0, &a_in)?)
        }
        Some(u) => {
            // _calc_cond_batch runs the negative and the prompt as one call when they fit, the
            // negative first, and that call rounds otherwise than two of one (debug-ltx-batch)
            let (v2, a2) = (Tensor::cat(&[&vb, &vb], 0)?, Tensor::cat(&[&ab, &ab], 0)?);
            let ctx = Tensor::cat(&[u, g.cond], 0)?;
            let (ov, oa) = dit.forward_streamed(st, cache, &v2, &a2, &v_t, &a_t, spatial, &ctx, fps, guides, |_, _, _| Ok(()))?;
            drop((v2, a2, ctx));
            let (uv, ua) = (denoised(&ov, 0, &v_in)?, denoised(&oa, 0, &a_in)?);
            let (cv, ca) = (denoised(&ov, 1, &v_in)?, denoised(&oa, 1, &a_in)?);
            if g.dual {
                // dual_cfg: cond = x - cond_pred, uncond = x - uncond_pred, out = uncond + (cond -
                // uncond) * scale, and cfg_function's x - out
                let f = |x: &Tensor, c: &Tensor, u: &Tensor, cfg: f64| -> Result<Tensor> {
                    let (c, u) = ((x - c)?, (x - u)?);
                    Ok((x - (&u + ((&c - &u)? * cfg)?)?)?)
                };
                (f(&v_in, &cv, &uv, g.video_cfg)?, f(&a_in, &ca, &ua, g.audio_cfg)?)
            } else {
                // uncond_pred + (cond_pred - uncond_pred) * cond_scale
                let f = |c: &Tensor, u: &Tensor| -> Result<Tensor> { Ok((u + ((c - u)? * g.video_cfg)?)?) };
                (f(&cv, &uv)?, f(&ca, &ua)?)
            }
        }
    };
    match inpaint {
        None => Ok((dv, da)),
        Some(m) => Ok((blend(&dv, &m.v_latent, &m.v_mask)?, blend(&da, &m.a_latent, &m.a_mask)?)),
    }
}

/// k-diffusion's sample_euler (no churn) over the given sigmas, from x (already noise-scaled);
/// `each` sees (step, x, denoised)
#[allow(clippy::too_many_arguments)]
pub fn sample_euler(
    dit: &Dit,
    st: &MmapedSafetensors,
    cache: &mut BlockCache,
    mut v: Tensor,
    mut a: Tensor,
    sigmas: &[f32],
    g: &Guidance,
    fps: f64,
    guides: &[Guide],
    inpaint: Option<&Inpaint>,
    mut each: impl FnMut(usize, (&Tensor, &Tensor), (&Tensor, &Tensor)) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let dev = v.device().clone();
    for i in 0..sigmas.len() - 1 {
        let t = std::time::Instant::now();
        let (dv, da) = denoise(dit, st, cache, &v, &a, sigmas[i], g, fps, guides, inpaint)?;
        dev.synchronize()?;
        eprintln!("[ltx] step {i}: {:.2} s, {} blocks resident", t.elapsed().as_secs_f64(), cache.resident());
        each(i, (&v, &a), (&dv, &da))?;
        let s = Tensor::new(&[sigmas[i]], &dev)?;
        let dt = Tensor::new(&[sigmas[i + 1] - sigmas[i]], &dev)?;
        // d = (x - denoised) / sigma; x = x + d * dt
        let step = |x: &Tensor, d: &Tensor| -> Result<Tensor> { Ok((x + (x - d)?.broadcast_div(&s)?.broadcast_mul(&dt)?)?) };
        (v, a) = (step(&v, &dv)?, step(&a, &da)?);
    }
    Ok((v, a))
}

/// k-diffusion's sample_euler_ancestral_RF (eta 1, s_noise 1), which ComfyUI runs for flow models
/// such as LTX: each step goes down to sigma_down and adds fresh noise to land on the next sigma.
/// The scalars are 0-dim f32 tensors in torch, rounded once per operation, as here; the noise is
/// torch's CUDA randn for `seed` over the packed latent (video then audio), made by curand's own
/// device code (ops.cu's torch_randn).
#[allow(clippy::too_many_arguments)]
pub fn sample_euler_ancestral_rf(
    dit: &Dit,
    st: &MmapedSafetensors,
    cache: &mut BlockCache,
    mut v: Tensor,
    mut a: Tensor,
    sigmas: &[f32],
    g: &Guidance,
    fps: f64,
    guides: &[Guide],
    inpaint: Option<&Inpaint>,
    seed: u64,
    sms: u32,
    mut each: impl FnMut(usize, (&Tensor, &Tensor), (&Tensor, &Tensor)) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let dev = v.device().clone();
    let mut rng = crate::rng::CudaRandn::new(seed, sms, 1536);
    let t = |x: f32| Tensor::new(&[x], &dev);
    for i in 0..sigmas.len() - 1 {
        let tm = std::time::Instant::now();
        let (dv, da) = denoise(dit, st, cache, &v, &a, sigmas[i], g, fps, guides, inpaint)?;
        dev.synchronize()?;
        eprintln!("[ltx] step {i}: {:.2} s, {} blocks resident", tm.elapsed().as_secs_f64(), cache.resident());
        each(i, (&v, &a), (&dv, &da))?;
        let (s0, s1) = (sigmas[i], sigmas[i + 1]);
        if s1 == 0.0 {
            (v, a) = (dv, da);
            continue;
        }
        let downstep_ratio = 1.0f32 + (s1 / s0 - 1.0f32) * 1.0f32;
        let sigma_down = s1 * downstep_ratio;
        let alpha_ip1 = 1.0f32 - s1;
        let alpha_down = 1.0f32 - sigma_down;
        // x**2 is x * x and **0.5 is sqrt in torch's pow
        let renoise = (s1 * s1 - sigma_down * sigma_down * (alpha_ip1 * alpha_ip1) / (alpha_down * alpha_down)).sqrt();
        let ratio = sigma_down / s0;
        let scale = alpha_ip1 / alpha_down;
        let noise = rng.randn_gpu(v.elem_count() + a.elem_count(), &dev)?;
        let (nv, na) = (noise.narrow(0, 0, v.elem_count())?, noise.narrow(0, v.elem_count(), a.elem_count())?);
        let step = |x: &Tensor, d: &Tensor, n: &Tensor| -> Result<Tensor> {
            // x = ratio * x + (1 - ratio) * denoised; x = (alpha_ip1 / alpha_down) * x + noise * 1.0 * renoise
            let x = (x.broadcast_mul(&t(ratio)?)? + d.broadcast_mul(&t(1.0 - ratio)?)?)?;
            let n = n.reshape(x.shape())?.broadcast_mul(&t(1.0)?)?.broadcast_mul(&t(renoise)?)?;
            Ok((x.broadcast_mul(&t(scale)?)? + n)?)
        };
        (v, a) = (step(&v, &dv, &nv)?, step(&a, &da, &na)?);
    }
    Ok((v, a))
}

/// debug-ltx-lora CHECKPOINT LORA_REF LORA... : layers with a set of LoRAs folded in, against
/// the weights ComfyUI made of them (refs/lora_ref.py), each LoRA at strength 1
pub fn debug_lora(ckpt: &str, reference: &str, loras: &[std::path::PathBuf]) -> Result<()> {
    // refs/lora_dynamic_ref.py: the weights ComfyUI's layers multiplied by on their first two
    // casts, with the LoRAs in the order the graph stacks them (give them in that order)
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let list: Vec<(std::path::PathBuf, f64)> = loras.iter().map(|p| (p.clone(), 1.0)).collect();
    let l = Loras::open(&list)?;
    for (p, n) in l.unmatched(&st) {
        println!("{}: {n} layers not in the checkpoint", p.display());
    }
    let mut layers: Vec<String> = r.tensors().iter().filter_map(|(k, _)| k.split(".call").next().map(str::to_string)).collect();
    layers.sort();
    layers.dedup();
    for layer in layers {
        let lin = Linear::load_lora(&st, &format!("{DIT}{layer}"), &dev, Some(&l))?;
        for call in 0..2 {
            let k = format!("{layer}.call{call}");
            match &lin {
                Linear::Fp8 { w, scale, .. } => {
                    let Ok(theirs) = r.load(&format!("{k}.scale"), &dev) else { continue };
                    let (ours_s, theirs_s) = (scale.flatten_all()?.to_vec1::<f32>()?[0], theirs.to_vec1::<f32>()?[0]);
                    println!(
                        "{k:52} fp8 scale ours {ours_s:e} theirs {theirs_s:e} {}",
                        if ours_s == theirs_s { "exact" } else { "DIFFER" }
                    );
                    compare(
                        &format!("{k} (fp8 bytes)"),
                        &w.to_dtype(DType::F32)?,
                        &r.load(&format!("{k}.qdata"), &dev)?.to_dtype(DType::F32)?,
                    )?;
                }
                Linear::Bf16 { w, .. } => {
                    let Ok(theirs) = r.load(&format!("{k}.weight"), &dev) else { continue };
                    compare(&format!("{k} (bf16)"), w, &theirs)?;
                }
            }
        }
    }
    Ok(())
}

/// debug-ltx-sample CHECKPOINT REF2 SAMPLE_REF: the scheduler, the noise and all eight euler
/// steps of sample_ref.py's run, from its text context
pub fn debug_sample(ckpt: &str, ref2: &str, sample_ref: &str) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let r2 = unsafe { MmapedSafetensors::new(ref2)? };
    let r = unsafe { MmapedSafetensors::new(sample_ref)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("sample reference has no {n}")) };
    let pack =
        |v: &Tensor, a: &Tensor| -> Result<Tensor> { Ok(Tensor::cat(&[v.flatten_all()?, a.flatten_all()?], 0)?.reshape((1, 1, ()))?) };
    let (vs, as_) = ([1usize, 128, 2, 20, 14], [1usize, 8, 8, 16]);
    let sigmas = scheduler_sigmas(8, 2.05, 0.95, true, 0.1, vs[2] * vs[3] * vs[4]);
    let theirs: Vec<f32> = get("sigmas")?.to_vec1()?;
    println!("sigmas ours   {sigmas:?}\nsigmas theirs {theirs:?}  {}", if sigmas == theirs { "exact" } else { "DIFFER" });
    let (nv, na) = noise(42, &vs, &as_, &dev)?;
    compare("noise, video", &nv, &get("noise.0")?)?;
    compare("noise, audio", &na, &get("noise.1")?)?;
    compare("x0 (packed)", &pack(&nv, &na)?, &get("x0")?)?;
    let dit = Dit::load(&st, &dev)?;
    let context = r2.load("in.context", &dev)?;
    let t = std::time::Instant::now();
    let mut cache = BlockCache::for_tokens(vs[2] * vs[3] * vs[4]);
    let (v, a) =
        sample_euler(&dit, &st, &mut cache, nv, na, &theirs, &Guidance::plain(&context), 30.0, &[], None, |i, (xv, xa), (dv, da)| {
            compare(&format!("step {i} x"), &pack(xv, xa)?, &get(&format!("step{i}.x"))?)?;
            compare(&format!("step {i} denoised"), &pack(dv, da)?, &get(&format!("step{i}.denoised"))?)?;
            Ok(())
        })?;
    println!("8 steps ({} of the blocks resident): {:.1} s", cache.resident(), t.elapsed().as_secs_f64());
    compare("sampled (packed)", &pack(&v, &a)?, &get("sampled")?)?;
    Ok(())
}
