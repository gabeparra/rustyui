//! Anima's diffusion model, as ComfyUI runs it in bf16 (comfy/ldm/anima/model.py on
//! comfy/ldm/cosmos/predict2.py's MiniTrainDIT): the LLM adapter that turns Qwen3's states and the
//! prompt's T5 ids into the 512-token context (Anima.extra_conds runs it once per sampling, padded
//! to 512 with zeros), and the DiT itself: 2x2 patches of the latent and a zero padding-mask
//! channel through one linear, Cosmos's 3D RoPE, 28 blocks of adaLN-LoRA-modulated self-attention
//! (comfy_kitchen's fused RMSNorm + RoPE on q and k), cross-attention and a GELU MLP, each added
//! back through addcmul with its gate, and the modulated final linear.
//!
//! Every op is the torch op ComfyUI runs, on the kernels LTX-2.3's port already matches to the bit
//! (ltx_layer_norm, ltx_modulate, ltx_addcmul, ltx_rms_norm, torch_gelu_erf_bf16, torch's linears,
//! flash attention). Unchecked against ComfyUI so far.

use crate::kernels as k;
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

const EPS: f32 = 1e-6;
/// the adapter's context length: preprocess_text_embeds pads to it
pub const CONTEXT: usize = 512;

fn bf16(st: &MmapedSafetensors, dev: &Device, name: &str) -> Result<Tensor> {
    Ok(st.load(name, dev).with_context(|| format!("Anima: {name}"))?.to_dtype(DType::BF16)?)
}

/// LoRAs over Anima's DiT as LoraLoader stacks them: kohya files (`lora_unet_<layer>` with
/// lora_down, lora_up and alpha), each with its model strength, in the graph's order
pub struct AnimaLoras {
    files: Vec<(MmapedSafetensors, f64)>,
}

impl AnimaLoras {
    pub fn open(list: &[(PathBuf, f64)]) -> Result<Self> {
        let mut files = vec![];
        for (p, s) in list {
            let st = unsafe { MmapedSafetensors::new(p)? };
            if let Some((k, _)) = st.tensors().into_iter().find(|(k, _)| !k.starts_with("lora_unet_")) {
                bail!("{}: {k} is not a kohya lora_unet_ key, the LoRA format rustyui reads for Anima so far", p.display());
            }
            files.push((st, *s));
        }
        Ok(AnimaLoras { files })
    }

    /// Layer `p`'s bf16 weight with the LoRAs that touch it folded in, as ComfyUI's DynamicVRAM
    /// loader folds them when the layer is first cast (LoRAAdapter.calculate_weight with the
    /// weight's dtype as the intermediate): mm(up, down) as a bf16 GEMM on cuBLAS, times
    /// strength * alpha / rank (a Python float, so in f32, rounded to bf16), added in bf16, LoRA
    /// after LoRA
    fn fold(&self, p: &str, w: Tensor, dev: &Device) -> Result<Tensor> {
        let key = format!("lora_unet_{}", p.strip_prefix("net.").unwrap_or(p).replace('.', "_"));
        let mut w = w;
        for (f, strength) in &self.files {
            if f.get(&format!("{key}.lora_down.weight")).is_err() {
                continue;
            }
            let down = bf16(f, dev, &format!("{key}.lora_down.weight"))?;
            let up = bf16(f, dev, &format!("{key}.lora_up.weight"))?;
            let rank = down.dim(0)?;
            // lora[alpha].item(), then v[2] / mat2.shape[0]
            let alpha = match f.get(&format!("{key}.alpha")) {
                Ok(_) => {
                    f.load(&format!("{key}.alpha"), &Device::Cpu)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?[0] / rank as f64
                }
                Err(_) => 1.0,
            };
            let factor = strength * alpha;
            let diff = k::torch_mm_bf16(&up.flatten_from(1)?, &down.flatten_from(1)?)?.reshape(w.dims())?;
            let diff = if factor != 1.0 {
                diff.to_dtype(DType::F32)?.broadcast_mul(&Tensor::new(factor as f32, dev)?)?.to_dtype(DType::BF16)?
            } else {
                diff
            };
            w = (w + diff)?;
        }
        Ok(w)
    }
}

/// What every layer loads from: the checkpoint, the card, the LoRAs over it
struct Ld<'a> {
    st: &'a MmapedSafetensors,
    dev: &'a Device,
    loras: Option<&'a AnimaLoras>,
}

impl Ld<'_> {
    fn t(&self, name: &str) -> Result<Tensor> {
        bf16(self.st, self.dev, name)
    }
}

struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    fn load(ld: &Ld, p: &str, bias: bool) -> Result<Self> {
        let w = ld.t(&format!("{p}.weight"))?;
        let w = match ld.loras {
            Some(l) => l.fold(p, w, ld.dev)?,
            None => w,
        };
        Ok(Lin { w, b: if bias { Some(ld.t(&format!("{p}.bias"))?) } else { None } })
    }

    fn f(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match &self.b {
            Some(b) => k::torch_linear(x, &self.w, b)?,
            None => k::torch_linear_nobias(x, &self.w)?,
        })
    }
}

/// F.rms_norm over the last dim with its bf16 weight (comfy.ops' RMSNorm)
fn rms(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(k::ltx_rms_norm(x, Some(w), EPS)?)
}

/// (B, T, heads * d) -> (B, T, heads, d)
fn split_heads(x: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, t, inner) = x.dims3()?;
    Ok(x.reshape((b, t, heads, inner / heads))?)
}

// ── the LLM adapter ──────────────────────────────────────────────────────────────────────────────

const AD_HEADS: usize = 16;
const AD_HEAD_DIM: usize = 64;

/// RotaryEmbedding(64)'s inv_freq as its __init__ makes it on the CPU: 1.0 / (10000 ** (arange(0,
/// 64, 2) / 64)), the power Sleef's vectorized powf (two 16-lane blocks), then the reciprocal
fn adapter_inv_freq() -> Vec<f32> {
    (0..AD_HEAD_DIM / 2).map(|i| 1.0f32 / crate::torchcpu::sleef_powf(10000.0, (2 * i) as f32 / AD_HEAD_DIM as f32)).collect()
}

struct AdAttn {
    q: Lin,
    q_norm: Tensor,
    k: Lin,
    k_norm: Tensor,
    v: Lin,
    o: Lin,
}

impl AdAttn {
    fn load(ld: &Ld, p: &str) -> Result<Self> {
        Ok(AdAttn {
            q: Lin::load(ld, &format!("{p}.q_proj"), false)?,
            q_norm: ld.t(&format!("{p}.q_norm.weight"))?,
            k: Lin::load(ld, &format!("{p}.k_proj"), false)?,
            k_norm: ld.t(&format!("{p}.k_norm.weight"))?,
            v: Lin::load(ld, &format!("{p}.v_proj"), false)?,
            o: Lin::load(ld, &format!("{p}.o_proj"), false)?,
        })
    }

    /// Attention.forward: q from x, k and v from the context, q and k normed per head and rotated
    /// by their own positions' tables, F.scaled_dot_product_attention (flash), o_proj
    fn forward(&self, x: &Tensor, ctx: &Tensor, rope_x: &(Tensor, Tensor), rope_ctx: &(Tensor, Tensor)) -> Result<Tensor> {
        let q = rms(&split_heads(&self.q.f(x)?, AD_HEADS)?, &self.q_norm)?;
        let kk = rms(&split_heads(&self.k.f(ctx)?, AD_HEADS)?, &self.k_norm)?;
        let v = split_heads(&self.v.f(ctx)?, AD_HEADS)?;
        let q = k::anima_rotary_apply(&q, &rope_x.0, &rope_x.1)?;
        let kk = k::anima_rotary_apply(&kk, &rope_ctx.0, &rope_ctx.1)?;
        let (b, t, _, _) = q.dims4()?;
        let a = crate::ltx::sdpa(
            &q.reshape((b, t, AD_HEADS * AD_HEAD_DIM))?,
            &kk.reshape((b, kk.dim(1)?, AD_HEADS * AD_HEAD_DIM))?,
            &v.reshape((b, v.dim(1)?, AD_HEADS * AD_HEAD_DIM))?,
            AD_HEADS,
        )?;
        self.o.f(&a)
    }
}

struct AdBlock {
    norm_self: Tensor,
    self_attn: AdAttn,
    norm_cross: Tensor,
    cross_attn: AdAttn,
    norm_mlp: Tensor,
    mlp1: Lin,
    mlp2: Lin,
}

impl AdBlock {
    fn load(ld: &Ld, p: &str) -> Result<Self> {
        Ok(AdBlock {
            norm_self: ld.t(&format!("{p}.norm_self_attn.weight"))?,
            self_attn: AdAttn::load(ld, &format!("{p}.self_attn"))?,
            norm_cross: ld.t(&format!("{p}.norm_cross_attn.weight"))?,
            cross_attn: AdAttn::load(ld, &format!("{p}.cross_attn"))?,
            norm_mlp: ld.t(&format!("{p}.norm_mlp.weight"))?,
            mlp1: Lin::load(ld, &format!("{p}.mlp.0"), true)?,
            mlp2: Lin::load(ld, &format!("{p}.mlp.2"), true)?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor, rope_x: &(Tensor, Tensor), rope_ctx: &(Tensor, Tensor)) -> Result<Tensor> {
        let n = rms(x, &self.norm_self)?;
        let x = k::ltx_add(x, &self.self_attn.forward(&n, &n, rope_x, rope_x)?)?;
        let n = rms(&x, &self.norm_cross)?;
        let x = k::ltx_add(&x, &self.cross_attn.forward(&n, ctx, rope_x, rope_ctx)?)?;
        let h = self.mlp2.f(&k::torch_gelu_erf_bf16(&self.mlp1.f(&rms(&x, &self.norm_mlp)?)?)?)?;
        Ok(k::ltx_add(&x, &h)?)
    }
}

struct Adapter {
    embed: Tensor,
    blocks: Vec<AdBlock>,
    out_proj: Lin,
    norm: Tensor,
    inv_freq: Vec<f32>,
}

impl Adapter {
    fn load(ld: &Ld) -> Result<Self> {
        let mut blocks = vec![];
        while ld.st.get(&format!("net.llm_adapter.blocks.{}.norm_mlp.weight", blocks.len())).is_ok() {
            blocks.push(AdBlock::load(ld, &format!("net.llm_adapter.blocks.{}", blocks.len()))?);
        }
        Ok(Adapter {
            embed: ld.t("net.llm_adapter.embed.weight")?,
            blocks,
            out_proj: Lin::load(ld, "net.llm_adapter.out_proj", true)?,
            norm: ld.t("net.llm_adapter.norm.weight")?,
            inv_freq: adapter_inv_freq(),
        })
    }

    /// Anima.preprocess_text_embeds: Qwen3's states (1, Tq, 1024) bf16 and the T5 ids with their
    /// weights -> the context (1, 512, 1024) bf16: LLMAdapter.forward, times the weights, padded
    fn context(&self, qwen: &Tensor, t5: &[(u32, f64)]) -> Result<Tensor> {
        let dev = qwen.device();
        let ids = Tensor::new(t5.iter().map(|&(id, _)| id).collect::<Vec<u32>>(), dev)?;
        let x = self.embed.index_select(&ids, 0)?.unsqueeze(0)?;
        let (t, tq) = (t5.len(), qwen.dim(1)?);
        let rope_x = k::anima_rotary_tables(&self.inv_freq, t, dev)?;
        let rope_ctx = k::anima_rotary_tables(&self.inv_freq, tq, dev)?;
        let mut x = x;
        for b in &self.blocks {
            x = b.forward(&x, qwen, &rope_x, &rope_ctx)?;
        }
        let out = rms(&self.out_proj.f(&x)?, &self.norm)?;
        // t5xxl_weights: a float32 tensor of the Python weights, then .to(bf16)
        let w: Vec<f32> = t5.iter().map(|&(_, w)| w as f32).collect();
        let w = Tensor::new(w, dev)?.to_dtype(DType::BF16)?.reshape((1, t, 1))?;
        let out = out.broadcast_mul(&w)?;
        if t >= CONTEXT {
            return Ok(out);
        }
        Ok(Tensor::cat(&[out, Tensor::zeros((1, CONTEXT - t, 1024), DType::BF16, dev)?], 1)?)
    }
}

// ── MiniTrainDIT ─────────────────────────────────────────────────────────────────────────────────

const HEADS: usize = 16;

/// adaln_modulation_*: SiLU, a linear to 256, a linear to `n` * D, plus the timestep's adaLN-LoRA
/// part; split in `n` (B, 1, D) pieces
struct Modulation {
    down: Lin,
    up: Lin,
}

impl Modulation {
    fn load(ld: &Ld, p: &str) -> Result<Self> {
        Ok(Modulation { down: Lin::load(ld, &format!("{p}.1"), false)?, up: Lin::load(ld, &format!("{p}.2"), false)? })
    }

    fn forward(&self, emb: &Tensor, lora: &Tensor, n: usize) -> Result<Vec<Tensor>> {
        let m = self.up.f(&self.down.f(&k::ltx_silu(emb)?)?)?;
        let d = m.dim(2)? / n;
        let lora = if lora.dim(2)? == m.dim(2)? { lora.clone() } else { lora.narrow(2, 0, m.dim(2)?)? };
        let m = k::ltx_add(&m, &lora)?;
        (0..n).map(|i| Ok(m.narrow(2, i * d, d)?.contiguous()?)).collect()
    }
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
}

impl Attn {
    fn load(ld: &Ld, p: &str) -> Result<Self> {
        Ok(Attn {
            q: Lin::load(ld, &format!("{p}.q_proj"), false)?,
            k: Lin::load(ld, &format!("{p}.k_proj"), false)?,
            v: Lin::load(ld, &format!("{p}.v_proj"), false)?,
            out: Lin::load(ld, &format!("{p}.output_proj"), false)?,
            q_norm: ld.t(&format!("{p}.q_norm.weight"))?,
            k_norm: ld.t(&format!("{p}.k_norm.weight"))?,
        })
    }

    /// self-attention: q and k through comfy_kitchen's rms_rope_split_half with the 3D RoPE
    fn self_attn(&self, x: &Tensor, rope: &Tensor) -> Result<Tensor> {
        let q = split_heads(&self.q.f(x)?, HEADS)?;
        let kk = split_heads(&self.k.f(x)?, HEADS)?;
        let v = self.v.f(x)?;
        let (q, kk) = k::ck_rms_rope(&q, &kk, rope, &self.q_norm, &self.k_norm, EPS)?;
        let (b, t, _, _) = q.dims4()?;
        let inner = v.dim(2)?;
        let a = crate::ltx::sdpa(&q.reshape((b, t, inner))?, &kk.reshape((b, t, inner))?, &v, HEADS)?;
        self.out.f(&a)
    }

    /// cross-attention into the context: q and k each through its RMSNorm, no RoPE
    fn cross_attn(&self, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let q = rms(&split_heads(&self.q.f(x)?, HEADS)?, &self.q_norm)?;
        let kk = rms(&split_heads(&self.k.f(ctx)?, HEADS)?, &self.k_norm)?;
        let v = self.v.f(ctx)?;
        let (b, t, _, _) = q.dims4()?;
        let inner = v.dim(2)?;
        let a = crate::ltx::sdpa(&q.reshape((b, t, inner))?, &kk.reshape((b, kk.dim(1)?, inner))?, &v, HEADS)?;
        self.out.f(&a)
    }
}

struct Block {
    mod_self: Modulation,
    mod_cross: Modulation,
    mod_mlp: Modulation,
    self_attn: Attn,
    cross_attn: Attn,
    mlp1: Lin,
    mlp2: Lin,
}

impl Block {
    fn load(ld: &Ld, p: &str) -> Result<Self> {
        Ok(Block {
            mod_self: Modulation::load(ld, &format!("{p}.adaln_modulation_self_attn"))?,
            mod_cross: Modulation::load(ld, &format!("{p}.adaln_modulation_cross_attn"))?,
            mod_mlp: Modulation::load(ld, &format!("{p}.adaln_modulation_mlp"))?,
            self_attn: Attn::load(ld, &format!("{p}.self_attn"))?,
            cross_attn: Attn::load(ld, &format!("{p}.cross_attn"))?,
            mlp1: Lin::load(ld, &format!("{p}.mlp.layer1"), false)?,
            mlp2: Lin::load(ld, &format!("{p}.mlp.layer2"), false)?,
        })
    }

    /// Block.forward on the residual stream x (B, L, D): each part's input LayerNorm'd (no affine)
    /// times 1 + scale plus shift, its result added back times its gate (torch.addcmul)
    fn forward(&self, x: &Tensor, emb: &Tensor, lora: &Tensor, ctx: &Tensor, rope: &Tensor) -> Result<Tensor> {
        let m = self.mod_self.forward(emb, lora, 3)?;
        let n = k::ltx_modulate(&k::ltx_layer_norm(x, EPS)?, &m[1], &m[0])?;
        let x = k::ltx_addcmul(x, &self.self_attn.self_attn(&n, rope)?, &m[2])?;
        let m = self.mod_cross.forward(emb, lora, 3)?;
        let n = k::ltx_modulate(&k::ltx_layer_norm(&x, EPS)?, &m[1], &m[0])?;
        let x = k::ltx_addcmul(&x, &self.cross_attn.cross_attn(&n, ctx)?, &m[2])?;
        let m = self.mod_mlp.forward(emb, lora, 3)?;
        let n = k::ltx_modulate(&k::ltx_layer_norm(&x, EPS)?, &m[1], &m[0])?;
        let h = self.mlp2.f(&k::torch_gelu_erf_bf16(&self.mlp1.f(&n)?)?)?;
        Ok(k::ltx_addcmul(&x, &h, &m[2])?)
    }
}

pub struct AnimaDit {
    adapter: Adapter,
    x_embed: Lin,
    t_l1: Lin,
    t_l2: Lin,
    t_norm: Tensor,
    blocks: Vec<Block>,
    final_mod: Modulation,
    final_linear: Lin,
    /// Timesteps(2048)'s exponents, -ln(10000) * k / 1024 as f32
    expo: Tensor,
    channels: usize,
}

/// Cosmos's VideoRopePosition3DEmb for Anima (head dim 128: time 44 dims, height and width 42
/// each; NTK extrapolation 4 on height and width, 1 on time): the per-axis exponents and thetas
fn rope_axes() -> ([Vec<f32>; 3], [f32; 3]) {
    let (dim_h, dim_t) = (128 / 6 * 2, 128 - 2 * (128 / 6 * 2));
    // arange(0, dim, 2)[: dim // 2].float() / dim (an exact or correctly rounded division)
    let spatial: Vec<f32> = (0..dim_h / 2).map(|i| (2 * i) as f32 / dim_h as f32).collect();
    let temporal: Vec<f32> = (0..dim_t / 2).map(|i| (2 * i) as f32 / dim_t as f32).collect();
    let ntk = |ratio: f64, dim: usize| -> f64 { ratio.powf(dim as f64 / (dim as f64 - 2.0)) };
    let theta = |ratio: f64, dim: usize| -> f32 { (10000.0 * ntk(ratio, dim)) as f32 };
    ([temporal, spatial.clone(), spatial], [theta(1.0, dim_t), theta(4.0, dim_h), theta(4.0, dim_h)])
}

impl AnimaDit {
    /// The model in `file`, with `loras` folded into its layers
    pub fn load(file: &Path, dev: &Device, loras: Option<&AnimaLoras>) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(file)? };
        st.get("net.llm_adapter.blocks.0.cross_attn.q_proj.weight").with_context(|| format!("{}: not an Anima model", file.display()))?;
        let ld = Ld { st: &st, dev, loras };
        let mut blocks = vec![];
        while st.get(&format!("net.blocks.{}.mlp.layer1.weight", blocks.len())).is_ok() {
            blocks.push(Block::load(&ld, &format!("net.blocks.{}", blocks.len()))?);
        }
        let x_embed = Lin::load(&ld, "net.x_embedder.proj.1", false)?;
        let channels = x_embed.w.dim(0)?;
        if channels != 2048 || x_embed.w.dim(1)? != 68 {
            bail!("{}: an Anima of {channels} channels and {} patch inputs, not 2048 and 68", file.display(), x_embed.w.dim(1)?);
        }
        let half = channels / 2;
        let neg_log = (-(10000f64.ln())) as f32;
        let inv = 1.0f32 / half as f32;
        let expo: Vec<f32> = (0..half).map(|i| (neg_log * i as f32) * inv).collect();
        Ok(AnimaDit {
            adapter: Adapter::load(&ld)?,
            x_embed,
            t_l1: Lin::load(&ld, "net.t_embedder.1.linear_1", false)?,
            t_l2: Lin::load(&ld, "net.t_embedder.1.linear_2", false)?,
            t_norm: ld.t("net.t_embedding_norm.weight")?,
            blocks,
            final_mod: Modulation::load(&ld, "net.final_layer.adaln_modulation")?,
            final_linear: Lin::load(&ld, "net.final_layer.linear", false)?,
            expo: Tensor::new(expo, dev)?,
            channels,
        })
    }

    /// The context a prompt gives the DiT (Anima.extra_conds): Qwen3's states (1, Tq, 1024), f32 as
    /// the text encoder leaves them, cast to bf16, and the T5 ids and weights
    pub fn context(&self, qwen: &Tensor, t5: &[(u32, f64)]) -> Result<Tensor> {
        self.adapter.context(&qwen.to_dtype(DType::BF16)?, t5)
    }

    /// MiniTrainDIT._forward: x (B, 16, 1, h, w) bf16, the timesteps (B,) f32, the contexts
    /// (B, 512, 1024) bf16 -> (B, 16, 1, h, w) bf16
    pub fn forward(&self, x: &Tensor, timesteps: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        if t != 1 || c != 16 {
            bail!("Anima runs one picture of 16 channels, not {c} x {t} frames");
        }
        let dev = x.device();
        // pad_to_patch_size: circular padding of an odd height or width to the 2x2 patches
        let mut xp = x.squeeze(2)?;
        if h % 2 == 1 {
            xp = Tensor::cat(&[&xp, &xp.narrow(2, 0, 1)?], 2)?;
        }
        if w % 2 == 1 {
            xp = Tensor::cat(&[&xp, &xp.narrow(3, 0, 1)?], 3)?;
        }
        let (hp, wp) = (xp.dim(2)?, xp.dim(3)?);
        let (h2, w2) = (hp / 2, wp / 2);
        // the zero padding-mask channel, then "b c (t r) (h m) (w n) -> b t h w (c r m n)"
        let xp = Tensor::cat(&[xp, Tensor::zeros((b, 1, hp, wp), DType::BF16, dev)?], 1)?;
        let patches = xp.reshape((b, 17, h2, 2, w2, 2))?.permute((0, 2, 4, 1, 3, 5))?.contiguous()?.reshape((b, h2 * w2, 68))?;
        let mut xs = self.x_embed.f(&patches)?;
        let (ranges, thetas) = rope_axes();
        let rope = k::cosmos_rope_table([&ranges[0], &ranges[1], &ranges[2]], thetas, (1, h2, w2), dev)?.unsqueeze(0)?.unsqueeze(2)?;
        // Timesteps then TimestepEmbedding with adaLN-LoRA: the sinusoid is the embedding (through
        // t_embedding_norm), linear_2's output the LoRA part
        let sinus = k::ltx_timestep_sinusoid(&timesteps.to_dtype(DType::F32)?, &self.expo)?.reshape((b, 1, self.channels))?;
        let lora = self.t_l2.f(&k::ltx_silu(&self.t_l1.f(&sinus)?)?)?;
        let emb = rms(&sinus, &self.t_norm)?;
        for blk in &self.blocks {
            xs = blk.forward(&xs, &emb, &lora, ctx, &rope)?;
        }
        let m = self.final_mod.forward(&emb, &lora, 2)?;
        let n = k::ltx_modulate(&k::ltx_layer_norm(&xs, EPS)?, &m[1], &m[0])?;
        let out = self.final_linear.f(&n)?;
        // "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)", cropped to the input's size
        let out = out.reshape((b, h2, w2, 2, 2, 16))?.permute((0, 5, 1, 3, 2, 4))?.contiguous()?.reshape((b, 16, hp, wp))?;
        Ok(out.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?.unsqueeze(2)?)
    }
}

#[cfg(test)]
mod tests {
    // torch 2.11 on this CPU (AVX-512): 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)),
    // the adapter RotaryEmbedding's inv_freq as its __init__ makes it, 2026-10-16
    #[test]
    fn adapter_inv_freq_like_torch() {
        let want: [u32; 32] = [
            0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77, 0x3dcccccd, 0x3d99940d,
            0x3d6655c2, 0x3d2cba15, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2, 0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77,
            0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e, 0x3a83126f, 0x3a44948c, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815,
            0x393a7753, 0x390bd472,
        ];
        let got: Vec<u32> = super::adapter_inv_freq().iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want);
    }

    // VideoRopePosition3DEmb(head_dim 128, extrapolation 4, 4, 1) built on the CPU in ComfyUI,
    // 2026-10-16: its dim_temporal_range and dim_spatial_range buffers and 10000 * each NTK factor
    // as the f32 the pow kernel takes
    #[test]
    fn cosmos_rope_axes_like_comfyui() {
        let ([t, h, w], [tt, th, tw]) = super::rope_axes();
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        let spatial: [u32; 21] = [
            0x0, 0x3d430c31, 0x3dc30c31, 0x3e124925, 0x3e430c31, 0x3e73cf3d, 0x3e924925, 0x3eaaaaab, 0x3ec30c31, 0x3edb6db7, 0x3ef3cf3d,
            0x3f061862, 0x3f124925, 0x3f1e79e8, 0x3f2aaaab, 0x3f36db6e, 0x3f430c31, 0x3f4f3cf4, 0x3f5b6db7, 0x3f679e7a, 0x3f73cf3d,
        ];
        let temporal: [u32; 22] = [
            0x0, 0x3d3a2e8c, 0x3dba2e8c, 0x3e0ba2e9, 0x3e3a2e8c, 0x3e68ba2f, 0x3e8ba2e9, 0x3ea2e8ba, 0x3eba2e8c, 0x3ed1745d, 0x3ee8ba2f,
            0x3f000000, 0x3f0ba2e9, 0x3f1745d1, 0x3f22e8ba, 0x3f2e8ba3, 0x3f3a2e8c, 0x3f45d174, 0x3f51745d, 0x3f5d1746, 0x3f68ba2f,
            0x3f745d17,
        ];
        assert_eq!(bits(&h), spatial);
        assert_eq!(bits(&w), spatial);
        assert_eq!(bits(&t), temporal);
        assert_eq!((tt.to_bits(), th.to_bits(), tw.to_bits()), (0x461c4000, 0x472776f0, 0x472776f0));
    }
}
