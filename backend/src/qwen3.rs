//! Qwen3-0.6B, Anima's text encoder (`qwen_3_06b_base.safetensors`), as ComfyUI runs it
//! (comfy/text_encoders/anima.py on llama.py's Llama2_): SDClipModel embeds the tokens as float32
//! and comfy.ops.manual_cast widens every bf16 weight to it, so the whole model runs in f32 like
//! Gemma 3 for LTX (gemma.rs), on the same kernels: F.linear as cublasSgemm, F.rms_norm as torch's
//! fused kernel (the weights as they are: Qwen3's norms add no 1), RoPE (theta 1e6) as
//! comfy_kitchen's eager formula, SDPA on the memory-efficient kernel with the key and value heads
//! repeated from 8 to 16 (no fused backend takes grouped heads with a mask). The mask is the causal
//! one, finfo(f32).min / 4 above the diagonal, plus that on keys process_tokens masks. The output
//! is the last layer's, through the final norm ("last"), (T, 1024) f32.
//! Unchecked against ComfyUI so far (a refs/qwen3_ref.py capture is the next step).

use crate::kernels;
use anyhow::{Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

const HIDDEN: usize = 1024;
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
/// Qwen3Tokenizer's pad and SDClipModel's special "pad" (<|endoftext|>)
const PAD: u32 = 151643;

pub struct Qwen3 {
    st: MmapedSafetensors,
}

/// SDClipModel.process_tokens' attention mask for one sequence with no end token and pad 151643:
/// leading pads are masked; the first pad after a real token is masked and ends the sequence, and
/// everything after it is masked
pub fn token_mask(ids: &[u32]) -> Vec<bool> {
    let mut mask = Vec::with_capacity(ids.len());
    let (mut eos, mut left_pad) = (false, false);
    for (index, &tok) in ids.iter().enumerate() {
        if index == 0 && tok == PAD {
            left_pad = true;
        }
        if eos || (left_pad && tok == PAD) {
            mask.push(false);
        } else {
            mask.push(true);
            left_pad = false;
        }
        if !eos && tok == PAD && !left_pad {
            if let Some(last) = mask.last_mut() {
                *last = false;
            }
            eos = true;
        }
    }
    mask
}

/// Llama2_.forward's mask: finfo(f32).min / 4 on masked keys, plus the same above the diagonal when
/// there is more than one token (so a masked key above it holds min / 2); (1, 1, t, t)
fn attention_mask(keep: &[bool], dev: &Device) -> Result<Tensor> {
    let t = keep.len();
    let min4 = f32::MIN / 4.0;
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let a = if keep[j] { 0.0 } else { min4 };
            let c = if t > 1 && j > i { min4 } else { 0.0 };
            m[i * t + j] = a + c;
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, t, t), dev)?)
}

/// (B, KV, T, D) -> (B, 2 KV, T, D), each key/value head serving two query heads (repeat_kv_for_gqa)
fn repeat_kv(x: &Tensor) -> Result<Tensor> {
    let (b, kv, t, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.broadcast_as((b, kv, HEADS / KV_HEADS, t, d))?.contiguous()?.reshape((b, HEADS, t, d))?)
}

/// (T, heads * D) -> (1, heads, T, D), contiguous (view(B, T, H, D).transpose(1, 2), made
/// contiguous by the norm that follows)
fn heads_first(x: &Tensor, heads: usize) -> Result<Tensor> {
    let t = x.dim(0)?;
    Ok(x.reshape((t, heads, HEAD_DIM))?.permute((1, 0, 2))?.contiguous()?.unsqueeze(0)?)
}

impl Qwen3 {
    pub fn open(path: &Path) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }.with_context(|| format!("open {}", path.display()))?;
        st.get("model.layers.27.self_attn.q_norm.weight").with_context(|| format!("{}: not Qwen3-0.6B", path.display()))?;
        Ok(Qwen3 { st })
    }

    /// a weight widened from bf16 to f32, as manual_cast casts it to the input's dtype
    fn w(&self, name: &str, dev: &Device) -> Result<Tensor> {
        Ok(self.st.load(name, dev).with_context(|| format!("qwen3: {name}"))?.to_dtype(DType::F32)?)
    }

    fn apply(&self, p: &str, x: &Tensor, dev: &Device) -> Result<Tensor> {
        Ok(kernels::torch_linear_f32(x, &self.w(&format!("{p}.weight"), dev)?)?)
    }

    fn rms(&self, x: &Tensor, name: &str, dev: &Device) -> Result<Tensor> {
        Ok(kernels::torch_rms_norm_f32(x, &self.w(name, dev)?, EPS)?)
    }

    /// The embedding rows of `ids` straight from the file, widened to f32: (T, 1024)
    fn embed(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let view = self.st.get("model.embed_tokens.weight")?;
        let data = view.data();
        let row = HIDDEN * 2;
        let mut bytes = Vec::with_capacity(ids.len() * row);
        for &id in ids {
            bytes.extend_from_slice(data.get(id as usize * row..(id as usize + 1) * row).context("token id past the embedding")?);
        }
        Ok(Tensor::from_raw_buffer(&bytes, DType::BF16, &[ids.len(), HIDDEN], &Device::Cpu)?.to_device(dev)?.to_dtype(DType::F32)?)
    }

    /// Layer `l`'s self-attention on the normed input h (T, 1024), before o_proj: (T, 2048)
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

    /// TransformerBlock.forward: x (T, 1024) -> (T, 1024)
    fn layer(&self, l: usize, x: &Tensor, mask: &Tensor, rope: &(Tensor, Tensor), dev: &Device) -> Result<Tensor> {
        let p = format!("model.layers.{l}");
        let a = self.attention(l, &self.rms(x, &format!("{p}.input_layernorm.weight"), dev)?, mask, rope, dev)?;
        let x = (x + self.apply(&format!("{p}.self_attn.o_proj"), &a, dev)?)?;
        let h = self.rms(&x, &format!("{p}.post_attention_layernorm.weight"), dev)?;
        // MLP: down_proj(F.silu(gate_proj(h)) * up_proj(h)), each op its own f32 kernel
        let m = (kernels::torch_silu(&self.apply(&format!("{p}.mlp.gate_proj"), &h, dev)?)?
            * self.apply(&format!("{p}.mlp.up_proj"), &h, dev)?)?;
        Ok((x + self.apply(&format!("{p}.mlp.down_proj"), &m, dev)?)?)
    }

    /// The prompt's tokens to the last layer's output through the final norm, (T, 1024) f32
    pub fn encode(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let t = ids.len();
        let mask = attention_mask(&token_mask(ids), dev)?;
        let rope = kernels::gemma_rope_tables(t, HEAD_DIM, THETA, 1.0, dev)?;
        let mut x = self.embed(ids, dev)?;
        for l in 0..LAYERS {
            x = self.layer(l, &x, &mask, &rope, dev)?;
        }
        self.rms(&x, "model.norm.weight", dev)
    }
}
