//! Gemma 3 from a GGUF file, for the prompt writer: candle-transformers' `quantized_gemma3`
//! (https://github.com/huggingface/candle, MIT OR Apache-2.0) with two changes that keep a 12B
//! model inside a 24 GB card. Its arithmetic is unchanged.
//!
//! - The token embedding stays on the CPU. candle dequantizes it to f32 on the card (4 GB for
//!   Gemma 3 12B's 262,208 x 3,840), where a lookup only ever reads the prompt's few rows. The
//!   output projection is the same tensor and stays quantized on the card.
//! - The RoPE tables are built once for each of the two frequencies, for `MAX_SEQ_LEN` positions,
//!   and shared by the layers. candle builds one pair per layer for all 131,072 positions of
//!   Gemma's context (130 MB a layer, 6 GB for 48 layers).

use anyhow::{bail, Result};
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

/// Positions the RoPE tables cover: the writer's instructions, a prompt to build on and its
/// answer take a few thousand at most.
const MAX_SEQ_LEN: usize = 8192;
const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

#[derive(Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate.forward(xs)?)? * self.up.forward(xs)?)?;
        self.down.forward(&gated)
    }
}

#[derive(Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> candle_core::Result<Self> {
        let theta: Vec<_> = (0..head_dim).step_by(2).map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32)).collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self { sin: idx_theta.sin()?, cos: idx_theta.cos()? })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, index_pos: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        Ok((candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?, candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?))
    }
}

#[derive(Clone)]
struct LayerWeights {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    sliding_window_size: Option<usize>,
    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn mask(&self, b_sz: usize, seq_len: usize, index_pos: usize, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let window = self.sliding_window_size;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i < j || window.is_some_and(|w| j + w < i) { 0u32 } else { 1u32 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask =
            if index_pos > 0 { Tensor::cat(&[&Tensor::zeros((seq_len, index_pos), DType::F32, device)?, &mask], D::Minus1)? } else { mask };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?.to_dtype(dtype)
    }

    fn forward_attn(&mut self, x: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.wq.forward(x)?.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = self.wk.forward(x)?.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = self.wv.forward(x)?.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let (q, k) = self.rotary_embedding.apply(&q, &k, index_pos)?;
        let (k, v) = match &self.kv_cache {
            Some((k_cache, v_cache)) if index_pos > 0 => (Tensor::cat(&[k_cache, &k], 2)?, Tensor::cat(&[v_cache, &v], 2)?),
            _ => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }
        let attn_output = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
        self.wo.forward(&attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.q_dim))?)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(ct: gguf_file::Content, reader: &mut R, device: &Device) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(&format!("gemma3.{s}")) {
            None => bail!("cannot find gemma3.{s} in metadata"),
            Some(v) => Ok(v),
        };
        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;
        let sliding_window_type =
            md_get("attention.sliding_window_type").and_then(|m| Ok(m.to_u32()? as usize)).unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);
        let rope_freq_base = md_get("rope.freq_base").and_then(|m| Ok(m.to_f32()?)).unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_sliding = md_get("rope.local_freq_base").and_then(|m| Ok(m.to_f32()?)).unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let rope_global = RotaryEmbedding::new(key_length, rope_freq_base, device)?;
        let rope_sliding = RotaryEmbedding::new(key_length, rope_freq_base_sliding, device)?;

        let cpu = Device::Cpu;
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", &cpu)?.dequantize(&cpu)?;
        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps)?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?, // tied to the embedding
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let mut qt = |name: &str| -> Result<QTensor> { Ok(ct.tensor(reader, &format!("{prefix}.{name}.weight"), device)?) };
            let (wq, wk, wv, wo) = (qt("attn_q")?, qt("attn_k")?, qt("attn_v")?, qt("attn_output")?);
            let q_norm = RmsNorm::from_qtensor(qt("attn_q_norm")?, rms_norm_eps)?;
            let k_norm = RmsNorm::from_qtensor(qt("attn_k_norm")?, rms_norm_eps)?;
            let attention_norm = RmsNorm::from_qtensor(qt("attn_norm")?, rms_norm_eps)?;
            let post_attention_norm = RmsNorm::from_qtensor(qt("post_attention_norm")?, rms_norm_eps)?;
            let ffn_norm = RmsNorm::from_qtensor(qt("ffn_norm")?, rms_norm_eps)?;
            let post_ffn_norm = RmsNorm::from_qtensor(qt("post_ffw_norm")?, rms_norm_eps)?;
            let mlp = Mlp {
                gate: QMatMul::from_qtensor(qt("ffn_gate")?)?,
                up: QMatMul::from_qtensor(qt("ffn_up")?)?,
                down: QMatMul::from_qtensor(qt("ffn_down")?)?,
            };
            // the sliding-window pattern is not in the file: every sixth layer attends to everything
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            layers.push(LayerWeights {
                wq: QMatMul::from_qtensor(wq)?,
                wk: QMatMul::from_qtensor(wk)?,
                wv: QMatMul::from_qtensor(wv)?,
                wo: QMatMul::from_qtensor(wo)?,
                q_norm,
                k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim: head_count * key_length,
                sliding_window_size: is_sliding.then_some(sliding_window_size),
                rotary_embedding: if is_sliding { rope_sliding.clone() } else { rope_global.clone() },
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            });
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        if index_pos + seq_len > MAX_SEQ_LEN {
            bail!("the writer's context is {MAX_SEQ_LEN} tokens; this is {}", index_pos + seq_len);
        }
        let mut layer_in =
            (self.tok_embeddings.forward(&x.to_device(&Device::Cpu)?)?.to_device(x.device())? * (self.embedding_length as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 { None } else { Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?) };
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = (layer.post_attention_norm.forward(&x)? + &layer_in)?;
            let y = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            layer_in = (layer.post_ffn_norm.forward(&y)? + &x)?;
        }
        let x = self.norm.forward(&layer_in.i((.., seq_len - 1, ..))?)?;
        Ok(self.output.forward(&x)?)
    }
}
