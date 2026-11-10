//! ComfyUI-RMBG's SAM3Segment: Meta's SAM 3 image model (its detector) as the node runs it, under
//! torch's bf16 autocast, from the pack's sam3.pt. Each op as torch 2.11 runs it: the picture's
//! transform (torchvision's resize on the card: torch's antialiased bilinear, rounded back to
//! uint8), the ViT's convolution, linears, FlashAttention and GELU in bf16 with its LayerNorms and
//! residual stream in f32, RoPE as a complex product in f32.
//!
//! In progress: the transform and the ViT so far, checked stage by stage against the pack's own
//! code (refs/sam3_ref.py, debug-sam3).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::kernels;

const TRUNK: &str = "detector.backbone.vision_backbone.trunk";
/// The ViT's input side, its patch and its attention windows
const RES: usize = 1008;
const PATCH: usize = 14;
const WINDOW: usize = 24;
const HEADS: usize = 16;

struct Block {
    norm1: (Tensor, Tensor),
    /// freqs_cis (L, D / 2) complex64 as (L, D / 2, 2) f32, the checkpoint's own
    freqs: Tensor,
    qkv: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    /// windowed attention (24 x 24), else global (blocks 7, 15, 23 and 31)
    window: bool,
}

/// One level of the ViTDet neck: transposed 2x2 convolutions (two with a GELU between for x4, one
/// for x2, none for x1), then a 1x1 and a 3x3 convolution, all bf16 with their biases
struct Level {
    up: Vec<(Tensor, Tensor)>,
    c1: (Tensor, Tensor),
    c3: (Tensor, Tensor),
}

/// A bf16 convolution (or a transposed 2x2 one, stride 2) on channels-last memory, its bias added
/// after as torch adds it (in f32, rounded to bf16): x and the result (N, H, W, C) contiguous
fn conv_nhwc(x: &Tensor, p: &(Tensor, Tensor), pad: usize, transposed: bool) -> Result<Tensor> {
    let x = x.permute((0, 3, 1, 2))?;
    let y = if transposed {
        kernels::cudnn_conv_transpose_nhwc(&x, &p.0, &[0, 0], &[2, 2], &[1, 1], &[0, 0], 1)?
    } else {
        kernels::cudnn_conv_nhwc(&x, &p.0, &[pad, pad], &[1, 1], &[1, 1])?
    };
    Ok(kernels::ltx_add(&y.permute((0, 2, 3, 1))?, &p.1)?)
}

/// One of the text encoder's residual attention blocks (open_clip's, ls none)
struct TextBlock {
    ln1: (Tensor, Tensor),
    in_proj: (Tensor, Tensor),
    out_proj: (Tensor, Tensor),
    ln2: (Tensor, Tensor),
    fc: (Tensor, Tensor),
    proj: (Tensor, Tensor),
}

/// The text encoder's context: 32 tokens
const CONTEXT: usize = 32;

pub struct Sam3 {
    /// the patch embedding's weight in bf16 (autocast's cast; no bias)
    patch: Tensor,
    /// pos_embed without its cls token, 24 x 24 tiled to (1, 72, 72, 1024) f32 (tile_abs_pos)
    pos: Tensor,
    ln_pre: (Tensor, Tensor),
    blocks: Vec<Block>,
    /// the neck's x4, x2 and x1 levels (its x0.5 one is computed and dropped: scalp 1)
    levels: Vec<Level>,
    tok_embed: Tensor,
    tok_pos: Tensor,
    text_blocks: Vec<TextBlock>,
    ln_final: (Tensor, Tensor),
    resizer: (Tensor, Tensor),
    tokenizer: crate::tokenizer::ClipTokenizer,
    geo_cls: Tensor,
    geo_proj: (Tensor, Tensor),
    geo_norm: (Tensor, Tensor),
    geo_layers: Vec<EncLayer>,
    geo_out_norm: (Tensor, Tensor),
    fusion: Vec<EncLayer>,
    dec: Decoder,
    scoring: Scoring,
    seg: SegHead,
}

type Dump<'a> = Option<&'a mut HashMap<String, Tensor>>;

fn keep(dbg: &mut Dump, name: &str, t: &Tensor) {
    if let Some(d) = dbg.as_deref_mut() {
        d.insert(name.to_string(), t.clone());
    }
}

/// LayerNorm (eps 1e-5) over the last dimension, in f32 as autocast runs it (a bf16 input is cast
/// up first), torch's kernel
fn ln(x: &Tensor, p: &(Tensor, Tensor)) -> Result<Tensor> {
    let c = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / c;
    Ok(kernels::layer_norm(&x.to_dtype(DType::F32)?.reshape((rows, c))?, &p.0, &p.1, 1e-5)?.reshape(x.dims())?)
}

/// An nn.MultiheadAttention's weights as autocast hands them to F.linear: bf16, the packed
/// in-projection whole and in the slices F.multi_head_attention_forward takes of it
struct Mha {
    all: (Tensor, Tensor),
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    kv: (Tensor, Tensor),
    out: (Tensor, Tensor),
}

/// How F.multi_head_attention_forward projects (_in_projection_packed): one linear when query,
/// key and value are one tensor, the query's and a packed key-value one when only key and value
/// are, else three. One wide GEMM and three narrow ones can round differently.
#[derive(Clone, Copy)]
enum Proj {
    Same,
    KeyIsValue,
    Separate,
}

/// nn.MultiheadAttention (8 heads, need_weights off) under autocast on rows (L, E), batch 1: the
/// in-projections in bf16, SDPA (FlashAttention without a mask, the memory-efficient kernel with
/// one: an additive (Lq, Lk) bf16 mask), out_proj in bf16: (Lq, E) bf16
fn attend(m: &Mha, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, proj: Proj, mask: Option<&Tensor>) -> Result<Tensor> {
    let e = q_in.dim(1)?;
    let (q, k, v) = match proj {
        Proj::Same => {
            let p = linear_bf16(q_in, &m.all)?;
            (p.narrow(1, 0, e)?.contiguous()?, p.narrow(1, e, e)?.contiguous()?, p.narrow(1, 2 * e, e)?.contiguous()?)
        }
        Proj::KeyIsValue => {
            let kv = linear_bf16(k_in, &m.kv)?;
            (linear_bf16(q_in, &m.q)?, kv.narrow(1, 0, e)?.contiguous()?, kv.narrow(1, e, e)?.contiguous()?)
        }
        Proj::Separate => (linear_bf16(q_in, &m.q)?, linear_bf16(k_in, &m.k)?, linear_bf16(v_in, &m.v)?),
    };
    let (q, k, v) = (q.unsqueeze(0)?, k.unsqueeze(0)?, v.unsqueeze(0)?);
    let o = match mask {
        None => crate::ltx::sdpa(&q, &k, &v, 8)?,
        Some(mk) => kernels::torch_mem_eff_attention(&q, &k, &v, 8, kernels::HeadLayout::Btk, Some(mk))?,
    };
    linear_bf16(&o.squeeze(0)?, &m.out)
}

/// nn.MultiheadAttention called with need_weights left on (SAM3's decoder self- and text
/// attention): F.multi_head_attention_forward's explicit path, not SDPA. q scaled by
/// sqrt(1 / head_dim) in bf16, q k^T per head on cuBLAS's strided batched bf16 GEMM (a baddbmm onto
/// the key-padding mask when there is one), softmax in f32, its bf16 cast times v likewise, then
/// out_proj: (Lq, E) bf16
fn attend_math(m: &Mha, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, proj: Proj, pad: Option<&[bool]>) -> Result<Tensor> {
    let e = q_in.dim(1)?;
    let (q, k, v) = match proj {
        Proj::Same => {
            let p = linear_bf16(q_in, &m.all)?;
            (p.narrow(1, 0, e)?.contiguous()?, p.narrow(1, e, e)?.contiguous()?, p.narrow(1, 2 * e, e)?.contiguous()?)
        }
        Proj::KeyIsValue => {
            let kv = linear_bf16(k_in, &m.kv)?;
            (linear_bf16(q_in, &m.q)?, kv.narrow(1, 0, e)?.contiguous()?, kv.narrow(1, e, e)?.contiguous()?)
        }
        Proj::Separate => (linear_bf16(q_in, &m.q)?, linear_bf16(k_in, &m.k)?, linear_bf16(v_in, &m.v)?),
    };
    let (lq, lk, heads) = (q.dim(0)?, k.dim(0)?, 8);
    let d = e / heads;
    let scale = (1.0f64 / d as f64).sqrt() as f32;
    let qs = (q.to_dtype(DType::F32)? * scale as f64)?.to_dtype(DType::BF16)?;
    // q_scaled @ k^T: bmm's transposed result, A = k (lda E, stride d), B = q (ldb E, stride d)
    let c = match pad {
        Some(p) => Some(pad_mask(p, lq, q.device())?.unsqueeze(0)?.repeat((heads, 1, 1))?),
        None => None,
    };
    let s = kernels::torch_bgemm_bf16(true, false, lk, lq, d, &k, e, d, &qs, e, d, c.as_ref(), lk, lq * lk, heads, heads * lq * lk)?;
    let w = kernels::torch_softmax_lastdim_f32(&s.to_dtype(DType::F32)?.reshape((heads * lq, lk))?)?.to_dtype(DType::BF16)?;
    // attn @ v: A = v (lda E, stride d), B = attn (ldb Lk, stride Lq * Lk)
    let o = kernels::torch_bgemm_bf16(false, false, d, lq, lk, &v, e, d, &w, lk, lq * lk, None, d, lq * d, heads, heads * lq * d)?;
    let o = o.reshape((heads, lq, d))?.transpose(0, 1)?.contiguous()?.reshape((lq, e))?;
    linear_bf16(&o, &m.out)
}

/// A pre-norm TransformerEncoderLayer (encoder.py): self-attention, cross-attention, ReLU FFN
struct EncLayer {
    self_attn: Mha,
    cross: Mha,
    lin1: (Tensor, Tensor),
    lin2: (Tensor, Tensor),
    norm1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    norm3: (Tensor, Tensor),
}

/// MLP (model_misc.py) under autocast: bf16 linears, ReLU between them
fn mlp_bf16(x: &Tensor, layers: &[(Tensor, Tensor)]) -> Result<Tensor> {
    let mut x = x.clone();
    for (i, l) in layers.iter().enumerate() {
        x = linear_bf16(&x, l)?;
        if i + 1 < layers.len() {
            x = x.relu()?;
        }
    }
    Ok(x)
}

/// A TransformerDecoderLayer (decoder.py, post-norm): self-attention, text cross-attention, image
/// cross-attention with the box bias, and an FFN run with autocast off (f32, TF32 GEMMs)
struct DecLayer {
    self_attn: Mha,
    ca_text: Mha,
    cross: Mha,
    norm1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    norm3: (Tensor, Tensor),
    catext_norm: (Tensor, Tensor),
    lin1: (Tensor, Tensor),
    lin2: (Tensor, Tensor),
}

struct Decoder {
    layers: Vec<DecLayer>,
    norm: (Tensor, Tensor),
    query_embed: Tensor,
    reference_points: Tensor,
    presence_token: Tensor,
    bbox: Vec<(Tensor, Tensor)>,
    rpb_x: Vec<(Tensor, Tensor)>,
    rpb_y: Vec<(Tensor, Tensor)>,
    presence_head: Vec<(Tensor, Tensor)>,
    presence_norm: (Tensor, Tensor),
    ref_point_head: Vec<(Tensor, Tensor)>,
}

/// What the decoder hands on: each layer's normed queries (Q, 256) f32, the reference boxes each
/// layer started from (Q, 4) f32, each layer's presence logit (1, 1) bf16, and the last layer's
/// boxes and presence features
struct Decoded {
    hs: Vec<Tensor>,
    refs: Vec<Tensor>,
    presence: Vec<Tensor>,
    boxes: Tensor,
    presence_feats: Tensor,
}

/// DotProductScoring: the prompt through its residual MLP and norm, mean-pooled over its valid
/// tokens, projected; the queries projected; their dot products scaled and clamped
struct Scoring {
    prompt_mlp: Vec<(Tensor, Tensor)>,
    prompt_norm: (Tensor, Tensor),
    prompt_proj: (Tensor, Tensor),
    hs_proj: (Tensor, Tensor),
}

/// UniversalSegmentationHead: the fused image tokens cross-attend the prompt, the pixel decoder
/// climbs the neck's finer levels, a 1x1 convolution makes per-pixel embeddings, and each query's
/// mask is its embedding's dot product with them
struct SegHead {
    cross: Mha,
    cross_norm: (Tensor, Tensor),
    convs: Vec<(Tensor, Tensor)>,
    norms: Vec<(Tensor, Tensor)>,
    instance: (Tensor, Tensor),
    mask_embed: Vec<(Tensor, Tensor)>,
}

/// An additive key-padding mask as SDPA receives it: -inf on padded keys, 0 elsewhere, one row
/// per query, bf16
fn pad_mask(pad: &[bool], queries: usize, dev: &Device) -> Result<Tensor> {
    let row: Vec<f32> = pad.iter().map(|&p| if p { f32::NEG_INFINITY } else { 0.0 }).collect();
    Ok(Tensor::from_vec(row, (1, pad.len()), dev)?.repeat((queries, 1))?.to_dtype(DType::BF16)?)
}

/// F.linear under autocast: x and the weights in bf16, one addmm over the flattened rows, on
/// cuBLASLt with the bias in its epilogue unless an input or output width is 1, where torch's addmm
/// runs cublasGemmEx on a result filled with the bias
fn linear_bf16(x: &Tensor, p: &(Tensor, Tensor)) -> Result<Tensor> {
    let c = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / c;
    let n = p.0.dim(0)?;
    let mut dims = x.dims().to_vec();
    *dims.last_mut().unwrap() = n;
    let x2 = x.to_dtype(DType::BF16)?.reshape((rows, c))?;
    let y = if c > 1 && n > 1 { kernels::torch_linear(&x2, &p.0, &p.1)? } else { kernels::torch_linear_bias_gemm(&x2, &p.0, &p.1)? };
    Ok(y.reshape(dims)?)
}

impl Sam3 {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut pth = candle_core::pickle::PthTensors::new(path, None)?;
        if pth.tensor_infos().keys().all(|k| !k.starts_with("detector.")) {
            pth = candle_core::pickle::PthTensors::new(path, Some("model"))?;
        }
        let get = |n: &str| -> Result<Tensor> { Ok(pth.get(n)?.with_context(|| format!("{}: no {n}", path.display()))?.to_device(dev)?) };
        let bf16 = |n: &str| -> Result<Tensor> { Ok(get(n)?.to_dtype(DType::BF16)?) };
        let pair = |p: &str| -> Result<(Tensor, Tensor)> { Ok((get(&format!("{p}.weight"))?, get(&format!("{p}.bias"))?)) };
        let pair_bf16 = |p: &str| -> Result<(Tensor, Tensor)> { Ok((bf16(&format!("{p}.weight"))?, bf16(&format!("{p}.bias"))?)) };
        let pos = get(&format!("{TRUNK}.pos_embed"))?;
        let (_, n, c) = pos.dims3()?;
        let side = ((n - 1) as f64).sqrt() as usize;
        let grid = pos.narrow(1, 1, n - 1)?.reshape((side, side, c))?;
        let tiles = RES / PATCH / side;
        let row = Tensor::cat(&vec![&grid; tiles], 1)?;
        let pos = Tensor::cat(&vec![&row; tiles], 0)?.unsqueeze(0)?.contiguous()?;
        let depth = (0..).take_while(|i| pth.tensor_infos().contains_key(&format!("{TRUNK}.blocks.{i}.norm1.weight"))).count();
        let blocks = (0..depth)
            .map(|i| -> Result<Block> {
                let b = format!("{TRUNK}.blocks.{i}");
                Ok(Block {
                    norm1: pair(&format!("{b}.norm1"))?,
                    freqs: get(&format!("{b}.attn.freqs_cis"))?,
                    qkv: pair_bf16(&format!("{b}.attn.qkv"))?,
                    proj: pair_bf16(&format!("{b}.attn.proj"))?,
                    norm2: pair(&format!("{b}.norm2"))?,
                    fc1: pair_bf16(&format!("{b}.mlp.fc1"))?,
                    fc2: pair_bf16(&format!("{b}.mlp.fc2"))?,
                    window: ![7, 15, 23, 31].contains(&i),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let neck = "detector.backbone.vision_backbone.convs";
        let levels = [vec!["dconv_2x2_0", "dconv_2x2_1"], vec!["dconv_2x2"], vec![]]
            .iter()
            .enumerate()
            .map(|(i, up)| -> Result<Level> {
                Ok(Level {
                    up: up.iter().map(|u| pair_bf16(&format!("{neck}.{i}.{u}"))).collect::<Result<Vec<_>>>()?,
                    c1: pair_bf16(&format!("{neck}.{i}.conv_1x1"))?,
                    c3: pair_bf16(&format!("{neck}.{i}.conv_3x3"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let te = "detector.backbone.language_backbone";
        let text_layers =
            (0..).take_while(|i| pth.tensor_infos().contains_key(&format!("{te}.encoder.transformer.resblocks.{i}.ln_1.weight"))).count();
        let text_blocks = (0..text_layers)
            .map(|i| -> Result<TextBlock> {
                let b = format!("{te}.encoder.transformer.resblocks.{i}");
                Ok(TextBlock {
                    ln1: pair(&format!("{b}.ln_1"))?,
                    in_proj: (bf16(&format!("{b}.attn.in_proj_weight"))?, bf16(&format!("{b}.attn.in_proj_bias"))?),
                    out_proj: pair_bf16(&format!("{b}.attn.out_proj"))?,
                    ln2: pair(&format!("{b}.ln_2"))?,
                    fc: pair_bf16(&format!("{b}.mlp.c_fc"))?,
                    proj: pair_bf16(&format!("{b}.mlp.c_proj"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mha = |p: &str| -> Result<Mha> {
            let (w, b) = (bf16(&format!("{p}.in_proj_weight"))?, bf16(&format!("{p}.in_proj_bias"))?);
            let e = w.dim(1)?;
            let part = |from: usize, n: usize| -> Result<(Tensor, Tensor)> {
                Ok((w.narrow(0, from, n)?.contiguous()?, b.narrow(0, from, n)?.contiguous()?))
            };
            Ok(Mha {
                q: part(0, e)?,
                k: part(e, e)?,
                v: part(2 * e, e)?,
                kv: part(e, 2 * e)?,
                all: (w.clone(), b.clone()),
                out: pair_bf16(&format!("{p}.out_proj"))?,
            })
        };
        let enc_layer = |p: &str| -> Result<EncLayer> {
            Ok(EncLayer {
                self_attn: mha(&format!("{p}.self_attn"))?,
                cross: mha(&format!("{p}.cross_attn_image"))?,
                lin1: pair_bf16(&format!("{p}.linear1"))?,
                lin2: pair_bf16(&format!("{p}.linear2"))?,
                norm1: pair(&format!("{p}.norm1"))?,
                norm2: pair(&format!("{p}.norm2"))?,
                norm3: pair(&format!("{p}.norm3"))?,
            })
        };
        let count = |p: &str| (0..).take_while(|i| pth.tensor_infos().contains_key(&format!("{p}.{i}.norm1.weight"))).count();
        let geo = "detector.geometry_encoder";
        let geo_layers =
            (0..count(&format!("{geo}.encode"))).map(|i| enc_layer(&format!("{geo}.encode.{i}"))).collect::<Result<Vec<_>>>()?;
        let fe = "detector.transformer.encoder.layers";
        let fusion = (0..count(fe)).map(|i| enc_layer(&format!("{fe}.{i}"))).collect::<Result<Vec<_>>>()?;
        let de = "detector.transformer.decoder";
        let mlp = |p: &str| -> Result<Vec<(Tensor, Tensor)>> {
            (0..)
                .take_while(|i| pth.tensor_infos().contains_key(&format!("{p}.layers.{i}.weight")))
                .map(|i| pair_bf16(&format!("{p}.layers.{i}")))
                .collect()
        };
        let dec_layers = (0..count(&format!("{de}.layers")))
            .map(|i| -> Result<DecLayer> {
                let p = format!("{de}.layers.{i}");
                Ok(DecLayer {
                    self_attn: mha(&format!("{p}.self_attn"))?,
                    ca_text: mha(&format!("{p}.ca_text"))?,
                    cross: mha(&format!("{p}.cross_attn"))?,
                    norm1: pair(&format!("{p}.norm1"))?,
                    norm2: pair(&format!("{p}.norm2"))?,
                    norm3: pair(&format!("{p}.norm3"))?,
                    catext_norm: pair(&format!("{p}.catext_norm"))?,
                    lin1: pair(&format!("{p}.linear1"))?,
                    lin2: pair(&format!("{p}.linear2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dec = Decoder {
            layers: dec_layers,
            norm: pair(&format!("{de}.norm"))?,
            query_embed: get(&format!("{de}.query_embed.weight"))?,
            reference_points: get(&format!("{de}.reference_points.weight"))?,
            presence_token: get(&format!("{de}.presence_token.weight"))?,
            bbox: mlp(&format!("{de}.bbox_embed"))?,
            rpb_x: mlp(&format!("{de}.boxRPB_embed_x"))?,
            rpb_y: mlp(&format!("{de}.boxRPB_embed_y"))?,
            presence_head: mlp(&format!("{de}.presence_token_head"))?,
            presence_norm: pair(&format!("{de}.presence_token_out_norm"))?,
            ref_point_head: mlp(&format!("{de}.ref_point_head"))?,
        };
        let sc = "detector.dot_prod_scoring";
        let scoring = Scoring {
            prompt_mlp: mlp(&format!("{sc}.prompt_mlp"))?,
            prompt_norm: pair(&format!("{sc}.prompt_mlp.out_norm"))?,
            prompt_proj: pair_bf16(&format!("{sc}.prompt_proj"))?,
            hs_proj: pair_bf16(&format!("{sc}.hs_proj"))?,
        };
        let sh = "detector.segmentation_head";
        let stages = (0..).take_while(|i| pth.tensor_infos().contains_key(&format!("{sh}.pixel_decoder.conv_layers.{i}.weight"))).count();
        let seg = SegHead {
            cross: mha(&format!("{sh}.cross_attend_prompt"))?,
            cross_norm: pair(&format!("{sh}.cross_attn_norm"))?,
            convs: (0..stages).map(|i| pair_bf16(&format!("{sh}.pixel_decoder.conv_layers.{i}"))).collect::<Result<Vec<_>>>()?,
            norms: (0..stages).map(|i| pair(&format!("{sh}.pixel_decoder.norms.{i}"))).collect::<Result<Vec<_>>>()?,
            instance: pair_bf16(&format!("{sh}.instance_seg_head"))?,
            mask_embed: mlp(&format!("{sh}.mask_predictor.mask_embed"))?,
        };
        Ok(Sam3 {
            scoring,
            seg,
            dec,
            geo_cls: get(&format!("{geo}.cls_embed.weight"))?,
            geo_proj: pair_bf16(&format!("{geo}.final_proj"))?,
            geo_norm: pair(&format!("{geo}.norm"))?,
            geo_layers,
            geo_out_norm: pair(&format!("{geo}.encode_norm"))?,
            fusion,
            patch: bf16(&format!("{TRUNK}.patch_embed.proj.weight"))?,
            pos,
            ln_pre: pair(&format!("{TRUNK}.ln_pre"))?,
            blocks,
            levels,
            tok_embed: get(&format!("{te}.encoder.token_embedding.weight"))?,
            tok_pos: get(&format!("{te}.encoder.positional_embedding"))?,
            text_blocks,
            ln_final: pair(&format!("{te}.encoder.ln_final"))?,
            resizer: pair_bf16(&format!("{te}.resizer"))?,
            tokenizer: crate::tokenizer::ClipTokenizer::new()?,
        })
    }

    /// One ViT block on (1, H, W, C) f32: x + attn(norm1(x)) (windowed or global), then
    /// x + mlp(norm2(x)), the residual stream in f32
    fn block(&self, b: &Block, x: &Tensor) -> Result<Tensor> {
        let (_, h, w, c) = x.dims4()?;
        let d = c / HEADS;
        let n = ln(x, &b.norm1)?;
        // window_partition: (1, h/24, 24, w/24, 24, C) to (windows, 24 * 24, C)
        let (gh, gw) = (h / WINDOW, w / WINDOW);
        let (n, nb, l) = if b.window {
            let t = n.reshape((1, gh, WINDOW, gw, WINDOW, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
            (t.reshape((gh * gw, WINDOW * WINDOW, c))?, gh * gw, WINDOW * WINDOW)
        } else {
            (n.reshape((1, h * w, c))?, 1, h * w)
        };
        let qkv = linear_bf16(&n, &b.qkv)?.reshape((nb * l, 3 * c))?;
        let (q, k, v) = kernels::sam3_rope_qkv(&qkv, &b.freqs, l, c, d / 2)?;
        let o = crate::ltx::sdpa(&q.reshape((nb, l, c))?, &k.reshape((nb, l, c))?, &v.reshape((nb, l, c))?, HEADS)?;
        let o = linear_bf16(&o, &b.proj)?;
        let o = if b.window {
            o.reshape((1, gh, gw, WINDOW, WINDOW, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((1, h, w, c))?
        } else {
            o.reshape((1, h, w, c))?
        };
        let x = (x + o.to_dtype(DType::F32)?)?;
        let m = linear_bf16(&ln(&x, &b.norm2)?, &b.fc1)?;
        let m = linear_bf16(&kernels::torch_gelu_erf_bf16(&m)?, &b.fc2)?;
        Ok((x + m.to_dtype(DType::F32)?)?)
    }

    /// The trunk on the transformed picture (1, 3, 1008, 1008) f32: (1, 72, 72, 1024) f32
    fn vit(&self, x: &Tensor, dbg: &mut Dump) -> Result<Tensor> {
        // patch_embed: autocast's bf16 convolution, NCHW memory
        let p = kernels::cudnn_conv(&x.to_dtype(DType::BF16)?, &self.patch, &[0, 0], &[PATCH, PATCH], &[1, 1])?.permute((0, 2, 3, 1))?;
        keep(dbg, "patch_embed", &p);
        let x = ln(&p.to_dtype(DType::F32)?.broadcast_add(&self.pos)?, &self.ln_pre)?;
        keep(dbg, "ln_pre", &x);
        let mut x = x;
        for (i, b) in self.blocks.iter().enumerate() {
            x = self.block(b, &x)?;
            if matches!(i, 0 | 1 | 7 | 8 | 15 | 31) {
                keep(dbg, &format!("block{i}"), &x);
            }
        }
        Ok(x)
    }
}

impl Sam3 {
    /// SimpleTokenizer (open_clip's, context 32): start, the prompt's BPE ids, end, cut to 32
    /// with the end kept, padded with 0
    fn tokens(&self, prompt: &str) -> Vec<u32> {
        let mut t = vec![49406];
        t.extend(self.tokenizer.encode(prompt));
        t.push(49407);
        if t.len() > CONTEXT {
            t.truncate(CONTEXT);
            t[CONTEXT - 1] = 49407;
        }
        t.resize(CONTEXT, 0);
        t
    }

    /// VETextEncoder on a prompt: the 32 tokens through the 24 blocks (causal attention on torch's
    /// memory-efficient kernel in bf16), ln_final, the resizer to 256: (32, 256) bf16, and which
    /// tokens are padding
    fn text(&self, prompt: &str, dbg: &mut Dump) -> Result<(Tensor, Vec<bool>)> {
        let dev = self.tok_embed.device();
        let ids = self.tokens(prompt);
        let pad: Vec<bool> = ids.iter().map(|&t| t == 0).collect();
        let idx = Tensor::from_vec(ids.clone(), CONTEXT, dev)?;
        keep(dbg, "tokens", &Tensor::from_vec(ids.iter().map(|&t| t as f32).collect::<Vec<_>>(), (1, CONTEXT), dev)?);
        let embeds = self.tok_embed.index_select(&idx, 0)?;
        keep(dbg, "text.2", &embeds);
        let mut x = (&embeds + &self.tok_pos)?;
        // build_causal_mask: -inf above the diagonal, as autocast hands SDPA the mask (bf16)
        let mask: Vec<f32> = (0..CONTEXT * CONTEXT).map(|i| if i % CONTEXT > i / CONTEXT { f32::NEG_INFINITY } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, (CONTEXT, CONTEXT), dev)?.to_dtype(DType::BF16)?;
        let width = x.dim(1)?;
        for (i, b) in self.text_blocks.iter().enumerate() {
            let qkv = linear_bf16(&ln(&x, &b.ln1)?, &b.in_proj)?;
            let part = |j: usize| -> Result<Tensor> { Ok(qkv.narrow(1, j * width, width)?.contiguous()?.unsqueeze(0)?) };
            let o = kernels::torch_mem_eff_attention(&part(0)?, &part(1)?, &part(2)?, HEADS, kernels::HeadLayout::Btk, Some(&mask))?;
            let o = linear_bf16(&o.squeeze(0)?, &b.out_proj)?;
            x = (x + o.to_dtype(DType::F32)?)?;
            let m = linear_bf16(&ln(&x, &b.ln2)?, &b.fc)?;
            let m = linear_bf16(&kernels::torch_gelu_erf_bf16(&m)?, &b.proj)?;
            x = (x + m.to_dtype(DType::F32)?)?;
            if i == 0 || i + 1 == self.text_blocks.len() {
                keep(dbg, &format!("text_block{i}"), &x);
            }
        }
        let x = ln(&x, &self.ln_final)?;
        keep(dbg, "text_encoder.1", &x);
        let r = linear_bf16(&x, &self.resizer)?;
        keep(dbg, "text.1", &r);
        Ok((r, pad))
    }

    /// SequenceGeometryEncoder on the node's dummy prompt (no boxes, no points): its cls token
    /// through final_proj and norm, then pre-norm layers whose self-attention sees only that token
    /// (with a key-padding mask of zeros) and whose cross-attention reads the image with the
    /// position encoding on the keys, then encode_norm: (1, 256) f32
    fn geometry(&self, img: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let mut x = ln(&linear_bf16(&self.geo_cls, &self.geo_proj)?, &self.geo_norm)?;
        let keys = kernels::ltx_add(img, pos)?;
        let zero = pad_mask(&[false], 1, img.device())?;
        for l in &self.geo_layers {
            let h = ln(&x, &l.norm1)?;
            x = (x + attend(&l.self_attn, &h, &h, &h, Proj::Same, Some(&zero))?.to_dtype(DType::F32)?)?;
            let h = ln(&x, &l.norm2)?;
            x = (x + attend(&l.cross, &h, &keys, img, Proj::Separate, None)?.to_dtype(DType::F32)?)?;
            let h = ln(&x, &l.norm3)?;
            x = (x + linear_bf16(&linear_bf16(&h, &l.lin1)?.relu()?, &l.lin2)?.to_dtype(DType::F32)?)?;
        }
        ln(&x, &self.geo_out_norm)
    }

    /// TransformerEncoderFusion: the image tokens (L, 256) bf16 through pre-norm layers whose
    /// self-attention adds the position encoding to queries and keys and whose cross-attention reads
    /// the prompt (33, 256) f32 with its padding masked; the residual stream stays bf16
    fn fusion(&self, img: &Tensor, pos: &Tensor, prompt: &Tensor, pad: &[bool]) -> Result<Tensor> {
        let mask = pad_mask(pad, img.dim(0)?, img.device())?;
        let pos = pos.to_dtype(DType::F32)?;
        let mut x = img.clone();
        for l in &self.fusion {
            let h = ln(&x, &l.norm1)?;
            let qk = (&h + &pos)?;
            x = kernels::ltx_add(&x, &attend(&l.self_attn, &qk, &qk, &h, Proj::Separate, None)?)?;
            let h = ln(&x, &l.norm2)?;
            x = kernels::ltx_add(&x, &attend(&l.cross, &h, prompt, prompt, Proj::KeyIsValue, Some(&mask))?)?;
            let h = ln(&x, &l.norm3)?;
            x = kernels::ltx_add(&x, &linear_bf16(&linear_bf16(&h, &l.lin1)?.relu()?, &l.lin2)?)?;
        }
        Ok(x)
    }

    /// _get_rpb_matrix for the current boxes on an (h, w) map, with the presence token's zero row:
    /// (8, 1 + Q, h * w) bf16
    fn rpb(&self, boxes: &Tensor, h: usize, w: usize, li: usize, dbg: &mut Dump) -> Result<Tensor> {
        let (dx, dy) = kernels::sam3_rpb_deltas(boxes, h, w)?;
        keep(dbg, &format!("dec.rpbx{li}.in"), &dx);
        keep(dbg, &format!("dec.rpby{li}.in"), &dy);
        let dx = mlp_bf16(&dx, &self.dec.rpb_x)?;
        let dy = mlp_bf16(&dy, &self.dec.rpb_y)?;
        keep(dbg, &format!("dec.rpbx{li}"), &dx);
        keep(dbg, &format!("dec.rpby{li}"), &dy);
        Ok(kernels::sam3_rpb_bias(&dy, &dx)?)
    }

    /// TransformerDecoder (eval: no DAC) on the fused image tokens (L, 256) bf16 and their position
    /// encoding, the prompt (33, 256) f32 and its padding: 200 queries and the presence token
    /// through the layers, the boxes refined after each
    fn decoder(
        &self,
        memory: &Tensor,
        pos: &Tensor,
        prompt: &Tensor,
        pad: &[bool],
        (h, w): (usize, usize),
        dbg: &mut Dump,
    ) -> Result<Decoded> {
        let d = &self.dec;
        let dev = memory.device();
        let q = d.query_embed.dim(0)?;
        let mut x = d.query_embed.clone();
        let mut refs = kernels::sam3_refine(&d.reference_points, None)?;
        let mut presence = d.presence_token.clone();
        let keys = kernels::ltx_add(memory, pos)?;

        let zero_pos = Tensor::zeros((1, x.dim(1)?), DType::F32, dev)?;
        let (mut hs, mut all_refs, mut pres) = (vec![], vec![refs.clone()], vec![]);
        for (li, l) in d.layers.iter().enumerate() {
            let qpos = mlp_bf16(&kernels::sam3_box_sine(&refs)?, &d.ref_point_head)?;
            let bias = self.rpb(&refs, h, w, li, dbg)?;

            keep(dbg, &format!("dec.rpb{li}"), &bias.narrow(1, 1, q)?);
            let tpos = Tensor::cat(&[&zero_pos, &qpos.to_dtype(DType::F32)?], 0)?;
            let t = Tensor::cat(&[&presence, &x], 0)?;
            let qk = (&t + &tpos)?;
            let a = attend_math(&l.self_attn, &qk, &qk, &t, Proj::Separate, None)?;
            keep(dbg, &format!("dl{li}.self_attn.0"), &a);
            let t = ln(&(&t + a.to_dtype(DType::F32)?)?, &l.norm2)?;
            keep(dbg, &format!("dl{li}.norm2"), &t);
            let qt = (&t + &tpos)?;
            let a = attend_math(&l.ca_text, &qt, prompt, prompt, Proj::KeyIsValue, Some(pad))?;
            keep(dbg, &format!("dl{li}.ca_text.0"), &a);
            let t = ln(&(&t + a.to_dtype(DType::F32)?)?, &l.catext_norm)?;
            keep(dbg, &format!("dl{li}.catext_norm"), &t);
            let qt = (&t + &tpos)?;
            let a = attend(&l.cross, &qt, &keys, memory, Proj::Separate, Some(&bias))?;
            keep(dbg, &format!("dl{li}.cross_attn.0"), &a);
            let t = ln(&(&t + a.to_dtype(DType::F32)?)?, &l.norm1)?;
            keep(dbg, &format!("dl{li}.norm1"), &t);
            let f = kernels::torch_linear_bias_f32_tf32(&t, &l.lin1.0, &l.lin1.1)?;
            keep(dbg, &format!("dl{li}.linear1"), &f);
            let f = kernels::torch_linear_bias_f32_tf32(&f.relu()?, &l.lin2.0, &l.lin2.1)?;
            keep(dbg, &format!("dl{li}.linear2"), &f);
            let t = ln(&(&t + f)?, &l.norm3)?;
            keep(dbg, &format!("dl{li}.norm3"), &t);
            presence = t.narrow(0, 0, 1)?;
            x = t.narrow(0, 1, q)?;
            let normed = ln(&x, &d.norm)?;
            refs = kernels::sam3_refine(&refs, Some(&mlp_bf16(&normed, &d.bbox)?))?;
            if li + 1 < d.layers.len() {
                all_refs.push(refs.clone());
            }
            hs.push(normed);
            pres.push(mlp_bf16(&ln(&presence, &d.presence_norm)?, &d.presence_head)?);
        }
        Ok(Decoded { hs, refs: all_refs, presence: pres, boxes: refs, presence_feats: presence })
    }

    /// DotProductScoring on each layer's queries (200, 256) f32 against the prompt (33, 256) f32:
    /// the last layer's logits (200) bf16. The image model is built without the joint box score
    /// (inverse_sigmoid of the score times the presence), so the presence is applied only by the
    /// processor.
    fn logits(&self, hs: &[Tensor], prompt: &Tensor, pad: &[bool], dbg: &mut Dump) -> Result<Tensor> {
        let s = &self.scoring;
        let p = ln(&(mlp_bf16(prompt, &s.prompt_mlp)?.to_dtype(DType::F32)? + prompt)?, &s.prompt_norm)?;
        // mean_pool_text: (prompt * valid).sum(0) as torch's reduction adds the 33 rows (one thread
        // a column: four accumulators taking rows i, i + 4, ..., the last row into the first, then
        // combined in order), divided by the number of valid tokens
        let rows = p.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let c = rows[0].len();
        let valid: Vec<f32> = pad.iter().map(|&x| if x { 0.0 } else { 1.0 }).collect();
        let count = valid.iter().sum::<f32>().max(1.0);
        let mut pooled = vec![0f32; c];
        for (j, out) in pooled.iter_mut().enumerate() {
            let mut acc = [0f32; 4];
            let full = rows.len() / 4 * 4;
            for (t, row) in rows.iter().enumerate().take(full) {
                acc[t % 4] += row[j] * valid[t];
            }
            for (i, t) in (full..rows.len()).enumerate() {
                acc[i] += rows[t][j] * valid[t];
            }
            *out = ((acc[0] + acc[1]) + acc[2]) + acc[3];
            *out /= count;
        }
        let pooled = linear_bf16(&Tensor::from_vec(pooled, (1, c), prompt.device())?, &s.prompt_proj)?;
        let last = hs.len() - 1;
        let q = hs[last].dim(0)?;
        let proj = linear_bf16(&Tensor::stack(hs, 0)?, &s.hs_proj)?;
        // matmul (6, 1, 200, 256) x (1, 256, 1): bmm's call, ldc raised to m for its single column
        let n = hs.len();
        let sc = kernels::torch_bgemm_bf16(true, true, q, 1, c, &proj, c, q * c, &pooled, 1, 0, None, q, q, n, n * q)?;
        let sc = (sc.to_dtype(DType::F32)? * 0.0625)?.clamp(-12f32, 12f32)?.to_dtype(DType::BF16)?;
        keep(dbg, "scoring", &sc);
        Ok(sc.narrow(0, last * q, q)?)
    }

    /// The segmentation head: (Q, 288 * 288) bf16 mask logits for the last layer's queries
    fn masks(
        &self,
        levels: &[(Tensor, Tensor)],
        memory: &Tensor,
        prompt: &Tensor,
        pad: &[bool],
        queries: &Tensor,
        dbg: &mut Dump,
    ) -> Result<Tensor> {
        let s = &self.seg;
        let mask = pad_mask(pad, memory.dim(0)?, memory.device())?;
        let enc = kernels::ltx_add(&attend(&s.cross, &ln(memory, &s.cross_norm)?, prompt, prompt, Proj::KeyIsValue, Some(&mask))?, memory)?;
        let (_, h0, w0, c) = levels[levels.len() - 1].0.dims4()?;
        // the pixel decoder, coarse to fine: the finer level plus the coarser one upsampled
        // (nearest, in f32 as autocast runs interpolate), a bf16 3x3 convolution, GroupNorm(8) in
        // f32, ReLU
        let mut prev = enc.reshape((1, h0, w0, c))?.to_dtype(DType::F32)?;
        for (i, (lvl, _)) in levels[..levels.len() - 1].iter().rev().enumerate() {
            let (_, h, w, _) = lvl.dims4()?;
            let (ph, pw) = (prev.dim(1)?, prev.dim(2)?);
            anyhow::ensure!(h == 2 * ph && w == 2 * pw, "SAM3's pixel decoder: {ph}x{pw} to {h}x{w} is not a doubling");
            let up = prev.reshape((1, ph, 1, pw, 1, c))?.repeat((1, 1, 2, 1, 2, 1))?.reshape((1, h, w, c))?;
            let sum = (lvl.to_dtype(DType::F32)? + up)?;
            let conv = conv_nhwc(&sum.to_dtype(DType::BF16)?, &s.convs[i], 1, false)?;
            let g = kernels::group_norm(
                &conv.to_dtype(DType::F32)?.permute((0, 3, 1, 2))?.contiguous()?,
                &s.norms[i].0,
                &s.norms[i].1,
                8,
                1e-5,
                false,
            )?
            .relu()?;
            prev = g.permute((0, 2, 3, 1))?.contiguous()?;
        }
        let pixel = prev.permute((0, 3, 1, 2))?.contiguous()?;
        keep(dbg, "pixel_embed", &pixel);
        let (_, _, h, w) = pixel.dims4()?;
        let inst = kernels::cudnn_conv(&pixel.to_dtype(DType::BF16)?, &s.instance.0, &[0, 0], &[1, 1], &[1, 1])?;
        let inst = (inst.to_dtype(DType::F32)?.broadcast_add(&s.instance.1.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?)?)
            .to_dtype(DType::BF16)?;
        let emb = mlp_bf16(queries, &s.mask_embed)?;
        let q = emb.dim(0)?;
        // einsum("bqc,bchw->bqhw"): one bmm of one batch, torch's gemm with the pixels as A
        Ok(kernels::torch_bgemm_bf16(
            false,
            false,
            h * w,
            q,
            c,
            &inst,
            h * w,
            c * h * w,
            &emb,
            c,
            q * c,
            None,
            h * w,
            q * h * w,
            1,
            q * h * w,
        )?
        .reshape((q, h * w))?)
    }

    /// The neck on the trunk's (1, 72, 72, 1024) f32: each kept level (1, H, W, 256) bf16 with its
    /// position encoding (1, 256, H, W) bf16, finest first
    fn neck(&self, x: &Tensor, dbg: &mut Dump) -> Result<Vec<(Tensor, Tensor)>> {
        let xb = x.to_dtype(DType::BF16)?;
        let mut out = vec![];
        for (i, l) in self.levels.iter().enumerate() {
            let mut y = xb.clone();
            for (j, u) in l.up.iter().enumerate() {
                if j > 0 {
                    y = kernels::torch_gelu_erf_bf16(&y)?;
                }
                y = conv_nhwc(&y, u, 0, true)?;
            }
            let y = conv_nhwc(&conv_nhwc(&y, &l.c1, 0, false)?, &l.c3, 1, false)?;
            let (_, h, w, _) = y.dims4()?;
            let pos = kernels::sam3_pos_sine(h, w, y.device())?;
            keep(dbg, &format!("seghead.kw.backbone_feats.{i}"), &y.permute((0, 3, 1, 2))?);
            out.push((y, pos));
        }
        keep(dbg, "fusion.kw.src_pos.0", &out[2].1);
        Ok(out)
    }
}

impl Sam3 {
    /// Sam3Processor on one picture (3, H, W) f32 and a prompt: set_image, set_text_prompt and
    /// _forward_grounding. Each query's probability is sigmoid(logit) * sigmoid(presence) in
    /// bf16; those above the threshold (rounded to bf16, as torch compares) keep their masks, each
    /// upsampled to the picture (bilinear, in f32 as autocast runs interpolate) and put through a
    /// sigmoid: (N, H, W) f32 in the processor's order
    pub fn ground(&self, img: &Tensor, prompt: &str, conf: f64, dbg: &mut Dump) -> Result<Tensor> {
        let dev = self.tok_embed.device().clone();
        let (_, h, w) = img.dims3()?;
        let (x, u8s) = net_input(img, &dev)?;
        keep(dbg, "pic_u8", &u8s);
        keep(dbg, "net_in", &x.squeeze(0)?);
        let feat = self.vit(&x, dbg)?;
        let levels = self.neck(&feat, dbg)?;
        let prompt = match prompt.trim() {
            "" => "object",
            p => p,
        };
        let (text, mut pad) = self.text(prompt, dbg)?;
        // the encoder's image tokens: the coarsest kept level and its position encoding, (L, 256)
        let (top, fpos) = &levels[levels.len() - 1];
        let (_, fh, fw, c) = top.dims4()?;
        let tokens = top.reshape((fh * fw, c))?;
        let ipos = fpos.squeeze(0)?.flatten_from(1)?.t()?.contiguous()?;
        let geo = self.geometry(&tokens, &ipos)?;
        keep(dbg, "geometry.0", &geo);
        let prompt_feats = Tensor::cat(&[&text.to_dtype(DType::F32)?, &geo], 0)?;
        pad.push(false);
        keep(dbg, "fusion.kw.prompt", &prompt_feats);
        let memory = self.fusion(&tokens, &ipos, &prompt_feats, &pad)?;
        keep(dbg, "fusion.memory", &memory);
        let dec = self.decoder(&memory, &ipos, &prompt_feats, &pad, (fh, fw), dbg)?;

        keep(dbg, "decoder.0", &Tensor::stack(&dec.hs, 0)?);
        keep(dbg, "decoder.1", &Tensor::stack(&dec.refs, 0)?);
        keep(dbg, "decoder.2", &Tensor::stack(&dec.presence, 0)?);
        keep(dbg, "decoder.3", &dec.presence_feats);
        keep(dbg, "grounding.pred_boxes", &dec.boxes);
        let presence = &dec.presence[dec.presence.len() - 1];
        keep(dbg, "grounding.presence_logit_dec", presence);
        let logits = self.logits(&dec.hs, &prompt_feats, &pad, dbg)?;
        keep(dbg, "grounding.pred_logits", &logits);
        let masks = self.masks(&levels, &memory, &prompt_feats, &pad, &dec.hs[dec.hs.len() - 1], dbg)?;
        keep(dbg, "grounding.pred_masks", &masks);
        let sig = |t: &Tensor| -> Result<Tensor> {
            Ok(kernels::sam3_refine(&t.to_dtype(DType::F32)?.flatten_all()?, None)?.to_dtype(DType::BF16)?)
        };
        let probs = sig(&logits)?.to_dtype(DType::F32)?.broadcast_mul(&sig(presence)?.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
        let pv = probs.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let thr = half::bf16::from_f32(conf as f32).to_f32();
        let kept: Vec<u32> = (0..pv.len() as u32).filter(|&i| pv[i as usize] > thr).collect();
        if kept.is_empty() {
            return Ok(Tensor::zeros((0, h, w), DType::F32, &dev)?);
        }
        keep(dbg, "post.scores", &probs.index_select(&Tensor::from_vec(kept.clone(), kept.len(), &dev)?, 0)?);

        let side = (masks.dim(1)? as f64).sqrt() as usize;
        let sel = masks.index_select(&Tensor::from_vec(kept.clone(), kept.len(), &dev)?, 0)?.to_dtype(DType::F32)?.reshape((
            kept.len(),
            1,
            side,
            side,
        ))?;
        let up = kernels::torch_upsample_bilinear2d_f32(&sel, h, w, None)?;
        let p = kernels::sam3_refine(&up, None)?;
        keep(dbg, "post.masks_logits", &p);
        Ok(p.squeeze(1)?)
    }
}

/// What SAM3Segment returns besides the picture: its options
pub struct Opts {
    pub prompt: String,
    pub merged: bool,
    pub confidence: f64,
    pub max_segments: usize,
    pub segment_pick: usize,
    pub mask_blur: usize,
    pub mask_offset: i64,
    pub invert: bool,
    /// background "Alpha" (else "Color" with background_color)
    pub alpha: bool,
}

/// Pillow 12.3's _gaussian_blur_radius (BoxBlur.c): the box radius for three passes, its float
/// and double steps as C takes them
fn pil_gaussian_radius(radius: f32, passes: i32) -> f32 {
    let sigma2: f32 = radius * radius / passes as f32;
    let big_l = (12.0f64 * sigma2 as f64 + 1.0).sqrt() as f32;
    let l = ((big_l as f64 - 1.0) / 2.0).floor() as f32;
    let mut a: f32 = (2.0 * l + 1.0) * (l * (l + 1.0) - 3.0 * sigma2);
    a /= 6.0 * (sigma2 - (l + 1.0) * (l + 1.0));
    l + a
}

/// ImagingHorizontalBoxBlur on each row of an 8-bit (w, h) picture: ImagingLineBoxBlur8's
/// fixed-point running sum (24 fractional bits, the fractional radius as weight on the far pixels)
fn pil_box_rows(px: &mut [u8], w: usize, h: usize, radius_f: f32) {
    let radius = radius_f as i64;
    let ww = (16777216.0f32 / (radius_f * 2.0 + 1.0)) as u32;
    let fw = (16777216u32.wrapping_sub(((radius * 2 + 1) as u32).wrapping_mul(ww))) / 2;
    let (edge_a, edge_b) = ((radius + 1).min(w as i64), (w as i64 - radius - 1).max(0));
    let lastx = w as i64 - 1;
    let mut out = vec![0u8; w];
    for y in 0..h {
        let line = &px[y * w..(y + 1) * w];
        let at = |i: i64| line[i as usize] as u32;
        let mut acc: u32 = at(0) * (radius as u32 + 1);
        for x in 0..edge_a - 1 {
            acc = acc.wrapping_add(at(x));
        }
        acc = acc.wrapping_add(at(lastx).wrapping_mul((radius - edge_a + 1) as u32));
        let mut step = |acc: &mut u32, x: i64, sub: i64, add: i64, left: i64, right: i64| {
            *acc = acc.wrapping_add(at(add)).wrapping_sub(at(sub));
            let bulk = acc.wrapping_mul(ww).wrapping_add((at(left) + at(right)).wrapping_mul(fw));
            out[x as usize] = (bulk.wrapping_add(1 << 23) >> 24) as u8;
        };
        if edge_a <= edge_b {
            for x in 0..edge_a {
                step(&mut acc, x, 0, x + radius, 0, x + radius + 1);
            }
            for x in edge_a..edge_b {
                step(&mut acc, x, x - radius - 1, x + radius, x - radius - 1, x + radius + 1);
            }
            for x in edge_b..=lastx {
                step(&mut acc, x, x - radius - 1, lastx, x - radius - 1, lastx);
            }
        } else {
            for x in 0..edge_b {
                step(&mut acc, x, 0, x + radius, 0, x + radius + 1);
            }
            for x in edge_b..edge_a {
                step(&mut acc, x, 0, lastx, 0, lastx);
            }
            for x in edge_a..=lastx {
                step(&mut acc, x, x - radius - 1, lastx, x - radius - 1, lastx);
            }
        }
        px[y * w..(y + 1) * w].copy_from_slice(&out);
    }
}

fn transpose_u8(px: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut t = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = px[y * w + x];
        }
    }
    t
}

/// ImageFilter.GaussianBlur(radius) on an L picture, as Pillow 12.3 runs it: three box passes
/// along rows, the picture transposed, three along its rows, transposed back
fn pil_gaussian_blur(px: &mut Vec<u8>, w: usize, h: usize, radius: f32) {
    let r = pil_gaussian_radius(radius, 3);
    if r != 0.0 {
        for _ in 0..3 {
            pil_box_rows(px, w, h, r);
        }
        let mut t = transpose_u8(px, w, h);
        for _ in 0..3 {
            pil_box_rows(&mut t, h, w, r);
        }
        *px = transpose_u8(&t, h, w);
    }
}

/// ImageFilter.MaxFilter / MinFilter(size) on an L picture: its edges replicated by size / 2
/// (Image.filter's expand), then each pixel's neighbourhood's max or min
fn pil_rank(px: &[u8], w: usize, h: usize, size: usize, max: bool) -> Vec<u8> {
    let m = (size / 2) as i64;
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut v = if max { 0u8 } else { 255u8 };
            for dy in -m..=m {
                for dx in -m..=m {
                    let (yy, xx) = ((y + dy).clamp(0, h as i64 - 1), (x + dx).clamp(0, w as i64 - 1));
                    let p = px[(yy * w as i64 + xx) as usize];
                    v = if max { v.max(p) } else { v.min(p) };
                }
            }
            out[(y * w as i64 + x) as usize] = v;
        }
    }
    out
}

/// The node's process_mask on an L mask: inverted, blurred (GaussianBlur), grown or shrunk
fn process_mask(px: &mut Vec<u8>, w: usize, h: usize, o: &Opts) {
    if o.invert {
        px.iter_mut().for_each(|v| *v = 255 - *v);
    }
    if o.mask_blur > 0 {
        pil_gaussian_blur(px, w, h, o.mask_blur as f32);
    }
    if o.mask_offset != 0 {
        let size = o.mask_offset.unsigned_abs() as usize * 2 + 1;
        for _ in 0..o.mask_offset.unsigned_abs() {
            *px = pil_rank(px, w, h, size, o.mask_offset > 0);
        }
    }
}

/// SAM3Segment on one picture (3, H, W) f32: its IMAGE (the picture with the mask as alpha), MASK
/// and MASK_IMAGE, each a batch of the masks it returns (none found gives one empty mask)
pub fn segment(model: &Sam3, img: &Tensor, o: &Opts) -> Result<(Tensor, Tensor, Tensor)> {
    anyhow::ensure!(o.alpha, "SAM3Segment: background \"Color\" is not ported yet (\"Alpha\" is)");
    let (_, h, w) = img.dims3()?;
    let dev = img.device().clone();
    let probs = model.ground(img, &o.prompt, o.confidence, &mut None)?;
    // _run_single_per_instance: the masks (probability above 0.5) ordered by mean probability
    let n = probs.dim(0)?;
    let mut order: Vec<usize> = (0..n).collect();
    let means: Vec<f64> = (0..n)
        .map(|i| -> Result<f64> { Ok(probs.get(i)?.to_dtype(DType::F64)?.mean_all()?.to_scalar::<f64>()?) })
        .collect::<Result<_>>()?;
    if o.max_segments > 0 && n > o.max_segments {
        order.sort_by(|&a, &b| means[b].total_cmp(&means[a]));
        order.truncate(o.max_segments);
    }
    order.sort_by(|&a, &b| means[b].total_cmp(&means[a]));
    if o.segment_pick > 0 {
        order = order.get(o.segment_pick - 1).map(|&i| vec![i]).unwrap_or_default();
    }
    let mut masks: Vec<Vec<u8>> = vec![];
    for &i in &order {
        let m = probs.get(i)?.to_device(&candle_core::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let mut px: Vec<u8> = m.iter().map(|&v| if v > 0.5 { 255 } else { 0 }).collect();
        process_mask(&mut px, w, h, o);
        masks.push(px);
    }
    if o.merged && !masks.is_empty() {
        // _run_single_merged: the processed masks' max as float, back to uint8 (a float32 product,
        // truncated) and processed again
        let merged: Vec<u8> = (0..w * h)
            .map(|j| {
                let v = masks.iter().map(|m| m[j] as f32 / 255.0).fold(0f32, f32::max);
                (v.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect();
        let mut px = merged;
        process_mask(&mut px, w, h, o);
        masks = vec![px];
    }
    if masks.is_empty() {
        masks.push(vec![0u8; w * h]);
    }
    // the picture as tensor2pil leaves it, with each mask as its alpha
    let px = img.to_dtype(DType::F32)?.to_device(&candle_core::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let rgb: Vec<f32> = px.iter().map(|&v| ((255.0f32 * v).clamp(0.0, 255.0) as u8) as f32 / 255.0).collect();
    let rgb = Tensor::from_vec(rgb, (3, h, w), &dev)?;
    let mut images = vec![];
    let mut mask_t = vec![];
    for m in &masks {
        let mt = Tensor::from_vec(m.iter().map(|&v| v as f32 / 255.0).collect::<Vec<_>>(), (1, h, w), &dev)?;
        images.push(Tensor::cat(&[&rgb, &mt], 0)?);
        mask_t.push(mt.squeeze(0)?);
    }
    let mask = Tensor::stack(&mask_t, 0)?;
    let mask_image = mask.unsqueeze(1)?.broadcast_as((masks.len(), 3, h, w))?.contiguous()?;
    Ok((Tensor::stack(&images, 0)?, mask, mask_image))
}

/// The node's picture (3, H, W) f32 in [0, 1] as Sam3Processor.set_image transforms it:
/// tensor2pil's uint8 (np.clip(255 * x, 0, 255), truncated), v2.Resize((1008, 1008)) on the card
/// (to float, torch's antialiased bilinear, rounded half to even, back to uint8),
/// ToDtype(float32, scale=True) (times float(1 / 255)) and Normalize(0.5, 0.5):
/// (1, 3, 1008, 1008) f32, and the uint8 planes
pub fn net_input(img: &Tensor, dev: &Device) -> Result<(Tensor, Tensor)> {
    let (c, h, w) = img.dims3()?;
    let px = img.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let u8s: Vec<u8> = px.iter().map(|&v| (255.0f32 * v).clamp(0.0, 255.0) as u8).collect();
    let planes = Tensor::from_vec(u8s.iter().map(|&v| v as f32).collect::<Vec<_>>(), (c, h, w), dev)?;
    let r = kernels::torch_upsample_aa_bilinear_f32(&planes, RES, RES)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let scale = (1.0f64 / 255.0) as f32;
    let v: Vec<f32> = r.iter().map(|&x| ((x.round_ties_even() as u8) as f32 * scale - 0.5) / 0.5).collect();
    Ok((Tensor::from_vec(v, (1, c, RES, RES), dev)?, Tensor::from_vec(u8s, (c, h, w), &Device::Cpu)?))
}

/// Where the node keeps sam3.pt: ComfyUI's models/sam3
pub fn checkpoint(models: &Path) -> Result<std::path::PathBuf> {
    let p = models.join("sam3").join("sam3.pt");
    anyhow::ensure!(p.is_file(), "SAM3Segment: no {}", p.display());
    Ok(p)
}

/// debug-sam3 PICTURE PROMPT REF [MODE CONFIDENCE MAX_SEGMENTS PICK BLUR] (refs/sam3_ref.py
/// with the same options; Separate 0.35 8 1 4 by default): every stage against the node's
pub fn debug_sam3(pic: &Path, prompt: &str, refs: &Path, models: &Path, dev: &Device, opts: &[String]) -> Result<()> {
    let t0 = std::time::Instant::now();
    let model = Sam3::load(&checkpoint(models)?, dev)?;
    println!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let (px, w, h) = crate::ltx_video::load_image(pic)?;
    let img = Tensor::from_vec(px, (h, w, 3), &Device::Cpu)?.permute((2, 0, 1))?.contiguous()?;
    let mut d = HashMap::new();
    let t1 = std::time::Instant::now();
    let opt = |i: usize, default: &str| -> String { opts.get(i).cloned().unwrap_or_else(|| default.to_string()) };
    let o = Opts {
        prompt: prompt.to_string(),
        merged: opt(0, "Separate") == "Merged",
        confidence: opt(1, "0.35").parse()?,
        max_segments: opt(2, "8").parse()?,
        segment_pick: opt(3, "1").parse()?,
        mask_blur: opt(4, "4").parse()?,
        mask_offset: 0,
        invert: false,
        alpha: true,
    };
    let probs = model.ground(&img.to_device(dev)?, prompt, o.confidence, &mut Some(&mut d))?;
    probs.device().synchronize()?;
    println!("grounded in {:.2} s: {} masks kept", t1.elapsed().as_secs_f64(), probs.dim(0)?);

    let (image, mask, mask_image) = segment(&model, &img.to_device(dev)?, &o)?;
    d.insert("node_image".to_string(), image.permute((0, 2, 3, 1))?);
    d.insert("node_mask".to_string(), mask);
    d.insert("node_mask_image".to_string(), mask_image.permute((0, 2, 3, 1))?);
    if let Some(path) = std::env::var_os("RUSTYUI_SAM3_DUMP") {
        let cpu: HashMap<String, Tensor> = d
            .iter()
            .map(|(k, v)| Ok((k.clone(), v.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?)))
            .collect::<Result<_>>()?;
        candle_core::safetensors::save(&cpu, path)?;
    }
    let mut keys: Vec<String> = d.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let ours = d[&k].to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        let Ok(theirs) = r.load(&k, &Device::Cpu) else {
            println!("{k:16} not in the reference");
            continue;
        };
        let theirs = theirs.to_dtype(DType::F32)?;
        if ours.elem_count() != theirs.elem_count() {
            println!("{k:16} sizes {:?} and {:?}", ours.dims(), theirs.dims());
            continue;
        }
        let ours = ours.contiguous()?.reshape(theirs.dims())?;
        let diff = (&ours - &theirs)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
        let n = diff.iter().filter(|&&v| v != 0.0).count();
        let max = diff.iter().cloned().fold(0f32, f32::max);
        println!("{k:16} {n:>10} of {:>10} differ, max {max:.3e}{}", diff.len(), if n == 0 { "  exact" } else { "" });
    }
    Ok(())
}
