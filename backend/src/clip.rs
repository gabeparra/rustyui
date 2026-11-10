//! SDXL's two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG), as ComfyUI runs them:
//! weights stored f16, every operation in f32, the penultimate layer's output (no final
//! layer norm) as the conditioning, and G's projected EOS state as the pooled vector.

use crate::kernels;
use crate::model::Source;
use crate::nn::{LayerNorm, Linear};
use crate::prompt::{Chunk, Embeddings, Encoder, Tok};
use crate::tokenizer::{ClipTokenizer, BOS, EOS};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

#[derive(Clone, Copy)]
pub enum Act {
    QuickGelu,
    Gelu,
}

struct Layer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

pub struct ClipText {
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    /// OpenCLIP's text_projection as ComfyUI holds it: a Linear weight, its transpose, contiguous
    projection: Option<Tensor>,
    heads: usize,
    act: Act,
    pub width: usize,
}

/// Loads one encoder. `hf` names the weights in transformers' layout under that prefix
/// (`text_model.encoder.layers.N.self_attn.q_proj` ...); the source maps them to the file.
fn load(src: &Source, heads: usize, act: Act, layers: usize, projection: bool) -> Result<ClipText> {
    let lin = |name: &str| -> Result<Linear> {
        Ok(Linear { w: src.weight(&format!("{name}.weight"))?, b: src.weight_opt(&format!("{name}.bias"))? })
    };
    let ln = |name: &str| -> Result<LayerNorm> {
        Ok(LayerNorm { w: src.weight(&format!("{name}.weight"))?, b: src.weight(&format!("{name}.bias"))?, eps: 1e-5 })
    };
    let mut out = vec![];
    for i in 0..layers {
        let p = format!("text_model.encoder.layers.{i}");
        out.push(Layer {
            ln1: ln(&format!("{p}.layer_norm1"))?,
            q: lin(&format!("{p}.self_attn.q_proj"))?,
            k: lin(&format!("{p}.self_attn.k_proj"))?,
            v: lin(&format!("{p}.self_attn.v_proj"))?,
            out: lin(&format!("{p}.self_attn.out_proj"))?,
            ln2: ln(&format!("{p}.layer_norm2"))?,
            fc1: lin(&format!("{p}.mlp.fc1"))?,
            fc2: lin(&format!("{p}.mlp.fc2"))?,
        });
    }
    let token_embedding = src.weight("text_model.embeddings.token_embedding.weight")?;
    let width = token_embedding.dim(1)?;
    Ok(ClipText {
        token_embedding,
        position_embedding: src.weight("text_model.embeddings.position_embedding.weight")?,
        layers: out,
        final_ln: ln("text_model.final_layer_norm")?,
        projection: if projection {
            // ComfyUI transposes it into a Linear weight when it loads the checkpoint, and a LoRA's
            // lora_te2_text_projection patches that weight
            Some(src.patch_as(src.weight("text_projection")?.t()?.contiguous()?, "text_projection")?)
        } else {
            None
        },
        heads,
        act,
        width,
    })
}

impl ClipText {
    /// Hidden states of layer `layer_idx` (negative counts from the end) and, for each
    /// sequence, the final-normed state at its first EOS, projected when the encoder has one.
    pub fn forward(&self, chunks: &[Chunk], layer_idx: i64, dev: &Device) -> Result<(Tensor, Tensor)> {
        crate::profile::scope("clip");
        let n_seq = chunks.len();
        let len = chunks[0].len();
        // token embeddings, with textual-inversion vectors in their slots
        let mut rows = vec![];
        let mut eos = vec![];
        for chunk in chunks {
            let mut ids = vec![];
            let mut parts: Vec<Tensor> = vec![];
            let flush = |ids: &mut Vec<u32>, parts: &mut Vec<Tensor>| -> Result<()> {
                if !ids.is_empty() {
                    let idx = Tensor::new(ids.as_slice(), dev)?;
                    parts.push(self.token_embedding.index_select(&idx, 0)?.to_dtype(DType::F32)?);
                    ids.clear();
                }
                Ok(())
            };
            let mut first_eos = None;
            for (pos, (tok, _)) in chunk.iter().enumerate() {
                match tok {
                    Tok::Id(id) => {
                        if *id == EOS && first_eos.is_none() {
                            first_eos = Some(pos);
                        }
                        ids.push(*id);
                    }
                    Tok::Vec(v) => {
                        flush(&mut ids, &mut parts)?;
                        anyhow::ensure!(v.len() == self.width, "embedding width {} for a {}-wide encoder", v.len(), self.width);
                        parts.push(Tensor::from_slice(v, (1, self.width), dev)?);
                    }
                }
            }
            flush(&mut ids, &mut parts)?;
            rows.push(Tensor::cat(&parts, 0)?);
            eos.push(first_eos.unwrap_or(len - 1));
        }
        let x = Tensor::stack(&rows, 0)?; // (n, len, width) f32
        let pos = self.position_embedding.narrow(0, 0, len)?.to_dtype(DType::F32)?;
        let mut x = x.broadcast_add(&pos)?.reshape((n_seq * len, self.width))?;

        // the causal mask: -finfo(f32).max above the diagonal
        let mut mask = vec![0f32; len * len];
        for i in 0..len {
            for j in (i + 1)..len {
                mask[i * len + j] = -f32::MAX;
            }
        }
        let mask = Tensor::from_vec(mask, (len, len), dev)?;

        // every op as torch runs it in f32 (weights stored f16, upcast exactly): F.layer_norm,
        // F.linear with its bias (cuBLASLt's bias epilogue), SDPA on the memory-efficient kernel, the
        // activations
        let lin = |l: &Linear, x: &Tensor| -> Result<Tensor> {
            let w = l.w.to_dtype(DType::F32)?;
            Ok(match &l.b {
                Some(b) => kernels::torch_linear_bias_f32(x, &w, &b.to_dtype(DType::F32)?)?,
                None => kernels::torch_linear_f32(x, &w)?,
            })
        };
        let norm = |n: &LayerNorm, x: &Tensor| -> Result<Tensor> {
            Ok(kernels::torch_layer_norm_f32(x, &n.w.to_dtype(DType::F32)?, &n.b.to_dtype(DType::F32)?, n.eps as f32)?)
        };
        let seq = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((n_seq, len, self.width))?) };
        let target = if layer_idx < 0 { self.layers.len() as i64 + layer_idx } else { layer_idx } as usize;
        let mut hidden = None;
        for (i, l) in self.layers.iter().enumerate() {
            let h = norm(&l.ln1, &x)?;
            let (q, k, v) = (seq(lin(&l.q, &h)?)?, seq(lin(&l.k, &h)?)?, seq(lin(&l.v, &h)?)?);
            let a = kernels::torch_mem_eff_attention(&q, &k, &v, self.heads, kernels::HeadLayout::Btk, Some(&mask))?;
            x = (x + lin(&l.out, &a.reshape((n_seq * len, self.width))?)?)?;
            let h = lin(&l.fc1, &norm(&l.ln2, &x)?)?;
            let h = kernels::torch_clip_act_f32(&h, matches!(self.act, Act::QuickGelu))?;
            x = (x + lin(&l.fc2, &h)?)?;
            if i == target {
                hidden = Some(x.reshape((n_seq, len, self.width))?);
            }
        }
        let fin = norm(&self.final_ln, &x)?.reshape((n_seq, len, self.width))?;
        let mut pooled = vec![];
        for (s, &e) in eos.iter().enumerate().take(n_seq) {
            pooled.push(fin.narrow(0, s, 1)?.narrow(1, e, 1)?.squeeze(1)?);
        }
        let mut pooled = Tensor::cat(&pooled, 0)?;
        if let Some(p) = &self.projection {
            // OpenCLIP's x @ text_projection, as ComfyUI runs it: F.linear on the transposed weight
            pooled = kernels::torch_linear_f32(&pooled, &p.to_dtype(DType::F32)?)?;
        }
        Ok((hidden.context("clip layer index out of range")?, pooled))
    }
}

pub struct SdxlClip {
    pub l: ClipText,
    pub g: ClipText,
    tok: ClipTokenizer,
    /// whether a prompt has been encoded since the encoders were loaded (ClipText::settle)
    settled: bool,
}

impl ClipText {
    /// What ComfyUI's DynamicVRAM loader leaves after an encoder's first use: it patches a LoRA'd
    /// layer in f32 as it first casts it, computes with that, and writes the result back into the
    /// resident f16 weight (resolve_cast_module_with_vbar's update_weight), so every later encode
    /// casts the rounded weight up again. Rounds every f32 weight (only patched ones are) to f16.
    fn settle(&mut self) -> Result<()> {
        let round = |w: &mut Tensor| -> Result<()> {
            if w.dtype() == DType::F32 {
                *w = w.to_dtype(DType::F16)?;
            }
            Ok(())
        };
        for l in &mut self.layers {
            for lin in [&mut l.q, &mut l.k, &mut l.v, &mut l.out, &mut l.fc1, &mut l.fc2] {
                round(&mut lin.w)?;
            }
        }
        if let Some(p) = &mut self.projection {
            round(p)?;
        }
        Ok(())
    }
}

/// One CLIPTextEncode result: the (1, 77*k, 2048) conditioning and the (1, 1280) pooled vector,
/// and what nodes after it set in its dict.
#[derive(Clone)]
pub struct Cond {
    pub cond: Tensor,
    pub pooled: Tensor,
    pub props: CondProps,
}

/// What ConditioningSetMask and the hook nodes set in a conditioning's dict
#[derive(Clone, Default)]
pub struct CondProps {
    /// 'mask' (b, h, w) f32 at the picture's size, and 'mask_strength'
    pub mask: Option<(Tensor, f64)>,
    /// 'set_area_to_bounds' ("mask bounds"), which the sampler does not run yet
    pub area_to_bounds: bool,
    /// 'hooks': the sampler patches the model with them for this conditioning, and keys its batches
    /// by the group itself (ComfyUI keys a dict by the object), so compare with Rc::ptr_eq
    pub hooks: Option<std::rc::Rc<crate::model::HookGroup>>,
    /// 'control': a ControlNetApplyAdvanced (and through it the ones applied before it)
    pub control: Option<std::rc::Rc<crate::controlnet::ControlApply>>,
}

impl CondProps {
    /// nothing set: the sampler's one-conditioning path runs it
    pub fn is_plain(&self) -> bool {
        self.mask.is_none() && self.hooks.is_none() && !self.area_to_bounds && self.control.is_none()
    }
}

impl SdxlClip {
    pub fn load(l: &Source, g: &Source) -> Result<Self> {
        Ok(SdxlClip {
            l: load(l, 12, Act::QuickGelu, 12, false)?,
            g: load(g, 20, Act::Gelu, 32, true)?,
            tok: ClipTokenizer::new()?,
            settled: false,
        })
    }

    pub fn tokenize(&self, text: &str, embs: &mut Embeddings) -> (Vec<Chunk>, Vec<Chunk>) {
        let l = Encoder { tok: &self.tok, pad: EOS, embedding_key: "clip_l", max_length: 77 };
        let g = Encoder { tok: &self.tok, pad: 0, embedding_key: "clip_g", max_length: 77 };
        (l.tokenize(text, embs), g.tokenize(text, embs))
    }

    /// ComfyUI's encode_token_weights for one encoder: all chunks in one batch, plus the
    /// empty prompt when any weight is not 1, whose states anchor the weighting.
    fn encode_one(&self, enc: &ClipText, chunks: &[Chunk], pad: u32, layer_idx: i64, dev: &Device) -> Result<(Tensor, Tensor)> {
        let weighted = chunks.iter().any(|c| c.iter().any(|(_, w)| *w != 1.0));
        let mut batch: Vec<Chunk> = chunks.to_vec();
        if weighted {
            let mut empty: Chunk = vec![(Tok::Id(BOS), 1.0), (Tok::Id(EOS), 1.0)];
            empty.extend(std::iter::repeat_n((Tok::Id(pad), 1.0), 77 - 2));
            batch.push(empty);
        }
        let (hidden, pooled) = enc.forward(&batch, layer_idx, dev)?;
        let first_pooled = pooled.narrow(0, 0, 1)?;
        let mut out = vec![];
        for (k, chunk) in chunks.iter().enumerate() {
            let mut z = hidden.narrow(0, k, 1)?;
            if weighted {
                let empty = hidden.narrow(0, chunks.len(), 1)?;
                // z[j] = (z[j] - empty[j]) * w + empty[j], in f32, only where w != 1
                let w: Vec<f32> = chunk.iter().map(|(_, w)| *w).collect();
                let keep: Vec<u8> = w.iter().map(|w| (*w != 1.0) as u8).collect();
                let wt = Tensor::from_vec(w, (1, chunk.len(), 1), dev)?;
                let zw = z.broadcast_sub(&empty)?.broadcast_mul(&wt)?.broadcast_add(&empty)?;
                let sel = Tensor::from_vec(keep, (1, chunk.len(), 1), dev)?.broadcast_as(z.shape())?;
                z = sel.where_cond(&zw, &z)?;
            }
            out.push(z);
        }
        Ok((Tensor::cat(&out, 1)?, first_pooled))
    }

    /// Settled as from the start: the state ComfyUI's encoders keep once they have encoded (they
    /// stay loaded), for an engine that let them go and loads them again
    pub fn settle_now(&mut self) -> Result<()> {
        if !self.settled {
            self.l.settle()?;
            self.g.settle()?;
            self.settled = true;
        }
        Ok(())
    }

    /// One CLIPTextEncode. The first after loading computes with the LoRAs' f32 patches, and the
    /// encoders then settle to the f16-rounded weights every later one uses, as in ComfyUI.
    pub fn encode(&mut self, text: &str, layer_idx: i64, embs: &mut Embeddings, dev: &Device) -> Result<Cond> {
        let (lc, gc) = self.tokenize(text, embs);
        let (g_out, g_pooled) = self.encode_one(&self.g, &gc, 0, layer_idx, dev)?;
        let (l_out, _) = self.encode_one(&self.l, &lc, EOS, layer_idx, dev)?;
        let cut = l_out.dim(1)?.min(g_out.dim(1)?);
        let cond = Tensor::cat(&[l_out.narrow(1, 0, cut)?, g_out.narrow(1, 0, cut)?], 2)?;
        if !self.settled {
            self.l.settle()?;
            self.g.settle()?;
            self.settled = true;
        }
        Ok(Cond { cond, pooled: g_pooled, props: CondProps::default() })
    }
}

/// debug-clip CHECKPOINT REF (refs/clip_ref.py): CLIP-G's and CLIP-L's layer 0 op by op, each op
/// on ComfyUI's input for it, then the final norm, the projection and CLIPTextEncode's outputs
pub fn debug_clip(ckpt: &std::path::Path, refs: &std::path::Path, prompt: &str, dev: &Device) -> Result<()> {
    let file = crate::model::File::open(ckpt)?;
    let src_l = Source {
        file: &file,
        map: crate::model::clip_l_map(),
        loras: vec![],
        lora_prefix: "lora_te1_",
        compute: DType::F32,
        dtype: DType::F16,
        dev,
        patching: Default::default(),
    };
    let src_g = Source {
        file: &file,
        map: crate::model::clip_g_map(),
        loras: vec![],
        lora_prefix: "lora_te2_",
        compute: DType::F32,
        dtype: DType::F16,
        dev,
        patching: Default::default(),
    };
    let mut clip = SdxlClip::load(&src_l, &src_g)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let get = |n: &str| -> Result<Tensor> { Ok(r.load(n, dev)?.to_dtype(DType::F32)?) };
    let report = |name: &str, ours: &Tensor, theirs: &Tensor| -> Result<()> {
        let ours = ours.reshape(theirs.dims())?;
        let d = (&ours - theirs)?.abs()?;
        let differ = d.ne(0f32)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        let max = d.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!(
            "{name:34} {differ:>8} of {:>8} differ, max |d| {max:.3e}{}",
            theirs.elem_count(),
            if differ == 0.0 { " exact" } else { "" }
        );
        Ok(())
    };
    for (name, enc) in [("clip_g", &clip.g), ("clip_l", &clip.l)] {
        println!("── {name}");
        let c = |n: &str| -> Result<Tensor> {
            let t = get(&format!("{name}.{n}"))?;
            let w = t.dim(candle_core::D::Minus1)?;
            Ok(t.reshape((t.elem_count() / w, w))?)
        };
        let l = &enc.layers[0];
        let lin = |l: &Linear, x: &Tensor| -> Result<Tensor> {
            let w = l.w.to_dtype(DType::F32)?;
            Ok(match &l.b {
                Some(b) => kernels::torch_linear_bias_f32(x, &w, &b.to_dtype(DType::F32)?)?,
                None => kernels::torch_linear_f32(x, &w)?,
            })
        };
        let norm = |n: &LayerNorm, x: &Tensor| -> Result<Tensor> {
            Ok(kernels::torch_layer_norm_f32(x, &n.w.to_dtype(DType::F32)?, &n.b.to_dtype(DType::F32)?, n.eps as f32)?)
        };
        report("layer_norm1", &norm(&l.ln1, &c("l0.ln1.in")?)?, &c("l0.ln1")?)?;
        report("q_proj", &lin(&l.q, &c("l0.ln1")?)?, &c("l0.q_proj")?)?;
        report("k_proj", &lin(&l.k, &c("l0.ln1")?)?, &c("l0.k_proj")?)?;
        report("v_proj", &lin(&l.v, &c("l0.ln1")?)?, &c("l0.v_proj")?)?;
        let seq = |t: Tensor| -> Result<Tensor> { Ok(t.unsqueeze(0)?) };
        let len = c("l0.q_proj")?.dim(0)?;
        let mut mask = vec![0f32; len * len];
        for i in 0..len {
            for j in (i + 1)..len {
                mask[i * len + j] = -f32::MAX;
            }
        }
        let mask = Tensor::from_vec(mask, (len, len), dev)?;
        report("mask", &mask, &get(&format!("{name}.sdpa.mask"))?.squeeze(0)?.squeeze(0)?)?;
        let a = kernels::torch_mem_eff_attention(
            &seq(c("l0.q_proj")?)?,
            &seq(c("l0.k_proj")?)?,
            &seq(c("l0.v_proj")?)?,
            enc.heads,
            kernels::HeadLayout::Btk,
            Some(&mask),
        )?;
        report("attention (mem-efficient)", &a, &c("l0.out_proj.in")?)?;
        report("out_proj", &lin(&l.out, &c("l0.out_proj.in")?)?, &c("l0.out_proj")?)?;
        report("layer_norm2", &norm(&l.ln2, &c("l0.ln2.in")?)?, &c("l0.ln2")?)?;
        report("fc1", &lin(&l.fc1, &c("l0.ln2")?)?, &c("l0.fc1")?)?;
        report("activation", &kernels::torch_clip_act_f32(&c("l0.fc1")?, matches!(enc.act, Act::QuickGelu))?, &c("l0.fc2.in")?)?;
        report("fc2", &lin(&l.fc2, &c("l0.fc2.in")?)?, &c("l0.fc2")?)?;
        report("final_layer_norm", &norm(&enc.final_ln, &c("final_ln.in")?)?, &c("final_ln")?)?;
        if let Some(p) = &enc.projection {
            report(
                "text_projection",
                &kernels::torch_linear_f32(&c("text_projection.in")?, &p.to_dtype(DType::F32)?)?,
                &c("text_projection")?,
            )?;
        }
    }
    let mut embs = crate::prompt::Embeddings::new(vec![]);
    let cond = clip.encode(prompt, -2, &mut embs, dev)?;
    report("CLIPTextEncode cond", &cond.cond, &get("cond")?)?;
    report("CLIPTextEncode pooled", &cond.pooled, &get("pooled")?)?;
    Ok(())
}
