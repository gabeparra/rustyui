//! SDXL's ControlNet as ComfyUI's comfy/cldm/cldm.py runs it, with the extras of a union model
//! (xinsir's ControlNet++ Union ProMax): the encoder half of the UNet on rustyui's own blocks, the
//! hint encoder, the control-type embedding and the merge of the hint by task. f16 throughout, as
//! the file is. The hint's convolutions run on the memory layout ComfyUI holds the picture in
//! (channels-last for a LoadImage picture), since torch's cuDNN engines and its reductions follow it.

use crate::model::{File, Source};
use crate::nn::{Conv2d, LayerNorm, Linear};
use crate::unet::{self, Layer, Unet};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use std::rc::Rc;

/// ComfyUI's UNION_CONTROLNET_TYPES (comfy/cldm/control_types.py)
pub const UNION_TYPES: [&str; 8] =
    ["openpose", "depth", "hed/pidi/scribble/ted", "canny/lineart/anime_lineart/mlsd", "normal", "segment", "tile", "repaint"];

/// One ControlNetApplyAdvanced on a prompt: which model (with SetUnionControlNetType's types),
/// the picture, its strength and the part of the schedule it acts in, and the control it was
/// applied over (ComfyUI's previous_controlnet), whose residuals are added to its own
pub struct ControlApply {
    pub net: PathBuf,
    pub types: Vec<usize>,
    /// the IMAGE (b, 3, h, w) f32
    pub image: Tensor,
    /// whether ComfyUI holds that picture as channels-last memory (Engine::image_nhwc)
    pub nhwc: bool,
    pub strength: f64,
    /// start_percent, end_percent
    pub range: (f64, f64),
    pub prev: Option<Rc<ControlApply>>,
}

struct Union {
    task: Tensor,
    ln1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    ln2: LayerNorm,
    fc: Linear,
    proj: Linear,
    spatial: Linear,
    add1: Linear,
    add2: Linear,
    types: usize,
}

pub struct ControlNet {
    time1: Linear,
    time2: Linear,
    label1: Linear,
    label2: Linear,
    input: Vec<Vec<Layer>>,
    zero: Vec<Conv2d>,
    hint: Vec<Conv2d>,
    middle: Vec<Layer>,
    middle_out: Conv2d,
    union: Option<Union>,
    model_channels: usize,
}

impl ControlNet {
    pub fn load(file: &File, dev: &Device) -> Result<Self> {
        let s = Source {
            file,
            map: crate::model::controlnet_map(),
            loras: vec![],
            lora_prefix: "",
            compute: DType::F16,
            dtype: DType::F16,
            dev,
            patching: Default::default(),
        };
        anyhow::ensure!(
            file.has("controlnet_cond_embedding.conv_in.weight"),
            "{}: not an SDXL ControlNet in diffusers' layout (the one kind rustyui reads yet)",
            file.path.display()
        );
        let mut input = vec![];
        loop {
            let l = unet::layers_at(&s, &format!("input_blocks.{}", input.len()))?;
            if l.is_empty() {
                break;
            }
            input.push(l);
        }
        let zero = (0..input.len()).map(|i| unet::conv(&s, &format!("zero_convs.{i}.0"), 1)).collect::<Result<Vec<_>>>()?;
        // input_hint_block: eight 3x3 convolutions, the third, fifth and seventh strided by 2
        let hint = [(0, 1), (2, 1), (4, 2), (6, 1), (8, 2), (10, 1), (12, 2), (14, 1)]
            .iter()
            .map(|&(k, stride)| unet::conv(&s, &format!("input_hint_block.{k}"), stride))
            .collect::<Result<Vec<_>>>()?;
        let union = if file.has("task_embedding") {
            let t = "transformer_layes.0";
            let task = s.weight("task_embedding")?;
            Some(Union {
                types: task.dim(0)?,
                task,
                ln1: unet::ln(&s, &format!("{t}.ln_1"))?,
                in_proj: Linear {
                    w: s.weight(&format!("{t}.attn.in_proj_weight"))?,
                    b: Some(s.weight(&format!("{t}.attn.in_proj_bias"))?),
                },
                out_proj: unet::lin(&s, &format!("{t}.attn.out_proj"))?,
                ln2: unet::ln(&s, &format!("{t}.ln_2"))?,
                fc: unet::lin(&s, &format!("{t}.mlp.c_fc"))?,
                proj: unet::lin(&s, &format!("{t}.mlp.c_proj"))?,
                spatial: unet::lin(&s, "spatial_ch_projs")?,
                add1: unet::lin(&s, "control_add_embedding.linear_1")?,
                add2: unet::lin(&s, "control_add_embedding.linear_2")?,
            })
        } else {
            None
        };
        let time1 = unet::lin(&s, "time_embed.0")?;
        Ok(ControlNet {
            model_channels: time1.w.dim(1)?,
            time1,
            time2: unet::lin(&s, "time_embed.2")?,
            label1: unet::lin(&s, "label_emb.0.0")?,
            label2: unet::lin(&s, "label_emb.0.2")?,
            input,
            zero,
            hint,
            middle: unet::layers_at(&s, "middle_block")?,
            middle_out: unet::conv(&s, "middle_block_out.0", 1)?,
            union,
        })
    }

    /// input_hint_block: the convolutions with SiLU between, on the picture's memory layout
    fn hint_block(&self, hint: &Tensor, nhwc: bool) -> Result<Tensor> {
        let mut h = hint.clone();
        for (i, c) in self.hint.iter().enumerate() {
            if i > 0 {
                h = crate::kernels::torch_silu(&h)?;
            }
            h = if nhwc { c.forward_nhwc(&h)? } else { c.forward(&h)? };
        }
        Ok(h)
    }

    /// union_controlnet_merge for the one type ComfyUI ever sets: the hint encoded, its spatial
    /// mean plus the type's task embedding through the one-token transformer layer and the
    /// projection, added back onto the encoded hint
    fn union_merge(&self, u: &Union, hint: &Tensor, nhwc: bool, ty: usize) -> Result<Tensor> {
        let cond = self.hint_block(hint, nhwc)?;
        let (b, c, hh, ww) = cond.dims4()?;
        // torch.mean over (2, 3) sums the memory as it lies: channels-last, the positions are the
        // reduced dimension and the channels the outputs (kernels::torch_mean's layout); NCHW (a
        // Canny picture), the positions are the fastest dimension (torchcpu::mean_fastest_f16)
        let feat = if nhwc {
            let flat = cond.permute((0, 2, 3, 1))?.contiguous()?.reshape((b, hh * ww, c))?;
            crate::kernels::torch_mean(&flat, 1)?.reshape((b, c))?
        } else {
            let v = cond.contiguous()?.flatten_all()?.to_vec1::<half::f16>()?;
            let (sms, per_sm) = crate::kernels::sm_info()?;
            let m = crate::torchcpu::mean_fastest_f16(&v, b * c, hh * ww, sms, per_sm)?;
            Tensor::from_vec(m, (b, c), cond.device())?
        };

        let x = feat.broadcast_add(&u.task.narrow(0, ty, 1)?)?.unsqueeze(1)?;
        // ResBlockUnionControlnet; attention over a single token returns its value exactly
        let qkv = u.in_proj.forward(&u.ln1.forward(&x)?)?;
        let v = qkv.narrow(2, 2 * c, c)?.contiguous()?;
        let x = (&x + u.out_proj.forward(&v)?)?;
        let m = u.proj.forward(&crate::kernels::torch_quick_gelu_f16(&u.fc.forward(&u.ln2.forward(&x)?)?)?)?;
        let x = (&x + m)?;
        let alpha = u.spatial.forward(&x.squeeze(1)?)?;
        Ok(cond.broadcast_add(&alpha.reshape((b, c, 1, 1))?)?)
    }

    /// One call's residuals, ComfyUI's control dict before control_merge: x the scaled noisy
    /// latent (b, 4, h, w) f16, t the discrete timesteps, ctx and y as the UNet gets them, hint
    /// (1, 3, 8h, 8w) f16, `types` SetUnionControlNetType's
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        t: &[f32],
        ctx: &Tensor,
        y: &Tensor,
        hint: &Tensor,
        nhwc: bool,
        types: &[usize],
        dev: &Device,
    ) -> Result<unet::Control> {
        let temb = crate::kernels::torch_timestep_embedding(t, self.model_channels, dev)?.to_dtype(DType::F16)?;
        let mut emb = self.time2.forward(&crate::kernels::torch_silu(&self.time1.forward(&temb)?)?)?;
        let mut guided = None;
        if let Some(u) = &self.union {
            if types.len() > 1 || types.iter().any(|&t| t >= u.types) {
                bail!("control types {types:?}: a union ControlNet takes one of its {}", u.types);
            }
            // control_add_embedding: a one-hot over the types, each value's 256-wide sinusoid on
            // the card, linear, SiLU, linear, added to the time embedding in place
            let onehot: Vec<f32> = (0..u.types).map(|i| if types.contains(&i) { 1.0 } else { 0.0 }).collect();
            let c = crate::kernels::torch_timestep_embedding(&onehot, 256, dev)?.to_dtype(DType::F16)?.reshape((1, u.types * 256))?;
            emb = emb.broadcast_add(&u.add2.forward(&crate::kernels::torch_silu(&u.add1.forward(&c)?)?)?)?;
            if let Some(&ty) = types.first() {
                guided = Some(self.union_merge(u, hint, nhwc, ty)?);
            }
        }
        let guided = match guided {
            Some(g) => g,
            None => self.hint_block(hint, nhwc)?,
        };
        let emb = (emb + self.label2.forward(&crate::kernels::torch_silu(&self.label1.forward(y)?)?)?)?;
        let mut output = vec![];
        let mut h = x.clone();
        for (i, (blk, zero)) in self.input.iter().zip(&self.zero).enumerate() {
            h = Unet::run(blk, h, &emb, ctx, None)?;
            if i == 0 {
                h = h.broadcast_add(&guided)?;
            }
            output.push(zero.forward(&h)?);
        }
        let h = Unet::run(&self.middle, h, &emb, ctx, None)?;
        Ok(unet::Control { output, middle: self.middle_out.forward(&h)? })
    }
}

/// `x *= strength` on an f16 tensor as torch does it: in float, rounded once
pub fn scale_f16(x: &Tensor, s: f64) -> Result<Tensor> {
    Ok((x.to_dtype(DType::F32)? * s)?.to_dtype(DType::F16)?)
}

/// The control picture as get_control prepares it for a latent of (h, w): common_upscale to
/// 8x the latent with nearest-exact and a centre crop (on the CPU, as ComfyUI's IMAGE is), then f16
/// on the card. nearest-exact takes source floor((i + 0.5) * scale) with the product rounded to
/// float, as torch's CPU kernel does (scale = in / out in float).
pub fn prepare_hint(image: &Tensor, h: usize, w: usize, dev: &Device) -> Result<Tensor> {
    let (b, c, ih, iw) = image.dims4()?;
    if b != 1 {
        bail!("a control picture batch of {b}: one picture is ported");
    }
    let (oh, ow) = (h * 8, w * 8);
    // the centre crop, Python's round (half to even) on doubles
    let round = |v: f64| -> usize { v.round_ties_even() as usize };
    let (old_aspect, new_aspect) = (iw as f64 / ih as f64, ow as f64 / oh as f64);
    let (mut x0, mut y0) = (0usize, 0usize);
    if old_aspect > new_aspect {
        x0 = round((iw as f64 - iw as f64 * (new_aspect / old_aspect)) / 2.0);
    } else if old_aspect < new_aspect {
        y0 = round((ih as f64 - ih as f64 * (old_aspect / new_aspect)) / 2.0);
    }
    let (ch, cw) = (ih - 2 * y0, iw - 2 * x0);
    let src = image.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let idx = |dst: usize, n_in: usize, n_out: usize| -> usize {
        let scale = n_in as f32 / n_out as f32;
        (((dst as f64 + 0.5) * scale as f64) as f32).floor().min((n_in - 1) as f32) as usize
    };
    let rows: Vec<usize> = (0..oh).map(|i| idx(i, ch, oh) + y0).collect();
    let cols: Vec<usize> = (0..ow).map(|j| idx(j, cw, ow) + x0).collect();
    let mut out = Vec::with_capacity(c * oh * ow);
    for k in 0..c {
        for &r in &rows {
            for &q in &cols {
                out.push(src[(k * ih + r) * iw + q]);
            }
        }
    }
    Ok(Tensor::from_vec(out, (1, c, oh, ow), dev)?.to_dtype(DType::F16)?)
}

/// The sigma ComfyUI's ControlNet stops (or starts) at for a percent of the schedule
/// (ModelSamplingDiscrete.percent_to_sigma), with the sigma table on the card as it is while
/// sampling: compared as f32
pub fn percent_to_sigma(disc: &crate::sampling::Discrete, percent: f64, dev: &Device) -> Result<f32> {
    if percent <= 0.0 {
        return Ok(999_999_999.9f64 as f32);
    }
    if percent >= 1.0 {
        return Ok(0.0);
    }
    let t = ((1.0 - percent) * 999.0) as f32;
    disc.sigma_on_card(t, dev)
}
