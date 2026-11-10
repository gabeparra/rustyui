//! The SDXL VAE decoder (ldm layout), in bf16 like ComfyUI on an Ada card.

use crate::model::{Mapped, Source};
use crate::nn::{attention, Conv2d, GroupNorm};
use anyhow::Result;
use candle_core::{DType, Tensor};

struct Res {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Res {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_cl(x, false)
    }

    /// `cl` when x is channels-last memory in torch: GroupNorm hands back contiguous NCHW, so
    /// conv1 and conv2 run plain, but the shortcut conv takes x itself and runs channels-last, and
    /// x + h keeps x's layout (the first operand orders TensorIterator's output)
    fn forward_cl(&self, x: &Tensor, cl: bool) -> Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward_silu(x)?)?;
        let h = self.conv2.forward(&self.norm2.forward_silu(&h)?)?;
        let x = match &self.shortcut {
            Some(c) if cl => c.forward_nhwc(x)?,
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok((x + h)?)
    }
}

struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj: Conv2d,
}

impl AttnBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.norm.forward(x)?;
        let a = Self::attend(&self.q.forward(&n)?, &self.k.forward(&n)?, &self.v.forward(&n)?)?;
        Ok((x + self.proj.forward(&a)?)?)
    }

    fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        pytorch_attention(q, k, v)
    }
}

/// ComfyUI's pytorch_attention (vae_attention on this box): q, k, v (b, c, h, w) as one head over
/// every pixel, (b, hw, c), through SDPA, back to (b, c, h, w). For bf16 that is torch's
/// mem-efficient kernel (flash attention stops at 256-wide heads, and so does cuDNN's on this card).
pub fn pytorch_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = q.dims4()?;
    let seq = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?) };
    let (q, k, v) = (seq(q)?, seq(k)?, seq(v)?);
    let a = if q.dtype() == DType::BF16 && q.device().is_cuda() {
        crate::kernels::torch_mem_eff_attention(&q, &k, &v, 1, crate::kernels::HeadLayout::Btk, None)?
    } else {
        attention(&q, &k, &v, 1, None)?
    };
    Ok(a.transpose(1, 2)?.reshape((b, c, h, w))?)
}

pub struct Decoder {
    post_quant: Conv2d,
    conv_in: Conv2d,
    mid1: Res,
    mid_attn: AttnBlock,
    mid2: Res,
    /// levels from the deepest up, each: its res blocks and an optional upsample conv
    up: Vec<(Vec<Res>, Option<Conv2d>)>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

fn has(s: &Source, name: &str) -> bool {
    match (s.map)(name) {
        Mapped::Key(k) | Mapped::Rows(k, _, _) => s.file.has(&k),
    }
}

fn conv(s: &Source, p: &str) -> Result<Conv2d> {
    let w = s.weight(&format!("{p}.weight"))?;
    let k = w.dim(2)?;
    Ok(Conv2d { w, b: s.weight_opt(&format!("{p}.bias"))?, stride: 1, padding: k / 2 })
}

fn gn(s: &Source, p: &str) -> Result<GroupNorm> {
    Ok(GroupNorm { w: s.weight(&format!("{p}.weight"))?, b: s.weight(&format!("{p}.bias"))?, groups: 32, eps: 1e-6 })
}

fn res(s: &Source, p: &str) -> Result<Res> {
    Ok(Res {
        norm1: gn(s, &format!("{p}.norm1"))?,
        conv1: conv(s, &format!("{p}.conv1"))?,
        norm2: gn(s, &format!("{p}.norm2"))?,
        conv2: conv(s, &format!("{p}.conv2"))?,
        shortcut: if has(s, &format!("{p}.nin_shortcut.weight")) { Some(conv(s, &format!("{p}.nin_shortcut"))?) } else { None },
    })
}

impl Decoder {
    pub fn load(s: &Source) -> Result<Self> {
        let mut levels = 0;
        while has(s, &format!("decoder.up.{levels}.block.0.conv1.weight")) {
            levels += 1;
        }
        let mut up = vec![];
        for i in (0..levels).rev() {
            let mut blocks = vec![];
            let mut j = 0;
            while has(s, &format!("decoder.up.{i}.block.{j}.conv1.weight")) {
                blocks.push(res(s, &format!("decoder.up.{i}.block.{j}"))?);
                j += 1;
            }
            let upsample = if has(s, &format!("decoder.up.{i}.upsample.conv.weight")) {
                Some(conv(s, &format!("decoder.up.{i}.upsample.conv"))?)
            } else {
                None
            };
            up.push((blocks, upsample));
        }
        Ok(Decoder {
            post_quant: conv(s, "post_quant_conv")?,
            conv_in: conv(s, "decoder.conv_in")?,
            mid1: res(s, "decoder.mid.block_1")?,
            mid_attn: AttnBlock {
                norm: gn(s, "decoder.mid.attn_1.norm")?,
                q: conv(s, "decoder.mid.attn_1.q")?,
                k: conv(s, "decoder.mid.attn_1.k")?,
                v: conv(s, "decoder.mid.attn_1.v")?,
                proj: conv(s, "decoder.mid.attn_1.proj_out")?,
            },
            mid2: res(s, "decoder.mid.block_2")?,
            up,
            norm_out: gn(s, "decoder.norm_out")?,
            conv_out: conv(s, "decoder.conv_out")?,
        })
    }

    /// latents (already divided by the scale factor) -> pixels in [0, 1], f32
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        crate::profile::scope("vae");
        let z = z.to_dtype(DType::BF16)?;
        let mut h = self.conv_in.forward(&self.post_quant.forward(&z)?)?;
        h = self.mid2.forward(&self.mid_attn.forward(&self.mid1.forward(&h)?)?)?;
        for (blocks, upsample) in &self.up {
            for b in blocks {
                h = b.forward(&h)?;
            }
            if let Some(c) = upsample {
                let (_, _, hh, ww) = h.dims4()?;
                h = c.forward(&h.upsample_nearest2d(hh * 2, ww * 2)?)?;
            }
        }
        let h = self.conv_out.forward(&self.norm_out.forward_silu(&h)?)?;
        // process_output: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0), in f32
        let x = h.to_dtype(DType::F32)?;
        let x = x.affine(1.0, 1.0)?.broadcast_div(&Tensor::new(2f32, x.device())?)?;
        Ok(x.clamp(0f32, 1f32)?)
    }
}

/// The SDXL VAE encoder (ldm layout) and quant_conv; returns the posterior's mean, as
/// ComfyUI's DiagonalGaussianRegularizer(sample=False) does.
pub struct Encoder {
    conv_in: Conv2d,
    /// per level: res blocks, then a stride-2 downsample conv (padded 0,1,0,1) except the last
    down: Vec<(Vec<Res>, Option<Conv2d>)>,
    mid1: Res,
    mid_attn: AttnBlock,
    mid2: Res,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    quant: Conv2d,
}

impl Encoder {
    pub fn load(s: &Source) -> Result<Self> {
        let mut down = vec![];
        let mut i = 0;
        while has(s, &format!("encoder.down.{i}.block.0.conv1.weight")) {
            let mut blocks = vec![];
            let mut j = 0;
            while has(s, &format!("encoder.down.{i}.block.{j}.conv1.weight")) {
                blocks.push(res(s, &format!("encoder.down.{i}.block.{j}"))?);
                j += 1;
            }
            let ds = if has(s, &format!("encoder.down.{i}.downsample.conv.weight")) {
                let mut c = conv(s, &format!("encoder.down.{i}.downsample.conv"))?;
                c.stride = 2;
                c.padding = 0;
                Some(c)
            } else {
                None
            };
            down.push((blocks, ds));
            i += 1;
        }
        Ok(Encoder {
            conv_in: conv(s, "encoder.conv_in")?,
            down,
            mid1: res(s, "encoder.mid.block_1")?,
            mid_attn: AttnBlock {
                norm: gn(s, "encoder.mid.attn_1.norm")?,
                q: conv(s, "encoder.mid.attn_1.q")?,
                k: conv(s, "encoder.mid.attn_1.k")?,
                v: conv(s, "encoder.mid.attn_1.v")?,
                proj: conv(s, "encoder.mid.attn_1.proj_out")?,
            },
            mid2: res(s, "encoder.mid.block_2")?,
            norm_out: gn(s, "encoder.norm_out")?,
            conv_out: conv(s, "encoder.conv_out")?,
            quant: conv(s, "quant_conv")?,
        })
    }

    /// pixels (b, 3, h, w) in [0, 1], f32 -> latent mean (b, 4, h/8, w/8), f32, unscaled.
    /// `nhwc` when ComfyUI holds the picture as NHWC memory (a LoadImage's): torch then keeps the
    /// residual stream channels-last (x * 2 - 1, the casts, pads and residual adds all keep it), so
    /// conv_in, the shortcut convs and the downsample convs run on cuDNN's NHWC engines, while
    /// every conv after a GroupNorm runs plain.
    pub fn encode(&self, pixels: &Tensor, nhwc: bool) -> Result<Tensor> {
        crate::profile::scope("vae");
        // process_input: x * 2 - 1 in f32, then the VAE's dtype
        let x = pixels.affine(2.0, 0.0)?.affine(1.0, -1.0)?.to_dtype(DType::BF16)?;
        let run = |c: &Conv2d, t: &Tensor| -> Result<Tensor> { Ok(if nhwc { c.forward_nhwc(t)? } else { c.forward(t)? }) };
        let mut h = run(&self.conv_in, &x)?;
        for (blocks, ds) in &self.down {
            for b in blocks {
                h = b.forward_cl(&h, nhwc)?;
            }
            if let Some(c) = ds {
                h = run(c, &h.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?)?;
            }
        }
        h = self.mid2.forward_cl(&self.mid_attn.forward(&self.mid1.forward_cl(&h, nhwc)?)?, nhwc)?;
        let h = self.conv_out.forward(&self.norm_out.forward_silu(&h)?)?;
        let moments = self.quant.forward(&h)?;
        let c = moments.dim(1)? / 2;
        Ok(moments.narrow(1, 0, c)?.to_dtype(DType::F32)?)
    }
}

/// The decoder block by block against refs/sdxl_vae_ref.py (the README lighthouse graph's
/// decode), each block on ComfyUI's input for it, then op by op inside the first middle res block
/// and the middle attention, and the whole decode from the KSampler's latent.
pub fn debug_vae(ckpt: &std::path::Path, refs: &std::path::Path, dev: &candle_core::Device) -> Result<()> {
    use anyhow::Context;
    let file = crate::model::File::open(ckpt)?;
    let src = Source {
        file: &file,
        map: crate::model::vae_map(),
        loras: vec![],
        lora_prefix: "",
        compute: DType::BF16,
        dtype: DType::BF16,
        dev,
        patching: Default::default(),
    };
    let dec = Decoder::load(&src)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, dev).with_context(|| format!("no {n} in the capture")) };
    let bf = |n: &str| -> Result<Tensor> { Ok(get(n)?.to_dtype(DType::BF16)?) };
    let report = |name: &str, ours: &Tensor, theirs: &Tensor| -> Result<()> {
        let ours = ours.to_dtype(DType::F32)?.reshape(theirs.dims())?;
        let d = (&ours - theirs)?.abs()?;
        let differ = d.ne(0f32)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        let max = d.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!(
            "{name:44} {differ:>9} of {:>9} differ, max |d| {max:.3e}{}",
            theirs.elem_count(),
            if differ == 0.0 { " exact" } else { "" }
        );
        Ok(())
    };
    let z = dec.post_quant.forward(&bf("ksampler.latent")?)?;
    report("post_quant_conv (the latent in bf16)", &z, &get("decoder.z")?)?;
    report("conv_in", &dec.conv_in.forward(&bf("conv_in.in")?)?, &get("conv_in")?)?;
    report("mid.block_1", &dec.mid1.forward(&bf("mid.block_1.in")?)?, &get("mid.block_1")?)?;
    report("mid.attn_1", &dec.mid_attn.forward(&bf("mid.attn_1.in")?)?, &get("mid.attn_1")?)?;
    report("mid.block_2", &dec.mid2.forward(&bf("mid.block_2.in")?)?, &get("mid.block_2")?)?;
    let levels = dec.up.len();
    for (li, (blocks, upsample)) in dec.up.iter().enumerate() {
        let i = levels - 1 - li;
        for (j, b) in blocks.iter().enumerate() {
            let n = format!("up.{i}.block.{j}");
            report(&n, &b.forward(&bf(&format!("{n}.in"))?)?, &get(&n)?)?;
        }
        if let Some(c) = upsample {
            let n = format!("up.{i}.upsample");
            let x = bf(&format!("{n}.in"))?;
            let (_, _, hh, ww) = x.dims4()?;
            let up = x.upsample_nearest2d(hh * 2, ww * 2)?;
            if i == 3 {
                report("up.3.upsample: nearest x2", &up, &get("up.3.upsample.conv.in")?)?;
            }
            report(&n, &c.forward(&up)?, &get(&n)?)?;
        }
    }
    report("norm_out", &dec.norm_out.forward(&bf("norm_out.in")?)?, &get("norm_out")?)?;
    report("SiLU (on their norm_out)", &crate::kernels::torch_silu(&bf("norm_out")?)?, &get("conv_out.in")?)?;
    report("norm_out + SiLU fused", &dec.norm_out.forward_silu(&bf("norm_out.in")?)?, &get("conv_out.in")?)?;
    report("conv_out (their SiLU)", &dec.conv_out.forward(&bf("conv_out.in")?)?, &get("conv_out")?)?;
    // inside mid.block_1
    let m = &dec.mid1;
    report("res: norm1 (GroupNorm)", &m.norm1.forward(&bf("mid.block_1.norm1.in")?)?, &get("mid.block_1.norm1")?)?;
    report("res: SiLU (on their norm1)", &crate::kernels::torch_silu(&bf("mid.block_1.norm1")?)?, &get("mid.block_1.conv1.in")?)?;
    report("res: conv1 (their SiLU)", &m.conv1.forward(&bf("mid.block_1.conv1.in")?)?, &get("mid.block_1.conv1")?)?;
    report("res: norm2 (GroupNorm)", &m.norm2.forward(&bf("mid.block_1.norm2.in")?)?, &get("mid.block_1.norm2")?)?;
    report("res: conv2 (their SiLU)", &m.conv2.forward(&bf("mid.block_1.conv2.in")?)?, &get("mid.block_1.conv2")?)?;
    // inside the attention
    let a = &dec.mid_attn;
    report("attn: norm", &a.norm.forward(&bf("mid.attn_1.norm.in")?)?, &get("mid.attn_1.norm")?)?;
    report("attn: q", &a.q.forward(&bf("mid.attn_1.norm")?)?, &get("mid.attn_1.q")?)?;
    report("attn: k", &a.k.forward(&bf("mid.attn_1.norm")?)?, &get("mid.attn_1.k")?)?;
    report("attn: v", &a.v.forward(&bf("mid.attn_1.norm")?)?, &get("mid.attn_1.v")?)?;
    let att = AttnBlock::attend(&bf("mid.attn_1.q")?, &bf("mid.attn_1.k")?, &bf("mid.attn_1.v")?)?;
    report("attn: SDPA (their q, k, v)", &att, &get("mid.attn_1.proj_out.in")?)?;
    report("attn: proj_out", &a.proj.forward(&bf("mid.attn_1.proj_out.in")?)?, &get("mid.attn_1.proj_out")?)?;
    // the whole decode
    let img = dec.decode(&get("ksampler.latent")?)?;
    report("decode (from the KSampler latent)", &img, &get("vae_decode.image")?.permute((0, 3, 1, 2))?)?;
    Ok(())
}
