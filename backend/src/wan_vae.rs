//! The Wan 2.1 VAE (`qwen_image_vae.safetensors`, Anima's and Krea2's) decoding one picture, as
//! ComfyUI's comfy/ldm/wan/vae.py runs it in bf16. A one-frame latent takes no feature cache, so
//! every CausalConv3d takes its fast path: comfy.ops' "causal_zero" autopad keeps the weight's last
//! time tap, and F.conv3d runs that 1x3x3 (or 1x1x1) kernel on the (1, C, 1, H, W) frame with zero
//! spatial padding, through cuDNN's 3D convolution, the bias added after in bf16. The upsamples'
//! time convolutions are skipped without a cache, so the network is a 2D one held in 5D tensors:
//! RMS_norm is F.normalize over the channels times sqrt(C) times gamma (kernels::wan_rms_norm),
//! the middle attention is one head over every pixel (vae::pytorch_attention), and each upsample is
//! nearest-exact x2 then a 2D convolution to half the channels.

use crate::kernels;
use crate::nn::Conv2d;
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

fn bf16(st: &MmapedSafetensors, dev: &Device, name: &str) -> Result<Tensor> {
    Ok(st.load(name, dev).with_context(|| format!("Wan VAE: {name}"))?.to_dtype(DType::BF16)?)
}

/// A CausalConv3d on one frame with nothing cached
struct CausalConv {
    w: Tensor,
    b: Tensor,
    pad: usize,
}

impl CausalConv {
    fn load(st: &MmapedSafetensors, dev: &Device, p: &str) -> Result<Self> {
        let w = bf16(st, dev, &format!("{p}.weight"))?;
        let (out, _, kt, kh, kw) = w.dims5()?;
        if kh != kw || kh % 2 == 0 {
            bail!("{p}: a {kt}x{kh}x{kw} kernel");
        }
        // weight[:, :, -1:], made contiguous as cudnn_convolution makes its weight
        let w = w.narrow(2, kt - 1, 1)?.contiguous()?;
        let b = bf16(st, dev, &format!("{p}.bias"))?.reshape((1, out, 1, 1, 1))?;
        // CausalConv3d(..., padding=p) keeps (0, p, p); every one here pads k // 2
        Ok(CausalConv { w, b, pad: kh / 2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = kernels::cudnn_conv(x, &self.w, &[0, self.pad, self.pad], &[1, 1, 1], &[1, 1, 1])?;
        Ok(y.broadcast_add(&self.b)?)
    }
}

/// ResidualBlock: RMS_norm, SiLU, conv, RMS_norm, SiLU, conv, plus the input (through a 1x1x1
/// conv when the channels change)
struct ResBlock {
    norm1: Tensor,
    conv1: CausalConv,
    norm2: Tensor,
    conv2: CausalConv,
    shortcut: Option<CausalConv>,
}

impl ResBlock {
    fn load(st: &MmapedSafetensors, dev: &Device, p: &str) -> Result<Self> {
        Ok(ResBlock {
            norm1: bf16(st, dev, &format!("{p}.residual.0.gamma"))?,
            conv1: CausalConv::load(st, dev, &format!("{p}.residual.2"))?,
            norm2: bf16(st, dev, &format!("{p}.residual.3.gamma"))?,
            conv2: CausalConv::load(st, dev, &format!("{p}.residual.6"))?,
            shortcut: if st.get(&format!("{p}.shortcut.weight")).is_ok() {
                Some(CausalConv::load(st, dev, &format!("{p}.shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&kernels::wan_rms_norm(x, &self.norm1, true)?)?;
        let h = self.conv2.forward(&kernels::wan_rms_norm(&h, &self.norm2, true)?)?;
        let s = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok((h + s)?)
    }
}

/// AttentionBlock: on (b t) c h w, RMS_norm, a 1x1 conv to q, k and v, one head of attention over
/// the pixels, a 1x1 projection, plus the input
struct AttnBlock {
    norm: Tensor,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl AttnBlock {
    fn load(st: &MmapedSafetensors, dev: &Device, p: &str) -> Result<Self> {
        Ok(AttnBlock {
            norm: bf16(st, dev, &format!("{p}.norm.gamma"))?,
            to_qkv: conv2d(st, dev, &format!("{p}.to_qkv"), 0)?,
            proj: conv2d(st, dev, &format!("{p}.proj"), 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let flat = x.reshape((b * t, c, h, w))?;
        let qkv = self.to_qkv.forward(&kernels::wan_rms_norm(&flat, &self.norm, false)?)?;
        let a = crate::vae::pytorch_attention(&qkv.narrow(1, 0, c)?, &qkv.narrow(1, c, c)?, &qkv.narrow(1, 2 * c, c)?)?;
        let y = self.proj.forward(&a)?.reshape((b, c, t, h, w))?;
        Ok((y + x)?)
    }
}

fn conv2d(st: &MmapedSafetensors, dev: &Device, p: &str, padding: usize) -> Result<Conv2d> {
    Ok(Conv2d { w: bf16(st, dev, &format!("{p}.weight"))?, b: Some(bf16(st, dev, &format!("{p}.bias"))?), stride: 1, padding })
}

enum Up {
    Res(ResBlock),
    /// Resample("upsample2d" or "upsample3d", the time conv unused for one frame): nearest-exact x2,
    /// then a 3x3 conv to half the channels
    Resample(Conv2d),
}

pub struct WanDecoder {
    conv2: CausalConv,
    conv1: CausalConv,
    middle: (ResBlock, AttnBlock, ResBlock),
    ups: Vec<Up>,
    head_norm: Tensor,
    head_conv: CausalConv,
}

impl WanDecoder {
    pub fn load(file: &Path, dev: &Device) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(file)? };
        if st.get("decoder.upsamples.0.upsamples.0.residual.2.weight").is_ok() || st.get("decoder.middle.0.residual.0.gamma").is_err() {
            bail!("{}: not a Wan 2.1 VAE", file.display());
        }
        let mut ups = vec![];
        let mut i = 0;
        loop {
            let p = format!("decoder.upsamples.{i}");
            if st.get(&format!("{p}.residual.0.gamma")).is_ok() {
                ups.push(Up::Res(ResBlock::load(&st, dev, &p)?));
            } else if st.get(&format!("{p}.resample.1.weight")).is_ok() {
                ups.push(Up::Resample(conv2d(&st, dev, &format!("{p}.resample.1"), 1)?));
            } else {
                break;
            }
            i += 1;
        }
        Ok(WanDecoder {
            conv2: CausalConv::load(&st, dev, "conv2")?,
            conv1: CausalConv::load(&st, dev, "decoder.conv1")?,
            middle: (
                ResBlock::load(&st, dev, "decoder.middle.0")?,
                AttnBlock::load(&st, dev, "decoder.middle.1")?,
                ResBlock::load(&st, dev, "decoder.middle.2")?,
            ),
            ups,
            head_norm: bf16(&st, dev, "decoder.head.0.gamma")?,
            head_conv: CausalConv::load(&st, dev, "decoder.head.2")?,
        })
    }

    /// WanVAE.decode of a (1, 16, 1, h, w) latent, already through Wan21's process_out, as ComfyUI's
    /// VAE.decode hands it over (in bf16) -> (1, 3, 1, 8h, 8w) bf16, before process_output
    pub fn decode_raw(&self, z: &Tensor, mut stage: impl FnMut(&str, &Tensor)) -> Result<Tensor> {
        let (b, _, t, _, _) = z.dims5()?;
        if b != 1 || t != 1 {
            bail!("the Wan VAE decodes one picture so far, not {b}x{t} frames");
        }
        let z = z.to_dtype(DType::BF16)?;
        let x = self.conv2.forward(&z)?;
        stage("conv2", &x);
        let x = self.conv1.forward(&x)?;
        stage("conv1", &x);
        let x = self.middle.0.forward(&x)?;
        let x = self.middle.1.forward(&x)?;
        let mut x = self.middle.2.forward(&x)?;
        stage("middle", &x);
        for (i, up) in self.ups.iter().enumerate() {
            x = match up {
                Up::Res(r) => r.forward(&x)?,
                Up::Resample(c) => {
                    let (b, ch, t, h, w) = x.dims5()?;
                    let y = c.forward(&x.reshape((b * t, ch, h, w))?.upsample_nearest2d(h * 2, w * 2)?)?;
                    let (_, co, h2, w2) = y.dims4()?;
                    y.reshape((b, co, t, h2, w2))?
                }
            };
            stage(&format!("up{i}"), &x);
        }
        let x = self.head_conv.forward(&kernels::wan_rms_norm(&x, &self.head_norm, true)?)?;
        stage("head", &x);
        Ok(x)
    }

    /// The picture in [0, 1], (1, 3, 8h, 8w) f32: decode_raw, then process_output
    /// (image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)) on its f32 copy
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        crate::profile::scope("vae");
        let x = self.decode_raw(z, |_, _| {})?.to_dtype(DType::F32)?;
        let x = x.affine(1.0, 1.0)?.broadcast_div(&Tensor::new(2f32, x.device())?)?.clamp(0f32, 1f32)?;
        Ok(x.squeeze(2)?)
    }
}

/// debug-wan-vae VAE REF.safetensors (refs/wan_vae_ref.py): the reference's latent through the
/// decoder here, every stage compared with ComfyUI's
pub fn debug_decode(vae: &Path, reference: &Path, dev: &Device) -> Result<()> {
    let refs = unsafe { MmapedSafetensors::new(reference)? };
    let dec = WanDecoder::load(vae, dev)?;
    let z = refs.load("latent", dev)?;
    let mut bad = 0;
    let mut check = |name: &str, got: &Tensor| {
        let Ok(want) = refs.load(name, dev) else {
            println!("{name:8} {:?} (no reference)", got.dims());
            return;
        };
        let report = (|| -> Result<String> {
            let (g, w) = (got.to_dtype(DType::F32)?, want.to_dtype(DType::F32)?);
            if g.dims() != w.dims() {
                return Ok(format!("shape {:?} against {:?}", g.dims(), w.dims()));
            }
            let d = (g - w)?.abs()?.flatten_all()?;
            let n = d.elem_count();
            let differ = d.gt(0f32)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
            let max = d.max(0)?.to_scalar::<f32>()?;
            Ok(if differ == 0 {
                format!("{:?} identical", got.dims())
            } else {
                format!("{:?} {differ} of {n} differ, max {max}", got.dims())
            })
        })();
        let report = report.unwrap_or_else(|e| format!("error {e}"));
        if !report.ends_with("identical") {
            bad += 1;
        }
        println!("{name:8} {report}");
    };
    let out = dec.decode_raw(&z, |name, t| check(name, t))?;
    let pic = out.to_dtype(DType::F32)?;
    let pic = pic.affine(1.0, 1.0)?.broadcast_div(&Tensor::new(2f32, pic.device())?)?.clamp(0f32, 1f32)?;
    check("image", &pic);
    println!("{}", if bad == 0 { "all identical".to_string() } else { format!("{bad} stages differ") });
    Ok(())
}
