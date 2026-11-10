//! LTX-2's latent upsamplers (comfy/ldm/lightricks/latent_upsampler.py: LatentUpsampler) as
//! LatentUpscaleModelLoader loads them and LTXVLatentUpsampler runs them: in bf16 (ComfyUI's VAE
//! dtype on this card), every convolution torch's cuDNN one (kernels::cudnn_conv), every GroupNorm
//! torch's, the VAE's per-channel statistics around the model as the VAE's dtype rounds them.

use crate::kernels;
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

/// The model's own config, from its safetensors metadata (LatentUpsampler.from_config)
#[derive(Debug, Clone)]
pub struct Config {
    pub num_blocks: usize,
    pub dims: usize,
    pub spatial: bool,
    pub temporal: bool,
    pub rational: bool,
}

/// A convolution and its bias, added afterwards as F.conv* adds it
struct Conv {
    w: Tensor,
    b: Tensor,
}

impl Conv {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            w: st.load(&format!("{p}.weight"), dev)?.to_dtype(DType::BF16)?,
            b: st.load(&format!("{p}.bias"), dev)?.to_dtype(DType::BF16)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ns = x.rank() - 2;
        let y = kernels::cudnn_conv(x, &self.w, &vec![1; ns], &vec![1; ns], &vec![1; ns])?;
        let mut shape = vec![1usize; x.rank()];
        shape[1] = self.b.dim(0)?;
        Ok(y.broadcast_add(&self.b.reshape(shape)?)?)
    }
}

/// GroupNorm(32, C), eps 1e-5
struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            w: st.load(&format!("{p}.weight"), dev)?.to_dtype(DType::BF16)?,
            b: st.load(&format!("{p}.bias"), dev)?.to_dtype(DType::BF16)?,
        })
    }

    fn forward(&self, x: &Tensor, silu: bool) -> Result<Tensor> {
        Ok(kernels::group_norm(x, &self.w, &self.b, 32, 1e-5, silu)?)
    }
}

/// ResBlock: conv, norm, SiLU, conv, norm, then SiLU of the sum with the input
struct ResBlock {
    conv1: Conv,
    norm1: Norm,
    conv2: Conv,
    norm2: Norm,
}

impl ResBlock {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(st, &format!("{p}.conv1"), dev)?,
            norm1: Norm::load(st, &format!("{p}.norm1"), dev)?,
            conv2: Conv::load(st, &format!("{p}.conv2"), dev)?,
            norm2: Norm::load(st, &format!("{p}.norm2"), dev)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(&self.conv1.forward(x)?, true)?;
        let h = self.norm2.forward(&self.conv2.forward(&h)?, false)?;
        Ok(kernels::ltx_silu(&kernels::ltx_add(&h, x)?)?)
    }
}

pub struct LatentUpsampler {
    pub config: Config,
    initial_conv: Conv,
    initial_norm: Norm,
    res_blocks: Vec<ResBlock>,
    upsampler: Conv,
    post: Vec<ResBlock>,
    final_conv: Conv,
}

/// A safetensors file's `__metadata__`
pub(crate) fn metadata(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len) as usize;
    anyhow::ensure!(len < 100 << 20, "{}: a {len}-byte safetensors header", path.display());
    let mut header = vec![0u8; len];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(v.get("__metadata__").and_then(|m| m.as_object()).cloned().unwrap_or_default())
}

impl LatentUpsampler {
    /// LatentUpscaleModelLoader's LTX branch (a file with post_upsample_res_blocks)
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path)? };
        if st.get("post_upsample_res_blocks.0.conv2.bias").is_err() {
            bail!("{}: not an LTX latent upsampler (the Hunyuan Video 1.5 ones are not implemented)", path.display());
        }
        let meta = metadata(path)?;
        let cfg: serde_json::Value =
            serde_json::from_str(meta.get("config").and_then(|c| c.as_str()).context("no config in the file's metadata")?)?;
        let get = |k: &str| cfg.get(k);
        let config = Config {
            num_blocks: get("num_blocks_per_stage").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
            dims: get("dims").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            spatial: get("spatial_upsample").and_then(|v| v.as_bool()).unwrap_or(true),
            temporal: get("temporal_upsample").and_then(|v| v.as_bool()).unwrap_or(false),
            rational: get("rational_resampler").and_then(|v| v.as_bool()).unwrap_or(false),
        };
        if config.dims != 3 {
            bail!("{}: a {}-D latent upsampler is not implemented (the 3-D ones are)", path.display(), config.dims);
        }
        if config.spatial && config.temporal {
            bail!("{}: a spatial and temporal upsampler in one is not implemented", path.display());
        }
        if config.spatial && config.rational {
            bail!("{}: the rational (x1.5) resampler is not implemented yet", path.display());
        }
        let blocks =
            |p: &str| -> Result<Vec<ResBlock>> { (0..config.num_blocks).map(|i| ResBlock::load(&st, &format!("{p}.{i}"), dev)).collect() };
        Ok(Self {
            initial_conv: Conv::load(&st, "initial_conv", dev)?,
            initial_norm: Norm::load(&st, "initial_norm", dev)?,
            res_blocks: blocks("res_blocks")?,
            upsampler: Conv::load(&st, "upsampler.0", dev)?,
            post: blocks("post_upsample_res_blocks")?,
            final_conv: Conv::load(&st, "final_conv", dev)?,
            config,
        })
    }

    /// The model on an un-normalized bf16 latent (1, 128, F, H, W)
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, _, f, h, w) = latent.dims5()?;
        let mut x = self.initial_norm.forward(&self.initial_conv.forward(latent)?, true)?;
        for blk in &self.res_blocks {
            x = blk.forward(&x)?;
        }
        x = if self.config.temporal {
            // Conv3d to 2C, "b (c p1) f h w -> b c (f p1) h w", the first frame dropped
            let y = self.upsampler.forward(&x)?;
            let c = y.dim(1)? / 2;
            let y = y.reshape((b, c, 2, f, h, w))?.permute((0, 1, 3, 2, 4, 5))?.reshape((b, c, 2 * f, h, w))?;
            y.narrow(2, 1, 2 * f - 1)?.contiguous()?
        } else {
            // per frame: Conv2d to 4C, "b (c p1 p2) h w -> b c (h p1) (w p2)"
            let c = x.dim(1)?;
            let frames = x.permute((0, 2, 1, 3, 4))?.reshape((b * f, c, h, w))?.contiguous()?;
            let y = self.upsampler.forward(&frames)?;
            let y = y.reshape((b * f, c, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.reshape((b * f, c, 2 * h, 2 * w))?;
            y.reshape((b, f, c, 2 * h, 2 * w))?.permute((0, 2, 1, 3, 4))?.contiguous()?
        };
        for blk in &self.post {
            x = blk.forward(&x)?;
        }
        self.final_conv.forward(&x)
    }
}

/// LTXVLatentUpsampler: the f32 latent in bf16, un-normalized by the VAE's per-channel statistics,
/// through the model, normalized again, back to f32. `std` and `mean` are the VAE's (128,) bf16.
pub fn upsample(model: &LatentUpsampler, latent: &Tensor, std: &Tensor, mean: &Tensor) -> Result<Tensor> {
    let x = kernels::ltx_channel_affine(&latent.to_dtype(DType::BF16)?, std, mean, false)?;
    let y = model.forward(&x)?;
    Ok(kernels::ltx_channel_affine(&y, std, mean, true)?.to_dtype(DType::F32)?)
}

/// debug-upsampler REF [--vae FILE]: both latent upscalers of a typical image-to-video graph against ComfyUI's
/// (refs/upsampler_ref.py), stage by stage
pub fn debug_upsampler(reference: &Path, models: &Path) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let (std, mean) = (get("std")?.to_dtype(DType::BF16)?, get("mean")?.to_dtype(DType::BF16)?);
    let latent = get("latent")?;
    for (tag, file) in
        [("spatial", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), ("temporal", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors")]
    {
        let m = LatentUpsampler::load(&models.join("latent_upscale_models").join(file), &dev)?;
        let x = kernels::ltx_channel_affine(&latent.to_dtype(DType::BF16)?, &std, &mean, false)?;
        crate::ltx::compare(&format!("{tag}: un-normalized input"), &x, &get(&format!("{tag}.model_in"))?)?;
        let c1 = m.initial_conv.forward(&x)?;
        crate::ltx::compare(&format!("{tag}: initial conv"), &c1, &get(&format!("{tag}.initial_conv"))?)?;
        let n1 = m.initial_norm.forward(&c1, false)?;
        crate::ltx::compare(&format!("{tag}: initial norm"), &n1, &get(&format!("{tag}.initial_norm"))?)?;
        let mut h = kernels::ltx_silu(&n1)?;
        for (i, blk) in m.res_blocks.iter().enumerate() {
            h = blk.forward(&h)?;
            if i == 0 {
                crate::ltx::compare(&format!("{tag}: first res block"), &h, &get(&format!("{tag}.res0"))?)?;
            }
        }
        crate::ltx::compare(&format!("{tag}: last res block"), &h, &get(&format!("{tag}.res_last"))?)?;
        let out = upsample(&m, &latent, &std, &mean)?;
        crate::ltx::compare(&format!("{tag}: LTXVLatentUpsampler's latent"), &out, &get(&format!("{tag}.out"))?)?;
    }
    Ok(())
}
