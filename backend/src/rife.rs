//! RIFE 4.7 (rife47.pth, rife49.pth) as ComfyUI-Frame-Interpolation's RIFE VFI node runs it
//! (vfi_models/rife: IFNet with arch "4.7"), f32 on the card with torch's own kernels: cuDNN
//! convolutions and transposed convolutions (cudnn_conv, TF32 engines allowed, as torch allows
//! them), each bias added after; torch's CUDA bilinear upsample, grid_sample (bilinear, border,
//! align_corners), linspace, leaky_relu and sigmoid (ops.cu).
//!
//! The node calls `IFNet(frame0, frame1, timestep, scale_list, fast_mode, ensemble)`, one position
//! off from IFNet.forward's (img0, img1, timestep, scale_list, training, fastmode, ensemble): its
//! ensemble lands in `fastmode`, which 4.7 never reads, so ensembling never runs. Ported as it runs.

use crate::kernels;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub struct Rife {
    w: HashMap<String, Tensor>,
}

impl Rife {
    /// The checkpoint's f32 state dict, onto the card. 4.7's layout only (rife47 and rife49).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let tensors = candle_core::pickle::read_all(path).with_context(|| format!("read {}", path.display()))?;
        let mut w = HashMap::new();
        for (k, v) in tensors {
            w.insert(k, v.to_dtype(DType::F32)?.to_device(dev)?);
        }
        for need in ["encode.0.weight", "encode.1.weight", "block0.conv0.0.0.weight", "block3.lastconv.0.weight"] {
            if !w.contains_key(need) {
                bail!("{}: not a RIFE 4.7 checkpoint (no {need})", path.display());
            }
        }
        if w.contains_key("block0.convblock.0.conv.weight") && w["block0.conv0.0.0.weight"].dim(1)? != 15 {
            bail!("{}: not RIFE 4.7's IFNet (its first block does not take 15 channels)", path.display());
        }
        Ok(Self { w })
    }

    fn t(&self, name: &str) -> Result<&Tensor> {
        self.w.get(name).with_context(|| format!("RIFE: no {name} in the checkpoint"))
    }

    /// nn.Conv2d (3x3, bias): cudnn_convolution, then the bias added; `nhwc` when torch's input is
    /// channels-last memory (it then runs on cuDNN's NHWC engines)
    fn conv(&self, x: &Tensor, p: &str, stride: usize, nhwc: bool) -> Result<Tensor> {
        let w = self.t(&format!("{p}.weight"))?;
        let y = if nhwc {
            kernels::cudnn_conv_nhwc(x, w, &[1, 1], &[stride, stride], &[1, 1])?
        } else {
            kernels::cudnn_conv(x, w, &[1, 1], &[stride, stride], &[1, 1])?
        };
        Ok(y.broadcast_add(&self.t(&format!("{p}.bias"))?.reshape((1, (), 1, 1))?)?)
    }

    /// nn.ConvTranspose2d (4x4, stride 2, padding 1, bias): cudnn_convolution_transpose, then the
    /// bias; `nhwc` as for conv
    fn deconv(&self, x: &Tensor, p: &str, nhwc: bool) -> Result<Tensor> {
        let w = self.t(&format!("{p}.weight"))?;
        let y = if nhwc {
            kernels::cudnn_conv_transpose_nhwc(x, w, &[1, 1], &[2, 2], &[1, 1], &[0, 0], 1)?
        } else {
            kernels::cudnn_conv_transpose(x, w, &[1, 1], &[2, 2], &[1, 1], &[0, 0], 1)?
        };
        Ok(y.broadcast_add(&self.t(&format!("{p}.bias"))?.reshape((1, (), 1, 1))?)?)
    }

    /// encode: Conv2d(3, 16, 3, 2, 1), ConvTranspose2d(16, 4, 4, 2, 1). `nhwc` when the node's IMAGE
    /// is NHWC memory: its einops rearrange is then a channels-last view, which .to(), clamp and
    /// pad keep, so torch runs both on cuDNN's NHWC engines (a VAEDecode's IMAGE is NCHW memory and
    /// runs plain). The blocks' inputs are contiguous either way (a cat with the timestep plane).
    fn encode(&self, x: &Tensor, nhwc: bool) -> Result<Tensor> {
        self.deconv(&self.conv(x, "encode.0", 2, nhwc)?, "encode.1", nhwc)
    }

    /// IFBlock.forward for 4.7: x and the flow resized by 1 / scale, conv0 (two stride-2 convs with
    /// LeakyReLU 0.2), eight ResConv (LeakyReLU(conv(x) * beta + x)), lastconv (a transposed conv
    /// and PixelShuffle(2)), resized back by scale; the flow is its first 4 channels times scale,
    /// the mask its fifth.
    fn block(&self, i: usize, x: &Tensor, flow: Option<&Tensor>, scale: f64) -> Result<(Tensor, Tensor)> {
        let p = format!("block{i}");
        let down = |t: &Tensor| -> Result<Tensor> {
            let (_, _, h, w) = t.dims4()?;
            let f = 1.0 / scale;
            Ok(kernels::torch_upsample_bilinear2d_f32(t, (h as f64 * f).floor() as usize, (w as f64 * f).floor() as usize, Some((f, f)))?)
        };
        let mut x = down(x)?;
        if let Some(flow) = flow {
            // * 1.0 / scale: a multiply by 1.0, then by the f32 reciprocal of scale
            let f = down(flow)?;
            let f = f.broadcast_mul(&Tensor::new(&[1.0f32], f.device())?)?;
            let f = f.broadcast_mul(&Tensor::new(&[1.0f32 / scale as f32], f.device())?)?;
            x = Tensor::cat(&[&x, &f], 1)?;
        }
        let leaky = |t: Tensor| kernels::torch_leaky_relu_f32(&t, 0.2);
        let mut feat = leaky(self.conv(&x, &format!("{p}.conv0.0.0"), 2, false)?)?;
        feat = leaky(self.conv(&feat, &format!("{p}.conv0.1.0"), 2, false)?)?;
        for j in 0..8 {
            let c = self.conv(&feat, &format!("{p}.convblock.{j}.conv"), 1, false)?;
            let r = (c.broadcast_mul(self.t(&format!("{p}.convblock.{j}.beta"))?)? + &feat)?;
            feat = leaky(r)?;
        }
        let tmp = pixel_shuffle(&self.deconv(&feat, &format!("{p}.lastconv.0"), false)?, 2)?;
        let (_, _, h, w) = tmp.dims4()?;
        let tmp = kernels::torch_upsample_bilinear2d_f32(
            &tmp,
            (h as f64 * scale).floor() as usize,
            (w as f64 * scale).floor() as usize,
            Some((scale, scale)),
        )?;
        let flow = tmp.narrow(1, 0, 4)?.broadcast_mul(&Tensor::new(&[scale as f32], tmp.device())?)?;
        let mask = tmp.narrow(1, 4, 1)?.contiguous()?;
        Ok((flow, mask))
    }

    /// IFNet.forward for 4.7 as the node calls it (no ensemble): img0 and img1 (B, 3, h, w) f32 on
    /// the card, the frame between them at `timestep`, `scales` the node's scale_list, `nhwc` as for
    /// encode. (B, 3, h, w).
    pub fn forward(&self, img0: &Tensor, img1: &Tensor, timestep: f32, scales: &[f64], nhwc: bool) -> Result<Tensor> {
        if scales.len() != 4 {
            bail!("RIFE 4.7 takes four scales, not {}", scales.len());
        }
        let (img0, img1) = (img0.clamp(0f32, 1f32)?, img1.clamp(0f32, 1f32)?);
        let (b, _, h, w) = img0.dims4()?;
        let (ph, pw) = (((h - 1) / 64 + 1) * 64, ((w - 1) / 64 + 1) * 64);
        let pad = |t: &Tensor| -> Result<Tensor> { Ok(t.pad_with_zeros(3, 0, pw - w)?.pad_with_zeros(2, 0, ph - h)?) };
        let (img0, img1) = (pad(&img0)?, pad(&img1)?);
        let dev = img0.device().clone();
        let ts = Tensor::full(timestep, (b, 1, ph, pw), &dev)?;
        let (f0, f1) = (self.encode(&img0, nhwc)?, self.encode(&img1, nhwc)?);
        let (mut wi0, mut wi1) = (img0.clone(), img1.clone());
        let mut flow: Option<Tensor> = None;
        let mut mask: Option<Tensor> = None;
        for (i, &scale) in scales.iter().enumerate() {
            let (fl, m) = match (&flow, &mask) {
                (Some(fl), Some(m)) => {
                    let x = Tensor::cat(&[&wi0, &wi1, &warp(&f0, &fl.narrow(1, 0, 2)?)?, &warp(&f1, &fl.narrow(1, 2, 2)?)?, &ts, m], 1)?;
                    let (fd, m0) = self.block(i, &x, Some(fl), scale)?;
                    ((fl + fd)?, m0)
                }
                _ => self.block(i, &Tensor::cat(&[&img0, &img1, &f0, &f1, &ts], 1)?, None, scale)?,
            };
            wi0 = warp(&img0, &fl.narrow(1, 0, 2)?)?;
            wi1 = warp(&img1, &fl.narrow(1, 2, 2)?)?;
            flow = Some(fl);
            mask = Some(m);
        }
        let m = kernels::torch_sigmoid_f32(mask.as_ref().context("RIFE: no mask")?)?;
        let merged = (wi0.broadcast_mul(&m)? + wi1.broadcast_mul(&m.affine(-1.0, 1.0)?)?)?;
        Ok(merged.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?)
    }
}

/// nn.PixelShuffle(r): (B, C * r * r, H, W) to (B, C, H * r, W * r)
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let oc = c / (r * r);
    Ok(x.reshape((b, oc, r, r, h, w))?.permute((0, 1, 4, 2, 5, 3))?.reshape((b, oc, h * r, w * r))?)
}

/// rife_arch.warp: grid_sample over a grid of linspace(-1, 1) plus the flow over (size - 1) / 2,
/// each divided as torch divides by a python float (a multiply by its f32 reciprocal), bilinear,
/// border padding, align_corners
fn warp(input: &Tensor, flow: &Tensor) -> Result<Tensor> {
    let (b, _, fh, fw) = flow.dims4()?;
    let (_, _, ih, iw) = input.dims4()?;
    let dev = flow.device();
    let gx = kernels::torch_linspace_f32(-1.0, 1.0, fw, dev)?.reshape((1, 1, 1, fw))?.broadcast_as((b, 1, fh, fw))?;
    let gy = kernels::torch_linspace_f32(-1.0, 1.0, fh, dev)?.reshape((1, 1, fh, 1))?.broadcast_as((b, 1, fh, fw))?;
    let base = Tensor::cat(&[&gx.contiguous()?, &gy.contiguous()?], 1)?;
    let inv_x = 1.0f32 / ((iw as f64 - 1.0) / 2.0) as f32;
    let inv_y = 1.0f32 / ((ih as f64 - 1.0) / 2.0) as f32;
    let fx = flow.narrow(1, 0, 1)?.broadcast_mul(&Tensor::new(&[inv_x], dev)?)?;
    let fy = flow.narrow(1, 1, 1)?.broadcast_mul(&Tensor::new(&[inv_y], dev)?)?;
    let grid = (base + Tensor::cat(&[&fx, &fy], 1)?)?;
    Ok(kernels::torch_grid_sample_border_f32(input, &grid)?)
}

/// RIFE VFI's frame loop for one checkpoint, as the node runs it with batch_size 1: between each
/// pair, multiplier - 1 frames at step / multiplier, each clamped to [0, 1]; the originals kept.
/// `frames` an IMAGE (T, 3, H, W) f32 in [0, 1], wherever it is; each pair goes to the model's
/// card and the frames come back where the input was; `nhwc` whether ComfyUI holds that IMAGE as
/// NHWC memory (Engine::image_nhwc). (T + (T - 1) * (multiplier - 1), 3, H, W).
pub fn interpolate(
    model: &Rife,
    frames: &Tensor,
    multiplier: usize,
    scale_factor: f64,
    nhwc: bool,
    mut each: impl FnMut(usize, usize),
) -> Result<Tensor> {
    let n = frames.dim(0)?;
    if n < 2 {
        bail!("RIFE VFI needs at least 2 frames, got {n}");
    }
    let card = model.w.values().next().context("RIFE: no weights")?.device().clone();
    let scales: Vec<f64> = [8.0, 4.0, 2.0, 1.0].iter().map(|s| s / scale_factor).collect();
    let total = (n - 1) * multiplier.saturating_sub(1);
    let mut out = Vec::with_capacity(n + total);
    let mut done = 0;
    for p in 0..n - 1 {
        out.push(frames.narrow(0, p, 1)?);
        let a = frames.narrow(0, p, 1)?.to_device(&card)?.to_dtype(DType::F32)?;
        let b = frames.narrow(0, p + 1, 1)?.to_device(&card)?.to_dtype(DType::F32)?;
        for step in 1..multiplier {
            // torch.tensor([step / m], dtype=float32): the python double rounded to f32
            let t = (step as f64 / multiplier as f64) as f32;
            let mid = model.forward(&a, &b, t, &scales, nhwc)?.clamp(0f32, 1f32)?;
            out.push(mid.to_device(frames.device())?);
            done += 1;
            each(done, total);
        }
    }
    out.push(frames.narrow(0, n - 1, 1)?);
    Ok(Tensor::cat(&out, 0)?)
}

/// debug-rife CKPT REF (refs/rife_ref.py): the first IFNet call stage by stage against
/// ComfyUI's, then the node's whole output. The capture's frames are NHWC memory (read from a
/// file, as a LoadImage's are), so its encoder ran channels-last.
pub fn debug_rife(ckpt: &Path, reference: &Path) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let model = Rife::load(ckpt, &dev)?;
    let (img0, img1) = (get("in.img0")?, get("in.img1")?);
    let (_, _, h, w) = img0.dims4()?;
    let (ph, pw) = (((h - 1) / 64 + 1) * 64, ((w - 1) / 64 + 1) * 64);
    let pad = |t: &Tensor| -> Result<Tensor> { Ok(t.clamp(0f32, 1f32)?.pad_with_zeros(3, 0, pw - w)?.pad_with_zeros(2, 0, ph - h)?) };
    let (p0, p1) = (pad(&img0)?, pad(&img1)?);
    crate::ltx::compare("encode(img0)", &model.encode(&p0, true)?, &get("encode0")?)?;
    crate::ltx::compare("encode(img1)", &model.encode(&p1, true)?, &get("encode1")?)?;
    // each block from ComfyUI's own input, then the warps from ComfyUI's tensors
    for i in 0..4 {
        let x = get(&format!("b{i}.x"))?;
        let flow_in = get(&format!("b{i}.flow_in")).ok();
        let (fl, m) = model.block(i, &x, flow_in.as_ref(), [8.0, 4.0, 2.0, 1.0][i])?;
        crate::ltx::compare(&format!("block{i} flow"), &fl, &get(&format!("b{i}.flow"))?)?;
        crate::ltx::compare(&format!("block{i} mask"), &m, &get(&format!("b{i}.mask"))?)?;
    }
    for k in 0..14 {
        let (inp, fl) = (get(&format!("warp{k}.in"))?, get(&format!("warp{k}.flow"))?);
        crate::ltx::compare(&format!("warp{k}"), &warp(&inp, &fl)?, &get(&format!("warp{k}.out"))?)?;
    }
    let t = get("in.timestep")?.flatten_all()?.to_vec1::<f32>()?[0];
    crate::ltx::compare("IFNet (whole call)", &model.forward(&img0, &img1, t, &[8.0, 4.0, 2.0, 1.0], true)?, &get("out")?)?;
    // the node's IMAGE (T, H, W, 3) as rustyui holds one, (T, 3, H, W)
    let frames = get("node.in")?.permute((0, 3, 1, 2))?.contiguous()?;
    let out = interpolate(&model, &frames, 2, 1.0, true, |_, _| {})?;
    crate::ltx::compare("RIFE VFI (the node's frames)", &out.permute((0, 2, 3, 1))?.contiguous()?, &get("node.out")?)?;
    Ok(())
}
