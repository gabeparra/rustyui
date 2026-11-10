//! comfyui_controlnet_aux's DepthAnythingV2Preprocessor: Depth Anything V2 (a DINOv2 ViT and its
//! DPT head; Small, Base or Large, the checkpoint the node names) as the node runs it. On the host,
//! as OpenCV 5.0.0 and numpy compute them: the picture's bicubic resize to the network's size (a
//! multiple of 14, the shorter side 518) in double, the normalisation, the depth's scaling to uint8
//! and its resize to the detect resolution. On the card in f32 as torch 2.11 runs each op: the
//! patch embedding and every DPT convolution on cuDNN in NCHW memory (see `conv2d`), the position
//! embedding's bicubic resize,
//! LayerNorm, the linears, attention as DINOv2 writes it (two strided batched GEMMs around torch's
//! warp softmax), GELU, the bilinear resizes with align_corners.
//!
//! The uint8 resize at the end goes where OpenCV sends it: a resolution above the picture's
//! shorter side is a bicubic enlargement, which OpenCV hands to Intel IPP, linked here as OpenCV
//! bundles it (`resize_u8_cubic`, kernels/ipp_resize.c); one below it is OpenCV's area downscale,
//! its own code, ported (`resize_u8_area`).

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

struct Block {
    norm1: (Tensor, Tensor),
    qkv: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    ls1: Tensor,
    norm2: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    ls2: Tensor,
}

/// A convolution's weight and bias (none for the scratch layers)
type Conv = (Tensor, Option<Tensor>);

struct Fusion {
    out: Conv,
    res1: [Conv; 2],
    res2: [Conv; 2],
}

pub struct DepthAnything {
    patch: Conv,
    cls: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    norm: (Tensor, Tensor),
    layers: Vec<usize>,
    projects: Vec<Conv>,
    resize: Vec<Option<Conv>>,
    rn: Vec<Conv>,
    fusion: Vec<Fusion>,
    out1: Conv,
    out2a: Conv,
    out2b: Conv,
}

/// LayerNorm (eps 1e-6) on the last dimension of an f32 tensor, torch's kernel
fn layer_norm(x: &Tensor, p: &(Tensor, Tensor)) -> Result<Tensor> {
    Ok(crate::kernels::layer_norm(x, &p.0, &p.1, 1e-6)?)
}

/// F.linear on (rows, k) f32 with a bias, as torch's addmm runs it
fn linear(x: &Tensor, p: &(Tensor, Tensor)) -> Result<Tensor> {
    Ok(crate::kernels::torch_linear_bias_f32(x, &p.0, &p.1)?)
}

/// Conv2d as torch runs the DPT head's, torch's bias added after: NCHW memory throughout. The
/// tokens' permute and reshape look channels-last, but reshape gives the size-one batch dim a
/// stride of C, which torch's channels-last test refuses, so the first convolution copies its
/// input to NCHW and every output after it is NCHW too.
fn conv2d(x: &Tensor, c: &Conv, stride: usize, padding: usize) -> Result<Tensor> {
    let y = crate::kernels::cudnn_conv(x, &c.0, &[padding, padding], &[stride, stride], &[1, 1])?;
    Ok(match &c.1 {
        Some(b) => y.broadcast_add(&b.reshape((1, (), 1, 1))?)?,
        None => y,
    })
}

impl DepthAnything {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let t: HashMap<String, Tensor> =
            candle_core::pickle::read_all(path).with_context(|| format!("read {}", path.display()))?.into_iter().collect();
        let get = |n: &str| -> Result<Tensor> {
            Ok(t.get(n).with_context(|| format!("{}: no {n}", path.display()))?.to_dtype(DType::F32)?.to_device(dev)?)
        };
        let pair = |p: &str| -> Result<(Tensor, Tensor)> { Ok((get(&format!("{p}.weight"))?, get(&format!("{p}.bias"))?)) };
        let conv = |p: &str| -> Result<Conv> {
            Ok((get(&format!("{p}.weight"))?, t.contains_key(&format!("{p}.bias")).then(|| get(&format!("{p}.bias"))).transpose()?))
        };
        let depth = (0..).take_while(|i| t.contains_key(&format!("pretrained.blocks.{i}.norm1.weight"))).count();
        if t.contains_key("pretrained.blocks.0.mlp.w12.weight") {
            bail!("{}: the Giant model's SwiGLU blocks are not ported", path.display());
        }
        let layers = match depth {
            12 => vec![2, 5, 8, 11],
            24 => vec![4, 11, 17, 23],
            n => bail!("{}: a DINOv2 of {n} blocks is not one Depth Anything V2 uses", path.display()),
        };
        let mut blocks = vec![];
        for i in 0..depth {
            let b = format!("pretrained.blocks.{i}");
            blocks.push(Block {
                norm1: pair(&format!("{b}.norm1"))?,
                qkv: pair(&format!("{b}.attn.qkv"))?,
                proj: pair(&format!("{b}.attn.proj"))?,
                ls1: get(&format!("{b}.ls1.gamma"))?,
                norm2: pair(&format!("{b}.norm2"))?,
                fc1: pair(&format!("{b}.mlp.fc1"))?,
                fc2: pair(&format!("{b}.mlp.fc2"))?,
                ls2: get(&format!("{b}.ls2.gamma"))?,
            });
        }
        let s = "depth_head.scratch";
        let fusion = (1..=4)
            .map(|k| -> Result<Fusion> {
                let f = format!("{s}.refinenet{k}");
                Ok(Fusion {
                    out: conv(&format!("{f}.out_conv"))?,
                    res1: [conv(&format!("{f}.resConfUnit1.conv1"))?, conv(&format!("{f}.resConfUnit1.conv2"))?],
                    res2: [conv(&format!("{f}.resConfUnit2.conv1"))?, conv(&format!("{f}.resConfUnit2.conv2"))?],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DepthAnything {
            patch: conv("pretrained.patch_embed.proj")?,
            cls: get("pretrained.cls_token")?,
            pos: get("pretrained.pos_embed")?,
            blocks,
            norm: pair("pretrained.norm")?,
            layers,
            projects: (0..4).map(|i| conv(&format!("depth_head.projects.{i}"))).collect::<Result<Vec<_>>>()?,
            resize: vec![
                Some(conv("depth_head.resize_layers.0")?),
                Some(conv("depth_head.resize_layers.1")?),
                None,
                Some(conv("depth_head.resize_layers.3")?),
            ],
            rn: (1..=4).map(|k| conv(&format!("{s}.layer{k}_rn"))).collect::<Result<Vec<_>>>()?,
            fusion,
            out1: conv(&format!("{s}.output_conv1"))?,
            out2a: conv(&format!("{s}.output_conv2.0"))?,
            out2b: conv(&format!("{s}.output_conv2.2"))?,
        })
    }

    /// One block: x + ls1 * attn(norm1(x)), then x + ls2 * mlp(norm2(x)); x (N, C)
    fn block(&self, b: &Block, x: &Tensor) -> Result<Tensor> {
        let (n, c) = x.dims2()?;
        let (heads, d) = (c / 64, 64);
        let qkv = linear(&layer_norm(x, &b.norm1)?, &b.qkv)?;
        let flat = qkv.flatten_all()?;
        // q * scale lands in memory as (N, heads, d) (TensorIterator keeps the input's order)
        let q = (qkv.narrow(1, 0, c)? * 0.125)?.contiguous()?.flatten_all()?;
        let k = flat.narrow(0, c, n * 3 * c - c)?;
        let v = flat.narrow(0, 2 * c, n * 3 * c - 2 * c)?;
        // attn = q @ k^T: baddbmm with the result transposed, k^T (lda 3C), q (ldb C)
        let attn = crate::kernels::torch_bgemm_f32(true, false, n, n, d, &k, 3 * c, d, &q, c, d, n, n * n, heads, heads * n * n)?;
        let attn = crate::kernels::torch_softmax_lastdim_f32(&attn.reshape((heads * n, n))?)?.flatten_all()?;
        // out = attn @ v, (heads, N, d)
        let out = crate::kernels::torch_bgemm_f32(false, false, d, n, n, &v, 3 * c, d, &attn, n, n * n, d, n * d, heads, heads * n * d)?;
        let out = out.reshape((heads, n, d))?.transpose(0, 1)?.contiguous()?.reshape((n, c))?;
        let x = (x + linear(&out, &b.proj)?.broadcast_mul(&b.ls1)?)?;
        let h = crate::kernels::torch_gelu_erf_f32(&linear(&layer_norm(&x, &b.norm2)?, &b.fc1)?)?;
        Ok((&x + linear(&h, &b.fc2)?.broadcast_mul(&b.ls2)?)?)
    }

    /// DINOv2's get_intermediate_layers(x, layers, return_class_token=True): each taken block's
    /// output through the final norm, its patch tokens (N - 1, C)
    fn dino(&self, x: &Tensor, dbg: &mut Option<&mut HashMap<String, Tensor>>) -> Result<Vec<Tensor>> {
        let (_, _, hh, ww) = x.dims4()?;
        let (ph, pw) = (hh / 14, ww / 14);
        let p = crate::kernels::cudnn_conv(x, &self.patch.0, &[0, 0], &[14, 14], &[1, 1])?;
        let p = p.broadcast_add(&self.patch.1.as_ref().context("patch bias")?.reshape((1, (), 1, 1))?)?;
        let c = p.dim(1)?;
        let tokens = p.reshape((c, ph * pw))?.t()?;
        if let Some(d) = dbg.as_deref_mut() {
            d.insert("patch_embed".into(), tokens.unsqueeze(0)?.contiguous()?);
        }
        // interpolate_pos_encoding: the 37x37 table, bicubic by (h0 + 0.1) / 37 and (w0 + 0.1) / 37
        let npos = self.pos.dim(1)? - 1;
        let side = (npos as f64).sqrt();
        let s = side as usize;
        let grid = self.pos.narrow(1, 1, npos)?.reshape((s, s, c))?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
        let (sx, sy) = ((ph as f64 + 0.1) / side, (pw as f64 + 0.1) / side);
        let (oh, ow) = ((s as f64 * sx).floor() as usize, (s as f64 * sy).floor() as usize);
        anyhow::ensure!((oh, ow) == (ph, pw), "position embedding resized to {oh}x{ow}, not {ph}x{pw}");
        let grid = crate::kernels::torch_upsample_bicubic2d_f32(&grid, ph, pw, (sx, sy))?;
        let patch_pos = grid.squeeze(0)?.reshape((c, ph * pw))?.t()?;
        let pos = Tensor::cat(&[&self.pos.narrow(1, 0, 1)?.squeeze(0)?, &patch_pos], 0)?;
        let mut x = (Tensor::cat(&[&self.cls.reshape((1, c))?, &tokens], 0)?.contiguous()? + pos)?;
        let mut outs = vec![];
        for (i, b) in self.blocks.iter().enumerate() {
            x = self.block(b, &x)?;
            if i == 0 {
                if let Some(d) = dbg.as_deref_mut() {
                    d.insert("block0".into(), x.unsqueeze(0)?);
                }
            }
            if self.layers.contains(&i) {
                let normed = layer_norm(&x, &self.norm)?;
                if let Some(d) = dbg.as_deref_mut() {
                    let j = outs.len();
                    d.insert(format!("feat{j}"), normed.narrow(0, 1, ph * pw)?.unsqueeze(0)?);
                    d.insert(format!("cls{j}"), normed.narrow(0, 0, 1)?);
                }
                outs.push(normed.narrow(0, 1, ph * pw)?);
            }
        }
        Ok(outs)
    }

    fn residual(&self, x: &Tensor, convs: &[Conv; 2]) -> Result<Tensor> {
        let out = conv2d(&x.relu()?, &convs[0], 1, 1)?;
        let out = conv2d(&out.relu()?, &convs[1], 1, 1)?;
        Ok((out + x)?)
    }

    /// FeatureFusionBlock: the skip through resConfUnit1 added, resConfUnit2, a bilinear resize
    /// (align_corners) to `size` or twice the size, the 1x1 out_conv
    fn fuse(&self, f: &Fusion, x: &Tensor, skip: Option<&Tensor>, size: Option<(usize, usize)>) -> Result<Tensor> {
        let mut out = x.clone();
        if let Some(s) = skip {
            out = (out + self.residual(s, &f.res1)?)?;
        }
        let out = self.residual(&out, &f.res2)?;
        let (_, _, h, w) = out.dims4()?;
        let (oh, ow) = size.unwrap_or((h * 2, w * 2));
        let out = crate::kernels::torch_upsample_bilinear2d_ac_f32(&out.contiguous()?, oh, ow)?;
        conv2d(&out, &f.out, 1, 0)
    }

    /// The network's depth (1, H, W) for its input (1, 3, H, W), max_depth 1
    pub fn net(&self, x: &Tensor, mut dbg: Option<&mut HashMap<String, Tensor>>) -> Result<Tensor> {
        let (_, _, hh, ww) = x.dims4()?;
        let (ph, pw) = (hh / 14, ww / 14);
        let feats = self.dino(x, &mut dbg)?;
        let mut layer = vec![];
        for (i, f) in feats.iter().enumerate() {
            let c = f.dim(1)?;
            // the tokens permuted to (1, C, ph, pw), copied to NCHW memory as torch's convolution does
            let x = f.t()?.reshape((1, c, ph, pw))?.contiguous()?;
            let x = conv2d(&x, &self.projects[i], 1, 0)?;
            if let Some(d) = dbg.as_deref_mut() {
                d.insert(format!("proj{i}"), x.clone());
            }
            let x = match (i, &self.resize[i]) {
                (0, Some(r)) => crate::kernels::cudnn_conv_transpose(&x, &r.0, &[0, 0], &[4, 4], &[1, 1], &[0, 0], 1)?
                    .broadcast_add(&r.1.as_ref().context("bias")?.reshape((1, (), 1, 1))?)?,
                (1, Some(r)) => crate::kernels::cudnn_conv_transpose(&x, &r.0, &[0, 0], &[2, 2], &[1, 1], &[0, 0], 1)?
                    .broadcast_add(&r.1.as_ref().context("bias")?.reshape((1, (), 1, 1))?)?,
                (3, Some(r)) => conv2d(&x, r, 2, 1)?,
                _ => x,
            };
            if let Some(d) = dbg.as_deref_mut() {
                d.insert(format!("resize{i}"), x.clone());
            }
            layer.push(x);
        }
        let rn: Vec<Tensor> = layer.iter().zip(&self.rn).map(|(x, c)| conv2d(x, c, 1, 1)).collect::<Result<_>>()?;
        let size = |t: &Tensor| -> Result<(usize, usize)> { Ok((t.dim(2)?, t.dim(3)?)) };
        let p4 = self.fuse(&self.fusion[3], &rn[3], None, Some(size(&rn[2])?))?;
        let p3 = self.fuse(&self.fusion[2], &p4, Some(&rn[2]), Some(size(&rn[1])?))?;
        let p2 = self.fuse(&self.fusion[1], &p3, Some(&rn[1]), Some(size(&rn[0])?))?;
        let p1 = self.fuse(&self.fusion[0], &p2, Some(&rn[0]), None)?;
        if let Some(d) = dbg.as_deref_mut() {
            for (k, p) in [("refine4", &p4), ("refine3", &p3), ("refine2", &p2), ("refine1", &p1)] {
                d.insert(k.into(), p.clone());
            }
        }
        let out = conv2d(&p1, &self.out1, 1, 1)?;
        if let Some(d) = dbg {
            d.insert("output_conv1".into(), out.clone());
        }
        let out = crate::kernels::torch_upsample_bilinear2d_ac_f32(&out.contiguous()?, ph * 14, pw * 14)?;
        let out = conv2d(&out, &self.out2a, 1, 1)?.relu()?;
        let out = conv2d(&out, &self.out2b, 1, 0)?.relu()?;

        Ok(out.squeeze(1)?)
    }
}

/// Resize(lower_bound, keep_aspect_ratio, ensure_multiple_of=14) to `size`: (width, height)
fn net_size(w: usize, h: usize, size: usize) -> (usize, usize) {
    let (sh, sw) = (size as f64 / h as f64, size as f64 / w as f64);
    let s = if sw > sh { sw } else { sh };
    let constrain = |x: f64| -> usize {
        let y = ((x / 14.0).round_ties_even() * 14.0) as usize;
        if y < size {
            ((x / 14.0).ceil() * 14.0) as usize
        } else {
            y
        }
    };
    (constrain(s * w as f64), constrain(s * h as f64))
}

/// OpenCV's cubic coefficients (resize.cpp interpolateCubic, A = -0.75, in float) and source
/// offsets for each destination index, the position (d + 0.5) * scale - 0.5 in double, as float
fn cv_cubic_tables(src: usize, dst: usize) -> Vec<(isize, [f32; 4])> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            let x = f - s;
            let a = -0.75f32;
            let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
            let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
            let c3 = 1.0 - c0 - c1 - c2;
            (s as isize, [c0, c1, c2, c3])
        })
        .collect()
}

/// cv2.resize(INTER_CUBIC) of a CV_64F picture (H, W, C interleaved): a horizontal pass then a
/// vertical one, each sum left to right in double, the taps clamped at the borders
fn resize_f64_cubic(src: &[f64], h: usize, w: usize, c: usize, oh: usize, ow: usize) -> Vec<f64> {
    let (xt, yt) = (cv_cubic_tables(w, ow), cv_cubic_tables(h, oh));
    let clampi = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    let mut tmp = vec![0f64; h * ow * c];
    for r in 0..h {
        for (dx, (s, a)) in xt.iter().enumerate() {
            for ch in 0..c {
                let mut v = 0f64;
                for (j, aj) in a.iter().enumerate() {
                    v += src[(r * w + clampi(s - 1 + j as isize, w)) * c + ch] * *aj as f64;
                }
                tmp[(r * ow + dx) * c + ch] = v;
            }
        }
    }
    let mut out = vec![0f64; oh * ow * c];
    for (dy, (s, b)) in yt.iter().enumerate() {
        let rows: Vec<usize> = (0..4).map(|k| clampi(s - 1 + k as isize, h)).collect();
        for i in 0..ow * c {
            let mut v = 0f64;
            for k in 0..4 {
                v += tmp[rows[k] * ow * c + i] * b[k] as f64;
            }
            out[dy * ow * c + i] = v;
        }
    }
    out
}

extern "C" {
    fn rustyui_ipp_resize_cubic_8u(
        src: *const u8,
        src_step: i64,
        sw: i32,
        sh: i32,
        dst: *mut u8,
        dst_step: i64,
        dw: i32,
        dh: i32,
        channels: i32,
        row0: i32,
        row1: i32,
    ) -> i32;
    fn rustyui_ipp_version() -> *const std::os::raw::c_char;
}

/// IPP's name, version and build date, as cv2.ipp.getIppVersion() prints them
pub fn ipp_version() -> String {
    // SAFETY: a static NUL-terminated string the shim keeps
    unsafe { std::ffi::CStr::from_ptr(rustyui_ipp_version()) }.to_string_lossy().into_owned()
}

/// cv2.resize(INTER_CUBIC) of a uint8 picture (H, W, C interleaved, C 1, 3 or 4): OpenCV 5.0.0
/// hands it to Intel IPP's resize (ipp_resize in imgproc/src/resize.cpp), which rustyui links as
/// OpenCV bundles it and calls with OpenCV's parameters, the whole picture as one tile
pub fn resize_u8_cubic(src: &[u8], h: usize, w: usize, c: usize, oh: usize, ow: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        matches!(c, 1 | 3 | 4) && src.len() == h * w * c,
        "resize_u8_cubic: a {w}x{h} picture of {c} channels in {} values",
        src.len()
    );
    let mut out = vec![0u8; oh * ow * c];
    // SAFETY: both buffers hold the sizes and steps passed
    let st = unsafe {
        rustyui_ipp_resize_cubic_8u(
            src.as_ptr(),
            (w * c) as i64,
            w as i32,
            h as i32,
            out.as_mut_ptr(),
            (ow * c) as i64,
            ow as i32,
            oh as i32,
            c as i32,
            0,
            oh as i32,
        )
    };
    anyhow::ensure!(st >= 0, "IPP's resize refused a {w}x{h} picture to {ow}x{oh} (status {st})");
    Ok(out)
}

/// debug-ipp-resize REF (refs/ipp_resize_ref.py): the bicubic resize against cv2's on the
/// reference's pictures
pub fn debug_ipp_resize(refs: &Path) -> Result<()> {
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let theirs = String::from_utf8(st.load("version", &Device::Cpu)?.to_vec1::<u8>()?)?;
    println!("rustyui's IPP: {}\ncv2's IPP:     {theirs}", ipp_version());
    let mut bad = 0;
    for i in 0.. {
        let Ok(input) = st.load(&format!("case{i}.in"), &Device::Cpu) else { break };
        let want = st.load(&format!("case{i}.out"), &Device::Cpu)?;
        let (h, w, c) = input.dims3()?;
        let (oh, ow, _) = want.dims3()?;
        let ours = resize_u8_cubic(&input.flatten_all()?.to_vec1::<u8>()?, h, w, c, oh, ow)?;
        let want = want.flatten_all()?.to_vec1::<u8>()?;
        let n = ours.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("{w}x{h}x{c} -> {ow}x{oh}: {n} of {} differ{}", want.len(), if n == 0 { "  exact" } else { "" });
        bad += (n > 0) as usize;
    }
    anyhow::ensure!(bad == 0, "{bad} cases differ");
    Ok(())
}

/// computeResizeAreaTab for one channel: each destination index's source indices and weights,
/// the cell's bounds in double, the weights rounded to float
fn cv_area_tab(src: usize, dst: usize, scale: f64) -> Vec<Vec<(usize, f32)>> {
    (0..dst)
        .map(|d| {
            let fsx1 = d as f64 * scale;
            let fsx2 = fsx1 + scale;
            let cell = scale.min(src as f64 - fsx1);
            let sx2 = (fsx2.floor() as i64).min(src as i64 - 1);
            let sx1 = (fsx1.ceil() as i64).min(sx2);
            let mut e = vec![];
            if sx1 as f64 - fsx1 > 1e-3 {
                e.push(((sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
            }
            for sx in sx1..sx2 {
                e.push((sx as usize, (1.0 / cell) as f32));
            }
            if fsx2 - sx2 as f64 > 1e-3 {
                e.push((sx2 as usize, ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32));
            }
            e
        })
        .collect()
}

/// cv2.resize(INTER_AREA) of a uint8 picture (H, W, C interleaved) to a smaller size, as OpenCV
/// runs it itself (area stays out of IPP while OpenCV's inexact IPP mode is off). A whole-number
/// scale sums each block in integers: (s + 2) >> 2 for 2x2, else s * (1 / area) in float. Any
/// other weighs each source pixel by its share of the cell (computeResizeAreaTab): each source
/// row along x, then the rows, every product and sum in float, rounded half to even.
fn resize_u8_area(src: &[u8], h: usize, w: usize, c: usize, oh: usize, ow: usize) -> Vec<u8> {
    let (sx, sy) = (1.0 / (ow as f64 / w as f64), 1.0 / (oh as f64 / h as f64));
    let (ix, iy) = (sx.round_ties_even(), sy.round_ties_even());
    let round = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
    let mut out = vec![0u8; oh * ow * c];
    if (sx - ix).abs() < f64::EPSILON && (sy - iy).abs() < f64::EPSILON {
        let (ix, iy) = (ix as usize, iy as usize);
        let scale = 1.0f32 / (ix * iy) as f32;
        for dy in 0..oh {
            for dx in 0..ow {
                for ch in 0..c {
                    let mut s = 0u32;
                    for a in 0..iy {
                        for b in 0..ix {
                            s += src[((dy * iy + a) * w + dx * ix + b) * c + ch] as u32;
                        }
                    }
                    out[(dy * ow + dx) * c + ch] =
                        if ix == 2 && iy == 2 && matches!(c, 1 | 3 | 4) { ((s + 2) >> 2) as u8 } else { round(s as f32 * scale) };
                }
            }
        }
        return out;
    }
    let (xt, yt) = (cv_area_tab(w, ow, sx), cv_area_tab(h, oh, sy));
    let mut rows = vec![0f32; h * ow * c];
    for r in 0..h {
        for (dx, e) in xt.iter().enumerate() {
            for ch in 0..c {
                let mut v = 0f32;
                for &(s, a) in e {
                    v += src[(r * w + s) * c + ch] as f32 * a;
                }
                rows[(r * ow + dx) * c + ch] = v;
            }
        }
    }
    for (dy, e) in yt.iter().enumerate() {
        for i in 0..ow * c {
            let mut acc = 0f32;
            for &(s, b) in e {
                acc += b * rows[s * ow * c + i];
            }
            out[dy * ow * c + i] = round(acc);
        }
    }
    out
}

/// The node for one picture (3, H, W) f32 in [0, 1]: its IMAGE (H', W', 3) as f32 values, the
/// shorter side `resolution` (resize_image_with_pad's), and the stages for debug-depth
pub fn preprocess(
    model: &DepthAnything,
    img: &Tensor,
    resolution: usize,
    dev: &Device,
    mut dbg: Option<&mut HashMap<String, Tensor>>,
) -> Result<(Vec<f32>, usize, usize)> {
    let (c, h, w) = img.dims3()?;
    anyhow::ensure!(c == 3, "DepthAnythingV2Preprocessor: a picture of {c} channels");
    // common_annotator_call: np.asarray(image * 255., dtype=uint8), HWC
    let px = img.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut u8s = vec![0u8; h * w * 3];
    for ch in 0..3 {
        for i in 0..h * w {
            u8s[i * 3 + ch] = (px[ch * h * w + i] * 255.0) as u8;
        }
    }
    // image2tensor: RGB / 255.0 in double, resized, normalised by ImageNet's mean and std, to f32
    let (nw, nh) = net_size(w, h, 518);
    let rgb: Vec<f64> = u8s.iter().map(|&v| v as f64 / 255.0).collect();
    let r = resize_f64_cubic(&rgb, h, w, 3, nh, nw);
    let (mean, std) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    let mut chw = vec![0f32; 3 * nh * nw];
    for i in 0..nh * nw {
        for ch in 0..3 {
            chw[ch * nh * nw + i] = ((r[i * 3 + ch] - mean[ch]) / std[ch]) as f32;
        }
    }
    let x = Tensor::from_vec(chw, (1, 3, nh, nw), dev)?;
    if let Some(d) = dbg.as_deref_mut() {
        d.insert("net_in".into(), x.clone());
    }
    let depth = model.net(&x, dbg.as_deref_mut())?;
    if let Some(d) = dbg.as_deref_mut() {
        d.insert("depth_net".into(), depth.clone());
    }
    // infer_image: back to the picture's size, bilinear with align_corners
    let depth =
        crate::kernels::torch_upsample_bilinear2d_ac_f32(&depth.unsqueeze(0)?.contiguous()?, h, w)?.flatten_all()?.to_vec1::<f32>()?;
    if let Some(d) = dbg.as_deref_mut() {
        d.insert("depth_hw".into(), Tensor::from_vec(depth.clone(), (h, w), &Device::Cpu)?);
    }
    // (depth - min) / (max - min) * 255.0 in f32, to uint8, three channels
    let (mn, mx) = depth.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let span = mx - mn;
    let mut map = vec![0u8; h * w * 3];
    for (i, &v) in depth.iter().enumerate() {
        let q = ((v - mn) / span * 255.0) as u8;
        map[i * 3..i * 3 + 3].fill(q);
    }
    if let Some(d) = dbg {
        d.insert("u8_map".into(), Tensor::from_vec(map.iter().map(|&v| v as f32).collect::<Vec<_>>(), (h, w, 3), &Device::Cpu)?);
    }
    // resize_image_with_pad: the shorter side to the resolution (cubic up, area down; the same
    // size is a copy), then the padding taken off again
    let k = resolution as f64 / h.min(w) as f64;
    let (th, tw) = ((h as f64 * k).round_ties_even() as usize, (w as f64 * k).round_ties_even() as usize);
    let out = if (th, tw) == (h, w) {
        map
    } else if k > 1.0 {
        resize_u8_cubic(&map, h, w, 3, th, tw)?
    } else {
        resize_u8_area(&map, h, w, 3, th, tw)
    };

    Ok((out.iter().map(|&v| v as f32 / 255.0).collect(), th, tw))
}

/// debug-depth PICTURE CKPT RESOLUTION REF (refs/depth_anything_ref.py): every stage
/// against the node's
pub fn debug_depth(pic: &Path, ckpt: &Path, resolution: usize, refs: &Path, dev: &Device) -> Result<()> {
    let model = DepthAnything::load(ckpt, dev)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    // the picture as LoadImage gives it
    let (px, w, h) = crate::ltx_video::load_image(pic)?;
    let img = Tensor::from_vec(px, (h, w, 3), &Device::Cpu)?.permute((2, 0, 1))?.contiguous()?;
    let mut d = HashMap::new();
    let (out, th, tw) = preprocess(&model, &img, resolution, dev, Some(&mut d))?;
    d.insert("node_out".into(), Tensor::from_vec(out, (th, tw, 3), &Device::Cpu)?);
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

/// Where comfyui_controlnet_aux keeps a Depth Anything V2 checkpoint (its own ckpts folder, by the
/// model's size), else rustyui's models/depth_anything
pub fn checkpoint(models: &Path, name: &str) -> Result<std::path::PathBuf> {
    let rel = crate::paths::contained(name).with_context(|| format!("DepthAnythingV2Preprocessor: {name:?} is not a file name"))?;
    let size =
        ["vits", "vitb", "vitl", "vitg"].iter().zip(["Small", "Base", "Large", "Giant"]).find(|(k, _)| name.contains(*k)).map(|(_, v)| v);
    let mut tries = vec![];
    if let Some(size) = size {
        tries.push(
            models
                .parent()
                .unwrap_or(models)
                .join(format!("custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-{size}"))
                .join(&rel),
        );
    }
    tries.push(models.join("depth_anything").join(&rel));
    tries.into_iter().find(|p| p.is_file()).with_context(|| {
        format!("DepthAnythingV2Preprocessor: {name} is in neither comfyui_controlnet_aux's ckpts nor models/depth_anything")
    })
}
