//! ESRGAN (RRDBNet, "old" key layout) as spandrel builds it, run the way ComfyUI's
//! ImageUpscaleWithModel does: f32, 512-pixel tiles overlapping by 32, each clamped to [0, 1] as
//! spandrel's call clamps it, then feathered and averaged. The convolutions run on NCHW memory,
//! which is what torch gets from a VAEDecode picture, or channels-last on cuDNN's NHWC engines
//! when the picture is NHWC memory (LoadImage's): torch keeps that layout through every tile, cat,
//! residual add and upsample, so every convolution of the model runs that way (`nhwc`).

use crate::nn::Conv2d;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

struct Rdb {
    convs: Vec<Conv2d>, // conv1..conv5
}

/// one of the model's convolutions, channels-last when `nhwc`
fn run_conv(c: &Conv2d, x: &Tensor, nhwc: bool) -> Result<Tensor> {
    Ok(if nhwc { c.forward_nhwc(x)? } else { c.forward(x)? })
}

impl Rdb {
    fn forward(&self, x: &Tensor, nhwc: bool) -> Result<Tensor> {
        let mut feats = vec![x.clone()];
        for (i, c) in self.convs.iter().enumerate() {
            let inp = Tensor::cat(&feats, 1)?;
            let y = run_conv(c, &inp, nhwc)?;
            if i < 4 {
                feats.push(leaky(&y)?);
            } else {
                // x5 * 0.2 + x
                return Ok((y.affine(0.2, 0.0)? + x)?);
            }
        }
        bail!("RDB without conv5")
    }
}

fn leaky(x: &Tensor) -> Result<Tensor> {
    // LeakyReLU(0.2): max(x, 0.2x) for these values
    Ok(x.maximum(&x.affine(0.2, 0.0)?)?)
}

pub struct Esrgan {
    first: Conv2d,
    blocks: Vec<[Rdb; 3]>,
    trunk: Conv2d,
    up: Vec<Conv2d>,
    hr: Conv2d,
    last: Conv2d,
    pub scale: usize,
}

fn conv(sd: &HashMap<String, Tensor>, p: &str, dev: &Device) -> Result<Conv2d> {
    let w = sd.get(&format!("{p}.weight")).with_context(|| format!("upscaler: no {p}.weight"))?.to_device(dev)?.to_dtype(DType::F32)?;
    let b = sd.get(&format!("{p}.bias")).map(|b| b.to_device(dev)?.to_dtype(DType::F32)).transpose()?;
    let k = w.dim(2)?;
    Ok(Conv2d { w, b, stride: 1, padding: k / 2 })
}

impl Esrgan {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut tensors = candle_core::pickle::read_all(path).with_context(|| format!("read {}", path.display()))?;
        if tensors.iter().any(|(k, _)| k.starts_with("params_ema.")) {
            tensors = tensors.into_iter().filter_map(|(k, v)| k.strip_prefix("params_ema.").map(|s| (s.to_string(), v))).collect();
        }
        let sd: HashMap<String, Tensor> = tensors.into_iter().collect();
        if !sd.contains_key("model.1.sub.0.RDB1.conv1.0.weight") {
            bail!("{}: not an old-layout ESRGAN (RRDBNet) file; only that kind is read so far", path.display());
        }
        let mut blocks = vec![];
        let mut i = 0;
        while sd.contains_key(&format!("model.1.sub.{i}.RDB1.conv1.0.weight")) {
            let rdb = |r: usize| -> Result<Rdb> {
                Ok(Rdb { convs: (1..=5).map(|c| conv(&sd, &format!("model.1.sub.{i}.RDB{r}.conv{c}.0"), dev)).collect::<Result<_>>()? })
            };
            blocks.push([rdb(1)?, rdb(2)?, rdb(3)?]);
            i += 1;
        }
        // after the trunk: upconv blocks (model.3, model.6, ...), then HR conv and the last conv
        let mut idx: Vec<usize> = sd
            .keys()
            .filter_map(|k| k.strip_prefix("model.")?.strip_suffix(".weight")?.parse::<usize>().ok())
            .filter(|&n| n >= 2)
            .collect();
        idx.sort();
        anyhow::ensure!(idx.len() >= 3, "upscaler: unexpected layout");
        let (last_i, hr_i) = (idx[idx.len() - 1], idx[idx.len() - 2]);
        let up: Vec<Conv2d> = idx[..idx.len() - 2].iter().map(|n| conv(&sd, &format!("model.{n}"), dev)).collect::<Result<_>>()?;
        let scale = 1 << up.len();
        Ok(Esrgan {
            first: conv(&sd, "model.0", dev)?,
            trunk: conv(&sd, &format!("model.1.sub.{i}"), dev)?,
            blocks,
            up,
            hr: conv(&sd, &format!("model.{hr_i}"), dev)?,
            last: conv(&sd, &format!("model.{last_i}"), dev)?,
            scale,
        })
    }

    fn forward(&self, x: &Tensor, nhwc: bool) -> Result<Tensor> {
        self.forward_with(x, nhwc, &mut |_, _| Ok(()))
    }

    /// The model as spandrel's ImageModelDescriptor calls it: the output clamped to [0, 1], so in
    /// tiled_scale each tile is clamped before the tiles are blended
    fn call(&self, x: &Tensor, nhwc: bool) -> Result<Tensor> {
        Ok(self.forward(x, nhwc)?.clamp(0f32, 1f32)?)
    }

    /// The model as spandrel's Sequential runs it, `stage` seeing each top-level module's output
    /// under its index there: 0 the first conv, 1 the trunk with its shortcut, then each upsample,
    /// conv and LeakyReLU, the HR conv and its LeakyReLU, and the last conv.
    fn forward_with(&self, x: &Tensor, nhwc: bool, stage: &mut dyn FnMut(usize, &Tensor) -> Result<()>) -> Result<Tensor> {
        let fea = run_conv(&self.first, x, nhwc)?;
        stage(0, &fea)?;
        let mut h = fea.clone();
        for [a, b, c] in &self.blocks {
            let o = c.forward(&b.forward(&a.forward(&h, nhwc)?, nhwc)?, nhwc)?;
            h = (o.affine(0.2, 0.0)? + &h)?;
        }
        let mut h = (run_conv(&self.trunk, &h, nhwc)? + &fea)?;
        drop(fea);
        stage(1, &h)?;
        let mut k = 2;
        for u in &self.up {
            let (_, _, hh, ww) = h.dims4()?;
            let wide = h.upsample_nearest2d(hh * 2, ww * 2)?;
            stage(k, &wide)?;
            let y = run_conv(u, &wide, nhwc)?;
            drop(wide);
            stage(k + 1, &y)?;
            h = leaky(&y)?;
            drop(y);
            stage(k + 2, &h)?;
            k += 3;
        }
        let y = run_conv(&self.hr, &h, nhwc)?;
        stage(k, &y)?;
        let h = leaky(&y)?;
        drop(y);
        stage(k + 1, &h)?;
        let out = run_conv(&self.last, &h, nhwc)?;
        stage(k + 2, &out)?;
        Ok(out)
    }

    /// comfy.utils.tiled_scale with tile 512, overlap 32, then clamp to [0, 1].
    /// image: (1, 3, h, w) f32 in [0, 1]; `nhwc` when ComfyUI holds it as NHWC memory.
    pub fn upscale(&self, image: &Tensor, nhwc: bool) -> Result<Tensor> {
        let (b, c, h, w) = image.dims4()?;
        anyhow::ensure!(b == 1 && c == 3, "upscale: one RGB image at a time");
        let (tile, overlap, s) = (512usize, 32usize, self.scale);
        let dev = image.device();
        if h <= tile && w <= tile {
            return self.call(image, nhwc);
        }
        let positions = |n: usize| -> Vec<usize> {
            if n > tile {
                (0..n - overlap).step_by(tile - overlap).collect()
            } else {
                vec![0]
            }
        };
        let (oh, ow) = (h * s, w * s);
        let mut out = Tensor::zeros((1, 3, oh, ow), DType::F32, dev)?;
        let mut div = Tensor::zeros((1, 1, oh, ow), DType::F32, dev)?;
        for py in positions(h) {
            for px in positions(w) {
                let y0 = py.min(h - overlap);
                let x0 = px.min(w - overlap);
                let th = tile.min(h - y0);
                let tw = tile.min(w - x0);
                let ps = self.call(&image.narrow(2, y0, th)?.narrow(3, x0, tw)?, nhwc)?;
                let (ph, pw) = (th * s, tw * s);
                // feather: ramps of (t+1)/f over the first and last f rows and columns
                let f = overlap * s;
                let ramp = |n: usize| -> Vec<f32> {
                    let mut m = vec![1f32; n];
                    if f < n {
                        for t in 0..f {
                            let a = (t + 1) as f32 / f as f32;
                            m[t] *= a;
                            m[n - 1 - t] *= a;
                        }
                    }
                    m
                };
                let my = Tensor::from_vec(ramp(ph), (1, 1, ph, 1), dev)?;
                let mx = Tensor::from_vec(ramp(pw), (1, 1, 1, pw), dev)?;
                let mask = my.broadcast_mul(&mx)?;
                let (uy, ux) = (y0 * s, x0 * s);
                let ly = ph.min(oh - uy);
                let lx = pw.min(ow - ux);
                let ps = ps.narrow(2, 0, ly)?.narrow(3, 0, lx)?;
                let mask = mask.narrow(2, 0, ly)?.narrow(3, 0, lx)?;
                let add = ps.broadcast_mul(&mask)?;
                let cur = out.narrow(2, uy, ly)?.narrow(3, ux, lx)?;
                out = out.slice_assign(&[0..1, 0..3, uy..uy + ly, ux..ux + lx], &(cur + add)?)?;
                let curd = div.narrow(2, uy, ly)?.narrow(3, ux, lx)?;
                div = div.slice_assign(&[0..1, 0..1, uy..uy + ly, ux..ux + lx], &(curd + mask)?)?;
            }
        }
        Ok(out.broadcast_div(&div)?.clamp(0f32, 1f32)?)
    }
}

/// debug-esrgan MODEL REF (refs/esrgan_ref.py): each tile ImageUpscaleWithModel cuts, the
/// model's output for it against ComfyUI's, then the node's result; in the tiles that hold the
/// pixel the capture names, every stage of the model in its window around that pixel
pub fn debug_esrgan(model: &Path, refs: &Path, dev: &Device) -> Result<()> {
    let up = Esrgan::load(model, dev)?;
    let meta = crate::ltx_upsampler::metadata(refs)?;
    let info: serde_json::Value = serde_json::from_str(meta.get("info").and_then(|v| v.as_str()).context("the capture has no info")?)?;
    let r = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, dev).with_context(|| format!("no {n} in the capture")) };
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
    // the capture's picture memory (RUSTYUI_ESRGAN_NHWC=1 there): NHWC runs every conv channels-last
    let nhwc = info["image_memory"].as_str() == Some("nhwc");
    println!("pixel {}, torch's cuDNN {}, the picture in {} memory", info["pixel"], info["cudnn"], if nhwc { "NHWC" } else { "NCHW" });
    for (i, tile) in info["tiles"].as_array().context("the capture has no tiles")?.iter().enumerate() {
        let wins = tile["windows"].as_object().cloned().unwrap_or_default();
        let mut seen = vec![];
        let out = up
            .forward_with(&get(&format!("tile{i}.in"))?, nhwc, &mut |k, t| {
                if let Some(w) = wins.get(&format!("model.{k}")) {
                    let w: Vec<usize> = serde_json::from_value(w.clone())?;
                    seen.push((k, t.narrow(2, w[0], w[2])?.narrow(3, w[1], w[3])?));
                }
                Ok(())
            })?
            .clamp(0f32, 1f32)?; // as Esrgan::call

        let repeats = if tile["repeats"].as_bool() == Some(true) { "" } else { " (torch's own run did not repeat)" };
        report(&format!("tile {i} at {} {}{repeats}", tile["pos"], tile["size"]), &out, &get(&format!("tile{i}.out"))?)?;
        for (k, w) in seen {
            report(&format!("  model.{k} around the pixel"), &w, &get(&format!("tile{i}.model.{k}"))?)?;
        }
    }
    let image = get("image")?.permute((0, 3, 1, 2))?.contiguous()?;
    report("ImageUpscaleWithModel", &up.upscale(&image, nhwc)?.permute((0, 2, 3, 1))?, &get("result")?)?;
    Ok(())
}
