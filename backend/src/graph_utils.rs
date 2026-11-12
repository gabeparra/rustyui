//! Image and plumbing nodes, each as its source defines it: ComfyUI's GetImageSize
//! (comfy_extras/nodes_images.py), ImageFromBatch (the same file), ImageScaleBy (nodes.py) and the deprecated
//! ResizeImagesByLongerEdge (comfy_extras/nodes_dataset.py), and KJNodes' ImageResizeKJv2
//! (nodes/image_nodes.py), PathchSageAttentionKJ (nodes/model_optimization_nodes.py), PlaySoundKJ
//! and INTConstant (nodes/nodes.py), ComfyUI's LatentUpscale, LatentUpscaleBy, LoadImageMask and
//! SetLatentNoiseMask (nodes.py),
//! PrimitiveStringMultiline (comfy_extras/nodes_primitive.py)
//! and PreviewAny (comfy_extras/nodes_preview_any.py), and ComfyUI-Frame-Interpolation's RIFE VFI
//! (vfi_models/rife, the model in rife.rs). What a node does not support yet stops the graph with its name, and the
//! editor lists only the choices that work.

use super::*;

/// The nodes this file runs
pub const UTIL_NODES: &[&str] = &[
    "GetImageSize",
    "ImageScaleBy",
    "ResizeImagesByLongerEdge",
    "ImageResizeKJv2",
    "PathchSageAttentionKJ",
    "PlaySoundKJ",
    "INTConstant",
    "ImageFromBatch",
    "RIFE VFI",
    "PrimitiveStringMultiline",
    "PreviewAny",
    "LatentUpscale",
    "LatentUpscaleBy",
    "LoadImageMask",
    "SetLatentNoiseMask",
    "Canny",
    "DepthAnythingV2Preprocessor",
    "SAM3Segment",
];

/// Python's round() on a float: halves to even
fn py_round(x: f64) -> usize {
    let r = x.round();
    (if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }) as usize
}

/// An 8-bit PNG as RGBA bytes the way Pillow's convert("RGBA") gives them: grey into R, G and B,
/// alpha 255 where the picture has none, palettes and low bit depths expanded. 16-bit pictures
/// are refused for now.
fn read_rgba(path: &std::path::Path) -> Result<(Vec<u8>, usize, usize)> {
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).with_context(|| path.display().to_string())?));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    anyhow::ensure!(
        info.bit_depth == png::BitDepth::Eight,
        "{}: {:?}-bit pictures are not implemented yet",
        path.display(),
        info.bit_depth
    );
    let px = &buf[..info.buffer_size()];
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        other => bail!("{}: color type {other:?} not handled", path.display()),
    };
    Ok((rgba, info.width as usize, info.height as usize))
}

/// comfy.utils.common_upscale on a LATENT (b, c, h, w) to (b, c, height, width), as ComfyUI runs it
/// on the CPU, where its latents live: "center" crops to the new aspect first; "nearest-exact"
/// picks min(floor((i + 0.5) * scale), n - 1), the product a double rounded to float as torch's
/// floorf takes it, scale in / out in float; "bilinear" is torch's CPU F.interpolate (torchcpu).
/// The other methods (area, bicubic, bislerp) are refused for now.
pub(super) fn latent_upscale(z: &Tensor, width: usize, height: usize, method: &str, crop: &str) -> Result<Tensor> {
    let (b, c, oh, ow) = z.dims4()?;
    anyhow::ensure!(width > 0 && height > 0, "cannot upscale a latent to {width}x{height}");
    let s = match crop {
        "disabled" => z.clone(),
        "center" => {
            let (old_aspect, new_aspect) = (ow as f64 / oh as f64, width as f64 / height as f64);
            let (mut x, mut y) = (0usize, 0usize);
            if old_aspect > new_aspect {
                x = py_round((ow as f64 - ow as f64 * (new_aspect / old_aspect)) / 2.0);
            } else if old_aspect < new_aspect {
                y = py_round((oh as f64 - oh as f64 * (old_aspect / new_aspect)) / 2.0);
            }
            z.narrow(2, y, oh - 2 * y)?.narrow(3, x, ow - 2 * x)?
        }
        other => bail!("LatentUpscale: crop {other} is not one of ComfyUI's"),
    };
    let (_, _, ih, iw) = s.dims4()?;
    let dev = z.device().clone();
    let x: Vec<f32> = s.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()?;
    let out = match method {
        "nearest-exact" => {
            let idx = |n_in: usize, n_out: usize| -> Vec<usize> {
                let scale = n_in as f32 / n_out as f32;
                (0..n_out).map(|i| ((((i as f64 + 0.5) * scale as f64) as f32).floor() as usize).min(n_in - 1)).collect()
            };
            let (ry, rx) = (idx(ih, height), idx(iw, width));
            let mut out = Vec::with_capacity(b * c * height * width);
            for p in 0..b * c {
                let plane = &x[p * ih * iw..(p + 1) * ih * iw];
                for &yy in &ry {
                    out.extend(rx.iter().map(|&xx| plane[yy * iw + xx]));
                }
            }
            out
        }
        "bilinear" => crate::torchcpu::interpolate_linear(&x, &[b, c, ih, iw], &[height, width])?,
        other => bail!("LatentUpscale: upscale_method {other} is not implemented yet (nearest-exact and bilinear are)"),
    };
    Ok(Tensor::from_vec(out, (b, c, height, width), &dev)?)
}

/// comfy.utils.common_upscale with crop "disabled" on an IMAGE (b, 3, h, w) in [0, 1]. lanczos
/// goes through 8 bits and Pillow as ComfyUI's does, so even the same size comes back quantized
/// (Pillow hands back a copy); bilinear is torch's CPU F.interpolate, bit for bit.
pub(super) fn resize_image(img: &Tensor, w: usize, h: usize, method: &str, dev: &Device) -> Result<Tensor> {
    let (b, _, ih, iw) = img.dims4()?;
    if !matches!(method, "lanczos" | "bilinear") {
        bail!("resizing with {method} is not implemented yet (lanczos and bilinear are)");
    }
    anyhow::ensure!(w > 0 && h > 0, "cannot resize to {w}x{h}");
    let data = img.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(b * h * w * 3);
    for i in 0..b {
        let one = &data[i * ih * iw * 3..(i + 1) * ih * iw * 3];
        if method == "bilinear" {
            out.extend(crate::resample::resize_bilinear(one, iw, ih, w, h));
        } else {
            // np.clip(255. * image, 0, 255).astype(np.uint8), then PIL's resize (a copy at the same size)
            let px: Vec<u8> = one.iter().map(|&x| (255.0f32 * x).clamp(0.0, 255.0) as u8).collect();
            let r = if (iw, ih) == (w, h) { px } else { crate::resample::resize_lanczos(&px, iw, ih, w, h) };
            out.extend(r.into_iter().map(|v| v as f32 / 255.0));
        }
    }
    Ok(Tensor::from_vec(out, (b, h, w, 3), dev)?.permute((0, 3, 1, 2))?.contiguous()?)
}

impl Engine {
    /// The nodes of this file, or None for a class it does not run
    pub(super) fn util_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        let dev = self.dev.clone();
        Ok(Some(match class {
            "GetImageSize" => {
                let Val::Image(img) = self.input(inp, "image")? else { bail!("GetImageSize: image is not an image") };
                let (b, _, h, w) = img.dims4()?;
                vec![Val::Int(w as i64), Val::Int(h as i64), Val::Int(b as i64)]
            }
            "ImageScaleBy" => {
                let method = self.text_arg(inp, "upscale_method")?;
                let scale = self.float_arg(inp, "scale_by")?;
                let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageScaleBy: image is not an image") };
                let (_, _, h, w) = img.dims4()?;
                // round(shape * scale_by), Python's round
                let t = Instant::now();
                let out = resize_image(&img, py_round(w as f64 * scale), py_round(h as f64 * scale), &method, &dev)?;
                self.timings.push((format!("{method} scale by {scale}"), t.elapsed().as_secs_f64()));
                vec![Val::Image(out)]
            }
            "ResizeImagesByLongerEdge" => {
                let longer = self.int_arg(inp, "longer_edge")?;
                let Val::Image(img) = self.input(inp, "images")? else { bail!("ResizeImagesByLongerEdge: images is not an image") };
                let (_, _, h, w) = img.dims4()?;
                // the other side is int(side * (longer_edge / longer side)), then Pillow's lanczos
                let (nw, nh) = if w > h {
                    (longer, (h as f64 * (longer as f64 / w as f64)) as usize)
                } else {
                    ((w as f64 * (longer as f64 / h as f64)) as usize, longer)
                };
                vec![Val::Image(resize_image(&img, nw, nh, "lanczos", &dev)?)]
            }
            "ImageResizeKJv2" => self.image_resize_kj(inp)?,
            "PathchSageAttentionKJ" => {
                let mode = self.text_arg(inp, "sage_attention")?;
                let model = self.input(inp, "model")?;
                anyhow::ensure!(
                    mode == "disabled",
                    "PathchSageAttentionKJ: {mode} is not implemented yet (disabled passes the model through)"
                );
                vec![model]
            }
            "PlaySoundKJ" => {
                // an output node that plays a chime in ComfyUI's browser tab once its input is
                // done; rustyui's editor has no player yet, so it only waits for that input
                let out = match inp.get("any_input") {
                    Some(Value::Array(_)) => self.input(inp, "any_input")?,
                    _ => Val::Nothing,
                };
                vec![out]
            }
            "INTConstant" => vec![Val::Int(self.arg(inp, "value")?.as_i64().context("INTConstant: value is not a whole number")?)],
            "PrimitiveStringMultiline" => vec![Val::Str(self.text_arg(inp, "value")?)],
            "LatentUpscale" => {
                // nodes.py: a 0 side follows the other's aspect (at least 64), both at least 64,
                // then common_upscale to width // 8 x height // 8
                // the noise mask stays as it is (the sampler stretches it to the new size)
                let (z, mask) = match self.input(inp, "samples")? {
                    Val::Latent(z) => (z, None),
                    Val::MaskedLatent(z, m) => (z, Some(m)),
                    _ => bail!("LatentUpscale: samples is not an SDXL latent"),
                };
                let wrap = |z: Tensor| match &mask {
                    Some(m) => Val::MaskedLatent(z, m.clone()),
                    None => Val::Latent(z),
                };
                let (method, crop) = (self.text_arg(inp, "upscale_method")?, self.text_arg(inp, "crop")?);
                let (mut w, mut h) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?);
                if w == 0 && h == 0 {
                    vec![wrap(z)]
                } else {
                    let (zh, zw) = (z.dim(2)?, z.dim(3)?);
                    if w == 0 {
                        h = h.max(64);
                        w = py_round(zw as f64 * h as f64 / zh as f64).max(64);
                    } else if h == 0 {
                        w = w.max(64);
                        h = py_round(zh as f64 * w as f64 / zw as f64).max(64);
                    } else {
                        (w, h) = (w.max(64), h.max(64));
                    }
                    vec![wrap(latent_upscale(&z, w / 8, h / 8, &method, &crop)?)]
                }
            }
            "LatentUpscaleBy" => {
                // nodes.py: each side round(side * scale_by), common_upscale without a crop
                let (z, mask) = match self.input(inp, "samples")? {
                    Val::Latent(z) => (z, None),
                    Val::MaskedLatent(z, m) => (z, Some(m)),
                    _ => bail!("LatentUpscaleBy: samples is not an SDXL latent"),
                };
                let method = self.text_arg(inp, "upscale_method")?;
                let by = self.float_arg(inp, "scale_by")?;
                let (zh, zw) = (z.dim(2)?, z.dim(3)?);
                let up = latent_upscale(&z, py_round(zw as f64 * by), py_round(zh as f64 * by), &method, "disabled")?;
                vec![match mask {
                    Some(m) => Val::MaskedLatent(up, m),
                    None => Val::Latent(up),
                }]
            }
            "LoadImageMask" => {
                // nodes.py: the picture as RGBA (Pillow's convert: grey into R, G and B, alpha 255
                // where there is none, palettes expanded), the channel / 255 in float32, alpha
                // inverted; (1, h, w)
                let name = self.text_arg(inp, "image")?;
                let name = name.strip_suffix(" [input]").unwrap_or(&name).to_string();
                let channel = self.text_arg(inp, "channel")?;
                let c = match channel.as_str() {
                    "red" => 0,
                    "green" => 1,
                    "blue" => 2,
                    "alpha" => 3,
                    other => bail!("LoadImageMask: channel {other} is not one of ComfyUI's"),
                };
                let rel = crate::paths::contained(&name).with_context(|| format!("LoadImageMask: {name:?} leaves the input folder"))?;
                let dir = self.models.parent().unwrap_or(&self.models).join("input");
                let path = dir.join(rel);
                anyhow::ensure!(
                    crate::paths::below(&dir, path.parent().unwrap_or(&dir)),
                    "LoadImageMask: {name:?} leads out of {}",
                    dir.display()
                );
                let (rgba, w, h) = read_rgba(&path)?;
                let mask: Vec<f32> = rgba.chunks(4).map(|p| p[c] as f32 / 255.0).map(|v| if c == 3 { 1.0 - v } else { v }).collect();
                vec![Val::Mask(Tensor::from_vec(mask, (1, h, w), &dev)?)]
            }
            "SetLatentNoiseMask" => {
                // the latent's noise_mask, the mask reshaped to (-1, 1, h, w)
                let z = match self.input(inp, "samples")? {
                    Val::Latent(z) | Val::MaskedLatent(z, _) => z,
                    _ => bail!("SetLatentNoiseMask: samples is not an SDXL latent"),
                };
                let Val::Mask(m) = self.input(inp, "mask")? else { bail!("SetLatentNoiseMask: mask is not a mask") };
                let (mh, mw) = (m.dim(m.rank() - 2)?, m.dim(m.rank() - 1)?);
                vec![Val::MaskedLatent(z, m.reshape(((), 1, mh, mw))?)]
            }
            "DepthAnythingV2Preprocessor" => {
                // comfyui_controlnet_aux: the checkpoint loaded for the call (the node loads and
                // drops it each time), each picture through Depth Anything V2; resolution as
                // common_annotator_call reads it (a whole number from 64, else 512)
                let Val::Image(img) = self.input(inp, "image")? else { bail!("DepthAnythingV2Preprocessor: image is not an image") };
                let name =
                    if inp.contains_key("ckpt_name") { self.text_arg(inp, "ckpt_name")? } else { "depth_anything_v2_vitl.pth".into() };
                let res = if inp.contains_key("resolution") { self.arg(inp, "resolution")?.as_i64() } else { None };
                let res = res.filter(|&r| r >= 64).unwrap_or(512) as usize;
                let path = crate::depth_anything::checkpoint(&self.models, &name)?;
                let model = crate::depth_anything::DepthAnything::load(&path, &dev)?;
                let mut outs = vec![];
                for i in 0..img.dim(0)? {
                    let (v, th, tw) = crate::depth_anything::preprocess(&model, &img.get(i)?, res, &dev, None)?;
                    outs.push(Tensor::from_vec(v, (th, tw, 3), &dev)?.permute((2, 0, 1))?.contiguous()?);
                }
                vec![Val::Image(Tensor::stack(&outs, 0)?)]
            }
            "SAM3Segment" => {
                // ComfyUI-RMBG: SAM 3 from models/sam3/sam3.pt, kept between graphs as the node's
                // processor cache keeps it (unload_model drops it after the call), each picture
                // segmented on its own and the outputs joined
                let Val::Image(img) = self.input(inp, "image")? else { bail!("SAM3Segment: image is not an image") };
                anyhow::ensure!(img.dim(1)? == 3, "SAM3Segment: a picture of {} channels (the node takes RGB)", img.dim(1)?);
                let get = |e: &mut Engine, n: &str| -> Result<Option<Value>> {
                    if inp.contains_key(n) {
                        Ok(Some(e.arg(inp, n)?))
                    } else {
                        Ok(None)
                    }
                };
                let int = |v: Option<Value>| v.and_then(|v| v.as_i64()).unwrap_or(0);
                let text = |v: Option<Value>| v.and_then(|v| v.as_str().map(str::to_string));
                let o = crate::sam3::Opts {
                    prompt: self.text_arg(inp, "prompt")?,
                    merged: self.text_arg(inp, "output_mode")? == "Merged",
                    confidence: self.float_arg(inp, "confidence_threshold")?,
                    max_segments: int(get(self, "max_segments")?).max(0) as usize,
                    segment_pick: int(get(self, "segment_pick")?).max(0) as usize,
                    mask_blur: int(get(self, "mask_blur")?).max(0) as usize,
                    mask_offset: int(get(self, "mask_offset")?),
                    invert: get(self, "invert_output")?.and_then(|v| v.as_bool()).unwrap_or(false),
                    alpha: text(get(self, "background")?).is_none_or(|b| b == "Alpha"),
                };
                if text(get(self, "device")?).is_some_and(|d| d == "CPU") {
                    bail!("SAM3Segment: device \"CPU\" is not ported (Auto and GPU run on the card)");
                }
                let unload = get(self, "unload_model")?.and_then(|v| v.as_bool()).unwrap_or(false);
                if self.sam3.is_none() {
                    let t = Instant::now();
                    self.sam3 = Some(Rc::new(crate::sam3::Sam3::load(&crate::sam3::checkpoint(&self.models)?, &dev)?));
                    self.timings.push(("load sam3".into(), t.elapsed().as_secs_f64()));
                }
                let model = self.sam3.clone().context("SAM3Segment: no model")?;
                let t = Instant::now();
                let (mut images, mut masks, mut mask_images) = (vec![], vec![], vec![]);
                for i in 0..img.dim(0)? {
                    let (a, m, mi) = crate::sam3::segment(&model, &img.get(i)?, &o)?;
                    images.push(a);
                    masks.push(m);
                    mask_images.push(mi);
                }
                self.timings.push(("sam3 segment".into(), t.elapsed().as_secs_f64()));
                if unload {
                    self.sam3 = None;
                }
                vec![Val::Image(Tensor::cat(&images, 0)?), Val::Mask(Tensor::cat(&masks, 0)?), Val::Image(Tensor::cat(&mask_images, 0)?)]
            }
            "Canny" => {
                // comfy_extras/nodes_canny.py: kornia's canny on the picture's first three channels
                let Val::Image(img) = self.input(inp, "image")? else { bail!("Canny: image is not an image") };
                let (lo, hi) = (self.float_arg(inp, "low_threshold")?, self.float_arg(inp, "high_threshold")?);
                anyhow::ensure!(lo <= hi && lo > 0.0 && hi < 1.0, "Canny: thresholds {lo} and {hi}: kornia takes 0 < low <= high < 1");
                vec![Val::Image(crate::canny::canny(&img.narrow(1, 0, 3)?, lo, hi, &dev)?)]
            }
            "PreviewAny" => {
                // an output node that shows its input as text in ComfyUI's editor; rustyui's
                // results carry no text yet, so it only runs that input
                let out = match inp.get("source") {
                    Some(Value::Array(_)) => self.input(inp, "source")?,
                    _ => Val::Nothing,
                };
                vec![out]
            }
            "ImageFromBatch" => {
                // a negative index counts from the end, then both are clamped to the batch
                let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageFromBatch: image is not an image") };
                let n = img.dim(0)? as i64;
                let mut index = self.arg(inp, "batch_index")?.as_i64().context("ImageFromBatch: batch_index is not a whole number")?;
                let length = self.arg(inp, "length")?.as_i64().context("ImageFromBatch: length is not a whole number")?;
                if index < 0 {
                    index += n;
                }
                let index = index.min(n - 1).max(0);
                let length = length.min(n - index).max(0);
                vec![Val::Image(img.narrow(0, index as usize, length as usize)?.contiguous()?)]
            }
            "RIFE VFI" => {
                // RIFE 4.7 checkpoints in f32, one interpolation a call (batch_size 1), uncompiled;
                // fast_mode and ensemble change nothing for 4.7 as the node passes them (rife.rs)
                let ckpt = self.text_arg(inp, "ckpt_name")?;
                anyhow::ensure!(
                    matches!(ckpt.as_str(), "rife47.pth" | "rife49.pth"),
                    "RIFE VFI: {ckpt} is not implemented yet (rife47.pth and rife49.pth are)"
                );
                let Val::Image(frames) = self.input(inp, "frames")? else { bail!("RIFE VFI: frames is not an image") };
                let multiplier = self.int_arg(inp, "multiplier")?.max(1);
                let scale = self.float_arg(inp, "scale_factor")?;
                anyhow::ensure!([0.25, 0.5, 1.0, 2.0, 4.0].contains(&scale), "RIFE VFI: scale_factor {scale} is not one of the node's");
                if inp.contains_key("dtype") {
                    let dtype = self.text_arg(inp, "dtype")?;
                    anyhow::ensure!(dtype == "float32", "RIFE VFI: dtype {dtype} is not implemented yet (float32 is)");
                }
                if inp.contains_key("torch_compile") {
                    anyhow::ensure!(
                        !self.flag_arg(inp, "torch_compile")?,
                        "RIFE VFI: torch_compile is not implemented (its compiled kernels round otherwise); turn it off"
                    );
                }
                if inp.contains_key("batch_size") {
                    let b = self.int_arg(inp, "batch_size")?;
                    anyhow::ensure!(b == 1, "RIFE VFI: batch_size {b} is not implemented yet (1 is)");
                }
                anyhow::ensure!(
                    !matches!(inp.get("optional_interpolation_states"), Some(Value::Array(_))),
                    "RIFE VFI: optional_interpolation_states is not implemented yet; leave it unconnected"
                );
                // vfi_utils: the checkpoint in the node's own ckpts/rife folder (config.yaml's ckpts_path)
                let root = self.models.parent().unwrap_or(&self.models).to_path_buf();
                let path = root
                    .join("custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife")
                    .join(crate::paths::contained(&ckpt).context("RIFE VFI: ckpt_name")?);
                anyhow::ensure!(path.is_file(), "RIFE VFI: {} is not there (rustyui does not download it)", path.display());
                (self.clip, self.unet, self.ltx_te) = (None, None, None);
                (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
                // the encoder runs channels-last when ComfyUI's frames are NHWC memory
                let nhwc = self.image_nhwc(inp, "frames")?;
                let t0 = Instant::now();
                let model = crate::rife::Rife::load(&path, &dev)?;
                let out = crate::rife::interpolate(&model, &frames, multiplier, scale, nhwc, |i, n| {
                    let _ = crate::progress::step(i, n);
                })?;
                self.dev.synchronize()?;
                self.timings.push((format!("RIFE VFI ({} frames from {})", out.dim(0)?, frames.dim(0)?), t0.elapsed().as_secs_f64()));
                vec![Val::Image(out)]
            }
            _ => return Ok(None),
        }))
    }

    /// KJNodes' ImageResizeKJv2 (resize()), for keep_proportion stretch, resize, crop and
    /// total_pixels with lanczos or bilinear on the CPU, no mask. Its outputs: the image, its
    /// width and height, and ComfyUI's empty 64x64 mask.
    fn image_resize_kj(&mut self, inp: &serde_json::Map<String, Value>) -> Result<Vec<Val>> {
        let (mut width, mut height) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?);
        let method = self.text_arg(inp, "upscale_method")?;
        let keep = self.text_arg(inp, "keep_proportion")?;
        let crop_position = self.text_arg(inp, "crop_position")?;
        let divisible_by = self.int_arg(inp, "divisible_by")?;
        if inp.get("device").and_then(Value::as_str).is_some_and(|d| d != "cpu") {
            bail!("ImageResizeKJv2: device gpu is not implemented (cpu is)");
        }
        if matches!(inp.get("mask"), Some(Value::Array(_))) {
            bail!("ImageResizeKJv2: a mask input is not implemented yet");
        }
        if !matches!(keep.as_str(), "stretch" | "resize" | "crop" | "total_pixels") {
            bail!("ImageResizeKJv2: keep_proportion {keep} is not implemented yet (stretch, resize, crop and total_pixels are)");
        }
        let Val::Image(img) = self.input(inp, "image")? else { bail!("ImageResizeKJv2: image is not an image") };
        let (_, _, ih, iw) = img.dims4()?;
        let (fw, fh) = (iw as f64, ih as f64);
        if keep == "resize" || keep == "total_pixels" {
            let (nw, nh) = if keep == "total_pixels" {
                let total = (width * height) as f64;
                let aspect = fw / fh;
                ((total * aspect).sqrt() as usize, (total / aspect).sqrt() as usize)
            } else if width == 0 && height == 0 {
                (iw, ih)
            } else if width == 0 {
                (py_round(fw * (height as f64 / fh)), height)
            } else if height == 0 {
                (width, py_round(fh * (width as f64 / fw)))
            } else {
                let ratio = (width as f64 / fw).min(height as f64 / fh);
                (py_round(fw * ratio), py_round(fh * ratio))
            };
            (width, height) = (nw, nh);
        } else {
            if width == 0 {
                width = iw;
            }
            if height == 0 {
                height = ih;
            }
        }
        if divisible_by > 1 {
            width -= width % divisible_by;
            height -= height % divisible_by;
        }
        // crop to the target's aspect first, where crop_position says
        let mut img = img;
        if keep == "crop" {
            let (old_aspect, new_aspect) = (fw / fh, width as f64 / height as f64);
            let (cw, ch) = if old_aspect > new_aspect { (py_round(fh * new_aspect), ih) } else { (iw, py_round(fw / new_aspect)) };
            let (x, y) = match crop_position.as_str() {
                "top" => ((iw - cw) / 2, 0),
                "bottom" => ((iw - cw) / 2, ih - ch),
                "left" => (0, (ih - ch) / 2),
                "right" => (iw - cw, (ih - ch) / 2),
                _ => ((iw - cw) / 2, (ih - ch) / 2),
            };
            img = img.narrow(3, x, cw)?.narrow(2, y, ch)?.contiguous()?;
        }
        let t = Instant::now();
        let out = resize_image(&img, width, height, &method, &self.dev)?;
        self.timings.push((format!("ImageResizeKJv2 {keep} {method} to {width}x{height}"), t.elapsed().as_secs_f64()));
        let mask = Tensor::zeros((1, 64, 64), DType::F32, &self.dev)?;
        Ok(vec![Val::Image(out), Val::Int(width as i64), Val::Int(height as i64), Val::Mask(mask)])
    }
}
