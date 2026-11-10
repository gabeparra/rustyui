//! Text to video and image to video with LTX-2.3, the base pass of a typical image-to-video graph
//! and nothing more yet: the prompt through Gemma 3, the projection and the two connectors, the
//! start image as LTXVImgToVideoInplace puts it in the first latent frame, LTXVScheduler's sigmas,
//! euler on the distilled DiT at CFG 1 (so no negative), the video VAE decoder, and an MP4 from
//! ffmpeg. Not yet: LTXVAddGuide, the LoRAs, the x2 upsamplers, the audio decoder; the clip is
//! silent. The DiT and the sampler match ComfyUI's numbers exactly, the VAE to rounding (ltx_vae.rs).

use crate::{gemma, image, kernels, ltx, ltx_vae, resample};
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct T2v {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: f64,
    pub steps: usize,
    pub seed: u64,
    pub out: PathBuf,
}

/// Python's round(): halves to even (round(8.5) is 8)
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// LTXVEmptyLatentAudio's length: frames / fps seconds at 25 latents a second
/// (16 kHz, a 160-sample mel hop, 4x down)
pub fn audio_latents(frames: usize, fps: f64) -> usize {
    round_half_even(frames as f64 / fps * 25.0) as usize
}

/// CLIPTextEncode with LTXAVTextEncoderLoader's CLIP: the prompt through Gemma 3's 49 hidden states
/// and the text projection, (1, L, 6144) bf16, as the DiT receives it before its connectors (bf16
/// values held in f32 until then; the connectors take them in bf16, as ComfyUI hands them over)
pub fn encode_prompt(g: &gemma::Gemma, proj: &MmapedSafetensors, prompt: &str, dev: &Device) -> Result<Tensor> {
    let (ids, n) = g.tok.encode_padded(prompt)?;
    let hidden = g.hidden_states(&ids, n, dev)?;
    let projected = gemma::ltx_project(&hidden, proj, dev)?;
    Ok(if projected.rank() == 2 { projected.unsqueeze(0)? } else { projected }.to_dtype(DType::BF16)?)
}

/// LoadImage's IMAGE for one PNG: (H, W, 3) values, each byte / 255 in f32, and its width and height
pub fn load_image(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let d = image::read_png(path)?;
    Ok((d.rgb.iter().map(|&v| v as f32 / 255.0).collect(), d.width as usize, d.height as usize))
}

/// comfy.utils.common_upscale(.., "bilinear", "center"): the middle of the picture at the target's
/// aspect (Python's arithmetic and round), then F.interpolate bilinear. (H, W, 3) in and out.
pub fn upscale_center_bilinear(px: &[f32], iw: usize, ih: usize, w: usize, h: usize) -> Vec<f32> {
    let (old_aspect, new_aspect) = (iw as f64 / ih as f64, w as f64 / h as f64);
    let (mut x, mut y) = (0usize, 0usize);
    if old_aspect > new_aspect {
        x = round_half_even((iw as f64 - iw as f64 * (new_aspect / old_aspect)) / 2.0) as usize;
    } else if old_aspect < new_aspect {
        y = round_half_even((ih as f64 - ih as f64 * (old_aspect / new_aspect)) / 2.0) as usize;
    }
    let (cw, ch) = (iw - 2 * x, ih - 2 * y);
    let mut crop = Vec::with_capacity(cw * ch * 3);
    for r in y..y + ch {
        crop.extend_from_slice(&px[(r * iw + x) * 3..(r * iw + x + cw) * 3]);
    }
    resample::resize_bilinear(&crop, cw, ch, w, h)
}

/// LTXVImgToVideoInplace: the picture, resized to the latent's size in pixels when it differs, as
/// (1, H, W, 3) frames for the encoder. ComfyUI's IMAGE as the encoder sees it.
pub fn start_pixels(px: &[f32], iw: usize, ih: usize, latent_w: usize, latent_h: usize, dev: &Device) -> Result<Tensor> {
    let (w, h) = (latent_w * 32, latent_h * 32);
    let pixels = if iw != w || ih != h { upscale_center_bilinear(px, iw, ih, w, h) } else { px.to_vec() };
    Ok(Tensor::from_vec(pixels, (1, h, w, 3), dev)?)
}

/// LTXVImgToVideoInplace's result from the encoded picture `t` (1, 128, n, h, w): the latent with
/// its first n frames replaced, and its noise mask, ones a frame (1, 1, F, 1, 1) with those frames
/// at 1 - strength, as an `Inpaint` with the audio's mask of ones like its samples that
/// LTXVConcatAVLatent adds, each made its latent's shape as the sampler makes it (reshape_mask).
pub fn img_to_video_inplace(t: &Tensor, latent: &Tensor, audio: &Tensor, strength: f64) -> Result<ltx::Inpaint> {
    let (_, _, f, _, _) = latent.dims5()?;
    let n = t.dim(2)?;
    let v_latent = Tensor::cat(&[&t.to_dtype(DType::F32)?, &latent.narrow(2, n, f - n)?.to_dtype(DType::F32)?], 2)?;
    let kept = (1.0 - strength) as f32;
    let per_frame: Vec<f32> = (0..f).map(|i| if i < n { kept } else { 1.0 }).collect();
    let v_mask = Tensor::from_vec(per_frame, (1, 1, f, 1, 1), latent.device())?;
    let a_mask = Tensor::ones(audio.dims(), DType::F32, audio.device())?;
    Ok(ltx::Inpaint {
        v_mask: crate::sampling::reshape_mask(&v_mask, latent.dims(), false)?,
        v_latent,
        a_mask: crate::sampling::reshape_mask(&a_mask, audio.dims(), false)?,
        a_latent: audio.to_dtype(DType::F32)?,
    })
}

/// `models`: a ComfyUI models folder; the files a typical image-to-video graph loads.
pub fn text_to_video(o: &T2v, models: &Path) -> Result<PathBuf> {
    generate(o, None, models)
}

/// The same with a start picture (a PNG) held in the first frame at `strength` (1 keeps it as is)
pub fn image_to_video(o: &T2v, start: &Path, strength: f64, models: &Path) -> Result<PathBuf> {
    generate(o, Some((start, strength)), models)
}

fn generate(o: &T2v, start: Option<(&Path, f64)>, models: &Path) -> Result<PathBuf> {
    if !o.width.is_multiple_of(32) || !o.height.is_multiple_of(32) || !(o.frames - 1).is_multiple_of(8) {
        bail!("LTX-2.3 needs width and height in 32s and 1 + 8n frames (got {}x{}, {})", o.width, o.height, o.frames);
    }
    let dev = Device::new_cuda(0)?;
    let t0 = Instant::now();

    // the text: Gemma 3's 49 hidden states, projected, through the connectors
    let g = gemma::Gemma::open(&models.join("text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"))?;
    let proj = unsafe { MmapedSafetensors::new(models.join("checkpoints/ltx-2.3-text-projection.safetensors"))? };
    let projected = encode_prompt(&g, &proj, &o.prompt, &dev)?;
    drop((g, proj));
    let st = unsafe { MmapedSafetensors::new(models.join("checkpoints/ltx-2.3-distilled.safetensors"))? };
    let context = ltx::text_context(&ltx::Connector::load(&st, "video", &dev)?, &ltx::Connector::load(&st, "audio", &dev)?, &projected)?;
    dev.synchronize()?;
    eprintln!("[t2v] text: {} tokens, context {:?} in {:.1}s", projected.dim(1)?, context.dims(), t0.elapsed().as_secs_f64());

    // the latents: empty (EmptyLTXVLatentVideo, LTXVEmptyLatentAudio), the start picture encoded
    // into the first frame
    let (f, h, w) = ((o.frames - 1) / 8 + 1, o.height / 32, o.width / 32);
    let video = [1usize, 128, f, h, w];
    let audio = [1usize, 8, audio_latents(o.frames, o.fps), 16];
    let empty_v = Tensor::zeros(&video, DType::F32, &dev)?;
    let empty_a = Tensor::zeros(&audio, DType::F32, &dev)?;
    let inpaint = match start {
        None => None,
        Some((path, strength)) => {
            let t = Instant::now();
            let (px, iw, ih) = load_image(path)?;
            let enc = ltx_vae::VideoEncoder::load(&models.join("vae/ltx-2.3-video-vae.safetensors"), &dev)?;
            let latent = enc.encode(&start_pixels(&px, iw, ih, w, h, &dev)?)?;
            drop(enc);
            dev.synchronize()?;
            eprintln!("[i2v] {} ({iw}x{ih}) encoded into the first frame in {:.1}s", path.display(), t.elapsed().as_secs_f64());
            Some(img_to_video_inplace(&latent, &empty_v, &empty_a, strength)?)
        }
    };

    // the eight distilled steps, from the noise scaled onto the latent (CONST.noise_scaling)
    let sigmas = ltx::scheduler_sigmas(o.steps, 2.05, 0.95, true, 0.1, f * h * w);
    let (nv, na) = ltx::noise(o.seed, &video, &audio, &dev)?;
    let (lv, la) = match &inpaint {
        Some(m) => (m.v_latent.clone(), m.a_latent.clone()),
        None => (empty_v, empty_a),
    };
    let (xv, xa) = (ltx::noise_scaling(sigmas[0], &nv, &lv)?, ltx::noise_scaling(sigmas[0], &na, &la)?);
    let dit = ltx::Dit::load(&st, &dev)?;
    let mut cache = ltx::BlockCache::for_tokens(f * h * w);
    let tl = Instant::now();
    let resident = cache.preload(&st, &dev)?;
    dev.synchronize()?;
    eprintln!("[t2v] {resident} of the DiT's blocks loaded onto the card in {:.1}s", tl.elapsed().as_secs_f64());
    let t1 = Instant::now();
    let (v, _a) = ltx::sample_euler(
        &dit,
        &st,
        &mut cache,
        xv,
        xa,
        &sigmas,
        &ltx::Guidance::plain(&context),
        o.fps,
        &[],
        inpaint.as_ref(),
        |_, _, _| Ok(()),
    )?;
    dev.synchronize()?;
    eprintln!("[t2v] {} steps on {f}x{h}x{w} latents in {:.1}s", o.steps, t1.elapsed().as_secs_f64());
    drop((dit, cache, context));

    // the frames, and the file
    let t2 = Instant::now();
    let vae = ltx_vae::VideoDecoder::load(&models.join("vae/ltx-2.3-video-vae.safetensors"), &dev)?;
    let frames = vae.decode(&v)?;
    dev.synchronize()?;
    eprintln!("[t2v] decoded {:?} in {:.1}s", frames.dims(), t2.elapsed().as_secs_f64());
    write_mp4(&frames, o.fps, &o.out)?;
    eprintln!("[t2v] {} written, {:.1}s in all", o.out.display(), t0.elapsed().as_secs_f64());
    Ok(o.out.clone())
}

/// debug-ltx-i2v CHECKPOINT I2V_REF IMAGE [--vae FILE]: i2v_ref.py's run piece by piece. The
/// resized picture against ComfyUI's, our encoding of ComfyUI's pixels against its latent, then,
/// from ComfyUI's latent so the encoder's rounding stays out of it, the sampler's latent and mask,
/// the sigmas, the noise, x0, the first call's per-token timesteps and all eight euler steps.
pub fn debug_i2v(ckpt: &Path, reference: &Path, start: &Path, vae: &Path, ancestral: bool) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let pack =
        |v: &Tensor, a: &Tensor| -> Result<Tensor> { Ok(Tensor::cat(&[v.flatten_all()?, a.flatten_all()?], 0)?.reshape((1, 1, ()))?) };
    let theirs_px = get("vae.encode_in")?;
    let (_, ph, pw, _) = theirs_px.dims4()?;
    let (px, iw, ih) = load_image(start)?;
    ltx::compare("resized picture (LoadImage, common_upscale)", &start_pixels(&px, iw, ih, pw / 32, ph / 32, &dev)?, &theirs_px)?;
    let enc = ltx_vae::VideoEncoder::load(vae, &dev)?;
    let theirs_t = get("vae.encode_out")?;
    ltx::compare("encoded picture (from ComfyUI's pixels)", &enc.encode(&theirs_px)?, &theirs_t)?;
    drop(enc);

    let (vs, as_) = (get("in.x.0")?.dims().to_vec(), get("in.x.1")?.dims().to_vec());
    let (empty_v, empty_a) = (Tensor::zeros(vs.as_slice(), DType::F32, &dev)?, Tensor::zeros(as_.as_slice(), DType::F32, &dev)?);
    let m = img_to_video_inplace(&theirs_t, &empty_v, &empty_a, 1.0)?;
    ltx::compare("latent_image (packed)", &pack(&m.v_latent, &m.a_latent)?, &get("latent_image")?)?;
    let mask = pack(&m.v_mask.broadcast_as(vs.as_slice())?, &m.a_mask.broadcast_as(as_.as_slice())?)?;
    ltx::compare("denoise mask (packed)", &mask, &get("denoise_mask")?)?;
    let (f, h, w) = (vs[2], vs[3], vs[4]);
    // i2v_ref.py: LTXVScheduler and euler; i2v_anc_ref.py: ManualSigmas and euler_ancestral
    let sigmas: Vec<f32> = if ancestral {
        "0.909375, 0.725, 0.421875, 0.0".split(", ").map(|v| v.parse::<f64>().map(|x| x as f32)).collect::<std::result::Result<_, _>>()?
    } else {
        ltx::scheduler_sigmas(8, 2.05, 0.95, true, 0.1, f * h * w)
    };
    let theirs: Vec<f32> = get("sigmas")?.to_vec1()?;
    println!("sigmas {}", if sigmas == theirs { "exact" } else { "DIFFER" });
    let (nv, na) = ltx::noise(42, &vs, &as_, &dev)?;
    ltx::compare("noise, video", &nv, &get("noise.0")?)?;
    ltx::compare("noise, audio", &na, &get("noise.1")?)?;
    let (xv, xa) = (ltx::noise_scaling(sigmas[0], &nv, &m.v_latent)?, ltx::noise_scaling(sigmas[0], &na, &m.a_latent)?);
    ltx::compare("x0 (packed)", &pack(&xv, &xa)?, &get("x0")?)?;
    let (tv, ta) = m.token_masks()?;
    ltx::compare("first call's video timesteps", &(tv * sigmas[0] as f64)?.reshape((1, (), 1))?, &get("in.timestep.0")?)?;
    ltx::compare("first call's audio timesteps", &(ta * sigmas[0] as f64)?.reshape((1, (), 1))?, &get("in.timestep.1")?)?;

    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let dit = ltx::Dit::load(&st, &dev)?;
    let context = get("in.context")?;
    let mut cache = ltx::BlockCache::for_tokens(f * h * w);
    let tl = Instant::now();
    let resident = cache.preload(&st, &dev)?;
    dev.synchronize()?;
    println!("{resident} blocks loaded in {:.1} s", tl.elapsed().as_secs_f64());
    let t = Instant::now();
    let each = |i: usize, (xv, xa): (&Tensor, &Tensor), (dv, da): (&Tensor, &Tensor)| -> Result<()> {
        ltx::compare(&format!("step {i} x"), &pack(xv, xa)?, &get(&format!("step{i}.x"))?)?;
        ltx::compare(&format!("step {i} denoised"), &pack(dv, da)?, &get(&format!("step{i}.denoised"))?)?;
        Ok(())
    };
    let (v, a) = if ancestral {
        use candle_core::cuda_backend::cudarc::driver::{sys::CUdevice_attribute, CudaContext};
        let sms = CudaContext::new(0)?.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        ltx::sample_euler_ancestral_rf(
            &dit,
            &st,
            &mut cache,
            xv,
            xa,
            &theirs,
            &ltx::Guidance::plain(&context),
            30.0,
            &[],
            Some(&m),
            42,
            sms,
            each,
        )?
    } else {
        ltx::sample_euler(&dit, &st, &mut cache, xv, xa, &theirs, &ltx::Guidance::plain(&context), 30.0, &[], Some(&m), each)?
    };
    println!("{} steps: {:.1} s", theirs.len() - 1, t.elapsed().as_secs_f64());
    ltx::compare("sampled (packed)", &pack(&v, &a)?, &get("sampled")?)?;
    Ok(())
}

/// The ffmpeg program on the libraries ComfyUI encodes with: PyAV's own build of FFmpeg 8.1.2 and
/// x264 (pyav-ffmpeg 8.1.2-1, third_party/ffmpeg-pyav/README.md), the system's program running on
/// them (FFmpeg keeps its ABI within a major version). The system's FFmpeg 8.0.1 AAC encoder writes
/// other packets for some sound. They are looked for in ~/.local/ffmpeg-8.1.2-pyav/lib, or where
/// RUSTYUI_FFMPEG_LIB says; without them the system's libraries encode, and that is said once.
pub fn ffmpeg() -> std::process::Command {
    static LIB: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    let lib = LIB.get_or_init(|| {
        let dir = match std::env::var_os("RUSTYUI_FFMPEG_LIB") {
            Some(d) => PathBuf::from(d),
            None => Path::new(&std::env::var_os("HOME")?).join(".local/ffmpeg-8.1.2-pyav/lib"),
        };
        if dir.join("libavcodec.so.62").exists() {
            Some(dir)
        } else {
            eprintln!("[engine] no FFmpeg 8.1.2 (PyAV's build) in {}: the system's FFmpeg encodes, and its AAC can differ from ComfyUI's (third_party/ffmpeg-pyav/README.md)", dir.display());
            None
        }
    });
    let mut c = std::process::Command::new("ffmpeg");
    if let Some(dir) = lib {
        let mut path = dir.as_os_str().to_owned();
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH").filter(|o| !o.is_empty()) {
            path.push(":");
            path.push(old);
        }
        c.env("LD_LIBRARY_PATH", path);
    }
    c
}

/// LTXVPreprocess's round trip for one picture (comfy_extras/nodes_lt.py: preprocess): rows and
/// columns cut to even counts, `x * 255` truncated to bytes, H.264 at `crf` (libx264, veryfast) in
/// an MP4, the first frame decoded back, / 255. ComfyUI does it with PyAV; ffmpeg's CLI writes the
/// same bytes on the same box when it converts with bilinear swscale, as PyAV does, and lets x264
/// thread within the frame with an automatic thread count, as PyAV asks (x264 then cuts the frame
/// into slices by the CPU count). (H, W, 3) f32 in, the same out with its new sides.
pub fn preprocess_picture(px: &[f32], w: usize, h: usize, crf: usize) -> Result<(Vec<f32>, usize, usize)> {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let (ew, eh) = (w / 2 * 2, h / 2 * 2);
    anyhow::ensure!(ew > 0 && eh > 0 && px.len() == w * h * 3, "LTXVPreprocess: a {w}x{h} picture");
    let mut bytes = Vec::with_capacity(ew * eh * 3);
    for row in px.chunks(w * 3).take(eh) {
        // .byte(): a truncating cast (the values are already within 0..=255)
        bytes.extend(row[..ew * 3].iter().map(|&v| (v * 255.0) as u8));
    }
    let mp4 = std::env::temp_dir().join(format!(
        "rustyui-preprocess-{}-{}.mp4",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let crf = crf.to_string();
    let mut enc = ffmpeg()
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{ew}x{eh}"),
            "-r",
            "1",
            "-i",
            "-",
        ])
        .args([
            "-sws_flags",
            "bilinear",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            &crf,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "0",
            "-thread_type",
            "slice",
        ])
        .arg(&mp4)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("running ffmpeg")?;
    enc.stdin.take().context("ffmpeg's stdin")?.write_all(&bytes)?;
    let status = enc.wait()?;
    let decoded = if status.success() {
        ffmpeg()
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(&mp4)
            .args(["-frames:v", "1", "-sws_flags", "bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .context("running ffmpeg")
    } else {
        Err(anyhow::anyhow!("ffmpeg could not encode the picture ({status})"))
    };
    let _ = std::fs::remove_file(&mp4);
    let out = decoded?;
    anyhow::ensure!(
        out.status.success() && out.stdout.len() == ew * eh * 3,
        "ffmpeg could not decode the picture back: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok((out.stdout.iter().map(|&b| b as f32 / 255.0).collect(), ew, eh))
}

/// debug-preprocess REF: LTXVPreprocess against ComfyUI's (refs/preprocess_ref.py)
pub fn debug_preprocess(reference: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    for name in ["full", "odd"] {
        let x = r.load(&format!("{name}.in"), &dev)?;
        let (h, w, _) = x.dims3()?;
        let px: Vec<f32> = x.flatten_all()?.to_vec1()?;
        for crf in [18, 35] {
            let (out, ow, oh) = preprocess_picture(&px, w, h, crf)?;
            ltx::compare(
                &format!("{name} {w}x{h}, crf {crf}"),
                &Tensor::from_vec(out, (oh, ow, 3), &dev)?,
                &r.load(&format!("{name}.out{crf}"), &dev)?,
            )?;
        }
    }
    Ok(())
}

/// debug-ltx-guide CHECKPOINT GUIDE_REF IMAGE [--strength 0.7]: LTXVAddGuide's guided sampling
/// (refs/guide_ref.py: the picture as a guide at frame 0 of a 640x448x17 latent, 8 euler
/// steps, then LTXVCropGuides) step by step against ComfyUI's, from ComfyUI's own encoding of the
/// guide (the encoder is compared on its own first)
pub fn debug_guide(ckpt: &Path, reference: &Path, start: &Path, vae: &Path, strength: f64) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let pack =
        |v: &Tensor, a: &Tensor| -> Result<Tensor> { Ok(Tensor::cat(&[v.flatten_all()?, a.flatten_all()?], 0)?.reshape((1, 1, ()))?) };
    let theirs_px = get("vae.encode_in")?;
    let (_, ph, pw, _) = theirs_px.dims4()?;
    let (px, iw, ih) = load_image(start)?;
    ltx::compare("resized guide (LoadImage, common_upscale)", &start_pixels(&px, iw, ih, pw / 32, ph / 32, &dev)?, &theirs_px)?;
    let enc = ltx_vae::VideoEncoder::load(vae, &dev)?;
    let theirs_t = get("vae.encode_out")?;
    ltx::compare("encoded guide (from ComfyUI's pixels)", &enc.encode(&theirs_px)?, &theirs_t)?;
    drop(enc);

    // LTXVAddGuide onto an empty latent: the guide's frame appended, its mask max(0, 1 - strength)
    let (vs, as_) = (get("in.x.0")?.dims().to_vec(), get("in.x.1")?.dims().to_vec());
    let (fg, h, w) = (vs[2], vs[3], vs[4]);
    let f = fg - theirs_t.dim(2)?;
    let v_latent = Tensor::cat(&[&Tensor::zeros((1, 128, f, h, w), DType::F32, &dev)?, &theirs_t], 2)?;
    let v_mask = Tensor::cat(
        &[&Tensor::ones((1, 1, f, h, w), DType::F32, &dev)?, &Tensor::full((1.0 - strength).max(0.0) as f32, (1, 1, fg - f, h, w), &dev)?],
        2,
    )?;
    let m = ltx::Inpaint {
        v_latent,
        v_mask,
        a_latent: Tensor::zeros(as_.as_slice(), DType::F32, &dev)?,
        a_mask: Tensor::ones((1, 1, as_[2], 1), DType::F32, &dev)?,
    };
    ltx::compare("latent_image (packed)", &pack(&m.v_latent, &m.a_latent)?, &get("latent_image")?)?;
    let mask = pack(&m.v_mask.broadcast_as(vs.as_slice())?, &m.a_mask.broadcast_as(as_.as_slice())?)?;
    ltx::compare("denoise mask (packed)", &mask, &get("denoise_mask")?)?;
    let guides = [ltx::Guide { frames: fg - f, h, w, frame_idx: 0, causal_fix: true, strength }];
    let ours_kf: Vec<i64> = guides[0].coords().flat_map(|c| c.into_iter().flatten()).collect();
    // theirs (1, 3, tokens, 2) as (tokens, 3, 2)
    let theirs_kf: Vec<i64> = get("in.keyframe_idxs")?.squeeze(0)?.permute((1, 0, 2))?.contiguous()?.flatten_all()?.to_vec1()?;
    println!("keyframe_idxs {}", if ours_kf == theirs_kf { "exact" } else { "DIFFER" });
    let sigmas = ltx::scheduler_sigmas(8, 2.05, 0.95, true, 0.1, fg * h * w);
    let theirs: Vec<f32> = get("sigmas")?.to_vec1()?;
    println!("sigmas {}", if sigmas == theirs { "exact" } else { "DIFFER" });
    let (nv, na) = ltx::noise(42, &vs, &as_, &dev)?;
    ltx::compare("noise, video", &nv, &get("noise.0")?)?;
    ltx::compare("noise, audio", &na, &get("noise.1")?)?;
    let (xv, xa) = (ltx::noise_scaling(sigmas[0], &nv, &m.v_latent)?, ltx::noise_scaling(sigmas[0], &na, &m.a_latent)?);
    ltx::compare("x0 (packed)", &pack(&xv, &xa)?, &get("x0")?)?;
    let (tv, ta) = m.token_masks()?;
    ltx::compare("first call's video timesteps", &(tv * sigmas[0] as f64)?.reshape((1, (), 1))?, &get("in.timestep.0")?)?;
    ltx::compare("first call's audio timesteps", &(ta * sigmas[0] as f64)?.reshape((1, (), 1))?, &get("in.timestep.1")?)?;

    let st = unsafe { MmapedSafetensors::new(ckpt)? };
    let dit = ltx::Dit::load(&st, &dev)?;
    // the first call's positions and block 0's guide-masked self-attention, on ComfyUI's q, k, v
    if let Ok(pe) = get("b0.v_pe") {
        let pos = dit.positions(fg, h, w, 30.0, as_[2], &guides, &dev)?;
        ltx::compare("video positions (with the guide's)", &pos.v, &pe)?;
        ltx::compare("video cross positions", &pos.v_cross, &get("b0.v_cross_pe")?)?;
        let gm = ltx::guide_mask(fg * h * w, &guides, &dev)?.context("no guide mask at this strength")?;
        ltx::compare("guide mask, noisy queries", &gm.noisy, &get("attn.noisy_mask")?)?;
        ltx::compare("guide mask, guide queries", &gm.tracked, &get("attn.tracked_mask")?)?;
        let out = kernels::guide_attention(&get("attn.q")?, &get("attn.k")?, &get("attn.v")?, 32, gm.guide_start, &gm.noisy, &gm.tracked)?;
        ltx::compare("block 0 guide attention (their q, k, v)", &out, &get("attn.out")?)?;
        if let Ok(x) = get("adaln.x") {
            let ad = kernels::ck_rms_adaln(&x, &get("adaln.scale")?, &get("adaln.shift")?, 1e-6)?;
            ltx::compare("block 0 fused AdaLN (their x, scale, shift)", &ad, &get("adaln.out")?)?;
            let blk = ltx::Block::load(&st, 0, &dev)?;
            let a1 = blk.attn1.forward_traced(&get("attn1.x")?, None, Some(&pos.v), None, Some(&gm), None)?;
            ltx::compare("block 0 attn1 (their input, our positions)", &a1, &get("attn1.out")?)?;
            let a1 = blk.attn1.forward_traced(&get("attn1.x")?, None, Some(&pe), None, Some(&gm), None)?;
            ltx::compare("block 0 attn1 (their input and positions)", &a1, &get("attn1.out")?)?;
            if let Ok(gx) = get("g0.in.x.0") {
                // what the DiT hands block 0, ours against theirs
                let n = |k: &str| get(&format!("g0.in.{k}"));
                let ts = dit.timesteps(&get("in.timestep.0")?, &get("in.timestep.1")?, h * w, false, 1)?;
                for (name, ours, key) in [
                    ("v timestep", &ts.v, "v_timestep.data"),
                    ("a timestep", &ts.a, "a_timestep"),
                    ("v cross scale/shift", &ts.v_cross_ss, "v_cross_scale_shift_timestep.data"),
                    ("a cross scale/shift", &ts.a_cross_ss, "a_cross_scale_shift_timestep"),
                    ("a2v gate", &ts.v_cross_gate, "v_cross_gate_timestep.data"),
                    ("v2a gate", &ts.a_cross_gate, "a_cross_gate_timestep"),
                    ("v prompt", &ts.v_prompt, "v_prompt_timestep"),
                    ("a prompt", &ts.a_prompt, "a_prompt_timestep"),
                ] {
                    ltx::compare(&format!("block 0's {name}"), ours, &n(key)?)?;
                }
                let (pv, pa) = dit.patchify(&get("in.x.0")?, &get("in.x.1")?, true)?;
                ltx::compare("block 0's video input (patchified)", &pv, &gx)?;
                ltx::compare("block 0's audio input (patchified)", &pa, &n("x.1")?)?;
                let args = ltx::BlockArgs {
                    v_context: &n("v_context")?,
                    a_context: &n("a_context")?,
                    v_timestep: &n("v_timestep.data")?,
                    a_timestep: &n("a_timestep")?,
                    v_pe: &n("v_pe.0")?,
                    a_pe: &n("a_pe.0")?,
                    v_cross_pe: &n("v_cross_pe.0")?,
                    a_cross_pe: &n("a_cross_pe.0")?,
                    v_cross_ss: &n("v_cross_scale_shift_timestep.data")?,
                    a_cross_ss: &n("a_cross_scale_shift_timestep")?,
                    v_cross_gate: &n("v_cross_gate_timestep.data")?,
                    a_cross_gate: &n("a_cross_gate_timestep")?,
                    v_prompt: &n("v_prompt_timestep")?,
                    a_prompt: &n("a_prompt_timestep")?,
                    v_mask: Some(&gm),
                };
                let (ov, oa) = blk.forward(&gx, &n("x.1")?, &args)?;
                ltx::compare("block 0 on their inputs, video", &ov, &get("b0.out.0")?)?;
                ltx::compare("block 0 on their inputs, audio", &oa, &get("b0.out.1")?)?;
            }
        }
    }
    let context = get("in.context")?;
    let mut cache = ltx::BlockCache::for_tokens(fg * h * w);
    cache.preload(&st, &dev)?;
    // the first DiT call on ComfyUI's own inputs, block 0 against its capture
    let (ov, oa) = dit.forward_streamed(
        &st,
        &mut cache,
        &get("in.x.0")?,
        &get("in.x.1")?,
        &get("in.timestep.0")?,
        &get("in.timestep.1")?,
        false,
        &context,
        30.0,
        &guides,
        |i, vx, ax| {
            if i == 0 {
                if let (Ok(v), Ok(a)) = (get("b0.out.0"), get("b0.out.1")) {
                    ltx::compare("first call, after block 0, video", vx, &v)?;
                    ltx::compare("first call, after block 0, audio", ax, &a)?;
                }
            }
            Ok(())
        },
    )?;
    ltx::compare("first call's output, video", &ov, &get("out.0")?)?;
    ltx::compare("first call's output, audio", &oa, &get("out.1")?)?;
    let t = Instant::now();
    let each = |i: usize, (xv, xa): (&Tensor, &Tensor), (dv, da): (&Tensor, &Tensor)| -> Result<()> {
        ltx::compare(&format!("step {i} x"), &pack(xv, xa)?, &get(&format!("step{i}.x"))?)?;
        ltx::compare(&format!("step {i} denoised"), &pack(dv, da)?, &get(&format!("step{i}.denoised"))?)?;
        Ok(())
    };
    let (v, a) = ltx::sample_euler(&dit, &st, &mut cache, xv, xa, &theirs, &ltx::Guidance::plain(&context), 30.0, &guides, Some(&m), each)?;
    println!("{} steps: {:.1} s", theirs.len() - 1, t.elapsed().as_secs_f64());
    ltx::compare("sampled (packed)", &pack(&v, &a)?, &get("sampled")?)?;
    // LTXVCropGuides: the guide's frame off the end, what VAEDecode then decodes
    ltx::compare("cropped video latent", &v.narrow(2, 0, f)?, &get("vae.decode_in")?)?;
    Ok(())
}

/// (T, H, W, 3) f32 in [0, 1] to an H.264 MP4, each value as ComfyUI's SaveVideo makes it
/// (`clamp(x * 255, 0, 255).to(uint8)`, truncating), yuv420p, through ffmpeg.
pub fn write_mp4(frames: &Tensor, fps: f64, out: &Path) -> Result<()> {
    write_mp4_meta(frames, fps, out, &[])
}

/// The same with container metadata, as ComfyUI's SaveVideo writes the prompt and the workflow (an
/// ffmetadata file, since a workflow can be longer than one command-line argument may be)
pub fn write_mp4_meta(frames: &Tensor, fps: f64, out: &Path, meta: &[(String, String)]) -> Result<()> {
    write_mp4_av(frames, fps, out, meta, None)
}

/// A soundtrack for write_mp4_av: `len` samples of `channels` interleaved, f32
pub struct Audio {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
    pub len: usize,
}

/// The same with sound, as ComfyUI's SaveVideo writes a VIDEO (VideoFromComponents.save_to, sRGB,
/// 8 bit): the rate as round(fps * 1000) / 1000, each frame `clamp(x * 255, 0, 255).to(uint8)`,
/// yuv420p by bilinear swscale with the BT.709 matrix (limited range), tagged BT.709 / sRGB, H.264 at
/// CRF 18 with x264 threading within each frame as PyAV asks, and the sound as AAC at its own rate,
/// cut to ceil(rate / fps * frames) samples, anything not finite made silence; ffmpeg() on PyAV's
/// FFmpeg libraries. At some widths ComfyUI's own H.264 differs from run to run for the same
/// frames (448 and 480 wide did, 384 and 1280 did not; 2026-10-15), so there it cannot be matched
/// to the byte; this is the stream PyAV writes when it converts every frame before encoding any.
pub fn write_mp4_av(frames: &Tensor, fps: f64, out: &Path, meta: &[(String, String)], audio: Option<&Audio>) -> Result<()> {
    let (t, h, w, _) = frames.dims4()?;
    let meta_file = if meta.is_empty() {
        None
    } else {
        let esc = |v: &str| v.replace('\\', "\\\\").replace('=', "\\=").replace(';', "\\;").replace('#', "\\#").replace('\n', "\\\n");
        let mut body = String::from(";FFMETADATA1\n");
        for (k, v) in meta {
            body.push_str(&format!("{}={}\n", esc(k), esc(v)));
        }
        let p = out.with_extension("ffmeta");
        std::fs::write(&p, body)?;
        Some(p)
    };
    let bytes: Vec<u8> = (frames * 255.0)?
        .clamp(0f32, 255f32)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as u8)
        .collect();
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Fraction(round(fps * 1000), 1000)
    let rate_num = (fps * 1000.0).round() as u64;
    // the soundtrack as raw f32 in a file beside the video (stdin carries the frames)
    let audio_file = match audio {
        Some(a) => {
            let keep = (a.sample_rate as u64 * 1000 * t as u64).div_ceil(rate_num.max(1)) as usize;
            let n = keep.min(a.len) * a.channels;
            let mut raw = Vec::with_capacity(n * 4);
            for &v in &a.samples[..n] {
                raw.extend_from_slice(&(if v.is_finite() { v } else { 0.0 }).to_le_bytes());
            }
            let p = out.with_extension("pcm");
            std::fs::write(&p, raw)?;
            Some((p, a.channels, a.sample_rate))
        }
        None => None,
    };
    let mut cmd = ffmpeg();
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]).args([
        "-s",
        &format!("{w}x{h}"),
        "-r",
        &format!("{rate_num}/1000"),
        "-i",
        "-",
    ]);
    // inputs first (frames, then metadata, then sound), the output's options after them all
    let mut next_input = 1;
    let mut outs = vec!["-map".to_string(), "0:v".to_string()];
    if let Some(p) = &meta_file {
        cmd.arg("-i").arg(p);
        outs.extend(["-map_metadata".to_string(), next_input.to_string()]);
        next_input += 1;
    }
    if let Some((p, c, sr)) = &audio_file {
        cmd.args(["-f", "f32le", "-ar", &sr.to_string(), "-ac", &c.to_string(), "-i"]).arg(p);
        outs.extend(["-map".to_string(), format!("{next_input}:a")]);
    }
    cmd.args(&outs)
        // the frames' own colour tags too (the encoder takes them from the frames), as
        // set_video_color_properties sets them for sRGB
        .args(["-vf", "scale=flags=bilinear:out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv"])
        .args(["-c:v", "libx264", "-crf", "18", "-threads", "0", "-thread_type", "slice"])
        .args(["-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-colorspace", "bt709", "-color_range", "tv"]);
    if audio_file.is_some() {
        cmd.args(["-c:a", "aac"]);
    }
    let mut ff = cmd
        .args(["-movflags", "+faststart+use_metadata_tags"])
        .arg(out)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("running ffmpeg")?;
    // a refused write (ffmpeg gave up) is reported by its exit status below
    let wrote = ff.stdin.as_mut().context("ffmpeg's stdin")?.write_all(&bytes);
    drop(ff.stdin.take());
    let status = ff.wait()?;
    if let Some(p) = meta_file {
        let _ = std::fs::remove_file(p);
    }
    if let Some((p, _, _)) = audio_file {
        let _ = std::fs::remove_file(p);
    }
    if status.success() {
        wrote?;
    }
    if !status.success() {
        bail!("ffmpeg failed writing {} ({} frames of {w}x{h})", out.display(), t);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_length_rounds_like_python() {
        assert_eq!(audio_latents(9, 30.0), 8); // 7.5 -> 8
        assert_eq!(audio_latents(289, 30.0), 241); // 240.83
        assert_eq!(round_half_even(8.5), 8.0);
        assert_eq!(round_half_even(9.5), 10.0);
    }
}
