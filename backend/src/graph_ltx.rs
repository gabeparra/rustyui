//! The LTX-2.3 nodes, as ComfyUI defines them (comfy_extras/nodes_lt.py, the core
//! loaders, CreateVideo and SaveVideo): the text, the latents, the start picture, the custom
//! sampler and its parts, the video decode and the file. What they compute lives in ltx.rs,
//! ltx_vae.rs and ltx_video.rs, the same code `t2v` and `i2v` run; here it is wired to ComfyUI's
//! inputs and outputs, so a ComfyUI workflow of these nodes runs as it is.
//!
//! Not yet: CFG other than 1, samplers other than euler and euler_ancestral. A graph that needs one
//! stops with what is missing.

use super::*;
use crate::{gemma, ltx, ltx_audio, ltx_upsampler, ltx_vae, ltx_video};
use candle_core::safetensors::MmapedSafetensors;

/// An LTX prompt: the projected text (1, L, 6144) bf16, as the DiT takes it before its connectors,
/// the frame rate LTXVConditioning gives it, and the guides LTXVAddGuide appended to the latent
/// (its keyframe_idxs and guide_attention_entries), which LTXVCropGuides clears
pub(super) struct LtxCond {
    projected: Tensor,
    frame_rate: Option<f64>,
    guides: Vec<ltx::Guide>,
}

impl LtxCond {
    /// the projected text, for RUSTYUI_DUMP_NODES
    pub(super) fn tensor(&self) -> &Tensor {
        &self.projected
    }

    /// conditioning_set_values: the same prompt with other guides
    fn with_guides(&self, guides: Vec<ltx::Guide>) -> Self {
        Self { projected: self.projected.clone(), frame_rate: self.frame_rate, guides }
    }
}

/// An LTX LATENT: video (1, 128, F, H, W) and audio (1, 8, T, 16) samples, f32, either or both
/// (LTXVConcatAVLatent's nested latent), and their noise masks, 1 where the sampler generates, in
/// the shapes the nodes make them: one value a frame (1, 1, F, 1, 1) from LTXVImgToVideo and the
/// guides, ones like the samples where LTXVConcatAVLatent fills a missing one. The sampler
/// stretches each to its latent (reshape_mask).
#[derive(Clone, Default)]
pub(super) struct LtxLatent {
    video: Option<Tensor>,
    audio: Option<Tensor>,
    video_mask: Option<Tensor>,
    audio_mask: Option<Tensor>,
}

impl LtxLatent {
    /// the parts that are there, named as ComfyUI's LATENT flattens them (a NestedTensor's video
    /// then audio), for RUSTYUI_DUMP_NODES
    pub(super) fn parts(&self) -> Vec<(&'static str, &Tensor)> {
        let mut v = vec![];
        let both = self.video.is_some() && self.audio.is_some();
        for (name, t) in [
            (if both { "samples.n0" } else { "samples" }, &self.video),
            (if both { "samples.n1" } else { "samples" }, &self.audio),
            (if both { "noise_mask.n0" } else { "noise_mask" }, &self.video_mask),
            (if both { "noise_mask.n1" } else { "noise_mask" }, &self.audio_mask),
        ] {
            if let Some(t) = t {
                v.push((name, t));
            }
        }
        v
    }
}

/// CFGGuider, or LTXVDualCFGGuider (audio_cfg set), over an LTX-2.3 model
pub(super) struct Guider {
    model: Rc<Stack>,
    positive: Rc<LtxCond>,
    negative: Rc<LtxCond>,
    /// CFGGuider's cfg, or the dual guider's video_cfg
    cfg: f64,
    audio_cfg: Option<f64>,
}

/// CreateVideo's VIDEO: frames (T, 3, H, W) f32 in [0, 1], their rate, and the sound with them
pub(super) struct VideoVal {
    frames: Tensor,
    fps: f64,
    audio: Option<Rc<AudioVal>>,
}

/// ComfyUI's AUDIO: {"waveform": (B, C, samples) f32, "sample_rate"}
pub(super) struct AudioVal {
    pub(super) waveform: Tensor,
    sample_rate: u32,
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

impl VideoVal {
    pub(super) fn bytes(&self) -> usize {
        tensor_bytes(&self.frames)
    }
}

impl LtxLatent {
    pub(super) fn bytes(&self) -> usize {
        [&self.video, &self.audio, &self.video_mask, &self.audio_mask].iter().filter_map(|t| t.as_ref()).map(tensor_bytes).sum()
    }
}

impl LtxCond {
    pub(super) fn bytes(&self) -> usize {
        tensor_bytes(&self.projected)
    }
}

/// The LTX-2.3 DiT left on the card between graphs, as ComfyUI keeps a loaded model: its
/// checkpoint, the weights outside the blocks and the blocks that fit
pub(super) struct DitCache {
    /// the checkpoint and the LoRAs folded into its blocks, with their strengths
    key: String,
    st: MmapedSafetensors,
    dit: ltx::Dit,
    blocks: ltx::BlockCache,
}

/// A CUDA allocation that failed for want of memory
pub(super) fn is_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    s.contains("OUT_OF_MEMORY") || s.contains("out of memory")
}

/// The LTX_NODES an image graph uses too (img2img, inpainting, a ControlNet's picture; Anima's
/// model and VAE), which `--no-video` lists and runs
pub const SHARED_NODES: &[&str] = &["LoadImage", "UNETLoader", "VAELoader"];

/// The nodes this file runs (CLIPTextEncode, VAEDecode and LoraLoaderModelOnly take LTX values too)
pub const LTX_NODES: &[&str] = &[
    "LTXAVTextEncoderLoader",
    "UNETLoader",
    "VAELoader",
    "LTXVAudioVAELoader",
    "LoadImage",
    "EmptyLTXVLatentVideo",
    "LTXVEmptyLatentAudio",
    "LTXVImgToVideoInplace",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVConditioning",
    "LTXVScheduler",
    "RandomNoise",
    "KSamplerSelect",
    "CFGGuider",
    "SamplerCustomAdvanced",
    "CreateVideo",
    "SaveVideo",
    "ManualSigmas",
    "LTXVAddGuide",
    "LTXVCropGuides",
    "LTXVPreprocess",
    "LatentUpscaleModelLoader",
    "LTXVLatentUpsampler",
    "LTXVAudioVAEDecode",
    "LTXVImgToVideo",
    "ModelSamplingLTXV",
    "LTXVDualCFGGuider",
];

/// the latent `name` of an LTX node
fn ltx_latent(v: Val, what: &str) -> Result<Rc<LtxLatent>> {
    match v {
        Val::LtxLatent(l) => Ok(l),
        _ => bail!("{what}: not an LTX latent"),
    }
}

impl Engine {
    /// The LTX nodes, or None for a class this file does not run
    pub(super) fn ltx_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        let dev = self.dev.clone();
        Ok(Some(match class {
            "LTXAVTextEncoderLoader" => {
                let (te_name, ckpt_name) = (self.text_arg(inp, "text_encoder")?, self.text_arg(inp, "ckpt_name")?);
                let te = self.model_path("text_encoders", &te_name)?;
                let ckpt = self.model_path("checkpoints", &ckpt_name)?;
                vec![Val::LtxClip(Rc::new((te, ckpt)))]
            }
            "UNETLoader" => {
                let name = self.text_arg(inp, "unet_name")?;
                let path = self.model_path("diffusion_models", &name).or_else(|_| self.model_path("unet", &name))?;
                if super::anima_nodes::is_anima(&path)? {
                    let dtype = self.text_arg(inp, "weight_dtype").unwrap_or_else(|_| "default".into());
                    anyhow::ensure!(
                        dtype == "default",
                        "UNETLoader: weight_dtype {dtype:?} for Anima (\"default\", its bf16, is implemented)"
                    );
                    return Ok(Some(vec![Val::AnimaModel(Rc::new(Stack::new(path)))]));
                }
                let st = unsafe { MmapedSafetensors::new(&path)? };
                if st.get("model.diffusion_model.audio_adaln_single.linear.weight").is_err() {
                    bail!("UNETLoader: {name} is neither an LTX-2.3 audio-video model nor an Anima, the diffusion models rustyui loads this way so far");
                }
                vec![Val::LtxModel(Rc::new(Stack::new(path)))]
            }
            "VAELoader" => {
                let name = self.text_arg(inp, "vae_name")?;
                vec![Val::Vae(self.model_path("vae", &name)?)]
            }
            "LTXVAudioVAELoader" => {
                let name = self.text_arg(inp, "ckpt_name")?;
                vec![Val::AudioVae(self.model_path("checkpoints", &name)?)]
            }
            "LoadImage" => {
                // ComfyUI's input folder, beside models/; "name [input]" annotations as the editor sends them
                let name = self.text_arg(inp, "image")?;
                let name = name.strip_suffix(" [input]").unwrap_or(&name).to_string();
                let rel = crate::paths::contained(&name).with_context(|| format!("LoadImage: {name:?} leaves the input folder"))?;
                let dir = self.models.parent().unwrap_or(&self.models).join("input");
                let path = dir.join(rel);
                if !crate::paths::below(&dir, path.parent().unwrap_or(&dir)) {
                    bail!("LoadImage: {name:?} leads out of {}", dir.display());
                }
                let (px, w, h) = ltx_video::load_image(&path)?;
                let img = Tensor::from_vec(px, (1, h, w, 3), &dev)?.permute((0, 3, 1, 2))?.contiguous()?;
                // a picture without alpha has ComfyUI's empty 64x64 mask
                vec![Val::Image(img), Val::Mask(Tensor::zeros((1, 64, 64), DType::F32, &dev)?)]
            }
            "EmptyLTXVLatentVideo" => {
                let (w, h, length) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?, self.int_arg(inp, "length")?);
                anyhow::ensure!(self.int_arg(inp, "batch_size")? == 1, "EmptyLTXVLatentVideo: one video at a time");
                let z = Tensor::zeros((1, 128, (length - 1) / 8 + 1, h / 32, w / 32), DType::F32, &dev)?;
                vec![Val::LtxLatent(Rc::new(LtxLatent { video: Some(z), ..Default::default() }))]
            }
            "LTXVEmptyLatentAudio" => {
                let (frames, fps) = (self.int_arg(inp, "frames_number")?, self.float_arg(inp, "frame_rate")?);
                anyhow::ensure!(self.int_arg(inp, "batch_size")? == 1, "LTXVEmptyLatentAudio: one clip at a time");
                // the audio VAE fixes 16 kHz, a 160-sample hop and 4x down: 25 latents a second, 16 bins
                let Val::AudioVae(_) = self.input(inp, "audio_vae")? else { bail!("LTXVEmptyLatentAudio: audio_vae is not an audio VAE") };
                let z = Tensor::zeros((1, 8, ltx_video::audio_latents(frames, fps), 16), DType::F32, &dev)?;
                vec![Val::LtxLatent(Rc::new(LtxLatent { audio: Some(z), ..Default::default() }))]
            }
            "LTXVImgToVideo" => {
                // comfy_extras/nodes_lt.py: a zero latent of `length` frames with the picture (resized
                // bilinear, center-cropped, then encoded) in its first frames, their mask 1 - strength;
                // the conditioning passes through
                let (width, height, length) = (self.int_arg(inp, "width")?, self.int_arg(inp, "height")?, self.int_arg(inp, "length")?);
                let batch = self.int_arg(inp, "batch_size")?;
                anyhow::ensure!(batch == 1, "LTXVImgToVideo: batch_size {batch} (one is implemented)");
                anyhow::ensure!(
                    width % 32 == 0 && height % 32 == 0,
                    "LTXVImgToVideo: {width}x{height} (multiples of 32, the node's step, are implemented)"
                );
                let strength = self.float_arg(inp, "strength")?;
                let Val::Image(img) = self.input(inp, "image")? else { bail!("LTXVImgToVideo: image is not an image") };
                let Val::Vae(vae) = self.input(inp, "vae")? else { bail!("LTXVImgToVideo: vae is not a VAE") };
                let (lh, lw, f) = (height / 32, width / 32, (length - 1) / 8 + 1);
                let t = self.ltx_encode_picture(&img, &vae, lw, lh, "encode the start picture")?;
                let n = t.dim(2)?;
                let video = Tensor::cat(&[&t.to_dtype(DType::F32)?, &Tensor::zeros((1, t.dim(1)?, f - n, lh, lw), DType::F32, &dev)?], 2)?;
                let kept = Tensor::full((1.0 - strength) as f32, (1, 1, n, 1, 1), &dev)?;
                let mask = Tensor::cat(&[&kept, &Tensor::ones((1, 1, f - n, 1, 1), DType::F32, &dev)?], 2)?;
                let latent = Val::LtxLatent(Rc::new(LtxLatent { video: Some(video), video_mask: Some(mask), ..Default::default() }));
                vec![self.input(inp, "positive")?, self.input(inp, "negative")?, latent]
            }
            "ModelSamplingLTXV" => {
                // swaps the model's sampling for ModelSamplingFlux with a shift from the latent's
                // token count. LTX already samples as ModelSamplingFlux, whose timestep is sigma,
                // so with sigmas given (LTXVScheduler, ManualSigmas) the shift changes nothing;
                // only a scheduler that reads the model's own sigmas would see it, and rustyui
                // has none for LTX yet
                let m = self.input(inp, "model")?;
                anyhow::ensure!(matches!(m, Val::LtxModel(_)), "ModelSamplingLTXV: model is not an LTX model");
                vec![m]
            }
            "LTXVImgToVideoInplace" => {
                let l = ltx_latent(self.input(inp, "latent")?, "LTXVImgToVideoInplace")?;
                if self.flag_arg(inp, "bypass")? {
                    return Ok(Some(vec![Val::LtxLatent(l)]));
                }
                let strength = self.float_arg(inp, "strength")?;
                let Val::Image(img) = self.input(inp, "image")? else { bail!("LTXVImgToVideoInplace: image is not an image") };
                let Val::Vae(vae) = self.input(inp, "vae")? else { bail!("LTXVImgToVideoInplace: vae is not a VAE") };
                anyhow::ensure!(l.audio.is_none(), "LTXVImgToVideoInplace: takes the video latent, before LTXVConcatAVLatent");
                let video = l.video.as_ref().context("LTXVImgToVideoInplace: the latent has no video")?;
                let (_, _, f, lh, lw) = video.dims5()?;
                let t = self.ltx_encode_picture(&img, &vae, lw, lh, "encode the start picture")?;
                // samples[:, :, :n] = t; get_noise_mask (the latent's own, or ones a frame) with
                // those frames at 1 - strength
                let n = t.dim(2)?;
                let video = Tensor::cat(&[&t.to_dtype(DType::F32)?, &video.narrow(2, n, f - n)?], 2)?;
                let mask = match &l.video_mask {
                    Some(m) => m.clone(),
                    None => Tensor::ones((1, 1, f, 1, 1), DType::F32, &dev)?,
                };
                let (mb, mc, _, mh, mw) = mask.dims5()?;
                let kept = Tensor::full((1.0 - strength) as f32, (mb, mc, n, mh, mw), &dev)?;
                let mask = Tensor::cat(&[&kept, &mask.narrow(2, n, f - n)?], 2)?;
                vec![Val::LtxLatent(Rc::new(LtxLatent { video: Some(video), video_mask: Some(mask), ..Default::default() }))]
            }
            "LTXVConcatAVLatent" => {
                let v = ltx_latent(self.input(inp, "video_latent")?, "LTXVConcatAVLatent")?;
                let a = ltx_latent(self.input(inp, "audio_latent")?, "LTXVConcatAVLatent")?;
                let video = v.video.clone().context("LTXVConcatAVLatent: video_latent has no video")?;
                let audio = a.audio.clone().context("LTXVConcatAVLatent: audio_latent has no audio")?;
                // a mask on either side gives both one, the other ones like its samples
                let (video_mask, audio_mask) = if v.video_mask.is_some() || a.audio_mask.is_some() {
                    (
                        Some(v.video_mask.clone().map_or_else(|| Tensor::ones(video.dims(), DType::F32, &dev), Ok)?),
                        Some(a.audio_mask.clone().map_or_else(|| Tensor::ones(audio.dims(), DType::F32, &dev), Ok)?),
                    )
                } else {
                    (None, None)
                };
                vec![Val::LtxLatent(Rc::new(LtxLatent { video: Some(video), audio: Some(audio), video_mask, audio_mask }))]
            }
            "LTXVSeparateAVLatent" => {
                let l = ltx_latent(self.input(inp, "av_latent")?, "LTXVSeparateAVLatent")?;
                anyhow::ensure!(l.video.is_some() && l.audio.is_some(), "LTXVSeparateAVLatent: not an audio-video latent");
                vec![
                    Val::LtxLatent(Rc::new(LtxLatent { video: l.video.clone(), video_mask: l.video_mask.clone(), ..Default::default() })),
                    Val::LtxLatent(Rc::new(LtxLatent { audio: l.audio.clone(), audio_mask: l.audio_mask.clone(), ..Default::default() })),
                ]
            }
            "LTXVConditioning" => {
                let fr = self.float_arg(inp, "frame_rate")?;
                let mut out = vec![];
                for side in ["positive", "negative"] {
                    let Val::LtxCond(c) = self.input(inp, side)? else { bail!("LTXVConditioning: {side} is not an LTX prompt") };
                    out.push(Val::LtxCond(Rc::new(LtxCond {
                        projected: c.projected.clone(),
                        frame_rate: Some(fr),
                        guides: c.guides.clone(),
                    })));
                }
                out
            }
            "LTXVScheduler" => {
                // the token count shifts the schedule: the video latent's F * H * W, else 4096
                let tokens = match inp.get("latent") {
                    Some(Value::Array(_)) => {
                        let l = ltx_latent(self.input(inp, "latent")?, "LTXVScheduler")?;
                        let v = l.video.as_ref().context("LTXVScheduler: the latent has no video")?;
                        v.dims()[2..].iter().product()
                    }
                    _ => 4096,
                };
                let sigmas = ltx::scheduler_sigmas(
                    self.int_arg(inp, "steps")?,
                    self.float_arg(inp, "max_shift")?,
                    self.float_arg(inp, "base_shift")?,
                    self.flag_arg(inp, "stretch")?,
                    self.float_arg(inp, "terminal")?,
                    tokens,
                );
                vec![Val::Sigmas(Rc::new(sigmas))]
            }
            "ManualSigmas" => {
                // re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas), each float() then a FloatTensor
                let text = self.text_arg(inp, "sigmas")?;
                let re = regex::Regex::new(r"[-+]?(?:\d*\.*\d+)").expect("the pattern compiles");
                let sigmas: Vec<f32> =
                    re.find_iter(&text).map(|m| m.as_str().parse::<f64>().map(|v| v as f32)).collect::<std::result::Result<_, _>>()?;
                anyhow::ensure!(sigmas.len() >= 2, "ManualSigmas: {text:?} holds fewer than two sigmas");
                vec![Val::Sigmas(Rc::new(sigmas))]
            }
            "RandomNoise" => vec![Val::Noise(self.arg(inp, "noise_seed")?.as_u64().context("noise_seed is not a whole number")?)],
            "KSamplerSelect" => vec![Val::Sampler(self.text_arg(inp, "sampler_name")?.to_string())],
            "CFGGuider" | "LTXVDualCFGGuider" => {
                let Val::LtxModel(model) = self.input(inp, "model")? else { bail!("{class}: only an LTX-2.3 model (UNETLoader) so far") };
                let Val::LtxCond(positive) = self.input(inp, "positive")? else { bail!("{class}: positive is not an LTX prompt") };
                let Val::LtxCond(negative) = self.input(inp, "negative")? else { bail!("{class}: negative is not an LTX prompt") };
                let (cfg, audio_cfg) = if class == "CFGGuider" {
                    (self.float_arg(inp, "cfg")?, None)
                } else {
                    (self.float_arg(inp, "video_cfg")?, Some(self.float_arg(inp, "audio_cfg")?))
                };
                vec![Val::Guider(Rc::new(Guider { model, positive, negative, cfg, audio_cfg }))]
            }
            "SamplerCustomAdvanced" => self.sampler_custom_advanced(inp)?,
            "LTXVAddGuide" => {
                let Val::LtxCond(pos) = self.input(inp, "positive")? else { bail!("LTXVAddGuide: positive is not an LTX prompt") };
                let Val::LtxCond(neg) = self.input(inp, "negative")? else { bail!("LTXVAddGuide: negative is not an LTX prompt") };
                let Val::Vae(vae) = self.input(inp, "vae")? else { bail!("LTXVAddGuide: vae is not a VAE") };
                let l = ltx_latent(self.input(inp, "latent")?, "LTXVAddGuide")?;
                let Val::Image(img) = self.input(inp, "image")? else { bail!("LTXVAddGuide: image is not an image") };
                for opt in ["attention_mask", "iclora_parameters"] {
                    if matches!(inp.get(opt), Some(Value::Array(_))) {
                        bail!("LTXVAddGuide: {opt} is not implemented yet; leave it unconnected");
                    }
                }
                let frame_idx = self.arg(inp, "frame_idx")?.as_i64().context("frame_idx is not a whole number")?;
                let strength = self.float_arg(inp, "strength")?;
                anyhow::ensure!(l.audio.is_none(), "LTXVAddGuide: Adding guide to a combined AV latent is not supported.");
                let video = l.video.as_ref().context("LTXVAddGuide: the latent has no video")?;
                let (_, _, f, lh, lw) = video.dims5()?;
                let n = img.dim(0)?;
                anyhow::ensure!(n == 1, "LTXVAddGuide: a guide video ({n} frames) is not implemented yet, one picture is");
                // one picture: always causal_fix, one latent frame
                let t = self.ltx_encode_picture(&img, &vae, lw, lh, "encode the guide picture")?;
                // get_latent_index: frame_idx against the latent's own frames (without earlier guides)
                let keyframes: usize = pos.guides.iter().map(|g| g.frames).sum();
                let count = f.checked_sub(keyframes).context("LTXVAddGuide: the latent is shorter than its guides")? as i64;
                let fi = if frame_idx >= 0 { frame_idx } else { ((count - 1) * 8 + 1 + frame_idx).max(0) };
                let latent_idx = (fi + 7).div_euclid(8);
                let tf = t.dim(2)?;
                anyhow::ensure!(
                    latent_idx + tf as i64 <= f as i64,
                    "LTXVAddGuide: Conditioning frames exceed the length of the latent sequence."
                );
                let guide = ltx::Guide { frames: tf, h: lh, w: lw, frame_idx: fi, causal_fix: true, strength };
                let add = |c: &LtxCond| {
                    let mut g = c.guides.clone();
                    g.push(guide.clone());
                    Val::LtxCond(Rc::new(c.with_guides(g)))
                };
                // the guide's latent appended, its mask max(0, 1 - strength) (more only through the
                // attention) in the plane of get_noise_mask's (the latent's own, or ones a frame)
                let mask = match &l.video_mask {
                    Some(m) => m.clone(),
                    None => Tensor::ones((1, 1, f, 1, 1), DType::F32, &dev)?,
                };
                let (mb, mc, _, mh, mw) = mask.dims5()?;
                anyhow::ensure!(mc == 1, "LTXVAddGuide: the latent's noise mask has {mc} channels (torch.cat refuses it)");
                let kept = Tensor::full((1.0 - strength).max(0.0) as f32, (mb, 1, tf, mh, mw), &dev)?;
                let video = Tensor::cat(&[video, &t.to_dtype(DType::F32)?], 2)?;
                let mask = Tensor::cat(&[&mask, &kept], 2)?;
                vec![
                    add(&pos),
                    add(&neg),
                    Val::LtxLatent(Rc::new(LtxLatent { video: Some(video), video_mask: Some(mask), ..Default::default() })),
                ]
            }
            "LatentUpscaleModelLoader" => {
                let name = self.text_arg(inp, "model_name")?;
                vec![Val::LatentUpscaler(self.model_path("latent_upscale_models", &name)?)]
            }
            "LTXVLatentUpsampler" => {
                let l = ltx_latent(self.input(inp, "samples")?, "LTXVLatentUpsampler")?;
                let Val::LatentUpscaler(model) = self.input(inp, "upscale_model")? else {
                    bail!("LTXVLatentUpsampler: upscale_model is not a latent upscale model")
                };
                let Val::Vae(vae) = self.input(inp, "vae")? else { bail!("LTXVLatentUpsampler: vae is not a VAE") };
                anyhow::ensure!(l.audio.is_none(), "LTXVLatentUpsampler: takes the video latent (LTXVSeparateAVLatent first)");
                let video = l.video.as_ref().context("LTXVLatentUpsampler: the latent has no video")?;
                // the VAE's per-channel statistics, in the VAE's dtype
                let st = unsafe { MmapedSafetensors::new(&vae)? };
                let std = st.load("per_channel_statistics.std-of-means", &dev)?.to_dtype(DType::BF16)?;
                let mean = st.load("per_channel_statistics.mean-of-means", &dev)?.to_dtype(DType::BF16)?;
                (self.clip, self.unet, self.ltx_te) = (None, None, None);
                (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
                let t0 = Instant::now();
                let run = || -> Result<Tensor> {
                    ltx_upsampler::upsample(&ltx_upsampler::LatentUpsampler::load(&model, &dev)?, video, &std, &mean)
                };
                let out = match run() {
                    Err(e) if is_oom(&e) && self.ltx_dit.is_some() => {
                        self.ltx_dit = None;
                        run()?
                    }
                    r => r?,
                };
                self.dev.synchronize()?;
                self.timings.push(("upsample the latent".into(), t0.elapsed().as_secs_f64()));
                // the noise mask is dropped
                vec![Val::LtxLatent(Rc::new(LtxLatent { video: Some(out), ..Default::default() }))]
            }
            "LTXVPreprocess" => {
                let Val::Image(img) = self.input(inp, "image")? else { bail!("LTXVPreprocess: image is not an image") };
                let crf = self.int_arg(inp, "img_compression")?;
                if crf == 0 {
                    return Ok(Some(vec![Val::Image(img)]));
                }
                let (n, _, h, w) = img.dims4()?;
                let t0 = Instant::now();
                let mut frames = Vec::with_capacity(n);
                for i in 0..n {
                    let px: Vec<f32> = img.narrow(0, i, 1)?.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1()?;
                    let (out, ow, oh) = ltx_video::preprocess_picture(&px, w, h, crf)?;
                    frames.push(Tensor::from_vec(out, (1, oh, ow, 3), &dev)?.permute((0, 3, 1, 2))?.contiguous()?);
                }
                self.timings.push(("LTXVPreprocess (an H.264 round trip)".into(), t0.elapsed().as_secs_f64()));
                vec![Val::Image(Tensor::cat(&frames, 0)?)]
            }
            "LTXVCropGuides" => {
                let Val::LtxCond(pos) = self.input(inp, "positive")? else { bail!("LTXVCropGuides: positive is not an LTX prompt") };
                let Val::LtxCond(neg) = self.input(inp, "negative")? else { bail!("LTXVCropGuides: negative is not an LTX prompt") };
                let l = ltx_latent(self.input(inp, "latent")?, "LTXVCropGuides")?;
                anyhow::ensure!(l.audio.is_none(), "LTXVCropGuides: takes the video latent (LTXVSeparateAVLatent first)");
                let video = l.video.as_ref().context("LTXVCropGuides: the latent has no video")?;
                let (_, _, f, h, w) = video.dims5()?;
                // get_noise_mask: the latent's own, or ones a frame (the output always carries one)
                let mask = match &l.video_mask {
                    Some(m) => m.clone(),
                    None => Tensor::ones((1, 1, f, 1, 1), DType::F32, &dev)?,
                };
                // get_keyframe_idxs(positive, latent_shape): its keyframe tokens in this latent's frames
                let n = pos.guides.iter().map(ltx::Guide::tokens).sum::<usize>() / (h * w);
                if n == 0 {
                    vec![
                        Val::LtxCond(pos),
                        Val::LtxCond(neg),
                        Val::LtxLatent(Rc::new(LtxLatent { video: Some(video.clone()), video_mask: Some(mask), ..Default::default() })),
                    ]
                } else {
                    anyhow::ensure!(n < f, "LTXVCropGuides: {n} guide frames in a latent of {f}");
                    let (video, mask) = (video.narrow(2, 0, f - n)?, mask.narrow(2, 0, f - n)?);
                    vec![
                        Val::LtxCond(Rc::new(pos.with_guides(vec![]))),
                        Val::LtxCond(Rc::new(neg.with_guides(vec![]))),
                        Val::LtxLatent(Rc::new(LtxLatent { video: Some(video), video_mask: Some(mask), ..Default::default() })),
                    ]
                }
            }
            "LTXVAudioVAEDecode" => {
                let l = ltx_latent(self.input(inp, "samples")?, "LTXVAudioVAEDecode")?;
                let Val::AudioVae(vae) = self.input(inp, "audio_vae")? else { bail!("LTXVAudioVAEDecode: audio_vae is not an audio VAE") };
                // a joint latent's audio, as unbind()[-1] picks it
                let latent = l.audio.as_ref().context("LTXVAudioVAEDecode: the latent has no audio")?;
                (self.clip, self.unet, self.ltx_te) = (None, None, None);
                (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
                let t0 = Instant::now();
                let run = || -> Result<(Tensor, u32)> {
                    let a = ltx_audio::AudioVae::load(&vae, &dev)?;
                    Ok((a.decode(latent)?, a.out_rate as u32))
                };
                let (waveform, sample_rate) = match run() {
                    Err(e) if is_oom(&e) && self.ltx_dit.is_some() => {
                        self.ltx_dit = None;
                        run()?
                    }
                    r => r?,
                };
                self.dev.synchronize()?;
                self.timings.push(("decode the audio".into(), t0.elapsed().as_secs_f64()));
                vec![Val::Audio(Rc::new(AudioVal { waveform, sample_rate }))]
            }
            "CreateVideo" => {
                let Val::Image(frames) = self.input(inp, "images")? else { bail!("CreateVideo: images is not an image") };
                let audio = match inp.get("audio") {
                    Some(Value::Array(_)) => match self.input(inp, "audio")? {
                        Val::Audio(a) => Some(a),
                        _ => bail!("CreateVideo: audio is not audio"),
                    },
                    _ => None,
                };
                for (key, ok) in
                    [("color_space", &["sRGB"][..]), ("bit_depth", &["auto", "8"][..]), ("codec", &["none", "auto", "h264"][..])]
                {
                    if let Some(v) = inp.get(key).filter(|v| !v.is_array()) {
                        let v = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                        anyhow::ensure!(ok.contains(&v.as_str()), "CreateVideo: {key} {v} is not implemented ({} is)", ok.join(", "));
                    }
                }
                vec![Val::Video(Rc::new(VideoVal { frames, fps: self.float_arg(inp, "fps")?, audio }))]
            }
            "SaveVideo" => {
                let prefix = self.text_arg(inp, "filename_prefix")?.to_string();
                for (key, ok) in [("format", ["auto", "mp4"]), ("codec", ["auto", "h264"])] {
                    if let Some(v) = inp.get(key).and_then(Value::as_str) {
                        anyhow::ensure!(ok.contains(&v), "SaveVideo: {key} {v} is not implemented (H.264 in an MP4 is)");
                    }
                }
                let Val::Video(v) = self.input(inp, "video")? else { bail!("SaveVideo: video is not a video") };
                let path = next_output(&self.out_dir, &prefix, "mp4")?;
                let mut meta = vec![("prompt".to_string(), image::ascii_json(&Value::Object(self.graph.clone())))];
                meta.extend(self.pnginfo.iter().cloned());
                let t0 = Instant::now();
                let audio = match &v.audio {
                    Some(a) => {
                        let (_, c, n) = a.waveform.dims3()?;
                        // waveform[0], channels interleaved
                        let samples: Vec<f32> = a.waveform.get(0)?.t()?.contiguous()?.flatten_all()?.to_vec1()?;
                        Some(ltx_video::Audio { samples, channels: c, sample_rate: a.sample_rate, len: n })
                    }
                    None => None,
                };
                ltx_video::write_mp4_av(&v.frames.permute((0, 2, 3, 1))?, v.fps, &path, &meta, audio.as_ref())?;
                self.timings.push(("write the video".into(), t0.elapsed().as_secs_f64()));
                self.saved.push(path);
                // ComfyUI's SaveVideo hands its video on (a PlaySoundKJ node can hang off it)
                vec![Val::Video(v)]
            }
            _ => return Ok(None),
        }))
    }

    /// CLIPTextEncode with LTXAVTextEncoderLoader's CLIP (Gemma kept loaded for the next prompt)
    pub(super) fn ltx_encode(&mut self, files: &(PathBuf, PathBuf), prompt: &str) -> Result<Val> {
        let key = format!("{}|{}", files.0.display(), files.1.display());
        if self.ltx_te.as_ref().map(|(k, _, _)| k != &key).unwrap_or(true) {
            self.free();
            let t = Instant::now();
            let g = gemma::Gemma::open(&files.0)?;
            let proj = unsafe { MmapedSafetensors::new(&files.1)? };
            self.timings.push(("load Gemma 3".into(), t.elapsed().as_secs_f64()));
            self.ltx_te = Some((key, g, proj));
        }
        let t = Instant::now();
        let (_, g, proj) = self.ltx_te.as_ref().unwrap();
        let projected = ltx_video::encode_prompt(g, proj, prompt, &self.dev)?;
        self.timings.push(("encode prompt".into(), t.elapsed().as_secs_f64()));
        Ok(Val::LtxCond(Rc::new(LtxCond { projected, frame_rate: None, guides: vec![] })))
    }

    /// VAE-encodes the first picture of `img` (N, 3, H, W) for a (lw, lh) latent, as
    /// LTXVImgToVideoInplace and LTXVAddGuide do: common_upscale bilinear, center crop, to the
    /// latent's pixel size. Everything else leaves the card first; the DiT too if it must.
    fn ltx_encode_picture(&mut self, img: &Tensor, vae: &Path, lw: usize, lh: usize, what: &str) -> Result<Tensor> {
        let dev = self.dev.clone();
        let (_, _, ih, iw) = img.dims4()?;
        let px: Vec<f32> = img.narrow(0, 0, 1)?.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1()?;
        (self.clip, self.unet, self.ltx_te) = (None, None, None);
        (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
        let t0 = Instant::now();
        let encode =
            || -> Result<Tensor> { ltx_vae::VideoEncoder::load(vae, &dev)?.encode(&ltx_video::start_pixels(&px, iw, ih, lw, lh, &dev)?) };
        let t = match encode() {
            Err(e) if is_oom(&e) && self.ltx_dit.is_some() => {
                self.ltx_dit = None;
                encode()?
            }
            r => r?,
        };
        self.dev.synchronize()?;
        self.timings.push((what.into(), t0.elapsed().as_secs_f64()));
        Ok(t)
    }

    /// VAEDecode of an LTX video latent: frames (T, 3, H, W). The DiT stays on the card for the
    /// next graph unless the decoder needs its memory.
    pub(super) fn ltx_decode(&mut self, l: &LtxLatent, vae: &Path) -> Result<Val> {
        let video = l.video.as_ref().context("VAEDecode: the latent has no video (LTXVSeparateAVLatent first)")?;
        (self.clip, self.unet, self.ltx_te) = (None, None, None);
        (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
        let size: usize = video.dims()[2..].iter().product();
        if self.ltx_decode_oom.is_some_and(|n| size >= n) {
            self.ltx_dit = None;
        }
        let t = Instant::now();
        let dev = self.dev.clone();
        let decode = || -> Result<Tensor> { ltx_vae::VideoDecoder::load(vae, &dev)?.decode(video) };
        let frames = match decode() {
            Err(e) if is_oom(&e) && self.ltx_dit.is_some() => {
                self.ltx_decode_oom = Some(self.ltx_decode_oom.map_or(size, |n| n.min(size)));
                self.ltx_dit = None;
                decode()?
            }
            r => r?,
        };
        self.dev.synchronize()?;
        self.vram();
        self.timings.push(("decode the video".into(), t.elapsed().as_secs_f64()));
        Ok(Val::Image(frames.permute((0, 3, 1, 2))?.contiguous()?))
    }

    fn sampler_custom_advanced(&mut self, inp: &serde_json::Map<String, Value>) -> Result<Vec<Val>> {
        let Val::Noise(seed) = self.input(inp, "noise")? else { bail!("SamplerCustomAdvanced: noise is not NOISE") };
        let Val::Guider(g) = self.input(inp, "guider")? else { bail!("SamplerCustomAdvanced: guider is not a GUIDER") };
        let Val::Sampler(sampler) = self.input(inp, "sampler")? else { bail!("SamplerCustomAdvanced: sampler is not a SAMPLER") };
        let Val::Sigmas(sigmas) = self.input(inp, "sigmas")? else { bail!("SamplerCustomAdvanced: sigmas is not SIGMAS") };
        let l = ltx_latent(self.input(inp, "latent_image")?, "SamplerCustomAdvanced")?;
        anyhow::ensure!(
            matches!(sampler.as_str(), "euler" | "euler_ancestral"),
            "SamplerCustomAdvanced: {sampler} is not implemented for LTX yet (euler and euler_ancestral are)"
        );
        // ComfyUI's CFG skips the negative at cfg 1 (math.isclose); the dual guider with equal scales
        // is plain CFG at that scale, and with different ones mixes in noise space, always running
        // the negative
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
        let (video_cfg, audio_cfg, dual) = match g.audio_cfg {
            Some(a) if !close(g.cfg, a) => (g.cfg, a, true),
            _ => (g.cfg, g.cfg, false),
        };
        let with_negative = dual || !close(video_cfg, 1.0);
        if with_negative {
            // both passes take the prompt's guides and frame rate; LTXVAddGuide and
            // LTXVConditioning give the negative the same
            anyhow::ensure!(
                g.negative.guides.len() == g.positive.guides.len(),
                "SamplerCustomAdvanced: the negative prompt has other guides than the positive"
            );
            anyhow::ensure!(
                g.negative.frame_rate == g.positive.frame_rate,
                "SamplerCustomAdvanced: the negative prompt has another frame rate than the positive"
            );
        }
        let video = l.video.clone().context("SamplerCustomAdvanced: the latent has no video")?;
        let audio = l.audio.clone().context("SamplerCustomAdvanced: LTX-2.3 samples audio and video together (LTXVConcatAVLatent)")?;
        // LTXAV.extra_conds: the conditioning's frame rate, 25 without LTXVConditioning
        let fps = g.positive.frame_rate.unwrap_or(25.0);
        // everything else off the card; the DiT stays when it is this checkpoint
        (self.clip, self.unet, self.ltx_te) = (None, None, None);
        (self.vae_dec, self.vae_enc, self.esrgan) = (None, None, None);
        if std::env::var_os("RUSTYUI_VRAM_DEBUG").is_some() {
            let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
            eprintln!(
                "[vram] sampler start: {:.2} GB free of {:.2}, DiT cached {}",
                free as f64 / 1e9,
                total as f64 / 1e9,
                self.ltx_dit.is_some()
            );
        }
        let key = g.model.loras.iter().fold(g.model.ckpt.display().to_string(), |k, (p, s)| format!("{k}|{}@{s}", p.display()));
        if self.ltx_dit.as_ref().is_some_and(|d| d.key != key) {
            self.ltx_dit = None;
        }
        let dev = self.dev.clone();
        // the DiT runs the negative and the prompt as one batch: twice the activations
        let tokens: usize = video.dims()[2..].iter().product::<usize>() * if with_negative { 2 } else { 1 };
        let t0 = Instant::now();
        if self.ltx_dit.is_none() {
            let st = unsafe { MmapedSafetensors::new(&g.model.ckpt)? };
            let dit = ltx::Dit::load(&st, &dev)?;
            // LoRAs fold into each block as it loads (ltx::Linear::load_lora, ComfyUI's arithmetic)
            let loras = if g.model.loras.is_empty() {
                None
            } else {
                let l = ltx::Loras::open(&g.model.loras)?;
                for (p, n) in l.unmatched(&st) {
                    eprintln!("[engine] {}: {n} LoRA layers match no weight of {}", p.display(), g.model.ckpt.display());
                }
                Some(Rc::new(l))
            };
            let blocks = ltx::BlockCache::for_tokens(tokens).with_loras(loras);
            self.ltx_dit = Some(DitCache { key, st, dit, blocks });
            dev.synchronize()?;
            self.timings.push(("load the DiT's weights outside its blocks".into(), t0.elapsed().as_secs_f64()));
        }
        // the connectors once per prompt (and checkpoint), before the blocks take the card
        let mut wanted = vec![g.positive.clone()];
        if with_negative {
            wanted.push(g.negative.clone());
        }
        self.ltx_context.retain(|(p, id, _)| *p == g.model.ckpt && wanted.iter().any(|c| c.projected.id() == *id));
        let missing: Vec<Rc<LtxCond>> =
            wanted.iter().filter(|c| !self.ltx_context.iter().any(|(_, id, _)| *id == c.projected.id())).cloned().collect();
        if !missing.is_empty() {
            let tc = Instant::now();
            let d = self.ltx_dit.as_mut().unwrap();
            // room for the connectors' weights and their work beside the activations' reserve
            d.blocks.fit_with(tokens, ltx::connector_bytes(&d.st) + (512 << 20), &dev)?;
            let (cv, ca) = (ltx::Connector::load(&d.st, "video", &dev)?, ltx::Connector::load(&d.st, "audio", &dev)?);
            for c in &missing {
                let t = ltx::text_context(&cv, &ca, &c.projected)?;
                self.ltx_context.push((g.model.ckpt.clone(), c.projected.id(), t));
            }
            dev.synchronize()?;
            self.timings.push(("the text context (the connectors)".into(), tc.elapsed().as_secs_f64()));
        }
        let context_of = |c: &LtxCond| {
            self.ltx_context
                .iter()
                .find(|(_, id, _)| *id == c.projected.id())
                .map(|(_, _, t)| t.clone())
                .context("the prompt's text context")
        };
        let context = context_of(&g.positive)?;
        let negative_context = if with_negative { Some(context_of(&g.negative)?) } else { None };
        let d = self.ltx_dit.as_mut().unwrap();
        // CFGGuider.sample: each mask made its latent's shape (prepare_mask), f32
        let inpaint = match (&l.video_mask, &l.audio_mask) {
            (Some(vm), Some(am)) => Some(ltx::Inpaint {
                v_latent: video.clone(),
                v_mask: crate::sampling::reshape_mask(vm, video.dims(), false)?,
                a_latent: audio.clone(),
                a_mask: crate::sampling::reshape_mask(am, audio.dims(), false)?,
            }),
            (None, None) => None,
            _ => bail!("SamplerCustomAdvanced: a mask on only one of video and audio"),
        };
        let (nv, na) = ltx::noise(seed, video.dims(), audio.dims(), &dev)?;
        let (xv, xa) = (ltx::noise_scaling(sigmas[0], &nv, &video)?, ltx::noise_scaling(sigmas[0], &na, &audio)?);
        // as many blocks on the card as leave room for this graph's activations
        let tb = Instant::now();
        d.blocks.fit(tokens, &dev)?;
        let before = d.blocks.resident();
        let resident = d.blocks.preload(&d.st, &dev)?;
        dev.synchronize()?;
        let load_secs = tb.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let steps = sigmas.len() - 1;
        let mut last = None;
        let each = |i: usize, _: (&Tensor, &Tensor), (dv, da): (&Tensor, &Tensor)| -> Result<()> {
            crate::progress::step(i + 1, steps)?;
            last = Some((dv.clone(), da.clone()));
            Ok(())
        };
        let guidance = ltx::Guidance { cond: &context, uncond: negative_context.as_ref(), video_cfg, audio_cfg, dual };
        let (v, a) = if sampler == "euler" {
            ltx::sample_euler(&d.dit, &d.st, &mut d.blocks, xv, xa, &sigmas, &guidance, fps, &g.positive.guides, inpaint.as_ref(), each)?
        } else {
            ltx::sample_euler_ancestral_rf(
                &d.dit,
                &d.st,
                &mut d.blocks,
                xv,
                xa,
                &sigmas,
                &guidance,
                fps,
                &g.positive.guides,
                inpaint.as_ref(),
                seed,
                self.sms,
                each,
            )?
        };
        self.timings.push((format!("{} of the DiT's blocks onto the card ({resident} there)", resident.saturating_sub(before)), load_secs));
        // inverse_noise_scaling: x / (1 - the last sigma), nothing when it is 0
        let end = sigmas[steps];
        let (v, a) = if end != 0.0 { ((v / (1.0 - end) as f64)?, (a / (1.0 - end) as f64)?) } else { (v, a) };
        self.dev.synchronize()?;
        self.vram();
        let secs = t1.elapsed().as_secs_f64();
        self.timings.push((format!("sample {steps} steps ({:.2} s/step)", secs / steps.max(1) as f64), secs));
        let keep = |v: Tensor, a: Tensor| LtxLatent {
            video: Some(v),
            audio: Some(a),
            video_mask: l.video_mask.clone(),
            audio_mask: l.audio_mask.clone(),
        };
        let (dv, da) = last.context("SamplerCustomAdvanced: no steps")?;
        Ok(vec![Val::LtxLatent(Rc::new(keep(v, a))), Val::LtxLatent(Rc::new(keep(dv, da)))])
    }
}

/// ComfyUI's get_save_image_path for one file: `prefix_NNNNN_.ext` in the prefix's folder under
/// `dir`, the counter one past the highest already there for that prefix, whatever the extension
fn next_output(dir: &Path, prefix: &str, ext: &str) -> Result<PathBuf> {
    let rel = crate::paths::contained(prefix).with_context(|| format!("filename_prefix {prefix:?} leaves the output folder"))?;
    let target = dir.join(rel);
    let folder = target.parent().unwrap_or(dir).to_path_buf();
    let stem = target.file_name().and_then(|s| s.to_str()).unwrap_or("rustyui").to_string();
    std::fs::create_dir_all(dir)?;
    if !crate::paths::below(dir, &folder) {
        bail!("filename_prefix {prefix:?} leads out of {}", dir.display());
    }
    std::fs::create_dir_all(&folder)?;
    let mut counter = 1;
    for entry in std::fs::read_dir(&folder)? {
        let name = entry?.file_name().to_string_lossy().to_string();
        if let Some(num) = name.strip_prefix(&format!("{stem}_")).and_then(|r| r.split('_').next()).and_then(|n| n.parse::<usize>().ok()) {
            counter = counter.max(num + 1);
        }
    }
    Ok(folder.join(format!("{stem}_{counter:05}_.{ext}")))
}
