//! The Anima nodes, as a typical Anima graph uses them: CLIPLoader on Qwen3-0.6B, CLIPTextEncode on
//! it, UNETLoader on an Anima checkpoint, KSampler with that model, VAEDecode of its latent through
//! the Wan 2.1 VAE. What they compute lives in anima_tokens.rs, qwen3.rs, anima.rs and wan_vae.rs.
//!
//! Not yet: LoRAs on the text encoder, denoise below 1, latents other than an empty one, a batch of more than one
//! picture, samplers other than euler and er_sde, schedulers other than simple and beta. Unchecked
//! against ComfyUI so far (graph_nodes_ref.py with CAPTURE_ANIMA=1 on a plain Anima graph is the
//! first check).

use super::*;
use crate::{anima, anima_tokens, qwen3, wan_vae};
use candle_core::safetensors::MmapedSafetensors;

/// Anima's ModelSamplingDiscreteFlow: sampling_settings {"shift": 3.0, "multiplier": 1.0}
const SHIFT: f64 = 3.0;

/// Wan21's latent format (process_out: latent * std / 1.0 + mean, per channel)
const LATENT_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503,
    -0.2921,
];
const LATENT_STD: [f32; 16] =
    [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160];

/// An Anima prompt as CLIPTextEncode leaves it: Qwen3's states (1, T, 1024) f32 and the T5 ids
/// with their weights (the conditioning's t5xxl_ids and t5xxl_weights)
pub(super) struct AnimaCond {
    qwen: Tensor,
    t5: Vec<(u32, f64)>,
}

impl AnimaCond {
    /// the parts named as ComfyUI's CONDITIONING flattens them, for RUSTYUI_DUMP_NODES
    pub(super) fn parts(&self) -> Result<Vec<(&'static str, Tensor)>> {
        let ids: Vec<i64> = self.t5.iter().map(|&(id, _)| id as i64).collect();
        let w: Vec<f32> = self.t5.iter().map(|&(_, w)| w as f32).collect();
        Ok(vec![
            ("0.0", self.qwen.clone()),
            ("0.1.t5xxl_ids", Tensor::new(ids, &Device::Cpu)?),
            ("0.1.t5xxl_weights", Tensor::new(w, &Device::Cpu)?),
        ])
    }

    pub(super) fn bytes(&self) -> usize {
        self.qwen.elem_count() * 4
    }
}

/// CLIPLoader's types under which ComfyUI's load_clip gives a Qwen3-0.6B file Anima's encoder
/// (the type only matters for the files it detects as other models)
pub const ANIMA_NODES: &[&str] = &["CLIPLoader"];

/// An Anima diffusion model file: MiniTrainDIT's keys with the LLM adapter's
pub(super) fn is_anima(path: &Path) -> Result<bool> {
    let st = unsafe { MmapedSafetensors::new(path)? };
    Ok(st.get("net.llm_adapter.blocks.0.cross_attn.q_proj.weight").is_ok()
        || st.get("llm_adapter.blocks.0.cross_attn.q_proj.weight").is_ok())
}

/// The CFG denoiser for Anima (a CONST model): the DiT on bf16 x with timestep sigma, its output
/// back to f32, x - output * sigma; at a cfg math.isclose to 1 the positive alone, else one batch
/// of the negative's then the positive's
struct AnimaDenoiser<'a> {
    dit: &'a anima::AnimaDit,
    pos: Tensor,
    neg: Tensor,
    cfg: f32,
    skip_uncond: bool,
    dev: &'a Device,
    last_uncond: Option<Tensor>,
}

impl crate::sampling::Denoise for AnimaDenoiser<'_> {
    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let b = x.dim(0)?;
        let xc = x.to_dtype(DType::BF16)?;
        let sig = Tensor::new(sigma, self.dev)?;
        let rep = |c: &Tensor| -> Result<Tensor> { Ok(if b > 1 { c.repeat((b, 1, 1))? } else { c.clone() }) };
        let den = |out: &Tensor| -> Result<Tensor> { Ok((x - out.to_dtype(DType::F32)?.broadcast_mul(&sig)?)?) };
        let (cond, uncond) = if self.skip_uncond {
            let out = self.dit.forward(&xc, &Tensor::new(vec![sigma; b], self.dev)?, &rep(&self.pos)?)?;
            (den(&out)?, x.zeros_like()?)
        } else {
            let xb = Tensor::cat(&[&xc, &xc], 0)?;
            let ctx = Tensor::cat(&[rep(&self.neg)?, rep(&self.pos)?], 0)?;
            let out = self.dit.forward(&xb, &Tensor::new(vec![sigma; 2 * b], self.dev)?, &ctx)?;
            (den(&out.narrow(0, b, b)?)?, den(&out.narrow(0, 0, b)?)?)
        };
        let cfg = Tensor::new(self.cfg, self.dev)?;
        let out = (&uncond + (&cond - &uncond)?.broadcast_mul(&cfg)?)?;
        self.last_uncond = Some(uncond);
        Ok(out)
    }

    fn uncond_denoised(&self) -> Option<Tensor> {
        self.last_uncond.clone()
    }
}

impl Engine {
    /// The Anima nodes graph.rs does not dispatch itself, or None for another class
    pub(super) fn anima_node(&mut self, class: &str, inp: &serde_json::Map<String, Value>) -> Result<Option<Vec<Val>>> {
        Ok(Some(match class {
            "CLIPLoader" => {
                let name = self.text_arg(inp, "clip_name")?;
                let path = self.model_path("text_encoders", &name).or_else(|_| self.model_path("clip", &name))?;
                let st = unsafe { MmapedSafetensors::new(&path)? };
                if st.get("model.layers.27.self_attn.q_norm.weight").is_err() || st.get("model.layers.28.self_attn.q_norm.weight").is_ok() {
                    bail!(
                        "CLIPLoader: {name} is not Qwen3-0.6B (Anima's text encoder), the only text encoder rustyui loads this way so far"
                    );
                }
                let device = self.text_arg(inp, "device").unwrap_or_else(|_| "default".into());
                anyhow::ensure!(device == "default", "CLIPLoader: device {device:?} (the card, \"default\", is implemented)");
                vec![Val::AnimaClip(Rc::new(path))]
            }
            _ => return Ok(None),
        }))
    }

    /// CLIPTextEncode on Qwen3-0.6B: AnimaTokenizer's two token lists, Qwen3's states
    pub(super) fn anima_encode(&mut self, clip: &Path, text: &str) -> Result<Val> {
        let t = Instant::now();
        let tok = match &self.anima_tok {
            Some(t) => t.clone(),
            None => {
                let t = Rc::new(anima_tokens::AnimaTokenizer::new()?);
                self.anima_tok = Some(t.clone());
                t
            }
        };
        let tokens = tok.tokenize(text, &self.embs)?;
        let qwen = qwen3::Qwen3::open(clip)?.encode(&tokens.qwen, &self.dev)?.unsqueeze(0)?;
        self.dev.synchronize()?;
        self.timings.push((format!("encode Anima prompt ({} tokens)", tokens.qwen.len()), t.elapsed().as_secs_f64()));
        Ok(Val::AnimaCond(Rc::new(AnimaCond { qwen, t5: tokens.t5 })))
    }

    /// The Anima DiT for the model (its checkpoint and LoRAs), loaded once and kept on the card
    fn anima_dit(&mut self, m: &Stack) -> Result<()> {
        let key = stack_key(m, "anima");
        if self.anima.as_ref().is_some_and(|(k, _)| *k == key) {
            return Ok(());
        }
        self.anima = None;
        self.unet = None;
        self.ltx_dit = None;
        let t = Instant::now();
        let loras = if m.loras.is_empty() { None } else { Some(anima::AnimaLoras::open(&m.loras)?) };
        let dit = anima::AnimaDit::load(&m.ckpt, &self.dev, loras.as_ref())?;
        self.timings.push((
            format!("load Anima{}", if m.loras.is_empty() { String::new() } else { format!(" with {} LoRAs", m.loras.len()) }),
            t.elapsed().as_secs_f64(),
        ));
        self.anima = Some((key, dit));
        Ok(())
    }

    /// KSampler with an Anima model
    pub(super) fn anima_ksampler(&mut self, inp: &serde_json::Map<String, Value>) -> Result<Val> {
        let Val::AnimaModel(m) = self.input(inp, "model")? else { bail!("KSampler: not an Anima model") };
        let cond = |v: Val, what: &str| -> Result<Rc<AnimaCond>> {
            match v {
                Val::AnimaCond(c) => Ok(c),
                _ => bail!("KSampler: {what} is not an Anima prompt (CLIPTextEncode on Qwen3-0.6B)"),
            }
        };
        let pos = cond(self.input(inp, "positive")?, "positive")?;
        let neg = cond(self.input(inp, "negative")?, "negative")?;
        let seed = self.arg(inp, "seed")?.as_u64().context("seed is not a whole number")?;
        let steps = self.int_arg(inp, "steps")?;
        let cfg = self.float_arg(inp, "cfg")?;
        let sampler = self.text_arg(inp, "sampler_name")?;
        let scheduler = self.text_arg(inp, "scheduler")?;
        let denoise = self.float_arg(inp, "denoise")?;
        anyhow::ensure!(denoise >= 0.9999, "KSampler: denoise {denoise} on Anima (1 is implemented)");
        let latent = match self.input(inp, "latent_image")? {
            Val::Latent(z) => z,
            _ => bail!("KSampler: latent_image is not a latent"),
        };
        // fix_empty_latent_channels: an empty latent takes the model's 16 channels and a time axis
        anyhow::ensure!(latent.abs()?.sum_all()?.to_scalar::<f32>()? == 0.0, "KSampler: only an empty latent is implemented for Anima");
        let (b, h, w) = match latent.rank() {
            4 => (latent.dim(0)?, latent.dim(2)?, latent.dim(3)?),
            5 => (latent.dim(0)?, latent.dim(3)?, latent.dim(4)?),
            r => bail!("KSampler: a latent of rank {r}"),
        };
        anyhow::ensure!(b == 1, "KSampler: a batch of {b} on Anima (one is implemented)");
        let zeros = Tensor::zeros((b, 16, 1, h, w), DType::F32, &self.dev)?;
        self.clip = None;
        self.anima_dit(&m)?;
        let dev = self.dev.clone();
        let (_, dit) = self.anima.as_ref().context("the Anima DiT")?;
        let t = Instant::now();
        // Anima.extra_conds: each prompt's context, once for the sampling
        let pos_ctx = dit.context(&pos.qwen, &pos.t5)?;
        let neg_ctx = dit.context(&neg.qwen, &neg.t5)?;
        let disc = crate::sampling::Discrete::flow(SHIFT, 1.0);
        let sigmas = disc.schedule(&scheduler, steps, None)?;
        let noise = Tensor::from_vec(rng::torch_randn_cpu(seed, b * 16 * h * w), (b, 16, 1, h, w), &dev)?;
        // CONST.noise_scaling: sigma * (1.0 * noise) + (1.0 - sigma) * latent_image, on the card
        let s0 = Tensor::new(sigmas[0], &dev)?;
        let one_minus = (Tensor::new(1f32, &dev)? - &s0)?;
        let x = ((noise * 1.0)?.broadcast_mul(&s0)? + zeros.broadcast_mul(&one_minus)?)?;
        let skip_uncond = (cfg - 1.0).abs() <= 1e-9 * cfg.abs().max(1.0);
        let mut den = AnimaDenoiser { dit, pos: pos_ctx, neg: neg_ctx, cfg: cfg as f32, skip_uncond, dev: &dev, last_uncond: None };
        let mut last = Instant::now();
        let mut progress = |i: usize, n: usize| -> Result<()> {
            crate::progress::step(i, n)?;
            if i == 1 || i == n || last.elapsed().as_secs_f64() > 5.0 {
                eprintln!("[engine] step {i}/{n}  {:.1}s", t.elapsed().as_secs_f64());
                last = Instant::now();
            }
            Ok(())
        };
        let out = crate::sampling::sample_flow(&sampler, &mut den, x, &sigmas, seed, self.sms, &dev, &mut progress, SHIFT)?;
        // inverse_noise_scaling (latent / (1 - 0)), then Wan21.process_out
        let out = out.broadcast_div(&(Tensor::new(1f32, &dev)? - Tensor::new(*sigmas.last().unwrap_or(&0.0), &dev)?)?)?;
        let std = Tensor::new(&LATENT_STD, &dev)?.reshape((1, 16, 1, 1, 1))?;
        let mean = Tensor::new(&LATENT_MEAN, &dev)?.reshape((1, 16, 1, 1, 1))?;
        let out = (out.broadcast_mul(&std)? * 1.0)?.broadcast_add(&mean)?;
        dev.synchronize()?;
        let secs = t.elapsed().as_secs_f64();
        self.timings.push((format!("sample Anima {steps} steps ({:.2} s/step)", secs / steps as f64), secs));
        self.vram();
        Ok(Val::Latent(out))
    }

    /// VAEDecode of a (1, 16, 1, h, w) latent with the Wan 2.1 VAE: (1, 3, 8h, 8w) f32
    pub(super) fn wan_decode(&mut self, z: &Tensor, vae: &Path) -> Result<Tensor> {
        let t = Instant::now();
        let dec = wan_vae::WanDecoder::load(vae, &self.dev)?;
        let img = dec.decode(z)?;
        self.dev.synchronize()?;
        self.timings.push(("decode (Wan VAE)".into(), t.elapsed().as_secs_f64()));
        self.vram();
        Ok(img)
    }
}
