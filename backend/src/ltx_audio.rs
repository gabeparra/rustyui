//! LTX-2's audio VAE decoder and its vocoder with bandwidth extension (comfy/ldm/lightricks/vae/
//! audio_vae.py, causal_audio_autoencoder.py, vocoders/vocoder.py), as LTXVAudioVAEDecode runs them:
//! in f32 (the LTX audio VAE's only working dtype in ComfyUI), every convolution torch's (cuDNN's,
//! or torch's own depthwise kernel where torch takes that), each elementwise step its own rounding.

use crate::kernels;
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor, D};
use std::path::Path;

fn load(st: &MmapedSafetensors, name: &str, dev: &Device) -> Result<Tensor> {
    Ok(st.load(name, dev).with_context(|| format!("audio VAE: no {name}"))?.to_dtype(DType::F32)?)
}

/// x + b over the channels (dim 1), as F.conv* adds its bias
fn add_bias(y: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    Ok(match b {
        Some(b) => {
            let mut shape = vec![1usize; y.rank()];
            shape[1] = b.dim(0)?;
            y.broadcast_add(&b.reshape(shape)?)?
        }
        None => y.clone(),
    })
}

/// F.conv1d as torch runs it on cuDNN: (N, C, L) seen as (N, C, 1, L)
fn conv1d(x: &Tensor, w: &Tensor, b: Option<&Tensor>, padding: usize, stride: usize, dilation: usize) -> Result<Tensor> {
    let y = kernels::cudnn_conv(&x.unsqueeze(2)?, &w.unsqueeze(2)?, &[0, padding], &[1, stride], &[1, dilation])?.squeeze(2)?;
    add_bias(&y, b)
}

/// F.conv_transpose1d on cuDNN (its backward-data convolution), w (in, out / groups, k)
fn conv_transpose1d(x: &Tensor, w: &Tensor, b: Option<&Tensor>, padding: usize, stride: usize, groups: usize) -> Result<Tensor> {
    let y = kernels::cudnn_conv_transpose(&x.unsqueeze(2)?, &w.unsqueeze(2)?, &[0, padding], &[1, stride], &[1, 1], &[0, 0], groups)?
        .squeeze(2)?;
    add_bias(&y, b)
}

/// F.pad(x, (left, right), mode="replicate") on the last dim of (N, C, L)
fn replicate_pad(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let l = x.dim(D::Minus1)?;
    let mut parts = vec![];
    if left > 0 {
        parts.push(x.narrow(D::Minus1, 0, 1)?.repeat((1, 1, left))?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(D::Minus1, l - 1, 1)?.repeat((1, 1, right))?);
    }
    Ok(Tensor::cat(&parts, D::Minus1)?.contiguous()?)
}

// ── the audio autoencoder's decoder ─────────────────────────────────────────

/// CausalConv2d along the height (time): (k - 1) rows of zeros on top, half the width each side
struct CConv {
    w: Tensor,
    b: Tensor,
}

impl CConv {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self { w: load(st, &format!("{p}.conv.weight"), dev)?, b: load(st, &format!("{p}.conv.bias"), dev)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (kh, kw) = (self.w.dim(2)?, self.w.dim(3)?);
        let (ph, pw) = (kh - 1, kw - 1);
        let x = x.pad_with_zeros(3, pw / 2, pw - pw / 2)?.pad_with_zeros(2, ph, 0)?;
        add_bias(&kernels::cudnn_conv(&x, &self.w, &[0, 0], &[1, 1], &[1, 1])?, Some(&self.b))
    }
}

/// PixelNorm(dim=1, eps=1e-6): x / sqrt(mean(x ** 2 over the channels) + eps), the mean summed
/// in torch's order
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let mean = kernels::torch_mean_dim1(&x.sqr()?)?;
    Ok(x.broadcast_div(&(mean + 1e-6)?.sqrt()?)?)
}

struct ResBlock {
    conv1: CConv,
    conv2: CConv,
    nin: Option<CConv>,
}

impl ResBlock {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        let nin = if st.get(&format!("{p}.nin_shortcut.conv.weight")).is_ok() {
            Some(CConv::load(st, &format!("{p}.nin_shortcut"), dev)?)
        } else {
            None
        };
        Ok(Self { conv1: CConv::load(st, &format!("{p}.conv1"), dev)?, conv2: CConv::load(st, &format!("{p}.conv2"), dev)?, nin })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&pixel_norm(x)?.silu()?)?;
        let h = self.conv2.forward(&pixel_norm(&h)?.silu()?)?;
        let x = match &self.nin {
            Some(n) => n.forward(x)?,
            None => x.clone(),
        };
        Ok((x + h)?)
    }
}

pub struct AudioDecoder {
    conv_in: CConv,
    mid: [ResBlock; 2],
    /// up[level][block], and each level but the first's upsample conv
    up: Vec<(Vec<ResBlock>, Option<CConv>)>,
    conv_out: CConv,
    std: Tensor,
    mean: Tensor,
}

impl AudioDecoder {
    fn load(st: &MmapedSafetensors, dev: &Device) -> Result<Self> {
        let p = "audio_vae.decoder";
        let mut up = vec![];
        let mut level = 0;
        while st.get(&format!("{p}.up.{level}.block.0.conv1.conv.weight")).is_ok() {
            let mut blocks = vec![];
            let mut j = 0;
            while st.get(&format!("{p}.up.{level}.block.{j}.conv1.conv.weight")).is_ok() {
                blocks.push(ResBlock::load(st, &format!("{p}.up.{level}.block.{j}"), dev)?);
                j += 1;
            }
            let upsample = if st.get(&format!("{p}.up.{level}.upsample.conv.conv.weight")).is_ok() {
                Some(CConv::load(st, &format!("{p}.up.{level}.upsample.conv"), dev)?)
            } else {
                None
            };
            up.push((blocks, upsample));
            level += 1;
        }
        Ok(Self {
            conv_in: CConv::load(st, &format!("{p}.conv_in"), dev)?,
            mid: [ResBlock::load(st, &format!("{p}.mid.block_1"), dev)?, ResBlock::load(st, &format!("{p}.mid.block_2"), dev)?],
            up,
            conv_out: CConv::load(st, &format!("{p}.conv_out"), dev)?,
            std: load(st, "audio_vae.per_channel_statistics.std-of-means", dev)?,
            mean: load(st, "audio_vae.per_channel_statistics.mean-of-means", dev)?,
        })
    }

    /// AudioLatentNormalizer.denormalize: "b c t f -> b t (c f)", x * std + mean, and back
    fn denormalize(&self, latent: &Tensor) -> Result<Tensor> {
        let (_, c, _, f) = latent.dims4()?;
        let (std, mean) = (self.std.reshape((1, c, 1, f))?, self.mean.reshape((1, c, 1, f))?);
        Ok(latent.broadcast_mul(&std)?.broadcast_add(&mean)?)
    }

    /// Decoder.forward on a denormalized latent (B, 8, T, 16), cut or padded to (B, 2, rows, 64)
    fn forward(&self, x: &Tensor, rows: usize) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.mid {
            h = b.forward(&h)?;
        }
        for (blocks, upsample) in self.up.iter().rev() {
            for b in blocks {
                h = b.forward(&h)?;
            }
            if let Some(conv) = upsample {
                // nearest x2, the causal conv, the first row dropped
                let (_, _, hh, ww) = h.dims4()?;
                let y = conv.forward(&h.upsample_nearest2d(2 * hh, 2 * ww)?)?;
                h = y.narrow(2, 1, 2 * hh - 1)?.contiguous()?;
            }
        }
        let y = self.conv_out.forward(&pixel_norm(&h)?.silu()?)?;
        // _adjust_output_shape
        let (_, _, t, f) = y.dims4()?;
        let y = y.narrow(2, 0, t.min(rows))?.narrow(3, 0, f.min(64))?;
        let y = if t < rows { y.pad_with_zeros(2, 0, rows - t)? } else { y };
        Ok(y.contiguous()?)
    }
}

// ── the vocoder (BigVGAN v2: AMP blocks, SnakeBeta, anti-aliased resampling) ──

/// Activation1d: UpSample1d(2, 12), SnakeBeta, DownSample1d(2, 12)
struct Act1d {
    alpha: Tensor,
    beta: Tensor,
    up: Tensor,
    down: Tensor,
}

impl Act1d {
    fn load(st: &MmapedSafetensors, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            alpha: load(st, &format!("{p}.act.alpha"), dev)?,
            beta: load(st, &format!("{p}.act.beta"), dev)?,
            up: load(st, &format!("{p}.upsample.filter"), dev)?,
            down: load(st, &format!("{p}.downsample.lowpass.filter"), dev)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        // UpSample1d: replicate pad 5, a depthwise transposed conv at stride 2, * 2, 15 off each end
        let k = self.up.dim(2)?;
        let pad = k / 2 - 1;
        let (left, right) = (pad * 2 + (k - 2) / 2, pad * 2 + (k - 2).div_ceil(2));
        let w = self.up.broadcast_as((c, 1, k))?.contiguous()?;
        let y = (conv_transpose1d(&replicate_pad(x, pad, pad)?, &w, None, 0, 2, c)? * 2.0)?;
        let n = y.dim(2)?;
        let y = y.narrow(2, left, n - left - right)?.contiguous()?;
        // SnakeBeta: x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha)) ** 2
        let a = self.alpha.exp()?.reshape((1, c, 1))?;
        let inv = (self.beta.exp()? + 1e-9)?.recip()?.reshape((1, c, 1))?;
        let s = y.broadcast_mul(&a)?.sin()?.sqr()?;
        let y = (&y + s.broadcast_mul(&inv)?)?;
        // DownSample1d: replicate pad (5, 6), torch's depthwise conv at stride 2
        let k = self.down.dim(2)?;
        let y = replicate_pad(&y, k / 2 - 1, k / 2)?;
        Ok(kernels::torch_depthwise_conv1d(&y, &self.down.broadcast_as((c, 1, k))?.contiguous()?, 2)?)
    }
}

struct AmpBlock {
    convs1: Vec<(Tensor, Tensor, usize)>,
    convs2: Vec<(Tensor, Tensor)>,
    acts1: Vec<Act1d>,
    acts2: Vec<Act1d>,
}

impl AmpBlock {
    fn load(st: &MmapedSafetensors, p: &str, dilations: &[usize], dev: &Device) -> Result<Self> {
        let mut b = AmpBlock { convs1: vec![], convs2: vec![], acts1: vec![], acts2: vec![] };
        for (i, &d) in dilations.iter().enumerate() {
            b.convs1.push((load(st, &format!("{p}.convs1.{i}.weight"), dev)?, load(st, &format!("{p}.convs1.{i}.bias"), dev)?, d));
            b.convs2.push((load(st, &format!("{p}.convs2.{i}.weight"), dev)?, load(st, &format!("{p}.convs2.{i}.bias"), dev)?));
            b.acts1.push(Act1d::load(st, &format!("{p}.acts1.{i}"), dev)?);
            b.acts2.push(Act1d::load(st, &format!("{p}.acts2.{i}"), dev)?);
        }
        Ok(b)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let (w1, b1, d) = &self.convs1[i];
            let k = w1.dim(2)?;
            let xt = conv1d(&self.acts1[i].forward(&x)?, w1, Some(b1), (k * d - d) / 2, 1, *d)?;
            let (w2, b2) = &self.convs2[i];
            let xt = conv1d(&self.acts2[i].forward(&xt)?, w2, Some(b2), (k - 1) / 2, 1, 1)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

struct Vocoder {
    conv_pre: (Tensor, Tensor),
    /// ConvTranspose1d weights and biases with their stride
    ups: Vec<(Tensor, Tensor, usize)>,
    resblocks: Vec<AmpBlock>,
    act_post: Act1d,
    conv_post: (Tensor, Option<Tensor>),
    final_clamp: bool,
}

impl Vocoder {
    fn load(st: &MmapedSafetensors, p: &str, cfg: &serde_json::Value, final_clamp: bool, dev: &Device) -> Result<Self> {
        let get = |k: &str| cfg.get(k).with_context(|| format!("vocoder config has no {k}"));
        let ints = |k: &str| -> Result<Vec<usize>> {
            Ok(get(k)?.as_array().context(k.to_string())?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
        };
        if get("resblock")?.as_str() != Some("AMP1") || get("activation")?.as_str() != Some("snakebeta") {
            bail!("{p}: only the AMP1 / snakebeta vocoder is implemented");
        }
        if cfg.get("use_tanh_at_final").and_then(|v| v.as_bool()).unwrap_or(true) {
            bail!("{p}: a tanh at the end is not implemented");
        }
        let rates = ints("upsample_rates")?;
        let kernels_ = ints("resblock_kernel_sizes")?;
        let dilations: Vec<Vec<usize>> = get("resblock_dilation_sizes")?
            .as_array()
            .context("resblock_dilation_sizes")?
            .iter()
            .map(|d| d.as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect()).unwrap_or_default())
            .collect();
        let mut ups = vec![];
        let mut resblocks = vec![];
        for (i, &u) in rates.iter().enumerate() {
            ups.push((load(st, &format!("{p}.ups.{i}.weight"), dev)?, load(st, &format!("{p}.ups.{i}.bias"), dev)?, u));
            for j in 0..kernels_.len() {
                resblocks.push(AmpBlock::load(st, &format!("{p}.resblocks.{}", i * kernels_.len() + j), &dilations[j], dev)?);
            }
        }
        let post_bias =
            if st.get(&format!("{p}.conv_post.bias")).is_ok() { Some(load(st, &format!("{p}.conv_post.bias"), dev)?) } else { None };
        Ok(Self {
            conv_pre: (load(st, &format!("{p}.conv_pre.weight"), dev)?, load(st, &format!("{p}.conv_pre.bias"), dev)?),
            ups,
            resblocks,
            act_post: Act1d::load(st, &format!("{p}.act_post"), dev)?,
            conv_post: (load(st, &format!("{p}.conv_post.weight"), dev)?, post_bias),
            final_clamp,
        })
    }

    /// (B, 2, 64, T) mel (stereo) -> (B, 2, samples)
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[mel.narrow(1, 0, 1)?.squeeze(1)?, mel.narrow(1, 1, 1)?.squeeze(1)?], 1)?;
        let mut x = conv1d(&x, &self.conv_pre.0, Some(&self.conv_pre.1), 3, 1, 1)?;
        let per = self.resblocks.len() / self.ups.len();
        for (i, (w, b, u)) in self.ups.iter().enumerate() {
            let k = w.dim(2)?;
            x = conv_transpose1d(&x, w, Some(b), (k - u) / 2, *u, 1)?;
            let mut xs = self.resblocks[i * per].forward(&x)?;
            for j in 1..per {
                xs = (xs + self.resblocks[i * per + j].forward(&x)?)?;
            }
            // xs / num_kernels: torch divides by a scalar as a multiply by its f32 reciprocal
            x = xs.affine((1.0f32 / per as f32) as f64, 0.0)?;
        }
        let x = conv1d(&self.act_post.forward(&x)?, &self.conv_post.0, self.conv_post.1.as_ref(), 3, 1, 1)?;
        Ok(if self.final_clamp { x.clamp(-1f32, 1f32)? } else { x })
    }
}

/// UpSample1d(3, window_type="hann")'s filter, as torch computes it at load (43 taps): kept as
/// ComfyUI's own f32 values (read off its loaded VAE), since torch's CPU sinc and cos round their
/// own way (the two halves are not even mirror images)
const HANN_X3: [u32; 43] = [
    0x22e91509, 0x24056011, 0x23dd7688, 0xb55b7488, 0xb93fe83f, 0xba0d1795, 0x39803c88, 0x3b2d1999, 0x3b6d95cb, 0xba61d448, 0xbc131632,
    0xbc31f525, 0x3add8e6c, 0x3caf188d, 0x3cd08081, 0xbb24b7fe, 0xbd3f9d77, 0xbd73a141, 0x3b4c1200, 0x3e0a9821, 0x3e8b3c14, 0x3ea8f5c3,
    0x3e8b3c14, 0x3e0a9821, 0x3b4c1200, 0xbd73a139, 0xbd3f9d69, 0xbb24b7fe, 0x3cd08077, 0x3caf187f, 0x3add8e6c, 0xbc31f523, 0xbc13162d,
    0xba61d448, 0x3b6d95cb, 0x3b2d1999, 0x39803c88, 0xba0d1791, 0xb93fe83f, 0xb55b7488, 0x23dd7688, 0x2405600b, 0x22e91509,
];

pub struct AudioVae {
    decoder: AudioDecoder,
    vocoder: Vocoder,
    bwe: Vocoder,
    forward_basis: Tensor,
    mel_basis: Tensor,
    resampler: Tensor,
    bwe_hop: usize,
    in_rate: usize,
    pub out_rate: usize,
}

impl AudioVae {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path)? };
        let meta = crate::ltx_upsampler::metadata(path)?;
        let cfg: serde_json::Value =
            serde_json::from_str(meta.get("config").and_then(|c| c.as_str()).context("the audio VAE has no config in its metadata")?)?;
        let voc = cfg.get("vocoder").context("no vocoder config")?;
        let (vcfg, bcfg) = (
            voc.get("vocoder").context("no vocoder.vocoder config")?,
            voc.get("bwe").context("only the vocoder with bandwidth extension is implemented")?,
        );
        let n = |k: &str| bcfg.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).with_context(|| format!("bwe config has no {k}"));
        let resampler = Tensor::from_vec(HANN_X3.iter().map(|&b| f32::from_bits(b)).collect::<Vec<f32>>(), (1, 1, 43), dev)?;
        Ok(Self {
            decoder: AudioDecoder::load(&st, dev)?,
            vocoder: Vocoder::load(&st, "vocoder.vocoder", vcfg, true, dev)?,
            bwe: Vocoder::load(&st, "vocoder.bwe_generator", bcfg, false, dev)?,
            forward_basis: load(&st, "vocoder.mel_stft.stft_fn.forward_basis", dev)?,
            mel_basis: load(&st, "vocoder.mel_stft.mel_basis", dev)?,
            resampler,
            bwe_hop: n("hop_length")?,
            in_rate: n("input_sampling_rate")?,
            out_rate: n("output_sampling_rate")?,
        })
    }

    /// MelSTFT.mel_spectrogram's log mel for (B*C, T) waveforms: causal STFT as a strided conv with
    /// the stored bases, magnitude, the mel basis, log of at least 1e-5
    fn log_mel(&self, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let n_fft = self.forward_basis.dim(2)?;
        let y = y.unsqueeze(1)?.pad_with_zeros(2, n_fft - self.bwe_hop, 0)?;
        let spec = conv1d(&y, &self.forward_basis, None, 0, self.bwe_hop, 1)?;
        let nf = spec.dim(1)? / 2;
        let (re, im) = (spec.narrow(1, 0, nf)?, spec.narrow(1, nf, nf)?);
        let mag = (re.sqr()? + im.sqr()?)?.sqrt()?;
        let mel = self.mel_basis.broadcast_matmul(&mag)?;
        Ok((mel.clamp(1e-5f32, f32::MAX)?.log()?, mag))
    }

    /// The UpSample1d(3, hann) of VocoderWithBWE's skip path
    fn resample(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let (k, ratio) = (self.resampler.dim(2)?, 3usize);
        let width = (k - 1) / (2 * ratio);
        let (left, right) = (2 * width * ratio, k - ratio);
        let w = self.resampler.broadcast_as((c, 1, k))?.contiguous()?;
        let y = (conv_transpose1d(&replicate_pad(x, width, width)?, &w, None, 0, ratio, c)? * ratio as f64)?;
        let n = y.dim(2)?;
        Ok(y.narrow(2, left, n - left - right)?.contiguous()?)
    }

    /// AudioVAE.decode: a normalized latent (B, 8, T, 16) to (B, 2, samples) at `out_rate`
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let rows = latent.dim(2)? * 4 - 3;
        let mel = self.decoder.forward(&self.decoder.denormalize(latent)?, rows)?;
        let x = self.vocoder.forward(&mel.transpose(2, 3)?.contiguous()?)?;
        let (b, c, t_low) = x.dims3()?;
        let t_out = t_low * self.out_rate / self.in_rate;
        let rem = t_low % self.bwe_hop;
        let x = if rem != 0 { x.pad_with_zeros(2, 0, self.bwe_hop - rem)? } else { x };
        let (mel, _) = self.log_mel(&x.reshape((b * c, ()))?)?;
        let mel = mel.reshape((b, c, mel.dim(1)?, mel.dim(2)?))?;
        let residual = self.bwe.forward(&mel)?;
        let skip = self.resample(&x)?;
        Ok((residual + skip)?.clamp(-1f32, 1f32)?.narrow(2, 0, t_out)?.contiguous()?)
    }
}

/// debug-audio REF: the audio decode against ComfyUI's (refs/audio_ref.py), stage by stage
pub fn debug_audio(reference: &Path, vae: &Path) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let r = unsafe { MmapedSafetensors::new(reference)? };
    let get = |n: &str| -> Result<Tensor> { r.load(n, &dev).with_context(|| format!("reference has no {n}")) };
    let cmp = |what: &str, ours: &Tensor, key: &str| -> Result<()> { crate::ltx::compare(what, ours, &get(key)?) };
    let a = AudioVae::load(vae, &dev)?;
    let d = &a.decoder;
    let x = d.denormalize(&get("latent")?)?;
    cmp("denormalized latent", &x, "decoder.in")?;
    let h = d.conv_in.forward(&x)?;
    cmp("decoder conv_in", &h, "decoder.conv_in")?;
    cmp("decoder mid block 1 (their input)", &d.mid[0].forward(&get("decoder.mid1.in")?)?, "decoder.mid1")?;
    let (blocks, up) = &d.up[2];
    let _ = blocks;
    if let Some(conv) = up {
        let u = get("decoder.up2.upsample.in")?;
        let (_, _, hh, ww) = u.dims4()?;
        let y = conv.forward(&u.upsample_nearest2d(2 * hh, 2 * ww)?)?;
        cmp("decoder level 2 upsample (their input)", &y.narrow(2, 1, 2 * hh - 1)?.contiguous()?, "decoder.up2.upsample")?;
    }
    let mel = d.forward(&x, get("latent")?.dim(2)? * 4 - 3)?;
    cmp("decoder (the mel)", &mel, "decoder")?;
    let v = &a.vocoder;
    let vin = get("voc.in")?;
    cmp("vocoder input", &mel.transpose(2, 3)?.contiguous()?, "voc.in")?;
    let pre = get("voc.conv_pre.in")?;
    cmp("vocoder conv_pre (their input)", &conv1d(&pre, &v.conv_pre.0, Some(&v.conv_pre.1), 3, 1, 1)?, "voc.conv_pre")?;
    let (w, b, u) = &v.ups[0];
    let k = w.dim(2)?;
    cmp("vocoder ups.0 (their input)", &conv_transpose1d(&get("voc.ups0.in")?, w, Some(b), (k - u) / 2, *u, 1)?, "voc.ups0")?;
    let act = &v.resblocks[0].acts1[0];
    cmp("anti-aliased SnakeBeta (their input)", &act.forward(&get("voc.res0.act0.in")?)?, "voc.res0.act0")?;
    cmp("AMP block 0 (their input)", &v.resblocks[0].forward(&get("voc.res0.in")?)?, "voc.res0")?;
    let wave = v.forward(&vin)?;
    cmp("vocoder (16 kHz, from their mel)", &wave, "voc")?;
    let (lm, mag) = a.log_mel(&get("bwe.mel_in")?)?;
    cmp("BWE magnitude (their waveform)", &mag, "bwe.magnitude")?;
    cmp("BWE log mel (their waveform)", &lm, "bwe.log_mel")?;
    cmp("BWE generator (their mel)", &a.bwe.forward(&get("bwe.in")?)?, "bwe")?;
    cmp("BWE skip (their waveform)", &a.resample(&get("bwe.skip.in")?)?, "bwe.skip")?;
    cmp("waveform (the whole decode)", &a.decode(&get("latent")?)?, "waveform")?;
    Ok(())
}
