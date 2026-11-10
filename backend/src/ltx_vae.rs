//! LTX-2.3's video VAE (LTX-2.3's `ltx-2.3-video-vae.safetensors`) as ComfyUI's
//! comfy/ldm/lightricks/vae/causal_video_autoencoder.py runs it, streaming. The decoder splits each
//! up block's output into time chunks of at most get_max_chunk_size bytes and runs the rest of the
//! network on one chunk at a time: every CausalConv3d keeps the last two frames it was given for the
//! next chunk, and every residual add keeps the inputs whose outputs have not come out yet. The
//! encoder runs the first frame alone, then chunks of frames. The chunks are ComfyUI's, so every
//! convolution sees the shapes torch's sees (cuDNN picks its engine by shape, kernels::cudnn_conv
//! picks it the way torch does) and every PixelNorm mean is laid out as torch lays it out, and the
//! frames and latents come out as ComfyUI's do, bit for bit.
//!
//! This file's config only: pixel norm, zero spatial padding, no timestep conditioning or injected
//! noise, patch 4, the non-causal decoder, the "uniform" log variance. Tensors are (1, C, T, H, W)
//! bf16 as ComfyUI's are, and `None` stands for a piece with no frames.

use crate::kernels;
use anyhow::{bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor, D};
use serde_json::Value;
use std::path::Path;

/// get_max_chunk_size: 32 MiB on a card of 6 GiB or less, 128 MiB from 24 GiB, linear between, from
/// the card's total memory as cudaMemGetInfo reports it (as comfy.model_management does)
fn max_chunk_size(dev: &Device) -> Result<usize> {
    let total = match dev {
        Device::Cuda(d) => d.cuda_stream().context().mem_get_info().map_err(|e| anyhow::anyhow!("VRAM: {e:?}"))?.1 as f64,
        _ => bail!("the video VAE runs on CUDA"),
    };
    const GIB: f64 = (1u64 << 30) as f64;
    let (min_v, max_v) = (6.0 * GIB, 24.0 * GIB);
    let (min_c, max_c) = ((32usize << 20) as f64, (128usize << 20) as f64);
    if total <= min_v {
        return Ok(min_c as usize);
    }
    if total >= max_v {
        return Ok(max_c as usize);
    }
    let interp = (total - min_v) / (max_v - min_v);
    Ok((min_c + interp * (max_c - min_c)) as usize)
}

fn frames_of(x: &Option<Tensor>) -> Result<usize> {
    Ok(match x {
        Some(x) => x.dim(2)?,
        None => 0,
    })
}

/// frames [from, from + len) of a piece, contiguous; None when that is no frames
fn frames(x: &Tensor, from: usize, len: usize) -> Result<Option<Tensor>> {
    Ok(if len == 0 { None } else { Some(x.narrow(2, from, len)?.contiguous()?) })
}

/// torch_cat_if_needed along time: the pieces with frames, joined
fn cat_frames(pieces: Vec<Option<Tensor>>) -> Result<Option<Tensor>> {
    let mut v: Vec<Tensor> = pieces.into_iter().flatten().collect();
    Ok(match v.len() {
        0 => None,
        1 => v.pop(),
        _ => Some(Tensor::cat(&v, 2)?),
    })
}

/// add_exchange_cache: the residual add of a block whose output lags its input. The inputs held
/// from earlier chunks, then the new ones, are added onto the output's frames in order; returns the
/// output and the inputs not added yet, the next chunk's cache.
fn add_exchange_cache(dest: Option<Tensor>, cache: Option<Tensor>, input: Option<Tensor>) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let Some(dest) = dest else {
        return Ok((None, cat_frames(vec![cache, input])?));
    };
    let td = dest.dim(2)?;
    let mut parts = Vec::new();
    let mut at = 0;
    let mut cache = cache;
    if let Some(c) = cache.take() {
        let tc = c.dim(2)?;
        let k = td.min(tc);
        parts.push((dest.narrow(2, 0, k)? + c.narrow(2, 0, k)?)?);
        cache = frames(&c, k, tc - k)?;
        at = k;
    }
    let mut input = input;
    let rest = td - at;
    if rest > 0 {
        let i = input.take().context("a residual add with no input frames for its output")?;
        let ti = i.dim(2)?;
        if ti < rest {
            bail!("a residual add with {ti} input frames for {rest} output frames");
        }
        parts.push((dest.narrow(2, at, rest)? + i.narrow(2, 0, rest)?)?);
        input = frames(&i, rest, ti - rest)?;
    }
    let out = if parts.len() == 1 { parts.pop().context("no frames")? } else { Tensor::cat(&parts, 2)? };
    Ok((Some(out), cat_frames(vec![cache, input])?))
}

/// What a decode or an encode remembers between chunks, per module (ComfyUI's
/// temporal_cache_state), indexed by the ids handed out at load
#[derive(Default)]
struct Run {
    convs: Vec<ConvState>,
    res: Vec<Option<Tensor>>,
    drop_first: Vec<bool>,
    s2d: Vec<S2dState>,
}

#[derive(Default)]
struct ConvState {
    /// the last two frames given (None before the first chunk, and after the last)
    cache: Option<Tensor>,
    is_end: bool,
}

#[derive(Default)]
struct S2dState {
    cached: Option<Tensor>,
    padded: bool,
    cached_x: Option<Tensor>,
    cached_input: Option<Tensor>,
}

#[derive(Default, Clone, Copy)]
struct Ids {
    convs: usize,
    res: usize,
    ups: usize,
    s2d: usize,
}

impl Run {
    fn new(n: Ids) -> Self {
        Run {
            convs: (0..n.convs).map(|_| ConvState::default()).collect(),
            res: vec![None; n.res],
            drop_first: vec![true; n.ups],
            s2d: (0..n.s2d).map(|_| S2dState::default()).collect(),
        }
    }
}

/// A CausalConv3d: 3x3x3, stride 1, the time axis padded from its own frames, the spatial axes
/// with zeros; torch's cuDNN convolution, then the bias added in bf16 (F.conv3d's order)
struct Conv {
    w: Tensor,
    b: Tensor,
    id: usize,
}

impl Conv {
    fn load(st: &MmapedSafetensors, dev: &Device, p: &str, ids: &mut Ids) -> Result<Self> {
        let w = st.load(&format!("{p}.weight"), dev)?.to_dtype(DType::BF16)?;
        let (out, _, kt, kh, kw) = w.dims5()?;
        if (kt, kh, kw) != (3, 3, 3) {
            bail!("{p}: a {kt}x{kh}x{kw} kernel, not 3x3x3");
        }
        let b = st.load(&format!("{p}.bias"), dev)?.to_dtype(DType::BF16)?.reshape((1, out, 1, 1, 1))?;
        ids.convs += 1;
        Ok(Conv { w, b, id: ids.convs - 1 })
    }

    fn channels(&self) -> Result<(usize, usize)> {
        let (o, i, _, _, _) = self.w.dims5()?;
        Ok((i, o))
    }

    /// CausalConv3d.forward: the frames cached from the last chunk (or, the first time, the first
    /// frame repeated: twice when causal, once when not) before x, and when this is the last chunk
    /// of a non-causal run its last frame once after; the last two frames kept for the next chunk.
    fn forward(&self, run: &mut Run, x: Option<Tensor>, causal: bool) -> Result<Option<Tensor>> {
        let s = &mut run.convs[self.id];
        let t = frames_of(&x)?;
        let cached = match s.cache.take() {
            Some(c) => c,
            None => {
                // the first chunk; with no frames, nothing happens yet
                let Some(x) = x.as_ref() else { return Ok(None) };
                let first = x.narrow(2, 0, 1)?;
                Tensor::cat(&vec![&first; if causal { 2 } else { 1 }], 2)?
            }
        };
        let mut pieces = vec![cached];
        if let Some(x) = &x {
            pieces.push(x.clone());
            if s.is_end && !causal {
                pieces.push(x.narrow(2, t - 1, 1)?);
            }
        }
        let mut needs_caching = !s.is_end;
        if needs_caching && t >= 2 {
            needs_caching = false;
            s.cache = frames(x.as_ref().context("frames")?, t - 2, 2)?;
        }
        let cat = Tensor::cat(&pieces, 2)?;
        drop(pieces);
        let tc = cat.dim(2)?;
        if needs_caching {
            s.cache = frames(&cat, tc - 2, 2)?;
        } else if s.is_end {
            s.cache = None;
        }
        if tc < 3 {
            return Ok(None);
        }
        let y = kernels::cudnn_conv(&cat, &self.w, &[0, 1, 1], &[1, 1, 1], &[1, 1, 1])?;
        Ok(Some(y.broadcast_add(&self.b)?))
    }
}

/// ResnetBlock3D with as many channels out as in: PixelNorm, SiLU, conv, PixelNorm, SiLU, conv,
/// plus the input (add_exchange_cache, since in a non-causal run the output lags the input)
struct Res {
    conv1: Conv,
    conv2: Conv,
    id: usize,
}

impl Res {
    fn load(st: &MmapedSafetensors, dev: &Device, p: &str, ids: &mut Ids) -> Result<Self> {
        let conv1 = Conv::load(st, dev, &format!("{p}.conv1.conv"), ids)?;
        let conv2 = Conv::load(st, dev, &format!("{p}.conv2.conv"), ids)?;
        ids.res += 1;
        Ok(Res { conv1, conv2, id: ids.res - 1 })
    }

    fn forward(&self, run: &mut Run, x: Option<Tensor>, causal: bool) -> Result<Option<Tensor>> {
        let h = x.as_ref().map(|x| kernels::ltx_pixel_norm(x, true)).transpose()?;
        let h = self.conv1.forward(run, h, causal)?;
        let h = h.as_ref().map(|h| kernels::ltx_pixel_norm(h, true)).transpose()?;
        let h = self.conv2.forward(run, h, causal)?;
        let cache = run.res[self.id].take();
        let (out, cache) = add_exchange_cache(h, cache, x)?;
        run.res[self.id] = cache;
        Ok(out)
    }
}

/// "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
fn depth_to_space(x: &Tensor, (p1, p2, p3): (usize, usize, usize)) -> Result<Tensor> {
    let (b, cp, d, h, w) = x.dims5()?;
    let c = cp / (p1 * p2 * p3);
    Ok(x.reshape(vec![b * c, p1, p2, p3, d, h, w])?.permute(vec![0, 4, 1, 5, 2, 6, 3])?.contiguous()?.reshape((
        b,
        c,
        d * p1,
        h * p2,
        w * p3,
    ))?)
}

/// "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"
fn space_to_depth(x: &Tensor, (p1, p2, p3): (usize, usize, usize)) -> Result<Tensor> {
    let (b, c, d, h, w) = x.dims5()?;
    let (d, h, w) = (d / p1, h / p2, w / p3);
    Ok(x.reshape(vec![b * c, d, p1, h, p2, w, p3])?.permute(vec![0, 2, 4, 6, 1, 3, 5])?.contiguous()?.reshape((
        b,
        c * p1 * p2 * p3,
        d,
        h,
        w,
    ))?)
}

/// "b c (f p) (h q) (w r) -> b (c p r q) f h w" with p = 1: note r (the width offset) before q
fn patchify(x: &Tensor, patch: usize) -> Result<Tensor> {
    let (b, c, f, h, w) = x.dims5()?;
    let (h, w) = (h / patch, w / patch);
    Ok(x.reshape(vec![b * c, f, h, patch, w, patch])?.permute(vec![0, 5, 3, 1, 2, 4])?.contiguous()?.reshape((
        b,
        c * patch * patch,
        f,
        h,
        w,
    ))?)
}

/// "b (c p r q) f h w -> b c (f p) (h q) (w r)" with p = 1
fn unpatchify(x: &Tensor, patch: usize) -> Result<Tensor> {
    let (b, cp, f, h, w) = x.dims5()?;
    let c = cp / (patch * patch);
    Ok(x.reshape(vec![b * c, patch, patch, f, h, w])?.permute(vec![0, 3, 4, 2, 5, 1])?.contiguous()?.reshape((
        b,
        c,
        f,
        h * patch,
        w * patch,
    ))?)
}

/// DepthToSpaceUpsample without its residual: the conv to (channels x stride), depth to space,
/// and when time doubles the very first frame dropped (once per run)
struct Up {
    conv: Conv,
    stride: (usize, usize, usize),
    id: usize,
}

impl Up {
    fn forward(&self, run: &mut Run, x: Option<Tensor>, causal: bool) -> Result<Option<Tensor>> {
        let Some(y) = self.conv.forward(run, x, causal)? else { return Ok(None) };
        let y = depth_to_space(&y, self.stride)?;
        if self.stride.0 == 2 && run.drop_first[self.id] {
            run.drop_first[self.id] = false;
            let t = y.dim(2)?;
            return frames(&y, 1, t - 1);
        }
        Ok(Some(y))
    }
}

/// SpaceToDepthDownsample: the conv's output and the input both space-to-depth, the input's
/// channels averaged in groups to match (torch's mean, kernels::torch_mean), and added; when time
/// halves, the first frame doubled first, and a lone frame kept until it has a partner
struct S2d {
    conv: Conv,
    stride: (usize, usize, usize),
    group: usize,
    id: usize,
}

impl S2d {
    fn shortcut(&self, x: &Tensor) -> Result<Tensor> {
        let x = space_to_depth(x, self.stride)?;
        if self.group == 1 {
            return Ok(x);
        }
        let (b, c, d, h, w) = x.dims5()?;
        Ok(kernels::torch_mean(&x.reshape(vec![b, c / self.group, self.group, d, h, w])?, 2)?.squeeze(2)?)
    }

    fn forward(&self, run: &mut Run, x: Option<Tensor>, causal: bool) -> Result<Option<Tensor>> {
        let mut s = std::mem::take(&mut run.s2d[self.id]);
        let mut x = cat_frames(vec![s.cached_input.take(), x])?.context("a downsample given no frames")?;
        if self.stride.0 == 2 && !s.padded {
            x = Tensor::cat(&[&x.narrow(2, 0, 1)?, &x], 2)?;
            s.padded = true;
        }
        if x.dim(2)? < self.stride.0 {
            s.cached_input = Some(x);
            run.s2d[self.id] = s;
            return Ok(None);
        }
        let x_in = self.shortcut(&x)?;
        let mut y = self.conv.forward(run, Some(x), causal)?;
        if self.stride.0 == 2 && frames_of(&y)? == 1 {
            if s.cached_x.is_some() {
                y = cat_frames(vec![s.cached_x.take(), y])?;
            } else {
                s.cached_x = y.take();
            }
        }
        let y = y.map(|y| space_to_depth(&y, self.stride)).transpose()?;
        let (y, cached) = add_exchange_cache(y, s.cached.take(), Some(x_in))?;
        s.cached = cached;
        run.s2d[self.id] = s;
        Ok(y)
    }
}

/// UNetMidBlock3D (res_x) or an up/down sampler
enum Block {
    Res(Vec<Res>),
    Up(Up),
    Down(S2d),
}

impl Block {
    fn forward(&self, run: &mut Run, x: Option<Tensor>, causal: bool) -> Result<Option<Tensor>> {
        match self {
            Block::Res(blocks) => {
                let mut x = x;
                for r in blocks {
                    x = r.forward(run, x, causal)?;
                }
                Ok(x)
            }
            Block::Up(u) => u.forward(run, x, causal),
            Block::Down(d) => d.forward(run, x, causal),
        }
    }

    /// mark_conv3d_ended: every convolution in the block takes this chunk as its last
    fn mark_ended(&self, run: &mut Run) {
        let mut end = |c: &Conv| run.convs[c.id].is_end = true;
        match self {
            Block::Res(blocks) => blocks.iter().for_each(|r| {
                end(&r.conv1);
                end(&r.conv2);
            }),
            Block::Up(u) => end(&u.conv),
            Block::Down(d) => end(&d.conv),
        }
    }
}

/// The VAE file's config and per-channel statistics
struct Common {
    cfg: Value,
    mean: Tensor,
    std: Tensor,
    patch: usize,
}

fn load_common(st: &MmapedSafetensors, path: &Path, dev: &Device) -> Result<Common> {
    let header: Value = {
        let raw = std::fs::read(path)?;
        let n = u64::from_le_bytes(raw[..8].try_into()?) as usize;
        serde_json::from_slice(&raw[8..8 + n])?
    };
    let config: Value = serde_json::from_str(header["__metadata__"]["config"].as_str().context("no config in the VAE file")?)?;
    let cfg = if config["vae"].is_object() { config["vae"].clone() } else { config };
    if cfg["norm_layer"].as_str() != Some("pixel_norm") {
        bail!("this VAE's norm is {:?}; rustyui's handles pixel_norm only", cfg["norm_layer"]);
    }
    let padding = cfg["spatial_padding_mode"].as_str().unwrap_or("zeros");
    if padding != "zeros" {
        bail!("this VAE pads with {padding:?}; rustyui's handles zeros only");
    }
    if cfg["timestep_conditioning"].as_bool().unwrap_or(false) {
        bail!("this VAE has timestep_conditioning; rustyui's handles it without only");
    }
    Ok(Common {
        mean: st.load("per_channel_statistics.mean-of-means", dev)?.to_dtype(DType::BF16)?,
        std: st.load("per_channel_statistics.std-of-means", dev)?.to_dtype(DType::BF16)?,
        patch: cfg["patch_size"].as_u64().unwrap_or(1) as usize,
        cfg,
    })
}

pub struct VideoDecoder {
    conv_in: Conv,
    blocks: Vec<Block>,
    conv_out: Conv,
    common: Common,
    ids: Ids,
}

impl VideoDecoder {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }.with_context(|| format!("open {}", path.display()))?;
        let common = load_common(&st, path, dev)?;
        let cfg = &common.cfg;
        if cfg["causal_decoder"].as_bool().unwrap_or(false) {
            bail!("this VAE has a causal decoder; rustyui's handles the non-causal one only");
        }
        let mut ids = Ids::default();
        let conv_in = Conv::load(&st, dev, "decoder.conv_in.conv", &mut ids)?;
        let mut blocks = vec![];
        for (i, b) in cfg["decoder_blocks"].as_array().context("no decoder_blocks in the VAE config")?.iter().rev().enumerate() {
            let (name, params) = (b[0].as_str().unwrap_or(""), &b[1]);
            let p = format!("decoder.up_blocks.{i}");
            blocks.push(match name {
                "res_x" => {
                    if params["inject_noise"].as_bool().unwrap_or(false) {
                        bail!("{p}: injected noise is not handled");
                    }
                    let n = params["num_layers"].as_u64().unwrap_or(1) as usize;
                    Block::Res((0..n).map(|j| Res::load(&st, dev, &format!("{p}.res_blocks.{j}"), &mut ids)).collect::<Result<_>>()?)
                }
                "compress_time" | "compress_space" | "compress_all" => {
                    if params["residual"].as_bool().unwrap_or(false) {
                        bail!("{p}: a residual upsample is not handled");
                    }
                    let stride = match name {
                        "compress_time" => (2, 1, 1),
                        "compress_space" => (1, 2, 2),
                        _ => (2, 2, 2),
                    };
                    let conv = Conv::load(&st, dev, &format!("{p}.conv.conv"), &mut ids)?;
                    ids.ups += 1;
                    Block::Up(Up { conv, stride, id: ids.ups - 1 })
                }
                other => bail!("{p}: decoder block {other:?} is not handled"),
            });
        }
        let conv_out = Conv::load(&st, dev, "decoder.conv_out.conv", &mut ids)?;
        Ok(VideoDecoder { conv_in, blocks, conv_out, common, ids })
    }

    /// latent (1, 128, F, H, W) -> frames (1 + 8(F-1), 32H, 32W, 3) in [0, 1], f32 (ComfyUI's
    /// IMAGE: the decoder's bf16 into an f32 buffer, then (x + 1) / 2 clamped)
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let dev = latent.device().clone();
        let mut run = Run::new(self.ids);
        // un_normalize in the VAE's dtype
        let x = kernels::ltx_channel_affine(&latent.to_dtype(DType::BF16)?, &self.common.std, &self.common.mean, false)?;
        // Decoder.forward_orig: conv_in on the whole latent, as its last chunk
        run.convs[self.conv_in.id].is_end = true;
        let x = self.conv_in.forward(&mut run, Some(x), false)?.context("the latent has no frames")?;
        let mut out = vec![];
        self.run_up(&mut run, 0, x, true, &mut out, max_chunk_size(&dev)?)?;
        let frames = Tensor::cat(&out, 0)?;
        let want = 1 + 8 * (latent.dim(2)? - 1);
        if frames.dim(0)? != want {
            bail!("the decoder gave {} frames for {want}", frames.dim(0)?);
        }
        Ok(frames)
    }

    /// Decoder.run_up: up block `idx` on this chunk, its output split into chunks of at most
    /// `max_chunk` bytes (torch.chunk's sizes) for the blocks after it; past the last block the
    /// final norm and conv, the frames out
    fn run_up(&self, run: &mut Run, idx: usize, sample: Tensor, ended: bool, out: &mut Vec<Tensor>, max_chunk: usize) -> Result<()> {
        let Some(block) = self.blocks.get(idx) else {
            let x = kernels::ltx_pixel_norm(&sample, true)?;
            drop(sample);
            if ended {
                run.convs[self.conv_out.id].is_end = true;
            }
            if let Some(y) = self.conv_out.forward(run, Some(x), false)? {
                let y = unpatchify(&y, self.common.patch)?;
                let y = ((y.to_dtype(DType::F32)? + 1.0)? / 2.0)?.clamp(0f32, 1f32)?;
                out.push(y.squeeze(0)?.permute((1, 2, 3, 0))?.contiguous()?);
            }
            return Ok(());
        };
        if ended {
            block.mark_ended(run);
        }
        let Some(y) = block.forward(run, Some(sample), false)? else { return Ok(()) };
        let chunks = (y.elem_count() * y.dtype().size_in_bytes()).div_ceil(max_chunk);
        if chunks <= 1 {
            return self.run_up(run, idx + 1, y, ended, out, max_chunk);
        }
        let t = y.dim(2)?;
        let size = t.div_ceil(chunks);
        let count = t.div_ceil(size);
        for i in 0..count {
            let piece = y.narrow(2, i * size, size.min(t - i * size))?.contiguous()?;
            self.run_up(run, idx + 1, piece, ended && i == count - 1, out, max_chunk)?;
        }
        Ok(())
    }
}

/// The VAE's encoder (causal throughout), then the per-channel normalization of the means
pub struct VideoEncoder {
    conv_in: Conv,
    blocks: Vec<Block>,
    conv_out: Conv,
    common: Common,
    ids: Ids,
}

impl VideoEncoder {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }.with_context(|| format!("open {}", path.display()))?;
        let common = load_common(&st, path, dev)?;
        let cfg = &common.cfg;
        if cfg["latent_log_var"].as_str().unwrap_or("per_channel") != "uniform" {
            bail!("this VAE's latent_log_var is {:?}; rustyui's handles uniform only", cfg["latent_log_var"]);
        }
        let mut ids = Ids::default();
        let conv_in = Conv::load(&st, dev, "encoder.conv_in.conv", &mut ids)?;
        let mut channels = cfg["encoder_base_channels"].as_u64().unwrap_or(128) as usize;
        let mut blocks = vec![];
        for (i, b) in cfg["encoder_blocks"].as_array().context("no encoder_blocks in the VAE config")?.iter().enumerate() {
            let (name, params) = (b[0].as_str().unwrap_or(""), &b[1]);
            let p = format!("encoder.down_blocks.{i}");
            blocks.push(match name {
                "res_x" => {
                    let n = params["num_layers"].as_u64().unwrap_or(1) as usize;
                    Block::Res((0..n).map(|j| Res::load(&st, dev, &format!("{p}.res_blocks.{j}"), &mut ids)).collect::<Result<_>>()?)
                }
                "compress_space_res" | "compress_time_res" | "compress_all_res" => {
                    let stride = match name {
                        "compress_space_res" => (1, 2, 2),
                        "compress_time_res" => (2, 1, 1),
                        _ => (2, 2, 2),
                    };
                    let out = params["multiplier"].as_u64().unwrap_or(2) as usize * channels;
                    let group = channels * stride.0 * stride.1 * stride.2 / out;
                    let conv = Conv::load(&st, dev, &format!("{p}.conv.conv"), &mut ids)?;
                    ids.s2d += 1;
                    channels = out;
                    Block::Down(S2d { conv, stride, group, id: ids.s2d - 1 })
                }
                other => bail!("{p}: encoder block {other:?} is not handled"),
            });
        }
        let conv_out = Conv::load(&st, dev, "encoder.conv_out.conv", &mut ids)?;
        Ok(VideoEncoder { conv_in, blocks, conv_out, common, ids })
    }

    /// frames (T, H, W, 3) in [0, 1] -> normalized latent (1, 128, 1 + (T-1)/8, H/32, W/32), f32
    pub fn encode(&self, frames: &Tensor) -> Result<Tensor> {
        let dev = frames.device().clone();
        let (t, h, w, _) = frames.dims4()?;
        // vae_encode_crop_pixels: the picture cropped to whole multiples of 32, centered
        let (ch, cw) = (h / 32 * 32, w / 32 * 32);
        let frames = frames.narrow(1, (h % 32) / 2, ch)?.narrow(2, (w % 32) / 2, cw)?;
        // process_input in f32, then the VAE's dtype, (1, 3, T, H, W); VideoVAE.encode keeps 1 + 8n
        let x = frames.permute((3, 0, 1, 2))?.contiguous()?.to_dtype(DType::F32)?;
        let x = ((x * 2.0)? - 1.0)?.to_dtype(DType::BF16)?.unsqueeze(0)?;
        let keep = 1 + (t.max(1) - 1) / 8 * 8;
        // Encoder.forward_orig's chunks: the first frame, then as many frames as fit twice the
        // chunk budget at conv_in's width (2, 4 or a multiple of 8)
        let (cin, cout) = self.conv_in.channels()?;
        let frame_size = ((3 * ch * cw * 2) as f64 * (cout as f64 / cin as f64)) as usize;
        let mut chunks = vec![(0, 1)];
        if keep > 1 {
            let ct = (max_chunk_size(&dev)? * 2 / frame_size).max(2);
            let ct = if ct < 4 {
                2
            } else if ct < 8 {
                4
            } else {
                ct / 8 * 8
            };
            let mut at = 1;
            while at < keep {
                let len = ct.min(keep - at);
                chunks.push((at, len));
                at += len;
            }
        }
        let mut run = Run::new(self.ids);
        let mut outs = vec![];
        for (i, &(at, len)) in chunks.iter().enumerate() {
            if i == chunks.len() - 1 {
                // mark_conv3d_ended(self): every convolution in the encoder
                run.convs.iter_mut().for_each(|c| c.is_end = true);
            }
            let chunk = patchify(&x.narrow(2, at, len)?.contiguous()?, self.common.patch)?;
            outs.push(self.forward_chunk(&mut run, chunk)?);
        }
        let out = cat_frames(outs)?.context("the encoder gave no frames")?;
        // the "uniform" log variance's 128 means come first
        let means = out.narrow(1, 0, 128)?.contiguous()?;
        Ok(kernels::ltx_channel_affine(&means, &self.common.std, &self.common.mean, true)?.to_dtype(DType::F32)?)
    }

    /// Encoder._forward_chunk: None when a block holds this chunk's frames back
    fn forward_chunk(&self, run: &mut Run, x: Tensor) -> Result<Option<Tensor>> {
        let mut x = self.conv_in.forward(run, Some(x), true)?;
        for b in &self.blocks {
            x = b.forward(run, x, true)?;
            if x.is_none() {
                return Ok(None);
            }
        }
        let x = kernels::ltx_pixel_norm(&x.context("no frames")?, true)?;
        self.conv_out.forward(run, Some(x), true)
    }
}

/// max |a - b|, mean |a - b| and PSNR (peak 1) of two same-shaped f32 tensors
pub fn compare(a: &Tensor, b: &Tensor) -> Result<(f32, f32, f32)> {
    let d = (a - b)?.abs()?.flatten_all()?;
    let max = d.max(D::Minus1)?.to_scalar::<f32>()?;
    let mean = d.mean(D::Minus1)?.to_scalar::<f32>()?;
    let mse = d.sqr()?.mean(D::Minus1)?.to_scalar::<f32>()?;
    Ok((max, mean, 10.0 * (1.0 / mse.max(1e-20)).log10()))
}
