//! KSampler as ComfyUI runs it for an eps model: the discrete sigma table, the simple, normal,
//! sgm_uniform, karras and beta schedules, CFG, and the euler / euler_ancestral / dpmpp_2m /
//! dpmpp_2m_sde / er_sde samplers (dpmpp_2m_sde with its Brownian-tree noise, brownian.rs). Latents stay
//! f32; every scalar step is f32 as in PyTorch: the schedules as torch's CPU kernels compute them
//! (torchcpu), the samplers' step sizes as 0-dim f32 tensors on the card.

use crate::kernels::{torch_er_noise_scaler_f32, torch_unary_math_f32, UnaryMath};
use crate::rng::CudaRandn;
use crate::torchcpu::{exp_f32, linspace_f32, linspace_f64, ln_f32, ln_f64, pow_scalar_f32};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

pub struct Discrete {
    pub sigmas: Vec<f32>,
    log_sigmas: Vec<f32>,
    /// ModelSamplingDiscreteFlow's table (CONST): no log table, the timestep is sigma itself
    flow: bool,
}

impl Discrete {
    /// ModelSamplingDiscrete for SDXL: scaled-linear betas 0.00085..0.012 over 1000 steps,
    /// computed in f64 and stored f32; the logs are torch's f64 log over the whole table (MKL).
    pub fn sdxl() -> Self {
        let betas: Vec<f64> = linspace_f64(0.00085f64.sqrt(), 0.012f64.sqrt(), 1000).into_iter().map(|b| b * b).collect();
        let mut acp = 1.0f64;
        let mut sigmas64 = vec![];
        for b in betas {
            acp *= 1.0 - b;
            sigmas64.push(((1.0 - acp) / acp).sqrt());
        }
        let sigmas = sigmas64.iter().map(|&s| s as f32).collect();
        let log_sigmas = ln_f64(&sigmas64).into_iter().map(|l| l as f32).collect();
        Discrete { sigmas, log_sigmas, flow: false }
    }

    /// ModelSamplingDiscreteFlow (Anima: shift 3, multiplier 1) as its set_parameters builds the
    /// table on the CPU in f32: (arange(1, 1001) / 1000) * multiplier, then time_snr_shift of that
    /// / multiplier, shift * t / (1 + (shift - 1) * t), each op rounded
    pub fn flow(shift: f64, multiplier: f64) -> Self {
        let m = multiplier as f32;
        let sigmas = (1..=1000)
            .map(|i| {
                let t = (i as f32 / 1000.0) * m / m;
                if shift == 1.0 {
                    t
                } else {
                    (shift as f32 * t) / (1.0 + (shift - 1.0) as f32 * t)
                }
            })
            .collect();
        Discrete { sigmas, log_sigmas: vec![], flow: true }
    }

    /// ModelSamplingDiscreteFlow.percent_to_sigma, in Python floats: time_snr_shift(shift, 1 - p)
    pub fn flow_percent_to_sigma(shift: f64, percent: f64) -> f64 {
        if percent <= 0.0 {
            return 1.0;
        }
        if percent >= 1.0 {
            return 0.0;
        }
        let t = 1.0 - percent;
        if shift == 1.0 {
            t
        } else {
            shift * t / (1.0 + (shift - 1.0) * t)
        }
    }

    pub fn sigma_min(&self) -> f32 {
        self.sigmas[0]
    }

    pub fn sigma_max(&self) -> f32 {
        *self.sigmas.last().unwrap()
    }

    /// The nearest table index to sigma (what the UNet is given as its timestep).
    pub fn timestep(&self, sigma: f32) -> f32 {
        self.nearest(sigma.ln())
    }

    /// timestep() with the table on the card (a loaded model's), as DifferentialDiffusion calls
    /// it while the model samples: CUDA's logf, then the nearest table index
    pub fn timestep_on_card(&self, sigma: f32, dev: &Device) -> Result<i64> {
        Ok(self.nearest(torch_unary_math_f32(&[sigma], UnaryMath::Log, dev)?[0]) as i64)
    }

    /// timestep() on the CPU, as the schedules call it in ComfyUI: the log is torch's (MKL's)
    fn timestep_cpu(&self, sigma: f32) -> f32 {
        self.nearest(ln_f32(&[sigma])[0])
    }

    /// The first table index whose log sigma is nearest `ls` (argmin of the distances)
    fn nearest(&self, ls: f32) -> f32 {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for (i, l) in self.log_sigmas.iter().enumerate() {
            let d = (ls - l).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best as f32
    }

    /// ModelSamplingDiscrete.sigma(t)'s log: the table's logs interpolated in f32, op by op
    fn log_sigma(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, (self.sigmas.len() - 1) as f32);
        let (lo, hi) = (t.floor() as usize, t.ceil() as usize);
        let w = t - t.trunc();
        (1.0 - w) * self.log_sigmas[lo] + w * self.log_sigmas[hi]
    }

    /// ModelSamplingDiscrete.sigma(t) on the CPU: torch's exp of the log, which is MKL's vmsExp
    /// with n = 1
    pub fn sigma(&self, t: f32) -> f32 {
        exp_f32(&[self.log_sigma(t)])[0]
    }

    /// sigma(t) with the table on the card (a loaded model's), CUDA's expf of the log
    pub fn sigma_on_card(&self, t: f32, dev: &Device) -> Result<f32> {
        Ok(torch_unary_math_f32(&[self.log_sigma(t)], UnaryMath::Exp, dev)?[0])
    }

    /// The sigmas a scheduler gives for `steps`. normal and sgm_uniform take sigma(t) where the
    /// model's sigma table is when the sampler is made, before it loads the model: on the CPU
    /// until ComfyUI has loaded that model, and with DynamicVRAM on the card from then until it
    /// unloads it (`on_card`), where torch's exp is CUDA's expf, which parts from MKL's by an ulp
    /// at some steps (a hires fix's second pass, or any prompt after the first on a server)
    pub fn schedule(&self, scheduler: &str, steps: usize, on_card: Option<&Device>) -> Result<Vec<f32>> {
        if self.flow && !matches!(scheduler, "simple" | "beta") {
            bail!("scheduler {scheduler:?} on a flow model is not implemented yet (simple and beta are)");
        }
        Ok(match scheduler {
            "normal" | "sgm_uniform" => {
                let start = self.timestep_cpu(self.sigma_max());
                let end = self.timestep_cpu(self.sigma_min());
                let ts = if scheduler == "sgm_uniform" {
                    let mut ts = linspace_f32(start, end, steps + 1);
                    ts.pop();
                    ts
                } else {
                    linspace_f32(start, end, steps)
                };
                let mut sigs: Vec<f32> = match on_card {
                    None => ts.into_iter().map(|t| self.sigma(t)).collect(),
                    Some(dev) => {
                        let logs: Vec<f32> = ts.into_iter().map(|t| self.log_sigma(t)).collect();
                        torch_unary_math_f32(&logs, UnaryMath::Exp, dev)?
                    }
                };
                sigs.push(0.0);
                sigs
            }

            "simple" => {
                // simple_scheduler: every len / steps-th sigma of the table from the top
                let n = self.sigmas.len();
                let ss = n as f64 / steps as f64;
                let mut sigs: Vec<f32> = (0..steps).map(|x| self.sigmas[n - 1 - (x as f64 * ss) as usize]).collect();
                sigs.push(0.0);
                sigs
            }
            "karras" => {
                // get_sigmas_karras on the CPU: min and max ** (1 / rho) as Python floats, the
                // ramp and its sum in f32 tensors, ** rho as torch's pow kernel takes it
                let rho = 7.0f64;
                let (smin, smax) = (self.sigma_min() as f64, self.sigma_max() as f64);
                let (min_inv, max_inv) = (smin.powf(1.0 / rho), smax.powf(1.0 / rho));
                let (span, top) = ((min_inv - max_inv) as f32, max_inv as f32);
                let base: Vec<f32> = linspace_f32(0.0, 1.0, steps).into_iter().map(|r| top + r * span).collect();
                let mut sigs = pow_scalar_f32(&base, rho);
                sigs.push(0.0);
                sigs
            }
            "beta" => self.beta_indices(steps).into_iter().map(|t| self.sigmas[t]).chain([0.0]).collect(),
            other => bail!("scheduler {other:?} is not implemented yet (simple, normal, karras, sgm_uniform, beta are)"),
        })
    }

    /// beta_scheduler's table indices (alpha = beta = 0.6): 1 - numpy.linspace(0, 1, steps,
    /// endpoint=False), scipy.stats.beta.ppf of each, times 999 and numpy.rint (half to even),
    /// an index that repeats the one before dropped
    pub fn beta_indices(&self, steps: usize) -> Vec<usize> {
        let total = (self.sigmas.len() - 1) as f64;
        let step = 1.0 / steps as f64;
        let mut out: Vec<usize> = vec![];
        let mut last = -1.0f64;
        for k in 0..steps {
            let t = (beta_ppf(1.0 - k as f64 * step, 0.6, 0.6) * total).round_ties_even();
            if t != last {
                out.push(t as usize);
            }
            last = t;
        }
        out
    }
}

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, nine terms; about 1e-15 relative)
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The regularized incomplete beta I_x(a, b), by its continued fraction (modified Lentz)
fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let cf = |a: f64, b: f64, x: f64| -> f64 {
        let tiny = 1e-300;
        let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..100_000 {
            let (m, m2) = (m as f64, 2.0 * m as f64);
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = 1.0 + aa / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = 1.0 + aa / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-17 {
                break;
            }
        }
        h
    };
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * cf(a, b, x) / a
    } else {
        1.0 - front * cf(b, a, 1.0 - x) / b
    }
}

/// scipy.stats.beta.ppf(q, a, b): the x with I_x(a, b) = q, by bisection to the last bit (the
/// schedule only needs it to land each x * 999 on the same integer as scipy)
fn beta_ppf(q: f64, a: f64, b: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    if q >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if beta_inc(a, b, mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// torch.sum of a contiguous float CUDA tensor of 128 to 511 values, in the order torch 2.11's
/// reduction adds them (Reduce.cuh, "vectorize along input"): one warp; lane t keeps four
/// accumulators over the vectors of four at t, t + 32, ..., then the tail value at the last whole
/// vector + t; the accumulators folded ((a0 + a1) + a2) + a3; then shuffle-down by 16, 8, 4, 2, 1
fn torch_cuda_sum_warp4(v: &[f32]) -> f32 {
    let n = v.len();
    assert!((128..512).contains(&n), "torch_cuda_sum_warp4: {n} values take another reduction layout");
    let mut lanes = [0f32; 32];
    for (t, lane) in lanes.iter_mut().enumerate() {
        let mut acc = [0f32; 4];
        let mut idx = t;
        while idx * 4 + 3 < n {
            for (i, a) in acc.iter_mut().enumerate() {
                *a += v[idx * 4 + i];
            }
            idx += 32;
        }
        let tail = n - n % 4 + t;
        if tail < n {
            acc[0] += v[tail];
        }
        *lane = ((acc[0] + acc[1]) + acc[2]) + acc[3];
    }
    let mut off = 16;
    while off > 0 {
        for i in 0..off {
            lanes[i] += lanes[i + off];
        }
        off /= 2;
    }
    lanes[0]
}

/// comfy.utils.reshape_mask, which the sampler's prepare_mask runs on each noise mask before it
/// goes to the card: the mask read as (-1, 1, H, W) for a 4-d latent and as (1, 1, -1, H, W) for a
/// 5-d one when it has fewer dims, F.interpolate to the latent's spatial size (bilinear,
/// trilinear: torch's CPU kernel, which leaves a mask of one value a frame a few ulps off that
/// value at the far columns), its channels repeated up to the latent's when fewer and its batch
/// made the latent's (repeat_to_batch_size). f32, on the mask's device.
pub fn reshape_mask(mask: &Tensor, shape: &[usize], on_card: bool) -> Result<Tensor> {
    let Some(dims) = shape.len().checked_sub(2).filter(|d| (1..=3).contains(d)) else {
        bail!("reshape_mask: a latent of {} dims", shape.len())
    };
    let md = mask.dims().to_vec();
    anyhow::ensure!(md.len() >= 2, "reshape_mask: a mask of {} dims", md.len());
    let (h, w) = (md[md.len() - 2], md[md.len() - 1]);
    let numel = mask.elem_count();
    let in_shape: Vec<usize> = match dims {
        2 => vec![numel / (h * w), 1, h, w],
        3 if md.len() < 5 => vec![1, 1, numel / (h * w), h, w],
        _ => md.clone(),
    };
    anyhow::ensure!(in_shape.len() == shape.len(), "reshape_mask: a {}-d mask for a {}-d latent", in_shape.len(), shape.len());
    let x: Vec<f32> = mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let out = if on_card {
        // a mask on the card goes through torch's CUDA bilinear (a 2-d latent's only)
        anyhow::ensure!(dims == 2, "reshape_mask on the card: a {dims}-d latent");
        let t = Tensor::from_vec(x, in_shape.clone(), mask.device())?;
        crate::kernels::torch_upsample_bilinear2d_f32(&t, shape[2], shape[3], None)?.flatten_all()?.to_vec1::<f32>()?
    } else {
        crate::torchcpu::interpolate_linear(&x, &in_shape, &shape[2..])?
    };
    let plane: usize = shape[2..].iter().product();
    let (n, c) = (in_shape[0], in_shape[1]);
    // mask.repeat((1, C) + (1,) * dims)[:, :C] when it has fewer channels, then the batch
    // repeated or cut to the latent's: channel j is the mask's j % c, sample b its b % n
    let (want_n, want_c) = (shape[0], if c < shape[1] { shape[1] } else { c });
    let mut v = Vec::with_capacity(want_n * want_c * plane);
    for b in 0..want_n {
        for j in 0..want_c {
            let at = ((b % n) * c + j % c) * plane;
            v.extend_from_slice(&out[at..at + plane]);
        }
    }
    let mut dims_out = vec![want_n, want_c];
    dims_out.extend_from_slice(&shape[2..]);
    Ok(Tensor::from_vec(v, dims_out, mask.device())?)
}

/// The CFG denoiser: x (f32 latent) and sigma -> denoised x0.
pub trait Denoise {
    fn denoised(&mut self, x: &Tensor, sigma: f32) -> Result<Tensor>;
    /// the last call's uncond prediction as a denoised latent (what a post-CFG function sees
    /// as uncond_denoised), for the CFG++ samplers
    fn uncond_denoised(&self) -> Option<Tensor> {
        None
    }
}

fn s(v: f32, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::new(v, dev)?)
}

/// to_d: (x - denoised) / sigma
fn to_d(x: &Tensor, sigma: f32, den: &Tensor, dev: &Device) -> Result<Tensor> {
    Ok((x - den)?.broadcast_div(&s(sigma, dev)?)?)
}

pub fn sample(
    name: &str,
    model: &mut dyn Denoise,
    x: Tensor,
    sigmas: &[f32],
    seed: u64,
    sms: u32,
    dev: &Device,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<Tensor> {
    sample_as(name, model, x, sigmas, seed, sms, dev, progress, None)
}

/// sample for a flow model (ModelSamplingDiscreteFlow with this shift, CONST): the samplers that
/// read the model's sampling take its branch (er_sde's half-log-SNR is -logit(sigma), its first
/// sigma moved off 1); the ones with a separate RF version in ComfyUI are refused until ported
#[allow(clippy::too_many_arguments)]
pub fn sample_flow(
    name: &str,
    model: &mut dyn Denoise,
    x: Tensor,
    sigmas: &[f32],
    seed: u64,
    sms: u32,
    dev: &Device,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
    shift: f64,
) -> Result<Tensor> {
    if !matches!(name, "euler" | "er_sde") {
        bail!("sampler {name:?} on a flow model is not implemented yet (euler and er_sde are)");
    }
    sample_as(name, model, x, sigmas, seed, sms, dev, progress, Some(shift))
}

#[allow(clippy::too_many_arguments)]
fn sample_as(
    name: &str,
    model: &mut dyn Denoise,
    x: Tensor,
    sigmas: &[f32],
    seed: u64,
    sms: u32,
    dev: &Device,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
    flow: Option<f64>,
) -> Result<Tensor> {
    // RUSTYUI_DUMP_SAMPLER=FILE: the sigmas and every step's x and denoised x0, named as
    // refs/sdxl_sample_ref.py names ComfyUI's
    let dump_path = std::env::var_os("RUSTYUI_DUMP_SAMPLER");
    let dump = std::cell::RefCell::new(std::collections::HashMap::<String, Tensor>::new());
    if dump_path.is_some() {
        dump.borrow_mut().insert("sigmas".into(), Tensor::new(sigmas, &Device::Cpu)?);
        dump.borrow_mut().insert("x_start".into(), x.to_device(&Device::Cpu)?);
    }
    let step_n = std::cell::Cell::new(0usize);
    let mut model = |x: &Tensor, sigma: f32| -> Result<(Tensor, Option<Tensor>)> {
        let den = model.denoised(x, sigma)?;
        let uncond = model.uncond_denoised();
        if dump_path.is_some() {
            let i = step_n.get();
            dump.borrow_mut().insert(format!("step{i}.x"), x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
            dump.borrow_mut().insert(format!("step{i}.denoised"), den.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
            step_n.set(i + 1);
        }
        Ok((den, uncond))
    };
    let x = sample_inner(name, &mut model, x, sigmas, seed, sms, dev, progress, flow)?;
    if let Some(p) = dump_path {
        dump.borrow_mut().insert("x_end".into(), x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
        candle_core::safetensors::save(&dump.borrow(), std::path::PathBuf::from(p))?;
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn sample_inner(
    name: &str,
    model: &mut dyn FnMut(&Tensor, f32) -> Result<(Tensor, Option<Tensor>)>,
    mut x: Tensor,
    sigmas: &[f32],
    seed: u64,
    sms: u32,
    dev: &Device,
    progress: &mut dyn FnMut(usize, usize) -> Result<()>,
    flow: Option<f64>,
) -> Result<Tensor> {
    let steps = sigmas.len() - 1;
    match name {
        "euler" => {
            for i in 0..steps {
                let den = model(&x, sigmas[i])?.0;
                let d = to_d(&x, sigmas[i], &den, dev)?;
                let dt = sigmas[i + 1] - sigmas[i];
                x = (&x + d.broadcast_mul(&s(dt, dev)?)?)?;
                progress(i + 1, steps)?;
            }
        }
        "euler_ancestral" => {
            let mut noise = CudaRandn::new(seed, sms, 1536);
            let shape = x.shape().clone();
            for i in 0..steps {
                let den = model(&x, sigmas[i])?.0;
                let (from, to) = (sigmas[i], sigmas[i + 1]);
                let up = to.min((to * to * (from * from - to * to) / (from * from)).sqrt());
                let down = (to * to - up * up).sqrt();
                if down == 0.0 {
                    x = den;
                } else {
                    let d = to_d(&x, from, &den, dev)?;
                    let n = noise.randn_gpu(shape.elem_count(), dev)?.reshape(&shape)?;
                    let step = (&x + d.broadcast_mul(&s(down - from, dev)?)?)?;
                    x = (step + n.broadcast_mul(&s(up, dev)?)?)?;
                }
                progress(i + 1, steps)?;
            }
        }
        "dpmpp_2m" => {
            // torch runs the step sizes on 0-dim f32 tensors on the card, one op at a time:
            // t = -log(sigma), sigma_fn(t) = exp(-t), h = t_next - t, (-h).expm1(), r = h_last / h,
            // 1 / (2r) as reciprocal(2r) * 1; log, exp and expm1 are CUDA's (not libm's)
            let logs = torch_unary_math_f32(sigmas, UnaryMath::Log, dev)?;
            let t: Vec<f32> = logs.iter().map(|l| -l).collect();
            let sig_fn = torch_unary_math_f32(&logs, UnaryMath::Exp, dev)?;
            let neg_h: Vec<f32> = (0..steps).map(|i| -(t[i + 1] - t[i])).collect();
            let em1 = torch_unary_math_f32(&neg_h, UnaryMath::Expm1, dev)?;
            let mut old: Option<Tensor> = None;
            for i in 0..steps {
                let den = model(&x, sigmas[i])?.0;
                let h = t[i + 1] - t[i];
                let a = sig_fn[i + 1] / sig_fn[i];
                let target = match (&old, sigmas[i + 1] == 0.0) {
                    (Some(prev), false) => {
                        let h_last = t[i] - t[i - 1];
                        let r = h_last / h;
                        let c2 = 1.0 / (2.0 * r);
                        let c1 = 1.0 + c2;
                        (den.broadcast_mul(&s(c1, dev)?)? - prev.broadcast_mul(&s(c2, dev)?)?)?
                    }
                    _ => den.clone(),
                };
                x = (x.broadcast_mul(&s(a, dev)?)? - target.broadcast_mul(&s(em1[i], dev)?)?)?;
                old = Some(den);
                progress(i + 1, steps)?;
            }
        }
        "dpmpp_2m_sde" | "dpmpp_2m_sde_gpu" => {
            // k_diffusion's sample_dpmpp_2m_sde with its defaults (eta 1, s_noise 1, midpoint): the
            // step sizes on 0-dim f32 tensors on the card as torch runs them, lambda = -log(sigma),
            // with CUDA's logf, expf and expm1f; the noise a Brownian tree over [the smallest
            // positive sigma, the largest] seeded with the sampler's seed, queried over
            // [sigma_next, sigma] (so times -1), over sqrt(|sigma_next - sigma|); _gpu draws the
            // tree's noise on the card
            let sigma_min = sigmas.iter().copied().filter(|&v| v > 0.0).fold(f32::INFINITY, f32::min);
            let sigma_max = sigmas.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let source = if name == "dpmpp_2m_sde_gpu" { crate::brownian::NoiseSource::Cuda { sms, dev: dev.clone() } } else { crate::brownian::NoiseSource::Cpu };
            let mut tree = crate::brownian::BrownianTree::new(sigma_min as f64, sigma_max as f64, seed, x.elem_count(), source)?;
            let one = |v: f32, op: UnaryMath| -> Result<f32> { Ok(torch_unary_math_f32(&[v], op, dev)?[0]) };
            let lam: Vec<f32> = torch_unary_math_f32(sigmas, UnaryMath::Log, dev)?.iter().map(|l| -l).collect();
            let mut old: Option<Tensor> = None;
            let mut h_last = 0f32;
            for i in 0..steps {
                let den = model(&x, sigmas[i])?.0;
                if sigmas[i + 1] == 0.0 {
                    x = den.clone();
                } else {
                    let (ls, lt) = (lam[i], lam[i + 1]);
                    let h = lt - ls;
                    let h_eta = h * 2.0;
                    let alpha_t = sigmas[i + 1] * one(lt, UnaryMath::Exp)?;
                    // sigma_next / sigma * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
                    let c1 = (sigmas[i + 1] / sigmas[i]) * one(-h, UnaryMath::Exp)?;
                    let em = -one(-h_eta, UnaryMath::Expm1)?;
                    x = (x.broadcast_mul(&s(c1, dev)?)? + den.broadcast_mul(&s(alpha_t * em, dev)?)?)?;
                    if let Some(prev) = &old {
                        // 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised),
                        // 1 / r as reciprocal(r) * 1
                        let r = h_last / h;
                        let c = ((0.5 * alpha_t) * em) * (1.0 / r);
                        x = (&x + (&den - prev)?.broadcast_mul(&s(c, dev)?)?)?;
                    }
                    let d = (sigmas[i + 1] - sigmas[i]).abs().sqrt();
                    let w: Vec<f32> = tree.query(sigmas[i + 1] as f64, sigmas[i] as f64)?.into_iter().map(|v| -v / d).collect();
                    let sq = (-one(-2.0 * h, UnaryMath::Expm1)?).sqrt();
                    let n = Tensor::from_vec(w, x.shape(), dev)?.broadcast_mul(&s(sigmas[i + 1], dev)?)?.broadcast_mul(&s(sq, dev)?)?;
                    x = (&x + n)?;
                    h_last = h;
                }
                old = Some(den);
                progress(i + 1, steps)?;
            }
        }
        "er_sde" => {
            // k_diffusion's sample_er_sde (s_noise 1, max_stage 3) for an eps model: er_lambda =
            // exp(-(-log(sigma))) with CUDA's logf and expf; each step's scalars on 0-dim f32
            // tensors as torch runs them; the 200-point integrals' sums in the order torch's CUDA
            // reduction adds them; the noise the card's randn from the seed, as euler_ancestral's
            // a flow model (CONST): offset_first_sigma_for_snr moves a first sigma of 1 or more to
            // percent_to_sigma(1e-4) (a Python float into the f32 table), and the half-log-SNR is
            // -logit(sigma), so er_lambda = exp(logit(sigma))
            let mut flow_sigmas = sigmas.to_vec();
            if let Some(shift) = flow {
                if flow_sigmas.len() > 1 && flow_sigmas[0] >= 1.0 {
                    flow_sigmas[0] = Discrete::flow_percent_to_sigma(shift, 1e-4) as f32;
                }
            }
            let sigmas = &flow_sigmas[..];
            let lam = match flow {
                Some(_) => torch_unary_math_f32(&torch_unary_math_f32(sigmas, UnaryMath::Logit, dev)?, UnaryMath::Exp, dev)?,
                None => torch_unary_math_f32(&torch_unary_math_f32(sigmas, UnaryMath::Log, dev)?, UnaryMath::Exp, dev)?,
            };
            let scaler = |v: &[f32]| torch_er_noise_scaler_f32(v, dev);
            // torch.arange(0, 200.0)
            let points: Vec<f32> = (0..200).map(|i| i as f32).collect();
            let mut noise = CudaRandn::new(seed, sms, 1536);
            let shape = x.shape().clone();
            let (mut old, mut old_d): (Option<Tensor>, Option<Tensor>) = (None, None);
            for i in 0..steps {
                let den = model(&x, sigmas[i])?.0;
                let stage = (i + 1).min(3);
                if sigmas[i + 1] == 0.0 {
                    x = den.clone();
                } else {
                    let (ls, lt) = (lam[i], lam[i + 1]);
                    let alpha_s = sigmas[i] / ls;
                    let alpha_t = sigmas[i + 1] / lt;
                    let r_alpha = alpha_t / alpha_s;
                    let ns = scaler(&[lt, ls])?;
                    let r = ns[0] / ns[1];
                    // r_alpha * r * x + alpha_t * (1 - r) * denoised
                    x = (x.broadcast_mul(&s(r_alpha * r, dev)?)? + den.broadcast_mul(&s(alpha_t * (1.0 - r), dev)?)?)?;
                    if stage >= 2 {
                        let dt = lt - ls;
                        // -dt / 200.0: a multiply by the f32 reciprocal of the python float
                        let step = -dt * (1.0f32 / 200.0);
                        let pos: Vec<f32> = points.iter().map(|&p| lt + p * step).collect();
                        let scaled = scaler(&pos)?;
                        let inv: Vec<f32> = scaled.iter().map(|&v| (1.0 / v) * 1.0).collect();
                        let s_int = torch_cuda_sum_warp4(&inv) * step;
                        let prev = old.as_ref().expect("a second stage has the step before");
                        let d = (&den - prev)?.broadcast_div(&s(ls - lam[i - 1], dev)?)?;
                        x = (&x + d.broadcast_mul(&s(alpha_t * (dt + s_int * ns[0]), dev)?)?)?;
                        if stage >= 3 {
                            let w: Vec<f32> = pos.iter().zip(&scaled).map(|(&p, &v)| (p - ls) / v).collect();
                            let s_u = torch_cuda_sum_warp4(&w) * step;
                            let prev_d = old_d.as_ref().expect("a third stage has the derivative before");
                            let u = (&d - prev_d)?.broadcast_div(&s((ls - lam[i - 2]) * 0.5, dev)?)?;
                            x = (&x + u.broadcast_mul(&s(alpha_t * ((dt * dt) * 0.5 + s_u * ns[0]), dev)?)?)?;
                        }
                        old_d = Some(d);
                    }
                    // + alpha_t * noise * s_noise * (lt ** 2 - ls ** 2 * r ** 2).sqrt().nan_to_num()
                    let root = ((lt * lt) - (ls * ls) * (r * r)).sqrt();
                    let root = if root.is_nan() { 0.0 } else { root };
                    let n = noise.randn_gpu(shape.elem_count(), dev)?.reshape(&shape)?;
                    x = (&x + n.broadcast_mul(&s(alpha_t, dev)?)?.broadcast_mul(&s(root, dev)?)?)?;
                }
                old = Some(den);
                progress(i + 1, steps)?;
            }
        }
        "euler_ancestral_cfg_pp" => {
            // k_diffusion's sample_euler_ancestral_cfg_pp (eta 1, s_noise 1) for an eps model: alpha
            // = sigma * exp(-log(sigma)) with CUDA's logf and expf; d from the uncond prediction
            // (scaled by alpha_s); the ancestral step on sigma / alpha; the card's noise from the
            // seed. The CFG denoiser runs the uncond pass at every scale here, as
            // disable_cfg1_optimization asks.
            let logs = torch_unary_math_f32(sigmas, UnaryMath::Log, dev)?;
            let neg: Vec<f32> = logs.iter().map(|l| -l).collect();
            let ex = torch_unary_math_f32(&neg, UnaryMath::Exp, dev)?;
            let alpha: Vec<f32> = sigmas.iter().zip(&ex).map(|(s, e)| s * e).collect();
            let mut noise = CudaRandn::new(seed, sms, 1536);
            let shape = x.shape().clone();
            for i in 0..steps {
                let (den, uncond) = model(&x, sigmas[i])?;
                if sigmas[i + 1] == 0.0 {
                    x = den;
                } else {
                    let uncond = uncond.context("euler_ancestral_cfg_pp: the denoiser gave no uncond prediction")?;
                    let (alpha_s, alpha_t) = (alpha[i], alpha[i + 1]);
                    // to_d(x, sigma, alpha_s * uncond_denoised)
                    let d = to_d(&x, sigmas[i], &uncond.broadcast_mul(&s(alpha_s, dev)?)?, dev)?;
                    let (from, to) = (sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t);
                    let up = to.min((to * to * (from * from - to * to) / (from * from)).sqrt());
                    let down = alpha_t * (to * to - up * up).sqrt();
                    x = (den.broadcast_mul(&s(alpha_t, dev)?)? + d.broadcast_mul(&s(down, dev)?)?)?;
                    let n = noise.randn_gpu(shape.elem_count(), dev)?.reshape(&shape)?;
                    x = (&x + n.broadcast_mul(&s(alpha_t, dev)?)?.broadcast_mul(&s(up, dev)?)?)?;
                }
                progress(i + 1, steps)?;
            }
        }
        other => bail!("sampler {other:?} is not implemented yet (euler, euler_ancestral, euler_ancestral_cfg_pp, dpmpp_2m, dpmpp_2m_sde, dpmpp_2m_sde_gpu, er_sde are)"),
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    // ComfyUI's ModelSamplingDiscreteFlow(shift 3, multiplier 1).sigmas on the CPU, 2026-10-16:
    // the 1000 f32 bit patterns folded as h = h * 1000003 + bits (mod 2^64), and three of them
    #[test]
    fn flow_table_like_comfyui() {
        let d = super::Discrete::flow(3.0, 1.0);
        let bits: Vec<u32> = d.sigmas.iter().map(|s| s.to_bits()).collect();
        assert_eq!((bits[0], bits[499], bits[999]), (0x3b443730, 0x3f400000, 0x3f800000));
        let h = bits.iter().fold(0u64, |h, &b| h.wrapping_mul(1000003).wrapping_add(b as u64));
        assert_eq!(h, 17827357453139572957);
        // percent_to_sigma(1e-4), the first sigma er_sde starts from, into an f32 table
        assert_eq!(super::Discrete::flow_percent_to_sigma(3.0, 1e-4) as f32, 0.9999666810035706f32);
    }

    // torch 2.11's CUDA sum of sin(i * 0.37) * (1 + i / 7), i < 200, as float32 (a plain
    // sequential sum gives 0x40286008)
    #[test]
    fn cuda_sum_order_like_torch() {
        let v: Vec<f32> = (0..200).map(|i| ((i as f64 * 0.37).sin() * (1.0 + i as f64 / 7.0)) as f32).collect();
        assert_eq!(super::torch_cuda_sum_warp4(&v).to_bits(), 0x40286040);
    }
}
