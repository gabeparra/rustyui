//! PyTorch's CPU arithmetic for the few tensors ComfyUI computes on the CPU before sampling (the
//! sigma tables and schedules, the noise masks), as torch 2.11's own kernels compute them:
//! linspace's loop and linear interpolation as GCC compiles them for the AVX2 / AVX-512 builds
//! (fused multiply-adds), pow by a scalar, which runs the vectorized part of a tensor through
//! Sleef's powf (Sleef_powf16_u10, the AVX-512F build) and the rest through libm's powf, and exp
//! and log, which torch hands to Intel MKL's VML.

// at::vml (aten/src/ATen/cpu/vml.h) with MKL: each contiguous run a CPU kernel gets goes to MKL's
// VML in one call, `vm[sd]<Op>(n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)`. MKL
// 2024.2.0 is linked statically (build.rs), the LP64 interface, so n is a 32-bit MKL_INT.
extern "C" {
    fn vmsExp(n: i32, a: *const f32, r: *mut f32, mode: i64);
    fn vmsLn(n: i32, a: *const f32, r: *mut f32, mode: i64);
    fn vmdLn(n: i32, a: *const f64, r: *mut f64, mode: i64);
    fn vmsCos(n: i32, a: *const f32, r: *mut f32, mode: i64);
    fn vmsSin(n: i32, a: *const f32, r: *mut f32, mode: i64);
}

/// VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE (mkl_vml_defines.h)
const VML_MODE: i64 = 0x0000_0002 | 0x0014_0000 | 0x0000_0100;

/// torch.exp on a contiguous CPU f32 tensor of these values (up to its 2048-element grain, one MKL
/// call): ComfyUI's normal and sgm_uniform schedules call it on one value at a time.
pub fn exp_f32(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    unsafe { vmsExp(x.len() as i32, x.as_ptr(), y.as_mut_ptr(), VML_MODE) };
    y
}

/// torch.cos on a contiguous CPU f32 tensor, as exp_f32
pub fn cos_f32(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    unsafe { vmsCos(x.len() as i32, x.as_ptr(), y.as_mut_ptr(), VML_MODE) };
    y
}

/// torch.sin on a contiguous CPU f32 tensor, as exp_f32
pub fn sin_f32(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    unsafe { vmsSin(x.len() as i32, x.as_ptr(), y.as_mut_ptr(), VML_MODE) };
    y
}

/// torch.mean over the last `n` contiguous values of each of `o` rows of an f16 CUDA tensor, as
/// torch 2.11's reduction (Reduce.cuh) orders the f32 sums on a card with `sms` multiprocessors,
/// worked through here on the host: vectorized along the input (each thread four accumulators over
/// aligned 4-value vectors, warp 0 taking an unaligned head and the tail), each warp's lanes summed
/// with offsets halving from 16 (a wider block first through shared memory), then the warps' sums
/// halving likewise; the mean is that times o / (o * n) in f32, rounded to f16. Layouts that need
/// torch's non-vectorized loop or a reduction across blocks are refused.
pub fn mean_fastest_f16(x: &[half::f16], o: usize, n: usize, sms: usize, threads_per_sm: usize) -> anyhow::Result<Vec<half::f16>> {
    anyhow::ensure!(x.len() == o * n && o > 0, "mean_fastest_f16: {} values for {o} rows of {n}", x.len());
    anyhow::ensure!(n >= 128, "mean_fastest_f16: rows of {n} take torch's non-vectorized loop, not ported");
    let last_pow2 = |v: usize| 1usize << (usize::BITS - 1 - v.max(1).leading_zeros());
    // setReduceConfig: vectorize_input divides dim0 by 4; set_block_dimension with 512 threads
    let (dim0, dim1, max) = (n / 4, o, 512usize);
    let p0 = if dim0 < max { last_pow2(dim0) } else { max };
    let p1 = if dim1 < max { last_pow2(dim1) } else { max };
    let mut bw = p0.min(32);
    let bh = p1.min(max / bw);
    bw = p0.min(max / bh);
    let mut step = bw;
    let split = n.div_ceil(step) >= (bh * 16).min(256);
    if split {
        step *= bh;
    }
    if split && n.div_ceil(step) >= 256 {
        let grid = o.div_ceil(1);
        let target = sms * (threads_per_sm / (bw * bh));
        if grid <= target {
            let vpt = n.div_ceil(step);
            let ctas = target.div_ceil(grid).min(vpt.div_ceil(16)).max(vpt.div_ceil(256));
            anyhow::ensure!(ctas <= 1, "mean_fastest_f16: rows of {n} take torch's reduction across {ctas} blocks, not ported");
        }
    }
    let ys = if split { bh } else { 1 };
    let factor = o as f32 / (o * n) as f32;
    let mut out = Vec::with_capacity(o);
    for row in 0..o {
        let s = &x[row * n..(row + 1) * n];
        let v = |i: usize| -> f32 { f32::from(s[i]) };
        // the slice's misalignment in values from an 8-byte boundary (the base is aligned)
        let shift0 = (row * n) % 4;
        let mut warps = vec![0f32; ys];
        for (y, warp) in warps.iter_mut().enumerate() {
            let mut lanes = vec![0f32; bw];
            for (lane, total) in lanes.iter_mut().enumerate() {
                let (mut first, mut end, mut head) = (0usize, n, 0f32);
                if shift0 > 0 {
                    if y == 0 && lane >= shift0 && lane < 4 {
                        head = v(lane - shift0);
                    }
                    first = 4 - shift0;
                    end = n - first;
                }
                let mut acc = [head, 0.0, 0.0, 0.0];
                let mut idx = lane + y * bw;
                while idx * 4 + 3 < end {
                    for (i, a) in acc.iter_mut().enumerate() {
                        *a += v(first + idx * 4 + i);
                    }
                    idx += step;
                }
                let tail = end - end % 4 + lane;
                if y == 0 && tail < end {
                    acc[0] += v(first + tail);
                }
                *total = ((acc[0] + acc[1]) + acc[2]) + acc[3];
            }
            // block_x_reduce: a block wider than a warp through shared memory first
            let mut dim = bw;
            while dim > 32 {
                let half = dim / 2;
                for l in 0..half {
                    lanes[l] += lanes[l + half];
                }
                dim = half;
            }
            let mut off = dim / 2;
            while off > 0 {
                let prev = lanes.clone();
                for l in 0..dim {
                    lanes[l] = prev[l] + if l + off < dim { prev[l + off] } else { prev[l] };
                }
                off /= 2;
            }
            *warp = lanes[0];
        }
        let mut off = ys / 2;
        while off > 0 {
            for y in 0..off {
                if y + off < ys {
                    warps[y] += warps[y + off];
                }
            }
            off /= 2;
        }
        out.push(half::f16::from_f32(warps[0] * factor));
    }
    Ok(out)
}

/// torch.log on a contiguous CPU f32 tensor, as exp_f32
pub fn ln_f32(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    unsafe { vmsLn(x.len() as i32, x.as_ptr(), y.as_mut_ptr(), VML_MODE) };
    y
}

/// torch.log on a contiguous CPU f64 tensor, as exp_f32 (the sigma table's 1000 values in one call)
pub fn ln_f64(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0f64; x.len()];
    unsafe { vmdLn(x.len() as i32, x.as_ptr(), y.as_mut_ptr(), VML_MODE) };
    y
}

/// torch.linspace(start, end, n) in f32: `start + step * i` for the first half and
/// `end - step * (n - 1 - i)` for the rest, each a fused multiply-add as the AVX2 and AVX-512
/// builds compile them.
pub fn linspace_f32(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    let half = n / 2;
    (0..n).map(|i| if i < half { step.mul_add(i as f32, start) } else { (-step).mul_add((n - i - 1) as f32, end) }).collect()
}

/// torch.linspace in f64, the same loop.
pub fn linspace_f64(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    let half = n / 2;
    (0..n).map(|i| if i < half { step.mul_add(i as f64, start) } else { (-step).mul_add((n - i - 1) as f64, end) }).collect()
}

/// How many leading elements of a contiguous f32 tensor torch's CPU loop hands to its vector
/// function: whole blocks of two vectors (32 lanes on AVX-512, 16 on AVX2), the rest one by one.
/// The capability is torch's: AVX-512 when the CPU has AVX512VL, BW, DQ and FMA, else AVX2 with
/// FMA, else none, unless ATEN_CPU_CAPABILITY says otherwise.
fn vector_block() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let fma = std::arch::is_x86_feature_detected!("fma");
        let avx512 = std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && fma;
        let avx2 = std::arch::is_x86_feature_detected!("avx2") && fma;
        match std::env::var("ATEN_CPU_CAPABILITY").ok().as_deref() {
            Some("default") => 0,
            Some("avx2") if avx2 => 16,
            _ if avx512 => 32,
            _ if avx2 => 16,
            _ => 0,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// One output position's two source indices and weights along a dimension, as
/// compute_source_index_and_lambda makes them (aten/src/ATen/native/UpSample.h) for align_corners
/// False and no scale factor: the same index with weights 1 and 0 when the size stays, else the
/// source index `scale * (i + 0.5) - 0.5` (fused, as the AVX2 / AVX-512 builds compile it) held at
/// 0 and above, its floor and the next index, and the fraction between them.
fn linear_taps(n_in: usize, n_out: usize, fused: bool) -> Vec<(usize, usize, f32, f32)> {
    (0..n_out)
        .map(|i| {
            if n_in == n_out {
                return (i, i, 1.0, 0.0);
            }
            let scale = n_in as f32 / n_out as f32;
            let x = i as f32 + 0.5;
            let src = if fused { scale.mul_add(x, -0.5) } else { scale * x - 0.5 };
            let src = if src < 0.0 { 0.0 } else { src };
            let i0 = (src.floor() as usize).min(n_in - 1);
            let l1 = (src - i0 as f32).max(0.0).min(1.0);
            let i1 = if i0 < n_in - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - l1, l1)
        })
        .collect()
}

/// `F.interpolate(x, size, mode="linear" | "bilinear" | "trilinear")` on a contiguous f32 CPU tensor
/// (N, C, *spatial), align_corners False and no scale factors, as upsample_generic_Nd_kernel_impl
/// computes it (aten/src/ATen/native/cpu/UpSampleKernel.cpp): each dimension's taps from
/// linear_taps, then the dimensions nested, the first outermost, each `t0 * w0 + t1 * w1` (which
/// GCC fuses to fma(t0, w0, t1 * w1) in the AVX2 / AVX-512 builds). Torch sends some tensors to
/// its channels-last kernel instead, which sums otherwise: 2-d ones with more than 3 channels in
/// channels-last memory or an output of H + W at most 128, 3-d ones with more than 3 channels in
/// channels-last memory. Those are refused unless no size changes, when every kernel copies.
pub fn interpolate_linear(x: &[f32], shape: &[usize], size: &[usize]) -> anyhow::Result<Vec<f32>> {
    let nd = shape.len().checked_sub(2).filter(|&d| (1..=3).contains(&d) && d == size.len());
    let Some(nd) = nd else { anyhow::bail!("interpolate: a {}-d tensor to a {}-d size", shape.len(), size.len()) };
    let (c, ins) = (shape[1], &shape[2..]);
    anyhow::ensure!(x.len() == shape.iter().product::<usize>(), "interpolate: {} values for the shape {shape:?}", x.len());
    if ins == size {
        return Ok(x.to_vec());
    }
    let in_plane: usize = ins.iter().product();
    // a contiguous tensor is channels-last too only with one channel or a 1x1 plane
    let channels_last = c == 1 || in_plane == 1;
    let vectorized = match nd {
        2 => (channels_last && c > 3) || size[0] + size[1] <= 128,
        3 => channels_last && c > 3,
        _ => false,
    };
    anyhow::ensure!(!vectorized, "interpolate: {shape:?} to {size:?} takes torch's channels-last kernel, not implemented yet");
    let fused = vector_block() > 0;
    let taps: Vec<Vec<(usize, usize, f32, f32)>> = (0..nd).map(|d| linear_taps(ins[d], size[d], fused)).collect();
    // each input dimension's stride within a plane
    let strides: Vec<usize> = (0..nd).map(|d| ins[d + 1..].iter().product()).collect();
    fn eval(x: &[f32], base: usize, d: usize, at: &[usize], taps: &[Vec<(usize, usize, f32, f32)>], strides: &[usize], fused: bool) -> f32 {
        let (i0, i1, w0, w1) = taps[d][at[d]];
        let (t0, t1) = if d + 1 == taps.len() {
            (x[base + i0 * strides[d]], x[base + i1 * strides[d]])
        } else {
            (
                eval(x, base + i0 * strides[d], d + 1, at, taps, strides, fused),
                eval(x, base + i1 * strides[d], d + 1, at, taps, strides, fused),
            )
        };
        if fused {
            t0.mul_add(w0, t1 * w1)
        } else {
            t0 * w0 + t1 * w1
        }
    }
    let out_plane: usize = size.iter().product();
    let mut out = Vec::with_capacity(shape[0] * c * out_plane);
    let mut at = vec![0usize; nd];
    for p in 0..shape[0] * c {
        for o in 0..out_plane {
            let mut r = o;
            for d in (0..nd).rev() {
                at[d] = r % size[d];
                r /= size[d];
            }
            out.push(eval(x, p * in_plane, 0, &at, &taps, &strides, fused));
        }
    }
    Ok(out)
}

/// `tensor ** exp` for a contiguous f32 tensor on the CPU, as pow_tensor_scalar_optimized_kernel
/// computes it for an exponent it has no shortcut for (not 2, 3, -2, 0.5, -0.5 or -1): Sleef's
/// powf on the vectorized blocks (the exponent as a float), and on the tail std::pow(float, double),
/// which is libm's pow in double rounded to float.
pub fn pow_scalar_f32(v: &[f32], exp: f64) -> Vec<f32> {
    debug_assert!(![2.0, 3.0, -2.0, 0.5, -0.5, -1.0].contains(&exp));
    let block = vector_block();
    let vec_end = v.len().checked_div(block).map_or(0, |blocks| blocks * block);
    v.iter().enumerate().map(|(i, &x)| if i < vec_end { sleef::powf_u10(x, exp as f32) } else { (x as f64).powf(exp) as f32 }).collect()
}

/// One lane of Sleef's powf, as the vectorized blocks of pow_scalar_f32 compute it.
pub fn sleef_powf(x: f32, y: f32) -> f32 {
    sleef::powf_u10(x, y)
}

/// Sleef 3.x's xpowf (src/libm/sleefsimdsp.c at the commit torch 2.11 pins, 5a1d179d), one lane of
/// the AVX-512F build: CONFIG 1 (vmla is a fused multiply-add), ENABLE_FMA_SP (the double-float
/// products in df.h are exact through FMA), logkf's exponent and mantissa from vgetexp / vgetmant.
/// Every operation is the one Sleef performs, in its order, so each lane rounds as Sleef's does.
mod sleef {
    #[derive(Clone, Copy)]
    struct F2 {
        x: f32,
        y: f32,
    }

    const fn f2(x: f32, y: f32) -> F2 {
        F2 { x, y }
    }

    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    /// vfmapn: a*b - c, rounded once
    fn fmapn(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, -c)
    }

    /// vfmanp: -(a*b) + c, rounded once
    fn fmanp(a: f32, b: f32, c: f32) -> f32 {
        (-a).mul_add(b, c)
    }

    fn df_normalize(t: F2) -> F2 {
        let s = t.x + t.y;
        f2(s, (t.x - s) + t.y)
    }

    fn df_scale(d: F2, s: f32) -> F2 {
        f2(d.x * s, d.y * s)
    }

    fn df_add2_f_f(x: f32, y: f32) -> F2 {
        let s = x + y;
        let v = s - x;
        f2(s, (x - (s - v)) + (y - v))
    }

    fn df_add2_f2_f(x: F2, y: f32) -> F2 {
        let s = x.x + y;
        let v = s - x.x;
        let t = (x.x - (s - v)) + (y - v);
        f2(s, t + x.y)
    }

    fn df_add_f_f2(x: f32, y: F2) -> F2 {
        let s = x + y.x;
        f2(s, ((x - s) + y.x) + y.y)
    }

    fn df_add_f2_f2(x: F2, y: F2) -> F2 {
        let s = x.x + y.x;
        f2(s, (((x.x - s) + y.x) + x.y) + y.y)
    }

    fn df_add2_f2_f2(x: F2, y: F2) -> F2 {
        let s = x.x + y.x;
        let v = s - x.x;
        let t = (x.x - (s - v)) + (y.x - v);
        f2(s, t + (x.y + y.y))
    }

    fn df_div(n: F2, d: F2) -> F2 {
        let t = 1.0 / d.x;
        let s = n.x * t;
        let u = fmapn(t, n.x, s);
        let v = fmanp(d.y, t, fmanp(d.x, t, 1.0));
        f2(s, fma(s, v, fma(n.y, t, u)))
    }

    fn df_mul_f2_f(x: F2, y: f32) -> F2 {
        let s = x.x * y;
        f2(s, fma(x.y, y, fmapn(x.x, y, s)))
    }

    fn df_mul_f2_f2(x: F2, y: F2) -> F2 {
        let s = x.x * y.x;
        f2(s, fma(x.x, y.y, fma(x.y, y.x, fmapn(x.x, y.x, s))))
    }

    fn df_squ(x: F2) -> F2 {
        let s = x.x * x.x;
        f2(s, fma(x.x + x.x, x.y, fmapn(x.x, x.x, s)))
    }

    /// _mm512_getexp_ps: floor(log2|d|) as a float (subnormals normalized), -inf for zero
    fn getexp(d: f32) -> f32 {
        let a = d.abs();
        if a.is_nan() {
            return d;
        }
        if a == 0.0 {
            return f32::NEG_INFINITY;
        }
        if a.is_infinite() {
            return f32::INFINITY;
        }
        let bits = a.to_bits();
        let e = (bits >> 23) as i32;
        if e == 0 {
            return (31 - bits.leading_zeros() as i32 - 149) as f32;
        }
        (e - 127) as f32
    }

    /// _mm512_getmant_ps(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan) for d >= 0: the significand
    /// in [0.75, 1.5)
    fn getmant_p75_1p5(d: f32) -> f32 {
        let bits = d.to_bits() & 0x7fff_ffff;
        if d.is_nan() {
            return f32::NAN;
        }
        if bits == 0 {
            return 0.0;
        }
        if bits == 0x7f80_0000 {
            return 1.0;
        }
        let mut frac = bits & 0x7f_ffff;
        if bits >> 23 == 0 {
            // a subnormal: shift its leading one out
            let shift = frac.leading_zeros() - 8;
            frac = (frac << shift) & 0x7f_ffff;
        }
        let exp = if frac & 0x40_0000 != 0 { 126 } else { 127 };
        f32::from_bits((exp << 23) | frac)
    }

    /// _mm512_cvtps_epi32 under the default rounding: nearest even, 0x80000000 when out of range
    fn cvtps_epi32(u: f32) -> i32 {
        let r = u.round_ties_even();
        if r.is_nan() || !(-2147483648.0..2147483648.0).contains(&r) {
            i32::MIN
        } else {
            r as i32
        }
    }

    /// _mm512_cvttps_epi32: truncation, 0x80000000 when out of range
    fn cvttps_epi32(u: f32) -> i32 {
        let r = u.trunc();
        if r.is_nan() || !(-2147483648.0..2147483648.0).contains(&r) {
            i32::MIN
        } else {
            r as i32
        }
    }

    /// vldexp_vf_vf_vi2: x * 2^q in Sleef's integer steps
    fn ldexp(x: f32, q: i32) -> f32 {
        let mut m = q >> 31;
        m = ((m.wrapping_add(q) >> 6).wrapping_sub(m)) << 4;
        let q = q.wrapping_sub(m << 2);
        m = m.wrapping_add(0x7f);
        if m <= 0 {
            m = 0;
        }
        if m > 0xff {
            m = 0xff;
        }
        let u = f32::from_bits((m as u32) << 23);
        let x = x * u * u * u * u;
        let u = f32::from_bits((q.wrapping_add(0x7f) as u32) << 23);
        x * u
    }

    fn mulsign(x: f32, y: f32) -> f32 {
        f32::from_bits(x.to_bits() ^ (y.to_bits() & 0x8000_0000))
    }

    fn logk(d: f32) -> F2 {
        let mut e = getexp(d * (1.0f32 / 0.75f32));
        if e == f32::INFINITY {
            e = 128.0;
        }
        let m = getmant_p75_1p5(d);
        let x = df_div(df_add2_f_f(-1.0, m), df_add2_f_f(1.0, m));
        let x2 = df_squ(x);
        let mut t = 0.240320354700088500976562f64 as f32;
        t = fma(t, x2.x, 0.285112679004669189453125f64 as f32);
        t = fma(t, x2.x, 0.400007992982864379882812f64 as f32);
        let c = f2(0.66666662693023681640625f32, 3.69183861259614332084311e-09f32);
        let mut s = df_mul_f2_f(f2(0.69314718246459960938f32, -1.904654323148236017e-09f32), e);
        s = df_add_f2_f2(s, df_scale(x, 2.0));
        df_add_f2_f2(s, df_mul_f2_f2(df_mul_f2_f2(x2, x), df_add2_f2_f2(df_mul_f2_f(x2, t), c)))
    }

    const R_LN2F: f32 = 1.442695040888963407359924681001892137426645954152985934135449406931f32;
    const L2UF: f32 = 0.693145751953125f32;
    const L2LF: f32 = 1.428606765330187045e-06f32;

    fn expk(d: F2) -> f32 {
        let u = (d.x + d.y) * R_LN2F;
        let q = cvtps_epi32(u);
        let mut s = df_add2_f2_f(d, q as f32 * -L2UF);
        s = df_add2_f2_f(s, q as f32 * -L2LF);
        s = df_normalize(s);
        let mut u = 0.00136324646882712841033936f32;
        u = fma(u, s.x, 0.00836596917361021041870117f32);
        u = fma(u, s.x, 0.0416710823774337768554688f32);
        u = fma(u, s.x, 0.166665524244308471679688f32);
        u = fma(u, s.x, 0.499999850988388061523438f32);
        let mut t = df_add_f2_f2(s, df_mul_f2_f(df_squ(s), u));
        t = df_add_f_f2(1.0, t);
        let u = ldexp(t.x + t.y, q);
        if d.x < -104.0 {
            0.0
        } else {
            u
        }
    }

    pub fn powf_u10(x: f32, y: f32) -> f32 {
        let two24 = (1 << 24) as f32;
        let yisint = y.trunc() == y || y.abs() > two24;
        let yisodd = (cvttps_epi32(y) & 1) == 1 && yisint && y.abs() < two24;
        let mut result = expk(df_mul_f2_f(logk(x.abs()), y));
        if result.is_nan() {
            result = f32::INFINITY;
        }
        result *= if x > 0.0 {
            1.0
        } else if yisint {
            if yisodd {
                -1.0
            } else {
                1.0
            }
        } else {
            f32::NAN
        };
        let efx = mulsign(x.abs() - 1.0, y);
        if y.is_infinite() {
            result = if efx < 0.0 {
                0.0
            } else if efx == 0.0 {
                1.0
            } else {
                f32::INFINITY
            };
        }
        if x.is_infinite() || x == 0.0 {
            let sign_y = y.to_bits() >> 31 == 1;
            let base = if sign_y ^ (x == 0.0) { 0.0 } else { f32::INFINITY };
            result = mulsign(base, if yisodd { x } else { 1.0 });
        }
        if x.is_nan() || y.is_nan() {
            result = f32::from_bits(u32::MAX);
        }
        if y == 0.0 || x == 1.0 {
            result = 1.0;
        }
        result
    }

    #[cfg(test)]
    mod tests {
        // (x, y, torch.pow's result bits through Sleef) where libm's powf rounds the other way,
        // from torch 2.11's AVX-512 build
        #[test]
        fn sleef_powf_matches_torch() {
            for &(x, y, want) in super::super::tests::SLEEF_POWF_CASES {
                let got = super::powf_u10(f32::from_bits(x), y).to_bits();
                assert_eq!(got, want, "powf({}, {y})", f32::from_bits(x));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// (x, y, torch.pow's result) for x ** 7 in torch 2.11's vector loop, all of them inputs where
    /// libm's powf rounds the other way
    pub(super) const SLEEF_POWF_CASES: &[(u32, f32, u32)] = &[
        (0x3f93c055, 7.0, 0x402ec140),
        (0x3fc243e1, 7.0, 0x41946214),
        (0x3f956f84, 7.0, 0x403d2f99),
        (0x3fa2d6c4, 7.0, 0x40ac9437),
        (0x3f71dafe, 7.0, 0x3f2bf82c),
        (0x3f5f49bb, 7.0, 0x3ec4a113),
    ];

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn linspace_like_torch() {
        // torch.linspace(0, 1, 10) and (999, 0, 39) from torch 2.11's AVX-512 build; the first has
        // an element (0x3f2aaaab) that the unfused formula rounds one ulp lower
        assert_eq!(
            bits(&super::linspace_f32(0.0, 1.0, 10)),
            [0x0, 0x3de38e39, 0x3e638e39, 0x3eaaaaab, 0x3ee38e39, 0x3f0e38e4, 0x3f2aaaab, 0x3f471c72, 0x3f638e39, 0x3f800000]
        );
        let t = super::linspace_f32(999.0, 0.0, 39);
        assert_eq!(bits(&t[..4]), [0x4479c000, 0x44732d79, 0x446c9af2, 0x4466086c]);
        assert_eq!(bits(&t[33..]), [0x43037287, 0x42d250d8, 0x429dbca2, 0x425250d8, 0x41d250d8, 0x0]);
    }

    #[test]
    fn interpolate_like_torch() {
        // F.interpolate(full((1, 1, 1, 1, 1), 1 - 0.85), (15, 1, 15), mode="trilinear") from torch
        // 2.11's AVX-512 build: 0.15 everywhere but these positions, a fused sum one or two ulps up
        let v = (1.0f64 - 0.85) as f32;
        let out = super::interpolate_linear(&[v], &[1, 1, 1, 1, 1], &[15, 1, 15]).unwrap();
        let mut want = vec![0x3e19999au32; 225];
        for i in (14..210).step_by(15).chain(210..224) {
            want[i] = 0x3e19999b;
        }
        want[224] = 0x3e19999c;
        assert_eq!(bits(&out), want);
        // torch.rand((1, 2, 2, 3, 2)) after manual_seed(3), to (3, 4, 5): the first six results and
        // the sum of all 120 results' bits
        let x: Vec<f32> = [
            0x3b8bb500u32,
            0x3dd834c0,
            0x3e9259f2,
            0x3cdcd060,
            0x3ef17770,
            0x3d763c80,
            0x3f459900,
            0x3f3e6315,
            0x3f182bfd,
            0x3f634a93,
            0x3ee6ed94,
            0x3f4cab6b,
            0x3e196fe4,
            0x3ecd8d52,
            0x3d5de8a0,
            0x3eeb372a,
            0x3e33d1dc,
            0x3f72fda6,
            0x3f58e9e0,
            0x3f5ff694,
            0x3f25f7ac,
            0x3e5be974,
            0x3f7305a7,
            0x3c466380,
        ]
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect();
        let out = super::interpolate_linear(&x, &[1, 2, 2, 3, 2], &[3, 4, 5]).unwrap();
        assert_eq!(bits(&out[..6]), [0x3b8bb500, 0x3c6bd541, 0x3d60f010, 0x3dc37569, 0x3dd834c0, 0x3e38938d]);
        assert_eq!(out.iter().fold(0u32, |s, v| s.wrapping_add(v.to_bits())), 1712105225);
    }
}
