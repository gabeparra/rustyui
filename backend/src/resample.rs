//! PIL's Image.resize(..., LANCZOS) on 8-bit RGB, ported from Pillow's Resample.c: separable,
//! horizontal pass then vertical, coefficients in fixed point (22 fractional bits). ComfyUI's
//! lanczos ImageScale goes through exactly this, after truncating the image to 8 bits.

use std::f64::consts::PI;

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * PI;
    x.sin() / x
}

fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// precompute_coeffs + normalize_coeffs_8bpc: (ksize, [(xmin, xlen)], fixed-point weights)
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let (in0, in1) = (0.0f64, in_size as f64);
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0f64; out_size * ksize];
    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) truncates toward zero
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xlen = (xmax - xmin) as usize;
        let k = &mut kk[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for x in 0..xlen {
            let w = lanczos(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for v in k.iter_mut().take(xlen) {
            if ww != 0.0 {
                *v /= ww;
            }
        }
        bounds.push((xmin as usize, xlen));
    }
    let fixed = kk
        .iter()
        .map(|&w| {
            let s = w * (1u64 << PRECISION_BITS) as f64;
            if w < 0.0 {
                (-0.5 + s) as i32
            } else {
                (0.5 + s) as i32
            }
        })
        .collect();
    (ksize, bounds, fixed)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// rgb: h*w*3 bytes. Returns oh*ow*3 bytes.
pub fn resize_lanczos(rgb: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let need_h = ow != w;
    let need_v = oh != h;
    let (ksize_v, bounds_v, kk_v) = coeffs(h, oh);
    // rows of the source the vertical pass reads
    let first = bounds_v[0].0;
    let last = bounds_v[oh - 1].0 + bounds_v[oh - 1].1;

    // horizontal pass over rows [first, last)
    let (tmp, tw, th, row0) = if need_h {
        let (ksize, bounds, kk) = coeffs(w, ow);
        let th = last - first;
        let mut out = vec![0u8; th * ow * 3];
        for y in 0..th {
            let src = &rgb[(y + first) * w * 3..(y + first + 1) * w * 3];
            for xx in 0..ow {
                let (xmin, xlen) = bounds[xx];
                let k = &kk[xx * ksize..xx * ksize + xlen];
                let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, kv) in k.iter().enumerate() {
                    let p = (xmin + x) * 3;
                    for c in 0..3 {
                        ss[c] = ss[c].wrapping_add(src[p + c] as i32 * kv);
                    }
                }
                for c in 0..3 {
                    out[(y * ow + xx) * 3 + c] = clip8(ss[c]);
                }
            }
        }
        (out, ow, th, first)
    } else {
        (rgb.to_vec(), w, h, 0)
    };
    if !need_v {
        return tmp;
    }
    // vertical pass; bounds shifted to the temporary image's first row
    let mut out = vec![0u8; oh * tw * 3];
    let _ = th;
    for yy in 0..oh {
        let (ymin, ylen) = bounds_v[yy];
        let ymin = ymin - row0;
        let k = &kk_v[yy * ksize_v..yy * ksize_v + ylen];
        for xx in 0..tw {
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for (y, kv) in k.iter().enumerate() {
                let p = ((ymin + y) * tw + xx) * 3;
                for c in 0..3 {
                    ss[c] = ss[c].wrapping_add(tmp[p + c] as i32 * kv);
                }
            }
            for c in 0..3 {
                out[(yy * tw + xx) * 3 + c] = clip8(ss[c]);
            }
        }
    }
    out
}

/// One axis of torch's bilinear interpolate: for each output index, the two source indices and
/// their weights. The source position is `scale * (i + 0.5) - 0.5` with `scale = input / output`,
/// one fused multiply-add as torch's CPU build computes it, clamped at 0; past the last row or
/// column the second index is the first one again.
fn bilinear_axis(input: usize, output: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|i| {
            let src = scale.mul_add(i as f32 + 0.5, -0.5).max(0.0);
            let i0 = (src.floor() as usize).min(input - 1);
            let lambda = (src - i0 as f32).clamp(0.0, 1.0);
            let i1 = if i0 < input - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - lambda, lambda)
        })
        .collect()
}

/// `torch.nn.functional.interpolate(mode="bilinear")` on the CPU (align_corners=False, no
/// antialias): ComfyUI's ImageScale "bilinear" and common_upscale's, in f32 with no trip through 8
/// bits, unlike its lanczos. Bit for bit with torch 2.11's (checked on whole pictures, cropped or
/// not, channels-last or not): along W then H, each pair as `a * wa + (b * wb)` with the first
/// product fused, which is how its compiler contracted the kernel. rgb: h*w*3 floats; returns oh*ow*3.
pub fn resize_bilinear(rgb: &[f32], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let (xs, ys) = (bilinear_axis(w, ow), bilinear_axis(h, oh));
    let mut out = Vec::with_capacity(oh * ow * 3);
    for &(y0, y1, wy0, wy1) in &ys {
        for &(x0, x1, wx0, wx1) in &xs {
            for c in 0..3 {
                let p = |y: usize, x: usize| rgb[(y * w + x) * 3 + c];
                let top = p(y0, x0).mul_add(wx0, p(y0, x1) * wx1);
                let bottom = p(y1, x0).mul_add(wx0, p(y1, x1) * wx1);
                out.push(top.mul_add(wy0, bottom * wy1));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_matches_torch() {
        // F.interpolate(torch.tensor([[[[0., 1.], [2., 3.]]]]), size=(4, 4), mode="bilinear")
        let rgb: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().flat_map(|&v| [v, v, v]).collect();
        let out = resize_bilinear(&rgb, 2, 2, 4, 4);
        let first: Vec<f32> = out.chunks(3).map(|p| p[0]).collect();
        let torch = [0.0, 0.25, 0.75, 1.0, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2.0, 2.25, 2.75, 3.0];
        assert_eq!(first, torch);
        // the same size back is the picture itself
        assert_eq!(resize_bilinear(&rgb, 2, 2, 2, 2), rgb);
    }
}
