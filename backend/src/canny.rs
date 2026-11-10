//! ComfyUI's Canny node: kornia 0.8.3's canny (kornia/filters/canny.py) with its defaults (a 5x5
//! Gaussian of sigma 1, hysteresis on, eps 1e-6), on the card in f32 as torch runs each op: the
//! grey picture, the separable Gaussian and the Sobel gradients on cuDNN (kernels::cudnn_conv, the
//! engines torch picks), magnitude, direction, non-maximum suppression and thresholds in one kernel
//! (ops.cu's kornia_canny_classify) that keeps every op's rounding, and the hysteresis loop as the
//! flood fill it settles to: a weak edge 8-connected to a strong one through weak ones becomes strong.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// F.pad's reflect (`reflect`) or edge (`replicate`) padding of a (B, 1, H, W) picture on the host
fn pad(x: &[f32], b: usize, h: usize, w: usize, top: usize, left: usize, reflect: bool) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h + 2 * top, w + 2 * left);
    let src = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let j = if reflect {
            let j = if i < 0 { -i } else { i };
            if j >= n {
                2 * (n - 1) - j
            } else {
                j
            }
        } else {
            i.clamp(0, n - 1)
        };
        j as usize
    };
    let mut out = Vec::with_capacity(b * ph * pw);
    for bi in 0..b {
        for r in 0..ph {
            let sr = src(r as isize - top as isize, h);
            for c in 0..pw {
                let sc = src(c as isize - left as isize, w);
                out.push(x[(bi * h + sr) * w + sc]);
            }
        }
    }
    (out, ph, pw)
}

/// kornia's get_gaussian_kernel1d(5, 1.0) on the card: exp(-x^2 / 2) by CUDA's expf, divided by
/// its sum in the order torch's reduction adds five values (four lanes, the fifth on lane 0's
/// second accumulator, then the lanes' tree)
fn gaussian5(dev: &Device) -> Result<[f32; 5]> {
    let e = crate::kernels::torch_unary_math_f32(&[-2.0, -0.5, 0.0, -0.5, -2.0], crate::kernels::UnaryMath::Exp, dev)?;
    let sum = ((e[0] + e[4]) + e[2]) + (e[1] + e[3]);
    Ok([e[0] / sum, e[1] / sum, e[2] / sum, e[3] / sum, e[4] / sum])
}

/// The Canny node's output for an IMAGE (B, 3, H, W) f32: the edges as 0 or 1 in all three
/// channels, as NCHW memory (ComfyUI repeats a (B, 1, H, W) tensor)
pub fn canny(img: &Tensor, low: f64, high: f64, dev: &Device) -> Result<Tensor> {
    let (b, c, h, w) = img.dims4()?;
    anyhow::ensure!(c >= 3, "Canny: a picture with {c} channels");
    let px = img.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    // rgb_to_grayscale: r * 0.299, then addcmul with g * 0.587 and b * 0.114, each op rounded
    let (wr, wg, wb) = (0.299f32, 0.587f32, 0.114f32);
    let mut gray = Vec::with_capacity(b * h * w);
    for bi in 0..b {
        let plane = |k: usize, i: usize| px[(bi * c + k) * h * w + i];
        for i in 0..h * w {
            gray.push((plane(0, i) * wr + plane(1, i) * wg) + plane(2, i) * wb);
        }
    }
    // gaussian_blur2d: filter2d with the row kernel on a reflect-padded picture, then the column one
    let g = gaussian5(dev)?;
    let conv = |x: Vec<f32>, ph: usize, pw: usize, k: &[f32], kshape: (usize, usize, usize, usize), padding: usize| -> Result<Tensor> {
        let x = Tensor::from_vec(x, (b, 1, ph, pw), dev)?;
        let k = Tensor::from_vec(k.to_vec(), kshape, dev)?;
        Ok(crate::kernels::cudnn_conv(&x, &k, &[padding, padding], &[1, 1], &[1, 1])?)
    };
    let (p, ph, pw) = pad(&gray, b, h, w, 0, 2, true);
    let bx = conv(p, ph, pw, &g, (1, 1, 1, 5), 0)?.flatten_all()?.to_vec1::<f32>()?;
    let (p, ph, pw) = pad(&bx, b, h, w, 2, 0, true);
    let blurred = conv(p, ph, pw, &g, (1, 1, 5, 1), 0)?.flatten_all()?.to_vec1::<f32>()?;
    // spatial_gradient (Sobel, not normalized): replicate padding, both kernels in one convolution
    let sobel = [-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let (p, ph, pw) = pad(&blurred, b, h, w, 1, 1, false);
    let grad = conv(p, ph, pw, &sobel, (2, 1, 3, 3), 0)?;
    // the magnitude for the NMS convolution (zero padding, eight neighbour differences); the
    // classification kernel computes it again the same way
    let mag = crate::kernels::kornia_canny_magnitude(&grad)?;
    let mut nms = vec![0f32; 72];
    let pairs = [(5, 4), (8, 4), (7, 4), (6, 4), (3, 4), (0, 4), (1, 4), (2, 4)];
    for (k, &(neg, pos)) in pairs.iter().enumerate() {
        nms[k * 9 + pos] = 1.0;
        nms[k * 9 + neg] = -1.0;
    }
    let nms = crate::kernels::cudnn_conv(&mag.contiguous()?, &Tensor::from_vec(nms, (8, 1, 3, 3), dev)?, &[1, 1], &[1, 1], &[1, 1])?;
    let mut edges = crate::kernels::kornia_canny_classify(&grad, &nms, low as f32, high as f32)?;
    // hysteresis: strong edges spread through 8-connected weak ones until nothing changes; the
    // weak ones left over are dropped
    let mut stack: Vec<usize> = (0..edges.len()).filter(|&i| edges[i] == 1.0).collect();
    while let Some(i) = stack.pop() {
        let (bi, p) = (i / (h * w), i % (h * w));
        let (r, q) = ((p / w) as isize, (p % w) as isize);
        for dr in -1isize..=1 {
            for dq in -1isize..=1 {
                let (rr, qq) = (r + dr, q + dq);
                if (dr, dq) == (0, 0) || rr < 0 || qq < 0 || rr >= h as isize || qq >= w as isize {
                    continue;
                }
                let j = bi * h * w + rr as usize * w + qq as usize;
                if edges[j] == 0.5 {
                    edges[j] = 1.0;
                    stack.push(j);
                }
            }
        }
    }
    for e in edges.iter_mut() {
        if *e != 1.0 {
            *e = 0.0;
        }
    }
    let e = Tensor::from_vec(edges, (b, 1, h, w), dev)?;
    Ok(Tensor::cat(&[&e, &e, &e], 1)?)
}
