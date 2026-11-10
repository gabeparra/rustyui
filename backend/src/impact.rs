//! ComfyUI-Impact-Pack's mask and detailer arithmetic, and ComfyUI's GrowMask, as their code runs
//! it on the CPU (the nodes are in graph_impact.rs). Masks stay in f32 there, so each step is
//! torch's, numpy's, OpenCV's or Pillow's own: torchvision's Gaussian blur is a kernel from MKL's
//! exp normalised by torch's cascade sum, F.pad's reflection and conv2d, which torch hands to
//! oneDNN's direct convolution (each output a row-major run of fused multiply-adds) or, for a 3x3
//! kernel on a small mask, to its own slow_conv2d (im2col and MKL's sgemm, linked as torch links
//! it); MaskToSEGS finds its segments as OpenCV 5's findContours and drawContours do; the
//! detailer's resizes are Pillow's LANCZOS on the picture as bytes (resample.rs).

use anyhow::{bail, Result};

extern "C" {
    fn sgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}

/// ComfyUI's GrowMask on one mask (H, W): scipy's grey_dilation (or erosion for a negative
/// expand) with a 3x3 footprint, a cross when tapered_corners, |expand| times; the edges as
/// scipy's reflect mode leaves them (the edge pixel repeated)
pub fn grow_mask(m: &[f32], h: usize, w: usize, expand: i64, tapered: bool) -> Vec<f32> {
    let mut cur = m.to_vec();
    for _ in 0..expand.unsigned_abs() {
        let mut next = cur.clone();
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut v = cur[(y * w as i64 + x) as usize];
                for (dy, dx) in [(-1i64, -1i64), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                    if tapered && dy != 0 && dx != 0 {
                        continue;
                    }
                    let (yy, xx) = ((y + dy).clamp(0, h as i64 - 1), (x + dx).clamp(0, w as i64 - 1));
                    let p = cur[(yy * w as i64 + xx) as usize];
                    v = if expand > 0 { v.max(p) } else { v.min(p) };
                }
                next[(y * w as i64 + x) as usize] = v;
            }
        }
        cur = next;
    }
    cur
}

/// multi_row_sum (SumKernel.cpp) for `rows` rows of `lanes` values: each lane's column summed
/// through four levels of 2^level_power rows, as torch's cascade sum adds them
fn multi_row_sum(x: &dyn Fn(usize, usize) -> f32, rows: usize, lanes: usize) -> Vec<f32> {
    let num_levels = 4usize;
    let ceil_log2 = if rows <= 1 { 0 } else { (usize::BITS - (rows - 1).leading_zeros()) as usize };
    let level_power = 4usize.max(ceil_log2 / num_levels);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = vec![vec![0f32; lanes]; num_levels];
    let mut i = 0usize;
    while i + level_step <= rows {
        for _ in 0..level_step {
            for k in 0..lanes {
                acc[0][k] += x(i, k);
            }
            i += 1;
        }
        for j in 1..num_levels {
            for k in 0..lanes {
                acc[j][k] += acc[j - 1][k];
                acc[j - 1][k] = 0.0;
            }
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    while i < rows {
        for k in 0..lanes {
            acc[0][k] += x(i, k);
        }
        i += 1;
    }
    for j in 1..num_levels {
        for k in 0..lanes {
            acc[0][k] += acc[j][k];
        }
    }
    acc.swap_remove(0)
}

/// torch's CPU sum of a contiguous f32 vector (cascade_sum on 8-lane vectors, which is what it runs
/// on these CPUs: checked against torch for sizes 5 to 100): from 8 values the vectors' row_sum,
/// then the scalar tail, then each lane; below that the scalar row_sum (four lanes, the remainder
/// into the first)
pub fn torch_cpu_sum_f32(v: &[f32]) -> f32 {
    const VEC: usize = 8;
    // row_sum: lanes of four accumulate rows of four (vectors or scalars), the rest into lane 0
    let row_sum = |n: usize, lanes: usize, get: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        let size_ilp = n / 4;
        let mut partial: Vec<Vec<f32>> = (0..4).map(|_| vec![0f32; lanes]).collect();
        let sums = multi_row_sum(&|r, l| get(r * 4 + l / lanes, l % lanes), size_ilp, 4 * lanes);
        for k in 0..4 {
            partial[k].copy_from_slice(&sums[k * lanes..(k + 1) * lanes]);
        }
        for i in size_ilp * 4..n {
            for l in 0..lanes {
                partial[0][l] += get(i, l);
            }
        }
        for k in 1..4 {
            for l in 0..lanes {
                let add = partial[k][l];
                partial[0][l] += add;
            }
        }
        partial.swap_remove(0)
    };
    if v.len() >= VEC {
        let vecs = v.len() / VEC;
        let lanes = row_sum(vecs, VEC, &|i, l| v[i * VEC + l]);
        let mut acc = 0f32;
        for x in &v[vecs * VEC..] {
            acc += x;
        }
        for x in lanes {
            acc += x;
        }
        acc
    } else {
        row_sum(v.len(), 1, &|i, _| v[i])[0]
    }
}

/// torchvision's _get_gaussian_kernel1d on the CPU in f32: linspace(-(k - 1) / 2, (k - 1) / 2, k)
/// divided by sigma, squared, times -0.5, MKL's exp, divided by torch's sum of it
fn gaussian_kernel1d(k: usize, sigma: f32) -> Vec<f32> {
    let half = ((k as f64 - 1.0) * 0.5) as f32;
    let (start, end) = (-half, half);
    let step = (end - start) / (k as f32 - 1.0);
    let x: Vec<f32> = (0..k).map(|i| if i < k / 2 { start + step * i as f32 } else { end - step * (k - i - 1) as f32 }).collect();
    let t: Vec<f32> = x.iter().map(|&v| -0.5f32 * ((v / sigma) * (v / sigma))).collect();
    let pdf = crate::torchcpu::exp_f32(&t);
    let s = torch_cpu_sum_f32(&pdf);
    pdf.iter().map(|&p| p / s).collect()
}

/// acc[i] = fma(x[i], w, acc[i]) for each i
fn fma_row(acc: &mut [f32], x: &[f32], w: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has AVX2 and FMA
        unsafe { fma_row_avx2(acc, x, w) };
        return;
    }
    for (a, &v) in acc.iter_mut().zip(x) {
        *a = v.mul_add(w, *a);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_row_avx2(acc: &mut [f32], x: &[f32], w: f32) {
    for (a, &v) in acc.iter_mut().zip(x) {
        *a = v.mul_add(w, *a);
    }
}

/// torch's CPU conv2d of `n` one-channel (PH, PW) pictures with one k x k kernel and no padding,
/// routed as use_mkldnn (aten/src/ATen/native/ConvUtils.h) routes it: to oneDNN unless the
/// kernel is 3x3, the batch one picture and the input at most 20480 values, which go to
/// slow_conv2d. oneDNN runs its direct convolution there (jit:avx512_core on these CPUs, checked
/// with ONEDNN_VERBOSE from 8x8 to 3000 wide, kernels 5 to 101): each output a fused multiply-add
/// per tap from zero, the kernel's rows in order and each row left to right. slow_conv2d unfolds
/// the picture (im2col) and multiplies it by the kernel with MKL's sgemm as torch calls it (m =
/// OH * OW, n = 1, k = k * k).
fn conv2d_valid(x: &[f32], n: usize, ph: usize, pw: usize, wt: &[f32], k: usize) -> Vec<f32> {
    let (oh, ow) = (ph + 1 - k, pw + 1 - k);
    let onednn = k > 3 || n > 1 || ph * pw > 20480;
    let mut out = vec![0f32; n * oh * ow];
    for i in 0..n {
        let src = &x[i * ph * pw..(i + 1) * ph * pw];
        let dst = &mut out[i * oh * ow..(i + 1) * oh * ow];
        if onednn {
            // the rows split between threads: each output is its own sum, so this changes nothing
            let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, 16).min(oh);
            let rows = oh.div_ceil(threads);
            std::thread::scope(|s| {
                for (ci, chunk) in dst.chunks_mut(rows * ow).enumerate() {
                    s.spawn(move || {
                        for (r, row) in chunk.chunks_mut(ow).enumerate() {
                            let y = ci * rows + r;
                            for ky in 0..k {
                                let line = &src[(y + ky) * pw..(y + ky + 1) * pw];
                                for kx in 0..k {
                                    fma_row(row, &line[kx..kx + ow], wt[ky * k + kx]);
                                }
                            }
                        }
                    });
                }
            });
        } else {
            let m = oh * ow;
            let mut cols = vec![0f32; k * k * m];
            for ky in 0..k {
                for kx in 0..k {
                    let row = &mut cols[(ky * k + kx) * m..(ky * k + kx + 1) * m];
                    for y in 0..oh {
                        row[y * ow..(y + 1) * ow].copy_from_slice(&src[(y + ky) * pw + kx..(y + ky) * pw + kx + ow]);
                    }
                }
            }
            let (mi, ni, ki) = (m as i32, 1i32, (k * k) as i32);
            let (alpha, beta) = (1f32, 0f32);
            // SAFETY: cols is (k*k, m) column-major as lda = m, wt k*k values, dst m values
            unsafe {
                sgemm_(
                    b"N".as_ptr(),
                    b"N".as_ptr(),
                    &mi,
                    &ni,
                    &ki,
                    &alpha,
                    cols.as_ptr(),
                    &mi,
                    wt.as_ptr(),
                    &ki,
                    &beta,
                    dst.as_mut_ptr(),
                    &mi,
                );
            }
        }
    }
    out
}

/// torchvision's GaussianBlur(k, sigma) on `n` f32 (H, W) masks on the CPU: the 2-D kernel as the
/// outer product of the 1-D one (torch.mm with one column, so each tap one product), reflection
/// padding by k / 2, then conv2d. The transform draws sigma with uniform_(sigma, sigma) on an f32
/// tensor, so it is the f32 nearest the widget's value.
pub fn gaussian_blur_cpu(img: &[f32], n: usize, h: usize, w: usize, k: usize, sigma: f32) -> Vec<f32> {
    let k1 = gaussian_kernel1d(k, sigma);
    let mut k2 = Vec::with_capacity(k * k);
    for a in &k1 {
        for b in &k1 {
            k2.push(a * b);
        }
    }
    let r = k / 2;
    let refl = |i: i64, n: usize| -> usize {
        let n = n as i64;
        let j = if i < 0 { -i } else { i };
        (if j >= n { 2 * (n - 1) - j } else { j }) as usize
    };
    let (ph, pw) = (h + 2 * r, w + 2 * r);
    let mut pad = vec![0f32; n * ph * pw];
    for i in 0..n {
        let (src, dst) = (&img[i * h * w..(i + 1) * h * w], &mut pad[i * ph * pw..(i + 1) * ph * pw]);
        for y in 0..ph {
            let sy = refl(y as i64 - r as i64, h);
            for x in 0..pw {
                dst[y * pw + x] = src[sy * w + refl(x as i64 - r as i64, w)];
            }
        }
    }
    conv2d_valid(&pad, n, ph, pw, &k2, k)
}

/// Impact's tensor_gaussian_blur_mask on `n` (H, W) masks: kernel_size * 2 + 1, shrunk to fit
/// the shorter side (no blur under 3), sigma; on the CPU, where the mask lies (its `mask.to(device)`
/// keeps no result)
pub fn tensor_gaussian_blur_mask(m: &[f32], n: usize, h: usize, w: usize, kernel_size: i64, sigma: f32) -> Vec<f32> {
    if kernel_size <= 0 {
        return m.to_vec();
    }
    let mut k = kernel_size * 2 + 1;
    let shortest = h.min(w) as i64;
    if shortest <= k {
        k = shortest / 2;
        if k % 2 == 0 {
            k += 1;
        }
        if k < 3 {
            return m.to_vec();
        }
    }
    gaussian_blur_cpu(m, n, h, w, k as usize, sigma)
}

/// One SEG of Impact's SEGS: where the detailer cuts it from the picture (crop_region) and the
/// part found there (bbox), both (x1, y1, x2, y2), and its mask over the crop (rows of the crop's
/// width)
#[derive(Clone, Debug)]
pub struct Seg {
    pub crop: [usize; 4],
    pub bbox: [i64; 4],
    pub mask: Vec<f32>,
}

/// Impact's SEGS: the picture's (height, width) and its segments
#[derive(Debug)]
pub struct Segs {
    pub h: usize,
    pub w: usize,
    pub segs: Vec<Seg>,
}

/// utils.normalize_region: a span of `size` from `startp`, kept inside 0..limit, in Python's
/// floats and int()
fn normalize_region(limit: i64, startp: i64, size: f64) -> (i64, i64) {
    let lim = limit as f64;
    let (s, e) = if startp < 0 {
        (0.0, lim.min(size))
    } else if startp as f64 + size > lim {
        ((lim - size).max(0.0), lim)
    } else {
        (startp as f64, lim.min(startp as f64 + size))
    };
    (s as i64, e as i64)
}

/// utils.make_crop_region: the bbox grown by crop_factor about its centre (at least
/// crop_min_size), moved inside the picture
pub fn make_crop_region(w: usize, h: usize, bbox: [i64; 4], crop_factor: f64, crop_min_size: Option<f64>) -> [usize; 4] {
    let (bw, bh) = (bbox[2] - bbox[0], bbox[3] - bbox[1]);
    let (mut cw, mut ch) = (bw as f64 * crop_factor, bh as f64 * crop_factor);
    if let Some(m) = crop_min_size {
        cw = if cw > m { cw } else { m };
        ch = if ch > m { ch } else { m };
    }
    let kx = bbox[0] as f64 + bw as f64 / 2.0;
    let ky = bbox[1] as f64 + bh as f64 / 2.0;
    let (x1, x2) = normalize_region(w as i64, (kx - cw / 2.0) as i64, cw);
    let (y1, y2) = normalize_region(h as i64, (ky - ch / 2.0) as i64, ch);
    [x1.max(0) as usize, y1.max(0) as usize, x2.max(0) as usize, y2.max(0) as usize]
}

/// The objects of a binary picture that cv2.findContours(RETR_TREE, CHAIN_APPROX_SIMPLE) gives a
/// top-level contour, in the order it lists them, each with cv2.boundingRect of its contour (x, y,
/// width, height) and cv2.drawContours(-1)'s fill of it (over that rectangle, rows of its width).
/// OpenCV's objects are 8-connected and its background 4-connected, and it pads the picture with a
/// frame of zeros: a top-level contour is the outer border of an object whose surrounding
/// background reaches the frame, which is the background left of the object's first pixel in a
/// raster scan, where findContours meets the object. Each new contour is prepended to its parent's
/// children and the tree is listed depth first, so the top level comes out last object first.
/// drawContours draws the contour's edges and fills the polygon between them; the contour's
/// vertices are border pixels joined by straight runs in the eight directions, so the edges pass
/// only through border pixels and the fill is the object with everything it encloses.
pub fn top_level_objects(bin: &[bool], h: usize, w: usize) -> Vec<([i64; 4], Vec<bool>)> {
    const NONE: u32 = u32::MAX;
    let mut label = vec![NONE; h * w];
    let mut firsts: Vec<(usize, usize)> = vec![];
    let mut boxes: Vec<[usize; 4]> = vec![];
    let mut stack = vec![];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !bin[i] || label[i] != NONE {
                continue;
            }
            let id = firsts.len() as u32;
            firsts.push((y, x));
            let mut bb = [x, y, x, y];
            label[i] = id;
            stack.push(i);
            while let Some(j) = stack.pop() {
                let (jy, jx) = (j / w, j % w);
                bb = [bb[0].min(jx), bb[1].min(jy), bb[2].max(jx), bb[3].max(jy)];
                for ny in jy.saturating_sub(1)..(jy + 2).min(h) {
                    for nx in jx.saturating_sub(1)..(jx + 2).min(w) {
                        let k = ny * w + nx;
                        if bin[k] && label[k] == NONE {
                            label[k] = id;
                            stack.push(k);
                        }
                    }
                }
            }
            boxes.push(bb);
        }
    }
    // the background the frame reaches, 4-connected
    let mut outside = vec![false; h * w];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if (y == 0 || x == 0 || y == h - 1 || x == w - 1) && !bin[i] && !outside[i] {
                outside[i] = true;
                stack.push(i);
            }
        }
    }
    while let Some(j) = stack.pop() {
        let (jy, jx) = (j / w, j % w);
        let mut go = |k: usize| {
            if !bin[k] && !outside[k] {
                outside[k] = true;
                stack.push(k);
            }
        };
        if jy > 0 {
            go(j - w);
        }
        if jy + 1 < h {
            go(j + w);
        }
        if jx > 0 {
            go(j - 1);
        }
        if jx + 1 < w {
            go(j + 1);
        }
    }
    let mut out = vec![];
    for id in (0..firsts.len()).rev() {
        let (y0, x0) = firsts[id];
        if x0 > 0 && !outside[y0 * w + x0 - 1] {
            continue;
        }
        // the fill: the box grown by one, flooded (4-connected) from its rim around the object;
        // what the flood cannot reach is the object or enclosed by it
        let [bx0, by0, bx1, by1] = boxes[id];
        let (gw, gh) = (bx1 - bx0 + 3, by1 - by0 + 3);
        let obj = |gx: usize, gy: usize| -> bool {
            let (x, y) = (gx as i64 + bx0 as i64 - 1, gy as i64 + by0 as i64 - 1);
            x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && label[y as usize * w + x as usize] == id as u32
        };
        let mut reach = vec![false; gw * gh];
        for gy in 0..gh {
            for gx in 0..gw {
                if (gy == 0 || gx == 0 || gy == gh - 1 || gx == gw - 1) && !reach[gy * gw + gx] {
                    reach[gy * gw + gx] = true;
                    stack.push(gy * gw + gx);
                }
            }
        }
        while let Some(j) = stack.pop() {
            let (jy, jx) = (j / gw, j % gw);
            for (ny, nx) in [(jy.wrapping_sub(1), jx), (jy + 1, jx), (jy, jx.wrapping_sub(1)), (jy, jx + 1)] {
                if ny < gh && nx < gw && !reach[ny * gw + nx] && !obj(nx, ny) {
                    reach[ny * gw + nx] = true;
                    stack.push(ny * gw + nx);
                }
            }
        }
        let (bw, bh) = (bx1 - bx0 + 1, by1 - by0 + 1);
        let mut fill = vec![false; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                fill[y * bw + x] = !reach[(y + 1) * gw + x + 1];
            }
        }
        out.push(([bx0 as i64, by0 as i64, bw as i64, bh as i64], fill));
    }
    out
}

/// A Python slice start:stop of a sequence of `len`, as a range
fn py_slice(start: i64, stop: i64, len: usize) -> std::ops::Range<usize> {
    let n = len as i64;
    let fix = |v: i64| (if v < 0 { (v + n).max(0) } else { v.min(n) }) as usize;
    let (a, b) = (fix(start), fix(stop));
    a..b.max(a)
}

/// core.mask_to_segs on `n` (H, W) masks, as MaskToSEGS calls it (no crop_min_size, no hook).
/// Combined, one segment per mask around every value that is not zero; else one per top-level
/// contour of the mask as uint8 (truncated, so values under 1/255 drop out) wider and taller than
/// drop_size, its mask the contour's fill (contour_fill) or the mask's values inside it.
#[allow(clippy::too_many_arguments)]
pub fn mask_to_segs(
    m: &[f32],
    n: usize,
    h: usize,
    w: usize,
    combined: bool,
    crop_factor: f64,
    bbox_fill: bool,
    drop_size: i64,
    contour_fill: bool,
) -> Segs {
    let drop_size = drop_size.max(1);
    let mut segs = vec![];
    let cut = |src: &dyn Fn(usize) -> f32, cr: [usize; 4]| -> Vec<f32> {
        let mut c = Vec::with_capacity((cr[2] - cr[0]) * (cr[3] - cr[1]));
        for y in cr[1]..cr[3] {
            for x in cr[0]..cr[2] {
                c.push(src(y * w + x));
            }
        }
        c
    };
    for i in 0..n {
        let mi = &m[i * h * w..(i + 1) * h * w];
        if combined {
            let mut bb: Option<[i64; 4]> = None;
            for y in 0..h {
                for x in 0..w {
                    if mi[y * w + x] != 0.0 {
                        let (x, y) = (x as i64, y as i64);
                        bb = Some(match bb {
                            None => [x, y, x, y],
                            Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
                        });
                    }
                }
            }
            let Some(bbox) = bb else { continue };
            let cr = make_crop_region(w, h, bbox, crop_factor, None);
            if cr[2] > cr[0] && cr[3] > cr[1] {
                let mut c = cut(&|j| mi[j], cr);
                if bbox_fill {
                    // the bbox in the picture's coordinates, as Impact writes it into the crop
                    let (cw, ch) = (cr[2] - cr[0], cr[3] - cr[1]);
                    for y in py_slice(bbox[1], bbox[3], ch) {
                        for x in py_slice(bbox[0], bbox[2], cw) {
                            c[y * cw + x] = 1.0;
                        }
                    }
                }
                segs.push(Seg { crop: cr, bbox, mask: c });
            }
        } else {
            let bin: Vec<bool> = mi.iter().map(|&v| (v * 255.0) as u8 != 0).collect();
            for (rect, fill) in top_level_objects(&bin, h, w) {
                let [x, y, bw, bh] = rect;
                let bbox = [x, y, x + bw, y + bh];
                let cr = make_crop_region(w, h, bbox, crop_factor, None);
                if bw <= drop_size || bh <= drop_size {
                    continue;
                }
                let inside = |j: usize| -> bool {
                    let (py, px) = ((j / w) as i64, (j % w) as i64);
                    px >= x && py >= y && px < x + bw && py < y + bh && fill[((py - y) * bw + px - x) as usize]
                };
                let mut c = if contour_fill {
                    cut(&|j| if inside(j) { 1.0 } else { 0.0 }, cr)
                } else {
                    cut(&|j| mi[j] * if inside(j) { 1.0 } else { 0.0 }, cr)
                };
                if bbox_fill {
                    let (cw, ch) = (cr[2] - cr[0], cr[3] - cr[1]);
                    let (bx1, by1) = (x - cr[0] as i64, y - cr[1] as i64);
                    for yy in py_slice(by1, by1 + bh, ch) {
                        for xx in py_slice(bx1, bx1 + bw, cw) {
                            c[yy * cw + xx] = 1.0;
                        }
                    }
                }
                // torch.clip(0, 1)
                for v in c.iter_mut() {
                    *v = if *v < 0.0 {
                        0.0
                    } else if *v > 1.0 {
                        1.0
                    } else {
                        *v
                    };
                }
                segs.push(Seg { crop: cr, bbox, mask: c });
            }
        }
    }
    Segs { h, w, segs }
}

/// The crop (x1, y1, x2, y2) of an (H, W, C) picture
pub fn crop_hwc(img: &[f32], w: usize, c: usize, cr: [usize; 4]) -> Vec<f32> {
    let mut out = Vec::with_capacity((cr[2] - cr[0]) * (cr[3] - cr[1]) * c);
    for y in cr[1]..cr[3] {
        out.extend_from_slice(&img[(y * w + cr[0]) * c..(y * w + cr[2]) * c]);
    }
    out
}

/// utils.tensor_resize on an RGB (H, W, 3) picture: tensor2pil (255 * x clipped and truncated to
/// bytes), Pillow's LANCZOS to (ow, oh) (a copy at the same size), pil2tensor (/ 255 in f32)
pub fn tensor_resize(img: &[f32], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let px: Vec<u8> = img.iter().map(|&x| (255.0f32 * x).clamp(0.0, 255.0) as u8).collect();
    let out = if (ow, oh) == (w, h) { px } else { crate::resample::resize_lanczos(&px, w, h, ow, oh) };
    out.into_iter().map(|v| v as f32 / 255.0).collect()
}

/// utils.tensor_paste of an RGB picture into another at (x, y) through a mask of its size: torch's
/// (1 - mask) * region1 + mask * region2, op by op in f32, cut to the picture's edges
#[allow(clippy::too_many_arguments)]
pub fn tensor_paste(
    img: &mut [f32],
    w: usize,
    h: usize,
    part: &[f32],
    pw: usize,
    ph: usize,
    x: usize,
    y: usize,
    mask: &[f32],
) -> Result<()> {
    if part.len() != pw * ph * 3 || mask.len() != pw * ph {
        bail!("tensor_paste: a {pw}x{ph} picture with {} values and a mask of {}", part.len(), mask.len());
    }
    let (cw, ch) = (w.min(x + pw).saturating_sub(x), h.min(y + ph).saturating_sub(y));
    for yy in 0..ch {
        for xx in 0..cw {
            let m = mask[yy * pw + xx];
            let inv = 1.0 - m;
            for c in 0..3 {
                let a = &mut img[((y + yy) * w + x + xx) * 3 + c];
                let b = part[(yy * pw + xx) * 3 + c];
                let keep = inv * *a;
                let put = m * b;
                *a = keep + put;
            }
        }
    }
    Ok(())
}

/// The count of values two f32 slices differ in, and the largest difference, as one line
fn report_line(name: &str, ours: &[f32], theirs: &[f32]) -> bool {
    if ours.len() != theirs.len() {
        println!("{name:18} {} values against {}", ours.len(), theirs.len());
        return false;
    }
    let n = ours.iter().zip(theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max = ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("{name:18} {n:>9} of {:>9} differ, max {max:.3e}{}", theirs.len(), if n == 0 { "  exact" } else { "" });
    n == 0
}

/// debug-impact-blur REF (refs/impact_blur_ref.py): GrowMask and Impact's mask blur against
/// torch's, on the reference's mask
pub fn debug_blur(refs: &std::path::Path) -> Result<()> {
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let cpu = candle_core::Device::Cpu;
    let get = |name: &str| -> Result<Vec<f32>> { Ok(st.load(name, &cpu)?.flatten_all()?.to_vec1::<f32>()?) };
    let mut ok = true;
    if let Ok(m) = st.load("mask", &cpu) {
        let (h, w) = m.dims2()?;
        let mv = m.flatten_all()?.to_vec1::<f32>()?;
        let grown = grow_mask(&mv, h, w, 8, true);
        ok &= report_line("grow8_tapered", &grown, &get("grow8_tapered")?);
        ok &= report_line("grow3_square", &grow_mask(&mv, h, w, 3, false), &get("grow3_square")?);
        ok &= report_line("shrink2", &grow_mask(&mv, h, w, -2, true), &get("shrink2")?);
        for (k, sigma) in [(31usize, 10.0f32), (11, 10.0), (41, 10.0), (7, 2.5)] {
            ok &= report_line(&format!("kernel1d_{k}"), &gaussian_kernel1d(k, sigma), &get(&format!("kernel1d_{k}"))?);
        }
        for (k, sigma) in [(15i64, 10.0f32), (5, 10.0), (20, 10.0), (3, 2.5)] {
            let name = format!("blur{k}_{sigma:?}");
            ok &= report_line(&name, &tensor_gaussian_blur_mask(&grown, 1, h, w, k, sigma), &get(&name)?);
        }
    }
    // small masks and batches: slow_conv2d's 3x3 path, oneDNN for a batch, the shrunk kernel
    for name in st.tensors().into_iter().map(|(n, _)| n).filter(|n| n.starts_with("small.")) {
        if name.ends_with(".in") {
            continue;
        }
        // small.<n>x<h>x<w>.k<kernel>
        let parts: Vec<&str> = name.trim_start_matches("small.").split('.').collect();
        let dims: Vec<usize> = parts[0].split('x').map(|v| v.parse().unwrap_or(0)).collect();
        let k: i64 = parts[1].trim_start_matches('k').parse().unwrap_or(0);
        let input = get(&format!("{name}.in"))?;
        ok &= report_line(&name, &tensor_gaussian_blur_mask(&input, dims[0], dims[1], dims[2], k, 10.0), &get(&name)?);
    }
    if !ok {
        bail!("the blur differs from torch's");
    }
    Ok(())
}

/// debug-impact-segs REF (refs/impact_segs_ref.py): MaskToSEGS against Impact's own on the
/// reference's masks, segment by segment (crop region, bbox, mask)
pub fn debug_segs(refs: &std::path::Path) -> Result<()> {
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(refs)? };
    let cpu = candle_core::Device::Cpu;
    let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let mut cases: Vec<String> =
        names.iter().filter(|n| n.ends_with(".mask") && !n.contains(".seg")).map(|n| n.trim_end_matches(".mask").to_string()).collect();
    cases.sort();
    let mut bad = 0;
    for case in cases {
        let m = st.load(&format!("{case}.mask"), &cpu)?;
        let (h, w) = m.dims2()?;
        let mv = m.flatten_all()?.to_vec1::<f32>()?;
        let opts = st.load(&format!("{case}.opts"), &cpu)?.to_vec1::<f32>()?;
        let (combined, crop_factor, bbox_fill, drop_size, contour_fill) =
            (opts[0] != 0.0, opts[1] as f64, opts[2] != 0.0, opts[3] as i64, opts[4] != 0.0);
        let segs = mask_to_segs(&mv, 1, h, w, combined, crop_factor, bbox_fill, drop_size, contour_fill);
        let want = st.load(&format!("{case}.count"), &cpu)?.to_vec1::<i64>()?[0] as usize;
        let mut ok = segs.segs.len() == want;
        let mut lines = vec![];
        for (i, s) in segs.segs.iter().enumerate().take(want) {
            let crop = st.load(&format!("{case}.seg{i}.crop"), &cpu)?.to_vec1::<i64>()?;
            let bbox = st.load(&format!("{case}.seg{i}.bbox"), &cpu)?.to_vec1::<i64>()?;
            let mask = st.load(&format!("{case}.seg{i}.mask"), &cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let crop_ok = crop.iter().zip(s.crop).all(|(a, b)| *a == b as i64);
            let bbox_ok = bbox.iter().zip(s.bbox).all(|(a, b)| *a == b);
            let diff = if mask.len() == s.mask.len() {
                mask.iter().zip(&s.mask).filter(|(a, b)| a.to_bits() != b.to_bits()).count()
            } else {
                usize::MAX
            };
            if !(crop_ok && bbox_ok && diff == 0) {
                ok = false;
                lines.push(format!(
                    "    seg{i}: crop {:?} want {crop:?}, bbox {:?} want {bbox:?}, mask {} differ",
                    s.crop,
                    s.bbox,
                    if diff == usize::MAX { "size".to_string() } else { diff.to_string() }
                ));
            }
        }
        println!("{case:28} {} segments (Impact {want}){}", segs.segs.len(), if ok { "  exact" } else { "" });
        for l in lines {
            println!("{l}");
        }
        if !ok {
            bad += 1;
        }
    }
    if bad > 0 {
        bail!("{bad} cases differ");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_region_like_python() {
        // a 247 box at (300, 200) in 1216x832, crop 3.0: 741 wide, centred, inside the picture
        assert_eq!(make_crop_region(1216, 832, [300, 200, 547, 447], 3.0, None), [53, 0, 794, 741]);
        // larger than the picture: the whole of it
        assert_eq!(make_crop_region(1216, 832, [100, 100, 627, 607], 3.0, None), [0, 0, 1216, 832]);
    }

    #[test]
    fn top_level_order_and_fill() {
        // two objects, the second (lower) listed first; a ring with an island inside: one
        // top-level contour whose fill takes the hole and the island
        let (h, w) = (12, 12);
        let mut bin = vec![false; h * w];
        for y in 1..6 {
            for x in 1..6 {
                if y == 1 || y == 5 || x == 1 || x == 5 || (y == 3 && x == 3) {
                    bin[y * w + x] = true;
                }
            }
        }
        for y in 8..10 {
            for x in 7..11 {
                bin[y * w + x] = true;
            }
        }
        let objs = top_level_objects(&bin, h, w);
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].0, [7, 8, 4, 2]);
        assert_eq!(objs[1].0, [1, 1, 5, 5]);
        assert!(objs[1].1.iter().all(|&v| v));
    }
}
