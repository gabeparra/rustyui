//! The handful of layers SDXL is made of, with PyTorch's numerics where they matter:
//! normalisation statistics in f32, attention scores and softmax in f32.

use candle_core::{DType, Result, Tensor, D};

/// Bytes of f32 attention scores computed at once; bounds VRAM for long sequences.
const ATTN_BUDGET: usize = 512 << 20;

pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("linear", x.device(), || self.forward_inner(x))
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        if x.device().is_cuda()
            && dt == self.w.dtype()
            && (dt == DType::F16 || dt == DType::BF16)
            && std::env::var_os("RUSTYUI_NO_LT").is_none()
        {
            // F.linear as torch runs it: with a bias, addmm's cuBLASLt gemm_and_bias; without,
            // mm's cublasGemmEx. Each picks its kernel by the shapes, so a batch of four latents
            // under CFG takes other kernels than one, and only torch's own calls follow them.
            return match &self.b {
                Some(b) => crate::kernels::torch_linear(x, &self.w, b),
                None => crate::kernels::torch_linear_nobias(x, &self.w),
            };
        }
        let wt = self.w.t()?;
        let y = match x.rank() {
            2 => x.matmul(&wt)?,
            _ => {
                let dims = x.dims().to_vec();
                let k = dims[dims.len() - 1];
                let rows: usize = dims[..dims.len() - 1].iter().product();
                let mut out_dims = dims.clone();
                *out_dims.last_mut().unwrap() = self.w.dim(0)?;
                x.reshape((rows, k))?.matmul(&wt)?.reshape(out_dims)?
            }
        };
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

pub struct LayerNorm {
    pub w: Tensor,
    pub b: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("layer_norm", x.device(), || self.forward_inner(x))
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cuda() && x.dtype() == self.w.dtype() {
            return crate::kernels::layer_norm(x, &self.w, &self.b, self.eps);
        }
        // CLIP runs in f32 over f16 weights: same kernel, weights widened
        if x.device().is_cuda() {
            let (w, b) = (self.w.to_dtype(x.dtype())?, self.b.to_dtype(x.dtype())?);
            return crate::kernels::layer_norm(x, &w, &b, self.eps);
        }
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(D::Minus1)?;
        let xc = xf.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let y = xn.broadcast_mul(&self.w.to_dtype(DType::F32)?)?.broadcast_add(&self.b.to_dtype(DType::F32)?)?;
        y.to_dtype(dt)
    }
}

pub struct GroupNorm {
    pub w: Tensor,
    pub b: Tensor,
    pub groups: usize,
    pub eps: f64,
}

impl GroupNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("group_norm", x.device(), || self.fused(x, false))
    }

    /// GroupNorm then SiLU in one pass, rounding between them as the two PyTorch kernels do.
    pub fn forward_silu(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("group_norm+silu", x.device(), || self.fused(x, true))
    }

    fn fused(&self, x: &Tensor, silu: bool) -> Result<Tensor> {
        if x.device().is_cuda() && x.dtype() == self.w.dtype() {
            return crate::kernels::group_norm(x, &self.w, &self.b, self.groups, self.eps, silu);
        }
        let y = self.forward_inner(x)?;
        if silu {
            y.silu()
        } else {
            Ok(y)
        }
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let (b, c, h, w) = x.dims4()?;
        let g = self.groups;
        let xf = x.to_dtype(DType::F32)?.reshape((b, g, (c / g) * h * w))?;
        let mean = xf.mean_keepdim(2)?;
        let xc = xf.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(2)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?.reshape((b, c, h, w))?;
        drop(xc);
        let wt = self.w.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?;
        let bt = self.b.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?;
        xn.broadcast_mul(&wt)?.broadcast_add(&bt)?.to_dtype(dt)
    }
}

pub struct Conv2d {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("conv2d", x.device(), || self.forward_inner(x))
    }

    /// forward as torch runs it on a channels-last input: cuDNN's NHWC engines (kernels::cudnn_conv_nhwc)
    pub fn forward_nhwc(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::time("conv2d", x.device(), || self.conv(x, true))
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        self.conv(x, false)
    }

    fn conv(&self, x: &Tensor, nhwc: bool) -> Result<Tensor> {
        // torch's cuDNN convolution, the plan chosen as torch chooses it (kernels::cudnn_conv)
        let cout = self.w.dim(0)?;
        let (p, s) = (self.padding, self.stride);
        let y = if nhwc {
            crate::kernels::cudnn_conv_nhwc(x, &self.w, &[p, p], &[s, s], &[1, 1])?
        } else {
            crate::kernels::cudnn_conv(x, &self.w, &[p, p], &[s, s], &[1, 1])?
        };
        match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, cout, 1, 1))?),
            None => Ok(y),
        }
    }
}

/// Attention over (b, n, heads*d) inputs, scores and softmax in f32, chunked over queries.
/// `mask` is added to the scores and broadcast over batch and heads.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize, mask: Option<&Tensor>) -> Result<Tensor> {
    let name = if k.dim(1)? == q.dim(1)? { "attention (self)" } else { "attention (cross)" };
    crate::profile::time(name, q.device(), || attention_inner(q, k, v, heads, mask))
}

fn attention_inner(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize, mask: Option<&Tensor>) -> Result<Tensor> {
    let dt = q.dtype();
    let (b, n, inner) = q.dims3()?;
    let d = inner / heads;
    // FlashAttention-2, the kernel PyTorch's sdpa picks for this case in ComfyUI: f16, no mask,
    // head dim 64. Scores and softmax stay in f32 on-chip; nothing n x m is ever stored.
    if dt == DType::F16 && mask.is_none() && d == 64 && std::env::var_os("RUSTYUI_NO_FLASH").is_none() {
        let m = k.dim(1)?;
        let q4 = q.reshape((b, n, heads, d))?;
        let k4 = k.reshape((b, m, heads, d))?;
        let v4 = v.reshape((b, m, heads, d))?;
        let scale = (1.0f64 / (d as f64).sqrt()) as f32;
        return candle_flash_attn::flash_attn(&q4, &k4, &v4, scale, false)?.reshape((b, n, inner));
    }
    let m = k.dim(1)?;
    let q = q.reshape((b, n, heads, d))?.transpose(1, 2)?.contiguous()?;
    let kt = k.reshape((b, m, heads, d))?.transpose(1, 2)?.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let v = v.reshape((b, m, heads, d))?.transpose(1, 2)?.contiguous()?;
    let scale = Tensor::new((d as f32).sqrt().recip(), q.device())?;
    let chunk = (ATTN_BUDGET / (b * heads * m * 4)).clamp(1, n);
    let mut outs = vec![];
    let mut start = 0;
    while start < n {
        let len = chunk.min(n - start);
        let qc = q.narrow(2, start, len)?.to_dtype(DType::F32)?;
        let mut s = qc.matmul(&kt)?.broadcast_mul(&scale)?;
        if let Some(mask) = mask {
            s = s.broadcast_add(&mask.narrow(0, start, len)?)?;
        }
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        // flash attention multiplies the probabilities in the input dtype; f32 input stays f32
        outs.push(p.to_dtype(dt)?.matmul(&v)?);
        start += len;
    }
    let o = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 2)? };
    o.transpose(1, 2)?.reshape((b, n, inner))
}

/// PyTorch's nearest interpolation to an exact size: src = floor(dst * in/out).
pub fn nearest_to(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    if oh == 2 * h && ow == 2 * w {
        return x.upsample_nearest2d(oh, ow);
    }
    let idx = |out: usize, inp: usize| -> Vec<u32> {
        let scale = inp as f32 / out as f32;
        (0..out).map(|i| ((i as f32 * scale).floor() as usize).min(inp - 1) as u32).collect()
    };
    let rows = Tensor::new(idx(oh, h), x.device())?;
    let cols = Tensor::new(idx(ow, w), x.device())?;
    x.index_select(&rows, 2)?.index_select(&cols, 3)
}
