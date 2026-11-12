//! Launching kernels/ops.cu from candle (compiled to PTX by build.rs).

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CudaDevice, CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));
/// kernels/ck.cu: comfy_kitchen's own kernels, built with its flags
const CK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ck.ptx"));

fn suffix(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        other => candle_core::bail!("rustyui kernels: {other:?} not supported"),
    })
}

fn contiguous(l: &Layout, what: &str) -> Result<usize> {
    if !l.is_contiguous() {
        candle_core::bail!("rustyui kernels: {what} must be contiguous");
    }
    Ok(l.start_offset())
}

/// Launches `$name` with (x, y, gamma, beta, extra...) for element type `$t`; the extra
/// arguments are pushed in the same scope so they outlive the launch builder.
macro_rules! launch3 {
    ($t:ty, $name:expr, $x:expr, $xl:expr, $g:expr, $gl:expr, $b:expr, $bl:expr, $cfg:expr, [$($extra:expr),*]) => {{
        let dev: &CudaDevice = $x.device();
        let n = $xl.shape().elem_count();
        let xs = $x.as_cuda_slice::<$t>()?.slice(contiguous($xl, "x")?..);
        let gs = $g.as_cuda_slice::<$t>()?.slice(contiguous($gl, "gamma")?..);
        let bs = $b.as_cuda_slice::<$t>()?.slice(contiguous($bl, "beta")?..);
        let out = unsafe { dev.alloc::<$t>(n)? };
        let func = dev.get_or_load_custom_func($name, "rustyui_ops", PTX)?;
        let mut builder = func.builder();
        builder.arg(&xs);
        builder.arg(&out);
        builder.arg(&gs);
        builder.arg(&bs);
        $(builder.arg($extra);)*
        unsafe { builder.launch($cfg) }.w()?;
        CudaStorage::wrap_cuda_slice(out, dev.clone())
    }};
}

pub struct GroupNormOp {
    pub groups: usize,
    pub eps: f32,
    pub silu: bool,
}

impl CustomOp3 for GroupNormOp {
    fn name(&self) -> &'static str {
        "rustyui-group-norm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-group-norm runs on the GPU only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dims = xl.shape().dims();
        let (n, c) = (dims[0], dims[1]);
        let hw: i64 = dims[2..].iter().product::<usize>() as i64;
        // RowwiseMomentsCUDAKernel's launch: a warp for fewer than 512 values a group, else 512
        let threads = if (c / self.groups) as i64 * hw < 512 { 32 } else { 512 };
        let cfg = LaunchConfig { grid_dim: ((n * self.groups) as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (ci, gi, eps, silu) = (c as i32, self.groups as i32, self.eps, self.silu as i32);
        let dt = x.dtype();
        let name = format!("group_norm_{}", suffix(dt)?);
        let out = match dt {
            DType::F16 => launch3!(half::f16, &name, x, xl, g, gl, b, bl, cfg, [&ci, &hw, &gi, &eps, &silu]),
            DType::BF16 => launch3!(half::bf16, &name, x, xl, g, gl, b, bl, cfg, [&ci, &hw, &gi, &eps, &silu]),
            _ => launch3!(f32, &name, x, xl, g, gl, b, bl, cfg, [&ci, &hw, &gi, &eps, &silu]),
        };
        Ok((out, xl.shape().clone()))
    }
}

pub struct LayerNormOp {
    pub eps: f32,
}

impl CustomOp3 for LayerNormOp {
    fn name(&self) -> &'static str {
        "rustyui-layer-norm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-layer-norm runs on the GPU only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dims = xl.shape().dims();
        let d = *dims.last().unwrap();
        let rows = xl.shape().elem_count() / d;
        // torch's vectorized layer norm: a (32, 4) block per row (ops.cu torch_layer_norm)
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
        let (di, eps) = (d as i32, self.eps);
        let dt = x.dtype();
        let name = format!("layer_norm_{}", suffix(dt)?);
        let out = match dt {
            DType::F16 => launch3!(half::f16, &name, x, xl, g, gl, b, bl, cfg, [&di, &eps]),
            DType::BF16 => launch3!(half::bf16, &name, x, xl, g, gl, b, bl, cfg, [&di, &eps]),
            _ => launch3!(f32, &name, x, xl, g, gl, b, bl, cfg, [&di, &eps]),
        };
        Ok((out, xl.shape().clone()))
    }
}

pub struct GegluOp;

impl CustomOp1 for GegluOp {
    fn name(&self) -> &'static str {
        "rustyui-geglu"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-geglu runs on the GPU only")
    }

    fn cuda_fwd(&self, h: &CudaStorage, hl: &Layout) -> Result<(CudaStorage, Shape)> {
        let dims = hl.shape().dims().to_vec();
        let two = *dims.last().unwrap();
        let half = two / 2;
        let rows = hl.shape().elem_count() / two;
        let mut out_dims = dims.clone();
        *out_dims.last_mut().unwrap() = half;
        let n = rows * half;
        let dev = h.device();
        let off = contiguous(hl, "h")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (rows_i, half_i) = (rows as i64, half as i32);
        let name = format!("geglu_{}", suffix(h.dtype())?);
        let func = dev.get_or_load_custom_func(&name, "rustyui_ops", PTX)?;
        macro_rules! go {
            ($t:ty) => {{
                let hs = h.as_cuda_slice::<$t>()?.slice(off..);
                let out = unsafe { dev.alloc::<$t>(n)? };
                let mut b = func.builder();
                b.arg(&hs);
                b.arg(&out);
                b.arg(&rows_i);
                b.arg(&half_i);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match h.dtype() {
            DType::F16 => go!(half::f16),
            DType::BF16 => go!(half::bf16),
            _ => go!(f32),
        };
        Ok((out, Shape::from(out_dims)))
    }
}

pub fn group_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, groups: usize, eps: f64, silu: bool) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(gamma, beta, &GroupNormOp { groups, eps: eps as f32, silu })
}

pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(gamma, beta, &LayerNormOp { eps: eps as f32 })
}

pub fn geglu(h: &Tensor) -> Result<Tensor> {
    h.contiguous()?.apply_op1_no_bwd(&GegluOp)
}

// ── the cuBLASLt handle, one per thread ──────────────────────────────────────
// PyTorch's F.linear on CUDA is addmm, which runs cublasLt with a BIAS epilogue: one f32
// accumulation, the bias added before the single rounding to f16. Same here.

use candle_core::cuda_backend::cudarc::cublaslt::CudaBlasLT;
use std::sync::Arc;

thread_local! {
    static LT: std::cell::RefCell<Option<Arc<CudaBlasLT>>> = const { std::cell::RefCell::new(None) };
}

fn lt(dev: &CudaDevice) -> Result<Arc<CudaBlasLT>> {
    LT.with(|c| {
        if let Some(h) = c.borrow().as_ref() {
            return Ok(h.clone());
        }
        let h = Arc::new(CudaBlasLT::new(dev.cuda_stream()).map_err(|e| candle_core::Error::Msg(format!("cublasLt: {e:?}")))?);
        *c.borrow_mut() = Some(h.clone());
        Ok(h)
    })
}

// ── NVFP4 dequantisation ─────────────────────────────────────────────────────
pub struct DequantNvfp4 {
    pub rows: usize,
    pub cols: usize,
}

impl CustomOp3 for DequantNvfp4 {
    fn name(&self) -> &'static str {
        "rustyui-dequant-nvfp4"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-dequant-nvfp4 runs on the GPU only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        s: &CudaStorage,
        sl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = q.device();
        let n = self.rows * self.cols;
        let qs = q.as_cuda_slice::<u8>()?.slice(contiguous(ql, "q")?..);
        let ss = s.as_cuda_slice::<u8>()?.slice(contiguous(sl, "scales")?..);
        let gs = g.as_cuda_slice::<f32>()?.slice(contiguous(gl, "global scale")?..);
        let out = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func("dequant_nvfp4_f32", "rustyui_ops", PTX)?;
        let (rows, cols) = (self.rows as i32, self.cols as i32);
        let n_cb = ((self.cols / 16).div_ceil(4)) as i32;
        let mut b = func.builder();
        b.arg(&qs);
        b.arg(&ss);
        b.arg(&gs);
        b.arg(&out);
        b.arg(&rows);
        b.arg(&cols);
        b.arg(&n_cb);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), Shape::from((self.rows, self.cols))))
    }
}

/// packed (rows, cols/2) u8, scales (u8 view of the fp8 tensor), global (f32, 1 element) -> (rows, cols) f32
pub fn dequant_nvfp4(q: &Tensor, scales: &Tensor, global: &Tensor) -> Result<Tensor> {
    let (rows, half) = q.dims2()?;
    q.contiguous()?.apply_op3_no_bwd(&scales.contiguous()?, &global.contiguous()?, &DequantNvfp4 { rows, cols: half * 2 })
}

pub struct DequantFp8;

impl CustomOp2 for DequantFp8 {
    fn name(&self) -> &'static str {
        "rustyui-dequant-fp8"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-dequant-fp8 runs on the GPU only")
    }

    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout, s: &CudaStorage, sl: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = q.device();
        let n = ql.shape().elem_count();
        let qs = q.as_cuda_slice::<u8>()?.slice(contiguous(ql, "q")?..);
        let ss = s.as_cuda_slice::<f32>()?.slice(contiguous(sl, "scale")?..);
        let out = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func("dequant_fp8_f32", "rustyui_ops", PTX)?;
        let n_i = n as i64;
        let mut b = func.builder();
        b.arg(&qs);
        b.arg(&ss);
        b.arg(&out);
        b.arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), ql.shape().clone()))
    }
}

/// fp8 e4m3 bytes (any shape, as u8), scale (f32, 1 element) -> the same shape in f32
pub fn dequant_fp8(q: &Tensor, scale: &Tensor) -> Result<Tensor> {
    q.contiguous()?.apply_op2_no_bwd(&scale.contiguous()?, &DequantFp8)
}

// ── cuBLASLt GEMMs set up the way PyTorch 2.11 sets them up ─────────────────
// ComfyUI's torch (2.11 on both boxes) gives cuBLASLt a 1 MiB workspace, and the heuristic's
// pick depends on it: with cudarc's 4 MiB the 4090 takes a split-K kernel for small row counts
// (LTX-2.3's 8 audio tokens), which rounds differently. Layouts are PyTorch's for a row-major
// x @ w^T: A is the weight (transposed), B the input, D (rows, n) row-major.

use candle_core::cuda_backend::cudarc::cublaslt::{result as ltr, sys as lts, MatmulShared};
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, DevicePtr};

const LT_WORKSPACE: usize = 1 << 20;

thread_local! {
    static LT_WS: std::cell::RefCell<Option<CudaSlice<u8>>> = const { std::cell::RefCell::new(None) };
}

fn lt_err(e: ltr::CublasError) -> candle_core::Error {
    candle_core::Error::Msg(format!("cublasLt: {e:?}"))
}

fn on_gpu(s: &candle_core::Storage) -> Result<&CudaStorage> {
    match s {
        candle_core::Storage::Cuda(c) => Ok(c),
        _ => candle_core::bail!("rustyui: a weight is not on the GPU"),
    }
}

/// PyTorch's _getAlignment: the largest power of two up to 256 that divides the address
fn alignment(ptr: u64) -> u32 {
    let mut a = 256u32;
    while !ptr.is_multiple_of(a as u64) {
        a /= 2;
    }
    a
}

/// One D = A^T B (+ bias) on the device's stream. Pointers are device addresses whose guards the
/// caller keeps alive across the call.
struct Gemm {
    k: usize,
    n: usize,
    rows: usize,
    /// CUDA_R_16BF or CUDA_R_32F (torch's addmm, gemm_and_bias) or CUDA_R_8F_E4M3 (torch's scaled_mm)
    ab: lts::cudaDataType_t,
    /// D's type: bf16, or f32 for an f32 addmm
    dt: lts::cudaDataType_t,
    a: u64,
    b: u64,
    d: u64,
    bias: Option<u64>,
    /// fp8: device pointers to A's and B's f32 scales
    scales: Option<(u64, u64)>,
    /// f32 on TF32 tensor cores (CUBLAS_COMPUTE_32F_FAST_TF32), as torch's gemm_and_bias runs an
    /// f32 matmul once torch.backends.cuda.matmul.allow_tf32 is set
    tf32: bool,
}

fn lt_gemm(dev: &CudaDevice, g: &Gemm) -> Result<()> {
    let handle = lt(dev)?;
    let stream = dev.cuda_stream();
    LT_WS.with(|c| -> Result<()> {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(unsafe { dev.alloc::<u8>(LT_WORKSPACE)? });
        }
        let (work, _gw) = c.as_ref().unwrap().device_ptr(&stream);
        let fp8 = g.scales.is_some();
        unsafe {
            use lts::cublasLtMatmulDescAttributes_t as A;
            use lts::cublasLtMatmulPreferenceAttributes_t as P;
            let compute =
                if g.tf32 { lts::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 } else { lts::cublasComputeType_t::CUBLAS_COMPUTE_32F };
            let desc = ltr::create_matmul_desc(compute, lts::cudaDataType_t::CUDA_R_32F).map_err(lt_err)?;
            let pref = ltr::create_matmul_pref().map_err(lt_err)?;
            let bf16 = lts::cudaDataType_t::CUDA_R_16BF;
            let a_l = ltr::create_matrix_layout(g.ab, g.k as u64, g.n as u64, g.k as i64).map_err(lt_err)?;
            let b_l = ltr::create_matrix_layout(g.ab, g.k as u64, g.rows as u64, g.k as i64).map_err(lt_err)?;
            let d_l = ltr::create_matrix_layout(g.dt, g.n as u64, g.rows as u64, g.n as i64).map_err(lt_err)?;
            let run = (|| -> std::result::Result<(), ltr::CublasError> {
                let set = |attr: A, v: *const std::ffi::c_void, size: usize| ltr::set_matmul_desc_attribute(desc, attr, v, size);
                let pset = |attr: P, v: *const std::ffi::c_void, size: usize| ltr::set_matmul_pref_attribute(pref, attr, v, size);
                let (ta, tb) = (1i32, 0i32);
                set(A::CUBLASLT_MATMUL_DESC_TRANSA, &ta as *const _ as _, 4)?;
                set(A::CUBLASLT_MATMUL_DESC_TRANSB, &tb as *const _ as _, 4)?;
                if let Some((sa, sb)) = &g.scales {
                    let fast = 0i8;
                    set(A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sa as *const _ as _, 8)?;
                    set(A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sb as *const _ as _, 8)?;
                    set(A::CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast as *const _ as _, 1)?;
                }
                if let Some(bias) = &g.bias {
                    let epi = lts::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as u32;
                    set(A::CUBLASLT_MATMUL_DESC_EPILOGUE, &epi as *const _ as _, 4)?;
                    set(A::CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias as *const _ as _, 8)?;
                    if fp8 {
                        let bdt = bf16 as i32;
                        set(A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt as *const _ as _, 4)?;
                    }
                }
                let wsz = LT_WORKSPACE as u64;
                pset(P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz as *const _ as _, 8)?;
                if !fp8 {
                    // gemm_and_bias limits the search to the buffers' alignment (D's taken from the bias)
                    let al = [alignment(g.a), alignment(g.b), alignment(g.d), alignment(g.bias.unwrap_or(0))];
                    pset(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al[0] as *const _ as _, 4)?;
                    pset(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al[1] as *const _ as _, 4)?;
                    pset(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al[2] as *const _ as _, 4)?;
                    pset(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al[3] as *const _ as _, 4)?;
                }
                let h = ltr::get_matmul_algo_heuristic(*handle.handle(), desc, a_l, b_l, d_l, d_l, pref)?;
                let (alpha, beta) = (1f32, 0f32);
                ltr::matmul(
                    *handle.handle(),
                    desc,
                    &alpha as *const _ as _,
                    &beta as *const _ as _,
                    g.a as *const _,
                    a_l,
                    g.b as *const _,
                    b_l,
                    g.d as *const _,
                    d_l,
                    g.d as *mut _,
                    d_l,
                    &h.algo,
                    work as *mut _,
                    LT_WORKSPACE,
                    stream.cu_stream() as *mut _,
                )
            })();
            let _ = ltr::destroy_matrix_layout(a_l);
            let _ = ltr::destroy_matrix_layout(b_l);
            let _ = ltr::destroy_matrix_layout(d_l);
            let _ = ltr::destroy_matmul_pref(pref);
            let _ = ltr::destroy_matmul_desc(desc);
            run.map_err(lt_err)
        }
    })
}

/// The (rows, k, n, out dims) of x @ w^T for x (..., k) and w (n, k)
fn linear_dims(xl: &Layout, w: &Tensor, what: &str) -> Result<(usize, usize, usize, Vec<usize>)> {
    let dims = xl.shape().dims().to_vec();
    let k = *dims.last().unwrap();
    let (n, wk) = w.dims2()?;
    if wk != k {
        candle_core::bail!("{what}: input width {k}, weight {n}x{wk}");
    }
    let mut out = dims.clone();
    *out.last_mut().unwrap() = n;
    Ok((xl.shape().elem_count() / k, k, n, out))
}

// ── bf16 linear as torch 2.11's F.linear runs it ─────────────────────────────
// addmm with the bias in cuBLASLt's epilogue (at::cuda::blas::gemm_and_bias), which torch 2.11
// takes for any row count once the weight has more than one row and column.

pub struct TorchLinear<'a> {
    /// (out, in), x's dtype (bf16 or f16)
    pub w: &'a Tensor,
    /// (out,), x's dtype
    pub bias: &'a Tensor,
}

impl CustomOp1 for TorchLinear<'_> {
    fn name(&self) -> &'static str {
        "rustyui-torch-linear"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-torch-linear runs on the GPU only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let (rows, k, n, out_dims) = linear_dims(xl, self.w, "rustyui-torch-linear")?;
        let dev = x.device();
        let stream = dev.cuda_stream();
        macro_rules! go {
            ($t:ty, $cuda:expr) => {{
                let xs = x.as_cuda_slice::<$t>()?.slice(contiguous(xl, "x")?..);
                let (w_st, w_l) = self.w.storage_and_layout();
                let ws = on_gpu(&w_st)?.as_cuda_slice::<$t>()?.slice(contiguous(w_l, "w")?..);
                let (b_st, b_l) = self.bias.storage_and_layout();
                let bs = on_gpu(&b_st)?.as_cuda_slice::<$t>()?.slice(contiguous(b_l, "bias")?..);
                let out = unsafe { dev.alloc::<$t>(rows * n)? };
                let (a, _g1) = ws.device_ptr(&stream);
                let (b, _g2) = xs.device_ptr(&stream);
                let (d, _g3) = out.device_ptr(&stream);
                let (bias, _g4) = bs.device_ptr(&stream);
                lt_gemm(dev, &Gemm { k, n, rows, ab: $cuda, dt: $cuda, a, b, d, bias: Some(bias), scales: None, tf32: false })?;
                drop((_g1, _g2, _g3, _g4));
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match x.dtype() {
            DType::BF16 => go!(half::bf16, lts::cudaDataType_t::CUDA_R_16BF),
            DType::F16 => go!(half::f16, lts::cudaDataType_t::CUDA_R_16F),
            other => candle_core::bail!("rustyui-torch-linear: {other:?}"),
        };
        Ok((out, Shape::from(out_dims)))
    }
}

/// y = x @ w^T + b in bf16 or f16, rounded the way ComfyUI's torch rounds it
pub fn torch_linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&TorchLinear { w, bias: b })
}

/// F.linear(x, w) without a bias for f16 or bf16 x (..., k) and w (n, k): what torch's mm makes of
/// the flattened x @ w.t() (gemm<at::Half> / gemm<at::BFloat16>: cublasGemmEx with op(A) = T on w,
/// op(B) = N on x, f32 compute, the default tensor-op algorithm, reduced-precision reductions
/// allowed as torch allows them, torch's workspace)
pub fn torch_linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    gemm_ex_linear(x, w, None)
}

/// F.linear(x, w, b) for f16 or bf16 where torch's addmm keeps off cuBLASLt (an input or output
/// width of 1: isInputCompliesAddmmCudaLt): the result filled with the bias, then the same
/// cublasGemmEx as torch_linear_nobias with beta 1
pub fn torch_linear_bias_gemm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    gemm_ex_linear(x, w, Some(b))
}

fn gemm_ex_linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let Some(&k) = dims.last() else { candle_core::bail!("torch_linear_nobias: a scalar input") };
    let rows = x.elem_count() / k;
    let (n, k2) = w.dims2()?;
    let dt = x.dtype();
    let ty = match dt {
        DType::BF16 => bls::cudaDataType_t::CUDA_R_16BF,
        DType::F16 => bls::cudaDataType_t::CUDA_R_16F,
        _ => candle_core::bail!("torch_linear_nobias: {dt:?}"),
    };
    if k != k2 || w.dtype() != dt {
        candle_core::bail!("torch_linear_nobias: x {:?} {:?}, w {:?} {:?}", dims, dt, w.dims(), w.dtype());
    }
    let d = cuda_dev(x)?;
    let stream = d.cuda_stream();
    let (x, w) = (x.contiguous()?, w.contiguous()?);
    let out = match bias {
        Some(b) => {
            if b.elem_count() != n || b.dtype() != dt {
                candle_core::bail!("torch_linear_bias_gemm: bias {:?} {:?} for {n} outputs", b.dims(), b.dtype());
            }
            // a fresh buffer: cuBLAS writes the result over it
            b.reshape((1, n))?.repeat((rows, 1))?.flatten_all()?.contiguous()?.copy()?
        }
        None => Tensor::zeros(rows * n, dt, x.device())?,
    };
    {
        let (xst, xl) = x.storage_and_layout();
        let (wst, wl) = w.storage_and_layout();
        let (ost, ol) = out.storage_and_layout();
        let (xp, _g1) = device_addr_any(&xst, xl, "x", &stream)?;
        let (wp, _g2) = device_addr_any(&wst, wl, "w", &stream)?;
        let (op, _g3) = device_addr_any(&ost, ol, "out", &stream)?;
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { d.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (kp, _g4) = work.device_ptr(&stream);
            unsafe {
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, if bias.is_some() { 1f32 } else { 0f32 });
                bls::cublasGemmEx(
                    *h,
                    bls::cublasOperation_t::CUBLAS_OP_T,
                    bls::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    rows as i32,
                    k as i32,
                    &alpha as *const f32 as *const _,
                    wp as *const _,
                    ty,
                    k as i32,
                    xp as *const _,
                    ty,
                    k as i32,
                    &beta as *const f32 as *const _,
                    op as *mut _,
                    ty,
                    n as i32,
                    bls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    bls::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(blas_err)?;
            }
            Ok(())
        })?;
    }
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = n;
    out.reshape(out_dims)
}

// ── fp8 linear: LTX-2.3's quantised DiT layers ───────────────────────────────
// ComfyUI runs an fp8 checkpoint layer through torch's scaled_mm: the input rounded to e4m3 by
// quant_fp8_e4m3_bf16 (scale 1.0), the weight with its own per-tensor scale, fp8 x fp8 with f32
// accumulation and no fast accumulation, and the bf16 bias added in the epilogue before the one
// rounding to bf16 (at::cuda::blas::scaled_gemm, A's scale being the weight's).

fn quant_fp8_into(dev: &CudaDevice, xs: &CudaView<'_, half::bf16>, count: usize) -> Result<CudaSlice<u8>> {
    let q = unsafe { dev.alloc::<u8>(count)? };
    let n = count as i64;
    let func = dev.get_or_load_custom_func("quant_fp8_e4m3_bf16", "rustyui_ops", PTX)?;
    let mut b = func.builder();
    b.arg(xs);
    b.arg(&q);
    b.arg(&n);
    unsafe { b.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
    Ok(q)
}

pub struct QuantFp8;

impl CustomOp1 for QuantFp8 {
    fn name(&self) -> &'static str {
        "rustyui-quant-fp8"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-quant-fp8 runs on the GPU only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = x.device();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(contiguous(xl, "x")?..);
        let q = quant_fp8_into(dev, &xs, xl.shape().elem_count())?;
        Ok((CudaStorage::wrap_cuda_slice(q, dev.clone()), xl.shape().clone()))
    }
}

/// bf16 -> the e4m3 bytes (as u8) an fp8 linear multiplies, the same shape
pub fn quant_fp8(x: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&QuantFp8)
}

pub struct Fp8Linear<'a> {
    /// (out, in) e4m3 bytes as u8
    pub w: &'a Tensor,
    /// (1,) f32, the weight's scale
    pub w_scale: &'a Tensor,
    /// (1,) f32 holding 1.0, the input's scale
    pub one: &'a Tensor,
    /// (out,) bf16
    pub bias: Option<&'a Tensor>,
}

impl CustomOp1 for Fp8Linear<'_> {
    fn name(&self) -> &'static str {
        "rustyui-fp8-linear"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-fp8-linear runs on the GPU only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let (rows, k, n, out_dims) = linear_dims(xl, self.w, "rustyui-fp8-linear")?;
        let dev = x.device();
        let stream = dev.cuda_stream();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(contiguous(xl, "x")?..);
        let q = quant_fp8_into(dev, &xs, rows * k)?;
        let (w_st, w_l) = self.w.storage_and_layout();
        let ws = on_gpu(&w_st)?.as_cuda_slice::<u8>()?.slice(contiguous(w_l, "w")?..);
        let (s_st, s_l) = self.w_scale.storage_and_layout();
        let ss = on_gpu(&s_st)?.as_cuda_slice::<f32>()?.slice(contiguous(s_l, "w_scale")?..);
        let (o_st, o_l) = self.one.storage_and_layout();
        let os = on_gpu(&o_st)?.as_cuda_slice::<f32>()?.slice(contiguous(o_l, "one")?..);
        let bias_st = self.bias.map(|t| t.storage_and_layout());
        let bs = match &bias_st {
            Some((st, l)) => Some(on_gpu(st)?.as_cuda_slice::<half::bf16>()?.slice(contiguous(l, "bias")?..)),
            None => None,
        };
        let out = unsafe { dev.alloc::<half::bf16>(rows * n)? };
        let (a, _g1) = ws.device_ptr(&stream);
        let (b, _g2) = q.device_ptr(&stream);
        let (d, _g3) = out.device_ptr(&stream);
        let (sa, _g4) = ss.device_ptr(&stream);
        let (sb, _g5) = os.device_ptr(&stream);
        let bp = bs.as_ref().map(|b| b.device_ptr(&stream));
        let e4m3 = lts::cudaDataType_t::CUDA_R_8F_E4M3;
        lt_gemm(
            dev,
            &Gemm {
                k,
                n,
                rows,
                ab: e4m3,
                dt: lts::cudaDataType_t::CUDA_R_16BF,
                a,
                b,
                d,
                bias: bp.as_ref().map(|p| p.0),
                scales: Some((sa, sb)),
                tf32: false,
            },
        )?;
        drop((_g1, _g2, _g3, _g4, _g5, bp));
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), Shape::from(out_dims)))
    }
}

/// y = x @ dequant(w)^T + b the way ComfyUI's fp8 layers compute it: x bf16 (any leading dims),
/// w (out, in) e4m3 bytes as u8, w_scale and one (1,) f32 on the GPU, b (out,) bf16; y bf16.
pub fn fp8_linear(x: &Tensor, w: &Tensor, w_scale: &Tensor, one: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&Fp8Linear { w, w_scale, one, bias: b })
}

// ── LTX-2.3's eager elementwise steps (ops.cu's ltx_* kernels) ──────────────
// Launched directly: bf16 operands, contiguous on the GPU, one thread per output element.

fn gpu_view<'a>(st: &'a candle_core::Storage, l: &Layout, what: &str) -> Result<CudaView<'a, half::bf16>> {
    Ok(on_gpu(st)?.as_cuda_slice::<half::bf16>()?.slice(contiguous(l, what)?..))
}

fn new_tensor(out: CudaSlice<half::bf16>, dev: &CudaDevice, shape: &[usize]) -> Tensor {
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone()));
    Tensor::from_storage(st, shape, candle_core::op::BackpropOp::none(), false)
}

fn cuda_dev(t: &Tensor) -> Result<CudaDevice> {
    match t.device() {
        candle_core::Device::Cuda(d) => Ok(d.clone()),
        _ => candle_core::bail!("rustyui ltx kernels: tensor not on the GPU"),
    }
}

/// The row stride of an operand repeated along a's rows: 0 for one row of `d`, `d` for a's shape.
/// How many of a's rows share each row of b: rows(a) for a single row, 1 for a's own shape, the
/// tokens per frame for a row per latent frame. Each b row covers a run of a's rows.
fn bcast_group(a: &Tensor, b: &Tensor) -> Result<i64> {
    let d = a.dim(candle_core::D::Minus1)?;
    if b.dim(candle_core::D::Minus1)? != d {
        candle_core::bail!("ltx: last dims differ: {:?} vs {:?}", a.dims(), b.dims());
    }
    let (ra, rb) = (a.elem_count() / d, b.elem_count() / d);
    if rb == 0 || ra % rb != 0 {
        candle_core::bail!("ltx: cannot repeat {:?} over {:?}", b.dims(), a.dims());
    }
    Ok((ra / rb) as i64)
}

/// name(ins..., out, n, d, groups...) for bf16 tensors; each input after the first is repeated
/// over ins[0]'s rows by its group (`bcast_group`); ins[0] and the output are full size.
fn ew(name: &str, ins: &[&Tensor], extra: &[i64]) -> Result<Tensor> {
    let a = ins[0];
    let dev = cuda_dev(a)?;
    let ins: Vec<Tensor> = ins.iter().map(|t| t.contiguous()).collect::<Result<_>>()?;
    let guards: Vec<_> = ins.iter().map(|t| t.storage_and_layout()).collect();
    let views: Vec<CudaView<'_, half::bf16>> = guards.iter().map(|(st, l)| gpu_view(st, l, name)).collect::<Result<_>>()?;
    let n = a.elem_count();
    let out = unsafe { dev.alloc::<half::bf16>(n)? };
    let func = dev.get_or_load_custom_func(name, "rustyui_ops", PTX)?;
    let mut b = func.builder();
    for v in &views {
        b.arg(v);
    }
    b.arg(&out);
    for e in extra {
        b.arg(e);
    }
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(views);
    drop(guards);
    Ok(new_tensor(out, &dev, a.dims()))
}

/// bf16(a + b), b repeated along a's rows
pub fn ltx_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (n, d, s) = (a.elem_count() as i64, a.dim(candle_core::D::Minus1)? as i64, bcast_group(a, b)?);
    ew("ltx_add", &[a, b], &[n, d, s])
}

/// bf16(a * b), b repeated along a's rows
pub fn ltx_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (n, d, s) = (a.elem_count() as i64, a.dim(candle_core::D::Minus1)? as i64, bcast_group(a, b)?);
    ew("ltx_mul", &[a, b], &[n, d, s])
}

/// torch.addcmul(a, b, c): bf16(a + b * c), b a's shape, c repeated along rows
pub fn ltx_addcmul(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    if a.dims() != b.dims() {
        candle_core::bail!("ltx_addcmul: {:?} vs {:?}", a.dims(), b.dims());
    }
    let (n, d, s) = (a.elem_count() as i64, a.dim(candle_core::D::Minus1)? as i64, bcast_group(a, c)?);
    ew("ltx_addcmul", &[a, b, c], &[n, d, s])
}

/// x * (1 + scale) + shift, three roundings, scale and shift repeated along rows
pub fn ltx_modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let (n, d, s) = (x.elem_count() as i64, x.dim(candle_core::D::Minus1)? as i64, bcast_group(x, scale)?);
    if bcast_group(x, shift)? != s {
        candle_core::bail!("ltx_modulate: scale {:?} and shift {:?} differ", scale.dims(), shift.dims());
    }
    ew("ltx_modulate", &[x, scale, shift], &[n, d, s])
}

/// F.gelu(x, approximate="tanh")
pub fn ltx_gelu_tanh(x: &Tensor) -> Result<Tensor> {
    ew("ltx_gelu_tanh", &[x], &[x.elem_count() as i64])
}

/// out (..., heads * dim) times 2 * sigmoid(logits (..., heads)), per head
pub fn ltx_head_gate(out: &Tensor, logits: &Tensor, heads: usize) -> Result<Tensor> {
    let dim = out.dim(candle_core::D::Minus1)? / heads;
    if logits.elem_count() * dim != out.elem_count() {
        candle_core::bail!("ltx_head_gate: {:?} vs logits {:?}", out.dims(), logits.dims());
    }
    let n = out.elem_count() as i64;
    let a = cuda_dev(out)?;
    let (o, l) = (out.contiguous()?, logits.contiguous()?);
    let (og, ol) = o.storage_and_layout();
    let (lg, ll) = l.storage_and_layout();
    let (ov, lv) = (gpu_view(&og, ol, "out")?, gpu_view(&lg, ll, "logits")?);
    let y = unsafe { a.alloc::<half::bf16>(n as usize)? };
    let func = a.get_or_load_custom_func("ltx_head_gate", "rustyui_ops", PTX)?;
    let (hi, di) = (heads as i32, dim as i32);
    let mut b = func.builder();
    b.arg(&ov);
    b.arg(&lv);
    b.arg(&y);
    b.arg(&n);
    b.arg(&hi);
    b.arg(&di);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((ov, lv));
    drop((og, lg));
    Ok(new_tensor(y, &a, out.dims()))
}

/// comfy_kitchen's apply_rope_split_half(1) as its CUDA backend runs it (kernels/ck.cu): x (B, T,
/// heads * 2 * half) against the rotation matrix (B or 1, T or 1, heads, half, 2, 2), computed in
/// the matrix's dtype (bf16, each product and the sum rounded) or in f32 for an f32 matrix
pub fn ltx_rope(x: &Tensor, rot: &Tensor) -> Result<Tensor> {
    let rd = rot.dims();
    if rd.len() != 6 || rd[4] != 2 || rd[5] != 2 {
        candle_core::bail!("ltx_rope: rotation matrix {:?}", rd);
    }
    let (b, t, inner) = x.dims3()?;
    let (heads, half) = (rd[2], rd[3]);
    if inner != heads * 2 * half || !(rd[0] == 1 || rd[0] == b) || !(rd[1] == 1 || rd[1] == t) {
        candle_core::bail!("ltx_rope: x {:?} vs rotation {:?}", x.dims(), rd);
    }
    let a = cuda_dev(x)?;
    let (xc, rc) = (x.contiguous()?, rot.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (rg, rl) = rc.storage_and_layout();
    let xv = gpu_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<half::bf16>(x.elem_count())? };
    let rows = (b * t * heads) as i64;
    let (hd, d1, d2) = ((2 * half) as i32, t as i64, heads as i64);
    let (fb, f1, f2) = (rd[0] as i64, rd[1] as i64, rd[2] as i64);
    let cfg = LaunchConfig::for_num_elems((rows as usize * half) as u32);
    match rot.dtype() {
        DType::BF16 => {
            let rv = gpu_view(&rg, rl, "rot")?;
            let func = a.get_or_load_custom_func("ck_rope_split_bf16", "ck", CK_PTX)?;
            let mut l = func.builder();
            l.arg(&xv).arg(&rv).arg(&y).arg(&rows).arg(&hd).arg(&d1).arg(&d2).arg(&fb).arg(&f1).arg(&f2);
            unsafe { l.launch(cfg) }.w()?;
        }
        DType::F32 => {
            let rv = on_gpu(&rg)?.as_cuda_slice::<f32>()?.slice(contiguous(rl, "rot")?..);
            let func = a.get_or_load_custom_func("ck_rope_split_bf16_f32", "ck", CK_PTX)?;
            let mut l = func.builder();
            l.arg(&xv).arg(&rv).arg(&y).arg(&rows).arg(&hd).arg(&d1).arg(&d2).arg(&fb).arg(&f1).arg(&f2);
            unsafe { l.launch(cfg) }.w()?;
        }
        other => candle_core::bail!("ltx_rope: a {other:?} rotation matrix"),
    }
    drop(xv);
    drop((xg, rg));
    Ok(new_tensor(y, &a, x.dims()))
}

/// comfy_kitchen's rms_rope_split_half as its CUDA backend runs it (kernels/ck.cu): q and k
/// (B, S, heads, head_dim) bf16 contiguous, each head RMS-normalised with its bf16 scale (head_dim
/// values) and rounded, then rotated by the f32 matrices (B or 1, S or 1, heads or 1, head_dim / 2,
/// 2, 2) in float. Returns (q, k).
pub fn ck_rms_rope(q: &Tensor, k: &Tensor, rot: &Tensor, q_scale: &Tensor, k_scale: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    let (b, s, h, d) = q.dims4()?;
    let rd = rot.dims();
    if k.dims() != q.dims()
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || rot.dtype() != DType::F32
        || rd.len() != 6
        || rd[3] != d / 2
        || rd[4] != 2
        || rd[5] != 2
        || !(rd[0] == 1 || rd[0] == b)
        || !(rd[1] == 1 || rd[1] == s)
        || !(rd[2] == 1 || rd[2] == h)
        || q_scale.elem_count() != d
        || k_scale.elem_count() != d
        || q_scale.dtype() != DType::BF16
        || k_scale.dtype() != DType::BF16
    {
        candle_core::bail!(
            "ck_rms_rope: q {:?} {:?}, k {:?}, rotation {:?} {:?}, scales {:?} {:?}",
            q.dims(),
            q.dtype(),
            k.dims(),
            rd,
            rot.dtype(),
            q_scale.dims(),
            k_scale.dims()
        );
    }
    let a = cuda_dev(q)?;
    let (qc, kc, rc, qsc, ksc) = (q.contiguous()?, k.contiguous()?, rot.contiguous()?, q_scale.contiguous()?, k_scale.contiguous()?);
    let (qg, ql) = qc.storage_and_layout();
    let (kg, kl) = kc.storage_and_layout();
    let (rg, rl) = rc.storage_and_layout();
    let (qsg, qsl) = qsc.storage_and_layout();
    let (ksg, ksl) = ksc.storage_and_layout();
    let (qv, kv) = (gpu_view(&qg, ql, "q")?, gpu_view(&kg, kl, "k")?);
    let (qsv, ksv) = (gpu_view(&qsg, qsl, "q_scale")?, gpu_view(&ksg, ksl, "k_scale")?);
    let rv = on_gpu(&rg)?.as_cuda_slice::<f32>()?.slice(contiguous(rl, "rot")?..);
    let n = q.elem_count();
    let (qo, ko) = unsafe { (a.alloc::<half::bf16>(n)?, a.alloc::<half::bf16>(n)?) };
    let rows = b * s * h;
    let (bi, si, hi, di) = (b as i64, s as i64, h as i64, d as i32);
    let (fb, f1, f2) = (rd[0] as i64, rd[1] as i64, rd[2] as i64);
    let func = a.get_or_load_custom_func("ck_rms_rope_split_bf16_f32", "ck", CK_PTX)?;
    let mut l = func.builder();
    l.arg(&qv)
        .arg(&kv)
        .arg(&rv)
        .arg(&qsv)
        .arg(&ksv)
        .arg(&qo)
        .arg(&ko)
        .arg(&bi)
        .arg(&si)
        .arg(&hi)
        .arg(&di)
        .arg(&fb)
        .arg(&f1)
        .arg(&f2)
        .arg(&eps);
    let cfg = LaunchConfig { grid_dim: (rows.div_ceil(4) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    unsafe { l.launch(cfg) }.w()?;
    drop((qv, kv, qsv, ksv, rv));
    drop((qg, kg, rg, qsg, ksg));
    Ok((new_tensor(qo, &a, q.dims()), new_tensor(ko, &a, k.dims())))
}

/// comfy_kitchen's rms_adaln as its CUDA backend runs it (kernels/ck.cu):
/// `rms_norm(x) * (1 + scale) + shift` in one pass, f32 statistics, a warp per row up to 1024 rows and a block per row beyond.
/// scale and shift each repeat over runs of x's rows (`bcast_group`), bf16 throughout.
pub fn ck_rms_adaln(x: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let n = x.elem_count() / d;
    let (sg, hg) = (bcast_group(x, scale)? as i32, bcast_group(x, shift)? as i32);
    let a = cuda_dev(x)?;
    let (xc, sc, hc) = (x.contiguous()?, scale.contiguous()?, shift.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (sgd, sl) = sc.storage_and_layout();
    let (hgd, hl) = hc.storage_and_layout();
    let (xv, sv, hv) = (gpu_view(&xg, xl, "x")?, gpu_view(&sgd, sl, "scale")?, gpu_view(&hgd, hl, "shift")?);
    let y = unsafe { a.alloc::<half::bf16>(x.elem_count())? };
    let (ni, di) = (n as i32, d as i32);
    if n <= 1024 {
        let func = a.get_or_load_custom_func("ck_rms_adaln_warp_bf16", "ck", CK_PTX)?;
        let mut l = func.builder();
        l.arg(&xv).arg(&sv).arg(&hv).arg(&y).arg(&ni).arg(&di).arg(&sg).arg(&hg).arg(&eps);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { l.launch(cfg) }.w()?;
    } else {
        let func = a.get_or_load_custom_func("ck_rms_adaln_bf16", "ck", CK_PTX)?;
        let mut l = func.builder();
        l.arg(&xv).arg(&sv).arg(&hv).arg(&y).arg(&di).arg(&sg).arg(&hg).arg(&eps);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { l.launch(cfg) }.w()?;
    }
    drop((xv, sv, hv));
    drop((xg, sgd, hgd));
    Ok(new_tensor(y, &a, x.dims()))
}

/// F.rms_norm over the last dim (a multiple of 4), with or without a bf16 weight, as torch's
/// vectorized kernel computes it
pub fn ltx_rms_norm(x: &Tensor, gamma: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let n_last = x.dim(candle_core::D::Minus1)?;
    if n_last % 4 != 0 {
        candle_core::bail!("ltx_rms_norm: width {n_last} is not a multiple of 4");
    }
    let rows = x.elem_count() / n_last;
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let gc = match gamma {
        Some(g) => g.contiguous()?,
        None => xc.clone(),
    };
    let (xg, xl) = xc.storage_and_layout();
    let (gg, gl) = gc.storage_and_layout();
    let (xv, gv) = (gpu_view(&xg, xl, "x")?, gpu_view(&gg, gl, "gamma")?);
    let y = unsafe { a.alloc::<half::bf16>(x.elem_count())? };
    let func = a.get_or_load_custom_func("ltx_rms_norm", "rustyui_ops", PTX)?;
    let (ni, has) = (n_last as i32, gamma.is_some() as i32);
    let mut b = func.builder();
    b.arg(&xv);
    b.arg(&gv);
    b.arg(&y);
    b.arg(&ni);
    b.arg(&eps);
    b.arg(&has);
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.w()?;
    drop((xv, gv));
    drop((xg, gg));
    Ok(new_tensor(y, &a, x.dims()))
}

/// nn.SiLU in bf16
pub fn ltx_silu(x: &Tensor) -> Result<Tensor> {
    ew("ltx_silu", &[x], &[x.elem_count() as i64])
}

/// F.layer_norm over the last dim without weight or bias, as torch's vectorized kernel computes it
pub fn ltx_layer_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let n_last = x.dim(candle_core::D::Minus1)?;
    if n_last % 4 != 0 {
        candle_core::bail!("ltx_layer_norm: width {n_last} is not a multiple of 4");
    }
    let rows = x.elem_count() / n_last;
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = gpu_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<half::bf16>(x.elem_count())? };
    let func = a.get_or_load_custom_func("ltx_layer_norm", "rustyui_ops", PTX)?;
    let ni = n_last as i32;
    let mut b = func.builder();
    b.arg(&xv);
    b.arg(&y);
    b.arg(&ni);
    b.arg(&eps);
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.w()?;
    drop(xv);
    drop(xg);
    Ok(new_tensor(y, &a, x.dims()))
}

/// [cos(t * exp(e_k)), sin(t * exp(e_k))] in bf16 for f32 timesteps t (n,) and exponents (half,)
pub fn ltx_timestep_sinusoid(t: &Tensor, expo: &Tensor) -> Result<Tensor> {
    let (n, half) = (t.elem_count(), expo.elem_count());
    let a = cuda_dev(t)?;
    let (tc, ec) = (t.contiguous()?, expo.contiguous()?);
    let (tg, tl) = tc.storage_and_layout();
    let (eg, el) = ec.storage_and_layout();
    let tv = on_gpu(&tg)?.as_cuda_slice::<f32>()?.slice(contiguous(tl, "t")?..);
    let ev = on_gpu(&eg)?.as_cuda_slice::<f32>()?.slice(contiguous(el, "exponents")?..);
    let y = unsafe { a.alloc::<half::bf16>(n * 2 * half)? };
    let func = a.get_or_load_custom_func("ltx_timestep_sinusoid", "rustyui_ops", PTX)?;
    let (ni, hi) = (n as i32, half as i32);
    let mut b = func.builder();
    b.arg(&tv);
    b.arg(&ev);
    b.arg(&y);
    b.arg(&ni);
    b.arg(&hi);
    unsafe { b.launch(LaunchConfig::for_num_elems((n * half) as u32)) }.w()?;
    drop((tv, ev));
    drop((tg, eg));
    Ok(new_tensor(y, &a, &[n, 2 * half]))
}

/// Cosmos-Predict2's VideoRopePosition3DEmb for a (t, h, w) grid of patches: the rotation matrices
/// (t * h * w, pairs, 2, 2) f32, from the per-axis exponents `ranges` (time's, then height's and
/// width's, each `dim_range` as the module holds it) and thetas (10000 times each NTK factor, as
/// floats). Each axis's frequencies are 1 / theta ** range; each angle is the position times one.
pub fn cosmos_rope_table(
    ranges: [&[f32]; 3],
    thetas: [f32; 3],
    (t, h, w): (usize, usize, usize),
    dev: &candle_core::Device,
) -> Result<Tensor> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("cosmos_rope_table: not on the GPU"),
    };
    let mut freqs = vec![];
    for (range, theta) in ranges.iter().zip(thetas) {
        let n = range.len();
        let r = Tensor::new(*range, dev)?;
        let (rg, rl) = r.storage_and_layout();
        let rv = f32_view(&rg, rl, "range")?;
        let f = unsafe { a.alloc::<f32>(n)? };
        let func = a.get_or_load_custom_func("cosmos_rope_freqs", "rustyui_ops", PTX)?;
        let ni = n as i32;
        let mut b = func.builder();
        b.arg(&rv).arg(&theta).arg(&f).arg(&ni);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        drop(rv);
        drop(rg);
        freqs.push(f32_tensor(f, &a, &[n]));
    }
    let all = Tensor::cat(&freqs, 0)?;
    let pairs = all.elem_count();
    let n = t * h * w * pairs;
    let (fg, fl) = all.storage_and_layout();
    let fv = f32_view(&fg, fl, "freqs")?;
    let out = unsafe { a.alloc::<f32>(n * 4)? };
    let func = a.get_or_load_custom_func("cosmos_rope_table", "rustyui_ops", PTX)?;
    let (ti, hi, wi) = (t as i32, h as i32, w as i32);
    let (dt, dh, dw) = (ranges[0].len() as i32, ranges[1].len() as i32, ranges[2].len() as i32);
    let mut b = func.builder();
    b.arg(&fv).arg(&out).arg(&ti).arg(&hi).arg(&wi).arg(&dt).arg(&dh).arg(&dw);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(fv);
    drop(fg);
    Ok(f32_tensor(out, &a, &[t * h * w, pairs, 2, 2]))
}

/// Anima's LLMAdapter RotaryEmbedding for positions 0..t: (cos, sin), each (t, 2 * half) bf16,
/// from its inv_freq (half f32 values)
pub fn anima_rotary_tables(inv_freq: &[f32], t: usize, dev: &candle_core::Device) -> Result<(Tensor, Tensor)> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("anima_rotary_tables: not on the GPU"),
    };
    let half = inv_freq.len();
    let inv = Tensor::new(inv_freq, dev)?;
    let (ig, il) = inv.storage_and_layout();
    let iv = f32_view(&ig, il, "inv_freq")?;
    let n = t * 2 * half;
    let (c, s) = unsafe { (a.alloc::<half::bf16>(n)?, a.alloc::<half::bf16>(n)?) };
    let func = a.get_or_load_custom_func("anima_rotary_tables", "rustyui_ops", PTX)?;
    let (ti, hi) = (t as i32, half as i32);
    let mut b = func.builder();
    b.arg(&iv).arg(&c).arg(&s).arg(&ti).arg(&hi);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(iv);
    drop(ig);
    Ok((new_tensor(c, &a, &[t, 2 * half]), new_tensor(s, &a, &[t, 2 * half])))
}

/// apply_rotary_pos_emb in bf16 on x (B, T, heads, D), tables (T, D) from anima_rotary_tables
pub fn anima_rotary_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, t, heads, d) = x.dims4()?;
    if cos.dims() != [t, d] || sin.dims() != [t, d] || x.dtype() != DType::BF16 {
        candle_core::bail!("anima_rotary_apply: x {:?} {:?}, tables {:?} {:?}", x.dims(), x.dtype(), cos.dims(), sin.dims());
    }
    let a = cuda_dev(x)?;
    let (xc, cc, sc) = (x.contiguous()?, cos.contiguous()?, sin.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (cg, cl) = cc.storage_and_layout();
    let (sg, sl) = sc.storage_and_layout();
    let (xv, cv, sv) = (gpu_view(&xg, xl, "x")?, gpu_view(&cg, cl, "cos")?, gpu_view(&sg, sl, "sin")?);
    let n = x.elem_count();
    let y = unsafe { a.alloc::<half::bf16>(n)? };
    let func = a.get_or_load_custom_func("anima_rotary_apply", "rustyui_ops", PTX)?;
    let (rows, di, hi, ti) = ((n / d) as i64, d as i32, heads as i32, t as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&cv).arg(&sv).arg(&y).arg(&rows).arg(&di).arg(&hi).arg(&ti);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((xv, cv, sv));
    drop((xg, cg, sg));
    Ok(new_tensor(y, &a, x.dims()))
}

/// The split-mode rotation matrix from f32 frequencies (rows, f): (rows, pad + f, 2, 2) bf16
pub fn ltx_rope_matrix(freqs: &Tensor, pad: usize) -> Result<Tensor> {
    let (rows, f) = freqs.dims2()?;
    let a = cuda_dev(freqs)?;
    let fc = freqs.contiguous()?;
    let (fg, fl) = fc.storage_and_layout();
    let fv = on_gpu(&fg)?.as_cuda_slice::<f32>()?.slice(contiguous(fl, "freqs")?..);
    let total = rows * (pad + f);
    let y = unsafe { a.alloc::<half::bf16>(total * 4)? };
    let func = a.get_or_load_custom_func("ltx_rope_matrix", "rustyui_ops", PTX)?;
    let (ri, fi, pi) = (rows as i64, f as i32, pad as i32);
    let mut b = func.builder();
    b.arg(&fv);
    b.arg(&y);
    b.arg(&ri);
    b.arg(&fi);
    b.arg(&pi);
    unsafe { b.launch(LaunchConfig::for_num_elems(total as u32)) }.w()?;
    drop(fv);
    drop(fg);
    Ok(new_tensor(y, &a, &[rows, pad + f, 2, 2]))
}

// ── torch's plain-cuBLAS mm: F.linear on a permuted view ──────────────────────
// F.linear fuses the bias (addmm on cuBLASLt) only for contiguous input. A permuted view, such as
// the patchified latent (channels-first memory read token by token), goes through matmul -> mm,
// which torch's default backend runs as cublasGemmEx with tensor ops and torch's own workspace
// (4096 KiB * 2 + 16 KiB * 8 off Hopper), rounds to bf16, and then adds the bias in its own kernel.

use candle_core::cuda_backend::cudarc::cublas::{result as blr, sys as bls};

const BLAS_WORKSPACE: usize = 4096 * 1024 * 2 + 16 * 1024 * 8;

/// A cuBLAS handle and the workspace torch gives it; destroyed with the thread that made it (the
/// server's worker), ahead of the CUDA context it lives in.
struct BlasHandle(bls::cublasHandle_t, CudaSlice<u8>);

impl Drop for BlasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = blr::destroy_handle(self.0);
        }
    }
}

thread_local! {
    static BLAS: std::cell::RefCell<Option<BlasHandle>> = const { std::cell::RefCell::new(None) };
}

fn blas_err(e: blr::CublasError) -> candle_core::Error {
    candle_core::Error::Msg(format!("cublas: {e:?}"))
}

/// x_cf (C, N) bf16 contiguous, read as the (N, C) view; w (out, C) bf16 -> (N, out) bf16, no bias
pub struct TorchMmChannelsFirst<'a> {
    pub w: &'a Tensor,
}

impl CustomOp1 for TorchMmChannelsFirst<'_> {
    fn name(&self) -> &'static str {
        "rustyui-torch-mm-cf"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("rustyui-torch-mm-cf runs on the GPU only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let (c, n) = xl.shape().dims2()?;
        let (out_f, wc) = self.w.dims2()?;
        if wc != c {
            candle_core::bail!("rustyui-torch-mm-cf: {c} channels against a weight of {out_f}x{wc}");
        }
        let dev = x.device();
        let stream = dev.cuda_stream();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(contiguous(xl, "x")?..);
        let (w_st, w_l) = self.w.storage_and_layout();
        let ws = on_gpu(&w_st)?.as_cuda_slice::<half::bf16>()?.slice(contiguous(w_l, "w")?..);
        let out = unsafe { dev.alloc::<half::bf16>(n * out_f)? };
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { dev.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (wp, _g1) = ws.device_ptr(&stream);
            let (xp, _g2) = xs.device_ptr(&stream);
            let (op, _g3) = out.device_ptr(&stream);
            let (kp, _g4) = work.device_ptr(&stream);
            unsafe {
                // getCurrentCUDABlasHandle: stream first (which resets the workspace), then the workspace
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, 0f32);
                let bf16 = bls::cudaDataType_t::CUDA_R_16BF;
                bls::cublasGemmEx(
                    *h,
                    bls::cublasOperation_t::CUBLAS_OP_T,
                    bls::cublasOperation_t::CUBLAS_OP_T,
                    out_f as i32,
                    n as i32,
                    c as i32,
                    &alpha as *const f32 as *const _,
                    wp as *const _,
                    bf16,
                    c as i32,
                    xp as *const _,
                    bf16,
                    n as i32,
                    &beta as *const f32 as *const _,
                    op as *mut _,
                    bf16,
                    out_f as i32,
                    bls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    bls::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(blas_err)?;
            }
            Ok(())
        })?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), Shape::from((n, out_f))))
    }
}

/// F.linear(x_cf viewed as (N, C), w, b) as torch runs it for that non-contiguous view
pub fn torch_linear_channels_first(x_cf: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let y = x_cf.contiguous()?.apply_op1_no_bwd(&TorchMmChannelsFirst { w })?;
    ltx_add(&y, b)
}

// ── checkpoint bytes onto the card through page-locked memory ──────────────────

use candle_core::cuda_backend::cudarc::driver::{result as drv, CudaEvent, DevicePtrMut, DeviceRepr, PinnedHostSlice};

/// A page-locked staging buffer in two halves, and for each half the event its last upload
/// recorded, so a half is refilled only once the card has read it
struct Staging {
    buf: PinnedHostSlice<u8>,
    done: [Option<CudaEvent>; 2],
    next: usize,
}

thread_local! {
    static STAGING: std::cell::RefCell<Option<Staging>> = const { std::cell::RefCell::new(None) };
}

const STAGE_HALF: usize = 64 << 20;

/// Where an upload's bytes come from
#[derive(Clone, Copy)]
pub enum UploadSrc<'a> {
    /// memory: a slice of a memory-mapped file, or bytes folded on the host
    Bytes(&'a [u8]),
    /// `len` bytes of a file from `offset`, read with pread: the kernel copies them from its page
    /// cache into the pinned buffer, and the process never maps those pages, so nothing has to
    /// be unmapped when the file is let go (reading a whole SDXL checkpoint through its mapping
    /// left 0.25 s of unmapping for the end of the session)
    File { file: &'a std::fs::File, offset: u64, len: usize },
}

impl UploadSrc<'_> {
    fn len(&self) -> usize {
        match self {
            UploadSrc::Bytes(b) => b.len(),
            UploadSrc::File { len, .. } => *len,
        }
    }

    /// bytes [off, off + dst.len()) of the source into `dst`, in pieces of at least 1 MiB on up to
    /// four threads, which take the page faults (or the reads) side by side
    fn copy_to(&self, off: usize, dst: &mut [u8]) -> Result<()> {
        let part = dst.len().div_ceil(4).max(1 << 20);
        let one = |at: usize, d: &mut [u8]| -> std::io::Result<()> {
            match self {
                UploadSrc::Bytes(b) => {
                    d.copy_from_slice(&b[at..at + d.len()]);
                    Ok(())
                }
                UploadSrc::File { file, offset, .. } => std::os::unix::fs::FileExt::read_exact_at(*file, d, offset + at as u64),
            }
        };
        let r = if dst.len() <= part {
            one(off, dst)
        } else {
            std::thread::scope(|sc| {
                let jobs: Vec<_> = dst.chunks_mut(part).enumerate().map(|(i, d)| sc.spawn(move || one(off + i * part, d))).collect();
                jobs.into_iter().try_for_each(|j| j.join().unwrap_or_else(|_| Err(std::io::Error::other("upload: a copy thread panicked"))))
            })
        };
        r.map_err(|e| candle_core::Error::Msg(format!("upload: reading the weights: {e}")))
    }
}

/// One checkpoint tensor's bytes (a slice of the memory-mapped file) onto the card as `dtype` of
/// `shape`. MmapedSafetensors::load copies them into a Vec and the driver stages that again, about
/// 4 GB/s; here they go a piece at a time into alternate halves of a pinned buffer, each piece
/// uploaded by DMA while the next is copied. Same bytes, same tensor.
pub fn upload(bytes: &[u8], dtype: DType, shape: &[usize], dev: &candle_core::Device) -> Result<Tensor> {
    upload_from(UploadSrc::Bytes(bytes), dtype, shape, dev)
}

/// upload, from memory or straight from the file (UploadSrc)
pub fn upload_from(src: UploadSrc, dtype: DType, shape: &[usize], dev: &candle_core::Device) -> Result<Tensor> {
    let d = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("upload: not a CUDA device"),
    };
    match dtype {
        DType::U8 => upload_as::<u8>(src, &d, shape),
        DType::BF16 => upload_as::<half::bf16>(src, &d, shape),
        DType::F16 => upload_as::<half::f16>(src, &d, shape),
        DType::F32 => upload_as::<f32>(src, &d, shape),
        other => candle_core::bail!("upload: {other:?} is not handled"),
    }
}

fn upload_as<T: candle_core::cuda_backend::CudaDType + DeviceRepr>(src: UploadSrc, d: &CudaDevice, shape: &[usize]) -> Result<Tensor> {
    let size = std::mem::size_of::<T>();
    let n_elem: usize = shape.iter().product();
    let len = src.len();
    if len != n_elem * size {
        candle_core::bail!("upload: {len} bytes for {shape:?} of {size}-byte values");
    }
    let stream = d.cuda_stream();
    let mut dst = unsafe { stream.alloc::<T>(n_elem) }.w()?;
    STAGING.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let buf = unsafe { stream.context().alloc_pinned::<u8>(2 * STAGE_HALF) }.w()?;
            *slot = Some(Staging { buf, done: [None, None], next: 0 });
        }
        let s = slot.as_mut().unwrap();
        let (base, _guard) = dst.device_ptr_mut(&stream);
        let mut off = 0;
        while off < len {
            let n = (len - off).min(STAGE_HALF);
            let h = s.next;
            s.next ^= 1;
            if let Some(e) = s.done[h].take() {
                e.synchronize().w()?;
            }
            let host = &mut s.buf.as_mut_slice().w()?[h * STAGE_HALF..h * STAGE_HALF + n];
            src.copy_to(off, host)?;
            unsafe { drv::memcpy_htod_async(base + off as u64, &host[..], stream.cu_stream()) }.w()?;
            s.done[h] = Some(stream.record_event(None).w()?);
            off += n;
        }
        Ok(())
    })?;
    let storage = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(dst, d.clone()));
    Ok(Tensor::from_storage(storage, shape, candle_core::op::BackpropOp::none(), false))
}

/// torch.randn(n) for a CUDA generator at (seed, offset), launched as torch launches it: blocks of
/// 256, `grid` of them (rng::CudaRandn works out the grid and the offset as torch does)
pub fn torch_randn(n: usize, seed: u64, offset: u64, grid: u32, dev: &candle_core::Device) -> Result<Tensor> {
    let d = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_randn: not a CUDA device"),
    };
    let out = unsafe { d.alloc::<f32>(n)? };
    let func = d.get_or_load_custom_func("torch_randn", "rustyui_ops", PTX)?;
    let mut b = func.builder();
    let n64 = n as i64;
    b.arg(&out);
    b.arg(&n64);
    b.arg(&seed);
    b.arg(&offset);
    unsafe { b.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, d));
    Ok(Tensor::from_storage(st, n, candle_core::op::BackpropOp::none(), false))
}

// ── LoRAs on the LTX DiT ─────────────────────────────────────────────────────────

/// torch.randint(0, 256, n, dtype=uint8) for a CUDA generator at (seed, offset), launched as torch
/// launches it (rng::CudaRandn works out the grid and the offset)
pub fn torch_randint_u8(n: usize, seed: u64, offset: u64, grid: u32, dev: &candle_core::Device) -> Result<Tensor> {
    let d = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_randint_u8: not a CUDA device"),
    };
    let out = unsafe { d.alloc::<u8>(n)? };
    let func = d.get_or_load_custom_func("torch_randint_u8", "rustyui_ops", PTX)?;
    let mut b = func.builder();
    let n64 = n as i64;
    b.arg(&out);
    b.arg(&n64);
    b.arg(&seed);
    b.arg(&offset);
    unsafe { b.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, d));
    Ok(Tensor::from_storage(st, n, candle_core::op::BackpropOp::none(), false))
}

/// comfy_kitchen's stochastic rounding of fp16 to e4m3 bytes with one random byte per value, as
/// its CUDA backend runs it (kernels/ck.cu), which ComfyUI uses under a cu130 torch, as the reference install's is
pub fn ck_stochastic_round_e4m3(w: &Tensor, rng: &Tensor) -> Result<Tensor> {
    let d = cuda_dev(w)?;
    let (w, rng) = (w.contiguous()?, rng.contiguous()?);
    let n = w.elem_count();
    if rng.elem_count() != n {
        candle_core::bail!("ck_stochastic_round_e4m3: {n} values, {} random bytes", rng.elem_count());
    }
    let out = unsafe { d.alloc::<u8>(n)? };
    {
        let (ws, wl) = w.storage_and_layout();
        let (rs, rl) = rng.storage_and_layout();
        let rv = on_gpu(&rs)?.as_cuda_slice::<u8>()?.slice(contiguous(rl, "rng")?..);
        let n64 = n as i64;
        match w.dtype() {
            DType::F16 => {
                let wv = on_gpu(&ws)?.as_cuda_slice::<half::f16>()?.slice(contiguous(wl, "w")?..);
                let func = d.get_or_load_custom_func("ck_stochastic_round_e4m3_f16", "ck", CK_PTX)?;
                let mut b = func.builder();
                b.arg(&wv).arg(&rv).arg(&out).arg(&n64);
                unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            }
            DType::BF16 => {
                let wv = gpu_view(&ws, wl, "w")?;
                let func = d.get_or_load_custom_func("ck_stochastic_round_e4m3_bf16", "ck", CK_PTX)?;
                let mut b = func.builder();
                b.arg(&wv).arg(&rv).arg(&out).arg(&n64);
                unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            }
            other => candle_core::bail!("ck_stochastic_round_e4m3: {other:?}"),
        }
    }
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, d));
    Ok(Tensor::from_storage(st, w.dims(), candle_core::op::BackpropOp::none(), false))
}

// ── cuDNN's attention with an additive bias (kernels/sdpa.cpp) ───────────────
// A masked attention in ComfyUI goes to torch's cuDNN SDPA backend (comfy.ops puts it right after
// FlashAttention, which takes no mask), whose kernels cuDNN generates and compiles with NVRTC at
// run time. sdpa.cpp builds that graph as torch 2.11 does, on the same cuDNN.

extern "C" {
    fn rustyui_sdpa_workspace(
        bf16: i32,
        q_dim: *const i64,
        q_stride: *const i64,
        k_dim: *const i64,
        k_stride: *const i64,
        v_dim: *const i64,
        v_stride: *const i64,
        o_dim: *const i64,
        o_stride: *const i64,
        bias_dim: *const i64,
        bias_stride: *const i64,
        workspace: *mut i64,
        err: *mut std::os::raw::c_char,
        err_len: i32,
    ) -> i32;
    fn rustyui_sdpa_forward(
        stream: *mut std::ffi::c_void,
        bf16: i32,
        q_dim: *const i64,
        q_stride: *const i64,
        k_dim: *const i64,
        k_stride: *const i64,
        v_dim: *const i64,
        v_stride: *const i64,
        o_dim: *const i64,
        o_stride: *const i64,
        bias_dim: *const i64,
        bias_stride: *const i64,
        scale: f32,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        bias: *mut std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        err: *mut std::os::raw::c_char,
        err_len: i32,
    ) -> i32;
}

/// cuDNN finds NVRTC (and NVRTC its builtins) only by name on the loader's path, which a user
/// prefix CUDA is not on: load both by path first, once, so the names resolve to them.
fn preload_nvrtc() -> Result<()> {
    static DONE: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    DONE.get_or_init(|| {
        let dir = std::path::Path::new(env!("RUSTYUI_CUDA_LIB"));
        let mut names = vec!["libnvrtc.so.12".to_string()];
        let mut builtins: Vec<String> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("libnvrtc-builtins.so.") && n.matches('.').count() == 3)
            .collect();
        builtins.sort();
        names.extend(builtins.into_iter().take(1));
        for n in names {
            let path = std::ffi::CString::new(dir.join(&n).to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
            let h = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if h.is_null() {
                return Err(format!("cannot load {}/{n}", dir.display()));
            }
        }
        Ok(())
    })
    .clone()
    .map_err(candle_core::Error::Msg)
}

/// One q, k, v, out or bias operand: a device pointer and torch's (B, H, S, D) sizes and strides
struct Operand {
    ptr: u64,
    dims: [i64; 4],
    strides: [i64; 4],
}

fn sdpa_call(dev: &CudaDevice, q: &Operand, k: &Operand, v: &Operand, o: &Operand, bias: Option<&Operand>, scale: f32) -> Result<()> {
    preload_nvrtc()?;
    let mut err = vec![0 as std::os::raw::c_char; 1024];
    let (bd, bs) = bias.map_or((std::ptr::null(), std::ptr::null()), |b| (b.dims.as_ptr(), b.strides.as_ptr()));
    let fail = |err: &[std::os::raw::c_char]| {
        let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().to_string();
        candle_core::Error::Msg(format!("cuDNN attention: {msg}"))
    };
    let mut ws = 0i64;
    let r = unsafe {
        rustyui_sdpa_workspace(
            1,
            q.dims.as_ptr(),
            q.strides.as_ptr(),
            k.dims.as_ptr(),
            k.strides.as_ptr(),
            v.dims.as_ptr(),
            v.strides.as_ptr(),
            o.dims.as_ptr(),
            o.strides.as_ptr(),
            bd,
            bs,
            &mut ws,
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if r != 0 {
        return Err(fail(&err));
    }
    let work = unsafe { dev.alloc::<u8>(ws.max(1) as usize)? };
    let stream = dev.cuda_stream();
    let (wp, _gw) = work.device_ptr(&stream);
    let p = |x: u64| x as *mut std::ffi::c_void;
    let r = unsafe {
        rustyui_sdpa_forward(
            stream.cu_stream() as *mut _,
            1,
            q.dims.as_ptr(),
            q.strides.as_ptr(),
            k.dims.as_ptr(),
            k.strides.as_ptr(),
            v.dims.as_ptr(),
            v.strides.as_ptr(),
            o.dims.as_ptr(),
            o.strides.as_ptr(),
            bd,
            bs,
            scale,
            p(q.ptr),
            p(k.ptr),
            p(v.ptr),
            p(o.ptr),
            bias.map_or(std::ptr::null_mut(), |b| p(b.ptr)),
            p(wp),
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if r != 0 {
        return Err(fail(&err));
    }
    Ok(())
}

/// _attention_with_guide_mask: q, k, v (1, T, heads * d) bf16; the queries before `guide_start`
/// attend with `noisy` (1, 1, 1, T) added to every row, the next `tracked` rows with
/// `tracked_mask` (1, 1, tracked, T), each as its own cuDNN call over all the keys, the way
/// torch hands them over: the (B, S, H, D) buffers seen as (B, H, S, D), the bias expanded to
/// the query rows, the output a fresh (B, S, H, D) buffer per call.
pub fn guide_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    guide_start: usize,
    noisy: &Tensor,
    tracked_mask: &Tensor,
) -> Result<Tensor> {
    let (b, t, inner) = q.dims3()?;
    let tracked = tracked_mask.dim(2)?;
    if b != 1 || k.dims() != q.dims() || v.dims() != q.dims() || guide_start + tracked != t {
        candle_core::bail!(
            "guide_attention: q {:?} k {:?} v {:?}, guides from {guide_start}, {tracked} tracked",
            q.dims(),
            k.dims(),
            v.dims()
        );
    }
    if noisy.dims() != [1, 1, 1, t] || tracked_mask.dims() != [1, 1, tracked, t] {
        candle_core::bail!("guide_attention: masks {:?} and {:?} for {t} tokens", noisy.dims(), tracked_mask.dims());
    }
    let d = inner / heads;
    let dev = cuda_dev(q)?;
    let stream = dev.cuda_stream();
    let (qc, kc, vc, nc, tc) = (q.contiguous()?, k.contiguous()?, v.contiguous()?, noisy.contiguous()?, tracked_mask.contiguous()?);
    let out = unsafe { dev.alloc::<half::bf16>(t * inner)? };
    {
        let (qs, ql) = qc.storage_and_layout();
        let (ks, kl) = kc.storage_and_layout();
        let (vs, vl) = vc.storage_and_layout();
        let (ns, nl) = nc.storage_and_layout();
        let (ts, tl) = tc.storage_and_layout();
        let (qv, kv, vv, nv, tv) = (
            gpu_view(&qs, ql, "q")?,
            gpu_view(&ks, kl, "k")?,
            gpu_view(&vs, vl, "v")?,
            gpu_view(&ns, nl, "noisy")?,
            gpu_view(&ts, tl, "tracked")?,
        );
        let (qp, _g1) = qv.device_ptr(&stream);
        let (kp, _g2) = kv.device_ptr(&stream);
        let (vp, _g3) = vv.device_ptr(&stream);
        let (np, _g4) = nv.device_ptr(&stream);
        let (tp, _g5) = tv.device_ptr(&stream);
        let (op, _g6) = out.device_ptr(&stream);
        let (h, d, t, inner) = (heads as i64, d as i64, t as i64, inner as i64);
        // a slice of the (1, T, inner) buffer keeps its batch stride; heads are d apart, rows inner
        let view = |ptr: u64, rows: i64, batch_stride: i64| Operand { ptr, dims: [1, h, rows, d], strides: [batch_stride, d, inner, 1] };
        let kk = view(kp, t, t * inner);
        let vv = view(vp, t, t * inner);
        // torch's own scale: 1 / sqrt(d) in double, then a float
        let scale = (1.0f64 / (d as f64).sqrt()) as f32;
        let row = (inner * 2) as u64;
        let gs = guide_start as i64;
        if gs > 0 {
            let qn = view(qp, gs, t * inner);
            let on = view(op, gs, gs * inner);
            let bias = Operand { ptr: np, dims: [1, 1, gs, t], strides: [t, t, 0, 1] };
            sdpa_call(&dev, &qn, &kk, &vv, &on, Some(&bias), scale)?;
        }
        let g = tracked as i64;
        let qt = view(qp + gs as u64 * row, g, t * inner);
        let ot = view(op + gs as u64 * row, g, g * inner);
        let bias = Operand { ptr: tp, dims: [1, 1, g, t], strides: [g * t, g * t, t, 1] };
        sdpa_call(&dev, &qt, &kk, &vv, &ot, Some(&bias), scale)?;
    }
    Ok(new_tensor(out, &dev, &[b, t, inner]))
}

// ── cuDNN convolutions as torch runs them (kernels/conv.cpp) ──────────────────
// torch 2.11 builds each convolution as a cuDNN v8 operation graph and, with benchmarking off as
// ComfyUI leaves it, runs the first engine config its instant heuristics give that works; conv.cpp
// does the same, so the kernel, and with it every rounding, is torch's.

extern "C" {
    fn rustyui_conv(
        stream: *mut std::ffi::c_void,
        op: i32,
        dtype: i32,
        nhwc: i32,
        nd: i32,
        x_dim: *const i64,
        x_stride: *const i64,
        w_dim: *const i64,
        w_stride: *const i64,
        y_dim: *const i64,
        y_stride: *const i64,
        pad: *const i64,
        stride: *const i64,
        dil: *const i64,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        err: *mut std::os::raw::c_char,
        err_len: i32,
    ) -> i32;
    fn rustyui_conv_plans(out: *mut std::os::raw::c_char, out_len: i32) -> i32;
}

/// The tags of the cuDNN plans the convolutions settled on so far, one per line
pub fn cudnn_conv_plans() -> String {
    let mut buf = vec![0 as std::os::raw::c_char; 1 << 16];
    unsafe { rustyui_conv_plans(buf.as_mut_ptr(), buf.len() as i32) };
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().to_string()
}

/// torch's cuDNN convolution without its bias (F.conv2d / F.conv3d add the bias afterwards, as a
/// broadcast add in the output's dtype): x (N, C, [D,] H, W), w (O, C / groups, [kd,] kh, kw), bf16,
/// f16 or f32, both made contiguous as torch makes them; the output contiguous.
pub fn cudnn_conv(x: &Tensor, w: &Tensor, padding: &[usize], stride: &[usize], dilation: &[usize]) -> Result<Tensor> {
    conv_impl(x, w, padding, stride, dilation, None, false)
}

/// cudnn_conv as torch runs it on a channels-last input (cudnn_conv_suggest_memory_format: the
/// input, the weight and the output all channels-last, on cuDNN's NHWC engines, which round
/// otherwise). The output is the same NCHW tensor, a view of NHWC memory.
pub fn cudnn_conv_nhwc(x: &Tensor, w: &Tensor, padding: &[usize], stride: &[usize], dilation: &[usize]) -> Result<Tensor> {
    conv_impl(x, w, padding, stride, dilation, None, true)
}

/// torch's cuDNN transposed convolution (cudnn_convolution_transpose: cuDNN's backward-data
/// convolution) without its bias: x (N, K, ...) against w (K, C / groups, ...), the output
/// (N, C, ...) with each side (in - 1) * stride - 2 * padding + dilation * (k - 1) +
/// output_padding + 1
pub fn cudnn_conv_transpose(
    x: &Tensor,
    w: &Tensor,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    output_padding: &[usize],
    groups: usize,
) -> Result<Tensor> {
    conv_impl(x, w, padding, stride, dilation, Some((output_padding, groups)), false)
}

/// cudnn_conv_transpose on a channels-last input, as cudnn_conv_nhwc
pub fn cudnn_conv_transpose_nhwc(
    x: &Tensor,
    w: &Tensor,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    output_padding: &[usize],
    groups: usize,
) -> Result<Tensor> {
    conv_impl(x, w, padding, stride, dilation, Some((output_padding, groups)), true)
}

fn conv_impl(
    x: &Tensor,
    w: &Tensor,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    transpose: Option<(&[usize], usize)>,
    channels_last: bool,
) -> Result<Tensor> {
    let nd = x.rank();
    if !(nd == 4 || nd == 5) || w.rank() != nd || x.dtype() != w.dtype() {
        candle_core::bail!("cudnn_conv: x {:?} {:?}, w {:?} {:?}", x.dims(), x.dtype(), w.dims(), w.dtype());
    }
    let ns = nd - 2;
    if padding.len() != ns || stride.len() != ns || dilation.len() != ns {
        candle_core::bail!("cudnn_conv: {ns} spatial dimensions, padding {padding:?}, stride {stride:?}, dilation {dilation:?}");
    }
    // channels-last memory: (N, [D,] H, W, C), the channels innermost
    let last = |t: &Tensor| -> Result<Tensor> {
        let mut order: Vec<usize> = vec![0];
        order.extend(2..t.rank());
        order.push(1);
        t.permute(order)?.contiguous()
    };
    let (xc, wc) = if channels_last { (last(x)?, last(w)?) } else { (x.contiguous()?, w.contiguous()?) };
    let (xd, wd) = (x.dims().to_vec(), w.dims().to_vec());
    let mut yd = match transpose {
        None => vec![xd[0], wd[0]],
        Some((_, groups)) => vec![xd[0], wd[1] * groups],
    };
    for i in 0..ns {
        let span = dilation[i] * (wd[2 + i] - 1) + 1;
        match transpose {
            None => {
                if xd[2 + i] + 2 * padding[i] < span {
                    candle_core::bail!("cudnn_conv: kernel {:?} larger than the padded input {:?}", wd, xd);
                }
                yd.push((xd[2 + i] + 2 * padding[i] - span) / stride[i] + 1);
            }
            Some((op, _)) => yd.push((xd[2 + i] - 1) * stride[i] + span + op[i] - 2 * padding[i]),
        }
    }
    let strides = |d: &[usize]| -> Vec<i64> {
        let mut s = vec![1i64; d.len()];
        if channels_last {
            // C, then the spatial dims from the last, then N
            let mut z = d[1] as i64;
            for i in (2..d.len()).rev() {
                s[i] = z;
                z *= d[i] as i64;
            }
            s[0] = z;
            return s;
        }
        for i in (0..d.len() - 1).rev() {
            s[i] = s[i + 1] * d[i + 1] as i64;
        }
        s
    };
    let as_i64 = |v: &[usize]| -> Vec<i64> { v.iter().map(|&u| u as i64).collect() };
    let (xdi, wdi, ydi) = (as_i64(&xd), as_i64(&wd), as_i64(&yd));
    let (xs, ws, ys) = (strides(&xd), strides(&wd), strides(&yd));
    let (pad, st, dil) = (as_i64(padding), as_i64(stride), as_i64(dilation));
    let dev = cuda_dev(x)?;
    let stream = dev.cuda_stream();
    let n: usize = yd.iter().product();
    let mut err = vec![0 as std::os::raw::c_char; 1024];
    macro_rules! run {
        ($t:ty, $code:expr) => {{
            let out = unsafe { dev.alloc::<$t>(n)? };
            {
                let (xg, xl) = xc.storage_and_layout();
                let (wg, wl) = wc.storage_and_layout();
                let xv = on_gpu(&xg)?.as_cuda_slice::<$t>()?.slice(contiguous(xl, "x")?..);
                let wv = on_gpu(&wg)?.as_cuda_slice::<$t>()?.slice(contiguous(wl, "w")?..);
                let (xp, _g1) = xv.device_ptr(&stream);
                let (wp, _g2) = wv.device_ptr(&stream);
                let (yp, _g3) = out.device_ptr(&stream);
                // a transposed convolution is cuDNN's backward-data one: its input plays the
                // forward's output ('y') and its output the forward's input ('x')
                let r = unsafe {
                    if transpose.is_none() {
                        rustyui_conv(
                            stream.cu_stream() as *mut _,
                            0,
                            $code,
                            channels_last as i32,
                            nd as i32,
                            xdi.as_ptr(),
                            xs.as_ptr(),
                            wdi.as_ptr(),
                            ws.as_ptr(),
                            ydi.as_ptr(),
                            ys.as_ptr(),
                            pad.as_ptr(),
                            st.as_ptr(),
                            dil.as_ptr(),
                            xp as *mut _,
                            wp as *mut _,
                            yp as *mut _,
                            err.as_mut_ptr(),
                            err.len() as i32,
                        )
                    } else {
                        rustyui_conv(
                            stream.cu_stream() as *mut _,
                            1,
                            $code,
                            channels_last as i32,
                            nd as i32,
                            ydi.as_ptr(),
                            ys.as_ptr(),
                            wdi.as_ptr(),
                            ws.as_ptr(),
                            xdi.as_ptr(),
                            xs.as_ptr(),
                            pad.as_ptr(),
                            st.as_ptr(),
                            dil.as_ptr(),
                            yp as *mut _,
                            wp as *mut _,
                            xp as *mut _,
                            err.as_mut_ptr(),
                            err.len() as i32,
                        )
                    }
                };
                if r != 0 {
                    let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().to_string();
                    candle_core::bail!("cudnn_conv: {msg}");
                }
            }
            let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone()));
            if channels_last {
                // the NHWC memory as (N, [D,] H, W, C), seen as (N, C, [D,] H, W)
                let mut mem: Vec<usize> = vec![yd[0]];
                mem.extend_from_slice(&yd[2..]);
                mem.push(yd[1]);
                let mut back: Vec<usize> = vec![0, nd - 1];
                back.extend(1..nd - 1);
                Tensor::from_storage(st, mem.as_slice(), candle_core::op::BackpropOp::none(), false).permute(back)?
            } else {
                Tensor::from_storage(st, yd.as_slice(), candle_core::op::BackpropOp::none(), false)
            }
        }};
    }
    Ok(match x.dtype() {
        DType::BF16 => run!(half::bf16, 0),
        DType::F16 => run!(half::f16, 1),
        DType::F32 => run!(f32, 2),
        other => candle_core::bail!("cudnn_conv: {other:?}"),
    })
}

/// The LTX VAE's per-channel statistics on a bf16 latent (N, C, ...), torch's rounding:
/// `normalize` (x - mean) / std, else x * std + mean, each step rounded to bf16 (ops.cu)
pub fn ltx_channel_affine(x: &Tensor, std: &Tensor, mean: &Tensor, normalize: bool) -> Result<Tensor> {
    let c = x.dim(1)?;
    if std.elem_count() != c || mean.elem_count() != c || x.dtype() != DType::BF16 {
        candle_core::bail!(
            "ltx_channel_affine: x {:?} {:?}, {} and {} statistics",
            x.dims(),
            x.dtype(),
            std.elem_count(),
            mean.elem_count()
        );
    }
    let n = x.elem_count();
    let inner = (n / x.dim(0)? / c) as i64;
    let a = cuda_dev(x)?;
    let (xc, sc, mc) = (x.contiguous()?, std.to_dtype(DType::BF16)?.contiguous()?, mean.to_dtype(DType::BF16)?.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (sg, sl) = sc.storage_and_layout();
    let (mg, ml) = mc.storage_and_layout();
    let (xv, sv, mv) = (gpu_view(&xg, xl, "x")?, gpu_view(&sg, sl, "std")?, gpu_view(&mg, ml, "mean")?);
    let y = unsafe { a.alloc::<half::bf16>(n)? };
    let func = a.get_or_load_custom_func("ltx_channel_affine", "rustyui_ops", PTX)?;
    let (nn, ci, norm) = (n as i64, c as i32, normalize as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&sv).arg(&mv).arg(&y).arg(&nn).arg(&ci).arg(&inner).arg(&norm);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((xv, sv, mv));
    drop((xg, sg, mg));
    Ok(new_tensor(y, &a, x.dims()))
}

/// torch's native depthwise convolution (conv_depthwise2d, which an f32 depthwise Conv1d takes
/// instead of cuDNN) for x (N, C, L) and a (C, 1, K) filter, no padding, no bias
pub fn torch_depthwise_conv1d(x: &Tensor, w: &Tensor, stride: usize) -> Result<Tensor> {
    let (nb, c, lin) = x.dims3()?;
    let k = w.dim(2)?;
    if w.dims() != [c, 1, k] || x.dtype() != DType::F32 || w.dtype() != DType::F32 || lin < k {
        candle_core::bail!("torch_depthwise_conv1d: x {:?} {:?}, w {:?} {:?}", x.dims(), x.dtype(), w.dims(), w.dtype());
    }
    let lout = (lin - k) / stride + 1;
    let n = nb * c * lout;
    let a = cuda_dev(x)?;
    let (xc, wc) = (x.contiguous()?, w.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (wg, wl) = wc.storage_and_layout();
    let xv = on_gpu(&xg)?.as_cuda_slice::<f32>()?.slice(contiguous(xl, "x")?..);
    let wv = on_gpu(&wg)?.as_cuda_slice::<f32>()?.slice(contiguous(wl, "w")?..);
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func("torch_depthwise_conv1d_f32", "rustyui_ops", PTX)?;
    let (ci, li, lo, ki, si, pi, nn) = (c as i32, lin as i32, lout as i32, k as i32, stride as i32, 0i32, n as i64);
    let mut b = func.builder();
    b.arg(&xv).arg(&wv).arg(&y).arg(&ci).arg(&li).arg(&lo).arg(&ki).arg(&si).arg(&pi).arg(&nn);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((xv, wv));
    drop((xg, wg));
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(y, a));
    Ok(Tensor::from_storage(st, (nb, c, lout), candle_core::op::BackpropOp::none(), false))
}

/// The card's multiprocessor count and threads per multiprocessor, which torch's reduction layout
/// depends on (setReduceConfig's target grid)
pub(crate) fn sm_info() -> Result<(usize, usize)> {
    use candle_core::cuda_backend::cudarc::driver::{sys::CUdevice_attribute as A, CudaContext};
    static INFO: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    if let Some(i) = INFO.get() {
        return Ok(*i);
    }
    let ctx = CudaContext::new(0).w()?;
    let sms = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).w()? as usize;
    let threads = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).w()? as usize;
    Ok(*INFO.get_or_init(|| (sms, threads)))
}

/// How torch's CUDA reduction lays out a sum over the middle of a contiguous (outer, R, inner)
/// tensor, inner > 1, so the reduced dimension is not the fastest-moving one (Reduce.cuh
/// setReduceConfig, its "vectorize along output" case): the block height, whether the R inputs are
/// split across the block's warps, and the CTAs each output is split across (a global reduction).
/// Together they fix the order of the sum.
fn torch_reduce_config(outer: usize, r: usize, inner: usize) -> Result<(usize, bool, usize)> {
    let num_outputs = outer * inner;
    // get_output_vec_size (fresh tensors are aligned, so the sizes decide)
    let mut vec = 4usize;
    while !inner.is_multiple_of(vec) {
        vec /= 2;
    }
    // set_block_dimension, MAX_NUM_THREADS 512
    let max_threads = 512 / vec;
    let last_pow2 = |n: usize| -> usize { 1usize << (usize::BITS - 1 - n.max(1).leading_zeros()) };
    let (dim0, dim1) = (num_outputs / vec, r);
    let d0 = if dim0 < max_threads { last_pow2(dim0) } else { max_threads };
    let d1 = if dim1 < max_threads { last_pow2(dim1) } else { max_threads };
    let bh = d1.min(max_threads / d0.min(32));
    let bw = d0.min(max_threads / bh);
    // outputs across the lanes; the inputs across the warps when each thread keeps 16 or more
    let split = r >= (bh * 16).min(256);
    let step_output = if split { bw } else { bw * bh };
    let mut ctas = 1;
    let per_thread = r.div_ceil(bh);
    if split && per_thread >= 256 {
        let (sms, threads_per_sm) = sm_info()?;
        let target = sms * (threads_per_sm / (bw * bh));
        let grid = (num_outputs / vec).div_ceil(step_output);
        if grid <= target {
            ctas = target.div_ceil(grid).min(per_thread.div_ceil(16)).max(per_thread.div_ceil(256));
        }
    }
    Ok((bh, split, ctas))
}

/// One pass of ops.cu's torch_mean_dim1 over dim `dim` of a contiguous bf16 or f32 tensor (mode
/// 0 the mean, 1 PixelNorm's root); the result keeps that dimension, as size 1
fn torch_mean_launch(x: &Tensor, dim: usize, mode: i32) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    if dim >= dims.len() {
        candle_core::bail!("torch_mean: dim {dim} of {dims:?}");
    }
    let outer: usize = dims[..dim].iter().product();
    let r = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    if inner < 2 || r == 0 {
        candle_core::bail!("torch_mean: {dims:?} over {dim} reduces along the fastest dimension, which torch lays out otherwise");
    }
    let (bh, split, ctas) = torch_reduce_config(outer, r, inner)?;
    // MeanOps' factor: num_output_elements / numel, as floats; none for the norm
    let factor = if mode == 2 { 1.0 } else { (outer * inner) as f32 / (outer * r * inner) as f32 };
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let n = outer * inner;
    let (oi, ri, ii, bhi, sp, ci) = (outer as i64, r as i32, inner as i64, bh as i32, split as i32, ctas as i32);
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let st = match x.dtype() {
        DType::BF16 => {
            let xv = on_gpu(&xg)?.as_cuda_slice::<half::bf16>()?.slice(contiguous(xl, "x")?..);
            let y = unsafe { a.alloc::<half::bf16>(n)? };
            let func = a.get_or_load_custom_func("torch_mean_dim1_bf16", "rustyui_ops", PTX)?;
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&oi).arg(&ri).arg(&ii).arg(&bhi).arg(&sp).arg(&ci).arg(&factor).arg(&mode);
            unsafe { b.launch(cfg) }.w()?;
            drop(xv);
            CudaStorage::wrap_cuda_slice(y, a)
        }
        DType::F32 => {
            let xv = on_gpu(&xg)?.as_cuda_slice::<f32>()?.slice(contiguous(xl, "x")?..);
            let y = unsafe { a.alloc::<f32>(n)? };
            let func = a.get_or_load_custom_func("torch_mean_dim1_f32", "rustyui_ops", PTX)?;
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&oi).arg(&ri).arg(&ii).arg(&bhi).arg(&sp).arg(&ci).arg(&factor).arg(&mode);
            unsafe { b.launch(cfg) }.w()?;
            drop(xv);
            CudaStorage::wrap_cuda_slice(y, a)
        }
        DType::F16 => {
            let xv = on_gpu(&xg)?.as_cuda_slice::<half::f16>()?.slice(contiguous(xl, "x")?..);
            let y = unsafe { a.alloc::<half::f16>(n)? };
            let func = a.get_or_load_custom_func("torch_mean_dim1_f16", "rustyui_ops", PTX)?;
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&oi).arg(&ri).arg(&ii).arg(&bhi).arg(&sp).arg(&ci).arg(&factor).arg(&mode);
            unsafe { b.launch(cfg) }.w()?;
            drop(xv);
            CudaStorage::wrap_cuda_slice(y, a)
        }
        other => candle_core::bail!("torch_mean: {other:?}"),
    };
    drop(xg);
    let mut out = dims;
    out[dim] = 1;
    Ok(Tensor::from_storage(candle_core::Storage::Cuda(st), out, candle_core::op::BackpropOp::none(), false))
}

/// torch.mean(x, dim=1, keepdim=True) for a contiguous bf16 or f32 tensor (N, C, ...), summed in
/// the order torch's CUDA reduction sums it, the result in x's dtype
pub fn torch_mean_dim1(x: &Tensor) -> Result<Tensor> {
    torch_mean_launch(x, 1, 0)
}

/// torch.mean(x, dim, keepdim=True), the same, over any dimension but the last
pub fn torch_mean(x: &Tensor, dim: usize) -> Result<Tensor> {
    torch_mean_launch(x, dim, 0)
}

/// kornia's canny magnitude from its gradients (B, 2, H, W) f32: (B, 1, H, W)
pub fn kornia_canny_magnitude(grad: &Tensor) -> Result<Tensor> {
    let (b, two, h, w) = grad.dims4()?;
    if two != 2 || grad.dtype() != DType::F32 {
        candle_core::bail!("kornia_canny_magnitude: grad {:?} {:?}", grad.dims(), grad.dtype());
    }
    let a = cuda_dev(grad)?;
    let gc = grad.contiguous()?;
    let (gg, gl) = gc.storage_and_layout();
    let gv = f32_view(&gg, gl, "grad")?;
    let n = b * h * w;
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func("kornia_canny_magnitude", "rustyui_ops", PTX)?;
    let (hw, ni) = ((h * w) as i64, n as i64);
    let mut bl = func.builder();
    bl.arg(&gv).arg(&y).arg(&hw).arg(&ni);
    unsafe { bl.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(gv);
    drop(gg);
    Ok(f32_tensor(y, &a, &[b, 1, h, w]))
}

/// kornia's canny from its gradients (B, 2, H, W) and NMS responses (B, 8, H, W), both f32: the
/// thresholded edges before hysteresis, 0, 0.5 (weak) or 1 (strong), (B, H*W) on the host
pub fn kornia_canny_classify(grad: &Tensor, nms: &Tensor, lo: f32, hi: f32) -> Result<Vec<f32>> {
    let (b, two, h, w) = grad.dims4()?;
    let (b2, eight, h2, w2) = nms.dims4()?;
    if two != 2 || eight != 8 || (b, h, w) != (b2, h2, w2) || grad.dtype() != DType::F32 || nms.dtype() != DType::F32 {
        candle_core::bail!("kornia_canny_classify: grad {:?} {:?}, nms {:?} {:?}", grad.dims(), grad.dtype(), nms.dims(), nms.dtype());
    }
    let a = cuda_dev(grad)?;
    let (gc, nc) = (grad.contiguous()?, nms.contiguous()?);
    let (gg, gl) = gc.storage_and_layout();
    let (ng, nl) = nc.storage_and_layout();
    let (gv, nv) = (f32_view(&gg, gl, "grad")?, f32_view(&ng, nl, "nms")?);
    let n = b * h * w;
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func("kornia_canny_classify", "rustyui_ops", PTX)?;
    let (hw, ni) = ((h * w) as i64, n as i64);
    let mut bl = func.builder();
    bl.arg(&gv).arg(&nv).arg(&y).arg(&hw).arg(&ni).arg(&lo).arg(&hi);
    unsafe { bl.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((gv, nv));
    drop((gg, ng));
    f32_tensor(y, &a, &[n]).to_vec1::<f32>()
}

/// `x * torch.sigmoid(1.702 * x)` on f16 as torch runs it (ops.cu's torch_quick_gelu_f16)
pub fn torch_quick_gelu_f16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::F16 {
        candle_core::bail!("torch_quick_gelu_f16: {:?}", x.dtype());
    }
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let n = xc.elem_count();
    let (xg, xl) = xc.storage_and_layout();
    let xv = on_gpu(&xg)?.as_cuda_slice::<half::f16>()?.slice(contiguous(xl, "x")?..);
    let y = unsafe { a.alloc::<half::f16>(n)? };
    let func = a.get_or_load_custom_func("torch_quick_gelu_f16", "rustyui_ops", PTX)?;
    let ni = n as i64;
    let mut b = func.builder();
    b.arg(&xv).arg(&y).arg(&ni);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(xv);
    drop(xg);
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(y, a));
    Ok(Tensor::from_storage(st, x.dims(), candle_core::op::BackpropOp::none(), false))
}

/// ComfyUI's PixelNorm (x / sqrt(mean(x**2, dim=1) + 1e-8)) on a contiguous bf16 (N, C, ...) as
/// torch runs it in bf16: the squares, the mean (summed in torch's order), the sum with eps, the root
/// and the quotient each rounded to bf16; then nn.SiLU of that when `silu` (ops.cu)
pub fn ltx_pixel_norm(x: &Tensor, silu: bool) -> Result<Tensor> {
    if x.dtype() != DType::BF16 || x.rank() < 3 {
        candle_core::bail!("ltx_pixel_norm: {:?} {:?}", x.dims(), x.dtype());
    }
    let xc = x.contiguous()?;
    let root = torch_mean_launch(&xc, 1, 1)?;
    let (n, c) = (xc.elem_count(), xc.dim(1)?);
    let inner = (n / xc.dim(0)? / c) as i64;
    let a = cuda_dev(x)?;
    let (xg, xl) = xc.storage_and_layout();
    let (rg, rl) = root.storage_and_layout();
    let (xv, rv) = (gpu_view(&xg, xl, "x")?, gpu_view(&rg, rl, "root")?);
    let y = unsafe { a.alloc::<half::bf16>(n)? };
    let func = a.get_or_load_custom_func("ltx_pixel_norm_apply", "rustyui_ops", PTX)?;
    let (nn, ci, si) = (n as i64, c as i32, silu as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&rv).arg(&y).arg(&nn).arg(&ci).arg(&inner).arg(&si);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((xv, rv));
    drop((xg, rg));
    Ok(new_tensor(y, &a, x.dims()))
}

/// The Wan VAE's RMS_norm (F.normalize(x, dim=1) * C ** 0.5 * gamma + 0) on a contiguous bf16
/// (N, C, ...) as torch runs it in bf16: the L2 norm over C summed in torch's order, then each op
/// rounded to bf16 in turn; then nn.SiLU of that when `silu` (ops.cu). `gamma` holds C values.
pub fn wan_rms_norm(x: &Tensor, gamma: &Tensor, silu: bool) -> Result<Tensor> {
    if x.dtype() != DType::BF16 || gamma.dtype() != DType::BF16 || x.rank() < 3 || gamma.elem_count() != x.dim(1)? {
        candle_core::bail!("wan_rms_norm: x {:?} {:?}, gamma {:?} {:?}", x.dims(), x.dtype(), gamma.dims(), gamma.dtype());
    }
    let xc = x.contiguous()?;
    let norm = torch_mean_launch(&xc, 1, 2)?;
    let gamma = gamma.flatten_all()?.contiguous()?;
    let (n, c) = (xc.elem_count(), xc.dim(1)?);
    let inner = (n / xc.dim(0)? / c) as i64;
    // RMS_norm's self.scale = dim ** 0.5 (libm's pow, as CPython's float power), a Python float the
    // kernel gets as a float
    let scale = (c as f64).powf(0.5) as f32;
    let a = cuda_dev(x)?;
    let (xg, xl) = xc.storage_and_layout();
    let (ng, nl) = norm.storage_and_layout();
    let (gg, gl) = gamma.storage_and_layout();
    let (xv, nv, gv) = (gpu_view(&xg, xl, "x")?, gpu_view(&ng, nl, "norm")?, gpu_view(&gg, gl, "gamma")?);
    let y = unsafe { a.alloc::<half::bf16>(n)? };
    let func = a.get_or_load_custom_func("wan_rms_norm_apply", "rustyui_ops", PTX)?;
    let (nn, ci, si) = (n as i64, c as i32, silu as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&nv).arg(&gv).arg(&y).arg(&nn).arg(&ci).arg(&inner).arg(&scale).arg(&si);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((xv, nv, gv));
    drop((xg, ng, gg));
    Ok(new_tensor(y, &a, x.dims()))
}

// ── Gemma 3 in f32 (ComfyUI's LTX-AV text encoder) ──────────────────────────────

fn f32_view<'a>(st: &'a candle_core::Storage, l: &Layout, what: &str) -> Result<CudaView<'a, f32>> {
    Ok(on_gpu(st)?.as_cuda_slice::<f32>()?.slice(contiguous(l, what)?..))
}

fn f32_tensor(out: CudaSlice<f32>, dev: &CudaDevice, shape: &[usize]) -> Tensor {
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone()));
    Tensor::from_storage(st, shape, candle_core::op::BackpropOp::none(), false)
}

/// F.rms_norm(x, (N,), w, eps) for f32 x (..., N) and w (N), torch's fused kernel (ops.cu)
pub fn torch_rms_norm_f32(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let n = x.dim(candle_core::D::Minus1)?;
    if x.dtype() != DType::F32 || w.dtype() != DType::F32 || w.elem_count() != n || n % 4 != 0 {
        candle_core::bail!("torch_rms_norm_f32: x {:?} {:?}, w {:?} {:?}", x.dims(), x.dtype(), w.dims(), w.dtype());
    }
    let rows = x.elem_count() / n;
    let a = cuda_dev(x)?;
    let (xc, wc) = (x.contiguous()?, w.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (wg, wl) = wc.storage_and_layout();
    let (xv, wv) = (f32_view(&xg, xl, "x")?, f32_view(&wg, wl, "w")?);
    let y = unsafe { a.alloc::<f32>(rows * n)? };
    let func = a.get_or_load_custom_func("torch_rms_norm_f32", "rustyui_ops", PTX)?;
    let (ni, eps) = (n as i32, eps);
    let mut b = func.builder();
    b.arg(&xv).arg(&wv).arg(&y).arg(&ni).arg(&eps);
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.w()?;
    drop((xv, wv));
    drop((xg, wg));
    Ok(f32_tensor(y, &a, x.dims()))
}

/// Gemma's RoPE tables for positions 0..t at one theta and scale, as ComfyUI computes them on the
/// card: (cos, sin), each (t, dim / 2)
pub fn gemma_rope_tables(t: usize, dim: usize, theta: f32, scale: f32, dev: &candle_core::Device) -> Result<(Tensor, Tensor)> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("gemma_rope_tables: not on the GPU"),
    };
    let half = dim / 2;
    let n = t * half;
    let (c, s) = unsafe { (a.alloc::<f32>(n)?, a.alloc::<f32>(n)?) };
    let func = a.get_or_load_custom_func("gemma_rope_tables", "rustyui_ops", PTX)?;
    let (ti, hi, inv) = (t as i32, half as i32, 1.0f32 / scale);
    let mut b = func.builder();
    b.arg(&c).arg(&s).arg(&ti).arg(&hi).arg(&theta).arg(&inv);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    Ok((f32_tensor(c, &a, &[t, half]), f32_tensor(s, &a, &[t, half])))
}

/// comfy_kitchen's eager split-half RoPE on f32 x (B, H, T, D) contiguous with tables (T, D / 2)
pub fn gemma_rope_f32(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, t, d) = x.dims4()?;
    if x.dtype() != DType::F32 || cos.dims() != [t, d / 2] || sin.dims() != [t, d / 2] {
        candle_core::bail!("gemma_rope_f32: x {:?}, tables {:?} {:?}", x.dims(), cos.dims(), sin.dims());
    }
    let a = cuda_dev(x)?;
    let (xc, cc, sc) = (x.contiguous()?, cos.contiguous()?, sin.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (cg, cl) = cc.storage_and_layout();
    let (sg, sl) = sc.storage_and_layout();
    let (xv, cv, sv) = (f32_view(&xg, xl, "x")?, f32_view(&cg, cl, "cos")?, f32_view(&sg, sl, "sin")?);
    let n = x.elem_count();
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func("gemma_rope_f32", "rustyui_ops", PTX)?;
    let (rows, ti, hi) = ((n / d) as i64, t as i32, (d / 2) as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&cv).arg(&sv).arg(&y).arg(&rows).arg(&ti).arg(&hi);
    unsafe { b.launch(LaunchConfig::for_num_elems((n / 2) as u32)) }.w()?;
    drop((xv, cv, sv));
    drop((xg, cg, sg));
    Ok(f32_tensor(y, &a, x.dims()))
}

/// F.gelu(gate, approximate="tanh") * up in f32, torch's GELU (ops.cu)
pub fn torch_gelu_tanh_mul_f32(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.dtype() != DType::F32 || gate.dims() != up.dims() {
        candle_core::bail!("torch_gelu_tanh_mul_f32: {:?} {:?} and {:?}", gate.dims(), gate.dtype(), up.dims());
    }
    let a = cuda_dev(gate)?;
    let (gc, uc) = (gate.contiguous()?, up.contiguous()?);
    let (gg, gl) = gc.storage_and_layout();
    let (ug, ul) = uc.storage_and_layout();
    let (gv, uv) = (f32_view(&gg, gl, "gate")?, f32_view(&ug, ul, "up")?);
    let n = gate.elem_count();
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func("torch_gelu_tanh_mul_f32", "rustyui_ops", PTX)?;
    let ni = n as i64;
    let mut b = func.builder();
    b.arg(&gv).arg(&uv).arg(&y).arg(&ni);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((gv, uv));
    drop((gg, ug));
    Ok(f32_tensor(y, &a, gate.dims()))
}

/// F.linear(x, w) without a bias for f32 x (rows, k) and w (n, k): what torch's mm makes of x @ w.t()
/// (cublasCommonArgs: the result row-major, so cuBLAS computes its transpose), cublasSgemm with
/// op(A) = T on w, op(B) = N on x, default math (TF32 off, ComfyUI's setting) and torch's workspace
pub fn torch_linear_f32(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (n, k2) = w.dims2()?;
    if k != k2 || x.dtype() != DType::F32 || w.dtype() != DType::F32 {
        candle_core::bail!("torch_linear_f32: x {:?} {:?}, w {:?} {:?}", x.dims(), x.dtype(), w.dims(), w.dtype());
    }
    let d = cuda_dev(x)?;
    let stream = d.cuda_stream();
    let (x, w) = (x.contiguous()?, w.contiguous()?);
    let out = unsafe { d.alloc::<f32>(rows * n)? };
    {
        let (xst, xl) = x.storage_and_layout();
        let (wst, wl) = w.storage_and_layout();
        let xv = f32_view(&xst, xl, "x")?;
        let wv = f32_view(&wst, wl, "w")?;
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { d.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (wp, _g1) = wv.device_ptr(&stream);
            let (xp, _g2) = xv.device_ptr(&stream);
            let (op, _g3) = out.device_ptr(&stream);
            let (kp, _g4) = work.device_ptr(&stream);
            unsafe {
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, 0f32);
                bls::cublasSgemm_v2(
                    *h,
                    bls::cublasOperation_t::CUBLAS_OP_T,
                    bls::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    rows as i32,
                    k as i32,
                    &alpha,
                    wp as *const f32,
                    k as i32,
                    xp as *const f32,
                    k as i32,
                    &beta,
                    op as *mut f32,
                    n as i32,
                )
                .result()
                .map_err(blas_err)?;
            }
            Ok(())
        })?;
    }
    Ok(f32_tensor(out, &d, &[rows, n]))
}

extern "C" {
    fn rustyui_mem_eff_attention(
        dtype: i32,
        stream: *mut std::ffi::c_void,
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        accum: *mut f32,
        batches: i32,
        heads: i32,
        queries: i32,
        keys: i32,
        head_dim: i32,
        q_strides: *const i64,
        k_strides: *const i64,
        v_strides: *const i64,
        bias_strides: *const i64,
        scale: f32,
        err: *mut std::os::raw::c_char,
        err_len: i32,
    ) -> i32;
}

/// How q, k and v sit in memory for the mem-efficient attention: (B, H, T, K) contiguous (Gemma's,
/// normed per head), or (B, T, H * K) contiguous (a projection's output, CLIP's, the SDXL VAE's)
#[derive(Clone, Copy, PartialEq)]
pub enum HeadLayout {
    Bhtk,
    Btk,
}

/// A contiguous f32 or bf16 tensor's device address, and the guard that keeps it valid
fn device_addr<'a>(
    st: &'a candle_core::Storage,
    l: &Layout,
    what: &str,
    stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> Result<(u64, candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>)> {
    let off = contiguous(l, what)? as u64;
    let g = on_gpu(st)?;
    Ok(match g.dtype() {
        DType::F32 => {
            let (p, guard) = g.as_cuda_slice::<f32>()?.device_ptr(stream);
            (p + off * 4, guard)
        }
        DType::BF16 => {
            let (p, guard) = g.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
            (p + off * 2, guard)
        }
        other => candle_core::bail!("{what}: {other:?} is neither f32 nor bf16"),
    })
}

/// SDPA's memory-efficient backend (torch's CUTLASS FMHA, kernels/mem_eff.cu, the variant torch's
/// dispatch picks for the head size) on f32 or bf16 q, k, v laid out as `layout` says, `heads`
/// heads, with an additive mask (M, N) of their type broadcast over batch and heads as SDPA expands
/// it, its rows first padded to a multiple of 8 floats as SDPA's preprocess_mask pads an unaligned
/// mask, and the default scale 1/sqrt(K). The result is (B, M, H * K) in q's type, the layout
/// attention_pytorch returns.
pub fn torch_mem_eff_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    layout: HeadLayout,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, m, n, kd) = match layout {
        HeadLayout::Bhtk => {
            let (b, h, m, kd) = q.dims4()?;
            let (_, h2, n, _) = k.dims4()?;
            if h2 != h || h != heads {
                candle_core::bail!("torch_mem_eff_attention: q {:?}, k {:?}, {heads} heads", q.dims(), k.dims());
            }
            (b, h, m, n, kd)
        }
        HeadLayout::Btk => {
            let (b, m, w) = q.dims3()?;
            let (_, n, _) = k.dims3()?;
            (b, heads, m, n, w / heads)
        }
    };
    let dt = q.dtype();
    let code = match dt {
        DType::F32 => 0,
        DType::BF16 => 1,
        _ => candle_core::bail!("torch_mem_eff_attention: {dt:?} (f32 and bf16 are built)"),
    };
    if v.dims() != k.dims() || k.dtype() != dt || v.dtype() != dt {
        candle_core::bail!("torch_mem_eff_attention: q {:?}, k {:?}, v {:?}", q.dims(), k.dims(), v.dims());
    }
    let d = cuda_dev(q)?;
    let stream = d.cuda_stream();
    let (qc, kc, vc) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    // the BMHK views SDPA hands the kernel: batch, sequence, head strides
    let strides = |t: usize| -> [i64; 3] {
        match layout {
            HeadLayout::Bhtk => [(h * t * kd) as i64, kd as i64, (t * kd) as i64],
            HeadLayout::Btk => [(t * h * kd) as i64, (h * kd) as i64, kd as i64],
        }
    };
    let (qs, ks) = (strides(m), strides(n));
    // preprocess_mask: rows padded with zeros to a multiple of 8 floats, then expanded over batch
    // and heads (stride 0 for both, the batch's stride being the one SDPA's view keeps)
    // a (H, M, N) mask is one per head (SAM3's box bias), its rows padded likewise
    let per_head = mask.is_some_and(|mk| mk.rank() == 3);
    let mask = match mask {
        Some(mk) => {
            if (mk.dims() != [m, n] && mk.dims() != [h, m, n]) || mk.dtype() != dt {
                candle_core::bail!("torch_mem_eff_attention: mask {:?} {:?} for {h} heads, {m} by {n}", mk.dims(), mk.dtype());
            }
            let padded = n.div_ceil(8) * 8;
            let last = mk.rank() - 1;
            Some(if padded == n {
                mk.contiguous()?
            } else {
                Tensor::cat(&[mk, &Tensor::zeros(&[&mk.dims()[..last], &[padded - n]].concat()[..], dt, mk.device())?], last)?
            })
        }
        None => None,
    };
    let bs = mask.as_ref().map_or([0i64; 3], |mk| {
        let row = mk.dim(mk.rank() - 1).unwrap_or(n) as i64;
        [mk.elem_count() as i64, if per_head { m as i64 * row } else { 0 }, row]
    });
    let scale = (1.0f64 / (kd as f64).sqrt()) as f32;
    let out = Tensor::zeros((b, m, h * kd), dt, q.device())?;
    // the bf16 kernels that loop over the value's columns accumulate in an f32 buffer
    let accum = if dt == DType::BF16 { Some(Tensor::zeros(b * m * h * kd, DType::F32, q.device())?) } else { None };
    let mut err = vec![0 as std::os::raw::c_char; 512];
    {
        let (qg, ql) = qc.storage_and_layout();
        let (kg, kl) = kc.storage_and_layout();
        let (vg, vl) = vc.storage_and_layout();
        let (og, ol) = out.storage_and_layout();
        let (qp, _g1) = device_addr(&qg, ql, "q", &stream)?;
        let (kp, _g2) = device_addr(&kg, kl, "k", &stream)?;
        let (vp, _g3) = device_addr(&vg, vl, "v", &stream)?;
        let (op, _g4) = device_addr(&og, ol, "out", &stream)?;
        let mask_store = mask.as_ref().map(|t| t.storage_and_layout());
        let mask_addr = match &mask_store {
            Some((g, l)) => Some(device_addr(g, l, "mask", &stream)?),
            None => None,
        };
        let bp = mask_addr.as_ref().map_or(std::ptr::null(), |(p, _)| *p as *const std::ffi::c_void);
        let accum_store = accum.as_ref().map(|t| t.storage_and_layout());
        let accum_addr = match &accum_store {
            Some((g, l)) => Some(device_addr(g, l, "accum", &stream)?),
            None => None,
        };
        let ap = accum_addr.as_ref().map_or(std::ptr::null_mut(), |(p, _)| *p as *mut f32);
        let r = unsafe {
            rustyui_mem_eff_attention(
                code,
                stream.cu_stream() as *mut _,
                qp as *const _,
                kp as *const _,
                vp as *const _,
                bp,
                op as *mut _,
                ap,
                b as i32,
                h as i32,
                m as i32,
                n as i32,
                kd as i32,
                qs.as_ptr(),
                ks.as_ptr(),
                ks.as_ptr(),
                bs.as_ptr(),
                scale,
                err.as_mut_ptr(),
                err.len() as i32,
            )
        };
        if r != 0 {
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().to_string();
            candle_core::bail!("{msg}");
        }
    }
    Ok(out)
}

/// Gemma's call: (B, H, T, K) q, k, v and a (1, 1, M, N) mask
pub fn torch_mem_eff_attention_f32(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let heads = q.dim(1)?;
    let mask = match mask {
        Some(mk) => Some(mk.squeeze(0)?.squeeze(0)?),
        None => None,
    };
    torch_mem_eff_attention(q, k, v, heads, HeadLayout::Bhtk, mask.as_ref())
}

/// F.linear(x, w, b) for f32 x (rows, k), w (n, k), b (n), as torch's addmm runs it: cuBLASLt with
/// the bias epilogue (gemm_and_bias), f32 compute (TF32 off), torch's workspace and alignments
pub fn torch_linear_bias_f32(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    linear_bias_f32(x, w, bias, false)
}

/// torch_linear_bias_f32 on TF32 tensor cores, as torch runs an f32 linear once
/// torch.backends.cuda.matmul.allow_tf32 is set (SAM3's import sets it for the whole process)
pub fn torch_linear_bias_f32_tf32(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    linear_bias_f32(x, w, bias, true)
}

fn linear_bias_f32(x: &Tensor, w: &Tensor, bias: &Tensor, tf32: bool) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (n, k2) = w.dims2()?;
    if k != k2 || bias.elem_count() != n || x.dtype() != DType::F32 || w.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        candle_core::bail!("torch_linear_bias_f32: x {:?} {:?}, w {:?}, bias {:?}", x.dims(), x.dtype(), w.dims(), bias.dims());
    }
    let dev = cuda_dev(x)?;
    let stream = dev.cuda_stream();
    let (x, w, bias) = (x.contiguous()?, w.contiguous()?, bias.contiguous()?);
    let out = unsafe { dev.alloc::<f32>(rows * n)? };
    {
        let (xs, xl) = x.storage_and_layout();
        let (ws, wl) = w.storage_and_layout();
        let (bs, bl) = bias.storage_and_layout();
        let (xv, wv, bv) = (f32_view(&xs, xl, "x")?, f32_view(&ws, wl, "w")?, f32_view(&bs, bl, "bias")?);
        let (a, _g1) = wv.device_ptr(&stream);
        let (b, _g2) = xv.device_ptr(&stream);
        let (d, _g3) = out.device_ptr(&stream);
        let (bp, _g4) = bv.device_ptr(&stream);
        let f32t = lts::cudaDataType_t::CUDA_R_32F;
        lt_gemm(&dev, &Gemm { k, n, rows, ab: f32t, dt: f32t, a, b, d, bias: Some(bp), scales: None, tf32 })?;
    }
    Ok(f32_tensor(out, &dev, &[rows, n]))
}

/// F.layer_norm over the last dimension of f32 x (rows, N) with w and b (N), torch's fused kernel
/// (ops.cu: Welford, as vectorized_layer_norm_kernel)
pub fn torch_layer_norm_f32(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let n = x.dim(candle_core::D::Minus1)?;
    if x.dtype() != DType::F32
        || w.dtype() != DType::F32
        || b.dtype() != DType::F32
        || w.elem_count() != n
        || b.elem_count() != n
        || n % 4 != 0
    {
        candle_core::bail!("torch_layer_norm_f32: x {:?} {:?}, w {:?}, b {:?}", x.dims(), x.dtype(), w.dims(), b.dims());
    }
    let rows = x.elem_count() / n;
    let a = cuda_dev(x)?;
    let (xc, wc, bc) = (x.contiguous()?, w.contiguous()?, b.contiguous()?);
    let (xg, xl) = xc.storage_and_layout();
    let (wg, wl) = wc.storage_and_layout();
    let (bg, bl) = bc.storage_and_layout();
    let (xv, wv, bv) = (f32_view(&xg, xl, "x")?, f32_view(&wg, wl, "w")?, f32_view(&bg, bl, "b")?);
    let y = unsafe { a.alloc::<f32>(rows * n)? };
    let func = a.get_or_load_custom_func("torch_layer_norm_f32", "rustyui_ops", PTX)?;
    let ni = n as i32;
    let mut lb = func.builder();
    lb.arg(&xv).arg(&wv).arg(&bv).arg(&y).arg(&ni).arg(&eps);
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
    unsafe { lb.launch(cfg) }.w()?;
    drop((xv, wv, bv));
    drop((xg, wg, bg));
    Ok(f32_tensor(y, &a, x.dims()))
}

/// CLIP's activations on f32 as torch computes them: `quick` a * sigmoid(1.702 * a) (three ops), else
/// F.gelu (erf)
pub fn torch_clip_act_f32(x: &Tensor, quick: bool) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        candle_core::bail!("torch_clip_act_f32: {:?}", x.dtype());
    }
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let n = x.elem_count();
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func(if quick { "torch_quick_gelu_f32" } else { "torch_gelu_erf_f32" }, "rustyui_ops", PTX)?;
    let ni = n as i64;
    let mut b = func.builder();
    b.arg(&xv).arg(&y).arg(&ni);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, x.dims()))
}

// ── LoRAs as ComfyUI's DynamicVRAM loader applies them ─────────────────────────

/// comfy_kitchen's dequantize_per_tensor_fp8 to bf16: e4m3 bytes times the f32 scale, rounded once
pub fn ck_dequant_e4m3_bf16(q: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let d = cuda_dev(q)?;
    let (q, scale) = (q.contiguous()?, scale.contiguous()?);
    let n = q.elem_count();
    let out = unsafe { d.alloc::<half::bf16>(n)? };
    {
        let (qs, ql) = q.storage_and_layout();
        let (ss, sl) = scale.storage_and_layout();
        let qv = on_gpu(&qs)?.as_cuda_slice::<u8>()?.slice(contiguous(ql, "q")?..);
        let sv = f32_view(&ss, sl, "scale")?;
        let func = d.get_or_load_custom_func("ck_dequant_e4m3_bf16", "ck", CK_PTX)?;
        let n64 = n as i64;
        let mut b = func.builder();
        b.arg(&qv).arg(&sv).arg(&out).arg(&n64);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    }
    Ok(new_tensor(out, &d, q.dims()))
}

/// comfy_kitchen's quantize_per_tensor_fp8 from bf16 with one f32 scale: e4m3 bytes, same shape
pub fn ck_quantize_e4m3_bf16(w: &Tensor, scale: &Tensor) -> Result<Tensor> {
    if w.dtype() != DType::BF16 || scale.dtype() != DType::F32 || scale.elem_count() != 1 {
        candle_core::bail!("ck_quantize_e4m3_bf16: {:?} {:?}, scale {:?} {:?}", w.dims(), w.dtype(), scale.dims(), scale.dtype());
    }
    let d = cuda_dev(w)?;
    let (w, scale) = (w.contiguous()?, scale.contiguous()?);
    let n = w.elem_count();
    let out = unsafe { d.alloc::<u8>(n)? };
    {
        let (ws, wl) = w.storage_and_layout();
        let (ss, sl) = scale.storage_and_layout();
        let wv = gpu_view(&ws, wl, "w")?;
        let sv = f32_view(&ss, sl, "scale")?;
        let func = d.get_or_load_custom_func("ck_quantize_e4m3_bf16", "ck", CK_PTX)?;
        let n64 = n as i64;
        let mut b = func.builder();
        b.arg(&wv).arg(&out).arg(&sv).arg(&n64);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    }
    let st = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(out, d));
    Ok(Tensor::from_storage(st, w.dims(), candle_core::op::BackpropOp::none(), false))
}

/// torch.mm(a, b) for bf16 a (m, k) and b (k, n): see torch_mm
pub fn torch_mm_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != DType::BF16 {
        candle_core::bail!("torch_mm_bf16: {:?}", a.dtype());
    }
    torch_mm(a, b)
}

/// torch.mm(a, b) for a (m, k) and b (k, n) of one dtype, as torch's mm runs it (cublasCommonArgs on
/// a row-major result: ops N, N on b then a): f16 and bf16 through cublasGemmEx with f32 compute,
/// the default tensor-op algorithm and reduced-precision reductions allowed (torch's defaults for
/// both), f32 through cublasSgemm with TF32 off; torch's workspace
pub fn torch_mm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (m, k) = a.dims2()?;
    let (k2, n) = b.dims2()?;
    let dt = a.dtype();
    let ty = match dt {
        DType::BF16 => bls::cudaDataType_t::CUDA_R_16BF,
        DType::F16 => bls::cudaDataType_t::CUDA_R_16F,
        DType::F32 => bls::cudaDataType_t::CUDA_R_32F,
        _ => candle_core::bail!("torch_mm: {dt:?}"),
    };
    if k != k2 || b.dtype() != dt {
        candle_core::bail!("torch_mm: {:?} {:?} by {:?} {:?}", a.dims(), dt, b.dims(), b.dtype());
    }
    let d = cuda_dev(a)?;
    let stream = d.cuda_stream();
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let out = Tensor::zeros((m, n), dt, a.device())?;
    {
        let (ast, al) = a.storage_and_layout();
        let (bst, bl) = b.storage_and_layout();
        let (ost, ol) = out.storage_and_layout();
        let (ap, _g1) = device_addr_any(&ast, al, "a", &stream)?;
        let (bp, _g2) = device_addr_any(&bst, bl, "b", &stream)?;
        let (op, _g3) = device_addr_any(&ost, ol, "out", &stream)?;
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { d.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (kp, _g4) = work.device_ptr(&stream);
            unsafe {
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, 0f32);
                let (opn, (n, m, k)) = (bls::cublasOperation_t::CUBLAS_OP_N, (n as i32, m as i32, k as i32));
                if dt == DType::F32 {
                    bls::cublasSgemm_v2(*h, opn, opn, n, m, k, &alpha, bp as *const f32, n, ap as *const f32, k, &beta, op as *mut f32, n)
                        .result()
                        .map_err(blas_err)?;
                } else {
                    bls::cublasGemmEx(
                        *h,
                        opn,
                        opn,
                        n,
                        m,
                        k,
                        &alpha as *const f32 as *const _,
                        bp as *const _,
                        ty,
                        n,
                        ap as *const _,
                        ty,
                        k,
                        &beta as *const f32 as *const _,
                        op as *mut _,
                        ty,
                        n,
                        bls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        bls::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                    .result()
                    .map_err(blas_err)?;
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// A contiguous f16, bf16 or f32 tensor's device address, and the guard that keeps it valid
fn device_addr_any<'a>(
    st: &'a candle_core::Storage,
    l: &Layout,
    what: &str,
    stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> Result<(u64, candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>)> {
    if on_gpu(st)?.dtype() == DType::F16 {
        let off = contiguous(l, what)? as u64;
        let (p, guard) = on_gpu(st)?.as_cuda_slice::<half::f16>()?.device_ptr(stream);
        return Ok((p + off * 2, guard));
    }
    device_addr(st, l, what, stream)
}

/// nn.SiLU as torch computes it for f16, bf16 or f32: in f32, rounded once (ops.cu)
pub fn torch_silu(x: &Tensor) -> Result<Tensor> {
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let n = x.elem_count();
    let ni = n as i64;
    let (xg, xl) = xc.storage_and_layout();
    let name = format!("torch_silu_{}", suffix(x.dtype())?);
    let func = a.get_or_load_custom_func(&name, "rustyui_ops", PTX)?;
    let st = match x.dtype() {
        DType::F16 => {
            let xv = on_gpu(&xg)?.as_cuda_slice::<half::f16>()?.slice(contiguous(xl, "x")?..);
            let y = unsafe { a.alloc::<half::f16>(n)? };
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&ni);
            unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            CudaStorage::wrap_cuda_slice(y, a.clone())
        }
        DType::BF16 => {
            let xv = gpu_view(&xg, xl, "x")?;
            let y = unsafe { a.alloc::<half::bf16>(n)? };
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&ni);
            unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            CudaStorage::wrap_cuda_slice(y, a.clone())
        }
        _ => {
            let xv = f32_view(&xg, xl, "x")?;
            let y = unsafe { a.alloc::<f32>(n)? };
            let mut b = func.builder();
            b.arg(&xv).arg(&y).arg(&ni);
            unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            CudaStorage::wrap_cuda_slice(y, a.clone())
        }
    };
    drop(xg);
    Ok(Tensor::from_storage(candle_core::Storage::Cuda(st), x.dims(), candle_core::op::BackpropOp::none(), false))
}

/// The unary math torch's samplers apply to 0-dim CUDA tensors.
#[derive(Clone, Copy)]
pub enum UnaryMath {
    Log = 0,
    Exp = 1,
    Expm1 = 2,
    /// torch.logit without eps
    Logit = 3,
}

/// torch's f32 log / exp / expm1 kernels on the card, element by element (CUDA's logf, expf and
/// expm1f, which round differently from libm's): the samplers' scalar steps
pub fn torch_unary_math_f32(x: &[f32], op: UnaryMath, dev: &candle_core::Device) -> Result<Vec<f32>> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_unary_math_f32: not on the GPU"),
    };
    if x.is_empty() {
        return Ok(vec![]);
    }
    let xt = Tensor::new(x, dev)?;
    let (xg, xl) = xt.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let out = unsafe { a.alloc::<f32>(x.len())? };
    let func = a.get_or_load_custom_func("torch_unary_math_f32", "rustyui_ops", PTX)?;
    let (ni, opi) = (x.len() as i32, op as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&out).arg(&ni).arg(&opi);
    unsafe { b.launch(LaunchConfig::for_num_elems(x.len() as u32)) }.w()?;
    drop(xv);
    drop(xg);
    f32_tensor(out, &a, &[x.len()]).to_vec1::<f32>()
}

/// comfy's timestep_embedding(t, dim) on the card, f32 (B, dim) (max period 10000, cos half first)
pub fn torch_timestep_embedding(t: &[f32], dim: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_timestep_embedding: not on the GPU"),
    };
    let tt = Tensor::new(t, dev)?;
    let (tg, tl) = tt.storage_and_layout();
    let tv = f32_view(&tg, tl, "t")?;
    let half = dim / 2;
    let out = unsafe { a.alloc::<f32>(t.len() * dim)? };
    let func = a.get_or_load_custom_func("torch_timestep_embedding", "rustyui_ops", PTX)?;
    let (bi, hi, nl) = (t.len() as i32, half as i32, (-(10000f64.ln())) as f32);
    let mut b = func.builder();
    b.arg(&tv).arg(&out).arg(&bi).arg(&hi).arg(&nl);
    unsafe { b.launch(LaunchConfig::for_num_elems((t.len() * half) as u32)) }.w()?;
    drop(tv);
    drop(tg);
    Ok(f32_tensor(out, &a, &[t.len(), dim]))
}

// ── RIFE: torch's own f32 kernels (ops.cu) ─────────────────────────────────────

/// F.interpolate(x, (out_h, out_w), mode="bilinear", align_corners=False) on contiguous f32 NCHW,
/// as torch's CUDA upsample_bilinear2d computes it. `scales` are the scale factors when the call
/// gave them (each side's kernel scale is then 1 / factor, rounded to f32), else in / out. The
/// same size is a copy, as torch makes it.
pub fn torch_upsample_bilinear2d_f32(x: &Tensor, out_h: usize, out_w: usize, scales: Option<(f64, f64)>) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if x.dtype() != DType::F32 {
        candle_core::bail!("torch_upsample_bilinear2d_f32: {:?}", x.dtype());
    }
    if (h, w) == (out_h, out_w) {
        return x.copy();
    }
    let side = |s: Option<f64>, i: usize, o: usize| match s {
        Some(s) if s > 0.0 => (1.0 / s) as f32,
        _ => i as f32 / o as f32,
    };
    let (rh, rw) = (side(scales.map(|s| s.0), h, out_h), side(scales.map(|s| s.1), w, out_w));
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<f32>(b * c * out_h * out_w)? };
    let func = a.get_or_load_custom_func("torch_upsample_bilinear2d_f32", "rustyui_ops", PTX)?;
    let n = (out_h * out_w) as i32;
    let dims = [b as i32, c as i32, h as i32, w as i32, out_h as i32, out_w as i32];
    let mut bl = func.builder();
    bl.arg(&xv).arg(&y).arg(&n);
    for d in &dims {
        bl.arg(d);
    }
    bl.arg(&rh).arg(&rw);
    let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(1024), 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    unsafe { bl.launch(cfg) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, &[b, c, out_h, out_w]))
}

/// F.interpolate(x, (out_h, out_w), mode="bilinear", align_corners=True) on contiguous f32 NCHW, as
/// torch's CUDA kernel computes it (source index scale * dst, scale (in - 1) / (out - 1) in f32)
pub fn torch_upsample_bilinear2d_ac_f32(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    if x.dtype() != DType::F32 {
        candle_core::bail!("torch_upsample_bilinear2d_ac_f32: {:?}", x.dtype());
    }
    if (h, w) == (out_h, out_w) {
        return x.copy();
    }
    let side = |i: usize, o: usize| if o > 1 { (i - 1) as f32 / (o - 1) as f32 } else { 0.0 };
    let (rh, rw) = (side(h, out_h), side(w, out_w));
    unary_launch4("torch_upsample_bilinear2d_ac_f32", x, out_h, out_w, rh, rw)
}

/// F.interpolate(x, scale_factor=(sh, sw), mode="bicubic", align_corners=False) on contiguous f32
/// NCHW to (out_h, out_w), torch's CUDA kernel (each side's scale 1 / factor in f32, as given)
pub fn torch_upsample_bicubic2d_f32(x: &Tensor, out_h: usize, out_w: usize, scales: (f64, f64)) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        candle_core::bail!("torch_upsample_bicubic2d_f32: {:?}", x.dtype());
    }
    unary_launch4("torch_upsample_bicubic2d_f32", x, out_h, out_w, (1.0 / scales.0) as f32, (1.0 / scales.1) as f32)
}

/// A ops.cu resampling kernel's launch: one thread per output position, every sample and channel
fn unary_launch4(name: &str, x: &Tensor, out_h: usize, out_w: usize, rh: f32, rw: f32) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<f32>(b * c * out_h * out_w)? };
    let func = a.get_or_load_custom_func(name, "rustyui_ops", PTX)?;
    let n = (out_h * out_w) as i32;
    let dims = [b as i32, c as i32, h as i32, w as i32, out_h as i32, out_w as i32];
    let mut bl = func.builder();
    bl.arg(&xv).arg(&y).arg(&n);
    for d in &dims {
        bl.arg(d);
    }
    bl.arg(&rh).arg(&rw);
    let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(1024), 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    unsafe { bl.launch(cfg) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, &[b, c, out_h, out_w]))
}

/// nn.GELU() (the erf form) on f32, torch's CUDA kernel (the CLIP port's)
pub fn torch_gelu_erf_f32(x: &Tensor) -> Result<Tensor> {
    torch_clip_act_f32(x, false)
}

/// x.softmax(dim=-1) on a contiguous f32 tensor whose rows have at most 9216 values, as
/// host_softmax runs it: its persistent warp softmax up to 2048 (128 threads a block), past that
/// cunn_SoftMaxForwardReg (1024 threads a row; torch has other kernels past 9216)
pub fn torch_softmax_lastdim_f32(x: &Tensor) -> Result<Tensor> {
    let cols = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / cols.max(1);
    if x.dtype() != DType::F32 || cols == 0 || cols > 9216 {
        candle_core::bail!("torch_softmax_lastdim_f32: {:?} {:?}: rows of 1 to 9216 f32 values", x.dims(), x.dtype());
    }
    if cols > 2048 {
        let a = cuda_dev(x)?;
        let xc = x.contiguous()?;
        let (xg, xl) = xc.storage_and_layout();
        let xv = f32_view(&xg, xl, "x")?;
        let y = unsafe { a.alloc::<f32>(rows * cols)? };
        let func = a.get_or_load_custom_func("torch_softmax_reg_f32", "rustyui_ops", PTX)?;
        let (c, regs) = (cols as i64, cols.div_ceil(1024) as i32);
        let mut bl = func.builder();
        bl.arg(&xv).arg(&y).arg(&c).arg(&regs);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        unsafe { bl.launch(cfg) }.w()?;
        drop(xv);
        drop(xg);
        return Ok(f32_tensor(y, &a, x.dims()));
    }
    let log2 = usize::BITS - (cols - 1).leading_zeros();
    let log2 = if cols == 1 { 0 } else { log2 };
    let np2 = 1usize << log2;
    let (warp, batch) = (np2.min(32), if np2 <= 128 { 2 } else { 1 });
    let warps_per_block = 128 / warp;
    let blocks = rows.div_ceil(warps_per_block * batch);
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<f32>(rows * cols)? };
    let func = a.get_or_load_custom_func(&format!("torch_softmax_warp_f32_{log2}"), "rustyui_ops", PTX)?;
    let (ri, ci) = (rows as i32, cols as i32);
    let mut bl = func.builder();
    bl.arg(&xv).arg(&y).arg(&ri).arg(&ci);
    let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (warp as u32, warps_per_block as u32, 1), shared_mem_bytes: 0 };
    unsafe { bl.launch(cfg) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, x.dims()))
}

/// torch's bgemm<float> as baddbmm_out_cuda_impl calls it: cublasSgemmStridedBatched on torch's
/// handle settings (default math, TF32 off; torch's workspace), the operands as raw f32 storage
/// with the leading dimensions and batch strides the caller worked out the way torch does.
/// Returns C's storage, `c_len` values.
#[allow(clippy::too_many_arguments)]
pub fn torch_bgemm_f32(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &Tensor,
    lda: usize,
    stride_a: usize,
    b: &Tensor,
    ldb: usize,
    stride_b: usize,
    ldc: usize,
    stride_c: usize,
    batches: usize,
    c_len: usize,
) -> Result<Tensor> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        candle_core::bail!("torch_bgemm_f32: {:?} {:?}", a.dtype(), b.dtype());
    }
    let d = cuda_dev(a)?;
    let stream = d.cuda_stream();
    let out = Tensor::zeros(c_len, DType::F32, a.device())?;
    {
        let (ast, al) = a.storage_and_layout();
        let (bst, bl) = b.storage_and_layout();
        let (ost, ol) = out.storage_and_layout();
        let (ap, _g1) = device_addr_any(&ast, al, "a", &stream)?;
        let (bp, _g2) = device_addr_any(&bst, bl, "b", &stream)?;
        let (op, _g3) = device_addr_any(&ost, ol, "out", &stream)?;
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { d.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (kp, _g4) = work.device_ptr(&stream);
            let op_of = |t: bool| if t { bls::cublasOperation_t::CUBLAS_OP_T } else { bls::cublasOperation_t::CUBLAS_OP_N };
            unsafe {
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, 0f32);
                bls::cublasSgemmStridedBatched(
                    *h,
                    op_of(transa),
                    op_of(transb),
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha,
                    ap as *const f32,
                    lda as i32,
                    stride_a as i64,
                    bp as *const f32,
                    ldb as i32,
                    stride_b as i64,
                    &beta,
                    op as *mut f32,
                    ldc as i32,
                    stride_c as i64,
                    batches as i32,
                )
                .result()
                .map_err(blas_err)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// F.grid_sample(input, grid, mode="bilinear", padding_mode="border", align_corners=True) in f32,
/// torch's CUDA kernel: `input` (N, C, H, W); `grid` (N, 2, Ho, Wo), the x and y planes that
/// RIFE's warp permutes into grid_sample's (N, Ho, Wo, 2). The output is (N, C, Ho, Wo).
pub fn torch_grid_sample_border_f32(input: &Tensor, grid: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = input.dims4()?;
    let (gn, two, oh, ow) = grid.dims4()?;
    if gn != n || two != 2 || input.dtype() != DType::F32 || grid.dtype() != DType::F32 {
        candle_core::bail!(
            "torch_grid_sample_border_f32: input {:?} {:?}, grid {:?} {:?}",
            input.dims(),
            input.dtype(),
            grid.dims(),
            grid.dtype()
        );
    }
    let a = cuda_dev(input)?;
    let (ic, gc) = (input.contiguous()?, grid.contiguous()?);
    let (ig, il) = ic.storage_and_layout();
    let iv = f32_view(&ig, il, "input")?;
    let (gg, gl) = gc.storage_and_layout();
    let gv = f32_view(&gg, gl, "grid")?;
    let y = unsafe { a.alloc::<f32>(n * c * oh * ow)? };
    let func = a.get_or_load_custom_func("torch_grid_sample_border_f32", "rustyui_ops", PTX)?;
    let count = (n * oh * ow) as i32;
    let dims = [c as i32, h as i32, w as i32, oh as i32, ow as i32];
    let mut bl = func.builder();
    bl.arg(&iv).arg(&gv).arg(&y).arg(&count);
    for d in &dims {
        bl.arg(d);
    }
    let cfg = LaunchConfig { grid_dim: ((count as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { bl.launch(cfg) }.w()?;
    drop((iv, gv));
    drop((ig, gg));
    Ok(f32_tensor(y, &a, &[n, c, oh, ow]))
}

/// torch.linspace(start, end, steps) as an f32 tensor on the card, torch's CUDA kernel
pub fn torch_linspace_f32(start: f32, end: f32, steps: usize, dev: &candle_core::Device) -> Result<Tensor> {
    if steps == 1 {
        return Tensor::new(&[start], dev);
    }
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_linspace_f32: not on the GPU"),
    };
    let step = (end - start) / (steps - 1) as f32;
    let y = unsafe { a.alloc::<f32>(steps)? };
    let func = a.get_or_load_custom_func("torch_linspace_f32", "rustyui_ops", PTX)?;
    let n = steps as i64;
    let mut bl = func.builder();
    bl.arg(&y).arg(&n).arg(&start).arg(&end).arg(&step);
    unsafe { bl.launch(LaunchConfig::for_num_elems(steps as u32)) }.w()?;
    Ok(f32_tensor(y, &a, &[steps]))
}

/// One of torch's elementwise f32 kernels from ops.cu, `extra` its scalar argument if it takes one
fn torch_unary_f32(x: &Tensor, name: &str, extra: Option<f32>) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        candle_core::bail!("{name}: {:?}", x.dtype());
    }
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let n = x.elem_count();
    let y = unsafe { a.alloc::<f32>(n)? };
    let func = a.get_or_load_custom_func(name, "rustyui_ops", PTX)?;
    let ni = n as i64;
    let mut bl = func.builder();
    bl.arg(&xv).arg(&y).arg(&ni);
    if let Some(e) = &extra {
        bl.arg(e);
    }
    unsafe { bl.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, x.dims()))
}

/// F.leaky_relu(x, negval) in f32, torch's kernel: x where positive, else x * negval
pub fn torch_leaky_relu_f32(x: &Tensor, negval: f32) -> Result<Tensor> {
    torch_unary_f32(x, "torch_leaky_relu_f32", Some(negval))
}

/// torch.sigmoid in f32, torch's kernel: 1 / (1 + exp(-x)) with CUDA's expf
pub fn torch_sigmoid_f32(x: &Tensor) -> Result<Tensor> {
    torch_unary_f32(x, "torch_sigmoid_f32", None)
}

/// er_sde's noise scaler on the card, x * ((x ** 0.3).exp() + 10.0) with CUDA's powf and expf
/// (ops.cu), for the handful of values the sampler scales each step
pub fn torch_er_noise_scaler_f32(x: &[f32], dev: &candle_core::Device) -> Result<Vec<f32>> {
    let a = match dev {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("torch_er_noise_scaler_f32: not on the GPU"),
    };
    let xt = Tensor::new(x, dev)?;
    let (xg, xl) = xt.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let out = unsafe { a.alloc::<f32>(x.len())? };
    let func = a.get_or_load_custom_func("torch_er_noise_scaler_f32", "rustyui_ops", PTX)?;
    let n = x.len() as i32;
    let mut b = func.builder();
    b.arg(&xv).arg(&out).arg(&n);
    unsafe { b.launch(LaunchConfig::for_num_elems(x.len() as u32)) }.w()?;
    drop(xv);
    drop(xg);
    f32_tensor(out, &a, &[x.len()]).to_vec1::<f32>()
}

// ── SAM3 (ComfyUI-RMBG's SAM3Segment) ───────────────────────────────────────────

/// nn.GELU() (erf) on bf16, torch's kernel: computed in f32, rounded to bf16
pub fn torch_gelu_erf_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        candle_core::bail!("torch_gelu_erf_bf16: {:?}", x.dtype());
    }
    ew("torch_gelu_erf_bf16", &[x], &[x.elem_count() as i64])
}

/// F.interpolate(mode="bilinear", antialias=True, align_corners=False) to (oh, ow) on f32 planes
/// (P, H, W): torch's CUDA kernel, its scales from the sizes
pub fn torch_upsample_aa_bilinear_f32(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (p, ih, iw) = x.dims3()?;
    if x.dtype() != DType::F32 {
        candle_core::bail!("torch_upsample_aa_bilinear_f32: {:?}", x.dtype());
    }
    let (hs, ws) = (ih as f32 / oh as f32, iw as f32 / ow as f32);
    for s in [hs, ws] {
        if 2 * (s.max(1.0).ceil() as usize) + 1 > 64 {
            candle_core::bail!("torch_upsample_aa_bilinear_f32: a scale of {s} spans more than the kernel's 64 weights");
        }
    }
    let a = cuda_dev(x)?;
    let xc = x.contiguous()?;
    let (xg, xl) = xc.storage_and_layout();
    let xv = f32_view(&xg, xl, "x")?;
    let y = unsafe { a.alloc::<f32>(p * oh * ow)? };
    let func = a.get_or_load_custom_func("torch_upsample_aa_bilinear_f32", "rustyui_ops", PTX)?;
    let (pi, ihi, iwi, ohi, owi) = (p as i32, ih as i32, iw as i32, oh as i32, ow as i32);
    let mut b = func.builder();
    b.arg(&xv).arg(&y).arg(&pi).arg(&ihi).arg(&iwi).arg(&ohi).arg(&owi).arg(&hs).arg(&ws);
    let cfg = LaunchConfig {
        grid_dim: (ow.div_ceil(32) as u32, oh.div_ceil(8) as u32, p.min(65535) as u32),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.w()?;
    drop(xv);
    drop(xg);
    Ok(f32_tensor(y, &a, &[p, oh, ow]))
}

/// SAM3's ViT attention inputs from its qkv rows (B * L, 3 * C) bf16: q and k rotated by freqs
/// (L, D / 2, 2) f32 (apply_rotary_enc), v copied; each (B * L, C) bf16, that is (B, L, heads, D)
pub fn sam3_rope_qkv(qkv: &Tensor, freqs: &Tensor, l: usize, c: usize, half_d: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let rows = qkv.dim(0)?;
    if qkv.dtype() != DType::BF16 || qkv.dim(1)? != 3 * c || freqs.dims() != [l, half_d, 2] || rows % l != 0 {
        candle_core::bail!("sam3_rope_qkv: qkv {:?} {:?}, freqs {:?}, L {l}, C {c}", qkv.dims(), qkv.dtype(), freqs.dims());
    }
    let dev = cuda_dev(qkv)?;
    let (qc, fc) = (qkv.contiguous()?, freqs.contiguous()?);
    let (qg, ql) = qc.storage_and_layout();
    let (fg, fl) = fc.storage_and_layout();
    let qv = gpu_view(&qg, ql, "qkv")?;
    let fv = f32_view(&fg, fl, "freqs")?;
    let n = rows * c;
    let (q, k, v) = unsafe { (dev.alloc::<half::bf16>(n)?, dev.alloc::<half::bf16>(n)?, dev.alloc::<half::bf16>(n)?) };
    let func = dev.get_or_load_custom_func("sam3_rope_qkv_bf16", "rustyui_ops", PTX)?;
    let (r, li, ci, hd) = (rows as i64, l as i32, c as i32, half_d as i32);
    let mut b = func.builder();
    b.arg(&qv).arg(&fv).arg(&q).arg(&k).arg(&v).arg(&r).arg(&li).arg(&ci).arg(&hd);
    unsafe { b.launch(LaunchConfig::for_num_elems((n / 2) as u32)) }.w()?;
    drop((qv, fv));
    drop((qg, fg));
    Ok((new_tensor(q, &dev, &[rows, c]), new_tensor(k, &dev, &[rows, c]), new_tensor(v, &dev, &[rows, c])))
}

/// SAM3's PositionEmbeddingSine for an (h, w) map: (1, 256, h, w) bf16, torch's ops on the card
pub fn sam3_pos_sine(h: usize, w: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let candle_core::Device::Cuda(a) = dev else { candle_core::bail!("sam3_pos_sine: not on the card") };
    let n = 256 * h * w;
    let y = unsafe { a.alloc::<half::bf16>(n)? };
    let func = a.get_or_load_custom_func("sam3_pos_sine_bf16", "rustyui_ops", PTX)?;
    let (hi, wi) = (h as i32, w as i32);
    let mut b = func.builder();
    b.arg(&y).arg(&hi).arg(&wi);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    Ok(new_tensor(y, a, &[1, 256, h, w]))
}

fn f32_out(name: &str, ins: &[&Tensor], n: usize, args: &[i64], shape: &[usize]) -> Result<Tensor> {
    let dev = cuda_dev(ins[0])?;
    let ins: Vec<Tensor> = ins.iter().map(|t| t.contiguous()).collect::<Result<_>>()?;
    let guards: Vec<_> = ins.iter().map(|t| t.storage_and_layout()).collect();
    let views: Vec<CudaView<'_, f32>> = guards.iter().map(|(st, l)| f32_view(st, l, name)).collect::<Result<_>>()?;
    let y = unsafe { dev.alloc::<f32>(n)? };
    let func = dev.get_or_load_custom_func(name, "rustyui_ops", PTX)?;
    let mut b = func.builder();
    for v in &views {
        b.arg(v);
    }
    b.arg(&y);
    let ints: Vec<i32> = args.iter().map(|&a| a as i32).collect();
    for a in &ints {
        b.arg(a);
    }
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop(views);
    drop(guards);
    Ok(f32_tensor(y, &dev, shape))
}

/// gen_sineembed_for_position for boxes (Q, 4) f32: (Q, 512) f32
pub fn sam3_box_sine(boxes: &Tensor) -> Result<Tensor> {
    let q = boxes.dim(0)?;
    f32_out("sam3_box_sine_f32", &[boxes], q * 512, &[q as i64], &[q, 512])
}

/// _get_rpb_matrix's log-scaled deltas for boxes (Q, 4) on an (h, w) map: dx (Q, w, 2) and
/// dy (Q, h, 2) f32
pub fn sam3_rpb_deltas(boxes: &Tensor, h: usize, w: usize) -> Result<(Tensor, Tensor)> {
    let q = boxes.dim(0)?;
    let dev = cuda_dev(boxes)?;
    let bc = boxes.contiguous()?;
    let (bg, bl) = bc.storage_and_layout();
    let bv = f32_view(&bg, bl, "boxes")?;
    let (nx, ny) = (q * w * 2, q * h * 2);
    let (dx, dy) = unsafe { (dev.alloc::<f32>(nx)?, dev.alloc::<f32>(ny)?) };
    let func = dev.get_or_load_custom_func("sam3_rpb_deltas_f32", "rustyui_ops", PTX)?;
    let (qi, hi, wi) = (q as i32, h as i32, w as i32);
    let mut b = func.builder();
    b.arg(&bv).arg(&dx).arg(&dy).arg(&qi).arg(&hi).arg(&wi);
    unsafe { b.launch(LaunchConfig::for_num_elems((nx + ny) as u32)) }.w()?;
    drop(bv);
    drop(bg);
    Ok((f32_tensor(dx, &dev, &[q, w, 2]), f32_tensor(dy, &dev, &[q, h, 2])))
}

/// _get_rpb_matrix's bias for the image cross-attention with the presence token's zero row first:
/// dy (Q, h, heads) + dx (Q, w, heads) bf16 as (heads, 1 + Q, h * w) bf16
pub fn sam3_rpb_bias(dy: &Tensor, dx: &Tensor) -> Result<Tensor> {
    let (q, h, heads) = dy.dims3()?;
    let w = dx.dim(1)?;
    let dev = cuda_dev(dy)?;
    let (yc, xc) = (dy.contiguous()?, dx.contiguous()?);
    let (yg, yl) = yc.storage_and_layout();
    let (xg, xl) = xc.storage_and_layout();
    let (yv, xv) = (gpu_view(&yg, yl, "dy")?, gpu_view(&xg, xl, "dx")?);
    let n = heads * (q + 1) * h * w;
    let out = unsafe { dev.alloc::<half::bf16>(n)? };
    let func = dev.get_or_load_custom_func("sam3_rpb_bias_bf16", "rustyui_ops", PTX)?;
    let (qi, hi, wi, he) = (q as i32, h as i32, w as i32, heads as i32);
    let mut b = func.builder();
    b.arg(&yv).arg(&xv).arg(&out).arg(&qi).arg(&hi).arg(&wi).arg(&he);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    drop((yv, xv));
    drop((yg, xg));
    Ok(new_tensor(out, &dev, &[heads, q + 1, h * w]))
}

/// SAM3's box refinement: sigmoid(delta + inverse_sigmoid(refs)) for refs f32 and delta bf16 of
/// one shape, or the sigmoid of refs alone; f32
pub fn sam3_refine(refs: &Tensor, delta: Option<&Tensor>) -> Result<Tensor> {
    let dev = cuda_dev(refs)?;
    let n = refs.elem_count();
    let rc = refs.contiguous()?;
    let (rg, rl) = rc.storage_and_layout();
    let rv = f32_view(&rg, rl, "refs")?;
    let y = unsafe { dev.alloc::<f32>(n)? };
    let func = dev.get_or_load_custom_func("sam3_refine_f32", "rustyui_ops", PTX)?;
    let ni = n as i64;
    match delta {
        Some(d) => {
            if d.elem_count() != n || d.dtype() != DType::BF16 {
                candle_core::bail!("sam3_refine: delta {:?} {:?} for {:?}", d.dims(), d.dtype(), refs.dims());
            }
            let dc = d.contiguous()?;
            let (dg, dl) = dc.storage_and_layout();
            let dv = gpu_view(&dg, dl, "delta")?;
            let mut b = func.builder();
            b.arg(&rv).arg(&dv).arg(&y).arg(&ni);
            unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        }
        None => {
            let null: u64 = 0;
            let mut b = func.builder();
            b.arg(&rv).arg(&null).arg(&y).arg(&ni);
            unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        }
    }
    drop(rv);
    drop(rg);
    Ok(f32_tensor(y, &dev, refs.dims()))
}

/// torch's bgemm<at::BFloat16> as bmm_out_cuda / baddbmm_out_cuda_impl call it:
/// cublasGemmStridedBatchedEx on bf16 with f32 compute and the default tensor-op algorithm, the
/// operands as raw bf16 storage with the leading dimensions and batch strides the caller worked
/// out the way torch does; `c` (c_len values) is baddbmm's result already filled with its input,
/// added with beta 1, or none for bmm. Returns C's storage.
#[allow(clippy::too_many_arguments)]
pub fn torch_bgemm_bf16(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &Tensor,
    lda: usize,
    stride_a: usize,
    b: &Tensor,
    ldb: usize,
    stride_b: usize,
    c: Option<&Tensor>,
    ldc: usize,
    stride_c: usize,
    batches: usize,
    c_len: usize,
) -> Result<Tensor> {
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        candle_core::bail!("torch_bgemm_bf16: {:?} {:?}", a.dtype(), b.dtype());
    }
    let d = cuda_dev(a)?;
    let stream = d.cuda_stream();
    let out = match c {
        Some(c) => {
            if c.elem_count() != c_len || c.dtype() != DType::BF16 {
                candle_core::bail!("torch_bgemm_bf16: C {:?} {:?} for {c_len} values", c.dims(), c.dtype());
            }
            c.flatten_all()?.contiguous()?.copy()?
        }
        None => Tensor::zeros(c_len, DType::BF16, a.device())?,
    };
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    {
        let (ast, al) = a.storage_and_layout();
        let (bst, bl) = b.storage_and_layout();
        let (ost, ol) = out.storage_and_layout();
        let (ap, _g1) = device_addr_any(&ast, al, "a", &stream)?;
        let (bp, _g2) = device_addr_any(&bst, bl, "b", &stream)?;
        let (op, _g3) = device_addr_any(&ost, ol, "out", &stream)?;
        BLAS.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let h = blr::create_handle().map_err(blas_err)?;
                let work = unsafe { d.alloc::<u8>(BLAS_WORKSPACE)? };
                *cell = Some(BlasHandle(h, work));
            }
            let BlasHandle(h, work) = cell.as_ref().unwrap();
            let (kp, _g4) = work.device_ptr(&stream);
            let op_of = |t: bool| if t { bls::cublasOperation_t::CUBLAS_OP_T } else { bls::cublasOperation_t::CUBLAS_OP_N };
            let ty = bls::cudaDataType_t::CUDA_R_16BF;
            unsafe {
                blr::set_stream(*h, stream.cu_stream() as _).map_err(blas_err)?;
                bls::cublasSetWorkspace_v2(*h, kp as *mut _, BLAS_WORKSPACE).result().map_err(blas_err)?;
                bls::cublasSetMathMode(*h, bls::cublasMath_t::CUBLAS_DEFAULT_MATH).result().map_err(blas_err)?;
                let (alpha, beta) = (1f32, if c.is_some() { 1f32 } else { 0f32 });
                // bmm of one batch is torch's gemm: cublasGemmEx, not the strided batched call
                if batches == 1 {
                    bls::cublasGemmEx(
                        *h,
                        op_of(transa),
                        op_of(transb),
                        m as i32,
                        n as i32,
                        k as i32,
                        &alpha as *const f32 as *const _,
                        ap as *const _,
                        ty,
                        lda as i32,
                        bp as *const _,
                        ty,
                        ldb as i32,
                        &beta as *const f32 as *const _,
                        op as *mut _,
                        ty,
                        ldc as i32,
                        bls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        bls::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                    .result()
                    .map_err(blas_err)?;
                    return Ok(());
                }
                bls::cublasGemmStridedBatchedEx(
                    *h,
                    op_of(transa),
                    op_of(transb),
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha as *const f32 as *const _,
                    ap as *const _,
                    ty,
                    lda as i32,
                    stride_a as i64,
                    bp as *const _,
                    ty,
                    ldb as i32,
                    stride_b as i64,
                    &beta as *const f32 as *const _,
                    op as *mut _,
                    ty,
                    ldc as i32,
                    stride_c as i64,
                    batches as i32,
                    bls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    bls::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .map_err(blas_err)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UploadSrc::copy_to gives the same bytes from memory and from the file, in one piece and in
    /// the pieces its threads copy
    #[test]
    fn upload_sources_agree() {
        let path = std::env::temp_dir().join(format!("rustyui-upload-{}", std::process::id()));
        let bytes: Vec<u8> = (0..(5usize << 20) + 3).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let src = UploadSrc::File { file: &file, offset: 0, len: bytes.len() };
        for (off, n) in [(0, 10), (12345, 1 << 20), (7, (4 << 20) + 11), (bytes.len() - 5, 5)] {
            let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
            src.copy_to(off, &mut a).unwrap();
            UploadSrc::Bytes(&bytes).copy_to(off, &mut b).unwrap();
            assert_eq!(a, &bytes[off..off + n]);
            assert_eq!(b, &bytes[off..off + n]);
        }
        let past = UploadSrc::File { file: &file, offset: 10, len: bytes.len() };
        assert!(past.copy_to(bytes.len() - 20, &mut [0u8; 20]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
