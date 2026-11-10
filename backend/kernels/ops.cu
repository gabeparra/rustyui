// rustyui's own kernels: one pass each for what candle does in many small ops.
// The arithmetic follows PyTorch's CUDA kernels (f32 accumulation, the same fused forms),
// so results round the way ComfyUI's do.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <climits>

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Sum over the block; `sh` holds 33 floats. Safe to call repeatedly.
__device__ float block_sum(float v, float *sh) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < nw ? sh[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) sh[32] = t;
    }
    __syncthreads();
    const float total = sh[32];
    __syncthreads();
    return total;
}

// PyTorch's silu: x / (1 + exp(-x)) in f32, applied to the already-rounded norm output.
template <typename T> __device__ __forceinline__ float silu_rounded(float v) {
    const float r = to_f<T>(from_f<T>(v));
    return r / (1.0f + expf(-r));
}

// PyTorch's Welford accumulator and its operations (SharedReduceOps.h: WelfordData, WelfordOps
// with correction 0), in f32 with an int64 count, written as torch writes them so nvcc contracts
// the same multiply-adds.
struct Welford {
    float mean, m2;
    long long n;
    float nf;
};

__device__ __forceinline__ Welford welford_reduce(Welford acc, float data) {
    long long new_n = acc.n + 1;
    float new_nf = static_cast<float>(new_n);
    float delta = data - acc.mean;
    float new_mean = acc.mean + delta / new_nf;
    float new_delta = data - new_mean;
    return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
}

__device__ __forceinline__ Welford welford_combine(Welford a, Welford b) {
    if (a.nf == 0) return b;
    if (b.nf == 0) return a;
    float delta = b.mean - a.mean;
    float new_count = a.nf + b.nf;
    float nb_over_n = b.nf / new_count;
    return {a.mean + delta * nb_over_n, a.m2 + b.m2 + delta * delta * a.nf * nb_over_n, -1, new_count};
}

__device__ __forceinline__ Welford welford_shfl_down(Welford v, int offset) {
    return {__shfl_down_sync(0xffffffff, v.mean, offset), __shfl_down_sync(0xffffffff, v.m2, offset),
            __shfl_down_sync(0xffffffff, v.n, offset), __shfl_down_sync(0xffffffff, v.nf, offset)};
}

// cuda_utils::WarpReduce
__device__ __forceinline__ Welford welford_warp(Welford v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = welford_combine(v, welford_shfl_down(v, offset));
    return v;
}

// GroupNorm as PyTorch's CUDA kernels compute it (group_norm_kernel.cu): one block per (n, group)
// of 32 threads when the group holds fewer than 512 values, else 512 (RowwiseMomentsCUDAKernel's
// launch), Welford per thread, the warp then block reduction of cuda_utils::BlockReduce. eps
// arrives in T, and the mean and rstd are stored in T (native_group_norm's outputs have the
// input's type), so both are rounded before ComputeFusedParamsCUDAKernel makes a = rstd * gamma[c]
// and b = -a * mean + beta[c] in f32 (in torch's build the product is rounded before the sum, no
// fused multiply-add: checked on SDXL's fp16 GroupNorms, where an FMA there moves a few hundred
// values); y = a * x + b, fused, rounded. silu != 0 also applies SiLU to the
// rounded result, as the separate PyTorch kernel would.
template <typename T>
__device__ void group_norm(const T *x, T *y, const T *gamma, const T *beta, int C, long HW, int G, float eps, int silu) {
    __shared__ Welford shared[32];
    __shared__ float stats[2];
    const int row = blockIdx.x, n = row / G, g = row % G, cpg = C / G;
    const long base = ((long)n * C + (long)g * cpg) * HW;
    const long len = (long)cpg * HW;
    const int tid = threadIdx.x, lid = tid % 32, wid = tid / 32;
    Welford val = {0.f, 0.f, 0, 0.f};
    for (long j = tid; j < len; j += blockDim.x) val = welford_reduce(val, to_f<T>(x[base + j]));
    if (blockDim.x <= 32) {
        val = welford_warp(val);
    } else {
        val = welford_warp(val);
        __syncthreads();
        if (lid == 0) shared[wid] = val;
        __syncthreads();
        val = (tid < (int)(blockDim.x / 32)) ? shared[lid] : Welford{0.f, 0.f, 0, 0.f};
        if (wid == 0) val = welford_warp(val);
    }
    if (tid == 0) {
        // project: var = m2 / n
        const float divisor = val.nf > 0 ? val.nf : 0;
        const float var = val.m2 / divisor;
        stats[0] = to_f<T>(from_f<T>(val.mean));
        stats[1] = to_f<T>(from_f<T>(rsqrtf(var + to_f<T>(from_f<T>(eps)))));
    }
    __syncthreads();
    const float mean = stats[0], rstd = stats[1];
    for (long i = tid; i < len; i += blockDim.x) {
        const int c = g * cpg + (int)(i / HW);
        const float a = rstd * to_f<T>(gamma[c]);
        const float b = __fadd_rn(__fmul_rn(-a, mean), to_f<T>(beta[c]));
        float v = fmaf(a, to_f<T>(x[base + i]), b);
        if (silu) v = silu_rounded<T>(v);
        y[base + i] = from_f<T>(v);
    }
}

// GEGLU: h = [value | gate] on the last dim -> value * gelu(gate), exact (erf) GELU,
// each step rounded to T as PyTorch's separate gelu and mul kernels round.
template <typename T>
__device__ void geglu(const T *h, T *y, long rows, int half) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * half) return;
    const long r = i / half, c = i % half;
    const float a = to_f<T>(h[r * 2 * half + c]);
    const float g = to_f<T>(h[r * 2 * half + half + c]);
    const float ge = to_f<T>(from_f<T>(g * 0.5f * (1.0f + erff(g * (float)M_SQRT1_2))));
    y[i] = from_f<T>(a * ge);
}

#define INSTANTIATE(T, SUFFIX)                                                                          \
    extern "C" __global__ void group_norm_##SUFFIX(const T *x, T *y, const T *gamma, const T *beta,      \
                                                   int C, long HW, int G, float eps, int silu) {         \
        group_norm<T>(x, y, gamma, beta, C, HW, G, eps, silu);                                           \
    }                                                                                                    \
    extern "C" __global__ void geglu_##SUFFIX(const T *h, T *y, long rows, int half) {                  \
        geglu<T>(h, y, rows, half);                                                                      \
    }

INSTANTIATE(__half, f16)
INSTANTIATE(__nv_bfloat16, bf16)
INSTANTIATE(float, f32)

// ── NVFP4 weights (comfy_kitchen's TensorCoreNVFP4Layout) ─────────────────────
// Packed E2M1 pairs (high nibble first), one E4M3 scale per 16 values stored in cuBLAS's
// 128x4 swizzled tile layout, one f32 global scale. Dequantised in f32 and left there: ComfyUI's
// text-encoder forward asks for the weight in its compute dtype (f32), and those values are not
// bf16-rounded (checked against the weight F.linear receives there).
__device__ __forceinline__ float fp8_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = ldexpf((float)m * 0.125f, -6);
    else if (e == 15 && m == 7) v = __int_as_float(0x7fc00000);
    else v = ldexpf(1.0f + (float)m * 0.125f, e - 7);
    return s ? -v : v;
}

extern "C" __global__ void dequant_nvfp4_f32(const uint8_t *q, const uint8_t *scales, const float *gscale,
                                             float *out, int rows, int cols, int n_cb) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * cols) return;
    const int r = (int)(idx / cols), c = (int)(idx % cols);
    const uint8_t byte = q[(long)r * (cols / 2) + c / 2];
    const int nib = (c & 1) ? (byte & 15) : (byte >> 4);
    const float lut[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const int bc = c / 16, rb = r / 128, i = r % 128, cb = bc / 4, j = bc % 4;
    const long s = ((long)rb * n_cb + cb) * 512 + (i % 32) * 16 + (i / 32) * 4 + j;
    out[idx] = (lut[nib] * fp8_e4m3(scales[s])) * gscale[0];
}

// fp8 e4m3 weights with one f32 scale for the whole tensor (comfy_kitchen's TensorCoreFP8Layout):
// f32(q) * scale, unrounded, which is the weight ComfyUI's f32 forward multiplies by.
extern "C" __global__ void dequant_fp8_f32(const uint8_t *q, const float *scale, float *out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp8_e4m3(q[i]) * scale[0];
}

// The input of one of LTX-2.3's fp8 linears, as ComfyUI quantises it: the checkpoint stores no
// input_scale, so ComfyUI's fp8 layout uses a scale of 1.0 (not the tensor's amax), clamps to
// e4m3's +-448 and rounds to nearest even (torch's .to(float8_e4m3fn)); NaN stays NaN.
extern "C" __global__ void quant_fp8_e4m3_bf16(const __nv_bfloat16 *x, uint8_t *q, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    if (isnan(v)) {
        q[i] = signbit(v) ? 0xff : 0x7f;
        return;
    }
    q[i] = (uint8_t)__nv_cvt_float_to_fp8(fminf(fmaxf(v, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
}

// ── LTX-2.3: the eager PyTorch steps of ComfyUI's av_model.py, rounding as they round ──
// Every tensor here is bf16. PyTorch computes each elementwise op in f32 and rounds its result to
// bf16, so a sequence of ops is reproduced by rounding after each one, never across. Operands
// marked "bcast" repeat along rows: index (i / d) * stride + i % d, stride 0 for one row.
#define BF __nv_bfloat16
__device__ __forceinline__ float bf(BF v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// b's element for a's element i, b repeated over a's rows: row r of a reads row r / g of b, so
// g = rows(a) for one row, 1 for a full-size b, and the tokens per frame for one row per frame
// (ComfyUI's CompressedTimestep, tokens frame-major)
__device__ __forceinline__ long bidx(long i, long d, long g) { return (i / d / g) * d + i % d; }

// a + b (b bcast): get_ada_values' table + timestep, the residual add_
extern "C" __global__ void ltx_add(const BF *a, const BF *b, BF *y, long n, long d, long sb) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16_rn(bf(a[i]) + bf(b[bidx(i, d, sb)]));
}

// a * b (b bcast): attn(...) * q_gate
extern "C" __global__ void ltx_mul(const BF *a, const BF *b, BF *y, long n, long d, long sb) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16_rn(bf(a[i]) * bf(b[bidx(i, d, sb)]));
}

// addcmul: a + 1 * (b * c), c bcast (the gated residuals). torch: a + alpha * (b * c) in f32.
extern "C" __global__ void ltx_addcmul(const BF *a, const BF *b, const BF *c, BF *y, long n, long d, long sc) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = 1.0f;
    y[i] = __float2bfloat16_rn(bf(a[i]) + alpha * (bf(b[i]) * bf(c[bidx(i, d, sc)])));
}

// x * (1 + scale) + shift, scale and shift bcast: rms_adaln's tail and the context modulation,
// three kernels in PyTorch, so three roundings.
extern "C" __global__ void ltx_modulate(const BF *x, const BF *scale, const BF *shift, BF *y, long n, long d, long s) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long j = bidx(i, d, s);
    const float one_plus = rbf(1.0f + bf(scale[j]));
    const float scaled = rbf(bf(x[i]) * one_plus);
    y[i] = __float2bfloat16_rn(scaled + bf(shift[j]));
}

// F.gelu(x, approximate="tanh") (ActivationGeluKernel.cu)
extern "C" __global__ void ltx_gelu_tanh(const BF *x, BF *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * float(0.5);
    constexpr float kKappa = 0.044715;
    const float v = bf(x[i]);
    auto x_cube = v * v * v;
    auto inner = kBeta * (v + kKappa * x_cube);
    y[i] = __float2bfloat16_rn(float(0.5) * v * (float(1) + tanhf(inner)));
}

// Gated attention: out (tokens, heads, dim) times 2 * sigmoid(logits (tokens, heads)), each of
// sigmoid, 2 * and the product a separate PyTorch kernel.
extern "C" __global__ void ltx_head_gate(const BF *o, const BF *logits, BF *y, long n, int heads, int dim) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long tok = i / ((long)heads * dim);
    const int h = (int)((i / dim) % heads);
    const float one = 1.0f;
    const float sig = rbf(one / (one + expf(-bf(logits[tok * heads + h]))));
    const float gate = rbf(2.0f * sig);
    y[i] = __float2bfloat16_rn(bf(o[i]) * gate);
}

// F.rms_norm as torch 2.11 runs it on CUDA (layer_norm_kernel.cu, vectorized_layer_norm_kernel
// with rms_norm=true; 32x4 threads per row): each thread folds the squares of its 4-element
// vectors in order, the warp folds lanes with shfl_down 16..1, warps 2,3 fold into 0,1 and 1
// into 0; rstd = rsqrtf(sum / N + eps); y = gamma * (rstd * x), or rstd * x without gamma.
extern "C" __global__ void ltx_rms_norm(const BF *x, const BF *gamma, BF *y, int N, float eps, int has_gamma) {
    __shared__ float buf[8];
    const BF *row = x + (long)blockIdx.x * N;
    BF *out = y + (long)blockIdx.x * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int nvec = N / 4;
    float s2 = 0.f;
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const float v = bf(row[i * 4 + ii]);
            s2 = s2 + v * v;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float o = __shfl_down_sync(0xffffffff, s2, offset, 32);
        s2 = s2 + o;
    }
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) buf[2 * (threadIdx.y - offset) + 1] = s2;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) s2 = s2 + buf[2 * threadIdx.y + 1];
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) buf[1] = s2 / float(N);
    __syncthreads();
    const float rstd = rsqrtf(buf[1] + eps);
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const int c = i * 4 + ii;
            const float v = bf(row[c]);
            out[c] = __float2bfloat16_rn(has_gamma ? bf(gamma[c]) * (rstd * v) : rstd * v);
        }
    }
}
#undef BF

// ── LTX-2.3: the DiT's own embeddings and output norm ──────────────────────────
#define BF __nv_bfloat16

// nn.SiLU on bf16: x / (1 + exp(-x)) in f32
extern "C" __global__ void ltx_silu(const BF *x, BF *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    y[i] = __float2bfloat16_rn(v / (1.0f + expf(-v)));
}

// get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0) then .to(bf16):
// [cos(t * exp(e_k)), sin(t * exp(e_k))], the exponents e_k computed by the caller (IEEE f32,
// as torch's mul and div kernels compute them)
extern "C" __global__ void ltx_timestep_sinusoid(const float *t, const float *expo, BF *y, int n, int half) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n * half) return;
    const int r = (int)(i / half), k = (int)(i % half);
    const float arg = t[r] * expf(expo[k]);
    y[(long)r * 2 * half + k] = __float2bfloat16_rn(cosf(arg));
    y[(long)r * 2 * half + half + k] = __float2bfloat16_rn(sinf(arg));
}

// ── Anima (Cosmos-Predict2's MiniTrainDIT): VideoRopePosition3DEmb in f32, torch's ops ──────────
// 1.0 / (theta ** range): torch.pow of a Python-float base by an f32 tensor (powf of the base as a
// float), then the reciprocal (an IEEE division) times 1.0
extern "C" __global__ void cosmos_rope_freqs(const float *range, float theta, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __fdiv_rn(1.0f, powf(theta, range[i]));
}

// the rotation matrices (T * H * W, dt + dh + dw, 2, 2) [[cos, -sin], [sin, cos]] of the outer
// products seq[t] * ft, seq[h] * fh, seq[w] * fw (freqs = ft ++ fh ++ fw), each its own f32 op
extern "C" __global__ void cosmos_rope_table(const float *freqs, float *out, int T, int H, int W, int dt, int dh, int dw) {
    const int pairs = dt + dh + dw;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T * H * W * pairs) return;
    const int d = (int)(i % pairs);
    const long l = i / pairs;
    const int w = (int)(l % W), h = (int)((l / W) % H), t = (int)(l / ((long)W * H));
    const float pos = d < dt ? (float)t : d < dt + dh ? (float)h : (float)w;
    const float a = __fmul_rn(pos, freqs[d]);
    const float c = cosf(a), s = sinf(a);
    out[i * 4] = c;
    out[i * 4 + 1] = -s;
    out[i * 4 + 2] = s;
    out[i * 4 + 3] = c;
}

// Anima's LLMAdapter RotaryEmbedding: cos and sin of position * inv_freq[k % half] in f32 (the
// k = 1 matmul is one product), rounded to bf16 (cos.to(x.dtype)); (T, 2 * half) each
extern "C" __global__ void anima_rotary_tables(const float *inv_freq, BF *cos_out, BF *sin_out, int T, int half) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T * 2 * half) return;
    const int d = (int)(i % (2 * half));
    const int t = (int)(i / (2 * half));
    const float a = __fmul_rn((float)t, inv_freq[d % half]);
    cos_out[i] = __float2bfloat16_rn(cosf(a));
    sin_out[i] = __float2bfloat16_rn(sinf(a));
}

// apply_rotary_pos_emb in bf16: (x * cos) + (rotate_half(x) * sin), each op rounded; x (rows, D)
// with its row's position `(row / heads) % T` (x laid out (B, T, heads, D)), tables (T, D)
extern "C" __global__ void anima_rotary_apply(const BF *x, const BF *cos_t, const BF *sin_t, BF *y, long rows, int D, int heads, int T) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    const int d = (int)(i % D);
    const long row = i / D;
    const int t = (int)((row / heads) % T);
    const int half = D / 2;
    const float xr = d < half ? -bf(x[i + half]) : bf(x[i - half]);
    const float a = rbf(bf(x[i]) * bf(cos_t[(long)t * D + d]));
    const float b = rbf(xr * bf(sin_t[(long)t * D + d]));
    y[i] = __float2bfloat16_rn(a + b);
}

// freqs_cis_matrix in split mode: freqs (rows, f) f32 -> (rows, pad + f, 2, 2) bf16
// [[cos, -sin], [sin, cos]], the first `pad` entries the identity
extern "C" __global__ void ltx_rope_matrix(const float *freqs, BF *out, long rows, int f, int pad) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int c_all = pad + f;
    if (i >= rows * c_all) return;
    const long r = i / c_all;
    const int c = (int)(i % c_all);
    BF cs, sn;
    if (c < pad) {
        cs = __float2bfloat16_rn(1.0f);
        sn = __float2bfloat16_rn(0.0f);
    } else {
        const float v = freqs[r * f + (c - pad)];
        cs = __float2bfloat16_rn(cosf(v));
        sn = __float2bfloat16_rn(sinf(v));
    }
    out[i * 4 + 0] = cs;
    out[i * 4 + 1] = __hneg(sn);
    out[i * 4 + 2] = sn;
    out[i * 4 + 3] = cs;
}

// F.layer_norm without weight or bias, as torch 2.11's vectorized_layer_norm_kernel computes it
// (32x4 threads per row, Welford over 4-element vectors, the same combine order)
struct WelfordLN {
    float mean, sigma2, count;
};

__device__ __forceinline__ WelfordLN welford_online(const float val, const WelfordLN &curr) {
    float delta = val - curr.mean;
    float new_count = curr.count + 1.f;
    float new_mean = curr.mean + delta * (1.f / new_count);
    return {new_mean, curr.sigma2 + delta * (val - new_mean), new_count};
}

__device__ __forceinline__ WelfordLN welford_combine(const WelfordLN dataB, const WelfordLN dataA) {
    float delta = dataB.mean - dataA.mean;
    float count = dataA.count + dataB.count;
    float mean, sigma2;
    if (count > 0.f) {
        auto coef = 1.f / count;
        auto nA = dataA.count * coef;
        auto nB = dataB.count * coef;
        mean = nA * dataA.mean + nB * dataB.mean;
        sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
        mean = 0.f;
        sigma2 = 0.f;
    }
    return {mean, sigma2, count};
}

extern "C" __global__ void ltx_layer_norm(const BF *x, BF *y, int N, float eps) {
    __shared__ float buf[8];
    const BF *row = x + (long)blockIdx.x * N;
    BF *out = y + (long)blockIdx.x * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int nvec = N / 4;
    WelfordLN wd = {0.f, 0.f, 0.f};
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) wd = welford_online(__bfloat162float(row[i * 4 + ii]), wd);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordLN wdB{__shfl_down_sync(0xffffffff, wd.mean, offset, 32), __shfl_down_sync(0xffffffff, wd.sigma2, offset, 32),
                      __shfl_down_sync(0xffffffff, wd.count, offset, 32)};
        wd = welford_combine(wd, wdB);
    }
    float *meansigmabuf = buf;
    float *countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            meansigmabuf[2 * wrt_y] = wd.mean;
            meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
            countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordLN wdB{meansigmabuf[2 * threadIdx.y], meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y]};
            wd = welford_combine(wd, wdB);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    const float mean = meansigmabuf[0];
    const float rstd = rsqrtf(meansigmabuf[1] + eps);
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const int c = i * 4 + ii;
            out[c] = __float2bfloat16_rn(rstd * (__bfloat162float(row[c]) - mean));
        }
    }
}
#undef BF

// ── torch's CUDA randn ───────────────────────────────────────────────────────────

#include <curand_kernel.h>

// torch.randn on a CUDA generator (normal_kernel through distribution_nullary_kernel): each
// thread a Philox4_32_10 state at (seed, its index, offset), curand_normal4, four values spread a
// grid apart. The curand device code torch compiles, so the same numbers bit for bit (the CPU port
// in rng.rs rounds the Box-Muller sin, cos and log its own way, about 1e-7 off).
extern "C" __global__ void torch_randn(float *out, long n, unsigned long long seed, unsigned long long offset) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const long rounded = ((n - 1) / (stride * 4) + 1) * stride * 4;
    for (long linear = idx; linear < rounded; linear += stride * 4) {
        const float4 r = curand_normal4(&state);
        const float v[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const long li = linear + stride * ii;
            if (li < n) out[li] = v[ii];
        }
        __syncthreads();
    }
}

// ── LoRAs on the LTX DiT ──────────────────────────────────────────────────────

// torch.randint(0, 256, n, dtype=uint8) on a CUDA generator (random_from_to_kernel): curand4 per
// thread at (seed, index, offset), each 32-bit value % 256, laid out as torch lays them out
extern "C" __global__ void torch_randint_u8(uint8_t *out, long n, unsigned long long seed, unsigned long long offset) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const long rounded = ((n - 1) / (stride * 4) + 1) * stride * 4;
    for (long linear = idx; linear < rounded; linear += stride * 4) {
        const uint4 r = curand4(&state);
        const unsigned int v[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const long li = linear + stride * ii;
            if (li < n) out[li] = (uint8_t)(v[ii] % 256u);
        }
        __syncthreads();
    }
}

// The LTX VAE's per-channel statistics on a bf16 (N, C, ...) latent, as torch runs them in the VAE's
// dtype: un_normalize x * std + mean, normalize (x - mean) / std, each operation rounded to bf16
extern "C" __global__ void ltx_channel_affine(const __nv_bfloat16 *x, const __nv_bfloat16 *stdv, const __nv_bfloat16 *meanv,
                                              __nv_bfloat16 *y, long n, int C, long inner, int normalize) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int)((i / inner) % C);
    const float v = __bfloat162float(x[i]), s = __bfloat162float(stdv[c]), m = __bfloat162float(meanv[c]);
    if (normalize) {
        const float d = __bfloat162float(__float2bfloat16_rn(v - m));
        y[i] = __float2bfloat16_rn(d / s);
    } else {
        const float p = __bfloat162float(__float2bfloat16_rn(v * s));
        y[i] = __float2bfloat16_rn(p + m);
    }
}

// PyTorch's conv_depthwise2d_forward_kernel_generic for a (C, 1, 1, K) f32 filter on (N, C, 1, L):
// one thread per output, the taps inside the input from kWmin to kWmax, each multiply-added in
// order into an f32 accumulator starting at 0 (no bias)
extern "C" __global__ void torch_depthwise_conv1d_f32(const float *x, const float *w, float *y, int C, int Lin, int Lout, int K, int stride,
                                                      int pad, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int wo = (int)(i % Lout);
    const int c = (int)((i / Lout) % C);
    const long nb = i / ((long)Lout * C);
    int kWmin = 0, kWmax = K;
    const int w_in_min = -pad + wo * stride;
    if (w_in_min < 0) kWmin = -w_in_min;
    const int w_in_max = w_in_min + (K - 1) - Lin + 1;
    if (w_in_max >= 0) kWmax = K - w_in_max;
    float value = 0.0f;
    const long offset0 = (nb * C + c) * (long)Lin;
    for (int kW = kWmin; kW < kWmax; ++kW) {
        const int w_in = -pad + wo * stride + kW;
        value += (w[c * K + kW] * x[offset0 + w_in]);
    }
    y[i] = value;
}

// torch.mean over dim 1 of a contiguous (outer, R, inner) tensor as torch's gpu_reduce_kernel sums it
// when that dimension is not the fastest-moving one (Reduce.cuh, setReduceConfig's "vectorize along
// output" case), one thread here per output, so outputs need no shared memory or second launch.
// Torch's thread (warp y, CTA z) sums the indices y + z * bh, y + z * bh + step, ... (from 0 in
// steps of 1 when the input is not split across warps) into vt0 = 4 interleaved accumulators that it
// combines in order; a block's bh partials are combined by block_y_reduce's tree, which is a
// pairwise sum of the partials in bit-reversed order; with ctas > 1 the last block's thread y adds
// the partials of CTAs y, y + bh, ... to 0 one by one (global_reduce), then the same tree. The sum
// is multiplied by `factor`. `square` squares each value first, rounded to its own type (PixelNorm's
// x**2 before the mean).
struct TreeSum {
    float v[10];
    int lvl[10];
    int n;
    __device__ __forceinline__ void push(float x) {
        int l = 0;
        while (n > 0 && lvl[n - 1] == l) {
            x = v[n - 1] + x;
            n--;
            l++;
        }
        v[n] = x;
        lvl[n] = l;
        n++;
    }
};

__device__ __forceinline__ int bit_reversed(int i, int bits) { return bits == 0 ? 0 : (int)(__brev((unsigned)i) >> (32 - bits)); }

template <typename T>
__device__ __forceinline__ float reduce_load(const T *x, long at, int square) {
    const float v = to_f<T>(x[at]);
    return square == 1 ? to_f<T>(from_f<T>(v * v)) : v;
}

// one accumulator step: acc + v, or with square 2 NormTwoOps' acc + v * v in float, which nvcc
// contracts to one fused multiply-add in torch's build
template <typename T>
__device__ __forceinline__ float reduce_step(float acc, const T *x, long at, int square) {
    if (square == 2) {
        const float v = to_f<T>(x[at]);
        return __fmaf_rn(v, v, acc);
    }
    return acc + reduce_load(x, at, square);
}

template <typename T>
__device__ float torch_thread_reduce(const T *x, long base, long inner, int R, int start, int step, int square) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int idx = start;
    while (idx + 3 * step < R) {
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i] = reduce_step(acc[i], x, base + (long)(idx + i * step) * inner, square);
        idx += 4 * step;
    }
    for (int i = 0; i < 4; i++) {
        if (idx >= R) break;
        acc[i] = reduce_step(acc[i], x, base + (long)idx * inner, square);
        idx += step;
    }
    return ((acc[0] + acc[1]) + acc[2]) + acc[3];
}

// one CTA's sum: its bh threads' sums combined by block_y_reduce's tree
template <typename T>
__device__ float torch_block_sum(const T *x, long base, long inner, int R, int bh, int bits, int z, int step, int square) {
    TreeSum block;
    block.n = 0;
    for (int j = 0; j < bh; j++) block.push(torch_thread_reduce(x, base, inner, R, bit_reversed(j, bits) + z * bh, step, square));
    return block.v[0];
}

template <typename T>
__device__ float torch_reduce_sum(const T *x, long base, long inner, int R, int bh, int split, int ctas, int square) {
    if (!split) return torch_thread_reduce(x, base, inner, R, 0, 1, square);
    const int bits = __ffs(bh) - 1, step = bh * ctas;
    if (ctas == 1) return torch_block_sum(x, base, inner, R, bh, bits, 0, step, square);
    TreeSum last;
    last.n = 0;
    for (int i = 0; i < bh; i++) {
        const int y = bit_reversed(i, bits);
        float t = 0.f;
        for (int z = y; z < ctas; z += bh) t = t + torch_block_sum(x, base, inner, R, bh, bits, z, step, square);
        last.push(t);
    }
    return last.v[0];
}

// mode 0: the mean, rounded to T; 1: PixelNorm's divisor from the mean of squares,
// sqrt(mean + 1e-8) with the mean, the sum and the root each rounded to T (torch's ops in T);
// 2: the L2 norm (linalg.vector_norm's NormTwoOps: the squares summed in float, the root of the
// sum rounded to T; `factor` 1)
template <typename T>
__device__ void torch_mean_dim1(const T *x, T *y, long outer, int R, long inner, int bh, int split, int ctas, float factor, int mode) {
    const long o = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer * inner) return;
    const long b = o / inner, i = o - b * inner;
    const float s = torch_reduce_sum(x, b * R * inner + i, inner, R, bh, split, ctas, mode) * factor;
    if (mode == 2) {
        y[o] = from_f<T>(sqrtf(s));
    } else if (mode == 1) {
        const float m = to_f<T>(from_f<T>(s));
        const float e = to_f<T>(from_f<T>(m + 1e-8f));
        y[o] = from_f<T>(sqrtf(e));
    } else {
        y[o] = from_f<T>(s);
    }
}

extern "C" __global__ void torch_mean_dim1_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, long outer, int R, long inner, int bh, int split,
                                                int ctas, float factor, int mode) {
    torch_mean_dim1(x, y, outer, R, inner, bh, split, ctas, factor, mode);
}

extern "C" __global__ void torch_mean_dim1_f32(const float *x, float *y, long outer, int R, long inner, int bh, int split, int ctas,
                                               float factor, int mode) {
    torch_mean_dim1(x, y, outer, R, inner, bh, split, ctas, factor, mode);
}

extern "C" __global__ void torch_mean_dim1_f16(const __half *x, __half *y, long outer, int R, long inner, int bh, int split, int ctas,
                                               float factor, int mode) {
    torch_mean_dim1(x, y, outer, R, inner, bh, split, ctas, factor, mode);
}

// kornia's canny magnitude sqrt(gx * gx + gy * gy + eps), each op rounded: grad (B, 2, HW), mag (B, HW)
extern "C" __global__ void kornia_canny_magnitude(const float *grad, float *mag, long hw, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long b = i / hw, p = i - b * hw;
    const float gx = grad[(b * 2) * hw + p], gy = grad[(b * 2 + 1) * hw + p];
    mag[i] = sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy)), 1e-6f));
}

// kornia 0.8.3's canny after its gradients, as torch's separate f32 kernels compute each op (so no

// contraction): the magnitude sqrt(gx * gx + gy * gy + eps), the direction
// round(atan2(gy, gx) * (4 / pi)), the non-maximum suppression's two neighbours (the NMS channel at
// the direction mod 8 and at the direction + 4 mod 8, torch.remainder), then the thresholds on the
// suppressed magnitude: 0, 0.5 (weak) or 1 (strong). grad (B, 2, HW), nms (B, 8, HW), edges (B, HW)
extern "C" __global__ void kornia_canny_classify(const float *grad, const float *nms, float *edges, long hw, long n, float lo, float hi) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long b = i / hw, p = i - b * hw;
    const float gx = grad[(b * 2) * hw + p], gy = grad[(b * 2 + 1) * hw + p];
    const float m = sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy)), 1e-6f));
    const float a45 = rintf(__fmul_rn(atan2f(gy, gx), (float)(4.0 / 3.14159265358979323846)));
    auto rem8 = [](float v) {
        float r = fmodf(v, 8.0f);
        if (r != 0.0f && (r < 0.0f)) r += 8.0f;
        return (int)r;
    };
    const int pos = rem8(a45), neg = rem8(__fadd_rn(a45, 4.0f));
    const float mp = nms[(b * 8 + pos) * hw + p], mn = nms[(b * 8 + neg) * hw + p];
    const float s = __fmul_rn(m, fminf(mp, mn) > 0.0f ? 1.0f : 0.0f);
    edges[i] = (s > lo ? 0.5f : 0.0f) + (s > hi ? 0.5f : 0.0f);
}

// ── Depth Anything V2 (DINOv2 and its DPT head): torch 2.11's own f32 kernels ─────────────────

// softmax_warp_forward (PersistentSoftmax.cuh) for f32 rows of `cols` <= 2048 values, which
// host_softmax takes up to 2048: a warp a row (two for rows up to 128), each lane the values lane,
// lane + width, ... (-inf past the row), max and the sum of exp(x - max) in that order, each
// through the warp's xor butterfly; out = exp(x - max) / sum
template <int LOG2>
__device__ void torch_softmax_warp_f32(const float *src, float *dst, int rows, int cols) {
    constexpr int NP2 = 1 << LOG2;
    constexpr int WS = NP2 < 32 ? NP2 : 32;
    constexpr int IT = NP2 / WS;
    constexpr int WB = NP2 <= 128 ? 2 : 1;
    const int first = (blockDim.y * blockIdx.x + threadIdx.y) * WB;
    int local = rows - first;
    if (local > WB) local = WB;
    if (local <= 0) return;
    const int lane = threadIdx.x;
    src += (long)first * cols + lane;
    dst += (long)first * cols + lane;
    float e[WB][IT];
    for (int i = 0; i < WB; ++i) {
        const int cnt = (i >= local) ? 0 : cols;
        for (int it = 0; it < IT; ++it) {
            const int idx = lane + it * WS;
            e[i][it] = idx < cnt ? src[(long)i * cols + it * WS] : -INFINITY;
        }
    }
    float mx[WB];
    for (int i = 0; i < WB; ++i) {
        mx[i] = e[i][0];
        for (int it = 0; it < IT; ++it) mx[i] = mx[i] > e[i][it] ? mx[i] : e[i][it];
    }
    for (int off = WS / 2; off > 0; off /= 2)
        for (int i = 0; i < WB; ++i) {
            const float b = __shfl_xor_sync(0xffffffff, mx[i], off, WS);
            mx[i] = mx[i] < b ? b : mx[i];
        }
    float sum[WB];
    for (int i = 0; i < WB; ++i) {
        sum[i] = 0.f;
        for (int it = 0; it < IT; ++it) {
            e[i][it] = expf(e[i][it] - mx[i]);
            sum[i] += e[i][it];
        }
    }
    for (int off = WS / 2; off > 0; off /= 2)
        for (int i = 0; i < WB; ++i) sum[i] = sum[i] + __shfl_xor_sync(0xffffffff, sum[i], off, WS);
    for (int i = 0; i < WB; ++i) {
        if (i >= local) break;
        for (int it = 0; it < IT; ++it) {
            const int idx = lane + it * WS;
            if (idx >= cols) break;
            dst[(long)i * cols + it * WS] = sum[i] == 0.f ? NAN : e[i][it] / sum[i];
        }
    }
}

#define SOFTMAX_WARP(L) \
    extern "C" __global__ void torch_softmax_warp_f32_##L(const float *src, float *dst, int rows, int cols) { \
        torch_softmax_warp_f32<L>(src, dst, rows, cols); \
    }
SOFTMAX_WARP(0) SOFTMAX_WARP(1) SOFTMAX_WARP(2) SOFTMAX_WARP(3) SOFTMAX_WARP(4) SOFTMAX_WARP(5)
SOFTMAX_WARP(6) SOFTMAX_WARP(7) SOFTMAX_WARP(8) SOFTMAX_WARP(9) SOFTMAX_WARP(10) SOFTMAX_WARP(11)

// cuda_utils::BlockReduce (block_reduce.cuh) with SoftMax.cu's Max or Add: each warp's
// shuffle-down tree, lane 0's result to shared memory, warp 0's tree over those; the result
// broadcast to the block as blockReduceWarp does
__device__ float torch_block_reduce_f32(float val, bool is_max, float identity, float *shared) {
    const int tid = threadIdx.x, lid = tid % 32, wid = tid / 32;
    for (int off = 16; off > 0; off >>= 1) {
        const float b = __shfl_down_sync(0xffffffff, val, off);
        val = is_max ? (val < b ? b : val) : val + b;
    }
    __syncthreads();
    if (lid == 0) shared[wid] = val;
    __syncthreads();
    val = tid < (int)blockDim.x / 32 ? shared[lid] : identity;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            const float b = __shfl_down_sync(0xffffffff, val, off);
            val = is_max ? (val < b ? b : val) : val + b;
        }
    }
    if (tid == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// cunn_SoftMaxForwardReg (SoftMax.cu), which host_softmax runs for f32 rows of 2049 to 9216
// values: a block of 1024 threads a row, thread t holding values t, t + 1024, ... (`regs` of
// them); the thread's max, then its sum of exp(x - max) in that order, each through the block
// reduction; out = exp(x - max) / sum
extern "C" __global__ void torch_softmax_reg_f32(const float *src, float *dst, long cols, int regs) {
    __shared__ float shared[32];
    src += (long)blockIdx.x * cols;
    dst += (long)blockIdx.x * cols;
    float reg[9];
    float tmax = -3.40282347e+38f;
    for (int r = 0; r < regs; ++r) {
        const long off = threadIdx.x + (long)r * blockDim.x;
        if (off < cols) {
            reg[r] = src[off];
            tmax = fmaxf(tmax, reg[r]);
        }
    }
    const float mx = torch_block_reduce_f32(tmax, true, -3.40282347e+38f, shared);
    float tsum = 0.f;
    for (int r = 0; r < regs; ++r) {
        const long off = threadIdx.x + (long)r * blockDim.x;
        if (off < cols) tsum = tsum + expf(reg[r] - mx);
    }
    const float sum = torch_block_reduce_f32(tsum, false, 0.f, shared);
    for (int r = 0; r < regs; ++r) {
        const long off = threadIdx.x + (long)r * blockDim.x;
        if (off < cols) dst[off] = expf(reg[r] - mx) / sum;
    }
}

// upsample_bicubic2d_out_frame (UpSampleBicubic2d.cu, UpSample.cuh's cubic helpers), align_corners
// false, contiguous NCHW f32: a thread per output position, the source index unclamped
__device__ __forceinline__ float torch_cubic_conv1(float x, float A) { return ((A + 2) * x - (A + 3)) * x * x + 1; }
__device__ __forceinline__ float torch_cubic_conv2(float x, float A) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; }
__device__ __forceinline__ float torch_cubic_interp1d(float x0, float x1, float x2, float x3, float t) {
    float coeffs[4];
    float A = -0.75;
    float t1 = t;
    coeffs[0] = torch_cubic_conv2(t1 + 1.0, A);
    coeffs[1] = torch_cubic_conv1(t1, A);
    float t2 = 1.0 - t;
    coeffs[2] = torch_cubic_conv1(t2, A);
    coeffs[3] = torch_cubic_conv2(t2 + 1.0, A);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

extern "C" __global__ void torch_upsample_bicubic2d_f32(const float *idata, float *odata, int n, int batchsize, int channels,
                                                       int height1, int width1, int height2, int width2, float hscale,
                                                       float wscale) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) return;
    const int ox = index % width2, oy = index / width2;
    const float real_x = wscale * (ox + static_cast<float>(0.5)) - static_cast<float>(0.5);
    const int in_x = floorf(real_x);
    const float t_x = real_x - in_x;
    const float real_y = hscale * (oy + static_cast<float>(0.5)) - static_cast<float>(0.5);
    const int in_y = floorf(real_y);
    const float t_y = real_y - in_y;
    auto at = [&](const float *p, int y, int x) {
        const int ay = max(min(y, height1 - 1), 0), ax = max(min(x, width1 - 1), 0);
        return p[(long)ay * width1 + ax];
    };
    for (int b = 0; b < batchsize; b++) {
        for (int c = 0; c < channels; c++) {
            const float *p = idata + ((long)b * channels + c) * height1 * width1;
            float coefficients[4];
            for (int k = 0; k < 4; k++)
                coefficients[k] = torch_cubic_interp1d(at(p, in_y - 1 + k, in_x - 1), at(p, in_y - 1 + k, in_x + 0),
                                                       at(p, in_y - 1 + k, in_x + 1), at(p, in_y - 1 + k, in_x + 2), t_x);
            odata[((long)b * channels + c) * height2 * width2 + (long)oy * width2 + ox] =
                torch_cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y);
        }
    }
}


// upsample_bilinear2d_out_frame with align_corners true (the source index scale * dst, the scale
// (in - 1) / (out - 1)), contiguous NCHW f32
extern "C" __global__ void torch_upsample_bilinear2d_ac_f32(const float *idata, float *odata, int n, int batchsize, int channels,
                                                           int height1, int width1, int height2, int width2, float rheight,
                                                           float rwidth) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        const int w2 = index % width2;
        const int h2 = index / width2;
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;
        const float w1r = rwidth * w2;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;
        for (int b = 0; b < batchsize; b++) {
            for (int c = 0; c < channels; ++c) {
                const float *p = idata + ((long)b * channels + c) * height1 * width1;
                const float val = h0lambda * (w0lambda * p[(long)h1 * width1 + w1] + w1lambda * p[(long)h1 * width1 + w1 + w1p]) +
                                  h1lambda * (w0lambda * p[(long)(h1 + h1p) * width1 + w1] +
                                              w1lambda * p[(long)(h1 + h1p) * width1 + w1 + w1p]);
                odata[((long)b * channels + c) * height2 * width2 + (long)h2 * width2 + w2] = val;
            }
        }
    }
}

// The union ControlNet's QuickGELU on f16, `x * torch.sigmoid(1.702 * x)` as torch runs it: three

// kernels, each in float and rounded to half (the scaled value, its sigmoid, the product)
extern "C" __global__ void torch_quick_gelu_f16(const __half *x, __half *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __half2float(x[i]);
    const float s = __half2float(__float2half_rn(a * 1.702f));
    const float g = __half2float(__float2half_rn(1.0f / (1.0f + expf(-s))));
    y[i] = __float2half_rn(a * g);
}


// PixelNorm's quotient x / root rounded to bf16, then (with `silu`) SiLU of that as nn.SiLU computes
// it, for a contiguous (outer, C, inner) x and its (outer, inner) roots
extern "C" __global__ void ltx_pixel_norm_apply(const __nv_bfloat16 *x, const __nv_bfloat16 *root, __nv_bfloat16 *y, long n, int C,
                                                long inner, int silu) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long b = i / ((long)C * inner), p = i % inner;
    const float q = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) / __bfloat162float(root[b * inner + p])));
    y[i] = __float2bfloat16_rn(silu ? q / (1.0f + expf(-q)) : q);
}

// The Wan VAE's RMS_norm after its norm (F.normalize(x, dim=1) * dim ** 0.5 * gamma + 0), each op
// one torch op in bf16: the norm clamped to 1e-12, the quotient, the product with the Python float
// scale (in float), the product with gamma (C values) and the + 0 each rounded to bf16; then nn.SiLU
// of that when `silu`
extern "C" __global__ void wan_rms_norm_apply(const __nv_bfloat16 *x, const __nv_bfloat16 *norm, const __nv_bfloat16 *gamma,
                                              __nv_bfloat16 *y, long n, int C, long inner, float scale, int silu) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long b = i / ((long)C * inner), c = (i / inner) % C, p = i % inner;
    const float d = __bfloat162float(__float2bfloat16_rn(fmaxf(__bfloat162float(norm[b * inner + p]), 1e-12f)));
    const float q = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) / d));
    const float r = __bfloat162float(__float2bfloat16_rn(q * scale));
    const float g = __bfloat162float(__float2bfloat16_rn(r * __bfloat162float(gamma[c])));
    const float z = __bfloat162float(__float2bfloat16_rn(g + 0.0f));
    y[i] = __float2bfloat16_rn(silu ? z / (1.0f + expf(-z)) : z);
}

// ── Gemma 3 in f32, as ComfyUI's LTX-AV text encoder runs it ───────────────────

// F.rms_norm over rows of N floats (N a multiple of 4, rows 16-byte aligned) as torch's fused kernel
// computes it (layer_norm_kernel.cu, vectorized_layer_norm_kernel<float, float, rms_norm=true>): a
// (32, 4) block per row; thread t adds the squares of the float4s t, t + 128, ... to its sum (each a
// fused multiply-add, as nvcc contracts cuWelfordOnlineSum's sigma2 + val * val), the lanes' sums
// meet by shuffles down, the four warps' through shared memory in halves, the total is divided by N,
// rstd = rsqrtf(that + eps), and each output is w * (rstd * x)
extern "C" __global__ void torch_rms_norm_f32(const float *x, const float *w, float *y, int N, float eps) {
    __shared__ float buf[6];
    const float *row = x + (long)blockIdx.x * N;
    float *out = y + (long)blockIdx.x * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int nvec = N / 4;
    const float4 *xv = reinterpret_cast<const float4 *>(row);
    float s = 0.f;
    for (int i = thrx; i < nvec; i += numx) {
        const float4 d = xv[i];
        s = s + d.x * d.x;
        s = s + d.y * d.y;
        s = s + d.z * d.z;
        s = s + d.w * d.w;
    }
    for (int offset = 16; offset > 0; offset >>= 1) s = s + __shfl_down_sync(0xffffffff, s, offset);
    // compute_stats' inter-warp reduction (sigma2 at buf[2 * y + 1])
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) buf[2 * (threadIdx.y - offset) + 1] = s;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) s = s + buf[2 * threadIdx.y + 1];
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) buf[1] = s / float(N);
    __syncthreads();
    const float rstd = rsqrtf(buf[1] + eps);
    const float4 *wv = reinterpret_cast<const float4 *>(w);
    float4 *ov = reinterpret_cast<float4 *>(out);
    for (int i = thrx; i < nvec; i += numx) {
        const float4 d = xv[i], g = wv[i];
        float4 o;
        o.x = g.x * (rstd * d.x);
        o.y = g.y * (rstd * d.y);
        o.z = g.z * (rstd * d.z);
        o.w = g.w * (rstd * d.w);
        ov[i] = o;
    }
}

// ComfyUI's precompute_freqs_cis for one theta, on the card as torch computes it: inv_freq =
// reciprocal(powf(theta, 2i / dim)) times the reciprocal of rope_scale, freqs = inv_freq * position
// (cuBLAS's one-term product), and CUDA's cosf and sinf of that; (T, dim / 2) each
extern "C" __global__ void gemma_rope_tables(float *cos_out, float *sin_out, int T, int half, float theta, float inv_scale) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T * half) return;
    const int t = (int)(i / half), j = (int)(i % half);
    const float e = (float)(2 * j) * (1.0f / (float)(2 * half));
    const float inv = (1.0f / powf(theta, e)) * inv_scale;
    const float f = inv * (float)t;
    cos_out[i] = cosf(f);
    sin_out[i] = sinf(f);
}

// comfy_kitchen's eager apply_rope_split_half1 (the path an f32 tensor takes: its CUDA kernels take
// f16 and bf16 only, and ComfyUI keeps the Triton backend off) on x (rows, D), row r at position
// r % T: out[i] = fma(x[i + D/2], -sin, x[i] * cos) and out[i + D/2] = fma(x[i], sin, x[i + D/2] * cos),
// the product rounded first (a multiply, then torch's addcmul, which is an explicit fma for alpha 1)
extern "C" __global__ void gemma_rope_f32(const float *x, const float *cosv, const float *sinv, float *y, long rows, int T, int half) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * half) return;
    const long r = i / half;
    const int j = (int)(i % half);
    const int t = (int)(r % T);
    const float c = cosv[(long)t * half + j], s = sinv[(long)t * half + j];
    const float *xr = x + r * 2 * half;
    float *yr = y + r * 2 * half;
    const float x0 = xr[j], x1 = xr[j + half];
    yr[j] = fmaf(x1, -s, x0 * c);
    yr[j + half] = fmaf(x0, s, x1 * c);
}

// The MLP's F.gelu(gate, approximate="tanh") * up in f32, the GELU as torch's GeluCUDAKernelImpl
// computes it (its constants folded in double, then float)
extern "C" __global__ void torch_gelu_tanh_mul_f32(const float *gate, const float *up, float *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kKappa = 0.044715;
    const float x = gate[i];
    const float x_cube = x * x * x;
    const float inner = kBeta * (x + kKappa * x_cube);
    const float g = 0.5f * x * (1.0f + tanhf(inner));
    y[i] = g * up[i];
}

// ── SDXL's CLIP in f32, as ComfyUI runs it ──────────────────────────────────────

// F.layer_norm over rows of N floats (N a multiple of 4, rows 16-byte aligned) as torch's fused kernel
// computes it (layer_norm_kernel.cu, vectorized_layer_norm_kernel<float, float, rms_norm=false>): a
// (32, 4) block per row, each thread a Welford pass over the float4s t, t + 128, ... (welford_online,
// as ltx_layer_norm above), the lanes merged by shuffles down and the warps through shared memory
// (welford_combine, the thread's own state first), sigma2 / N, rstd = rsqrtf(that + eps), and each
// output w * (rstd * (x - mean)) + b
template <typename T>
__device__ void torch_layer_norm(const T *x, const T *w, const T *b, T *y, int N, float eps) {
    __shared__ float buf[6];
    const T *row = x + (long)blockIdx.x * N;
    T *out = y + (long)blockIdx.x * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int nvec = N / 4;
    WelfordLN wd{0.f, 0.f, 0.f};
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) wd = welford_online(to_f<T>(row[i * 4 + ii]), wd);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordLN wdB{__shfl_down_sync(0xffffffff, wd.mean, offset), __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                      __shfl_down_sync(0xffffffff, wd.count, offset)};
        wd = welford_combine(wd, wdB);
    }
    float *meansigmabuf = buf;
    float *countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            meansigmabuf[2 * wrt_y] = wd.mean;
            meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
            countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordLN wdB{meansigmabuf[2 * threadIdx.y], meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y]};
            wd = welford_combine(wd, wdB);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    const float mean = meansigmabuf[0];
    const float rstd = rsqrtf(meansigmabuf[1] + eps);
    for (int i = thrx; i < nvec; i += numx) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const int j = i * 4 + ii;
            out[j] = from_f<T>(to_f<T>(w[j]) * (rstd * (to_f<T>(row[j]) - mean)) + to_f<T>(b[j]));
        }
    }
}

extern "C" __global__ void torch_layer_norm_f32(const float *x, const float *w, const float *b, float *y, int N, float eps) {
    torch_layer_norm<float>(x, w, b, y, N, eps);
}
extern "C" __global__ void layer_norm_f32(const float *x, float *y, const float *w, const float *b, int N, float eps) {
    torch_layer_norm<float>(x, w, b, y, N, eps);
}
extern "C" __global__ void layer_norm_f16(const __half *x, __half *y, const __half *w, const __half *b, int N, float eps) {
    torch_layer_norm<__half>(x, w, b, y, N, eps);
}
extern "C" __global__ void layer_norm_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, const __nv_bfloat16 *w, const __nv_bfloat16 *b, int N,
                                           float eps) {
    torch_layer_norm<__nv_bfloat16>(x, w, b, y, N, eps);
}

// nn.SiLU as torch's silu_kernel computes it: x / (1 + exp(-x)) in f32, rounded once
template <typename T>
__device__ void torch_silu(const T *x, T *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = to_f<T>(x[i]);
    y[i] = from_f<T>(v / (1.0f + expf(-v)));
}
extern "C" __global__ void torch_silu_f16(const __half *x, __half *y, long n) { torch_silu<__half>(x, y, n); }
extern "C" __global__ void torch_silu_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, long n) { torch_silu<__nv_bfloat16>(x, y, n); }
extern "C" __global__ void torch_silu_f32(const float *x, float *y, long n) { torch_silu<float>(x, y, n); }

// comfy's timestep_embedding on the card: freqs = exp(-log(max_period) * arange(half) / half), the
// scalar product rounded, the division by half a multiply by its f32 reciprocal; then t * freqs and
// [cos | sin], f32 (B, 2 half)
extern "C" __global__ void torch_timestep_embedding(const float *t, float *out, int B, int half, float neg_log_period) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * half) return;
    const int bi = (int)(i / half), j = (int)(i % half);
    const float freq = expf((neg_log_period * (float)j) * (1.0f / (float)half));
    const float arg = t[bi] * freq;
    out[(long)bi * 2 * half + j] = cosf(arg);
    out[(long)bi * 2 * half + half + j] = sinf(arg);
}

// ComfyUI's quick_gelu, a * torch.sigmoid(1.702 * a): the scalar product, torch's sigmoid
// (1 / (1 + exp(-x))) and the product, each in f32
extern "C" __global__ void torch_quick_gelu_f32(const float *x, float *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = x[i];
    const float t = a * 1.702f;
    const float s = 1.0f / (1.0f + expf(-t));
    y[i] = a * s;
}

// F.gelu (erf) in f32 as torch's GeluCUDAKernelImpl computes it: x * 0.5 * (1 + erf(x * sqrt(1/2)))
extern "C" __global__ void torch_gelu_erf_f32(const float *x, float *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kAlpha = M_SQRT1_2;
    const float v = x[i];
    y[i] = v * 0.5f * (1.0f + erff(v * kAlpha));
}

// torch's float log (op 0), exp (1) and expm1 (2) kernels, ::log / std::exp / ::expm1 on a float
// built without fast math: the samplers' step sizes, which torch computes on 0-dim CUDA tensors
extern "C" __global__ void torch_unary_math_f32(const float *x, float *y, int n, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = x[i];
  // op 3: torch.logit with no eps, log(x / (1 - x)) in float
  y[i] = op == 0 ? logf(a) : op == 1 ? expf(a) : op == 2 ? expm1f(a) : logf(a / (1.0f - a));
}

// ── RIFE (ComfyUI-Frame-Interpolation's RIFE VFI): torch 2.11's own f32 CUDA kernels ─────────────

// area_pixel_compute_source_index (aten/src/ATen/native/cuda/UpSample.cuh), align_corners false
__device__ __forceinline__ static float torch_area_source_index(float scale, int dst_index) {
    float src_idx = scale * (dst_index + static_cast<float>(0.5)) - static_cast<float>(0.5);
    return (src_idx < static_cast<float>(0)) ? static_cast<float>(0) : src_idx;
}

// upsample_bilinear2d_out_frame (aten/src/ATen/native/cuda/UpSampleBilinear2d.cu), align_corners
// false, on contiguous NCHW f32: a thread per output position, every sample and channel in it
extern "C" __global__ void torch_upsample_bilinear2d_f32(const float *idata, float *odata, int n, int batchsize, int channels,
                                                        int height1, int width1, int height2, int width2, float rheight,
                                                        float rwidth) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        const int w2 = index % width2;
        const int h2 = index / width2;
        const float h1r = torch_area_source_index(rheight, h2);
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;
        const float w1r = torch_area_source_index(rwidth, w2);
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;
        for (int b = 0; b < batchsize; b++) {
            for (int c = 0; c < channels; ++c) {
                const float *p = idata + ((long)b * channels + c) * height1 * width1;
                const float val = h0lambda * (w0lambda * p[(long)h1 * width1 + w1] + w1lambda * p[(long)h1 * width1 + w1 + w1p]) +
                                  h1lambda * (w0lambda * p[(long)(h1 + h1p) * width1 + w1] +
                                              w1lambda * p[(long)(h1 + h1p) * width1 + w1 + w1p]);
                odata[((long)b * channels + c) * height2 * width2 + (long)h2 * width2 + w2] = val;
            }
        }
    }
}

// grid_sampler_compute_source_index (aten/src/ATen/native/cuda/GridSampler.cuh) for border padding
// and align_corners: unnormalize, clip_coordinates, safe_downgrade_to_int_range
__device__ __forceinline__ static float torch_grid_source_index_border(float coord, int size) {
    coord = ((coord + 1.f) / 2) * (size - 1);
    coord = ::min(static_cast<float>(size - 1), ::max(coord, static_cast<float>(0)));
    if (coord > INT_MAX - 1 || coord < INT_MIN || !::isfinite(static_cast<double>(coord)))
        return static_cast<float>(-100.0);
    return coord;
}

__device__ __forceinline__ static bool torch_within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

// grid_sampler_2d_kernel (aten/src/ATen/native/cuda/GridSampler.cu), bilinear with border padding
// and align_corners, f32: input (N, C, H, W) contiguous; the grid given as (N, 2, Ho, Wo)
// contiguous, x then y, which is what RIFE's warp permutes into grid_sample's (N, Ho, Wo, 2)
extern "C" __global__ void torch_grid_sample_border_f32(const float *input, const float *grid, float *output, int nthreads, int C,
                                                       int inp_H, int inp_W, int out_H, int out_W) {
    const long inp_sN = (long)C * inp_H * inp_W, inp_sC = (long)inp_H * inp_W, inp_sH = inp_W, inp_sW = 1;
    const long grid_sN = 2L * out_H * out_W, grid_sH = out_W, grid_sW = 1, grid_sCoor = (long)out_H * out_W;
    const long out_sN = (long)C * out_H * out_W, out_sC = (long)out_H * out_W, out_sH = out_W, out_sW = 1;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        const int w = index % out_W;
        const int h = (index / out_W) % out_H;
        const int n = index / (out_H * out_W);
        const long grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
        float x = grid[grid_offset];
        float y = grid[grid_offset + grid_sCoor];
        float ix = torch_grid_source_index_border(x, inp_W);
        float iy = torch_grid_source_index_border(y, inp_H);
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);
        const float *inp_ptr_NC = input + n * inp_sN;
        float *out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
            float out_acc = 0;
            if (torch_within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                out_acc += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (torch_within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                out_acc += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (torch_within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                out_acc += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (torch_within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                out_acc += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
            }
            *out_ptr_NCHW = out_acc;
        }
    }
}

// linspace_cuda_out's float lambda (aten/src/ATen/native/cuda/RangeFactories.cu); `step` is
// (end - start) / (steps - 1) in float, made on the host
extern "C" __global__ void torch_linspace_f32(float *out, long steps, float scalar_start, float scalar_end, float step) {
    const long ind = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= steps) return;
    const long halfway = steps / 2;
    if (ind < halfway) {
        out[ind] = scalar_start + (step * ind);
        return;
    }
    out[ind] = scalar_end - step * (steps - ind - 1);
}

// leaky_relu_kernel (aten/src/ATen/native/cuda/ActivationLeakyReluKernel.cu) on f32
extern "C" __global__ void torch_leaky_relu_f32(const float *x, float *y, long n, float negval) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float aop = x[i];
    y[i] = aop > 0.0f ? aop : aop * negval;
}

// sigmoid_kernel_cuda's float lambda (aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu)
extern "C" __global__ void torch_sigmoid_f32(const float *x, float *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float one = 1.0f;
    y[i] = one / (one + expf(-x[i]));
}

// er_sde's default noise scaler, x * ((x ** 0.3).exp() + 10.0), as torch's float kernels compute
// it one op at a time: powf with the exponent as a float, expf, the add, the product
extern "C" __global__ void torch_er_noise_scaler_f32(const float *x, float *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = x[i];
    const float p = powf(a, 0.3f);
    const float e = expf(p);
    const float s = e + 10.0f;
    y[i] = a * s;
}

// ── SAM3 (ComfyUI-RMBG's SAM3Segment): torch 2.11's kernels as its bf16 autocast runs them ────

// upsample_gen2d_aa_out_frame (UpSampleBilinear2d.cu) with the bilinear filter, f32 planes: each
// output pixel's span and weights (UpSample.cuh: the filter at (j + xmin - center + 0.5) *
// invscale, invscale 1 / scale in double when downscaling, each weight divided by their float
// sum), the source rows of the span interpolated along x, then those along y, each a running
// fused multiply-add from the first product
__device__ int sam3_aa_weights(float scale, float support, int i, int in_size, float *w, int &xmin) {
    const float center = scale * (i + 0.5f);
    xmin = max((int)(center - support + 0.5f), 0);
    const int xsize = min((int)(center + support + 0.5f), in_size) - xmin;
    const float invscale = scale >= 1.0f ? (float)(1.0 / (double)scale) : 1.0f;
    const float xmin_m_center = xmin - center;
    float total = 0.0f;
    for (int j = 0; j < xsize; j++) {
        float x = (j + xmin_m_center + 0.5f) * invscale;
        if (x < 0) x = -x;
        const float v = x < 1 ? 1 - x : 0;
        w[j] = v;
        total += v;
    }
    for (int j = 0; j < xsize; j++)
        if (total != 0.0f) w[j] /= total;
    return xsize;
}

extern "C" __global__ void torch_upsample_aa_bilinear_f32(const float *src, float *dst, int planes, int ih, int iw, int oh, int ow, float hs, float ws) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= ow || oy >= oh) return;
    const float support_h = hs >= 1.0f ? 1.0f * hs : 1.0f;
    const float support_w = ws >= 1.0f ? 1.0f * ws : 1.0f;
    float wx[64], wy[64], col[64];
    int xmin, ymin;
    const int xsize = sam3_aa_weights(ws, support_w, ox, iw, wx, xmin);
    const int ysize = sam3_aa_weights(hs, support_h, oy, ih, wy, ymin);
    for (int p = blockIdx.z; p < planes; p += gridDim.z) {
        const float *plane = src + (long)p * ih * iw;
        for (int y = 0; y < ysize; y++) {
            const float *row = plane + (long)(ymin + y) * iw + xmin;
            float acc = row[0] * wx[0];
            for (int j = 1; j < xsize; j++) acc = __fmaf_rn(row[j], wx[j], acc);
            col[y] = acc;
        }
        float acc = col[0] * wy[0];
        for (int j = 1; j < ysize; j++) acc = __fmaf_rn(col[j], wy[j], acc);
        dst[((long)p * oh + oy) * ow + ox] = acc;
    }
}

// SAM3's ViT attention on its qkv rows (B * L, 3 * C) bf16: apply_rotary_enc on q and k, each
// head's (x[2i], x[2i + 1]) a complex number times freqs_cis[l][i] in f32, c10::complex's
// (a*c - b*d, a*d + b*c) fused as nvcc fuses them, rounded back to bf16; q, k and v written as
// (B, L, heads, D) for FlashAttention
extern "C" __global__ void sam3_rope_qkv_bf16(const __nv_bfloat16 *qkv, const float *freqs, __nv_bfloat16 *q, __nv_bfloat16 *k, __nv_bfloat16 *v, long rows, int L, int C, int half_d) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = C / 2;
    if (idx >= rows * pairs) return;
    const long row = idx / pairs;
    const int pair = (int)(idx % pairs);
    const int l = (int)(row % L);
    const int i = pair % half_d;
    const float c = freqs[((long)l * half_d + i) * 2], d = freqs[((long)l * half_d + i) * 2 + 1];
    const __nv_bfloat16 *s = qkv + row * 3 * C;
    const long o = row * C + 2 * pair;
    float a = __bfloat162float(s[2 * pair]), b = __bfloat162float(s[2 * pair + 1]);
    q[o] = __float2bfloat16(__fmaf_rn(a, c, -__fmul_rn(b, d)));
    q[o + 1] = __float2bfloat16(__fmaf_rn(a, d, __fmul_rn(b, c)));
    a = __bfloat162float(s[C + 2 * pair]);
    b = __bfloat162float(s[C + 2 * pair + 1]);
    k[o] = __float2bfloat16(__fmaf_rn(a, c, -__fmul_rn(b, d)));
    k[o + 1] = __float2bfloat16(__fmaf_rn(a, d, __fmul_rn(b, c)));
    v[o] = s[2 * C + 2 * pair];
    v[o + 1] = s[2 * C + 2 * pair + 1];
}

// nn.GELU() (erf) on bf16 as torch's GeluCUDAKernelImpl computes it: in f32, x * 0.5 * (1 +
// erf(x * sqrt(1/2))), rounded to bf16
extern "C" __global__ void torch_gelu_erf_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kAlpha = M_SQRT1_2;
    const float v = __bfloat162float(x[i]);
    y[i] = __float2bfloat16(v * 0.5f * (1.0f + erff(v * kAlpha)));
}

// PositionEmbeddingSine (SAM3's: 256 features, temperature 10000, normalized, scale 2 pi) for an
// (H, W) map, op by op as torch computes it on the card in f32: (h + 1) / (H + 1e-6) * 2pi (and
// likewise for w), divided by powf(10000, 2 * (k // 2) / 128), sin on even features and cos on
// odd ones; y's 128 features then x's, (256, H, W) rounded to bf16
extern "C" __global__ void sam3_pos_sine_bf16(__nv_bfloat16 *out, int H, int W) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256L * H * W) return;
    const int f = (int)(i / ((long)H * W));
    const int hw = (int)(i % ((long)H * W));
    const int h = hw / W, w = hw % W;
    const float eps = 1e-6f, scale = 6.283185307179586f;
    const bool is_x = f >= 128;
    const int k = f % 128;
    const float e = is_x ? (float)(w + 1) / ((float)W + eps) * scale : (float)(h + 1) / ((float)H + eps) * scale;
    const float q = (float)(k / 2);
    const float dim_t = powf(10000.0f, (2.0f * q) / 128.0f);
    const float p = e / dim_t;
    out[i] = __float2bfloat16(k % 2 == 0 ? sinf(p) : cosf(p));
}

// gen_sineembed_for_position (SAM3's model_misc.py) for boxes (Q, 4) f32 (cx, cy, w, h), 256
// features: each coordinate times 2pi, divided by powf(10000, 2 * (k // 2) / 128), sin on even
// features and cos on odd ones; y's 128, then x's, w's and h's: (Q, 512) f32
extern "C" __global__ void sam3_box_sine_f32(const float *box, float *out, int q) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)q * 512) return;
    const int row = (int)(i / 512), f = (int)(i % 512);
    const int part = f / 128, k = f % 128;
    const int coord = part == 0 ? 1 : (part == 1 ? 0 : part);
    const float e = __fmul_rn(box[row * 4 + coord], 6.283185307179586f);
    const float dim_t = powf(10000.0f, (2.0f * (float)(k / 2)) / 128.0f);
    const float p = __fdiv_rn(e, dim_t);
    out[i] = k % 2 == 0 ? sinf(p) : cosf(p);
}

// _get_rpb_matrix's log-scaled deltas for boxes (Q, 4) cxcywh f32 on an (H, W) map: the box's
// corners (c -/+ 0.5 * size), each map coordinate (j / W, i / H) minus them, times 8, then
// sign(d) * log2(|d| + 1) / 3; dx (Q, W, 2) and dy (Q, H, 2), op by op as torch runs them. The
// division by 3 (a Python number) is a product with its float reciprocal (div_true_kernel_cuda);
// the coordinates divide by the map size as a 0-dim tensor, a true division.
extern "C" __global__ void sam3_rpb_deltas_f32(const float *box, float *dx, float *dy, int q, int H, int W) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long nx = (long)q * W * 2, ny = (long)q * H * 2;
    if (i >= nx + ny) return;
    const bool is_x = i < nx;
    const long j = is_x ? i : i - nx;
    const int n = is_x ? W : H;
    const int row = (int)(j / (n * 2)), pos = (int)((j / 2) % n), side = (int)(j % 2);
    const float c = box[row * 4 + (is_x ? 0 : 1)], s = box[row * 4 + (is_x ? 2 : 3)];
    const float half = __fmul_rn(0.5f, s);
    const float corner = side == 0 ? __fsub_rn(c, half) : __fadd_rn(c, half);
    const float coord = __fdiv_rn((float)pos, (float)n);
    const float d = __fmul_rn(__fsub_rn(coord, corner), 8.0f);
    const float sg = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
    const float v = __fmul_rn(__fmul_rn(sg, log2f(__fadd_rn(fabsf(d), 1.0f))), __fdiv_rn(1.0f, 3.0f));
    (is_x ? dx : dy)[j] = v;
}

// _get_rpb_matrix's B with the presence token's row of zeros first: (heads, 1 + Q, H * W) bf16,
// dy (Q, H, heads) + dx (Q, W, heads) added in f32 and rounded as torch adds bf16 tensors
extern "C" __global__ void sam3_rpb_bias_bf16(const __nv_bfloat16 *dy, const __nv_bfloat16 *dx, __nv_bfloat16 *out, int q, int H, int W, int heads) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long per_head = (long)(q + 1) * H * W;
    if (i >= heads * per_head) return;
    const int h = (int)(i / per_head);
    const long r = i % per_head;
    const int qi = (int)(r / ((long)H * W));
    const int hw = (int)(r % ((long)H * W));
    if (qi == 0) {
        out[i] = __float2bfloat16(0.0f);
        return;
    }
    const int y = hw / W, x = hw % W;
    const float a = __bfloat162float(dy[((long)(qi - 1) * H + y) * heads + h]);
    const float b = __bfloat162float(dx[((long)(qi - 1) * W + x) * heads + h]);
    out[i] = __float2bfloat16_rn(a + b);
}

// SAM3's box refinement in f32 op by op as torch runs it: sigmoid(delta + inverse_sigmoid(ref)),
// inverse_sigmoid being log(clamp(x, 1e-3) / clamp(1 - x, 1e-3)) of x clamped to [0, 1] and
// sigmoid 1 / (1 + exp(-x)); with no delta (null) the sigmoid of ref alone
extern "C" __global__ void sam3_refine_f32(const float *ref, const __nv_bfloat16 *delta, float *out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = ref[i];
    if (delta != nullptr) {
        const float x = fminf(fmaxf(s, 0.0f), 1.0f);
        const float x1 = fmaxf(x, 0.001f), x2 = fmaxf(__fsub_rn(1.0f, x), 0.001f);
        s = __fadd_rn(__bfloat162float(delta[i]), logf(__fdiv_rn(x1, x2)));
    }
    out[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-s)));
}
