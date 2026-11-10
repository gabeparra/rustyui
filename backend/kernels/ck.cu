// comfy_kitchen's CUDA kernels that ComfyUI's LTX path runs under a cu130 torch (comfy_kitchen
// 0.2.37: backends/cuda/ops/adaln.cu, rms_rope.cu and rope_device.cuh, per_tensor_quantize.cu).
// Their arithmetic is copied unchanged and this file is compiled as comfy_kitchen compiles them
// (--use_fast_math -O3), so the results round the way ComfyUI's do; only the entry points are
// rustyui's, one per instantiation it launches. comfy_kitchen is Apache-2.0: copyright (c) 2025
// Comfy Org and NVIDIA CORPORATION & AFFILIATES, see kernels/LICENSE-comfy-kitchen.
//
// Changes from the originals: the template kernels are __device__ functions called from
// extern "C" __global__ wrappers; the RoPE kernel keeps only its no-RMS, contiguous-head paths
// (what ComfyUI's apply_rope* calls take for LTX's (B, T, heads, dim) tensors), one thread per
// rotated pair rather than two (each pair's arithmetic is its own, so the results are the same).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace comfy {

constexpr int kThreadsPerWarp = 32;

// utils.cuh: which modulation row output row `row` reads (`group` consecutive rows share one)
__device__ __forceinline__ int modulation_row(int row, int group, int n_rows) {
    if (group == 1) return row;
    if (group == n_rows) return 0;
    if ((group & (group - 1)) == 0) return row >> (__ffs(group) - 1);
    return row / group;
}

// ── adaln.cu ────────────────────────────────────────────────────────────────

constexpr int kAdaLNThreads = 256;
constexpr int kAdaLNWarps = kAdaLNThreads / kThreadsPerWarp;  // 8
constexpr int kAdaLNRowsPerWarpBlock = kAdaLNWarps;

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

template<typename T> struct VecWidth { static constexpr int value = 16 / sizeof(T); };

template<typename T>
struct alignas(16) Vec {
    static constexpr int W = VecWidth<T>::value;
    T elts[W];
};

__device__ __forceinline__ float2 warp_reduce_sum2(float2 v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(0xffffffff, v.x, offset);
        v.y += __shfl_down_sync(0xffffffff, v.y, offset);
    }
    return v;
}

__device__ __forceinline__ float2 block_reduce_sum2(float2 v, float2* warp_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum2(v);
    if (lane == 0) warp_smem[wid] = v;
    __syncthreads();

    float2 total = make_float2(0.0f, 0.0f);
    for (int w = 0; w < kAdaLNWarps; ++w) { total.x += warp_smem[w].x; total.y += warp_smem[w].y; }
    return total;
}

template<bool kSubtractMean>
__device__ __forceinline__ void norm_stats(float2 acc, float inv_d, float eps,
                                           float& mean, float& rstd) {
    if constexpr (kSubtractMean) {
        mean = acc.x * inv_d;
        rstd = rsqrtf(fmaxf(acc.y * inv_d - mean * mean, 0.0f) + eps);
    } else {
        mean = 0.0f;
        rstd = rsqrtf(acc.y * inv_d + eps);
    }
}

// one row per 256-thread block (more than 1024 rows)
template<typename T, bool kSubtractMean>
__device__ __forceinline__ void adaln_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int D,
    int scale_group,
    int shift_group,
    float eps)
{
    constexpr int VEC = VecWidth<T>::value;
    const int row = static_cast<int>(blockIdx.x);
    const int n_rows = static_cast<int>(gridDim.x);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float2 warp_smem[kAdaLNWarps];

    const int scale_row = modulation_row(row, scale_group, n_rows);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, n_rows);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    const int n_vec   = (D % VEC == 0) ? (D / VEC) : 0;
    const int vec_end = n_vec * VEC;

    const Vec<T>* x_vec = reinterpret_cast<const Vec<T>*>(x_row);

    float2 acc = make_float2(0.0f, 0.0f);
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv = x_vec[v];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = to_float(xv.elts[j]);
            if constexpr (kSubtractMean) acc.x += f;
            acc.y += f * f;
        }
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float f = to_float(x_row[i]);
        if constexpr (kSubtractMean) acc.x += f;
        acc.y += f * f;
    }

    acc = block_reduce_sum2(acc, warp_smem);
    const float inv_d = 1.0f / static_cast<float>(D);
    float mean, rstd;
    norm_stats<kSubtractMean>(acc, inv_d, eps, mean, rstd);

    const Vec<T>* s_vec  = reinterpret_cast<const Vec<T>*>(s_row);
    const Vec<T>* sh_vec = reinterpret_cast<const Vec<T>*>(sh_row);
    Vec<T>*       o_vec  = reinterpret_cast<Vec<T>*>(o_row);
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv  = x_vec[v];
        Vec<T> sv  = s_vec[v];
        Vec<T> shv = sh_vec[v];
        Vec<T> ov;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float result = (to_float(xv.elts[j]) - mean) * rstd
                           * (1.0f + to_float(sv.elts[j])) + to_float(shv.elts[j]);
            ov.elts[j] = from_float<T>(result);
        }
        o_vec[v] = ov;
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float result = (to_float(x_row[i]) - mean) * rstd
                       * (1.0f + to_float(s_row[i])) + to_float(sh_row[i]);
        o_row[i] = from_float<T>(result);
    }
}

// one row per warp, 8 rows per 256-thread block (up to 1024 rows)
template<typename T, bool kSubtractMean>
__device__ __forceinline__ void adaln_warp_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int N,
    int D,
    int scale_group,
    int shift_group,
    float eps)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int row = blockIdx.x * kAdaLNRowsPerWarpBlock + warp_in_block;
    if (row >= N) return;

    const int scale_row = modulation_row(row, scale_group, N);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, N);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    float2 acc = make_float2(0.0f, 0.0f);
    for (int i = lane; i < D; i += kThreadsPerWarp) {
        float f = to_float(x_row[i]);
        if constexpr (kSubtractMean) acc.x += f;
        acc.y += f * f;
    }

    acc = warp_reduce_sum2(acc);
    acc.x = __shfl_sync(0xffffffff, acc.x, 0);
    acc.y = __shfl_sync(0xffffffff, acc.y, 0);
    const float inv_d = 1.0f / static_cast<float>(D);
    float mean, rstd;
    norm_stats<kSubtractMean>(acc, inv_d, eps, mean, rstd);

    for (int i = lane; i < D; i += kThreadsPerWarp) {
        float result = (to_float(x_row[i]) - mean) * rstd
                       * (1.0f + to_float(s_row[i])) + to_float(sh_row[i]);
        o_row[i] = from_float<T>(result);
    }
}

// ── rope_device.cuh and rms_rope.cu ─────────────────────────────────────────

namespace rope {

template <typename C>
__device__ __forceinline__ void rotate(C x0, C x1, C f00, C f01, C f10,
                                       C f11, C &y0, C &y1) {
  y0 = f00 * x0 + f01 * x1;
  y1 = f10 * x0 + f11 * x1;
}

} // namespace rope

// rms_sum<T, ContigHead=true>: lane l's fused multiply-adds of elements l, l + 32, ..., then the
// warp's shuffle-down sum, lane 0's broadcast
template <typename T>
__device__ __forceinline__ float rms_sum_contig(const T *row, int head_dim, int lane) {
  float sum = 0.0f;
  for (int element = lane; element < head_dim; element += 32) {
    const float value = static_cast<float>(row[element]);
    sum = fmaf(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  return __shfl_sync(0xffffffffu, sum, 0);
}

// rope_kernel<InputType, FreqsType, ScaleType, HasRms=true, SplitHalf=true, HasK=true,
// InPlace=false, ContigHead=true> with rot_dim = head_dim (rms_rope_split_half): a warp a row of q
// and of k (B, S, H, D) contiguous, 4 warps a block; each head RMS-normalised with its scale and
// rounded to InputType, then rotated in float; freqs (freqs_batch, freqs_dim1, freqs_dim2, pairs,
// 2, 2) contiguous. Each lane's pairs are its own arithmetic, so lane l takes pairs l, l + 32, ...
template <typename InputType, typename FreqsType, typename ScaleType>
__device__ __forceinline__ void rms_rope_split_row(const InputType *q, const InputType *k, const FreqsType *freqs,
                                                   const ScaleType *q_scale, const ScaleType *k_scale, InputType *q_out,
                                                   InputType *k_out, long batch, long dim1, long dim2, int head_dim,
                                                   long freqs_batch, long freqs_dim1, long freqs_dim2, float epsilon) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long row = (long)blockIdx.x * 4 + warp;
  const long rows = batch * dim1 * dim2;
  if (row >= rows) return;
  const long i2 = row % dim2;
  const long tmp = row / dim2;
  const long i1 = tmp % dim1;
  const long i0 = tmp / dim1;
  const long base = row * head_dim;
  const float q_sum = rms_sum_contig(q + base, head_dim, lane);
  const float q_rrms = rsqrtf(q_sum / static_cast<float>(head_dim) + epsilon);
  const float k_sum = rms_sum_contig(k + base, head_dim, lane);
  const float k_rrms = rsqrtf(k_sum / static_cast<float>(head_dim) + epsilon);
  const long fi0 = freqs_batch == 1 ? 0 : i0;
  const long fi1 = freqs_dim1 == 1 ? 0 : i1;
  const long fi2 = freqs_dim2 == 1 ? 0 : i2;
  const int pairs = head_dim / 2;
  const long freq_row = ((fi0 * freqs_dim1 + fi1) * freqs_dim2 + fi2) * (long)pairs * 4;
  for (int pair = lane; pair < pairs; pair += 32) {
    const int first = pair, second = pair + pairs;
    float q0 = static_cast<float>(q[base + first]);
    float q1 = static_cast<float>(q[base + second]);
    q0 = static_cast<float>(static_cast<InputType>(q0 * q_rrms * static_cast<float>(q_scale[first])));
    q1 = static_cast<float>(static_cast<InputType>(q1 * q_rrms * static_cast<float>(q_scale[second])));
    const long fb = freq_row + (long)pair * 4;
    const float f00 = static_cast<float>(freqs[fb]);
    const float f01 = static_cast<float>(freqs[fb + 1]);
    const float f10 = static_cast<float>(freqs[fb + 2]);
    const float f11 = static_cast<float>(freqs[fb + 3]);
    float y0, y1;
    rope::rotate(q0, q1, f00, f01, f10, f11, y0, y1);
    q_out[base + first] = static_cast<InputType>(y0);
    q_out[base + second] = static_cast<InputType>(y1);
    float k0 = static_cast<float>(k[base + first]);
    float k1 = static_cast<float>(k[base + second]);
    k0 = static_cast<float>(static_cast<InputType>(k0 * k_rrms * static_cast<float>(k_scale[first])));
    k1 = static_cast<float>(static_cast<InputType>(k1 * k_rrms * static_cast<float>(k_scale[second])));
    rope::rotate(k0, k1, f00, f01, f10, f11, y0, y1);
    k_out[base + first] = static_cast<InputType>(y0);
    k_out[base + second] = static_cast<InputType>(y1);
  }
}

// rope_kernel<InputType, FreqsType, float, HasRms=false, SplitHalf, HasK=false, InPlace=false,
// ContigHead=true>, for one rotated pair: x (rows, head_dim) contiguous, freqs (rows of the
// broadcast, pairs, 2, 2) contiguous. ComputeType is FreqsType without the RMS step.
template <typename InputType, typename FreqsType, bool SplitHalf>
__device__ __forceinline__ void rope_pair(const InputType *x, const FreqsType *freqs, InputType *out,
                                          long rows, int head_dim, long dim1, long dim2,
                                          long freqs_batch, long freqs_dim1, long freqs_dim2) {
  using ComputeType = FreqsType;
  const int pairs = head_dim / 2;
  const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rows * pairs) return;
  const long row = gid / pairs;
  const int pair = (int)(gid % pairs);

  const long i2 = row % dim2;
  const long tmp = row / dim2;
  const long i1 = tmp % dim1;
  const long i0 = tmp / dim1;
  const long fi0 = freqs_batch == 1 ? 0 : i0;
  const long fi1 = freqs_dim1 == 1 ? 0 : i1;
  const long fi2 = freqs_dim2 == 1 ? 0 : i2;
  // freqs (freqs_batch, freqs_dim1, freqs_dim2, pairs, 2, 2), contiguous
  const long freq_row = ((fi0 * freqs_dim1 + fi1) * freqs_dim2 + fi2) * (long)pairs * 4;

  const InputType *xr = x + row * head_dim;
  InputType *yr = out + row * head_dim;
  const int first = SplitHalf ? pair : pair * 2;
  const int second = SplitHalf ? pair + pairs : first + 1;
  const ComputeType q0 = static_cast<ComputeType>(xr[first]);
  const ComputeType q1 = static_cast<ComputeType>(xr[second]);
  const long base = freq_row + (long)pair * 4;
  const ComputeType f00 = static_cast<ComputeType>(freqs[base]);
  const ComputeType f01 = static_cast<ComputeType>(freqs[base + 1]);
  const ComputeType f10 = static_cast<ComputeType>(freqs[base + 2]);
  const ComputeType f11 = static_cast<ComputeType>(freqs[base + 3]);
  ComputeType y0, y1;
  rope::rotate(q0, q1, f00, f01, f10, f11, y0, y1);
  yr[first] = static_cast<InputType>(y0);
  yr[second] = static_cast<InputType>(y1);
}

// ── per_tensor_quantize.cu ───────────────────────────────────────────────────

__forceinline__ __device__ float clamp(float val, float min, float max) {
    return fminf(max, fmaxf(min, val));
}

// float_utils.cuh's FP8LimitsTrait, for e4m3
constexpr float kFP8E4M3Max = 448.0f;

// dequantize_fp8_tensor_kernel, one value: f32(e4m3) * the f32 scale, rounded once
template<typename OutputType>
__device__ __forceinline__ void dequantize_fp8(const uint8_t *src, OutputType *dst, const float *scale_f, long n) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __nv_fp8_e4m3 v;
    v.__x = src[idx];
    dst[idx] = static_cast<OutputType>(static_cast<float>(v) * *scale_f);
}

// stochastic_round_fp8_kernel<InputType, __nv_fp8_e4m3>
template<typename InputType>
__device__ __forceinline__ void stochastic_round_fp8_kernel(
    const uint8_t *rng,
    uint8_t *dst,
    const InputType *src,
    const long size) {

    using OutputType = __nv_fp8_e4m3;
    constexpr float kFP8Max = kFP8E4M3Max;
    constexpr int kExponentBits = 4;
    constexpr int kMantissaBits = 3;
    constexpr int kExponentBias = 7;
    constexpr float kMantissaLevels = static_cast<float>(1 << kMantissaBits);
    const float subnormal_mantissa_scale = exp2f(-kExponentBias + 1 - kMantissaBits);
    const float subnormal_value_scale = exp2f(-kExponentBias + 1);

    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    const float value = __half2float(__float2half(static_cast<float>(src[idx])));
    const float abs_value = fabsf(value);
    const float sign = value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);

    float exponent = floorf(log2f(abs_value)) + kExponentBias;
    exponent = clamp(exponent, 0.0f, static_cast<float>((1 << kExponentBits) - 1));
    const bool normal = exponent != 0.0f;
    const float exponent_scale = exp2f(exponent - kExponentBias);

    const float normal_mantissa = (abs_value / exponent_scale - 1.0f) * kMantissaLevels;
    const float subnormal_mantissa = abs_value / subnormal_mantissa_scale;
    const float random = static_cast<float>(rng[idx]) * (1.0f / 256.0f);
    const float mantissa = floorf((normal ? normal_mantissa : subnormal_mantissa) + random) / kMantissaLevels;
    const float rounded = sign * (normal ? exponent_scale * (1.0f + mantissa) : subnormal_value_scale * mantissa);

    union {
        OutputType fp8;
        uint8_t u8;
    } out;
    out.fp8 = static_cast<OutputType>(clamp(rounded, -kFP8Max, kFP8Max));
    dst[idx] = out.u8;
}

} // namespace comfy

using comfy::adaln_kernel;
using comfy::adaln_warp_kernel;

extern "C" __global__ void __launch_bounds__(256) ck_rms_adaln_bf16(
    const nv_bfloat16 *x, const nv_bfloat16 *scale, const nv_bfloat16 *shift, nv_bfloat16 *out,
    int D, int scale_group, int shift_group, float eps) {
    adaln_kernel<nv_bfloat16, false>(x, scale, shift, out, D, scale_group, shift_group, eps);
}

extern "C" __global__ void __launch_bounds__(256) ck_rms_adaln_warp_bf16(
    const nv_bfloat16 *x, const nv_bfloat16 *scale, const nv_bfloat16 *shift, nv_bfloat16 *out,
    int N, int D, int scale_group, int shift_group, float eps) {
    adaln_warp_kernel<nv_bfloat16, false>(x, scale, shift, out, N, D, scale_group, shift_group, eps);
}

// apply_rope_split_half1 / apply_rope_split_half: bf16 x, bf16 or f32 rotation matrices
extern "C" __global__ void ck_rope_split_bf16(const nv_bfloat16 *x, const nv_bfloat16 *freqs, nv_bfloat16 *out, long rows, int head_dim,
                                              long dim1, long dim2, long freqs_batch, long freqs_dim1, long freqs_dim2) {
    comfy::rope_pair<nv_bfloat16, nv_bfloat16, true>(x, freqs, out, rows, head_dim, dim1, dim2, freqs_batch, freqs_dim1, freqs_dim2);
}

// rms_rope_split_half: bf16 q and k with bf16 RMSNorm scales and f32 rotation matrices
extern "C" __global__ void __launch_bounds__(128) ck_rms_rope_split_bf16_f32(
    const nv_bfloat16 *q, const nv_bfloat16 *k, const float *freqs, const nv_bfloat16 *q_scale, const nv_bfloat16 *k_scale,
    nv_bfloat16 *q_out, nv_bfloat16 *k_out, long batch, long dim1, long dim2, int head_dim, long freqs_batch, long freqs_dim1,
    long freqs_dim2, float epsilon) {
    comfy::rms_rope_split_row<nv_bfloat16, float, nv_bfloat16>(q, k, freqs, q_scale, k_scale, q_out, k_out, batch, dim1, dim2,
                                                                head_dim, freqs_batch, freqs_dim1, freqs_dim2, epsilon);
}

extern "C" __global__ void ck_rope_split_bf16_f32(const nv_bfloat16 *x, const float *freqs, nv_bfloat16 *out, long rows, int head_dim,
                                                  long dim1, long dim2, long freqs_batch, long freqs_dim1, long freqs_dim2) {
    comfy::rope_pair<nv_bfloat16, float, true>(x, freqs, out, rows, head_dim, dim1, dim2, freqs_batch, freqs_dim1, freqs_dim2);
}

// dequantize_per_tensor_fp8: e4m3 bytes and one f32 scale, to bf16
extern "C" __global__ void ck_dequant_e4m3_bf16(const uint8_t *q, const float *scale, nv_bfloat16 *out, long n) {
    comfy::dequantize_fp8<nv_bfloat16>(q, out, scale, n);
}

// stochastic_rounding_fp8 to e4m3 from fp16: the random bytes in, the e4m3 bytes out (comfy_kitchen
// writes them over the random bytes)
extern "C" __global__ void ck_stochastic_round_e4m3_f16(const half *src, const uint8_t *rng, uint8_t *out, long n) {
    comfy::stochastic_round_fp8_kernel<half>(rng, out, src, n);
}

// quantize_fp8_tensor_kernel (per_tensor_quantize.cu) from bf16, one value: f32(x) / scale (a
// fast-math division, as comfy_kitchen builds it), clamped to the e4m3 range, rounded to nearest
extern "C" __global__ void ck_quantize_e4m3_bf16(const nv_bfloat16 *src, uint8_t *dst, const float *scale_f, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float scaled_val = static_cast<float>(src[i]) / *scale_f;
    scaled_val = comfy::clamp(scaled_val, -comfy::kFP8E4M3Max, comfy::kFP8E4M3Max);
    dst[i] = static_cast<__nv_fp8_e4m3>(scaled_val).__x;
}

// stochastic_rounding_fp8 to e4m3 from bf16 (a LoRA'd weight requantized by ComfyUI's DynamicVRAM path)
extern "C" __global__ void ck_stochastic_round_e4m3_bf16(const nv_bfloat16 *src, const uint8_t *rng, uint8_t *out, long n) {
    comfy::stochastic_round_fp8_kernel<nv_bfloat16>(rng, out, src, n);
}
