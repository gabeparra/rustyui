// PyTorch's memory-efficient attention forward (xformers' CUTLASS FMHA as torch 2.11 carries it,
// third_party/torch-mem-eff), the kernels torch picks for ComfyUI's attention where flash attention
// does not take the call (Gemma's masked f32 one, SDXL CLIP's causal f32 one, the SDXL VAE's bf16
// one with its 512-wide head), with no other code path in between: torch's three sm80 variants for
// f32 and for bf16, defined here as torch's generated cutlassF_f32_aligned.cu and
// cutlassF_bf16_aligned.cu define them, the one for the call chosen as _efficient_attention_forward
// chooses it. Their products run on the tensor cores (f32 as CUTLASS's 3xTF32 emulation) and their
// softmax block by block, so their numbers come only from these kernels, built from this source
// against torch's own CUTLASS commit.

#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#include <cstdint>
#include <cstdio>

using namespace PyTorchMemEffAttention;

template <typename T, int Q, int K, int MAXK>
using Fmha = AttentionKernel<T, cutlass::arch::Sm80, true, Q, K, MAXK, true, true>;

// torch's generated cutlassF_*_aligned.cu define these for sm80, and dispatch_cutlassF_*_sm80 offers
// them in this order; _efficient_attention_forward launches the first that takes the call
#define RUSTYUI_FMHA(NAME, T, Q, K, MAXK)                                                                \
  __global__ void __launch_bounds__(Fmha<T, Q, K, MAXK>::kNumThreads, Fmha<T, Q, K, MAXK>::kMinBlocksPerSm) \
  NAME(typename Fmha<T, Q, K, MAXK>::Params p) {                                                         \
    RUSTYUI_FMHA_BODY(NAME, T, Q, K, MAXK)                                                               \
  }

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ <= 1200
#define RUSTYUI_FMHA_BODY(NAME, T, Q, K, MAXK) \
  if (!p.advance_to_block()) {                  \
    return;                                     \
  }                                             \
  Fmha<T, Q, K, MAXK>::attention_kernel(p);     \
  return;
#else
#define RUSTYUI_FMHA_BODY(NAME, T, Q, K, MAXK) printf("FATAL: kernel `" #NAME "` is for sm80-sm100, but was built for sm%d\n", int(__CUDA_ARCH__ + 0) / 10);
#endif
#else
#define RUSTYUI_FMHA_BODY(NAME, T, Q, K, MAXK)
#endif

RUSTYUI_FMHA(fmha_cutlassF_f32_aligned_64x64_rf_sm80, float, 64, 64, 64)
RUSTYUI_FMHA(fmha_cutlassF_f32_aligned_64x128_rf_sm80, float, 64, 128, 128)
RUSTYUI_FMHA(fmha_cutlassF_f32_aligned_32x128_gmem_sm80, float, 32, 128, 65536)
RUSTYUI_FMHA(fmha_cutlassF_bf16_aligned_64x64_rf_sm80, cutlass::bfloat16_t, 64, 64, 64)
RUSTYUI_FMHA(fmha_cutlassF_bf16_aligned_64x128_rf_sm80, cutlass::bfloat16_t, 64, 128, 128)
RUSTYUI_FMHA(fmha_cutlassF_bf16_aligned_32x128_gmem_sm80, cutlass::bfloat16_t, 32, 128, 65536)

namespace {

bool fits_i32(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

struct Call {
  cudaStream_t stream;
  const void *q, *k, *v, *bias;
  void *out;
  float *accum;
  int batches, heads, queries, keys, head_dim;
  const int64_t *q_strides, *k_strides, *v_strides, *bias_strides;
  float scale;
};

// launchKernel for one kernel: -1 when it does not take the call (the next is tried), 0 when it ran,
// 1 on an error (in err)
template <typename Kernel, typename Fn>
int try_kernel(Fn kernel_fn, const Call &c, int max_shmem, char *err, int err_len) {
  using T = typename Kernel::scalar_t;
  if (c.head_dim > Kernel::kMaxK) return -1;
  if ((c.q_strides[2] % Kernel::kAlignmentQ) || (c.k_strides[2] % Kernel::kAlignmentK) || (c.v_strides[2] % Kernel::kAlignmentV)) return -1;
  const size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  if (smem_bytes > (size_t)max_shmem) return -1;
  for (const int64_t *s : {c.q_strides, c.k_strides, c.v_strides}) {
    if (!fits_i32(s[1]) || !fits_i32(s[2])) {
      snprintf(err, err_len, "mem-efficient attention: a stride overflows int32");
      return 1;
    }
  }
  typename Kernel::Params p;
  if (Kernel::kNeedsOutputAccumulatorBuffer && c.accum == nullptr) {
    snprintf(err, err_len, "mem-efficient attention: this kernel needs an f32 accumulator");
    return 1;
  }
  p.query_ptr = static_cast<const T *>(c.q);
  p.key_ptr = static_cast<const T *>(c.k);
  p.value_ptr = static_cast<const T *>(c.v);
  p.logsumexp_ptr = nullptr;
  p.output_accum_ptr = Kernel::kNeedsOutputAccumulatorBuffer ? reinterpret_cast<typename Kernel::output_accum_t *>(c.accum) : nullptr;
  p.output_ptr = static_cast<typename Kernel::output_t *>(c.out);
  p.num_heads = c.heads;
  p.head_dim = c.head_dim;
  p.head_dim_value = c.head_dim;
  p.num_queries = c.queries;
  p.num_keys = c.keys;
  p.num_batches = c.batches;
  p.custom_mask_type = 0;
  p.seqlen_k_ptr = nullptr;
  p.scale = c.scale;
  p.q_strideB = c.q_strides[0];
  p.k_strideB = c.k_strides[0];
  p.v_strideB = c.v_strides[0];
  p.q_strideM = (int32_t)c.q_strides[1];
  p.k_strideM = (int32_t)c.k_strides[1];
  p.v_strideM = (int32_t)c.v_strides[1];
  p.q_strideH = (int32_t)c.q_strides[2];
  p.k_strideH = (int32_t)c.k_strides[2];
  p.v_strideH = (int32_t)c.v_strides[2];
  p.o_strideM = c.heads * c.head_dim;
  if (c.bias != nullptr) {
    if (!fits_i32(c.bias_strides[2])) {
      snprintf(err, err_len, "mem-efficient attention: the bias's row stride overflows int32");
      return 1;
    }
    p.attn_bias_ptr = static_cast<const T *>(c.bias);
    p.bias_strideB = c.bias_strides[0];
    p.bias_strideH = c.bias_strides[1];
    p.bias_strideM = (int32_t)c.bias_strides[2];
  }
  p.use_dropout = false;
  if (smem_bytes > 0xc000) {
    cudaError_t e = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (e != cudaSuccess) {
      snprintf(err, err_len, "mem-efficient attention: %s", cudaGetErrorString(e));
      return 1;
    }
  }
  auto blocks = p.getBlocksGrid();
  if (blocks.x * blocks.y * blocks.z == 0 || c.keys == 0) {
    cudaMemsetAsync(c.out, 0, sizeof(T) * (size_t)c.batches * c.queries * c.heads * c.head_dim, c.stream);
    return 0;
  }
  if (!Kernel::check_supported(p)) {
    snprintf(err, err_len, "mem-efficient attention: the kernel's check_supported refused these tensors");
    return 1;
  }
  kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, c.stream>>>(p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    snprintf(err, err_len, "mem-efficient attention: %s", cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

}  // namespace

extern "C" {

// The attention of q (B, M, H, K), k and v (B, N, H, K) in torch's BMHK view (strides in elements:
// batch, sequence, head; the last dimension contiguous), f32 (dtype 0) or bf16 (dtype 1), with an
// additive bias (B, H, M, N) of the same type (strides batch, head, query; a broadcast dimension
// has stride 0) or none, into out (B, M, H, K) contiguous; accum is an f32 (B, M, H, K) scratch for
// the bf16 kernels that accumulate outside their output. What _efficient_attention_forward does for
// a call without dropout, causal mask or log-sum-exp, the kernel chosen as its dispatch chooses it.
// Returns 0, or -1 with the reason.
int rustyui_mem_eff_attention(int dtype, void *stream_, const void *q, const void *k, const void *v, const void *bias, void *out, float *accum,
                              int batches, int heads, int queries, int keys, int head_dim, const int64_t *q_strides, const int64_t *k_strides,
                              const int64_t *v_strides, const int64_t *bias_strides, float scale, char *err, int err_len) {
  Call c{static_cast<cudaStream_t>(stream_), q, k, v, bias, out, accum, batches, heads, queries, keys, head_dim, q_strides, k_strides, v_strides, bias_strides, scale};
  int device = 0, max_shmem = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  int r = -1;
  if (dtype == 0) {
    r = try_kernel<Fmha<float, 64, 64, 64>>(fmha_cutlassF_f32_aligned_64x64_rf_sm80, c, max_shmem, err, err_len);
    if (r < 0) r = try_kernel<Fmha<float, 64, 128, 128>>(fmha_cutlassF_f32_aligned_64x128_rf_sm80, c, max_shmem, err, err_len);
    if (r < 0) r = try_kernel<Fmha<float, 32, 128, 65536>>(fmha_cutlassF_f32_aligned_32x128_gmem_sm80, c, max_shmem, err, err_len);
  } else if (dtype == 1) {
    r = try_kernel<Fmha<cutlass::bfloat16_t, 64, 64, 64>>(fmha_cutlassF_bf16_aligned_64x64_rf_sm80, c, max_shmem, err, err_len);
    if (r < 0) r = try_kernel<Fmha<cutlass::bfloat16_t, 64, 128, 128>>(fmha_cutlassF_bf16_aligned_64x128_rf_sm80, c, max_shmem, err, err_len);
    if (r < 0) r = try_kernel<Fmha<cutlass::bfloat16_t, 32, 128, 65536>>(fmha_cutlassF_bf16_aligned_32x128_gmem_sm80, c, max_shmem, err, err_len);
  } else {
    snprintf(err, err_len, "mem-efficient attention: dtype %d is neither f32 (0) nor bf16 (1)", dtype);
    return -1;
  }
  if (r < 0) {
    snprintf(err, err_len, "mem-efficient attention: no kernel takes head size %d with these strides", head_dim);
    return -1;
  }
  return r == 0 ? 0 : -1;
}

}  // extern "C"
