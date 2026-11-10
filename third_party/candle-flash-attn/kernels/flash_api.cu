#include <cstdio>
#include <cstdlib>
#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"

void rustyui_splitkv_bf16_hdim32(Flash_fwd_params &params, cudaStream_t stream);
void rustyui_splitkv_bf16_hdim64(Flash_fwd_params &params, cudaStream_t stream);
void rustyui_splitkv_bf16_hdim128(Flash_fwd_params &params, cudaStream_t stream);

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  // rustyui trims the build to the instantiations it runs (SDXL: fp16 head dim 64; LTX-2.3:
  // bf16 head dims 64 and 128, split-KV when torch's heuristic splits), all non-causal; anything
  // else is refused loudly.
  if (!params.is_causal && params.block_table == nullptr && params.num_splits > 1 && params.is_bf16) {
      if (params.d <= 32) return rustyui_splitkv_bf16_hdim32(params, stream);
      if (params.d <= 64) return rustyui_splitkv_bf16_hdim64(params, stream);
      if (params.d == 128) return rustyui_splitkv_bf16_hdim128(params, stream);
  }
  if (!params.is_causal && params.block_table == nullptr && params.num_splits <= 1) {
      if (!params.is_bf16 && params.d <= 64) return run_mha_fwd_<cutlass::half_t, 64, false>(params, stream);
      if (params.is_bf16 && params.d <= 32) return run_mha_fwd_<cutlass::bfloat16_t, 32, false>(params, stream);
      if (params.is_bf16 && params.d <= 64) return run_mha_fwd_<cutlass::bfloat16_t, 64, false>(params, stream);
      if (params.is_bf16 && params.d == 128) return run_mha_fwd_<cutlass::bfloat16_t, 128, false>(params, stream);
  }
  fprintf(stderr, "flash attention: not built for d=%d bf16=%d causal=%d splits=%d (fp16 d<=64, bf16 d<=64 or 128, non-causal)\n",
          (int)params.d, (int)params.is_bf16, (int)params.is_causal, (int)params.num_splits);
  abort();
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,
    uint32_t total_q,

    int is_bf16,
    int is_causal,
    int unpadded_lse,

    int window_size_left,
    int window_size_right,

    float softcap,

    int32_t *block_table_ptr,
    uint32_t block_table_batch_stride,
    int page_block_size,

    int32_t *mm_prefix_ranges_ptr,
    uint32_t mm_prefix_range_batch_stride,
    int max_mm_prefix_ranges,
    int num_splits,
    void *softmax_lseaccum_ptr,
    void *oaccum_ptr,
    void *stream_ptr
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.total_q = total_q;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_k = nullptr;
    params.block_table = block_table_ptr;
    params.block_table_batch_stride = block_table_batch_stride;
    params.page_block_size = page_block_size;
    params.mm_prefix_ranges = mm_prefix_ranges_ptr;
    params.mm_prefix_range_batch_stride = mm_prefix_range_batch_stride;
    params.max_mm_prefix_ranges = max_mm_prefix_ranges;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.num_splits = num_splits;
    params.softmax_lseaccum_ptr = softmax_lseaccum_ptr;
    params.oaccum_ptr = oaccum_ptr;
    params.unpadded_lse = unpadded_lse;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    run_mha_fwd(params, stream);
}
