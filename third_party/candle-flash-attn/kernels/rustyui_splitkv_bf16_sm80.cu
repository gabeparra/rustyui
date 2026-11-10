// rustyui: the split-KV kernels torch's SDPA runs when its heuristic splits the keys (few
// queries, as with LTX-2.3's audio tokens), with upstream's tile sizes (run_mha_fwd_splitkv_dispatch),
// not this fork's paged ones.

#include "flash_fwd_launch_template.h"

void rustyui_splitkv_bf16_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, false>(params, stream);
}

void rustyui_splitkv_bf16_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, false>(params, stream);
}

void rustyui_splitkv_bf16_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(params, stream);
}
