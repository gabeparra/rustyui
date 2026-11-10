# torch-mem-eff

PyTorch v2.11.0's memory-efficient attention forward (`aten/src/ATen/native/transformers/cuda/mem_eff_attention`,
xformers' CUTLASS FMHA as PyTorch carries it), headers only, unchanged, under their original include
paths, with PyTorch's LICENSE (BSD-3). Two shims stand in for the ATen headers they include:
`ATen/cuda/PhiloxUtils.cuh` (the Philox state type, copied from ATen; dropout is never used) and
`c10/util/Exception.h` (TORCH_CHECK as a `return false` in `check_supported`).

`backend/kernels/mem_eff.cu` instantiates the one kernel ComfyUI's f32 masked attention runs on
(`fmha_cutlassF_f32_aligned_32x128_gmem_sm80`) and launches it as `_efficient_attention_forward`
does. It builds against CUTLASS at PyTorch v2.11.0's pinned commit (`../cutlass`).
