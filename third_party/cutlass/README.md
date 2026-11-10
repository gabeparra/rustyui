# cutlass (pruned)

NVIDIA CUTLASS at commit 0d2b201e8c1c4a03efa6e9c468161916e2334725, the one PyTorch v2.11.0 pins as
third_party/cutlass, cut down to the headers `backend/kernels/mem_eff.cu` includes (the list
`nvcc -M` gives for it), unchanged, with CUTLASS's LICENSE.txt (BSD-3). The commit matters: the
attention kernel built against it has to round as torch's does.
