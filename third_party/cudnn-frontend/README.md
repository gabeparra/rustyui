# cudnn-frontend (vendored)

NVIDIA's header-only C++ API over cuDNN's graph backend, MIT (LICENSE.txt), at the commit PyTorch
2.11.0 pins in its third_party/cudnn_frontend: b8c0656e6f6c84fc194f4d57329b55d609eff596 (v1.18.0).
Only `include/` is kept. `backend/kernels/sdpa.cpp` uses it to build cuDNN's attention graph the way
torch's cuDNN SDPA backend builds it, so a masked attention runs the same cuDNN kernel ComfyUI does.
