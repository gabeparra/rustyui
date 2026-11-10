# Changelog

All notable changes to rustyui are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### In progress

- LTX-2.3: CFG above 1 together with image guides.
- Anima (DiT, Qwen3-0.6B text encoder, Wan 2.1 VAE): implemented, not yet checked against ComfyUI.

## [0.1.0]

First release.

### Added

- `rustyui-backend`: runs ComfyUI API graphs in Rust on candle and answers ComfyUI's HTTP API and
  websocket (`/prompt`, `/queue`, `/history`, `/jobs`, `/object_info`, `/view`, `/upload/image`,
  `/interrupt`, `/free`, `/ws` and the user-data routes ComfyUI's editor uses), also under `/api/`.
- ComfyUI's editor (ComfyUI_frontend v1.55.14) served at `/`, and a simple Forge-style editor with a
  node-graph view at `/simple/`, both compiled into the binary.
- SDXL: text to image, img2img, inpainting, the hires fix, LoRAs, textual-inversion embeddings,
  regional prompts, per-prompt LoRA hooks and ControlNet, matching ComfyUI's output bit for bit on
  the reference setup.
- Samplers `euler`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `dpmpp_2m`, `dpmpp_2m_sde`,
  `dpmpp_2m_sde_gpu` and `er_sde`; schedulers `simple`, `normal`, `karras`, `sgm_uniform` and `beta`.
- LTX-2.3 text to video and image to video with sound, through ComfyUI's LTX nodes: guides,
  latent upscalers, LoRAs, the video and audio VAEs, and MP4 output identical to ComfyUI's.
- Nodes ported from KJNodes, ComfyUI-Frame-Interpolation (RIFE), comfyui_controlnet_aux (Depth
  Anything V2, Canny), ComfyUI-RMBG (SAM 3), ComfyUI-Impact-Pack (detailer) and
  ComfyUI-TiledAttentionCouple.
- PyTorch's CPU and CUDA `randn`, torchsde's Brownian tree, and torch's CPU math (through Intel
  MKL) reproduced, so seeds and sigma schedules match ComfyUI's.
- FlashAttention-2, cuDNN graphs, cuBLASLt, torch's memory-efficient attention, comfy_kitchen's
  kernels and fused kernels written to PyTorch's arithmetic.
- PNG and MP4 outputs that carry their graph, as ComfyUI writes them.
- Loopback-only serving with Host, Origin and Sec-Fetch-Site checks.
- systemd socket activation, idle model release and idle exit.
- Optional prompt writer (`--writer`) on a GGUF Gemma 3 instruct model, and optional mirroring of
  renders to another machine (`--mirror`).
- A pinned development container (`dev/`) and the capture scripts that check rustyui against
  ComfyUI (`backend/refs/`).
