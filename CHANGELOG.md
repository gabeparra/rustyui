# Changelog

All notable changes to rustyui are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `comfyui-rustyui/`, a ComfyUI node pack: a **Run on rustyui** node sends the workflow to
  rustyui-backend and shows the result in ComfyUI.
- The README's install paths, including ComfyUI's official editor (`comfyui-frontend-package`)
  served by the backend with `--frontend`.

### Fixed

- A prompt sent from inside a running ComfyUI no longer waits for ComfyUI's queue to empty, which
  holds that very prompt.

### Removed

- The copy of ComfyUI's editor (ComfyUI_frontend v1.55.14, `ui/`). The repository no longer
  includes it, and building no longer needs pnpm.

### Changed

- rustyui's own editor is served at `/` (the node graph at `/#graph`); `/simple/` redirects there.
  `--frontend DIR` serves a folder at `/` as before, so point it at `frontend/dist` rather than
  `ui/dist`.
- SDXL is faster with the same output: on an RTX 4090 a warm run takes 87 to 92 percent of ComfyUI's
  time and the sampling steps 93 percent (they took 106 percent before), from faster adds, norms,
  GEGLU and layout copies, no zero-filling of buffers that are overwritten, and the prompt projected
  for cross-attention once a sampler run.
- LTX-2.3: the blocks of the 42 GB DiT that do not fit on a 24 GB card are streamed from page-locked
  memory one block ahead; the video decode and a new prompt let only the blocks they need leave the
  card; the video VAE loads in 0.08 s instead of 1 s. On an RTX 4090 the video example takes 35.5 s
  from process start against ComfyUI's 47.8 s and 15.6 s against 17.2 s for a repeat, with identical
  output.
- LTX-2.3: a sampling that runs out of memory part-way keeps more memory free and starts over
  instead of failing the job.
- `debug-passes` checks the fast kernels bit for bit against the forms they replace.

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
