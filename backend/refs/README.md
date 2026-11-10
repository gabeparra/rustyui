# refs: captures from ComfyUI

rustyui aims to give ComfyUI's output exactly, and these scripts are how that is checked. Each one
runs part of a graph inside ComfyUI's own Python (in process, no server needed), saves the tensors
along the way to a safetensors file, and changes nothing else. A `rustyui-backend debug-*` command,
or a node dump from `rustyui-backend run`, is then compared against that file, stage by stage, to
find the first place the two part.

ComfyUI itself is only the reference. rustyui does not need it to run.

## Running a capture

Run the scripts with ComfyUI's Python (its venv), and tell them where ComfyUI is with
`COMFYUI_DIR` (the current directory by default):

```bash
export COMFYUI_DIR=/path/to/ComfyUI
cd "$COMFYUI_DIR" && .venv/bin/python /path/to/rustyui/backend/refs/unet_ref.py /tmp/rustyui_unet_ref.safetensors
rustyui-backend debug-unet /tmp/rustyui_unet_ref.safetensors
```

Inputs and outputs:

- The SDXL scripts take the file to write as their first argument, and read the lighthouse graph
  (the API graph inside `docs/examples/lighthouse.png`) from
  [`fixtures/readme-graph.json`](fixtures/readme-graph.json) unless given another graph file, so no
  prompt text lives in the scripts.
- The LTX-2.3 scripts write `/tmp/rustyui_*.safetensors` and read their prompt from
  `/tmp/rustyui_ltx_prompt.json` (`{"prompt": "..."}`). The image-to-video scripts read
  `/tmp/rustyui_i2v_prompt.json` and load `input/rustyui-i2v-lighthouse.png` from ComfyUI's input
  folder, a copy of `docs/examples/lighthouse.png`. On a system where `/tmp` is cleared at boot,
  write these again after a restart.
- A model is found by name in ComfyUI's own `models/` folder, so both programs read the same files.

A masked attention runs on cuDNN's runtime-compiled kernels, which need NVRTC's builtins on the
loader path. If ComfyUI's torch is the CUDA 13.0 build, that means its venv's
`nvidia/cu13/lib`:

```bash
LD_LIBRARY_PATH="$COMFYUI_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib" .venv/bin/python ...
```

The DiT captures need the GPU for a minute or two at about 20 GB of VRAM.

**Load models the way the server does.** ComfyUI's server starts with DynamicVRAM (comfy-aimdo's
`ModelPatcherDynamic`, on by default on NVIDIA). With LoRAs that is not the arithmetic of the legacy
`ModelPatcher` a script gets by loading through `comfy.sd` or `nodes` alone: the LoRAs go into each
layer in bf16 when it is first cast, and fp8 layers are quantized again. The scripts that run a
graph (`graph_nodes_ref.py`, `lora_dynamic_ref.py`, and the others that say "started as the server
starts") import ComfyUI's `main.py` first, so their render of a graph is the server's to the byte.
The older scripts load the legacy way, which gives the same numbers for anything without LoRAs.

**Capture against the ComfyUI you compare with.** Under torch's CUDA 13.0 build ComfyUI runs
comfy_kitchen's CUDA kernels instead of their eager versions, and they round differently. A capture
made under another torch, cuDNN or ComfyUI version may no longer match: capture again rather than
compare against an old file.

## Node packs that are not installed

Some nodes rustyui runs come from packs a reference ComfyUI may not have (comfyui_controlnet_aux,
ComfyUI-RMBG, ComfyUI-Impact-Pack). Those scripts load the pack's own code from a copy instead:

- `graph_nodes_ref.py` and `object_info_ref.py` take `CAPTURE_EXTRA_NODES=DIR` (a pack to load as a
  custom node) and `CAPTURE_STUBS=DIR` (stand-ins for imports the venv cannot satisfy).
- `depth_anything_ref.py`, `sam3_ref.py` and `impact_segs_ref.py` read their pack from
  `~/.cache/rustyui/controlnet_aux_ref`, `~/.cache/rustyui/rmbg_ref` and
  `~/.cache/rustyui/impact_ref`.

## SDXL

| Script | Writes | Checked by |
|---|---|---|
| `clip_ref.py OUT` | CLIP-G and CLIP-L for the lighthouse prompt: layer 0 op by op, the attention's inputs and SDPA backend, the final norm, the projection, CLIPTextEncode's outputs | `debug-clip` |
| `unet_ref.py OUT` | the UNet on the graph's first call: its inputs, every block's output, the first ResBlock's and transformer block's ops, what SDPA receives and the backend it picks | `debug-unet` |
| `sdxl_sample_ref.py OUT [GRAPH]` | the KSampler step by step: the sigmas, each step's x and CFG-combined denoised, the final latent | `RUSTYUI_DUMP_SAMPLER=FILE rustyui-backend run GRAPH` |
| `sdxl_vae_ref.py OUT [GRAPH]` | the VAE decode: every block's input and output, the first middle res block and the attention op by op, the SDPA backend, the image | `debug-sdxl-vae` |
| `esrgan_ref.py NODES KEY OUT [MODEL [ROW COL]]` | ImageUpscaleWithModel on a node's IMAGE from a `graph_nodes_ref.py` dump: each tile's input and output, the node's result, and every stage of the model around output pixel (ROW, COL). `RUSTYUI_ESRGAN_NHWC=1` holds the picture as channels-last memory, as LoadImage leaves it | `debug-esrgan MODEL.pth OUT` |
| `order_ref.py GRAPH` | prints the order ComfyUI runs the graph's nodes in (with LoRAs on the text encoders, the first encode after loading differs from later ones, so order is part of the result) | `comfy_order` in `graph.rs` |
| `depth_anything_ref.py PICTURE CKPT RES OUT` | comfyui_controlnet_aux's DepthAnythingV2Preprocessor: the resize, DINOv2's stages, the DPT head, the map | `debug-depth` |
| `ipp_resize_ref.py OUT` | `cv2.resize(INTER_CUBIC)` on uint8 pictures (1, 3 and 4 channels, enlarged and shrunk), which OpenCV hands to Intel IPP | `debug-ipp-resize` |
| `sam3_ref.py PICTURE PROMPT OUT [MODE CONF MAX PICK BLUR]` | ComfyUI-RMBG's SAM3Segment: every stage from the transform to the node's mask | `debug-sam3` |
| `impact_blur_ref.py MASK OUT` | GrowMask and Impact's mask blur on a test mask | `debug-impact-blur` |
| `impact_segs_ref.py MASK SEGS_OUT BLUR_OUT` | ComfyUI-Impact-Pack's own `mask_to_segs` and `tensor_gaussian_blur_mask`: segments of drawn and grown masks, blurs of small masks and batches | `debug-impact-segs`, `debug-impact-blur` |
| `conv_ref.py` | torch's cuDNN Conv3d and Conv2d (bf16, fp16, f32) on seeded inputs, with cuDNN set up as ComfyUI sets it up | `debug-conv` |
| `gn_ref.py` | torch's GroupNorm, and SiLU after it, in bf16, fp16 and f32 | `debug-group-norm` |

## LTX-2.3

| Script | Writes | Checked by |
|---|---|---|
| `ltx_te_dump.py` | Gemma 3's hidden states through ComfyUI's LTX-AV text encoder | `debug-ltx-te` |
| `gemma_layer_ref.py` | Gemma 3 inside: both RoPE tables, every op of layers 0 and 5 on the padded sequence, the attention's inputs and SDPA backend, all 49 hidden states | `debug-gemma-layer` |
| `dequant_ref.py` | dequantized NVFP4 and fp8 weights | `debug-dequant` |
| `proj_ref.py` | the text projection's GEMM inputs and outputs | `debug-ltx-proj` |
| `fp8_ref.py` | six real fp8 linears on fixed inputs | `debug-fp8` |
| `dit_ref2.py` | one DiT call: connector, timesteps, blocks 0 and 2 piece by piece | `debug-ltx-block`, `debug-ltx-io`, `debug-ltx-connector`, `debug-ltx-forward` |
| `dit_batch_ref.py GRAPH OUT` | the server's first DiT call for a graph, with CFG above 1 batching the negative and the prompt: its arguments, timesteps, every block's output, the result | `debug-ltx-batch` |
| `sample_ref.py` | a whole 8-step euler run, then the video VAE decode | `debug-ltx-sample` |
| `i2v_ref.py` | image to video: the start picture through LTXVImgToVideoInplace (640x448x17), the sampler's latent and mask, the first call's per-token timesteps, all 8 steps | `debug-ltx-i2v` |
| `i2v_anc_ref.py` | the same with ManualSigmas and euler_ancestral (3 steps) | `debug-ltx-i2v --ancestral` |
| `guide_ref.py [STRENGTH]` | the picture as an LTXVAddGuide guide (default 0.7), then LTXVCropGuides: keyframe indices, block 0's inputs, its guide-masked attention, all 8 steps | `debug-ltx-guide` |
| `guide_attn_ref.py` | ComfyUI's guide-masked attention alone on seeded q, k, v (three sizes) | `debug-guide-attn` |
| `lora_dynamic_ref.py GRAPH OUT` | the weights a graph's LoRA'd DiT layers multiply by under DynamicVRAM, on their first two casts | `debug-ltx-lora` (LoRAs in the graph's order) |
| `preprocess_ref.py` | LTXVPreprocess's H.264 round trip on the lighthouse picture and an odd-sized crop, CRF 18 and 35 | `debug-preprocess` |
| `upsampler_ref.py` | LTXVLatentUpsampler with the spatial and temporal x2 upscalers, every stage | `debug-upsampler` |
| `vae_enc_ref.py` | the video VAE encoder on one and nine frames | `debug-ltx-vae-enc` |
| `vae_stream_ref.py` | the video VAE where it streams: a 49-frame 896x1280 decode and its encode, with the chunk budget ComfyUI computes on the GPU | `debug-ltx-vae-stream` |
| `audio_ref.py` | LTXVAudioVAEDecode on a seeded latent, every stage to the 48 kHz waveform | `debug-audio` |
| `rife_ref.py NODES KEY FRAMES OUT` | RIFE VFI (rife49, multiplier 2) on the first frames of a node dump's video: the node's input and output, and every stage of its first IFNet call | `debug-rife` |

## Anima

| Script | Writes | Checked by |
|---|---|---|
| `anima_tokens_ref.py OUT.json [GRAPH...]` | Anima's prompt tokens (Qwen3's ids, T5's ids and weights) for a set of awkward prompts and each CLIPTextEncode text of the graphs, CPU only. `anima_tokens_ref.py assets backend/assets/anima` writes the tokenizer data rustyui ships | `debug-anima-tokens OUT.json` |
| `wan_vae_ref.py OUT [H W [SEED]]` | the Wan 2.1 VAE decoding a seeded one-frame latent through VAELoader and VAEDecode, stage by stage | `debug-wan-vae VAE OUT` |

## Whole graphs

| Script | What |
|---|---|
| `graph_nodes_ref.py GRAPH[,GRAPH2...] OUT [server\|legacy]` | a whole API graph, every node's output tensors. Several graphs run in one process, as a server's prompts do (the last one's tensors kept). `CAPTURE_ORDER=1` prints the nodes as they finish, `CAPTURE_CALLS=1` each model call's batch. Compare with `RUSTYUI_DUMP_NODES=FILE rustyui-backend run GRAPH`, then `cmp_nodes.py OUT FILE` |
| `cmp_nodes.py A B` | lines up two node dumps in ComfyUI's order and reports where they first differ |
| `object_info_ref.py OUT CLASS...` | the definitions `/object_info` gives those nodes, for `backend/assets/comfy_object_info.json` |
| `cmp_video.py A.mp4 B.mp4` | two renders compared frame by frame, sound sample by sample, and packet by packet |
| `cmp_schedule.py [BACKEND] [--powf]` | rustyui's sigma schedules against `calculate_sigmas` at every step count (karras and simple 1 to 300, normal and sgm_uniform 1 to 80) through `debug-schedule`; with `--powf`, Sleef's `powf` against `torch.pow` on three million inputs through `debug-powf` |
| `regress.py [CASE...]` | every SDXL fixture in `~/.cache/rustyui/fixtures` through rustyui, against ComfyUI's capture of it, pixel for pixel. Run in ComfyUI's venv; `RUSTYUI_BIN` names the build to test. The fixtures are not part of the repository |
| `post.py URL GRAPH PREFIX` | a deploy test: one graph through a running server, its outputs under `PREFIX` (which must start with `rustyui-tests/`), then compare with `rustyui-backend compare` |

## Other checks

These `rustyui-backend` commands print or write values for comparison with PyTorch, Pillow or
scipy directly, or reuse a capture above:

| Command | Checks |
|---|---|
| `debug-randn [--seed N] [--n 16] [--cuda --sms N]` | PyTorch's CPU `randn`, or its CUDA `randn` for a GPU with that many SMs |
| `debug-schedule SCHEDULER STEPS...`, `debug-powf TRIPLES.bin` | the sigma schedules and Sleef's `powf` (driven by `cmp_schedule.py`) |
| `debug-beta-schedule MAX_STEPS` | the beta schedule's table indices, against scipy's `beta.ppf` |
| `debug-lanczos IN.png W H OUT.png` | the lanczos resize, to compare with Pillow's |
| `debug-gemma-tokens TE.safetensors PROMPT.json [REF]` | Gemma 3's tokenizer, as ComfyUI tokenizes |
| `debug-ltx-attn REF2` | FlashAttention on each attention call's inputs from `dit_ref2.py`, against torch's SDPA |
| `debug-conv-nhwc REF STRIDE PADDING` | one channels-last convolution (`x`, `w`, `y` saved from torch) against torch's |
