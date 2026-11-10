# Examples

ComfyUI API graphs (the format of ComfyUI's "Save (API)"). Each one runs on rustyui and on ComfyUI
alike. Open one in the editor, or queue it directly:

```bash
curl -s http://127.0.0.1:8190/prompt -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat sdxl_txt2img_api.json)}"
```

| File | What it does |
|---|---|
| `sdxl_txt2img_api.json` | SDXL text to image, 1024x1024, dpmpp_2m karras |
| `sdxl_lora_api.json` | the same with a LoRA and CLIP skip 2 |
| `sdxl_hires_fix_api.json` | 832x1216, then a 4x ESRGAN upscale, a lanczos resize to 1.5x and a 0.4 denoise second pass |

Change `ckpt_name`, `lora_name` and `model_name` to files in your models folder.
