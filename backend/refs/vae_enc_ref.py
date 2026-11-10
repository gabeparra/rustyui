# ComfyUI's LTX-2.3 video VAE encode of known pixels: one frame (a start image) and nine (1 + 8),
# for rustyui's encoder (debug-ltx-vae-enc).
import os, sys, torch
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.sd, comfy.utils
from safetensors.torch import load_file, save_file
sd, meta = comfy.utils.load_torch_file("models/vae/ltx-2.3-video-vae.safetensors", return_metadata=True)
vae = comfy.sd.VAE(sd=sd, metadata=meta)
frames = load_file("/tmp/rustyui_sample_ref.safetensors")["vae.decode_out"][0]
one, nine = frames[:1].clone(), frames.clone()
out = {"one.image": one, "one.latent": vae.encode(one).float().contiguous(), "nine.image": nine, "nine.latent": vae.encode(nine).float().contiguous()}
for k, v in out.items(): print(k, tuple(v.shape), v.dtype)
save_file(out, "/tmp/rustyui_vae_enc_ref.safetensors")
