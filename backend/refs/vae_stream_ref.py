# ComfyUI's LTX video VAE as it streams: a decode of a seeded (1, 128, 7, 40, 28) latent (49 frames
# at 896x1280, so the up blocks split into time chunks) and an encode of 49 frames at that size (time
# chunks of 8), through ComfyUI's own VAE object (VAELoader), with the chunk budget it computes on
# this card. For rustyui's streaming port. Read-only apart from the output file.
import os, sys
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import asyncio
import torch
from safetensors.torch import save_file
import nodes
import comfy.ldm.lightricks.vae.causal_video_autoencoder as cva

OUT = "/tmp/rustyui_vae_stream_ref.safetensors"
t = {}


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("ltx-2.3-video-vae.safetensors")[0]
    dev = torch.device("cuda")
    t["max_chunk_size"] = torch.tensor([cva.get_max_chunk_size(dev)], dtype=torch.int64)
    g = torch.Generator().manual_seed(21)
    latent = torch.randn(1, 128, 7, 40, 28, generator=g)
    t["latent"] = latent
    with torch.inference_mode():
        frames = vae.decode(latent)
    t["decoded"] = frames.float().cpu().contiguous()
    print("decoded", tuple(frames.shape), frames.dtype, "chunk", int(t["max_chunk_size"][0]))
    # the nodes encode an IMAGE batch, (frames, H, W, 3)
    pixels = frames[0] if frames.dim() == 5 else frames
    with torch.inference_mode():
        enc = vae.encode(pixels)
    t["encode_in"] = pixels.float().cpu().contiguous()
    t["encoded"] = enc.float().cpu().contiguous()
    print("encoded", tuple(enc.shape), enc.dtype)


asyncio.run(main())
save_file(t, OUT)
print("saved", OUT, {k: tuple(v.shape) for k, v in t.items()})
