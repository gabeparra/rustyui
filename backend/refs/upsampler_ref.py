# ComfyUI's LTXVLatentUpsampler with the two latent upscalers of a typical image-to-video graph (spatial x2 1.1, temporal x2
# 1.0) on a seeded 5x14x20 latent, loaded by ComfyUI's own nodes, every stage of the model kept, for
# rustyui's port. Read-only apart from the output file.
import os, sys
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import asyncio
import torch
from safetensors.torch import save_file
import nodes
import comfy.model_management as mm

OUT = "/tmp/rustyui_upsampler_ref.safetensors"
t = {}


def keep(name, x):
    t[name] = x.detach().cpu().contiguous().clone()


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("ltx-2.3-video-vae.safetensors")[0]
    loader = nodes.NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]
    up = nodes.NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]
    g = torch.Generator().manual_seed(5)
    latent = torch.randn(1, 128, 5, 14, 20, generator=g)
    keep("latent", latent)
    for tag, name in [("spatial", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), ("temporal", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors")]:
        model = loader.execute(name)[0]
        net = model.model
        hooks = []
        def hook(stage):
            def f(mod, args, out):
                if f"{tag}.{stage}" not in t:
                    keep(f"{tag}.{stage}", out)
            return f
        hooks.append(net.initial_conv.register_forward_hook(hook("initial_conv")))
        hooks.append(net.initial_norm.register_forward_hook(hook("initial_norm")))
        hooks.append(net.res_blocks[0].register_forward_hook(hook("res0")))
        hooks.append(net.res_blocks[-1].register_forward_hook(hook("res_last")))
        hooks.append(net.upsampler.register_forward_hook(hook("upsampler")))
        hooks.append(net.post_upsample_res_blocks[-1].register_forward_hook(hook("post_last")))
        hooks.append(net.final_conv.register_forward_hook(hook("final_conv")))
        def pre(mod, args):
            if f"{tag}.model_in" not in t:
                keep(f"{tag}.model_in", args[0])
        hooks.append(net.register_forward_pre_hook(pre))
        with torch.inference_mode():
            out = up.execute({"samples": latent.clone(), "noise_mask": torch.ones(1, 1, 5, 1, 1)}, model, vae)[0]
        for h in hooks:
            h.remove()
        keep(f"{tag}.out", out["samples"])
        print(tag, tuple(out["samples"].shape), out["samples"].dtype, "noise_mask" in out, "dtype", net.initial_conv.weight.dtype)
    stats = vae.first_stage_model.per_channel_statistics
    keep("std", stats.get_buffer("std-of-means"))
    keep("mean", stats.get_buffer("mean-of-means"))


asyncio.run(main())
save_file(t, OUT)
print("saved", OUT, sorted(t))
