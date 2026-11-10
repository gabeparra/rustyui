# The Wan 2.1 VAE (qwen_image_vae.safetensors, Anima's and Krea2's) decoding one picture inside
# ComfyUI, started as the server starts (main.py): a seeded latent through Wan21's process_out, then
# VAELoader and VAEDecode as the server runs them, with every stage's output kept (WanVAE.conv2, the
# decoder's conv1, its middle, each upsample layer, the head conv) and the image, laid out
# (1, 3, 1, H, W) as the decoder gives it. Also which SDPA backend torch picks for the middle
# attention, in ComfyUI's priority. Checked by `rustyui-backend debug-wan-vae VAE OUT`.
#   python wan_vae_ref.py OUT.safetensors [HEIGHT WIDTH [SEED]]
import sys, os

OUT = sys.argv[1]
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1216, 832)
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 1
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401
import torch  # noqa: E402
import nodes  # noqa: E402
import comfy.latent_formats  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

t, info = {}, {}


def keep(name, x):
    t[name] = x.detach().cpu().contiguous().clone()


vae = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")[0]
g = torch.Generator().manual_seed(SEED)
z = comfy.latent_formats.Wan21().process_out(torch.randn((1, 16, 1, H // 8, W // 8), generator=g))
keep("latent", z)
m = vae.first_stage_model
hooks = [(m.conv2, "conv2"), (m.decoder.conv1, "conv1"), (m.decoder.middle[2], "middle"), (m.decoder.head[2], "head")]
hooks += [(u, f"up{i}") for i, u in enumerate(m.decoder.upsamples)]
for mod, name in hooks:
    mod.register_forward_hook(lambda mod, args, out, name=name: keep(name, out))


# the middle attention's q, k, v as SDPA gets them, and the backend torch chooses for them
def attn_hook(mod, args, out):
    x = mod.norm(args[0].reshape(-1, *args[0].shape[1:2], *args[0].shape[3:]))
    q, k, v = mod.to_qkv(x).chunk(3, dim=1)
    B, C = q.shape[0], q.shape[1]
    q, k, v = (a.view(B, 1, C, -1).transpose(2, 3).contiguous() for a in (q, k, v))
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    info["sdpa_flash"] = str(torch.backends.cuda.can_use_flash_attention(params))
    info["sdpa_efficient"] = str(torch.backends.cuda.can_use_efficient_attention(params))
    info["sdpa_cudnn"] = str(torch.backends.cuda.can_use_cudnn_attention(params))
    info["sdpa_choice"] = str(torch._fused_sdp_choice(q, k, v))


m.decoder.middle[1].register_forward_hook(attn_hook)
img = nodes.VAEDecode().decode(vae, {"samples": z})[0]
keep("image", img.permute(0, 3, 1, 2).unsqueeze(2))
info["vae_dtype"] = str(vae.vae_dtype)
info["size"] = f"{H}x{W}"
info["seed"] = str(SEED)
save_file(t, OUT, metadata=info)
print(info)
for k, v in t.items():
    print(k, tuple(v.shape), v.dtype)
