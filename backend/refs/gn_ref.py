# torch's CUDA GroupNorm (F.group_norm, then F.silu) on seeded inputs, for rustyui's group_norm:
# bf16, fp16 and f32, groups of fewer and more than 512 values, the latent upsampler's and SDXL's
# shapes. Read-only apart from the output file.
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUT = "/tmp/rustyui_gn_ref.safetensors"
g = torch.Generator(device="cuda").manual_seed(11)
cases = [
    ("up_bf16", torch.bfloat16, (1, 1024, 5, 14, 20), 32, 1e-5),
    ("small_bf16", torch.bfloat16, (1, 64, 2, 3, 4), 32, 1e-5),
    ("sdxl_f16", torch.float16, (1, 320, 104, 152), 32, 1e-5),
    ("vae_f32", torch.float32, (1, 512, 64, 64), 32, 1e-6),
    ("small_f32", torch.float32, (2, 128, 3, 5), 32, 1e-6),
]
t = {}
for name, dt, shape, groups, eps in cases:
    x = (torch.randn(shape, device="cuda", generator=g) * 3 + 0.5).to(dt)
    w = (1 + 0.2 * torch.randn(shape[1], device="cuda", generator=g)).to(dt)
    b = (0.1 * torch.randn(shape[1], device="cuda", generator=g)).to(dt)
    with torch.inference_mode():
        y = F.group_norm(x, groups, w, b, eps)
        ys = F.silu(y)
    t.update({f"{name}.x": x, f"{name}.w": w, f"{name}.b": b, f"{name}.y": y, f"{name}.ys": ys})
    print(name, dt, shape)
save_file({k: v.cpu().contiguous() for k, v in t.items()}, OUT)
print("saved", OUT, torch.__version__)
