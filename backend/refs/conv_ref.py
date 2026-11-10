# torch's cuDNN convolutions as ComfyUI runs them (benchmark off, TF32 allowed, not deterministic)
# on seeded inputs, for rustyui's cudnn_conv: the LTX latent upsampler's Conv3d and Conv2d shapes and
# an SDXL-sized fp16 Conv2d. Read-only apart from the output file.
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
OUT = "/tmp/rustyui_conv_ref.safetensors"
g = torch.Generator(device="cuda").manual_seed(7)
cases = [
    ("up3d_in", torch.bfloat16, (1, 128, 5, 14, 20), (1024, 128, 3, 3, 3)),
    ("up3d_mid", torch.bfloat16, (1, 1024, 5, 14, 20), (1024, 1024, 3, 3, 3)),
    ("up2d", torch.bfloat16, (5, 1024, 14, 20), (4096, 1024, 3, 3)),
    ("tmp3d", torch.bfloat16, (1, 512, 5, 14, 20), (1024, 512, 3, 3, 3)),
    ("sdxl2d", torch.float16, (1, 320, 104, 152), (320, 320, 3, 3)),
    ("vae3d_f32", torch.float32, (1, 128, 3, 16, 16), (128, 128, 3, 3, 3)),
]
t = {}
for name, dt, xs, ws in cases:
    x = torch.randn(xs, device="cuda", generator=g).to(dt)
    w = (torch.randn(ws, device="cuda", generator=g) * (1.0 / (ws[1] * 9) ** 0.5)).to(dt)
    b = (torch.randn(ws[0], device="cuda", generator=g) * 0.1).to(dt)
    conv = F.conv3d if len(xs) == 5 else F.conv2d
    with torch.inference_mode():
        y = conv(x, w, None, 1, 1)
        yb = conv(x, w, b, 1, 1)
    torch.cuda.synchronize()
    t.update({f"{name}.x": x, f"{name}.w": w, f"{name}.b": b, f"{name}.y": y, f"{name}.yb": yb})
    print(name, dt, tuple(y.shape))
save_file({k: v.cpu().contiguous() for k, v in t.items()}, OUT)
print("saved", OUT, torch.__version__, torch.backends.cudnn.version())
