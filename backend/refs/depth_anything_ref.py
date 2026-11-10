# Depth Anything V2 as comfyui_controlnet_aux's DepthAnythingV2Preprocessor runs it, for making
# rustyui's bit for bit. The node pack is not installed in the reference ComfyUI, so its own package is
# imported from a copy (~/.cache/rustyui/controlnet_aux_ref/custom_controlnet_aux: __init__.py,
# util.py and depth_anything_v2/ from the pack's src/) and the checkpoint is read by path. Records
# the node's steps for one picture: the uint8 picture, the network input, DINOv2's patch embedding,
# first block and four intermediate layers, the DPT head's stages, the network's depth, the depth at
# the picture's size, the uint8 map before its resize, and the node's IMAGE.
#   python depth_anything_ref.py PICTURE CKPT RESOLUTION OUT.safetensors
import os
import sys

PIC, CKPT, RES, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, os.path.expanduser("~/.cache/rustyui/controlnet_aux_ref"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import custom_controlnet_aux.depth_anything_v2 as da2  # noqa: E402
from custom_controlnet_aux.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

t = {}


def keep(name, v):
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(np.ascontiguousarray(v))
    if isinstance(v, torch.Tensor):
        t[name] = v.detach().cpu().contiguous().clone()


name = os.path.basename(CKPT)
model = DepthAnythingV2(**da2.model_configs[name])
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
det = da2.DepthAnythingV2Detector(model.eval(), name).to("cuda")
m = det.model

_image2tensor = m.image2tensor


def image2tensor(raw, input_size=518):
    keep("bgr", raw)
    x, hw = _image2tensor(raw, input_size)
    keep("net_in", x)
    return x, hw


m.image2tensor = image2tensor

_infer = m.infer_image


def infer_image(raw, input_size=518, max_depth=20.0):
    d = _infer(raw, input_size=input_size, max_depth=max_depth)
    keep("depth_hw", d)
    return d


m.infer_image = infer_image

_forward = m.forward


def forward(x, max_depth):
    d = _forward(x, max_depth)
    keep("depth_net", d)
    return d


m.forward = forward

_gil = m.pretrained.get_intermediate_layers


def gil(*a, **k):
    out = _gil(*a, **k)
    for i, (f, c) in enumerate(out):
        keep(f"feat{i}", f)
        keep(f"cls{i}", c)
    return out


m.pretrained.get_intermediate_layers = gil


def grab(name):
    def f(mod, args, out):
        keep(name, out)
    return f


m.pretrained.patch_embed.register_forward_hook(grab("patch_embed"))
m.pretrained.blocks[0].register_forward_hook(grab("block0"))
for i, r in enumerate(m.depth_head.resize_layers):
    r.register_forward_hook(grab(f"resize{i}"))
for i, r in enumerate(m.depth_head.projects):
    r.register_forward_hook(grab(f"proj{i}"))
for i in range(1, 5):
    getattr(m.depth_head.scratch, f"refinenet{i}").register_forward_hook(grab(f"refine{i}"))
m.depth_head.scratch.output_conv1.register_forward_hook(grab("output_conv1"))

_resize_pad = da2.resize_image_with_pad


def resize_image_with_pad(img, *a, **k):
    keep("u8_map", img)
    return _resize_pad(img, *a, **k)


da2.resize_image_with_pad = resize_image_with_pad

# the node: LoadImage's IMAGE, then common_annotator_call(model, image, resolution=RES, max_depth=1)
img = torch.from_numpy(np.array(Image.open(PIC).convert("RGB")).astype(np.float32) / 255.0)[None]
keep("image", img)
np_image = np.asarray(img[0].cpu() * 255., dtype=np.uint8)
keep("u8", np_image)
res = det(np_image, output_type="np", detect_resolution=RES, max_depth=1)
keep("result_u8", res)
keep("node_out", torch.from_numpy(res.astype(np.float32) / 255.0))
save_file(t, OUT)
print("saved", OUT, {k: tuple(v.shape) for k, v in t.items()})
