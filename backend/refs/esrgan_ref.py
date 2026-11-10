# ImageUpscaleWithModel's ESRGAN inside ComfyUI, tile by tile, for making rustyui's bit for bit: the
# picture it upscales (a node's IMAGE output from a graph_nodes_ref.py dump), every tile's input
# and the model's output for it as tiled_scale cuts them (each run twice, to see whether torch's own
# result repeats), the node's result, and, in the tiles that hold output pixel (ROW, COL), each of
# the model's top-level stages in a window of 33x33 around that pixel at the stage's own size (the
# window's place is in the metadata). Started as the server starts (main.py).
#   python esrgan_ref.py NODES.safetensors IMAGE_KEY OUT.safetensors [MODEL [ROW COL]]
import sys, os, json, itertools

NODES, KEY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "4x_foolhardy_Remacri.pth"
ROW, COL = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (-1, -1)
HALF = 16
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

t, info = {}, {"model": MODEL, "image_key": KEY, "pixel": [ROW, COL]}


def keep(name, x):
    t[name] = x.detach().float().cpu().contiguous().clone()


image = load_file(NODES)[KEY]
keep("image", image)
# the memory the IMAGE had in the graph decides torch's convolution layout: VAEDecode's output is
# NCHW memory seen as NHWC, so movedim hands the model a contiguous NCHW tensor; a LoadImage
# picture is NHWC memory, and torch then runs every convolution channels-last (RUSTYUI_ESRGAN_NHWC=1)
if os.environ.get("RUSTYUI_ESRGAN_NHWC") != "1":
    image = image.movedim(-1, 1).contiguous().movedim(1, -1)
info["image_memory"] = "nhwc" if image.is_contiguous() else "nchw"
up = UpscaleModelLoader.execute(MODEL)[0]
keep("result", ImageUpscaleWithModel.execute(up, image)[0])

# the same tiles again, by hand, as comfy.utils.tiled_scale cuts them (tile 512, overlap 32)
mm.load_models_gpu([up.patcher], force_full_load=True)
s = image.movedim(-1, -3).to(up.patcher.load_device)
tile, overlap, scale = 512, 32, up.scale
positions = [range(0, s.shape[d + 2] - overlap, tile - overlap) if s.shape[d + 2] > tile else [0] for d in range(2)]
names = dict(up.model.named_modules())
stages = sorted((n for n in names if n.count(".") == 1 and n.startswith("model.")), key=lambda n: int(n.split(".")[1]))
info["stages"] = stages
tiles, unstable = [], 0
for i, it in enumerate(itertools.product(*positions)):
    pos = [max(0, min(s.shape[d + 2] - overlap, it[d])) for d in range(2)]
    ln = [min(tile, s.shape[d + 2] - pos[d]) for d in range(2)]
    s_in = s.narrow(2, pos[0], ln[0]).narrow(3, pos[1], ln[1])
    local = (ROW - pos[0] * scale, COL - pos[1] * scale)
    covers = 0 <= local[0] < ln[0] * scale and 0 <= local[1] < ln[1] * scale
    windows, handles = {}, []
    if covers:
        for n in stages:
            def hook(mod, args, out, n=n, i=i):
                f = out.shape[2] / ln[0]  # this stage's size against the tile's
                r, c = int(local[0] * f / scale), int(local[1] * f / scale)
                r0, c0 = max(0, r - HALF), max(0, c - HALF)
                h, w = min(out.shape[2], r + HALF + 1) - r0, min(out.shape[3], c + HALF + 1) - c0
                windows[n] = [r0, c0, h, w]
                keep(f"tile{i}.{n}", out[:, :, r0:r0 + h, c0:c0 + w])
            handles.append(names[n].register_forward_hook(hook))
    with torch.no_grad():
        a = up(s_in.float())
    for h in handles:
        h.remove()
    with torch.no_grad():
        b = up(s_in.float())
    same = torch.equal(a, b)
    unstable += not same
    keep(f"tile{i}.in", s_in)
    keep(f"tile{i}.out", a)
    tiles.append({"pos": pos, "size": ln, "repeats": same, "windows": windows})
info["tiles"] = tiles
info["unstable_tiles"] = unstable
info["cudnn"] = {"benchmark": torch.backends.cudnn.benchmark, "allow_tf32": torch.backends.cudnn.allow_tf32,
                 "deterministic": torch.backends.cudnn.deterministic, "version": torch.backends.cudnn.version()}
print(json.dumps(info))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
