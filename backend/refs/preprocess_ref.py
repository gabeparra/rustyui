# ComfyUI's LTXVPreprocess (comfy_extras/nodes_lt.py: preprocess, one picture through libx264 at a
# CRF and back, by PyAV) on the README picture and on a crop with odd sides, recorded for the Rust
# port. Read-only apart from the output file.
import os, sys
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from comfy_extras.nodes_lt import preprocess

OUT = "/tmp/rustyui_preprocess_ref.safetensors"
img = Image.open(os.path.join(COMFYUI_DIR, "input/rustyui-i2v-lighthouse.png")).convert("RGB")
full = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
tensors = {}
for name, x in [("full", full), ("odd", full[3:3 + 601, 5:5 + 875])]:
    tensors[f"{name}.in"] = x.contiguous()
    for crf in (18, 35):
        tensors[f"{name}.out{crf}"] = preprocess(x, crf).contiguous()
        print(name, tuple(x.shape), crf, tuple(tensors[f"{name}.out{crf}"].shape))
save_file(tensors, OUT)
import av
print("saved", OUT, "PyAV", av.__version__, av.library_versions["libavcodec"], av.library_versions["libswscale"])
