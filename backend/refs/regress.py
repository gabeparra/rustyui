"""backend/refs/regress.py: run each SDXL fixture through rustyui (a fresh process each, as a capture is) and compare every
picture it writes with ComfyUI's capture of the same graph, pixel for pixel. Run it in ComfyUI's
venv (numpy, Pillow): "$COMFYUI_DIR"/.venv/bin/python backend/refs/regress.py [CASE...]"""
import glob, os, shutil, subprocess, sys
import numpy as np
from PIL import Image

FIX = os.path.expanduser("~/.cache/rustyui/fixtures")
CAP = os.path.join(os.environ.get("COMFYUI_DIR", "."), "output/rustyui-tests/capture-server_{:05d}_.png")
# RUSTYUI_BIN names another build (the dev container builds into target-dev/)
BIN = os.environ.get("RUSTYUI_BIN", os.path.expanduser("~/rustyui/target/release/rustyui-backend"))
CASES = [
    ("readme-graph", [os.path.join(os.environ.get("COMFYUI_DIR", "."), "output/rustyui-readme/lighthouse-comfyui-cu130_00001_.png")]),
    ("readme-lora", [7]), ("readme-ea", [8]), ("readme-euler", [9]), ("readme-ea33", [10]),
    ("frombatch", [11, 12, 13]), ("upscale-loaded", [25]), ("latent-upscale", [26, 27]), ("inpaint", [29, 30]), ("readme-sde", [28]), ("img2img", [29]),
    ("readme-ersde", [31]), ("readme-sde-gpu", [32]), ("readme-cfgpp", [33]),
    ("duo", [34]), ("duo-hires", [36]), ("plain-hires-odd", [38]), ("cn-depth", [40]), ("cn-chain", [41]), ("canny", [42, 43]), ("cn-canny", [44]),
    ("depth-node", [45, 46]),
    # the pack's own SAM3Segment output through ComfyUI's MaskToImage and SaveImage arithmetic
    # (the reference ComfyUI has no ComfyUI-RMBG): refs/sam3_ref.py on the same picture
    ("sam3-mask", [os.path.join(os.environ.get("COMFYUI_DIR", "."), "output/rustyui-tests/sam3-mask-expected.png")]),
    # Impact's detailer chain (GrowMask, ImpactGaussianBlurMask, MaskToSEGS, DetailerForEach),
    # captured with the pack loaded from ~/.cache/rustyui/impact_ref (CAPTURE_EXTRA_NODES)
    ("detailer", [47]), ("detailer-real", [48]), ("detailer-variants", [49]), ("detailer-skip", [50]), ("dd-ksampler", [51]),
    # TiledAttentionCouple: the hub's shape (a LoRA on the text encoders, hires with a second TAC)
    # and three vertical tiles with negative tile prompts
    ("tac-hub", [54]), ("tac-three", [55]),
]
only = sys.argv[1:]
bad = 0
for name, caps in CASES:
    if only and name not in only:
        continue
    out = f"/tmp/regress/{name}"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([BIN, "run", f"{FIX}/{name}.json", "--out", out], capture_output=True, text=True)
    if r.returncode:
        print(f"{name:18} FAILED: {r.stderr.strip().splitlines()[-1:]}")
        bad += 1
        continue
    ours = sorted(glob.glob(f"{out}/**/*.png", recursive=True))
    caps = [c if isinstance(c, str) else CAP.format(c) for c in caps]
    # each capture against the picture of ours nearest it (the file names do not keep its order)
    res = []
    pics = [np.asarray(Image.open(o).convert("RGB")) for o in ours]
    for c in caps:
        b = np.asarray(Image.open(c).convert("RGB"))
        diffs = [int((a != b).any(-1).sum()) for a in pics if a.shape == b.shape]
        res.append("no picture of that size" if not diffs else "identical" if min(diffs) == 0 else f"{min(diffs)} px differ")
    if len(ours) != len(caps):
        res.append(f"{len(ours)} pictures for {len(caps)} captures")
    ok = all(x == "identical" for x in res)
    bad += not ok
    print(f"{name:18} {'ok ' if ok else 'BAD'} {', '.join(res)}")
print("all identical" if bad == 0 else f"{bad} cases differ")
