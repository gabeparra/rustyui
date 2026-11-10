# cv2.resize(INTER_CUBIC) on uint8 pictures as OpenCV 5.0.0 runs it (it hands these to
# Intel IPP), for rustyui's debug-ipp-resize: random pictures of 1, 3 and 4 channels, enlarged and
# shrunk to odd and even sizes, Depth Anything's 720-to-1024 case among them.
#   python ipp_resize_ref.py OUT.safetensors
import sys
import numpy as np
import cv2
import torch
from safetensors.torch import save_file

assert cv2.ipp.useIPP(), "OpenCV's IPP is off"
rng = np.random.default_rng(11)
cases = [(720, 1280, 3, 1024, 1820), (720, 1280, 3, 1820, 1024), (480, 640, 3, 1024, 1365), (37, 53, 1, 101, 77),
         (100, 100, 3, 333, 333), (64, 48, 4, 129, 97), (518, 700, 3, 1024, 1384), (1000, 667, 3, 640, 427),
         (31, 29, 3, 200, 211), (256, 256, 1, 255, 257), (900, 600, 3, 1536, 1024), (7, 5, 3, 64, 33)]
t = {"version": torch.tensor(list(cv2.ipp.getIppVersion().encode()), dtype=torch.uint8)}
for i, (h, w, c, oh, ow) in enumerate(cases):
    # smooth fields plus noise, as depth maps and photos have both
    base = rng.integers(0, 256, size=(h // 8 + 2, w // 8 + 2, c)).astype(np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img.astype(np.int16) + rng.integers(-20, 21, size=img.shape), 0, 255).astype(np.uint8)
    # cv2 hands a one-channel picture back as (h, w)
    if c == 1 and img.ndim == 3:
        img = img[..., 0]
    out = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
    t[f"case{i}.in"] = torch.from_numpy(np.ascontiguousarray(img).reshape(h, w, c))
    t[f"case{i}.out"] = torch.from_numpy(np.ascontiguousarray(out).reshape(oh, ow, c))
save_file(t, sys.argv[1])
print("saved", len(cases), "cases;", cv2.ipp.getIppVersion())
