# ComfyUI-Impact-Pack's MaskToSEGS (core.mask_to_segs) and mask blur (utils.tensor_gaussian_blur_mask)
# run from the pack's own files, for rustyui's debug-impact-segs and debug-impact-blur. The pack is
# not installed in the reference ComfyUI, so it is imported from a reference copy
# (~/.cache/rustyui/impact_ref/custom_nodes/ComfyUI-Impact-Pack, with stubs/ for segment_anything
# and skimage, which these functions never call).
#   python impact_segs_ref.py MASK.png SEGS_OUT.safetensors BLUR_OUT.safetensors
# MASK.png is the test mask (its red channel); the segment cases are that mask grown and blurred
# as tools/inpaint.py's graph does, and drawn ones: rings with islands, objects on the edges,
# diagonal links, specks under drop_size, an object inside another's hole.
import os
import sys

PNG, SEGS_OUT, BLUR_OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REF = os.path.expanduser("~/.cache/rustyui/impact_ref")
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.path[:0] = [os.path.join(REF, "stubs"), os.path.join(REF, "custom_nodes/ComfyUI-Impact-Pack/modules"), COMFY]
sys.argv = [sys.argv[0]]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import scipy.ndimage  # noqa: E402
from impact import core, utils  # noqa: E402

mask = torch.from_numpy(np.array(Image.open(PNG).convert("RGBA"))[..., 0].astype(np.float32) / 255.0)


def grow(m, expand):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    out = m.numpy()
    for _ in range(expand):
        out = scipy.ndimage.grey_dilation(out, footprint=kernel)
    return torch.from_numpy(out)


def blurred(m, k=15, sigma=10.0):
    return utils.tensor_gaussian_blur_mask(m[None, ..., None], k, sigma)[0, ..., 0]


cases = {}
cases["fixture"] = (blurred(grow(mask, 8)), (False, 3.0, False, 10, False))
cases["fixture_g0"] = (blurred(mask), (False, 3.0, False, 10, False))
cases["fixture_contour"] = (blurred(grow(mask, 8)), (False, 3.0, False, 10, True))
cases["fixture_combined"] = (blurred(grow(mask, 8)), (True, 3.0, False, 10, False))
cases["fixture_combined_fill"] = (blurred(grow(mask, 8)), (True, 3.0, True, 10, False))
cases["fixture_bboxfill"] = (blurred(grow(mask, 8)), (False, 1.5, True, 10, False))

# drawn masks: values in [0, 1], some under 1/255
h, w = 300, 420
d = np.zeros((h, w), np.float32)
yy, xx = np.mgrid[0:h, 0:w]
ring = ((yy - 80) ** 2 + (xx - 90) ** 2 < 60 ** 2) & ((yy - 80) ** 2 + (xx - 90) ** 2 > 35 ** 2)
d[ring] = 0.8
d[(yy - 80) ** 2 + (xx - 90) ** 2 < 12 ** 2] = 1.0          # island in the ring's hole
d[0:40, 300:420] = 0.5                                        # on the top and right edges
d[250:300, 0:30] = 1.0                                        # bottom left corner
for i in range(40):                                           # a diagonal line, 8-connected only
    d[150 + i, 200 + i] = 1.0
d[200:203, 330:333] = 1.0                                     # a speck under drop_size
d[120:180, 300:360] = 0.6                                     # a square with a hole holding a square
d[135:165, 315:345] = 0.0
d[145:155, 325:335] = 0.9
d[260:280, 100:300] = 0.003                                   # under 1/255: dropped by the uint8 cast
d[230:290, 330:400] = np.linspace(0.0, 1.0, 70)[None, :]      # a ramp starting under 1/255
d[100:104, 150:250] = 1.0                                     # two blobs joined by one diagonal pixel
d[104, 250] = 1.0
d[105:110, 251:280] = 1.0
cases["drawn"] = (torch.from_numpy(d), (False, 3.0, False, 10, False))
cases["drawn_drop1"] = (torch.from_numpy(d), (False, 2.0, False, 1, False))
cases["drawn_contour"] = (torch.from_numpy(d), (False, 3.0, False, 2, True))
cases["drawn_bboxfill"] = (torch.from_numpy(d), (False, 1.0, True, 2, False))
cases["drawn_combined"] = (torch.from_numpy(d), (True, 1.3, True, 10, False))
rng = np.random.default_rng(7)
noise = (rng.random((180, 260)) > 0.93).astype(np.float32) * rng.random((180, 260)).astype(np.float32)
cases["speckle"] = (torch.from_numpy(noise), (False, 3.0, False, 1, False))
cases["speckle_blur"] = (blurred(torch.from_numpy(noise), 2, 3.0), (False, 3.0, False, 3, False))

segs_t = {}
for name, (m, (combined, crop_factor, bbox_fill, drop_size, contour_fill)) in cases.items():
    m = m.contiguous().float()
    segs_t[f"{name}.mask"] = m
    segs_t[f"{name}.opts"] = torch.tensor([float(combined), crop_factor, float(bbox_fill), float(drop_size), float(contour_fill)])
    shape, segs = core.mask_to_segs(m, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
    segs_t[f"{name}.count"] = torch.tensor([len(segs)], dtype=torch.int64)
    for i, s in enumerate(segs):
        segs_t[f"{name}.seg{i}.crop"] = torch.tensor([int(v) for v in s.crop_region], dtype=torch.int64)
        segs_t[f"{name}.seg{i}.bbox"] = torch.tensor([int(v) for v in s.bbox], dtype=torch.int64)
        segs_t[f"{name}.seg{i}.mask"] = torch.from_numpy(np.ascontiguousarray(s.cropped_mask)).float()
    print(name, len(segs), [tuple(int(v) for v in s.crop_region) for s in segs][:6])
save_file({k: v.clone().contiguous() for k, v in segs_t.items()}, SEGS_OUT)

# the blur on small masks and batches: slow_conv2d's 3x3 path at most 20480 values, oneDNN above
# that or for a batch, and the kernel shrunk to fit a short side
torch.manual_seed(3)
blur_t = {}
for n, bh, bw, k in [(1, 40, 60, 1), (1, 150, 150, 1), (2, 40, 60, 1), (1, 9, 30, 5), (1, 5, 30, 5), (1, 4, 30, 5),
                     (1, 3, 30, 5), (1, 741, 741, 20), (1, 741, 741, 5), (1, 64, 64, 1), (3, 30, 50, 4), (1, 1024, 1024, 20)]:
    x = torch.rand(n, bh, bw)
    name = f"small.{n}x{bh}x{bw}.k{k}"
    blur_t[f"{name}.in"] = x
    blur_t[name] = utils.tensor_gaussian_blur_mask(x[..., None], k, 10.0)[..., 0].contiguous()
save_file({k: v.clone().contiguous() for k, v in blur_t.items()}, BLUR_OUT)
print("saved", SEGS_OUT, len(segs_t), BLUR_OUT, len(blur_t))
