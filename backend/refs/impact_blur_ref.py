# ComfyUI's GrowMask and Impact's tensor_gaussian_blur_mask (torchvision's GaussianBlur on the
# CPU) as those nodes call them, on a test mask, for rustyui's debug-impact-blur.
#   python impact_blur_ref.py MASK.png OUT.safetensors
import sys
import numpy as np
import scipy.ndimage
import torch
import torchvision
from PIL import Image
from safetensors.torch import save_file

PNG, OUT = sys.argv[1], sys.argv[2]
mask = torch.from_numpy(np.array(Image.open(PNG).convert("L")).astype(np.float32) / 255.0)
t = {"mask": mask}


def grow(m, expand, tapered):
    c = 0 if tapered else 1
    kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    out = m.numpy()
    for _ in range(abs(expand)):
        out = scipy.ndimage.grey_erosion(out, footprint=kernel) if expand < 0 else scipy.ndimage.grey_dilation(out, footprint=kernel)
    return torch.from_numpy(out)


def blur(m, kernel_size, sigma):
    mask = m[None, ..., None]
    kernel_size = kernel_size * 2 + 1
    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return m
    mask = mask[:, None, ..., 0]
    return torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)[0, 0]


grown = grow(mask, 8, True)
t["grow8_tapered"] = grown
t["grow3_square"] = grow(mask, 3, False)
t["shrink2"] = grow(mask, -2, True)
for k, sigma in [(15, 10.0), (5, 10.0), (20, 10.0), (3, 2.5)]:
    t[f"blur{k}_{sigma}"] = blur(grown, k, sigma).contiguous()
from torchvision.transforms import _functional_tensor as F_t  # noqa: E402
for k, sigma in [(31, 10.0), (11, 10.0), (41, 10.0), (7, 2.5)]:
    t[f"kernel1d_{k}"] = F_t._get_gaussian_kernel1d(k, sigma, dtype=torch.float32, device="cpu")
import torch.nn.functional as F  # noqa: E402
k2 = F_t._get_gaussian_kernel2d([31, 31], [10.0, 10.0], dtype=torch.float32, device="cpu")
pad31 = F.pad(grown[None, None], [15, 15, 15, 15], mode="reflect")
t["pad31"] = pad31[0, 0]
t["conv31"] = F.conv2d(pad31, k2.expand(1, 1, 31, 31))[0, 0]
save_file({k: v.contiguous() for k, v in t.items()}, OUT)
print("saved", OUT, {k: tuple(v.shape) for k, v in t.items()})
