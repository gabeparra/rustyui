"""ComfyUI's guide-masked self-attention (comfy/ldm/lightricks/model.py: GuideAttentionMask,
_attention_with_guide_mask) on seeded inputs, saved for rustyui's debug-guide-attn.

Two sizes: a typical image-to-video graph's first pass (38 latent frames of 14x20, the last one the guide) and a 5-frame
one. q, k, v are (1, T, 32 * 128) bf16 as CrossAttention hands them over (after RoPE)."""
import sys, os
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "."))
sys.argv = [sys.argv[0]]
import torch
from safetensors.torch import save_file
import comfy.ops  # noqa: F401  (sets the SDPA backend priority)
from comfy.ldm.lightricks.model import GuideAttentionMask, _attention_with_guide_mask

out_path = "/tmp/rustyui_guide_attn_ref.safetensors"
H, D = 32, 128
tensors = {}
g = torch.Generator(device="cuda").manual_seed(1234)
for name, frames, ppf, strength in [("a", 38, 280, 0.7), ("b", 5, 280, 0.7), ("c", 5, 280, 0.35)]:
    T = frames * ppf
    G = ppf
    q = (torch.randn(1, T, H * D, device="cuda", generator=g) * 2.0).to(torch.bfloat16)
    k = (torch.randn(1, T, H * D, device="cuda", generator=g) * 2.0).to(torch.bfloat16)
    v = torch.randn(1, T, H * D, device="cuda", generator=g).to(torch.bfloat16)
    weights = torch.full((1, G), strength, device="cuda", dtype=torch.bfloat16)
    mask = GuideAttentionMask(T, T - G, G, weights)
    with torch.inference_mode():
        out = _attention_with_guide_mask(q, k, v, H, mask, attn_precision=None, transformer_options={})
    torch.cuda.synchronize()
    tensors.update({
        f"{name}.q": q, f"{name}.k": k, f"{name}.v": v,
        f"{name}.noisy_mask": mask.noisy_mask.contiguous(), f"{name}.tracked_mask": mask.tracked_mask.contiguous(),
        f"{name}.out": out.contiguous(),
        f"{name}.strength": torch.tensor([strength], dtype=torch.float64),
    })
    print(name, T, G, strength, "log_w", mask.noisy_mask[0, 0, 0, -1].item(), "out", tuple(out.shape), out.float().abs().max().item())
save_file({k: v.cpu() for k, v in tensors.items()}, out_path)
print("saved", out_path, "torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
