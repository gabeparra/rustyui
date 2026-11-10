# The f32 weights ComfyUI's Gemma forward actually multiplies by (cast_bias_weight, as F.linear
# gets them) for one fp8 layer and one NVFP4 layer, saved for the Rust kernels to be checked
# against bit for bit; and which fp8 formula reproduces them. Read-only apart from the output.
import os, sys, json, struct, torch
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.sd, comfy.ops, folder_paths
from safetensors.torch import save_file
prompt = json.load(open("/tmp/rustyui_ltx_prompt.json"))["prompt"]
te = folder_paths.get_full_path_or_raise("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")
ck = folder_paths.get_full_path_or_raise("checkpoints", "ltx-2.3-text-projection.safetensors")
clip = comfy.sd.load_clip(ckpt_paths=[te, ck], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.LTXV)
clip.load_model(clip.tokenize(prompt))
m = clip.cond_stage_model.gemma3_12b.transformer.model
dev = clip.patcher.load_device

def weight_used(mod):
    x = torch.zeros(1, mod.in_features, dtype=torch.float32, device=dev)
    r = comfy.ops.cast_bias_weight(mod, x, compute_dtype=torch.float32)
    return r[0].float().contiguous()

fp8_mod, nv_mod = m.layers[10].mlp.gate_proj, m.layers[0].self_attn.q_proj
w8, w4 = weight_used(fp8_mod), weight_used(nv_mod)
print("fp8 layer:", type(fp8_mod.weight).__name__, getattr(fp8_mod.weight, "_layout_cls", None), "->", tuple(w8.shape), w8.dtype)
print("nvfp4 layer:", type(nv_mod.weight).__name__, getattr(nv_mod.weight, "_layout_cls", None), "->", tuple(w4.shape), w4.dtype)

# raw fp8 bytes and scale from the file
from safetensors import safe_open
with safe_open(te, "pt", device="cuda") as f:
    q = f.get_tensor("model.layers.10.mlp.gate_proj.weight")
    s = f.get_tensor("model.layers.10.mlp.gate_proj.weight_scale")
print("raw:", q.dtype, tuple(q.shape), "scale", s.dtype, s.item())
qf = q.float()
cands = {
    "bf16(f32(q) * s)": (qf * s.float()).bfloat16().float(),
    "bf16(q) * bf16(s), rounded": (q.bfloat16().float() * s.bfloat16().float()).bfloat16().float(),
    "f32(q) * s, unrounded": qf * s.float(),
    "f16(f32(q) * s)": (qf * s.float()).half().float(),
}
for name, c in cands.items():
    diff = (c != w8.to(c.device)).sum().item()
    print(f"  {name:32s} differs on {diff} of {c.numel()}")
w4_is_bf16 = (w4.bfloat16().float() == w4).all().item()
print("nvfp4 weight values all bf16-representable:", w4_is_bf16)
save_file({"fp8_l10_gate": w8.cpu(), "nvfp4_l0_q": w4.cpu()}, "/tmp/rustyui_dequant_ref.safetensors")
print("saved")
