# ComfyUI's LTX text projection taken apart at its two linears: what each receives and returns
# (fed the saved reference hidden states), and whether PyTorch's bf16 GEMM there equals an
# exactly accumulated one. Read-only apart from the output file.
import os, sys, json, torch
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.sd, folder_paths
from safetensors.torch import save_file, load_file
te = folder_paths.get_full_path_or_raise("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")
ck = folder_paths.get_full_path_or_raise("checkpoints", "ltx-2.3-text-projection.safetensors")
clip = comfy.sd.load_clip(ckpt_paths=[te, ck], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.LTXV)
proj = clip.cond_stage_model.text_embedding_projection
dev = torch.device("cuda")
proj.to(dev)
print("torch", torch.__version__, "cublasLt", torch.backends.cuda.preferred_blas_library(),
      "| bf16 reduced-precision reduction allowed:", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
      "| tf32:", torch.backends.cuda.matmul.allow_tf32)
print("ops:", type(proj.video_aggregate_embed).__module__, type(proj.video_aggregate_embed).__qualname__)
ref = load_file("/tmp/rustyui_ltx_te_ref.safetensors")
hidden = ref["hidden"].to(dev).to(torch.bfloat16)          # (1, 49, T, 3840), as encode_token_weights hands it over
seen = {}
def hook(name):
    def f(mod, inp, out):
        seen[name + "_in"] = inp[0].detach().clone()
        seen[name + "_out"] = out.detach().clone()
    return f
proj.video_aggregate_embed.register_forward_hook(hook("video"))
proj.audio_aggregate_embed.register_forward_hook(hook("audio"))
with torch.no_grad():
    full = proj(hidden)
print("full", tuple(full.shape), full.dtype, "| equals saved cond:", torch.equal(full.float().cpu(), ref["cond"]))
for name in ["video", "audio"]:
    lin = getattr(proj, f"{name}_aggregate_embed")
    x, y = seen[name + "_in"], seen[name + "_out"]
    w, b = lin.weight, lin.bias
    print(f"{name}: in {tuple(x.shape)} {x.dtype} contiguous={x.is_contiguous()}, w {w.dtype}, out {y.dtype}")
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        exact = (x.float() @ w.float().t() + b.float()).to(torch.bfloat16)
        exact64 = (x.double() @ w.double().t() + b.double()).to(torch.bfloat16)
        again = torch.nn.functional.linear(x, w, b)
    for label, c in [("f32-accumulated", exact), ("f64-accumulated", exact64), ("F.linear again", again)]:
        d = (c != y).sum().item()
        print(f"   {label:16s} differs from ComfyUI's output on {d} of {y.numel()}")
save_file({k: v.float().cpu().contiguous() for k, v in seen.items()}, "/tmp/rustyui_proj_ref.safetensors")
print("saved")
