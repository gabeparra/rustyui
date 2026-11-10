# Reference outputs of LTX-2.3's fp8 linears inside ComfyUI, for the Rust port. Each layer is
# built by comfy.ops.mixed_precision_ops from the checkpoint's own tensors (weight, weight_scale,
# bias, comfy_quant) and fed a fixed bf16 input. Saves the input, ComfyUI's quantised input bytes,
# its output, an f64 product of the same fp8 values (to tell rounding from mistakes), and the
# layer's tensors, so a smaller machine can run the same cases without the 27 GB checkpoint.
# Read-only apart from the output file; a few hundred MB of VRAM for a few seconds.
import ctypes, glob, json, os, sys, torch
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.options
comfy.options.enable_args_parsing(False)
import comfy.ops, comfy.quant_ops
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = os.path.join(COMFYUI_DIR, "models/checkpoints/ltx-2.3-distilled.safetensors")
OUT = "/tmp/rustyui_fp8_ref.safetensors"
P = "model.diffusion_model.transformer_blocks.2."
# (layer, rows, input scale, outliers): rows as in the 448x640x9 test (560 video tokens,
# 8 audio, 1024 context); ff.net.2 also gets values past fp8's 448 to exercise the clamp.
CASES = [
    ("attn1.to_q", 560, 1.0, False),
    ("ff.net.0.proj", 560, 1.0, False),
    ("ff.net.2", 560, 3.0, True),
    ("audio_attn1.to_q", 8, 1.0, False),
    ("attn2.to_k", 1024, 0.5, False),
    ("video_to_audio_attn.to_k", 560, 1.0, False),
]

tensors, meta = {}, {"torch": torch.__version__, "cuda": torch.version.cuda, "cases": []}
import nvidia.cublas
lt = glob.glob(os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib", "libcublasLt.so*"))
if lt:
    meta["cublaslt_version"] = ctypes.CDLL(lt[0]).cublasLtGetVersion()
meta["CUBLASLT_WORKSPACE_SIZE"] = os.environ.get("CUBLASLT_WORKSPACE_SIZE")

ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
f = safe_open(CKPT, "pt", device="cpu")
gen = torch.Generator().manual_seed(1234)
for i, (name, rows, scale, outliers) in enumerate(CASES):
    w = f.get_tensor(P + name + ".weight")
    sd = {"weight": w, "weight_scale": f.get_tensor(P + name + ".weight_scale"),
          "bias": f.get_tensor(P + name + ".bias"), "comfy_quant": f.get_tensor(P + name + ".comfy_quant")}
    out_f, in_f = w.shape
    lin = ops.Linear(in_f, out_f, bias=True, device="cuda", dtype=torch.bfloat16)
    lin.load_state_dict(sd, strict=False)
    assert type(lin.weight.data).__name__ == "QuantizedTensor", type(lin.weight.data)
    x = (torch.randn(rows, in_f, generator=gen) * scale)
    if outliers:
        x.view(-1)[::997] *= 300.0
    x = x.to(torch.bfloat16).cuda()
    with torch.inference_mode():
        y = lin(x)
        qx, params = comfy.quant_ops.TensorCoreFP8E4M3Layout.quantize(x)
    torch.cuda.synchronize()
    wq = w.cuda()
    exact = (qx.to(torch.float64) @ wq.to(torch.float64).t()) * float(params.scale) * float(sd["weight_scale"]) \
        + sd["bias"].cuda().to(torch.float64)
    k = f"c{i}"
    tensors[f"{k}.x"] = x.cpu()
    tensors[f"{k}.qx"] = qx.view(torch.uint8).cpu()
    tensors[f"{k}.y"] = y.cpu()
    tensors[f"{k}.exact"] = exact.to(torch.float32).cpu()
    tensors[f"{k}.w"] = w.view(torch.uint8).contiguous()
    tensors[f"{k}.w_scale"] = sd["weight_scale"].reshape(1).float()
    tensors[f"{k}.bias"] = sd["bias"]
    yb = exact.to(torch.bfloat16)
    meta["cases"].append({"name": name, "rows": rows, "in": in_f, "out": out_f, "y_dtype": str(y.dtype),
                          "act_scale": float(params.scale),
                          "y_vs_exact_bf16_mismatch": float((y != yb).float().mean()),
                          "max_abs_x": float(x.abs().max())})
    del lin, wq
save_file(tensors, OUT, metadata={"meta": json.dumps(meta)})
print(json.dumps(meta, indent=1))
print("saved", OUT, "vram_peak_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2))
