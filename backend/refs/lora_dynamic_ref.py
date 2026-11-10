# The weights ComfyUI's DiT layers actually multiply by when LoRAs are on and the server runs with
# DynamicVRAM (comfy-aimdo's ModelPatcherDynamic): a LoRA is applied when a layer is first cast
# (comfy/ops.py resolve_cast_module_with_vbar), to the weight dequantized to the compute dtype, and
# an fp8 layer is quantized again with a recalculated scale. Runs an API graph in process as
# graph_nodes_ref.py does and records, for a few layers, the weight (an fp8 layer's bytes and
# scale, or the bf16 values) on the first two casts.
#   python lora_dynamic_ref.py GRAPH.json OUT.safetensors
import sys, os, json, asyncio

GRAPH, OUT = sys.argv[1], sys.argv[2]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401
import torch  # noqa: E402
import execution  # noqa: E402
import nodes  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

WANT = [
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.audio_attn1.to_q",
    "transformer_blocks.0.audio_to_video_attn.to_k",
    "transformer_blocks.23.attn2.to_v",
    "transformer_blocks.47.ff.net.2",
]
captured, info, seen = {}, {}, {}
original = comfy.ops.resolve_cast_module_with_vbar


def recording(s, dtype, device, bias_dtype, compute_dtype, want_requant, return_weights=True):
    r = original(s, dtype, device, bias_dtype, compute_dtype, want_requant, return_weights)
    key = getattr(s, "seed_key", None) or ""
    name = next((w for w in WANT if key.endswith(w + ".weight") or key.endswith(w)), None)
    if name is None or not return_weights:
        return r
    n = seen.get(name, 0)
    seen[name] = n + 1
    if n >= 2:
        return r
    w = r[0]
    p = f"{name}.call{n}"
    if isinstance(w, QuantizedTensor):
        q, scale = w._qdata, w._params.scale
        captured[p + ".qdata"] = q.view(torch.uint8).cpu().contiguous().clone()
        captured[p + ".scale"] = scale.float().reshape(-1).cpu().clone()
        info[p] = {"quantized": True, "orig_dtype": str(w._params.orig_dtype), "dtype": str(dtype), "compute_dtype": str(compute_dtype),
                   "want_requant": want_requant, "seed_key": key}
    else:
        captured[p + ".weight"] = w.float().cpu().contiguous().clone()
        info[p] = {"quantized": False, "weight_dtype": str(w.dtype), "dtype": str(dtype), "compute_dtype": str(compute_dtype),
                   "want_requant": want_requant, "seed_key": key}
    if r[1] is not None:
        captured[p + ".bias"] = r[1].float().cpu().contiguous().clone()
    return r


comfy.ops.resolve_cast_module_with_vbar = recording

import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
graph = json.load(open(GRAPH))
graph = graph.get("prompt", graph)
outs = []
for k, n in graph.items():
    if n["class_type"] in ("SaveVideo", "SaveImage"):
        n["inputs"]["filename_prefix"] = "rustyui-tests/capture-lora"
        outs.append(k)
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "capture-lora", {}, outs)
print(json.dumps(info, indent=1))
save_file(captured, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(captured), "tensors")
