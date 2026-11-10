# The README lighthouse graph's KSampler inside ComfyUI, step by step: the sigmas, and at every step
# the x it gives the model, the CFG-combined denoised output, and the final latent; for making
# rustyui's sampler loop bit for bit. Started as the server starts (main.py).
#   python sdxl_sample_ref.py OUT.safetensors [GRAPH.json]
import sys, os, json, asyncio

OUT = sys.argv[1]
GRAPH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/readme-graph.json")
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401
import torch  # noqa: E402
import execution  # noqa: E402
import nodes  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

t = {}
info = {}


def keep(name, x):
    t[name] = x.detach().float().cpu().contiguous().clone()


graph = json.load(open(GRAPH))
graph = graph.get("prompt", graph)
sampler = next(n["inputs"]["sampler_name"] for n in graph.values() if n["class_type"] == "KSampler")
fn_name = f"sample_{sampler}"
original = getattr(kds, fn_name)


def recording(model, x, sigmas, *a, **kw):
    keep("sigmas", sigmas)
    keep("x_start", x)
    step = [0]

    def wrapped(xx, sigma, **extra):
        i = step[0]
        step[0] += 1
        keep(f"step{i}.x", xx)
        keep(f"step{i}.sigma", sigma)
        den = model(xx, sigma, **extra)
        keep(f"step{i}.denoised", den)
        return den

    for attr in ("inner_model",):
        if hasattr(model, attr):
            setattr(wrapped, attr, getattr(model, attr))
    out = original(wrapped, x, sigmas, *a, **kw)
    keep("x_end", out)
    info["steps"] = step[0]
    return out


setattr(kds, fn_name, recording)

import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
outs = []
for k, n in graph.items():
    if n["class_type"] in ("SaveVideo", "SaveImage"):
        n["inputs"]["filename_prefix"] = "rustyui-tests/capture-sample"
        outs.append(k)
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "capture-sample", {}, outs)
info["sampler"] = sampler
print(json.dumps(info))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
