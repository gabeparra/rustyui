# SDXL's UNet inside ComfyUI on its first call of the README lighthouse graph
# (backend/refs/fixtures/readme-graph.json), for making rustyui's bit for bit: the call's inputs (x,
# timesteps, context, y), every input/middle/output block's output, op by op inside the first
# ResBlock (input_blocks.1.0) and the first transformer block (input_blocks.4.1), what SDPA receives
# for its self- and cross-attention (and which backend torch picks, in ComfyUI's priority), and the
# KSampler's latent. Runs the graph in process, started as the server starts (main.py).
#   python unet_ref.py OUT.safetensors [GRAPH.json]
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
import comfy.ldm.modules.diffusionmodules.openaimodel as oai  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

t, info = {}, {}


def keep(name, x):
    if isinstance(x, torch.Tensor):
        t[name] = x.detach().float().cpu().contiguous().clone()


def out_hook(name, inputs=False):
    # a hook that returns something replaces the output, so this returns nothing
    def f(mod, args, out):
        if inputs and args:
            keep(name + ".in", args[0])
        keep(name, out[0] if isinstance(out, tuple) else out)
    return f


INTERNALS = {
    "input_blocks.1.0.in_layers.0": True, "input_blocks.1.0.in_layers": False, "input_blocks.1.0.emb_layers": True,
    "input_blocks.1.0.out_layers.0": True, "input_blocks.1.0.out_layers": False,
    "input_blocks.4.1.norm": True, "input_blocks.4.1.proj_in": False, "input_blocks.4.1.proj_out": True,
    "input_blocks.4.1.transformer_blocks.0.norm1": True, "input_blocks.4.1.transformer_blocks.0.attn1.to_q": False,
    "input_blocks.4.1.transformer_blocks.0.attn1.to_k": False, "input_blocks.4.1.transformer_blocks.0.attn1.to_v": False,
    "input_blocks.4.1.transformer_blocks.0.attn1.to_out.0": True, "input_blocks.4.1.transformer_blocks.0.norm2": True,
    "input_blocks.4.1.transformer_blocks.0.attn2.to_q": False, "input_blocks.4.1.transformer_blocks.0.attn2.to_k": False,
    "input_blocks.4.1.transformer_blocks.0.attn2.to_v": False, "input_blocks.4.1.transformer_blocks.0.attn2.to_out.0": True,
    "input_blocks.4.1.transformer_blocks.0.norm3": True, "input_blocks.4.1.transformer_blocks.0.ff.net.0.proj": False,
    "input_blocks.4.1.transformer_blocks.0.ff.net.0": False, "input_blocks.4.1.transformer_blocks.0.ff.net.2": True,
    "input_blocks.4.1.transformer_blocks.0": True,
}
calls = [0]
original_forward = oai.UNetModel.forward


def recording_forward(self, x, timesteps=None, context=None, y=None, *a, **kw):
    first = calls[0] == 0
    calls[0] += 1
    if not first:
        return original_forward(self, x, timesteps, context, y, *a, **kw)
    keep("unet.x", x)
    keep("unet.timesteps", timesteps)
    keep("unet.context", context)
    keep("unet.y", y)
    info["dtype"] = str(self.dtype)
    info["x_dtype"] = str(x.dtype)
    info["batch"] = int(x.shape[0])
    info["kwargs"] = sorted(kw.keys())
    handles = []
    names = dict(self.named_modules())
    for n, with_in in INTERNALS.items():
        if n in names:
            handles.append(names[n].register_forward_hook(out_hook(n, inputs=with_in)))
        else:
            info.setdefault("missing", []).append(n)
    handles.append(self.time_embed.register_forward_hook(out_hook("time_embed", inputs=True)))
    handles.append(self.label_emb.register_forward_hook(out_hook("label_emb", inputs=True)))
    try:
        out = original_forward(self, x, timesteps, context, y, *a, **kw)
    finally:
        for h in handles:
            h.remove()
    keep("unet.out", out)
    return out


oai.UNetModel.forward = recording_forward

# each block's input and output: _forward runs a block's layers through forward_timestep_embed (the
# block module itself is never called), tagged in transformer_options["block"]
original_fte = oai.forward_timestep_embed


def recording_fte(ts, x, emb, context=None, transformer_options={}, *a, **kw):
    out = original_fte(ts, x, emb, context, transformer_options, *a, **kw)
    if calls[0] == 1:
        kind, idx = transformer_options.get("block", ("?", 0))
        name = {"input": f"input_blocks.{idx}", "middle": "middle_block", "output": f"output_blocks.{idx}"}.get(kind, f"{kind}.{idx}")
        keep(name + ".in", x)
        keep(name, out)
    return out


oai.forward_timestep_embed = recording_fte

# what SDPA receives for the first transformer block's self- and cross-attention
sdpa_original = torch.nn.functional.scaled_dot_product_attention
sdpa_calls = [0]


def sdpa_recording(q, k, v, attn_mask=None, *a, **kw):
    # counted only inside the UNet's first call (the text encoders' calls come before it)
    i = sdpa_calls[0] if calls[0] == 1 else 99
    if calls[0] == 1:
        sdpa_calls[0] += 1
    if i < 2:
        p = ["sdpa.self", "sdpa.cross"][i]
        keep(p + ".q", q)
        keep(p + ".k", k)
        keep(p + ".v", v)
        info[p] = {"shapes": [list(x.shape) for x in (q, k, v)], "strides": [list(x.stride()) for x in (q, k, v)],
                   "dtype": str(q.dtype), "mask": attn_mask is not None, "kwargs": {k2: str(v2) for k2, v2 in kw.items()}}
        import comfy.ops as cops
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(cops.SDPA_BACKEND_PRIORITY, set_priority=True):
            info[p]["choice"] = int(torch._fused_sdp_choice(q, k, v, attn_mask, 0.0, False, scale=None, enable_gqa=False))
        out = sdpa_original(q, k, v, attn_mask, *a, **kw)
        keep(p + ".out", out)
        return out
    return sdpa_original(q, k, v, attn_mask, *a, **kw)


torch.nn.functional.scaled_dot_product_attention = sdpa_recording

import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))

original_get = execution.get_output_data


async def get_recording(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
    r = await original_get(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
    if str(unique_id) == "5":
        keep("ksampler.samples", r[0][0][0]["samples"])
    return r


execution.get_output_data = get_recording
graph = json.load(open(GRAPH))
graph = graph.get("prompt", graph)
outs = []
for k, n in graph.items():
    if n["class_type"] in ("SaveVideo", "SaveImage"):
        n["inputs"]["filename_prefix"] = "rustyui-tests/capture-unet"
        outs.append(k)
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "capture-unet", {}, outs)
info["unet_calls"] = calls[0]
print(json.dumps(info, indent=1))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
