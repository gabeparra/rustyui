# The first call ComfyUI's server makes to LTX-2.3's DiT for a graph, recorded for the Rust
# port: its arguments (the packed or nested latent, timestep, context, masks, frame rate, keyframes),
# the prepared timesteps, every transformer block's output, the head's input and the call's result.
# With CFG above 1 the server batches the negative and the prompt into this one call (uncond first),
# which is what this was written to capture. Started as the server starts (main.py); the run is cut
# short after the first call, and nothing is written apart from the output file.
#   python dit_batch_ref.py GRAPH.json OUT.safetensors
import sys, os, json, asyncio

GRAPH, OUT = sys.argv[1], sys.argv[2]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import torch  # noqa: E402
import execution  # noqa: E402
import nodes  # noqa: E402
import server  # noqa: E402
import comfy.ldm.lightricks.av_model as avm  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

tensors, meta = {}, {}


def keep(name, obj, depth=0):
    """Every tensor in obj (tuples, lists, dicts, objects with __dict__ or __slots__, nested
    tensors part by part), copied to the CPU in its own dtype; the rest by type."""
    if hasattr(obj, "tensors") and not torch.is_tensor(obj):
        meta[name] = f"{type(obj).__name__}[{len(obj.tensors)}]"
        for i, t in enumerate(obj.tensors):
            keep(f"{name}.n{i}", t, depth + 1)
    elif torch.is_tensor(obj):
        t = obj.detach()
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.uint8, torch.bool):
            t = t.to(torch.float32)
        tensors[name] = t.cpu().contiguous().clone()
        meta[name] = {"dtype": str(obj.dtype), "shape": list(obj.shape)}
    elif isinstance(obj, (tuple, list)) and depth < 5:
        meta[name] = f"{type(obj).__name__}[{len(obj)}]"
        for i, v in enumerate(obj):
            keep(f"{name}.{i}", v, depth + 1)
    elif isinstance(obj, dict) and depth < 5:
        for k, v in obj.items():
            if k != "transformer_options":
                keep(f"{name}.{k}", v, depth + 1)
    elif hasattr(obj, "__slots__") and not callable(obj) and depth < 5:
        meta[name] = type(obj).__name__
        for k in obj.__slots__:
            keep(f"{name}.{k}", getattr(obj, k, None), depth + 1)
    elif hasattr(obj, "__dict__") and depth < 5 and not callable(obj):
        meta[name] = type(obj).__name__
        for k, v in vars(obj).items():
            keep(f"{name}.{k}", v, depth + 1)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        meta[name] = obj
    else:
        meta[name] = type(obj).__name__


class Done(Exception):
    pass


state = {"calls": 0}
orig_fwd = avm.LTXAVModel.forward
orig_ts = avm.LTXAVModel._prepare_timestep
orig_out = avm.LTXAVModel._process_output


def prep_ts(self, timestep, batch_size, hidden_dtype, **kwargs):
    out = orig_ts(self, timestep, batch_size, hidden_dtype, **kwargs)
    if state["calls"] == 1 and "ts.out" not in meta:
        keep("ts.in", timestep)
        keep("ts.a_timestep_in", kwargs.get("a_timestep"))
        meta["ts.batch_size"] = batch_size
        meta["ts.orig_shape"] = list(kwargs["orig_shape"]) if kwargs.get("orig_shape") is not None else None
        keep("ts.out", out)
    return out


def proc_out(self, x, embedded_timestep, keyframe_idxs, **kwargs):
    if state["calls"] == 1 and "final.in" not in meta:
        keep("final.in", x)
        keep("final.embedded_timestep", embedded_timestep)
    return orig_out(self, x, embedded_timestep, keyframe_idxs, **kwargs)


def fwd(self, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options={}, keyframe_idxs=None, **kwargs):
    state["calls"] += 1
    if state["calls"] > 1:
        raise Done()
    keep("in.x", x)
    keep("in.timestep", timestep)
    keep("in.context", context)
    keep("in.attention_mask", attention_mask)
    meta["in.frame_rate"] = frame_rate if not torch.is_tensor(frame_rate) else frame_rate.tolist()
    meta["in.cond_or_uncond"] = transformer_options.get("cond_or_uncond")
    keep("in.keyframe_idxs", keyframe_idxs)
    for k, v in kwargs.items():
        keep(f"in.kw.{k}", v)
    hooks = []
    for bi, blk in enumerate(self.transformer_blocks):
        def block_post(mod, args, kw, out, bi=bi):
            keep(f"b{bi}.out", out)
        if bi == 0:
            def block_pre(mod, args, kw):
                keep("b0.in.x", args[0] if args else kw.get("x"))
                for k, v in kw.items():
                    if k != "transformer_options":
                        keep(f"b0.in.{k}", v)
            hooks.append(blk.register_forward_pre_hook(block_pre, with_kwargs=True))
        hooks.append(blk.register_forward_hook(block_post, with_kwargs=True))
    meta["blocks"] = len(self.transformer_blocks)
    try:
        out = orig_fwd(self, x, timestep, context, attention_mask=attention_mask, frame_rate=frame_rate, transformer_options=transformer_options, keyframe_idxs=keyframe_idxs, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    keep("out", out)
    return out


avm.LTXAVModel.forward = fwd
avm.LTXAVModel._prepare_timestep = prep_ts
avm.LTXAVModel._process_output = proc_out

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
graph = json.load(open(GRAPH))
graph = graph.get("prompt", graph)
outs = [k for k, n in graph.items() if n["class_type"] in ("SaveImage", "SaveVideo")]
for k in outs:
    graph[k]["inputs"]["filename_prefix"] = "rustyui-tests/dit-batch"
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "dit-batch", {}, outs)  # stops at the second DiT call (Done), as intended
print(json.dumps({k: v for k, v in meta.items() if not k.startswith("b") or k.startswith("b0.")}, indent=1)[:4000])
save_file(tensors, OUT, metadata={"info": json.dumps(meta)})
print("saved", OUT, len(tensors), "tensors")
