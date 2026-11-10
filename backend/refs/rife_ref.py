# ComfyUI-Frame-Interpolation's RIFE VFI node as a typical video graph runs it (rife49.pth, multiplier 2,
# fast_mode and ensemble on, scale 1, float32, batch 1), inside ComfyUI started as main.py starts it,
# on the first frames of a node dump's decoded video. Records the node's input and output and, for
# its first IFNet call, every stage: the encoder's features, each IFBlock's input, convolutions and
# (flow, mask), each warp. Read-only apart from the output file.
#   python rife_ref.py NODES.safetensors KEY FRAMES OUT.safetensors
import sys, os, json, asyncio, importlib
DUMP, KEY, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import torch  # noqa: E402
import nodes  # noqa: E402
import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
node_cls = nodes.NODE_CLASS_MAPPINGS["RIFE VFI"]
arch = importlib.import_module(node_cls.__module__ + ".rife_arch")

tensors = {}
def keep(name, t):
    tensors[name] = t.detach().float().cpu().contiguous().clone()

state = {"calls": 0}
orig_fwd = arch.IFNet.forward
orig_warp = arch.warp
warps = {"n": 0}

def warp(inp, flow):
    out = orig_warp(inp, flow)
    if state["calls"] == 1:
        i = warps["n"]
        warps["n"] += 1
        keep(f"warp{i}.in", inp)
        keep(f"warp{i}.flow", flow)
        keep(f"warp{i}.out", out)
    return out

def fwd(self, img0, img1, timestep=0.5, scale_list=None, training=True, fastmode=True, ensemble=False, return_flow=False):
    state["calls"] += 1
    hooks = []
    if state["calls"] == 1:
        keep("in.img0", img0)
        keep("in.img1", img1)
        keep("in.timestep", timestep)
        tensors["in.flags"] = torch.tensor([float(training), float(fastmode), float(ensemble)])
        tensors["in.scale_list"] = torch.tensor([float(s) for s in scale_list])
        enc_n = {"n": 0}
        def enc_hook(m, a, out):
            keep(f"encode{enc_n['n']}", out)
            enc_n["n"] += 1
        hooks.append(self.encode.register_forward_hook(enc_hook))
        for bi, blk in enumerate([self.block0, self.block1, self.block2, self.block3]):
            def pre(m, a, kw, bi=bi):
                keep(f"b{bi}.x", a[0])
                fl = a[1] if len(a) > 1 else kw.get("flow")
                if fl is not None:
                    keep(f"b{bi}.flow_in", fl)
            def post(m, a, kw, out, bi=bi):
                keep(f"b{bi}.flow", out[0])
                keep(f"b{bi}.mask", out[1])
            hooks.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
            hooks.append(blk.register_forward_hook(post, with_kwargs=True))
            for sub in ("conv0", "convblock", "lastconv"):
                def sub_hook(m, a, out, bi=bi, sub=sub):
                    keep(f"b{bi}.{sub}", out)
                hooks.append(getattr(blk, sub).register_forward_hook(sub_hook))
    try:
        out = orig_fwd(self, img0, img1, timestep, scale_list, training, fastmode, ensemble, return_flow)
    finally:
        for h in hooks:
            h.remove()
    if state["calls"] == 1:
        keep("out", out)
    return out

arch.IFNet.forward = fwd
arch.warp = warp
with safe_open(DUMP, "pt") as f:
    frames = f.get_tensor(KEY)[:N].contiguous()
keep("node.in", frames)
(result,) = node_cls().vfi("rife49.pth", frames, clear_cache_after_n_frames=10, multiplier=2, fast_mode=True,
                           ensemble=True, scale_factor=1.0, dtype="float32", torch_compile=False, batch_size=1)
keep("node.out", result)
print(json.dumps({k: list(v.shape) for k, v in tensors.items()})[:3000])
save_file(tensors, OUT)
print("saved", OUT, len(tensors), "tensors; IFNet calls", state["calls"], "warps in the first", warps["n"])
