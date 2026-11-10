# SDXL's VAE decode inside ComfyUI on the README lighthouse graph's latent, for making rustyui's bit
# for bit: the decoder's input, every block's output (each up level's res blocks and upsample), op by
# op inside the first res block and the middle attention, which SDPA backend torch picks for that
# attention (in ComfyUI's priority) and the VAEDecode image. Runs the graph in process, started as
# the server starts (main.py).
#   python sdxl_vae_ref.py OUT.safetensors [GRAPH.json]
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
import comfy.ldm.modules.diffusionmodules.model as vae_model  # noqa: E402
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
    "mid.block_1.norm1": True, "mid.block_1.conv1": True, "mid.block_1.norm2": True, "mid.block_1.conv2": True,
    "mid.attn_1.norm": True, "mid.attn_1.q": False, "mid.attn_1.k": False, "mid.attn_1.v": False, "mid.attn_1.proj_out": True,
    "up.3.upsample.conv": True, "norm_out": True, "conv_out": True,
}
original_forward = vae_model.Decoder.forward
calls = [0]


def recording_forward(self, z, **kw):
    if calls[0]:
        return original_forward(self, z, **kw)
    calls[0] += 1
    keep("decoder.z", z)
    info["dtype"] = str(z.dtype)
    info["weight_dtype"] = str(self.conv_in.weight.dtype)
    names = dict(self.named_modules())
    handles = []
    for n, with_in in INTERNALS.items():
        handles.append(names[n].register_forward_hook(out_hook(n, inputs=with_in)))
    blocks = ["conv_in", "mid.block_1", "mid.attn_1", "mid.block_2"]
    for i in reversed(range(self.num_resolutions)):
        blocks += [f"up.{i}.block.{j}" for j in range(self.num_res_blocks + 1)]
        if i != 0:
            blocks.append(f"up.{i}.upsample")
    for n in blocks:
        handles.append(names[n].register_forward_hook(out_hook(n, inputs=True)))
    info["blocks"] = blocks
    # the backend torch picks for the middle attention's SDPA, in ComfyUI's priority
    a = names["mid.attn_1"]
    info["attention"] = getattr(a.optimized_attention, "__name__", str(a.optimized_attention))
    try:
        out = original_forward(self, z, **kw)
    finally:
        for h in handles:
            h.remove()
    keep("decoder.out", out)
    q = t["mid.attn_1.q"]
    B, C = q.shape[0], q.shape[1]
    qq = q.view(B, 1, C, -1).transpose(2, 3).contiguous().to("cuda", torch.bfloat16)
    import comfy.ops as cops
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel(cops.SDPA_BACKEND_PRIORITY, set_priority=True):
        info["sdpa_choice"] = str(torch._fused_sdp_choice(qq, qq, qq, None, 0.0, False))
    return out


vae_model.Decoder.forward = recording_forward

original_out = execution.get_output_data


async def recording_out(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
    r = await original_out(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
    if type(obj).__name__ == "VAEDecode":
        keep("vae_decode.image", r[0][0][0])
    if type(obj).__name__ == "KSampler":
        keep("ksampler.latent", r[0][0][0]["samples"])
    return r


execution.get_output_data = recording_out

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
    if n["class_type"] == "SaveImage":
        n["inputs"]["filename_prefix"] = "rustyui-tests/capture-vae"
        outs.append(k)
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "capture-vae", {}, outs)
print(json.dumps(info, indent=1))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
