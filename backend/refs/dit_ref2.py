# One real sampling step of LTX-2.3's DiT inside ComfyUI, recorded piece by piece for the
# Rust port. A typical text-to-video graph's nodes (Gemma encoder, LTXVConditioning, UNETLoader, empty AV
# latents, LTXVScheduler, CFGGuider cfg 1, euler) at a small size, no image guide or LoRAs. Saves,
# in their own dtypes: the connector's input and output, the first DiT call's arguments and result,
# the prepared timesteps (CompressedTimestep included), and for blocks 0 (bf16) and 2 (the first
# fp8 block) every submodule's input and output plus each attention call's q, k, v and result,
# with the CUDA kernels each attention ran. Copies are taken when recorded (the blocks work in
# place). Read-only apart from the output file.
import os, asyncio, json, sys, time, torch
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.options
comfy.options.enable_args_parsing(False)
import folder_paths, nodes, execution
import comfy.ldm.lightricks.av_model as avm
import comfy.ldm.modules.attention as cattn
from safetensors.torch import save_file

OUT = "/tmp/rustyui_dit_ref2.safetensors"
LENGTH = 9
prompt_text = json.load(open("/tmp/rustyui_ltx_prompt.json"))["prompt"]
tensors, meta = {}, {"size": [448, 640, LENGTH]}
BLOCKS = (0, 2)
SUBMODULES = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn", "ff", "audio_ff")


def keep(name, obj, depth=0):
    """Every tensor in obj (tuples, lists, dicts, objects with __dict__ or __slots__), copied to the CPU
    in its own dtype; the rest by type."""
    if torch.is_tensor(obj):
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


state = {"block": None, "attn": None, "attn_n": 0}

orig_pre = avm.LTXAVModel.preprocess_text_embeds
orig_fwd = avm.LTXAVModel.forward
orig_ts = avm.LTXAVModel._prepare_timestep
orig_out = avm.LTXAVModel._process_output
orig_attention = cattn.optimized_attention


def pre(self, context, unprocessed=False):
    out = orig_pre(self, context, unprocessed=unprocessed)
    if "connector_in" not in tensors:
        keep("connector_in", context)
        keep("connector_out", out)
        meta["connector_unprocessed"] = unprocessed
    return out


def prep_ts(self, timestep, batch_size, hidden_dtype, **kwargs):
    out = orig_ts(self, timestep, batch_size, hidden_dtype, **kwargs)
    if "ts.out" not in meta:
        keep("ts.in", timestep)
        keep("ts.a_timestep_in", kwargs.get("a_timestep"))
        meta["ts.hidden_dtype"] = str(hidden_dtype)
        meta["ts.orig_shape"] = list(kwargs["orig_shape"]) if kwargs.get("orig_shape") is not None else None
        keep("ts.out", out)
    return out


def proc_out(self, x, embedded_timestep, keyframe_idxs, **kwargs):
    if "final.in" not in meta:
        keep("final.in", x)
        keep("final.embedded_timestep", embedded_timestep)
    return orig_out(self, x, embedded_timestep, keyframe_idxs, **kwargs)


def attention(q, k, v, heads, *args, **kwargs):
    """Records every attention call made inside a recorded block, with the kernels it ran."""
    if state["block"] is None:
        return orig_attention(q, k, v, heads, *args, **kwargs)
    unwrap = lambda t: t.tensor if hasattr(t, "tensor") else t
    name = f"b{state['block']}.{state['attn'] or 'x'}.sdpa{state['attn_n']}"
    state["attn_n"] += 1
    keep(f"{name}.q", unwrap(q))
    keep(f"{name}.k", unwrap(k))
    keep(f"{name}.v", unwrap(v))
    meta[f"{name}.heads"] = heads
    meta[f"{name}.mask"] = None if kwargs.get("mask") is None else str(type(kwargs.get("mask")))
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = orig_attention(q, k, v, heads, *args, **kwargs)
        torch.cuda.synchronize()
    meta[f"{name}.kernels"] = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    keep(f"{name}.out", out)
    return out


def fwd(self, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options={}, keyframe_idxs=None, **kwargs):
    keep("in.x", x)
    keep("in.timestep", timestep)
    keep("in.context", context)
    keep("in.attention_mask", attention_mask)
    meta["in.frame_rate"] = frame_rate if not torch.is_tensor(frame_rate) else frame_rate.tolist()
    keep("in.keyframe_idxs", keyframe_idxs)
    for k, v in kwargs.items():
        keep(f"in.kw.{k}", v)
    hooks = []
    for bi in BLOCKS:
        blk = self.transformer_blocks[bi]

        def block_pre(mod, args, kw, bi=bi):
            keep(f"b{bi}.in.x", args[0] if args else kw.get("x"))
            for k, v in kw.items():
                if k != "transformer_options":
                    keep(f"b{bi}.in.{k}", v)
            state["block"] = bi

        def block_post(mod, args, kw, out, bi=bi):
            keep(f"b{bi}.out", out)
            state["block"] = None

        hooks.append(blk.register_forward_pre_hook(block_pre, with_kwargs=True))
        hooks.append(blk.register_forward_hook(block_post, with_kwargs=True))
        for sub in SUBMODULES:
            m = getattr(blk, sub)

            def sub_pre(mod, args, kw, bi=bi, sub=sub):
                keep(f"b{bi}.{sub}.in", args[0] if args else kw.get("x"))
                for k in ("context", "pe", "k_pe"):
                    if kw.get(k) is not None:
                        keep(f"b{bi}.{sub}.in_{k}", kw[k])
                state["attn"], state["attn_n"] = sub, 0

            def sub_post(mod, args, kw, out, bi=bi, sub=sub):
                keep(f"b{bi}.{sub}.out", out)
                state["attn"] = None

            hooks.append(m.register_forward_pre_hook(sub_pre, with_kwargs=True))
            hooks.append(m.register_forward_hook(sub_post, with_kwargs=True))
            if hasattr(m, "to_gate_logits") and m.to_gate_logits is not None:
                def gate_post(mod, args, out, bi=bi, sub=sub):
                    keep(f"b{bi}.{sub}.gate_logits", out)
                hooks.append(m.to_gate_logits.register_forward_hook(gate_post))
    t = time.time()
    try:
        out = orig_fwd(self, x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    torch.cuda.synchronize()
    meta["forward_seconds"] = round(time.time() - t, 2)
    keep("out", out)
    meta["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    raise Done()


avm.LTXAVModel.preprocess_text_embeds = pre
avm.LTXAVModel.forward = fwd
avm.LTXAVModel._prepare_timestep = prep_ts
avm.LTXAVModel._process_output = proc_out
cattn.optimized_attention = attention


class Server:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *a, **k):
        pass

    def queue_updated(self):
        pass


graph = {
    "te": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2.3-text-projection.safetensors", "device": "default"}},
    "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["te", 0]}},
    "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["te", 0]}},
    "cond": {"class_type": "LTXVConditioning", "inputs": {"frame_rate": 30, "positive": ["pos", 0], "negative": ["neg", 0]}},
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.3-distilled.safetensors", "weight_dtype": "default"}},
    "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-audio-vae.safetensors"}},
    "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 448, "height": 640, "length": LENGTH, "batch_size": 1}},
    "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": LENGTH, "frame_rate": 30, "batch_size": 1, "audio_vae": ["avae", 0]}},
    "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
    "sched": {"class_type": "LTXVScheduler", "inputs": {"steps": 8, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["av", 0]}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "guider": {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1]}},
    "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "samp": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0], "sigmas": ["sched", 0], "latent_image": ["av", 0]}},
}


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    missing = sorted({n["class_type"] for n in graph.values()} - set(nodes.NODE_CLASS_MAPPINGS))
    if missing:
        sys.exit(f"nodes not available: {missing}")
    # the cache settings main.py gives the executor by default
    import comfy.model_management as mm
    cache_ram = min(10.0, max(2.0, mm.total_ram * 0.10 / 1024.0))
    cache_ram_inactive = min(128.0, mm.total_ram / 1024.0)
    ex = execution.PromptExecutor(Server(), cache_type=execution.CacheType.RAM_PRESSURE,
                                  cache_args={"lru": 0, "ram": cache_ram, "ram_inactive": cache_ram_inactive})
    t = time.time()
    await ex.execute_async(graph, "rustyui-dit-ref2", {}, ["samp"])
    meta["total_seconds"] = round(time.time() - t, 1)


asyncio.run(main())
meta["captured"] = "out" in tensors
small = {k: v for k, v in meta.items() if not isinstance(v, dict)}
print(json.dumps(small, indent=0, default=str)[:6000])
print("tensors:", len(tensors), "MB:", round(sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20))
if tensors:
    save_file(tensors, OUT, metadata={"meta": json.dumps(meta, default=str)})
    print("saved", OUT)
