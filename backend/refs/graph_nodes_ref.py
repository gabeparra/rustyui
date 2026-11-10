# A whole API graph through ComfyUI in process, started the way the server starts (main.py's own
# setup: paths, prestartup scripts, DynamicVRAM through comfy-aimdo, every custom node), recording
# each node's outputs: every tensor in them, by "<node>.<output>.<path>" (a LATENT's
# samples.n0 / n1 are a NestedTensor's video and audio). For finding the first node where rustyui
# and ComfyUI part. Its SaveVideo / SaveImage write under output/rustyui-tests/capture-<mode>.
#   python graph_nodes_ref.py GRAPH.json[,GRAPH2.json...] OUT.safetensors [server|legacy]
# Several graphs run one after another in the one process, as a server's prompts do (the model
# stays loaded, its sigma table on the card); the tensors kept are the last graph's.
# CAPTURE_ORDER=1 prints each node as it finishes, in the order ComfyUI ran them; CAPTURE_CALLS=1
# each model call's batch; CAPTURE_CONTROLNET=1 the first two ControlNet calls inside out;
# CAPTURE_ANIMA=1 Anima piece by piece (its notes in the file's metadata).


# "legacy" turns DynamicVRAM off (--disable-dynamic-vram), the ModelPatcher the other refs load with.
import sys, os, json, asyncio

GRAPH, OUT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "server"
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"] + (["--disable-dynamic-vram"] if MODE == "legacy" else [])
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import torch  # noqa: E402
import execution  # noqa: E402
import nodes  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

captured = {}


def flat(prefix, v, depth=0):
    if depth > 8:
        return
    if isinstance(v, torch.Tensor):
        if v.is_floating_point() or v.dtype in (torch.int64, torch.int32, torch.bool):
            # a copy each: outputs share storage (a node passing its input on), which safetensors refuses
            captured[prefix] = (v.detach().float() if v.is_floating_point() else v.detach()).cpu().contiguous().clone()
    elif hasattr(v, "tensors") and isinstance(getattr(v, "tensors"), (list, tuple)):
        for i, t in enumerate(v.tensors):
            flat(f"{prefix}.n{i}", t, depth + 1)
    elif isinstance(v, dict):
        for k, x in v.items():
            flat(f"{prefix}.{k}", x, depth + 1)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            flat(f"{prefix}.{i}", x, depth + 1)


original = execution.get_output_data


async def recording(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
    r = await original(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
    if os.environ.get("CAPTURE_ORDER"):
        print("[capture] ran node", unique_id, type(obj).__name__, flush=True)
    flat(str(unique_id), r[0])
    return r


execution.get_output_data = recording

if os.environ.get("CAPTURE_CONTROLNET"):
    # the first two ControlNet model calls: their inputs (and the hint's memory layout), the hint
    # block's output, the union merge's steps (the spatial mean with the task embedding, the
    # transformer layer, the projection), the guided hint and every residual
    import comfy.cldm.cldm as cldm  # noqa: E402
    _cn_forward = cldm.ControlNet.forward
    cn_calls = [0]

    def _cn_keep(name, v):
        if isinstance(v, torch.Tensor):
            captured[name] = v.detach().float().cpu().contiguous().clone()

    def _cn_logged(self, x, hint, timesteps, context, y=None, **kwargs):
        i = cn_calls[0]
        cn_calls[0] += 1
        if i >= 2:
            return _cn_forward(self, x, hint, timesteps, context, y=y, **kwargs)
        p = f"cn.{i}"
        for n, v in (("x", x), ("hint", hint), ("timesteps", timesteps), ("context", context), ("y", y)):
            _cn_keep(f"{p}.{n}", v)
        print("[capture] controlnet call", i, "x", tuple(x.shape), x.dtype, "hint", tuple(hint.shape), hint.dtype, "stride", hint.stride(),
              "channels_last", hint.is_contiguous(memory_format=torch.channels_last), "control_type", kwargs.get("control_type"), flush=True)
        hooks = []
        def grab(name):
            def f(mod, args, out):
                if args:
                    _cn_keep(f"{p}.{name}.in", args[0])
                _cn_keep(f"{p}.{name}", out)
            return f
        hooks.append(self.input_hint_block.register_forward_hook(grab("input_hint_block")))
        if getattr(self, "transformer_layes", None) is not None:
            hooks.append(self.transformer_layes.register_forward_hook(grab("transformer_layes")))
            hooks.append(self.spatial_ch_projs.register_forward_hook(grab("spatial_ch_projs")))
            hooks.append(self.control_add_embedding.register_forward_hook(grab("control_add_embedding")))
        _merge = self.union_controlnet_merge
        def merged(*a, **k):
            out = _merge(*a, **k)
            _cn_keep(f"{p}.guided_hint", out)
            return out
        self.union_controlnet_merge = merged
        try:
            out = _cn_forward(self, x, hint, timesteps, context, y=y, **kwargs)
        finally:
            for h in hooks:
                h.remove()
            del self.union_controlnet_merge
        for j, o in enumerate(out["output"]):
            _cn_keep(f"{p}.output.{j}", o)
        _cn_keep(f"{p}.middle", out["middle"][0])
        return out

    cldm.ControlNet.forward = _cn_logged

if os.environ.get("CAPTURE_CALLS"):
    # each model call's batch: its input shape, which sides it holds (0 the positive) and the hook
    # group's id, to see how _calc_cond_batch split a step
    import comfy.model_base  # noqa: E402
    _apply = comfy.model_base.BaseModel.apply_model

    def _logged(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        to = transformer_options or {}
        print("[capture] model call", tuple(x.shape), "sides", to.get("cond_or_uncond"), "sigma", float(t[0]), flush=True)
        return _apply(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)

    comfy.model_base.BaseModel.apply_model = _logged



if os.environ.get("CAPTURE_ANIMA"):
    # Anima, piece by piece (rustyui's qwen3.rs, anima.rs): each Qwen3 run's tokens and final
    # states, each preprocess_text_embeds' inputs and context, the first DiT call's inputs, RoPE
    # table, timestep embedding, blocks 0, 1, 14 and 27 and output, every sampler call's denoised,
    # the position embedder's buffers and the SDPA backends torch picks for each attention shape
    import comfy.text_encoders.llama as llama  # noqa: E402
    import comfy.ldm.anima.model as anima_model  # noqa: E402
    import comfy.ldm.cosmos.predict2 as predict2  # noqa: E402
    import comfy.samplers  # noqa: E402

    def _a_keep(name, v):
        if isinstance(v, torch.Tensor):
            captured[name] = (v.detach().float() if v.is_floating_point() else v.detach()).cpu().contiguous().clone()

    a_info = {}
    _llama_forward = llama.Llama2_.forward
    te_calls = [0]

    def _llama_logged(self, x, attention_mask=None, embeds=None, *a, **kw):
        i = te_calls[0]
        te_calls[0] += 1
        _a_keep(f"te.{i}.embeds", embeds)
        _a_keep(f"te.{i}.attention_mask", attention_mask)
        out = _llama_forward(self, x, attention_mask, embeds, *a, **kw)
        _a_keep(f"te.{i}.out", out[0])
        return out

    llama.Llama2_.forward = _llama_logged
    _pre = anima_model.Anima.preprocess_text_embeds
    pre_calls = [0]

    def _pre_logged(self, text_embeds, text_ids, t5xxl_weights=None):
        i = pre_calls[0]
        pre_calls[0] += 1
        _a_keep(f"adapter.{i}.qwen", text_embeds)
        _a_keep(f"adapter.{i}.ids", text_ids)
        _a_keep(f"adapter.{i}.weights", t5xxl_weights)
        out = _pre(self, text_embeds, text_ids, t5xxl_weights=t5xxl_weights)
        _a_keep(f"adapter.{i}.context", out)
        a_info[f"adapter.{i}.dtype"] = str(out.dtype)
        return out

    anima_model.Anima.preprocess_text_embeds = _pre_logged
    _dit_forward = predict2.MiniTrainDIT._forward
    dit_calls = [0]

    def _dit_logged(self, x, timesteps, context, fps=None, padding_mask=None, **kw):
        i = dit_calls[0]
        dit_calls[0] += 1
        if i > 0:
            return _dit_forward(self, x, timesteps, context, fps, padding_mask, **kw)
        _a_keep("dit.x", x)
        _a_keep("dit.timesteps", timesteps)
        _a_keep("dit.context", context)
        pe = self.pos_embedder
        for n in ("dim_spatial_range", "dim_temporal_range"):
            _a_keep(f"dit.pos_embedder.{n}", getattr(pe, n))
        # the adapter's RoPE frequencies (made at __init__: on the CPU unless a default device says
        # otherwise, which decides their rounding)
        if hasattr(self, "llm_adapter"):
            _a_keep("adapter.inv_freq", self.llm_adapter.rotary_emb.inv_freq)
        a_info["dit.ntk"] = f"{pe.h_ntk_factor!r} {pe.w_ntk_factor!r} {pe.t_ntk_factor!r}"
        a_info["dit.dtype"] = f"x {x.dtype} weights {self.x_embedder.proj[1].weight.dtype}"
        hooks = []

        def grab(name):
            def f(mod, args, out):
                _a_keep(name, out[0] if isinstance(out, tuple) else out)
            return f
        hooks.append(self.x_embedder.register_forward_hook(grab("dit.x_embedded")))
        hooks.append(self.pos_embedder.register_forward_hook(grab("dit.rope")))
        hooks.append(self.t_embedder.register_forward_hook(lambda m, a, o: (_a_keep("dit.t_emb", o[0]), _a_keep("dit.adaln_lora", o[1]))))
        hooks.append(self.t_embedding_norm.register_forward_hook(grab("dit.t_emb_norm")))
        for b in (0, 1, 14, len(self.blocks) - 1):
            hooks.append(self.blocks[b].register_forward_hook(grab(f"dit.block{b}")))
        hooks.append(self.final_layer.register_forward_hook(grab("dit.final")))
        try:
            out = _dit_forward(self, x, timesteps, context, fps, padding_mask, **kw)
        finally:
            for h in hooks:
                h.remove()
        _a_keep("dit.out", out)
        return out

    predict2.MiniTrainDIT._forward = _dit_logged
    _sf = comfy.samplers.sampling_function
    sf_calls = [0]

    def _sf_logged(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        i = sf_calls[0]
        sf_calls[0] += 1
        out = _sf(model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed)
        _a_keep(f"step{i}.x", x)
        _a_keep(f"step{i}.sigma", timestep)
        _a_keep(f"step{i}.denoised", out)
        return out

    comfy.samplers.sampling_function = _sf_logged
    # the SDPA backend torch chooses for each attention call's shapes, in ComfyUI's priority
    _sdpa = torch.nn.functional.scaled_dot_product_attention
    seen_shapes = set()

    def _sdpa_logged(q, k, v, *a, **kw):
        key = (tuple(q.shape), tuple(k.shape), str(q.dtype), kw.get("attn_mask") is not None or (len(a) > 0 and a[0] is not None), bool(kw.get("enable_gqa", False)))
        if key not in seen_shapes:
            seen_shapes.add(key)
            mask = kw.get("attn_mask", a[0] if a else None)
            try:
                params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, bool(kw.get("enable_gqa", False)))
                a_info[f"sdpa {key}"] = f"flash {torch.backends.cuda.can_use_flash_attention(params)} cudnn {torch.backends.cuda.can_use_cudnn_attention(params)} efficient {torch.backends.cuda.can_use_efficient_attention(params)} choice {torch._fused_sdp_choice(q, k, v, mask, 0.0, False, enable_gqa=bool(kw.get('enable_gqa', False)))}"
            except Exception as e:  # noqa: BLE001
                a_info[f"sdpa {key}"] = f"? {e}"
        return _sdpa(q, k, v, *a, **kw)

    torch.nn.functional.scaled_dot_product_attention = _sdpa_logged

# a PromptServer as main.py's start_comfyui makes one (custom nodes add routes to it as they
# load), never started; the assets system stays off, as it does for an unknown asset manager
import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
# CAPTURE_EXTRA_NODES=DIR adds a folder of custom nodes not installed in this ComfyUI (a reference
# copy of a node pack), CAPTURE_STUBS=DIR a folder of stand-in modules for imports it cannot satisfy
if os.environ.get("CAPTURE_STUBS"):
    sys.path.insert(0, os.environ["CAPTURE_STUBS"])
if os.environ.get("CAPTURE_EXTRA_NODES"):
    import folder_paths  # noqa: E402
    folder_paths.add_model_folder_path("custom_nodes", os.environ["CAPTURE_EXTRA_NODES"])
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
# main.py's prompt_worker settings with the classic cache (which changes nothing computed)
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
for i, path in enumerate(GRAPH.split(",")):
    graph = json.load(open(path))
    graph = graph.get("prompt", graph)
    outs = []
    for k, n in graph.items():
        if n["class_type"] in ("SaveVideo", "SaveImage"):
            n["inputs"]["filename_prefix"] = f"rustyui-tests/capture-{MODE}"
            outs.append(k)
    ex.execute(graph, f"capture-{MODE}-{i}", {}, outs)
print("success", ex.success, "messages", [m[0] for m in ex.status_messages][-5:])
meta = a_info if os.environ.get("CAPTURE_ANIMA") else None
save_file(captured, OUT, metadata=meta)
for k, v in (meta or {}).items():
    print("[capture]", k, v)
print("saved", OUT, len(captured), "tensors")
