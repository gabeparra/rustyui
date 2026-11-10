# ComfyUI's Gemma 3 12B (the LTX-AV text encoder) inside its first layers, for making rustyui's
# bit for bit: the padded ids and their mask, both RoPE tables, and every op of layer 0 (a sliding
# layer) and layer 5 (the first global one) as ComfyUI computes them, the attention's inputs as SDPA
# receives them, which SDPA backend torch picks for them, and all 49 hidden states. Runs ComfyUI's own
# model in process; read-only apart from the output file.
import os, sys, json
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import torch
import comfy.sd, folder_paths
import comfy.ldm.modules.attention as cattn
from safetensors.torch import save_file

OUT = "/tmp/rustyui_gemma_layer_ref.safetensors"
LAYERS = (0, 5)
prompt = json.load(open("/tmp/rustyui_ltx_prompt.json"))["prompt"]
te = folder_paths.get_full_path_or_raise("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")
ck = folder_paths.get_full_path_or_raise("checkpoints", "ltx-2.3-text-projection.safetensors")
clip = comfy.sd.load_clip(ckpt_paths=[te, ck], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.LTXV)
tokens = clip.tokenize(prompt)
ids = [t[0] for t in tokens["gemma3_12b"][0]]
m = clip.cond_stage_model.gemma3_12b.transformer.model

t = {"ids": torch.tensor(ids, dtype=torch.int64)}
info = {"n_padded": len(ids), "n_real": sum(1 for i in ids if i != 0)}


def keep(name, x):
    t[name] = x.detach().float().cpu().contiguous()


def hook(name, inputs=False):
    def f(mod, args, out):
        if inputs:
            keep(name + ".in", args[0])
        keep(name, out[0] if isinstance(out, tuple) else out)
    return f


for li in LAYERS:
    L = m.layers[li]
    p = f"l{li}."
    # the input before the layer runs: it ends with torch.add(residual, x, out=x), which writes the
    # output into the input tensor; and a hook that returns something replaces what it hooks, so
    # these return nothing
    def layer_in(mod, args, kwargs, p=p):
        keep(p + "x", kwargs.get("x", args[0] if args else None))

    def layer_out(mod, args, kwargs, out, p=p):
        keep(p + "out", out[0])

    L.register_forward_pre_hook(layer_in, with_kwargs=True)
    L.register_forward_hook(layer_out, with_kwargs=True)
    L.input_layernorm.register_forward_hook(hook(p + "input_norm"))
    a = L.self_attn
    for n in ("q_proj", "k_proj", "v_proj"):
        getattr(a, n).register_forward_hook(hook(p + n))
    a.q_norm.register_forward_hook(hook(p + "q_norm"))
    a.k_norm.register_forward_hook(hook(p + "k_norm"))
    a.o_proj.register_forward_hook(hook(p + "o_proj", inputs=True))
    L.post_attention_layernorm.register_forward_hook(hook(p + "post_attn_norm"))
    L.pre_feedforward_layernorm.register_forward_hook(hook(p + "pre_ff_norm"))
    L.mlp.gate_proj.register_forward_hook(hook(p + "gate"))
    L.mlp.up_proj.register_forward_hook(hook(p + "up"))
    L.mlp.down_proj.register_forward_hook(hook(p + "down", inputs=True))
    L.post_feedforward_layernorm.register_forward_hook(hook(p + "post_ff_norm"))

# the attention ComfyUI hands the layers: record what SDPA receives in the layers above, and which
# backend torch picks for it
original = cattn.attention_pytorch
calls = [0]


def recording(q, k, v, heads, mask=None, **kw):
    i = calls[0]
    calls[0] += 1
    if i in LAYERS:
        p = f"l{i}."
        keep(p + "sdpa.q", q)
        keep(p + "sdpa.k", k)
        keep(p + "sdpa.v", v)
        if mask is not None:
            keep(p + "sdpa.mask", mask)
        # what comfy.ops.scaled_dot_product_attention decides, in its own backend priority: native
        # GQA if a fused backend takes it, else k and v repeated; then the backend torch picks
        import comfy.ops as cops
        from torch.nn.attention import sdpa_kernel
        m4 = mask if mask is None or mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        with sdpa_kernel(cops.SDPA_BACKEND_PRIORITY, set_priority=True):
            params = torch.backends.cuda.SDPAParams(q, k, v, m4, 0.0, False, True)
            info[p + "native_gqa"] = [torch.backends.cuda.can_use_flash_attention(params), torch.backends.cuda.can_use_cudnn_attention(params),
                                      torch.backends.cuda.can_use_efficient_attention(params)]
            kr, vr = cops.repeat_kv_for_gqa(k, v, q.shape[-3], -3)
            info[p + "choice_repeated"] = str(torch._fused_sdp_choice(q, kr, vr, m4, 0.0, False, scale=None, enable_gqa=False))
            try:
                info[p + "choice_gqa"] = str(torch._fused_sdp_choice(q, k, v, m4, 0.0, False, scale=None, enable_gqa=True))
            except Exception as e:
                info[p + "choice_gqa"] = repr(e)
        info[p + "sdpa_kwargs"] = {k2: str(v2) for k2, v2 in kw.items()}
        info[p + "dtypes"] = [str(q.dtype), str(k.dtype), str(v.dtype), str(mask.dtype) if mask is not None else None]
        info[p + "shapes"] = [list(q.shape), list(k.shape), list(v.shape), list(mask.shape) if mask is not None else None]
        info[p + "strides"] = [list(q.stride()), list(k.stride()), list(v.stride())]
        out = original(q, k, v, heads, mask=mask, **kw)
        keep(p + "sdpa.out", out)
        return out
    return original(q, k, v, heads, mask=mask, **kw)


cattn.attention_pytorch = recording

# both RoPE tables as the model computes them for the padded positions
pos = torch.arange(0, len(ids), device="cuda").unsqueeze(0)
fc = m.compute_freqs_cis(pos, torch.device("cuda"))
for j, (cos, sin, nsin) in enumerate(fc):
    keep(f"rope{j}.cos", cos)
    keep(f"rope{j}.sin", sin)
    keep(f"rope{j}.neg_sin", nsin)

clip.load_model(tokens)
gm = clip.cond_stage_model
gm.set_clip_options({"execution_device": clip.patcher.load_device})
with torch.inference_mode():
    out, pooled, extra = gm.gemma3_12b.encode_token_weights(tokens["gemma3_12b"])
n = int(torch.sum(extra["attention_mask"]).item())
keep("attention_mask", extra["attention_mask"])
keep("hidden", out[:, :, -n:])
info["gemma_dtype"] = str(m.layers[0].input_layernorm.weight.dtype)
info["calls"] = calls[0]
print(json.dumps(info, indent=1))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
