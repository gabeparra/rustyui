# SDXL's two text encoders inside ComfyUI, for making rustyui's bit for bit: for CLIP-G and CLIP-L,
# the embeddings, every op of layer 0, what SDPA receives (and which backend torch picks for it),
# the final layer norm, the projection, and CLIPTextEncode's conditioning and pooled output, for the
# README lighthouse prompt (backend/refs/fixtures/readme-graph.json, node 2). Started as the server
# starts (main.py); read-only apart from the output file.
#   python clip_ref.py OUT.safetensors
import sys, os, json

OUT = sys.argv[1]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

graph = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/readme-graph.json")))
graph = graph.get("prompt", graph)
prompt = graph["2"]["inputs"]["text"]
ckpt = folder_paths.get_full_path_or_raise("checkpoints", graph["1"]["inputs"]["ckpt_name"])
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(ckpt, output_vae=False, output_clipvision=False,
                                                           embedding_directory=folder_paths.get_folder_paths("embeddings"))
t, info = {}, {}


def keep(name, x):
    t[name] = x.detach().float().cpu().contiguous().clone()


def hook(name, inputs=False):
    def f(mod, args, out):
        if inputs:
            keep(name + ".in", args[0])
        keep(name, out[0] if isinstance(out, tuple) else out)
    return f


cm = clip.cond_stage_model
for enc in ("clip_g", "clip_l"):
    tm = getattr(cm, enc).transformer
    info[enc + ".dtype"] = str(next(tm.parameters()).dtype)
    tp = tm.text_projection.weight
    info[enc + ".text_projection"] = {"shape": list(tp.shape), "stride": list(tp.stride()), "contiguous": tp.is_contiguous()}
    m = tm.text_model
    m.embeddings.register_forward_hook(hook(f"{enc}.embeddings"))
    L = m.encoder.layers[0]
    L.layer_norm1.register_forward_hook(hook(f"{enc}.l0.ln1", inputs=True))
    for n in ("q_proj", "k_proj", "v_proj"):
        getattr(L.self_attn, n).register_forward_hook(hook(f"{enc}.l0.{n}"))
    L.self_attn.out_proj.register_forward_hook(hook(f"{enc}.l0.out_proj", inputs=True))
    L.layer_norm2.register_forward_hook(hook(f"{enc}.l0.ln2", inputs=True))
    L.mlp.fc1.register_forward_hook(hook(f"{enc}.l0.fc1"))
    L.mlp.fc2.register_forward_hook(hook(f"{enc}.l0.fc2", inputs=True))
    L.register_forward_hook(lambda mod, args, out, enc=enc: keep(f"{enc}.l0.out", out) or None)
    m.final_layer_norm.register_forward_hook(hook(f"{enc}.final_ln", inputs=True))
    tm.text_projection.register_forward_hook(hook(f"{enc}.text_projection", inputs=True))

# the first SDPA call of each encoder (G runs first, 32 layers, then L)
original = torch.nn.functional.scaled_dot_product_attention
calls = [0]


def recording(q, k, v, attn_mask=None, *a, **kw):
    i = calls[0]
    calls[0] += 1
    if i in (0, 32):
        p = "clip_g" if i == 0 else "clip_l"
        keep(f"{p}.sdpa.q", q)
        keep(f"{p}.sdpa.k", k)
        keep(f"{p}.sdpa.v", v)
        if attn_mask is not None:
            keep(f"{p}.sdpa.mask", attn_mask)
        info[p + ".sdpa"] = {"shapes": [list(x.shape) for x in (q, k, v)], "strides": [list(x.stride()) for x in (q, k, v)],
                             "mask": None if attn_mask is None else [list(attn_mask.shape), list(attn_mask.stride()), str(attn_mask.dtype)],
                             "kwargs": {k2: str(v2) for k2, v2 in kw.items()},
                             "choice": int(torch._fused_sdp_choice(q, k, v, attn_mask, 0.0, False, scale=None, enable_gqa=False))}
        out = original(q, k, v, attn_mask, *a, **kw)
        keep(f"{p}.sdpa.out", out)
        return out
    return original(q, k, v, attn_mask, *a, **kw)


torch.nn.functional.scaled_dot_product_attention = recording
tokens = clip.tokenize(prompt)
for enc in ("g", "l"):
    t[f"tokens_{enc}"] = torch.tensor([x[0] for x in tokens[enc][0]], dtype=torch.int64)
with torch.inference_mode():
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
keep("cond", cond)
keep("pooled", pooled)
info["calls"] = calls[0]
print(json.dumps(info, indent=1))
save_file(t, OUT, metadata={"info": json.dumps(info)})
print("saved", OUT, len(t), "tensors")
