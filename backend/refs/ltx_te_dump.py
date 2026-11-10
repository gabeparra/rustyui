# Runs ComfyUI's own LTX-AV text encoder (Gemma 3 12B fp4 + LTX-2.3 projection) on one prompt and
# saves what the Rust port must reproduce. Read-only: loads models, writes one output file.
import os, sys, json, torch
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.sd, folder_paths
from safetensors.torch import save_file
prompt = json.load(open(sys.argv[0] if False else "/tmp/rustyui_ltx_prompt.json"))["prompt"]
te = folder_paths.get_full_path_or_raise("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")
ck = folder_paths.get_full_path_or_raise("checkpoints", "ltx-2.3-text-projection.safetensors")
clip = comfy.sd.load_clip(ckpt_paths=[te, ck], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.LTXV)
tokens = clip.tokenize(prompt)
ids = [t[0] for t in tokens["gemma3_12b"][0]]
m = clip.cond_stage_model.gemma3_12b.transformer.model
q = m.layers[0].self_attn.q_proj.weight
info = {
    "n_tokens_padded": len(ids), "n_real": sum(1 for i in ids if i != 0),
    "embed_dtype": str(m.embed_tokens.weight.dtype),
    "norm_dtype": str(m.layers[0].input_layernorm.weight.dtype),
    "qnorm_dtype": str(m.layers[0].self_attn.q_norm.weight.dtype),
    "q_proj_type": type(q).__name__, "q_proj_dtype": str(getattr(q, "dtype", None)),
    "q_proj_layout": str(getattr(q, "_layout_cls", getattr(q, "layout", None))),
    "projection_dtype": str(clip.cond_stage_model.text_embedding_projection.video_aggregate_embed.weight.dtype),
}
print(json.dumps(info))
# all hidden states of the real tokens, straight from the Gemma wrapper
clip.load_model(tokens)
gm = clip.cond_stage_model
gm.set_clip_options({"execution_device": clip.patcher.load_device})
out, pooled, extra = gm.gemma3_12b.encode_token_weights(tokens["gemma3_12b"])
n = int(torch.sum(extra["attention_mask"]).item())
hidden = out[:, :, -n:].float().cpu().contiguous()
gm.reset_clip_options()
res = clip.encode_from_tokens(tokens, return_dict=True)
cond = res["cond"].float().cpu().contiguous()
print("hidden", tuple(hidden.shape), hidden.dtype, "| cond", tuple(cond.shape), "| extra keys", [k for k in res if k != "cond"])
save_file({"ids": torch.tensor(ids, dtype=torch.int64), "hidden": hidden, "cond": cond}, "/tmp/rustyui_ltx_te_ref.safetensors", metadata={"info": json.dumps(info)})
print("saved")
