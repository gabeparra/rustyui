# SAM3Segment (ComfyUI-RMBG) as the node runs it, for making rustyui's bit for bit. The pack is
# not installed in the reference ComfyUI, so its own files are imported from a copy
# (~/.cache/rustyui/rmbg_ref: py/AILab_SAM3Segment.py, AILab_ImageMaskTools.py, AILab_utils.py and
# models/sam3, plus stubs/ for iopath, decord and pycocotools, which only training code uses),
# with the checkpoint at ComfyUI's models/sam3/sam3.pt. Records each stage for one picture and one
# prompt: the picture as the node sees it, the network input, the ViT's patch embedding and some
# blocks, the neck's levels, the text encoder, the geometry encoder, the fusion encoder, the
# decoder, the mask head, the post-processing and the node's MASK.
#   python sam3_ref.py PICTURE PROMPT OUT.safetensors [output_mode confidence max_segments segment_pick mask_blur]
import os
import sys

PIC, PROMPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else "Separate"
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.35
MAXSEG = int(sys.argv[6]) if len(sys.argv) > 6 else 8
PICK = int(sys.argv[7]) if len(sys.argv) > 7 else 1
BLUR = int(sys.argv[8]) if len(sys.argv) > 8 else 4
REF = os.path.expanduser("~/.cache/rustyui/rmbg_ref")
sys.path[:0] = [os.path.join(REF, "stubs"), os.path.join(REF, "py"), os.environ.get("COMFYUI_DIR", ".")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from AILab_SAM3Segment import SAM3Segment  # noqa: E402

t = {}


def keep(name, v):
    """every tensor in v (tensors, tuples, lists, dicts), under name.i / name.key"""
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(np.ascontiguousarray(v))
    if isinstance(v, torch.Tensor):
        if v.is_complex():
            v = torch.view_as_real(v)
        if v.dtype == torch.bool:
            v = v.to(torch.uint8)
        t[name] = v.detach().cpu().contiguous().clone()
    elif isinstance(v, (tuple, list)):
        for i, x in enumerate(v):
            keep(f"{name}.{i}", x)
    elif isinstance(v, dict):
        for k, x in v.items():
            keep(f"{name}.{k}", x)


def grab(name, inputs=False):
    def f(mod, args, kwargs, out):
        keep(name, out)
        if inputs:
            keep(name + ".in", list(args))
            keep(name + ".kw", {k: v for k, v in kwargs.items() if isinstance(v, (torch.Tensor, list, tuple, dict))})
    return f


node = SAM3Segment()
proc, _ = node._load_processor("Auto")
m = proc.model

_transform = proc.transform


def transform(img):
    out = _transform(img)
    keep("pic_u8", img)
    keep("net_in", out)
    return out


proc.transform = transform

trunk = m.backbone.vision_backbone.trunk
trunk.patch_embed.register_forward_hook(grab("patch_embed"), with_kwargs=True)
trunk.ln_pre.register_forward_hook(grab("ln_pre"), with_kwargs=True)
for i in (0, 1, 7, 8, 15, 31):
    trunk.blocks[i].register_forward_hook(grab(f"block{i}"), with_kwargs=True)
trunk.register_forward_hook(grab("vit_out"), with_kwargs=True)
m.backbone.vision_backbone.register_forward_hook(grab("neck"), with_kwargs=True)
lb = m.backbone.language_backbone
lb.encoder.register_forward_hook(grab("text_encoder"), with_kwargs=True)
for i in (0, 23):
    lb.encoder.transformer.resblocks[i].register_forward_hook(grab(f"text_block{i}"), with_kwargs=True)
lb.register_forward_hook(grab("text"), with_kwargs=True)
_tok = lb.tokenizer


def tokenizer(texts, context_length):
    out = _tok(texts, context_length=context_length)
    keep("tokens", out)
    return out


lb.tokenizer = tokenizer
m.geometry_encoder.register_forward_hook(grab("geometry", inputs=True), with_kwargs=True)
m.transformer.encoder.register_forward_hook(grab("fusion", inputs=True), with_kwargs=True)
m.transformer.decoder.register_forward_hook(grab("decoder", inputs=True), with_kwargs=True)
m.segmentation_head.register_forward_hook(grab("seghead", inputs=True), with_kwargs=True)
m.dot_prod_scoring.register_forward_hook(grab("scoring", inputs=True), with_kwargs=True)
# inside the decoder's first layer, and its inputs
dec = m.transformer.decoder
LAYERS = [int(x) for x in os.environ.get("SAM3_REF_LAYERS", "0").split(",")]
for li in LAYERS:
    for n in ("self_attn", "ca_text", "cross_attn", "norm1", "norm2", "norm3", "catext_norm", "linear1", "linear2"):
        getattr(dec.layers[li], n).register_forward_hook(grab(f"dl{li}.{n}", inputs=True), with_kwargs=True)
dec.ref_point_head.register_forward_hook(grab("dec.ref_point_head", inputs=True), with_kwargs=True)
dec.bbox_embed.register_forward_hook(grab("dec.bbox_embed", inputs=True), with_kwargs=True)
dec.presence_token_head.register_forward_hook(grab("dec.presence_head", inputs=True), with_kwargs=True)
sc = m.dot_prod_scoring
sc.prompt_proj.register_forward_hook(grab("scoring_pooled_proj", inputs=True), with_kwargs=True)
sc.hs_proj.register_forward_hook(grab("scoring_hs_proj"), with_kwargs=True)
sc.prompt_mlp.register_forward_hook(grab("scoring_prompt_mlp"), with_kwargs=True)
_rpb = dec._get_rpb_matrix
_rpb_n = [0]


def rpb(boxes, size):
    out = _rpb(boxes, size)
    keep(f"dec.rpb{_rpb_n[0]}", out)
    keep(f"dec.rpb{_rpb_n[0]}_boxes", boxes)
    _rpb_n[0] += 1
    return out


dec._get_rpb_matrix = rpb
_rpb_calls = {"x": [0], "y": [0]}


def grab_rpb(axis):
    def f(mod, args, kwargs, out):
        i = _rpb_calls[axis][0]
        keep(f"dec.rpb{axis}{i}", out)
        keep(f"dec.rpb{axis}{i}.in", kwargs.get("x", args[0] if args else None))
        _rpb_calls[axis][0] += 1
    return f


dec.boxRPB_embed_x.register_forward_hook(grab_rpb("x"), with_kwargs=True)
dec.boxRPB_embed_y.register_forward_hook(grab_rpb("y"), with_kwargs=True)

_ground = proc._forward_grounding


def forward_grounding(state):
    _fg = m.forward_grounding

    def fg(*a, **k):
        out = _fg(*a, **k)
        keep("grounding", {k2: v for k2, v in out.items() if k2 in ("pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec")})
        return out

    m.forward_grounding = fg
    try:
        st = _ground(state)
    finally:
        m.forward_grounding = _fg
    keep("post", {k: st[k] for k in ("masks_logits", "masks", "boxes", "scores") if k in st})
    return st


proc._forward_grounding = forward_grounding

img = torch.from_numpy(np.array(Image.open(PIC).convert("RGB")).astype(np.float32) / 255.0)[None]
keep("image", img)
r = node.segment(img, PROMPT, "Auto", confidence_threshold=CONF, max_segments=MAXSEG, segment_pick=PICK,
                 mask_blur=BLUR, mask_offset=0, invert_output=False, unload_model=False, background="Alpha",
                 background_color="#222222", output_mode=MODE)
keep("node_image", r[0])
keep("node_mask", r[1])
keep("node_mask_image", r[2])
save_file(t, OUT)
print("saved", OUT, len(t), "tensors")
for k in sorted(t):
    print(f"  {k} {tuple(t[k].shape)} {str(t[k].dtype).replace('torch.', '')}")
