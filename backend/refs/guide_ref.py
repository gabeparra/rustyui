# A small LTX-2.3 run guided by a picture, inside ComfyUI, recorded for the Rust port: the start image
# (input/rustyui-i2v-lighthouse.png, docs/examples/lighthouse.png) through LTXVAddGuide (frame_idx 0,
# strength STRENGTH, as a typical image-to-video graph guides) onto a 640x448x17 latent at 30 fps, seed 42,
# LTXVScheduler 8 steps, euler, cfg 1, no LoRAs, then LTXVCropGuides and the video VAE decode. Saves
# what i2v_ref.py saves plus the first DiT call's keyframe_idxs and guide entries and the
# cropped latent. Read-only apart from the output file.
import os, asyncio, json, sys, time, torch
ARGS = sys.argv[1:]
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import comfy.options
comfy.options.enable_args_parsing(False)
import folder_paths, nodes, execution
import comfy.sample
import comfy.sd
import comfy.k_diffusion.sampling as kds
from safetensors.torch import save_file

STRENGTH = float(ARGS[0]) if ARGS else 0.7
OUT = f"/tmp/rustyui_guide_ref_{STRENGTH}.safetensors"
LENGTH = 17
W, H = 640, 448
prompt_text = json.load(open("/tmp/rustyui_i2v_prompt.json"))["prompt"]
tensors, meta = {}, {"size": [W, H, LENGTH], "strength": STRENGTH}


def keep(name, t):
    if isinstance(t, (list, tuple)):
        for i, v in enumerate(t):
            keep(f"{name}.{i}", v)
        return
    if hasattr(t, "unbind") and getattr(t, "is_nested", False):
        keep(name, t.unbind())
        return
    tensors[name] = t.detach().cpu().contiguous().clone()
    meta[name] = {"dtype": str(t.dtype), "shape": list(t.shape)}


orig_noise = comfy.sample.prepare_noise
orig_euler = kds.sample_euler


def prepare_noise(latent_image, seed, noise_inds=None):
    out = orig_noise(latent_image, seed, noise_inds)
    if "noise.0" not in tensors and "noise" not in tensors:
        keep("noise", out)
        meta["seed"] = seed
    return out


def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
    keep("sigmas", sigmas)
    keep("x0", x)
    t = time.time()

    def cb(d):
        i = d["i"]
        keep(f"step{i}.x", d["x"])
        keep(f"step{i}.denoised", d["denoised"])
        if callback is not None:
            callback(d)

    out = orig_euler(model, x, sigmas, extra_args=extra_args, callback=cb, disable=disable, **kw)
    torch.cuda.synchronize()
    meta["sampling_seconds"] = round(time.time() - t, 2)
    keep("sampled", out)
    return out


comfy.sample.prepare_noise = prepare_noise
kds.sample_euler = sample_euler


orig_vae_decode = comfy.sd.VAE.decode


def vae_decode(self, samples_in, *a, **k):
    out = orig_vae_decode(self, samples_in, *a, **k)
    if "vae.decode_out" not in tensors:
        keep("vae.decode_in", samples_in)
        keep("vae.decode_out", out)
    return out


comfy.sd.VAE.decode = vae_decode


orig_vae_encode = comfy.sd.VAE.encode


def vae_encode(self, pixel_samples, *a, **k):
    out = orig_vae_encode(self, pixel_samples, *a, **k)
    if "vae.encode_out" not in tensors:
        keep("vae.encode_in", pixel_samples)
        keep("vae.encode_out", out)
    return out


comfy.sd.VAE.encode = vae_encode

import comfy.samplers
orig_ksampler_sample = comfy.samplers.KSAMPLER.sample


def ksampler_sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
    keep("latent_image", latent_image)
    if denoise_mask is not None:
        keep("denoise_mask", denoise_mask)
    return orig_ksampler_sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)


comfy.samplers.KSAMPLER.sample = ksampler_sample

import comfy.ldm.lightricks.av_model as avm
orig_forward = avm.LTXAVModel.forward


def forward(self, x, timestep, context, *a, **k):
    out = orig_forward(self, x, timestep, context, *a, **k)
    if "in.context" not in tensors:
        keep("in.x", x)
        keep("in.timestep", timestep)
        keep("in.context", context)
        for name in ("denoise_mask", "audio_denoise_mask", "keyframe_idxs"):
            if k.get(name) is not None:
                keep(f"in.{name}", k[name])
        if k.get("guide_attention_entries") is not None:
            meta["in.guide_attention_entries"] = [{a: (b if isinstance(b, (int, float, list, type(None))) else str(b)) for a, b in e.items()} for e in k["guide_attention_entries"]]
        meta["in.kwargs"] = sorted(k)
        keep("out", out)
    return out


avm.LTXAVModel.forward = forward

# the first call's first guide-masked attention (block 0's attn1) and the positions it saw
import comfy.ldm.lightricks.model as ltxm
orig_guided = ltxm._attention_with_guide_mask


def guided(q, k, v, heads, guide_mask, *a, **kw):
    out = orig_guided(q, k, v, heads, guide_mask, *a, **kw)
    if "attn.out" not in tensors:
        keep("attn.q", q)
        keep("attn.k", k)
        keep("attn.v", v)
        keep("attn.noisy_mask", guide_mask.noisy_mask)
        keep("attn.tracked_mask", guide_mask.tracked_mask)
        meta["attn.guide_start"] = guide_mask.guide_start
        keep("attn.out", out)
    return out


ltxm._attention_with_guide_mask = guided

# block 0: its first fused AdaLN (the video self-attention's input) and the whole attn1 call
import comfy.quant_ops
orig_rms_adaln = comfy.quant_ops.ck.rms_adaln


def rms_adaln(x, scale, shift, *a, **kw):
    out = orig_rms_adaln(x, scale, shift, *a, **kw)
    if "adaln.out" not in tensors:
        keep("adaln.x", x)
        keep("adaln.scale", scale)
        keep("adaln.shift", shift)
        keep("adaln.out", out)
    return out


comfy.quant_ops.ck.rms_adaln = rms_adaln
orig_cross = ltxm.CrossAttention.forward


def cross(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    out = orig_cross(self, x, context=context, mask=mask, pe=pe, k_pe=k_pe, transformer_options=transformer_options)
    if isinstance(mask, ltxm.GuideAttentionMask) and "attn1.out" not in tensors:
        keep("attn1.x", x)
        keep("attn1.out", out)
    return out


ltxm.CrossAttention.forward = cross

orig_block = avm.BasicAVTransformerBlock.forward


def deep(name, obj, depth=0):
    """keep() through tuples, lists and CompressedTimestep (its __slots__)"""
    if torch.is_tensor(obj):
        keep(name, obj)
    elif isinstance(obj, (tuple, list)) and depth < 4:
        for i, v in enumerate(obj):
            deep(f"{name}.{i}", v, depth + 1)
    elif hasattr(obj, "__slots__") and depth < 4:
        for k in obj.__slots__:
            deep(f"{name}.{k}", getattr(obj, k, None), depth + 1)


def block(self, x, *a, **kw):
    if "b0.v_pe" not in tensors:
        deep("g0.in.x", x)
        for k, v in kw.items():
            if k != "transformer_options":
                deep(f"g0.in.{k}", v)
        keep("b0.v_pe", kw["v_pe"][0] if isinstance(kw["v_pe"], (list, tuple)) else kw["v_pe"])
        keep("b0.v_cross_pe", kw["v_cross_pe"][0] if isinstance(kw["v_cross_pe"], (list, tuple)) else kw["v_cross_pe"])
        keep("b0.in.0", x[0])
        out = orig_block(self, x, *a, **kw)
        keep("b0.out.0", out[0])
        keep("b0.out.1", out[1])
        return out
    return orig_block(self, x, *a, **kw)


avm.BasicAVTransformerBlock.forward = block


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
    "cond": {"class_type": "LTXVConditioning", "inputs": {"frame_rate": 30, "positive": ["guide", 0], "negative": ["guide", 1]}},
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.3-distilled.safetensors", "weight_dtype": "default"}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.3-video-vae.safetensors"}},
    "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-audio-vae.safetensors"}},
    "img": {"class_type": "LoadImage", "inputs": {"image": "rustyui-i2v-lighthouse.png"}},
    "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": LENGTH, "batch_size": 1}},
    "guide": {"class_type": "LTXVAddGuide", "inputs": {"frame_idx": 0, "strength": STRENGTH, "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "latent": ["vlat", 0], "image": ["img", 0]}},
    "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": LENGTH, "frame_rate": 30, "batch_size": 1, "audio_vae": ["avae", 0]}},
    "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["guide", 2], "audio_latent": ["alat", 0]}},
    "sched": {"class_type": "LTXVScheduler", "inputs": {"steps": 8, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["av", 0]}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "guider": {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1]}},
    "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "samp": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0], "sigmas": ["sched", 0], "latent_image": ["av", 0]}},
    "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp", 0]}},
    "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "latent": ["sep", 0]}},
    "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["crop", 2], "vae": ["vae", 0]}},
}


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    missing = sorted({n["class_type"] for n in graph.values()} - set(nodes.NODE_CLASS_MAPPINGS))
    if missing:
        sys.exit(f"nodes not available: {missing}")
    import comfy.model_management as mm
    cache_ram = min(10.0, max(2.0, mm.total_ram * 0.10 / 1024.0))
    cache_ram_inactive = min(128.0, mm.total_ram / 1024.0)
    ex = execution.PromptExecutor(Server(), cache_type=execution.CacheType.RAM_PRESSURE,
                                  cache_args={"lru": 0, "ram": cache_ram, "ram_inactive": cache_ram_inactive})
    t = time.time()
    await ex.execute_async(graph, "rustyui-guide-ref", {}, ["vdec"])
    meta["total_seconds"] = round(time.time() - t, 1)
    # the decoders' results from the executor's cache
    for node_id, name in (("vdec", "images"), ("sep", "separated"), ("crop", "cropped")):
        try:
            cached = ex.caches.outputs.get(node_id)
            val = cached.outputs if hasattr(cached, "outputs") else cached
            # LTXVCropGuides: its third output, the latent
            val = val[2] if node_id == "crop" else (val[0] if isinstance(val, (list, tuple)) else val)
            first = val[0] if isinstance(val, (list, tuple)) else val
            if isinstance(first, dict):
                for k2, v2 in first.items():
                    if torch.is_tensor(v2) or isinstance(v2, (list, tuple)) or getattr(v2, "is_nested", False):
                        keep(f"{name}.{k2}", v2)
                    else:
                        meta[f"{name}.{k2}"] = v2 if isinstance(v2, (int, float, str)) else str(type(v2))
            elif torch.is_tensor(first):
                keep(name, first)
            else:
                meta[name] = str(type(first))
        except Exception as e:
            meta[f"{name}.error"] = repr(e)[:300]


asyncio.run(main())
meta["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print(json.dumps({k: v for k, v in meta.items()}, indent=0, default=str)[:5000])
print("tensors:", len(tensors), "MB:", round(sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20, 1))
if tensors:
    save_file(tensors, OUT, metadata={"meta": json.dumps(meta, default=str)})
    print("saved", OUT)
