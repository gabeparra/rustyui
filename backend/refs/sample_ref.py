# A whole small LTX-2.3 text-to-video run inside ComfyUI, recorded for the Rust port: the graph of
# dit_ref2.py (448x640x9 at 30 fps, seed 42, LTXVScheduler 8 steps, euler, cfg 1, no guide or
# LoRAs) run to the end, then the video VAE and the audio VAE decodes. Saves the sigmas, the
# noise (video then audio, one generator), the packed latent and the denoised prediction at every
# step, the sampler's result, the decoded frames and the waveform. Read-only apart from the
# output file.
import os, asyncio, json, sys, time, torch
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

OUT = "/tmp/rustyui_sample_ref.safetensors"
LENGTH = 9
prompt_text = json.load(open("/tmp/rustyui_ltx_prompt.json"))["prompt"]
tensors, meta = {}, {"size": [448, 640, LENGTH]}


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
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.3-video-vae.safetensors"}},
    "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-audio-vae.safetensors"}},
    "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 448, "height": 640, "length": LENGTH, "batch_size": 1}},
    "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": LENGTH, "frame_rate": 30, "batch_size": 1, "audio_vae": ["avae", 0]}},
    "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
    "sched": {"class_type": "LTXVScheduler", "inputs": {"steps": 8, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["av", 0]}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "guider": {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1]}},
    "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "samp": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0], "sigmas": ["sched", 0], "latent_image": ["av", 0]}},
    "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp", 0]}},
    "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["sep", 0], "vae": ["vae", 0]}},
    "adec": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep", 1], "audio_vae": ["avae", 0]}},
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
    await ex.execute_async(graph, "rustyui-sample-ref", {}, ["vdec", "adec"])
    meta["total_seconds"] = round(time.time() - t, 1)
    # the decoders' results from the executor's cache
    for node_id, name in (("vdec", "images"), ("adec", "audio"), ("sep", "separated")):
        try:
            cached = ex.caches.outputs.get(node_id)
            val = cached.outputs if hasattr(cached, "outputs") else cached
            val = val[0] if isinstance(val, (list, tuple)) else val
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
