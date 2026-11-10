# ComfyUI's LTXVAudioVAEDecode on a seeded audio latent (1, 8, 14, 16), loaded by its own nodes, with
# every stage kept: the denormalized latent, the decoder's mel, the vocoder's 16 kHz waveform and its
# first layers, the BWE's mel, residual and skip, and the 48 kHz result. For rustyui's port.
# Read-only apart from the output file.
import os, sys
sys.argv = ["x"]
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFYUI_DIR)
import asyncio
import torch
from safetensors.torch import save_file
import nodes

OUT = "/tmp/rustyui_audio_ref.safetensors"
t = {}


def keep(name, x):
    if torch.is_tensor(x) and name not in t:
        t[name] = x.detach().float().cpu().contiguous().clone()


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    vae = nodes.NODE_CLASS_MAPPINGS["LTXVAudioVAELoader"].execute("ltx-2.3-audio-vae.safetensors")[0]
    m = vae.first_stage_model
    hooks = []

    def hook(name):
        def f(mod, args, out):
            keep(name + ".in", args[0])
            keep(name, out[0] if isinstance(out, tuple) else out)
        return f
    for name, mod in [
        ("decoder", m.autoencoder.decoder), ("decoder.conv_in", m.autoencoder.decoder.conv_in),
        ("decoder.mid1", m.autoencoder.decoder.mid.block_1), ("decoder.up2.upsample", m.autoencoder.decoder.up[2].upsample),
        ("voc", m.vocoder.vocoder), ("voc.conv_pre", m.vocoder.vocoder.conv_pre), ("voc.ups0", m.vocoder.vocoder.ups[0]),
        ("voc.res0", m.vocoder.vocoder.resblocks[0]), ("voc.res0.act0", m.vocoder.vocoder.resblocks[0].acts1[0]),
        ("voc.res0.act0.up", m.vocoder.vocoder.resblocks[0].acts1[0].upsample),
        ("voc.res0.act0.snake", m.vocoder.vocoder.resblocks[0].acts1[0].act),
        ("voc.res0.act0.down", m.vocoder.vocoder.resblocks[0].acts1[0].downsample),
        ("voc.act_post", m.vocoder.vocoder.act_post), ("voc.conv_post", m.vocoder.vocoder.conv_post),
        ("bwe", m.vocoder.bwe_generator), ("bwe.skip", m.vocoder.resampler),
    ]:
        hooks.append(mod.register_forward_hook(hook(name)))
    orig_mel = m.vocoder.mel_stft.mel_spectrogram

    def mel(y):
        out = orig_mel(y)
        keep("bwe.mel_in", y)
        keep("bwe.log_mel", out[0])
        keep("bwe.magnitude", out[1])
        return out
    m.vocoder.mel_stft.mel_spectrogram = mel
    g = torch.Generator().manual_seed(9)
    latent = torch.randn(1, 8, 14, 16, generator=g)
    keep("latent", latent)
    with torch.inference_mode():
        audio = nodes.NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"].execute({"samples": latent}, vae)[0]
    keep("waveform", audio["waveform"])
    print("waveform", tuple(audio["waveform"].shape), audio["sample_rate"], "dtype", next(m.parameters()).dtype)


asyncio.run(main())
save_file(t, OUT)
print("saved", OUT)
for k in sorted(t):
    print(k, tuple(t[k].shape))
