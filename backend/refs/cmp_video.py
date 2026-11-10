# Compare two MP4s frame by frame (decoded RGB) and their audio (decoded f32), and the encoded
# streams themselves (ffmpeg's per-packet hashes). Usage: cmp_video.py A.mp4 B.mp4
import subprocess, sys, json
import numpy as np

a, b = sys.argv[1], sys.argv[2]

def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", p], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"]

def frames(p, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float64)

def audio(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vn", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)

def packets(p, sel):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", sel, "-c", "copy", "-f", "framemd5", "-"], capture_output=True, text=True, check=True)
    return [l.split(",")[-1].strip() for l in r.stdout.splitlines() if not l.startswith("#")]

sa, sb = probe(a), probe(b)
va = next(s for s in sa if s["codec_type"] == "video")
vb = next(s for s in sb if s["codec_type"] == "video")
for k in ["codec_name", "width", "height", "pix_fmt", "r_frame_rate", "nb_frames", "color_space", "color_primaries", "color_transfer", "color_range"]:
    if va.get(k) != vb.get(k):
        print(f"video {k}: {va.get(k)} vs {vb.get(k)}")
w, h = va["width"], va["height"]
fa, fb = frames(a, w, h), frames(b, w, h)
print(f"video: {fa.shape[0]} and {fb.shape[0]} frames of {w}x{h}")
n = min(len(fa), len(fb))
ps = []
for i in range(n):
    mse = np.mean((fa[i] - fb[i]) ** 2)
    ps.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
same = sum(1 for p in ps if p == 99.0)
print(f"  frames identical: {same} of {n}; PSNR min {min(ps):.2f} mean {np.mean(ps):.2f} dB")
pa, pb = packets(a, "0:v"), packets(b, "0:v")
print(f"  video packets identical: {sum(x == y for x, y in zip(pa, pb))} of {max(len(pa), len(pb))}")
if any(s["codec_type"] == "audio" for s in sa) and any(s["codec_type"] == "audio" for s in sb):
    aa, ab = audio(a), audio(b)
    m = min(len(aa), len(ab))
    d = aa[:m] - ab[:m]
    snr = 99.0 if not d.any() else 10 * np.log10(np.sum(ab[:m] ** 2) / np.sum(d ** 2))
    print(f"audio: {len(aa)} and {len(ab)} samples, identical {int(np.sum(d == 0))} of {m}, SNR {snr:.2f} dB")
    qa, qb = packets(a, "0:a"), packets(b, "0:a")
    print(f"  audio packets identical: {sum(x == y for x, y in zip(qa, qb))} of {max(len(qa), len(qb))}")
