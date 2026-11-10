# rustyui's sigma schedules against ComfyUI's, bit for bit: calculate_sigmas on the SDXL table for
# every step count from 1 to 300 (karras's pow runs through Sleef for the vectorized part from 32 on),
# and, with --powf, Sleef's powf alone on a few million inputs torch.pow computes in its vector loop.
#   python cmp_schedule.py [BACKEND] [--powf]
import sys, os, json, subprocess, tempfile
sys.argv, args = ["x"], sys.argv[1:]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch  # noqa: E402
import comfy.model_sampling as ms_mod  # noqa: E402
import comfy.samplers as samplers  # noqa: E402

backend = next((a for a in args if not a.startswith("--")), os.path.expanduser("~/rustyui/target/release/rustyui-backend"))


class Config:
    sampling_settings = {}


ms = ms_mod.ModelSamplingDiscrete(Config())
total = 0
for sched, counts in (("karras", range(1, 301)), ("simple", range(1, 301)), ("normal", range(1, 301)), ("sgm_uniform", range(1, 301))):
    out = subprocess.run([backend, "debug-schedule", sched, *map(str, counts)], capture_output=True, text=True, check=True).stdout
    ours = json.loads(out.strip().splitlines()[-1])
    bad = []
    for n in counts:
        theirs = samplers.calculate_sigmas(ms, sched, n).float().view(torch.int32).tolist()
        theirs = [t & 0xFFFFFFFF for t in theirs]
        if ours[str(n)] != theirs:
            diff = sum(1 for a, b in zip(ours[str(n)], theirs) if a != b) + abs(len(theirs) - len(ours[str(n)]))
            bad.append(f"{n} ({diff} of {len(theirs)})")
    print(f"{sched} {counts.start} to {counts.stop - 1} steps: {len(bad)} differ" + (": " + ", ".join(bad) if bad else ""))
    total += len(bad)
print("schedules that differ:", total)

if "--powf" in args:
    g = torch.Generator().manual_seed(0)
    xs = [torch.rand(1 << 20, generator=g) * 1.6 + 0.4,  # karras's bases sit in about [0.6, 1.5]
          torch.exp(torch.randn(1 << 20, generator=g) * 20),  # every magnitude
          torch.rand(1 << 20, generator=g)]
    ys = [7.0, 7.0, 2.5]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for x, y in zip(xs, ys):
            r = x ** y  # a multiple of 32 elements: all of it through Sleef
            f.write(torch.stack([x, torch.full_like(x, y), r], 1).contiguous().numpy().tobytes())
        path = f.name
    print(subprocess.run([backend, "debug-powf", path], capture_output=True, text=True).stdout)
    os.unlink(path)
