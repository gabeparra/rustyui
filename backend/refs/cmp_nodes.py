# Node outputs of one graph, ComfyUI's (refs/graph_nodes_ref.py) against rustyui's
# (RUSTYUI_DUMP_NODES), in ComfyUI's execution order: how many values differ in each.
#   cmp_nodes.py COMFY.safetensors RUSTY.safetensors
import sys, re
import torch
from safetensors import safe_open

a = safe_open(sys.argv[1], "pt")
b = safe_open(sys.argv[2], "pt")
bkeys = set(b.keys())
seen = set()
for k in a.keys():  # insertion order: the order ComfyUI ran the nodes
    m = re.match(r"^([^.]+)\.(\d+)\.(\d+)(.*)$", k)
    if not m:
        continue
    node, out, call, rest = m.groups()
    if call != "0":
        continue
    key = f"{node}.{out}{rest}"
    if key not in bkeys:
        print(f"{key:44} only in ComfyUI {tuple(a.get_slice(k).get_shape())}")
        continue
    seen.add(key)
    x, y = a.get_tensor(k), b.get_tensor(key)
    if x.shape != y.shape:
        try:
            y = y.reshape(x.shape)
        except Exception:
            print(f"{key:44} shape {tuple(x.shape)} vs rustyui's {tuple(y.shape)}")
            continue
    x, y = x.double(), y.double()
    d = (x - y).abs()
    n = int((d != 0).sum())
    mx = float(d.max()) if d.numel() else 0.0
    rel = float(d.norm() / x.norm()) if x.norm() > 0 else 0.0
    print(f"{key:44} {n:>10} of {x.numel():>10} differ, max {mx:.3e} rel {rel:.2e}{'  exact' if n == 0 else ''}")
for key in sorted(bkeys - seen):
    print(f"{key:44} only in rustyui")
