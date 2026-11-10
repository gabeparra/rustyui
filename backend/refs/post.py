# backend/refs/post.py: a deploy test. Queue an API graph on a ComfyUI-compatible server, with the SaveImage prefix changed, and wait
# for it: prints the prompt id, the seconds it took and the files the server reports.
#   python3 post.py URL GRAPH.json PREFIX
import json, sys, time, urllib.request
url, path, prefix = sys.argv[1:4]
# test renders stay in their own subfolder of the output folder
assert prefix.startswith("rustyui-tests/"), f"prefix {prefix!r} must start with rustyui-tests/"
g = json.load(open(path))
g = g.get("prompt", g)
for n in g.values():
    if n["class_type"] == "SaveImage":
        n["inputs"]["filename_prefix"] = prefix
t0 = time.time()
req = urllib.request.Request(url + "/prompt", data=json.dumps({"prompt": g, "client_id": "deploy-test"}).encode(),
                             headers={"Content-Type": "application/json"})
pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
print("prompt", pid)
while True:
    h = json.load(urllib.request.urlopen(f"{url}/history/{pid}", timeout=30))
    if pid in h:
        e = h[pid]
        st = e.get("status", {})
        print("status", st.get("status_str"), "completed", st.get("completed"), f"{time.time() - t0:.1f} s")
        for node, out in e.get("outputs", {}).items():
            for im in out.get("images", []):
                print("image", im.get("subfolder"), im.get("filename"))
        for m in st.get("messages", []):
            if m[0] in ("execution_error",):
                print("error", json.dumps(m[1])[:400])
        break
    if time.time() - t0 > 600:
        print("timed out"); break
    time.sleep(1)
