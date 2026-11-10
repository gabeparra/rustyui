# The editor definitions ComfyUI's /object_info/{node_class} gives the named nodes, from the nodes as
# the server loads them (main.py's setup, every custom node; CAPTURE_EXTRA_NODES / CAPTURE_STUBS as
# graph_nodes_ref.py takes them, for a pack that is not installed), for rustyui's
# assets/comfy_object_info.json.
#   python object_info_ref.py OUT.json CLASS...
import sys, os, json, asyncio

OUT, CLASSES = sys.argv[1], sys.argv[2:]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import nodes  # noqa: E402
import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
if os.environ.get("CAPTURE_STUBS"):
    sys.path.insert(0, os.environ["CAPTURE_STUBS"])
if os.environ.get("CAPTURE_EXTRA_NODES"):
    import folder_paths  # noqa: E402
    folder_paths.add_model_folder_path("custom_nodes", os.environ["CAPTURE_EXTRA_NODES"])
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
handler = next(r.handler for r in prompt_server.routes if getattr(r, "path", None) == "/object_info/{node_class}")


class Request:
    def __init__(self, cls):
        self.match_info = {"node_class": cls}


out = {}
for cls in CLASSES:
    resp = loop.run_until_complete(handler(Request(cls)))
    out.update(json.loads(resp.body))
    print(cls, "found" if cls in out else "MISSING")
json.dump(out, open(OUT, "w"), indent=1)
