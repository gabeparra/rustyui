# The order ComfyUI runs a graph's nodes in, started as the server starts (main.py), for checking
# rustyui's port of its ExecutionList (graph.rs comfy_order). Prints one line, ORDER [(id, class)...].
# Its SaveImage / SaveVideo write under output/rustyui-tests/order.
#   python order_ref.py GRAPH.json
import sys, os, json, asyncio

GRAPH = sys.argv[1]
COMFY = os.environ.get("COMFYUI_DIR", ".")
sys.argv = ["main.py"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)
import main  # noqa: E402,F401  ComfyUI's startup, without the server
import execution  # noqa: E402
import nodes  # noqa: E402
import server  # noqa: E402
from app.assets.manager import NoAssets  # noqa: E402

order = []
original = execution.get_output_data


async def recording(prompt_id, unique_id, obj, *args, **kwargs):
    order.append((unique_id, type(obj).__name__))
    return await original(prompt_id, unique_id, obj, *args, **kwargs)


execution.get_output_data = recording
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
prompt_server = server.PromptServer(loop, NoAssets(main.args))
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
graph = json.load(open(GRAPH))
graph = graph.get("prompt", graph)
outs = [k for k, n in graph.items() if n["class_type"] in ("SaveImage", "SaveVideo")]
for k in outs:
    graph[k]["inputs"]["filename_prefix"] = "rustyui-tests/order"
ex = execution.PromptExecutor(prompt_server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
ex.execute(graph, "order", {}, outs)
print("ORDER", order)
