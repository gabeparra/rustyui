"""The two nodes, and the prompt handler that connects them.

"Run on rustyui" is the node people add. When a workflow holding one is queued, the handler
below replaces the whole workflow with a single "rustyui run" node, under the same node id, that
carries the rest of the graph. ComfyUI then runs only that node: it sends the graph to
rustyui-backend, waits, and shows what came back on "Run on rustyui". With the node disabled (or
bypassed or muted) the workflow runs in ComfyUI as usual."""

import json
import logging
import os

from . import client

DEFAULT_URL = os.environ.get("RUSTYUI_URL", "http://127.0.0.1:8190")
MARK = "RustyUIBackend"
RUN = "RustyUIRun"


class RustyUIBackend:
    DESCRIPTION = (
        "Runs this whole workflow on rustyui-backend instead of ComfyUI. Add one anywhere and queue as "
        "usual; the pictures and videos it makes show here and land in ComfyUI's output folder. "
        "Turn it off to run the workflow in ComfyUI again."
    )
    CATEGORY = "rustyui"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": DEFAULT_URL, "tooltip": "where rustyui-backend serve listens"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "off: the workflow runs in ComfyUI"}),
                "timeout": ("INT", {"default": 3600, "min": 10, "max": 86400, "tooltip": "seconds before giving up"}),
            }
        }

    def run(self, **_):
        # only reached when the prompt handler did not run (it always rewrites or removes this node)
        raise RuntimeError("rustyui's prompt handler is not installed; restart ComfyUI")


class RustyUIRun:
    DESCRIPTION = "Internal: what a workflow with 'Run on rustyui' becomes when it is queued. Not meant to be added by hand."
    CATEGORY = "rustyui/internal"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": DEFAULT_URL}),
                "graph": ("STRING", {"multiline": True, "default": "{}"}),
                "timeout": ("INT", {"default": 3600, "min": 10, "max": 86400}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, url, graph, timeout, extra_pnginfo=None):
        import comfy.model_management as mm
        import folder_paths
        from comfy.utils import ProgressBar

        # both programs share the card: let ComfyUI's models go before rustyui loads its own
        mm.unload_all_models()
        mm.soft_empty_cache()
        pid = client.queue(url, json.loads(graph), extra_pnginfo)
        bar = ProgressBar(1)
        entry = client.wait(url, pid, timeout, mm.processing_interrupted, lambda v, total: bar.update_absolute(v, total))
        if entry is None:
            mm.throw_exception_if_processing_interrupted()
            raise client.RustyError("interrupted")
        files, animated = client.outputs(entry)
        out, tmp = folder_paths.get_output_directory(), folder_paths.get_temp_directory()
        shown = [client.place(client.fetch(url, item), item, out, tmp) for item in files]
        ui = {"images": shown}
        if animated:
            ui["animated"] = [True]
        return {"ui": ui}


def _scalar(inputs, name, default):
    v = inputs.get(name, default)
    return default if isinstance(v, (list, dict)) or v is None else v


def on_prompt(json_data):
    """ComfyUI's on_prompt hook: a workflow with an enabled 'Run on rustyui' becomes one RustyUIRun."""
    prompt = json_data.get("prompt")
    if not isinstance(prompt, dict):
        return json_data
    marks = [k for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == MARK]
    if not marks:
        return json_data
    rest = {k: v for k, v in prompt.items() if k not in marks}
    inputs = prompt[marks[0]].get("inputs") or {}
    if not _scalar(inputs, "enabled", True):
        json_data["prompt"] = rest
        return json_data
    run = {
        "url": str(_scalar(inputs, "url", DEFAULT_URL)),
        "graph": json.dumps(rest),
        "timeout": int(_scalar(inputs, "timeout", 3600)),
    }
    json_data["prompt"] = {marks[0]: {"class_type": RUN, "inputs": run, "_meta": {"title": "Run on rustyui"}}}
    # "queue selected outputs" names nodes that are no longer in this prompt; rustyui runs all of them
    json_data.pop("partial_execution_targets", None)
    logging.info("[rustyui] sending the workflow (%d nodes) to %s", len(rest), run["url"])
    return json_data


def install():
    from server import PromptServer

    PromptServer.instance.add_on_prompt_handler(on_prompt)


NODE_CLASS_MAPPINGS = {MARK: RustyUIBackend, RUN: RustyUIRun}
NODE_DISPLAY_NAME_MAPPINGS = {MARK: "Run on rustyui", RUN: "rustyui run (internal)"}
