"""rustyui for ComfyUI: add a "Run on rustyui" node to a workflow and it runs on rustyui-backend
instead of ComfyUI's executor. See README.md."""

from .rustyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, install

install()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
