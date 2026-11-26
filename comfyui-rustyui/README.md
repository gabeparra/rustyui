# rustyui for ComfyUI

A ComfyUI custom node that runs a whole workflow on
[rustyui-backend](https://github.com/gabeparra/rustyui) instead of ComfyUI's own executor. You
keep ComfyUI's editor, your saved workflows and your models folder; the rendering happens in
rustyui, which gives the same pixels for the graphs it supports, often faster.

It is one node, **Run on rustyui** (category `rustyui`). Add it anywhere in a workflow and queue
as usual. Turn its `enabled` switch off, or bypass it, and the workflow runs in ComfyUI again.

## What you need

- ComfyUI (tested with 0.39.0).
- rustyui-backend running on the same machine, built as the
  [main README](https://github.com/gabeparra/rustyui#installing) describes. The node talks to it
  over HTTP on `127.0.0.1:8190`; rustyui-backend only answers its own machine.
- Nothing else: the node pack uses only Python's standard library and ComfyUI itself.

## Install

The node pack is the `comfyui-rustyui/` folder of the rustyui repository. Link or copy it into
ComfyUI's `custom_nodes/`, then restart ComfyUI:

```bash
git clone https://github.com/gabeparra/rustyui.git ~/rustyui
ln -s ~/rustyui/comfyui-rustyui /path/to/ComfyUI/custom_nodes/comfyui-rustyui
```

A link follows `git pull`; a copy (`cp -r`) stays as it is. ComfyUI's log says
`[rustyui]` when a workflow is sent.

Start the backend with ComfyUI's own folders, so both programs read the same models and the
results land where ComfyUI keeps its outputs:

```bash
rustyui-backend serve --models /path/to/ComfyUI/models --out /path/to/ComfyUI/output
```

Under systemd socket activation (`deploy/` in the repository) the backend starts on the node's
first request and exits when idle, so it holds no GPU memory between uses.

## Use

1. Open a workflow in ComfyUI. Add **Run on rustyui** (double-click the canvas, type "rustyui").
2. Queue. The node's `url` says where the backend is (`RUSTYUI_URL` in ComfyUI's environment
   changes the default).
3. The pictures or video show on the Run on rustyui node, and in ComfyUI's queue and assets.

What happens on Queue: ComfyUI hands the workflow to the node pack before it runs anything. The
pack sends the graph, without the Run on rustyui node, to rustyui-backend, unloads ComfyUI's own
models so the two programs do not hold the card at once, waits, and copies back what rustyui made.
ComfyUI's progress bar follows rustyui's sampler steps, and ComfyUI's Interrupt cancels the
render in rustyui too.

The saved PNGs carry the workflow, as ComfyUI's do, so a picture dropped on ComfyUI reopens it.

## Limits

- rustyui runs the nodes its README lists under
  [Supported nodes](https://github.com/gabeparra/rustyui#supported-nodes) (SDXL, LTX-2.3 and a set
  of custom-node ports). A workflow with any other node is refused before it starts, and the
  error names the node: turn the switch off to run that workflow in ComfyUI.
- Preview nodes (PreviewImage and the like) are not ported; use SaveImage or SaveVideo.
- The whole workflow runs in one place. Mixing ComfyUI nodes and rustyui nodes in one run is not
  possible: the two programs do not share tensors.
- "Queue selected output nodes" runs every output node of the workflow.
- Like ComfyUI, the node does not run again when nothing in the workflow changed; change a seed
  or a widget to render again.

## Tests

```bash
python3 -m unittest discover -s comfyui-rustyui/tests
```

They run against a stand-in for the backend and need neither ComfyUI nor a GPU.

## License

GPL-3.0, as the rest of rustyui.
