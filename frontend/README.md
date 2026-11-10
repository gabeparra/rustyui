# rustyui simple editor

The editor rustyui serves at **`/simple/`**, beside ComfyUI's own editor at `/`. It is written in
TypeScript and Vue 3, bundled by Vite into `dist/`, and compiled into the backend binary. It has two
views:

- **`/simple/`**: a form laid out as Stable Diffusion WebUI Forge lays out its own, for SDXL.
- **`/simple/#graph`**: a node editor on LiteGraph (`@comfyorg/litegraph`, the canvas ComfyUI's
  frontend draws with), with a queue, a history with thumbnails and a picture viewer.

Both talk to the backend that serves them and build ComfyUI API graphs, so everything they make
opens in ComfyUI and in ComfyUI's editor at `/`.

## Development

```bash
bun install --frozen-lockfile            # once, and again when bun.lock changes
bun run dev       # http://127.0.0.1:5173/simple/ with hot reload; API calls go to the backend on :8190
bun run build     # type-check, then bundle into dist/
node --experimental-strip-types --test tests/forge.test.ts   # the form's graph and PNG Info tests
```

After `bun run build`, build the backend again (see the root README): it compiles in whatever
`dist/` holds.

The dev server listens on 127.0.0.1 only and refuses what the backend refuses (a non-loopback
`Host`, a foreign `Origin`, a cross-site `Sec-Fetch-Site`). It drops the browser's `Origin` when it
passes a request on, since the backend would turn its port away as another origin.

## The form (`/simple/`)

The checkpoint and clip skip sit on top, then the **txt2img**, **img2img** and **PNG Info** tabs.
Each tab has the prompt and negative prompt beside **Generate**, the generation settings on the
left (with a **Lora** sub-tab), and the results with their infotext on the right.

- **Generation settings**: sampler and schedule type (shown under A1111's names), steps, CFG
  scale, width and height, seed (-1 for a random one), batch count and batch size.
- **Hires. fix** (txt2img): through a model upscaler and then resized to the target, or in latent
  space (`Latent`, `Latent (nearest-exact)`); hires steps, denoise, scale or target size, and an
  optional sampler, schedule type and CFG of its own.
- **Upscale after**: an ESRGAN-family upscaler after the last decode, then a resize to the size
  asked for.
- **img2img**: drop a picture in, or send a result there; denoising strength sets how much is
  redrawn.
- **Lora**: the backend's LoRAs, filterable. A click adds a `<lora:name:1>` tag to the prompt.
  Tags use A1111's syntax: they are taken out of the text and become a chain of `LoraLoader`
  nodes, each at its weight.
- **Prompt writer**: when the backend runs with `--writer`, a ✍️ button opens a box for a short
  idea; **Write** replaces the prompt with the writer's answer, keeping its quality and style tags.
- **Ctrl+Enter** generates, **Esc** interrupts. **Interrupt** cancels the rest of the run and stops
  the render in progress; **Skip** stops only the one in progress.

Each run is one ComfyUI API graph per batch-count entry (seeds counting up from the first), built
from the nodes rustyui runs: `CheckpointLoaderSimple`, `LoraLoader`, `CLIPSetLastLayer` for clip
skip, `CLIPTextEncode` and `ConditioningConcat` for `BREAK`, `KSampler`, `UpscaleModelLoader`,
`ImageUpscaleWithModel`, `ImageScale`, `LatentUpscale`, `VAEEncode`, `VAEDecode`, `SaveImage`.

**PNG Info** reads a picture's text chunks: ComfyUI's graph (every ComfyUI and rustyui render
carries one) or the A1111/Forge `parameters` infotext. It lists what the form can't hold or the
backend doesn't have, and from there:

- **Send to txt2img** / **Send to img2img** fill a tab with the picture's settings. Settings read
  from an A1111 or Forge picture make a different picture here: rustyui samples as ComfyUI does,
  not as A1111 does.
- Replaying fills the tab and generates with the seed kept, which gives the same picture when the
  form holds all of the graph and the backend has the same checkpoint.
- A ComfyUI PNG's own graph can also be queued exactly as it is, with its workflow, whatever the
  form could hold of it.

The form's settings are kept in the browser's local storage, per tab.

## The graph view (`/simple/#graph`)

LiteGraph's controls, as in ComfyUI:

| Keybind | Explanation |
|---|---|
| `Ctrl` + `Enter` | Queue the graph (the number box beside **Queue** queues several; seeds follow *control after generate*) |
| `Ctrl` + `S` | Save the workflow, in ComfyUI's format |
| `Ctrl` + `O` | Open a workflow, an API graph or a PNG (or drop one on the canvas) |
| `Ctrl` + `A` | Select all nodes |
| `Ctrl` + `C` / `Ctrl` + `V` | Copy and paste the selected nodes |
| `Ctrl` + `Shift` + `V` | Paste, keeping the links into the pasted nodes |
| `Delete` / `Backspace` | Delete the selected nodes |
| `Space` | Pan the canvas while held |
| `Esc` | Close the picture viewer or a menu |
| Double-click | Search nodes |
| Right-click | Add a node by category. On a node: bypass, mute, colour, remove |
| Drag from a slot | Draw a link. Let go on empty canvas to pick a node that fits |
| `Ctrl` + drag | Select several nodes |
| Wheel | Zoom |

- **Open** takes ComfyUI workflows (also what **Save** writes), ComfyUI API graphs, and PNGs from
  ComfyUI or rustyui. An API graph's seeds come in as *fixed*, so queueing it again makes the same
  picture. An A1111 or Forge PNG holds settings rather than a graph: open it in the form's PNG Info.
- **Default** puts the starting graph back, **Models** reads the backend's nodes and models again,
  **Free VRAM** unloads the backend's models.
- Only what a Save node needs is sent, and it is checked first: nodes the backend does not run
  (shown red), inputs left unconnected and choices the backend does not have are listed, and
  nothing is queued.
- The queue panel lists every prompt the backend ran, from either editor or from a script, with its
  pictures. **Open graph** puts a prompt's graph back on the canvas, **Forget** takes it off the
  list (the files stay). A picture opens large in the viewer, with its own graph a click away.
- The canvas is saved in the browser every few seconds, so a reload keeps it.

## Source

| Path | What |
|---|---|
| `src/main.ts` | picks the view from the URL |
| `src/forge/ForgeApp.vue`, `forge.css`, `FSlider.vue` | the form |
| `src/forge/graph.ts` | the form's settings as a ComfyUI API graph, LoRA tags, hires and upscale sizes, infotext |
| `src/forge/png.ts` | PNG Info: text chunks, A1111 infotext and ComfyUI graphs read back into settings |
| `src/App.vue` | the graph view: top bar, canvas, queue, picture viewer, messages, polling |
| `src/graph/registry.ts` | the backend's `/object_info` as LiteGraph node types, registered as ComfyUI registers them (link inputs as slots, values as widgets in the definition's order, *control after generate* after a seed), plus ComfyUI's editor-only nodes: Note, Reroute, PrimitiveNode |
| `src/graph/prompt.ts` | the graph as ComfyUI's API prompt (as ComfyUI's `graphToPrompt` builds it), and the check before queueing |
| `src/graph/load.ts` | opening workflows, API graphs and PNGs; the default graph |
| `src/graph/textarea.ts` | prompt boxes as real textareas over the canvas, as ComfyUI has them |
| `src/components/QueuePanel.vue` | the queue and history |
| `src/api.ts` | the backend's routes and what they answer |
| `tests/forge.test.ts` | the form's graphs, LoRA tags and PNG Info round trips |

## What it asks the backend for

- ComfyUI's API, under `/api`: `/api/object_info` (the nodes and the installed models),
  `/api/prompt`, `/api/view`, `/api/upload/image`, `/api/interrupt`, `/api/jobs/<id>/cancel`,
  `/api/free`.
- rustyui's own routes, under `/rustyui`: `/rustyui/status` (polled while rendering: the queue, the
  node running and its sampler step, whether a prompt writer is loaded), `/rustyui/history` (every
  prompt the backend has run, with the graph each one ran), `/rustyui/history/forget` and
  `/rustyui/write`.

Pictures go to the backend's output folder, named from the `SaveImage` prefix (`rustyui` unless
`--prefix` says otherwise) as `rustyui_00001_.png` and on. Each holds ComfyUI's `prompt` chunk (the
API graph). Those queued from the graph view also hold `workflow` (the editor's graph in ComfyUI's
workflow format). The form shows each result's settings as A1111-style infotext beside it.

## Limits

- Only the backend's nodes: the root README lists them. The form builds SDXL graphs only; the graph
  view also plays LTX-2.3 videos from `SaveVideo` in its queue and viewer.
- The graph view has no undo and no interrupt button (the form and the editor at `/` have one).
- Progress is polled (`/rustyui/status`), not taken from the websocket.
- ComfyUI subgraphs open in the graph view as red nodes; use the editor at `/` for them.
