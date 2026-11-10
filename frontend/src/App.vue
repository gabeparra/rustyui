<script setup lang="ts">
// rustyui's editor: LiteGraph's canvas with the backend's nodes, a top bar, and the queue.
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { LGraph, LGraphCanvas, LiteGraph, type LGraphNode } from '@comfyorg/litegraph'
import { api, isVideo, ui, viewUrl, type HistoryItem, type ObjectInfo, type Status } from './api'
import { comboValues, previews, registerNodes } from './graph/registry'
import { attachTextareas } from './graph/textarea'
import { graphToPrompt, titleOf } from './graph/prompt'
import { advanceSeeds, defaultGraph, isWorkflow, loadApiGraph, loadWorkflow, openFile, type Opened } from './graph/load'
import QueuePanel from './components/QueuePanel.vue'

type Picture = { filename: string; subfolder: string }

const SAVED = 'rustyui.workflow'
const PANEL = 'rustyui.panel'

// the brand icon, beside the editor's own files wherever it is served
const icon = import.meta.env.BASE_URL + 'favicon.svg'
const stage = ref<HTMLElement>()
const canvasEl = ref<HTMLCanvasElement>()
const layerEl = ref<HTMLElement>()
const fileEl = ref<HTMLInputElement>()
const status = ref<Status | null>(null)
const history = ref<HistoryItem[]>([])
const batch = ref(1)
const panelOpen = ref(read(PANEL) !== '0')
const viewing = ref<Picture | null>(null)
const toasts = reactive<{ id: number; text: string; kind: string }[]>([])

let graph: LGraph
let canvas: LGraphCanvas
let historyVersion = -1
let pollTimer = 0
let saveTimer = 0
let lastSaved = ''
let failures = 0
/** prompts queued from this editor: their pictures go on their nodes when done */
const mine = new Set<string>()
const reported = new Set<string>()

function read(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // a private window or full storage: the graph is just not remembered
  }
}

let toastId = 0
function toast(text: string, kind = 'info', ms = 4500) {
  const id = ++toastId
  toasts.push({ id, text, kind })
  setTimeout(() => {
    const i = toasts.findIndex((t) => t.id === id)
    if (i >= 0) toasts.splice(i, 1)
  }, ms)
}

// ── status, progress and the queue ───────────────────────────────────────────

const statusText = computed(() => {
  const st = status.value
  if (!st) return 'the backend is not answering'
  if (st.running) return st.executing ? `rendering · ${st.executing.class_type}` : 'rendering'
  return st.loaded ? 'ready · models loaded' : 'ready'
})
const statusClass = computed(() => (!status.value ? 'gone' : status.value.running ? 'busy' : 'up'))
const queued = computed(() => (status.value ? status.value.running + status.value.pending : 0))
const progress = computed(() => {
  const ex = status.value?.executing
  return ex && ex.max > 0 ? Math.round((100 * ex.value) / ex.max) : 0
})

/** a prompt's pictures, drawn in the nodes that saved them */
function showPictures(item: HistoryItem) {
  const byNode = new Map<string, HTMLImageElement[]>()
  for (const o of item.outputs) {
    const img = new Image()
    img.onload = () => graph.setDirtyCanvas(true, false)
    img.src = viewUrl(o)
    byNode.set(o.node, [...(byNode.get(o.node) ?? []), img])
  }
  for (const [node, imgs] of byNode) previews.set(node, imgs)
}

async function loadHistory() {
  try {
    const h = await ui('/history')
    history.value = h.items
  } catch {
    return // the backend went away; the status pill says so
  }
  for (const it of history.value) {
    if (!mine.has(it.id) || reported.has(it.id) || it.status === 'queued' || it.status === 'running') continue
    reported.add(it.id)
    const secs = typeof it.seconds === 'number' ? it.seconds.toFixed(1) : '?'
    if (it.status === 'success') {
      showPictures(it)
      toast(`#${it.number} done in ${secs} s`, 'ok', 2500)
    } else {
      toast(`#${it.number} ${it.status === 'lost' ? 'lost' : 'failed'}: ${it.error ?? ''}`, 'error', 15000)
    }
  }
}

async function poll() {
  try {
    status.value = await ui('/status')
    failures = 0
  } catch {
    // one miss is the backend restarting (it starts on demand); two, and it is gone
    if (++failures > 1) status.value = null
  }
  const st = status.value
  if (st && st.history_version !== historyVersion) {
    historyVersion = st.history_version
    await loadHistory()
  }
  graph?.setDirtyCanvas(true, false)
  pollTimer = window.setTimeout(poll, st && (st.running || st.pending) ? 600 : 2500)
}

/** the node running now, outlined, when the prompt came from this editor */
function drawRunning(ctx: CanvasRenderingContext2D) {
  const ex = status.value?.executing
  if (!ex || !mine.has(ex.prompt_id)) return
  const node = graph.getNodeById(Number(ex.node)) as LGraphNode | null
  if (!node) return
  const t = LiteGraph.NODE_TITLE_HEIGHT
  ctx.save()
  ctx.strokeStyle = '#7bc47f'
  ctx.lineWidth = 3
  ctx.strokeRect(node.pos[0] - 5, node.pos[1] - t - 5, node.size[0] + 10, node.size[1] + t + 10)
  ctx.restore()
}

// ── the graph ────────────────────────────────────────────────────────────────

function told(r: Opened, what: string) {
  if (r.missing.length) toast(`${what}: the red nodes are not something rustyui runs yet (${r.missing.join(', ')})`, 'error', 12000)
  else toast(`${what} opened`, 'ok', 2500)
}

async function loadInfo(): Promise<ObjectInfo> {
  const objectInfo: ObjectInfo = await api('/object_info')
  registerNodes(objectInfo)
  return objectInfo
}

async function refreshModels() {
  try {
    const objectInfo = await loadInfo()
    // the combos of nodes already on the canvas take the new lists (models added since)
    for (const node of graph._nodes as LGraphNode[]) {
      const def = objectInfo[node.type ?? '']
      if (!def) continue
      const all = { ...(def.input.required ?? {}), ...(def.input.optional ?? {}) }
      for (const w of (node.widgets ?? []) as any[]) {
        const spec = all[w.name]
        const values = comboValues(spec)
        if (w.type === 'combo' && values) w.options.values = values
      }
    }
    graph.setDirtyCanvas(true, true)
    toast('models and nodes read again', 'ok', 2500)
  } catch (e: any) {
    toast(`cannot read what the backend runs: ${e.message}`, 'error', 12000)
  }
}

async function queue() {
  const times = Math.max(1, Math.min(64, Math.floor(batch.value) || 1))
  for (let i = 0; i < times; i++) {
    // built again each time: the seeds move after each one, as ComfyUI's do
    const { prompt, problems } = graphToPrompt(graph)
    if (problems.length) return toast(problems.join('\n'), 'error', 15000)
    try {
      const r = await api('/prompt', {
        method: 'POST',
        body: { prompt, title: titleOf(prompt), extra_data: { extra_pnginfo: { workflow: graph.serialize() } } },
      })
      mine.add(r.prompt_id)
      advanceSeeds(graph)
    } catch (e: any) {
      return toast(e.message, 'error', 15000)
    }
  }
  toast(times > 1 ? `${times} queued` : 'queued', 'ok', 2000)
  clearTimeout(pollTimer)
  poll()
}

function saveWorkflow() {
  const blob = new Blob([JSON.stringify(graph.serialize(), null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = 'rustyui-workflow.json'
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
}

async function open(file: File) {
  try {
    previews.clear()
    told(await openFile(graph, file), file.name)
  } catch (e: any) {
    toast(`${file.name}: ${e.message}`, 'error', 10000)
  }
}

function onPick(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (f) open(f)
  input.value = ''
}

function onDrop(e: DragEvent) {
  const f = e.dataTransfer?.files?.[0]
  if (f) open(f)
}

function resetGraph() {
  previews.clear()
  defaultGraph(graph)
  toast('the default graph', 'ok', 2000)
}

async function openHistory(id: string) {
  try {
    const e = await ui(`/history/${encodeURIComponent(id)}`)
    // the editor's workflow when it sent one, else the graph that ran (tools send only that)
    const r = isWorkflow(e.workflow) ? loadWorkflow(graph, e.workflow) : loadApiGraph(graph, e.graph)
    previews.clear()
    showPictures(e)
    told(r, `#${e.number}`)
  } catch (err: any) {
    toast(err.message, 'error')
  }
}

/** a picture's own graph, as dropping the PNG would open it */
async function openPicture(o: Picture) {
  viewing.value = null
  try {
    const r = await fetch(viewUrl(o))
    if (!r.ok) throw new Error(`the backend answered ${r.status}`)
    await open(new File([await r.arrayBuffer()], o.filename, { type: 'image/png' }))
  } catch (e: any) {
    toast(`${o.filename}: ${e.message}`, 'error', 10000)
  }
}

async function forget(id: string) {
  try {
    await ui('/history/forget', { method: 'POST', body: { ids: [id] } })
    await loadHistory()
  } catch (e: any) {
    toast(e.message, 'error')
  }
}

// ── the prompt writer ─────────────────────────────────────────────────────────

const idea = ref('')
const writing = ref(false)

/** the CLIPTextEncode feeding a sampler's positive, else the first one */
function positiveNode(): LGraphNode | null {
  const nodes = graph._nodes as LGraphNode[]
  for (const n of nodes) {
    const slot = n.inputs?.find((i) => i.name === 'positive' && i.link != null)
    const link = slot ? graph.links.get(slot.link!) : undefined
    const from = link ? graph.getNodeById(link.origin_id) : null
    if (from?.type === 'CLIPTextEncode') return from
  }
  return nodes.find((n) => n.type === 'CLIPTextEncode') ?? null
}

/** The idea, with the prompt as it stands (its character and style tags stay), into that node. */
async function writePrompt() {
  const node = positiveNode()
  const widget = node?.widgets?.find((w: any) => w.name === 'text') as any
  if (!widget) return toast('there is no prompt node to write into', 'error')
  if (!idea.value.trim() && !String(widget.value ?? '').trim()) return toast('write an idea first', 'error')
  writing.value = true
  try {
    const r = await ui('/write', { method: 'POST', body: { idea: idea.value.trim(), prompt: String(widget.value ?? '') } })
    widget.value = r.prompt
    graph.setDirtyCanvas(true, true)
    toast(`written in ${r.seconds.toFixed(1)} s into "${node!.title}"`, 'ok', 3000)
  } catch (e: any) {
    toast(`the writer: ${e.message}`, 'error', 12000)
  } finally {
    writing.value = false
  }
}

async function freeVram() {
  try {
    const r = await api('/free', { method: 'POST', body: {} })
    toast(r.loaded ? 'the backend lets its models go after the current prompt' : 'nothing is loaded', 'ok')
  } catch (e: any) {
    toast(e.message, 'error')
  }
}

function togglePanel() {
  panelOpen.value = !panelOpen.value
  write(PANEL, panelOpen.value ? '1' : '0')
}

// ── wiring ───────────────────────────────────────────────────────────────────

/** the canvas at the screen's pixel density, as LiteGraph expects it (and ComfyUI sizes it) */
function resize() {
  const el = canvasEl.value
  if (!el || !canvas || !stage.value) return
  const { width, height } = stage.value.getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  el.width = Math.round(width * ratio)
  el.height = Math.round(height * ratio)
  el.getContext('2d')?.scale(ratio, ratio)
  canvas.setDirty(true, true)
  canvas.draw(true, true)
}

function onKey(e: KeyboardEvent) {
  const typing = e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement
  const mod = e.ctrlKey || e.metaKey
  if (e.key === 'Escape' && viewing.value) {
    viewing.value = null
  } else if (mod && e.key === 'Enter') {
    e.preventDefault()
    queue()
  } else if (mod && e.key.toLowerCase() === 's' && !typing) {
    e.preventDefault()
    saveWorkflow()
  } else if (mod && e.key.toLowerCase() === 'o' && !typing) {
    e.preventDefault()
    fileEl.value?.click()
  }
}

function autosave() {
  if (!graph) return
  const json = JSON.stringify(graph.serialize())
  if (json !== lastSaved) {
    lastSaved = json
    write(SAVED, json)
  }
}

let observer: ResizeObserver | null = null
let dprQuery: MediaQueryList | null = null

onMounted(async () => {
  LiteGraph.release_link_on_empty_shows_menu = true
  LiteGraph.middle_click_slot_add_default_node = true
  LiteGraph.alt_drag_do_clone_nodes = true
  graph = new LGraph()
  canvas = new LGraphCanvas(canvasEl.value!, graph)
  canvas.clear_background_color = '#1d1d1d'
  canvas.onDrawForeground = drawRunning
  attachTextareas(canvas, layerEl.value!)
  observer = new ResizeObserver(resize)
  observer.observe(stage.value!)
  // a window moved to a screen of another density
  dprQuery = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
  dprQuery.addEventListener('change', resize)
  resize()
  // for poking at from the console (and the headless tests): check() is what Queue would send
  ;(window as any).rustyui = { graph, canvas, previews, check: () => graphToPrompt(graph) }
  window.addEventListener('keydown', onKey)
  window.addEventListener('pagehide', autosave)
  try {
    await loadInfo()
  } catch (e: any) {
    toast(`cannot read what the backend runs: ${e.message}`, 'error', 15000)
  }
  // the graph left open last time (the first editor's own format starts over)
  try {
    const saved = JSON.parse(read(SAVED) ?? 'null')
    if (!isWorkflow(saved)) throw new Error('nothing saved')
    loadWorkflow(graph, saved)
  } catch {
    defaultGraph(graph)
  }
  lastSaved = JSON.stringify(graph.serialize())
  saveTimer = window.setInterval(autosave, 2000)
  poll()
})

onBeforeUnmount(() => {
  clearTimeout(pollTimer)
  clearInterval(saveTimer)
  observer?.disconnect()
  dprQuery?.removeEventListener('change', resize)
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('pagehide', autosave)
})
</script>

<template>
  <div class="app" :class="{ 'panel-open': panelOpen }">
    <header class="bar">
      <div class="brand"><img :src="icon" alt="" /><span>rustyui</span></div>
      <button id="queue-btn" class="primary" title="Queue the graph (Ctrl+Enter)" @click="queue">Queue</button>
      <input v-model.number="batch" class="batch" type="number" min="1" max="64" title="how many to queue" />
      <span class="sep"></span>
      <button title="Open a workflow, an API graph or a PNG (Ctrl+O), or drop one on the canvas" @click="fileEl?.click()">Open</button>
      <button title="Save the workflow (Ctrl+S)" @click="saveWorkflow">Save</button>
      <button title="Start again from the default graph" @click="resetGraph">Default</button>
      <button title="Read the models and nodes again" @click="refreshModels">Models</button>
      <button title="Unload the backend's models (frees the card)" @click="freeVram">Free VRAM</button>
      <template v-if="status?.writer">
        <span class="sep"></span>
        <input v-model="idea" class="idea" placeholder="an idea for the prompt writer" @keydown.enter="writePrompt" />
        <button :disabled="writing" title="An LLM on the card writes the tags into the prompt node" @click="writePrompt">{{ writing ? 'Writing…' : 'Write' }}</button>
      </template>
      <div class="spacer"></div>
      <div v-if="progress" class="progress" :title="`${progress}%`"><div :style="{ width: progress + '%' }"></div></div>
      <div id="engine" class="pill" :class="statusClass"><span class="dot"></span>{{ statusText }}</div>
      <button class="panel-btn" :title="panelOpen ? 'Hide the queue' : 'Show the queue'" @click="togglePanel">
        Queue{{ queued ? ` (${queued})` : '' }} {{ panelOpen ? '▸' : '◂' }}
      </button>
    </header>
    <main ref="stage" class="stage" @dragover.prevent @drop.prevent="onDrop">
      <canvas ref="canvasEl" class="graph" tabindex="0"></canvas>
      <div ref="layerEl" class="layer"></div>
    </main>
    <QueuePanel v-if="panelOpen" :items="history" @open="openHistory" @forget="forget" @view="(o) => (viewing = o)" />
    <input ref="fileEl" type="file" accept=".json,.png,application/json,image/png" hidden @change="onPick" />
    <div v-if="viewing" class="lightbox" @click="viewing = null">
      <figure>
        <video v-if="isVideo(viewing)" :src="viewUrl(viewing)" controls autoplay loop @click.stop />
        <img v-else :src="viewUrl(viewing)" alt="" />
        <figcaption>{{ (viewing.subfolder ? viewing.subfolder + '/' : '') + viewing.filename }}</figcaption>
      </figure>
      <div class="lacts" @click.stop>
        <button @click="openPicture(viewing)">Open its graph</button>
        <a class="button" :href="viewUrl(viewing)" target="_blank" rel="noopener">Full size</a>
      </div>
    </div>
    <div class="toasts">
      <div v-for="t in toasts" :key="t.id" class="toast" :class="t.kind">{{ t.text }}</div>
    </div>
  </div>
</template>
