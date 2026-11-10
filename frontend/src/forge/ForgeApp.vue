<script setup lang="ts">
// rustyui's simple editor, laid out as Stable Diffusion WebUI Forge lays out its own: the
// checkpoint above, txt2img / img2img / PNG Info, the prompts beside a big Generate, the
// generation settings on the left and the gallery with its infotext on the right. Each run is a
// ComfyUI graph (graph.ts) the backend renders as ComfyUI would.
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api, ui, viewUrl, BASE, type HistoryItem, type ObjectInfo, type Status } from '../api'
import FSlider from './FSlider.vue'
import { buildGraph, hiresSize, infotext, LATENT_UPSCALERS, samplerLabel, schedulerLabel, upscaleSize, type Settings } from './graph'
import { fromParameters, pngText, readComfyGraph, unrunnable, type GraphReading } from './png'
import './forge.css'

// the defaults: a common SDXL setup with a hires pass
const DEFAULTS: Settings = {
  checkpoint: '', clipSkip: 1, prompt: '', negative: '', sampler: 'euler_ancestral', scheduler: 'normal', steps: 28, cfg: 4.5,
  width: 832, height: 1216, seed: -1, batchCount: 1, batchSize: 1, hires: true, hrUpscaler: '4x_foolhardy_Remacri.pth', hrSteps: 14,
  hrDenoise: 0.4, hrScale: 1.5, hrResizeX: 0, hrResizeY: 0, hrResize: 'lanczos', hrSampler: '', hrScheduler: '', hrCfg: 0,
  up: false, upModel: '4x_foolhardy_Remacri.pth', upScale: 1.5, upResizeX: 0, upResizeY: 0, upResize: 'lanczos', denoise: 0.6,
}
const STORE = 'rustyui-forge-v1'
function stored(key: 't2i' | 'i2i'): Settings {
  try {
    const all = JSON.parse(localStorage.getItem(STORE) ?? '{}')
    return { ...DEFAULTS, ...(all[key] ?? {}), initImage: undefined }
  } catch {
    return { ...DEFAULTS }
  }
}

const tab = ref<'txt2img' | 'img2img' | 'pnginfo'>('txt2img')
const sub = ref<'generation' | 'lora'>('generation')
const t2i = reactive<Settings>(stored('t2i'))
const i2i = reactive<Settings>({ ...stored('i2i'), hires: false })
const cur = computed(() => (tab.value === 'img2img' ? i2i : t2i))
watch([t2i, i2i], () => {
  try {
    localStorage.setItem(STORE, JSON.stringify({ t2i, i2i: { ...i2i, initImage: undefined } }))
  } catch {
    // a private window: settings last for this visit
  }
}, { deep: true })

// ── what the backend has ─────────────────────────────────────────────────────────
const info = ref<ObjectInfo | null>(null)
const status = ref<Status | null>(null)
const choices = (node: string, input: string): string[] => {
  const spec = info.value?.[node]?.input?.required?.[input] ?? info.value?.[node]?.input?.optional?.[input]
  const c = spec?.[0]
  return Array.isArray(c) ? (c as string[]) : Array.isArray(spec?.[1]?.options) ? spec![1]!.options : []
}
const checkpoints = computed(() => choices('CheckpointLoaderSimple', 'ckpt_name'))
const loras = computed(() => choices('LoraLoader', 'lora_name'))
const samplers = computed(() => choices('KSampler', 'sampler_name'))
const schedulers = computed(() => choices('KSampler', 'scheduler'))
const modelUpscalers = computed(() => choices('UpscaleModelLoader', 'model_name'))
const upscalers = computed(() => [...Object.keys(LATENT_UPSCALERS), ...modelUpscalers.value])
const resizeMethods = computed(() => choices('ImageScale', 'upscale_method'))
const prefix = computed(() => (info.value?.SaveImage?.input?.required?.filename_prefix?.[1] as any)?.default ?? 'rustyui')

async function refresh() {
  try {
    info.value = await api('/object_info')
    for (const s of [t2i, i2i]) if (!checkpoints.value.includes(s.checkpoint)) s.checkpoint = checkpoints.value[0] ?? ''
  } catch (e: any) {
    toast(`the backend: ${e.message}`, 'error')
  }
}

// ── toasts ───────────────────────────────────────────────────────────────────────
const toasts = ref<{ id: number; text: string; kind: string }[]>([])
let toastId = 0
function toast(text: string, kind = 'info', ms = 6000) {
  const id = ++toastId
  toasts.value.push({ id, text, kind })
  setTimeout(() => (toasts.value = toasts.value.filter((t) => t.id !== id)), ms)
}

// ── generating ───────────────────────────────────────────────────────────────────
interface Result {
  url: string
  filename: string
  subfolder: string
  info: string
  settings: Settings
  seed: number
}
const results = ref<Result[]>([])
const selected = ref(0)
const running = ref(false)
const done = ref(0)
const total = ref(0)
const lastSeed = ref<number | null>(null)
let jobs: { id: string; seed: number; settings: Settings; info: string }[] = []
const clientId = `forge-${Math.random().toString(36).slice(2)}`

const progress = computed(() => {
  const ex = status.value?.executing
  const step = ex && ex.max ? ex.value / ex.max : 0
  return total.value ? Math.min(1, (done.value + step) / total.value) : 0
})
const progressText = computed(() => {
  const ex = status.value?.executing
  const part = total.value > 1 ? ` (${Math.min(done.value + 1, total.value)}/${total.value})` : ''
  return ex && ex.max ? `${ex.class_type} ${ex.value}/${ex.max}${part}` : `waiting${part}`
})
const hrTarget = computed(() => hiresSize(t2i))
const upFrom = computed(() => (cur.value.hires && tab.value === 'txt2img' ? hrTarget.value : [cur.value.width, cur.value.height]))
const upTarget = computed(() => upscaleSize({ ...cur.value, hires: cur.value.hires && tab.value === 'txt2img' }))

async function generate() {
  if (running.value) return
  const s = cur.value
  if (tab.value === 'img2img' && !s.initImage) return toast('drop a picture into img2img first', 'error')
  const base = s.seed < 0 ? Math.floor(Math.random() * 4294967295) : s.seed
  const snapshot = JSON.parse(JSON.stringify(s)) as Settings
  const graphs: { seed: number; graph: unknown }[] = []
  try {
    for (let i = 0; i < Math.max(1, s.batchCount); i++) graphs.push({ seed: base + i, graph: buildGraph(snapshot, base + i, loras.value, prefix.value) })
  } catch (e: any) {
    return toast(e.message, 'error')
  }
  results.value = []
  selected.value = 0
  done.value = 0
  total.value = graphs.length
  running.value = true
  lastSeed.value = base
  jobs = []
  try {
    for (const g of graphs) {
      const r = await api('/prompt', { method: 'POST', body: { prompt: g.graph, client_id: clientId } })
      jobs.push({ id: r.prompt_id, seed: g.seed, settings: snapshot, info: infotext(snapshot, g.seed) })
    }
  } catch (e: any) {
    toast(`queueing: ${e.message}`, 'error', 12000)
  }
  await follow()
}

/** A ComfyUI PNG's own graph queued as it is, byte for byte (its seeds may be past what a JS number
 * holds), with the workflow it carried; the result shows in the tab the graph belongs to */
async function replayExactly() {
  const p = pngInfo.value
  if (running.value || !p?.graph) return
  const extra = p.workflow ? `,"extra_data":{"extra_pnginfo":{"workflow":${p.workflow}}}` : ''
  const body = `{"prompt":${p.graph},"client_id":${JSON.stringify(clientId)}${extra}}`
  const settings = { ...DEFAULTS, ...p.settings } as Settings
  tab.value = p.tab
  results.value = []
  selected.value = 0
  done.value = 0
  total.value = 1
  running.value = true
  jobs = []
  try {
    const r = await api('/prompt', { method: 'POST', body })
    jobs.push({ id: r.prompt_id, seed: settings.seed, settings, info: p.text })
  } catch (e: any) {
    toast(`queueing: ${e.message}`, 'error', 12000)
  }
  await follow()
}

/** PNG Info's settings into the tab the picture belongs to, then Generate: the same picture when the
 * form carries all of the graph (and the seed is kept) */
function replay() {
  const p = pngInfo.value
  if (!p || running.value) return
  fill(p, p.tab)
  // without the picture's own model it would only be another picture with the tab's
  const target = p.tab === 'img2img' ? i2i : t2i
  const own = listed(p.settings.checkpoint, checkpoints.value)
  if (!own || target.checkpoint !== own) return toast(`Not replayed: this picture's model isn't one the form can load here, so it would be another picture with ${target.checkpoint || 'no checkpoint'}.`, 'error', 15000)
  generate()
}

/** the queued jobs through the backend's status and history until each has ended */
async function follow() {
  const reported = new Set<string>()
  while (jobs.length && reported.size < jobs.length) {
    try {
      status.value = await ui('/status')
      const h: { items: HistoryItem[] } = await ui('/history')
      for (const j of jobs) {
        const it = h.items.find((x) => x.id === j.id)
        if (!it || reported.has(j.id) || it.status === 'queued' || it.status === 'running') continue
        reported.add(j.id)
        done.value = reported.size
        if (it.status === 'success') {
          for (const o of it.outputs) {
            results.value.push({ url: viewUrl(o), filename: o.filename, subfolder: o.subfolder, info: j.info, settings: j.settings, seed: j.seed })
          }
        } else {
          toast(`${it.status}: ${it.error ?? ''}`, 'error', 15000)
        }
      }
    } catch {
      // the backend went away for a moment; keep asking
    }
    if (reported.size < jobs.length) await new Promise((r) => setTimeout(r, 500))
  }
  running.value = false
  status.value = null
}

async function interrupt() {
  // the rest of this run, then the one running
  for (const j of jobs.slice(done.value + 1)) api(`/jobs/${j.id}/cancel`, { method: 'POST', body: {} }).catch(() => {})
  await api('/interrupt', { method: 'POST', body: {} }).catch(() => {})
}

async function skip() {
  await api('/interrupt', { method: 'POST', body: {} }).catch(() => {})
}

function onKey(e: KeyboardEvent) {
  if (tab.value === 'pnginfo') return
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault()
    generate()
  } else if (e.key === 'Escape' && running.value) {
    interrupt()
  }
}

// ── the prompt tools ─────────────────────────────────────────────────────────────
function swapSize() {
  const s = cur.value
  ;[s.width, s.height] = [s.height, s.width]
}
function addLora(file: string) {
  const name = file.replace(/\.(safetensors|pt|ckpt)$/i, '')
  const s = cur.value
  s.prompt = `${s.prompt.trim()}${s.prompt.trim() ? ', ' : ''}<lora:${name}:1>`
  toast(`added ${name}`, 'ok', 2000)
}
const loraFilter = ref('')
const loraList = computed(() => {
  const q = loraFilter.value.trim().toLowerCase()
  return loras.value.filter((l) => !q || l.toLowerCase().includes(q))
})
function reuseLast() {
  const r = results.value[selected.value]
  if (!r) return toast('nothing generated yet', 'error')
  Object.assign(cur.value, { ...r.settings, seed: r.seed, initImage: cur.value.initImage })
}

const writing = ref(false)
const idea = ref('')
const showIdea = ref(false)
async function writePrompt() {
  const s = cur.value
  if (!idea.value.trim() && !s.prompt.trim()) return toast('write an idea first', 'error')
  writing.value = true
  try {
    const r = await ui('/write', { method: 'POST', body: { idea: idea.value.trim(), prompt: s.prompt } })
    s.prompt = r.prompt
    toast(`written in ${r.seconds.toFixed(1)} s`, 'ok', 3000)
  } catch (e: any) {
    toast(`the writer: ${e.message}`, 'error', 12000)
  } finally {
    writing.value = false
  }
}

// ── img2img's picture ────────────────────────────────────────────────────────────
const initPreview = ref('')
async function upload(file: Blob, name: string) {
  const form = new FormData()
  form.append('image', file, name)
  form.append('overwrite', 'true')
  const r = await fetch(BASE + '/api/upload/image', { method: 'POST', body: form })
  if (!r.ok) throw new Error(`the upload: ${r.status} ${await r.text()}`)
  const j = await r.json()
  i2i.initImage = j.subfolder ? `${j.subfolder}/${j.name}` : j.name
  initPreview.value = URL.createObjectURL(file)
}
async function pickInit(files: FileList | null | undefined) {
  const f = files?.[0]
  if (!f) return
  try {
    await upload(f, f.name)
  } catch (e: any) {
    toast(e.message, 'error')
  }
}
async function sendToImg2img() {
  const r = results.value[selected.value]
  if (!r) return
  try {
    const blob = await (await fetch(r.url)).blob()
    await upload(blob, `forge-${r.filename}`)
    Object.assign(i2i, { ...r.settings, seed: r.seed, hires: false, initImage: i2i.initImage })
    tab.value = 'img2img'
  } catch (e: any) {
    toast(e.message, 'error')
  }
}

// ── PNG Info ─────────────────────────────────────────────────────────────────────
interface PngInfo extends GraphReading {
  text: string
  kind: string
  /** what stops this backend running the picture's own graph */
  unrunnable: string[]
  /** a ComfyUI PNG's graph and workflow, as the PNG has them */
  graph?: string
  workflow?: string
}
const pngInfo = ref<PngInfo | null>(null)
const pngPreview = ref('')
/** a PNG's settings: a ComfyUI render's graph is what made it (a PNG that carries A1111
 * parameters beside it says less in those), else A1111 / Forge / Civitai's parameters */
async function readInfo(f: Blob): Promise<PngInfo> {
  const chunks = await pngText(await f.arrayBuffer())
  if (chunks.prompt) {
    const r = readComfyGraph(chunks.prompt)
    const text = infotext({ ...DEFAULTS, ...r.settings } as Settings, r.settings.seed ?? 0)
    let workflow: string | undefined = chunks.workflow
    try {
      if (workflow) JSON.parse(workflow)
    } catch {
      workflow = undefined
    }
    return { ...r, text, kind: 'ComfyUI graph', unrunnable: info.value ? unrunnable(chunks.prompt, info.value) : [], graph: chunks.prompt, workflow }
  }
  if (chunks.parameters) {
    const kind = 'A1111 / Forge parameters: rustyui samples as ComfyUI does, so the same settings make another picture'
    return { ...fromParameters(chunks.parameters), text: chunks.parameters, kind, unrunnable: [] }
  }
  const text = Object.entries(chunks).map(([k, v]) => `${k}: ${v}`).join('\n') || 'no text in this picture'
  return { settings: {}, leftOut: [], tab: 'txt2img', text, kind: 'no generation settings', unrunnable: [] }
}
async function readPng(files: FileList | null | undefined) {
  const f = files?.[0]
  if (!f) return
  try {
    pngInfo.value = await readInfo(f)
    pngPreview.value = URL.createObjectURL(f)
  } catch (e: any) {
    toast(e.message, 'error')
  }
}
/** a file this backend lists under `name`, matched as Forge matches them (case and extension aside) */
function listed(name: string | undefined, list: string[]): string | undefined {
  if (!name) return undefined
  const bare = (x: string) => x.toLowerCase().replace(/\.(safetensors|ckpt|pth|pt|bin)$/, '')
  return list.includes(name) ? name : list.find((x) => bare(x) === bare(name))
}
/** A PNG's settings into a tab: what this backend doesn't have stays as the tab has it, and the
 * toast says what the form left out */
function fill(reading: GraphReading, to: 'txt2img' | 'img2img') {
  const target = to === 'img2img' ? i2i : t2i
  const s = { ...reading.settings }
  const notes = [...reading.leftOut]
  if (to === 'txt2img' && s.initImage) {
    notes.push(`img2img's picture ${s.initImage}`)
    delete s.initImage
  }
  const check = (key: keyof Settings, list: string[], what: string) => {
    const v = s[key] as string | undefined
    if (!v) return
    const found = listed(v, list)
    if (found) (s as any)[key] = found
    else {
      notes.push(`the ${what} ${v}, which this backend doesn't have`)
      delete s[key]
    }
  }
  check('checkpoint', checkpoints.value, 'checkpoint')
  check('sampler', samplers.value, 'sampler')
  check('scheduler', schedulers.value, 'schedule type')
  check('hrSampler', samplers.value, 'hires sampler')
  check('hrScheduler', schedulers.value, 'hires schedule type')
  check('hrUpscaler', upscalers.value, 'hires upscaler')
  check('upModel', modelUpscalers.value, 'upscaler')
  check('hrResize', resizeMethods.value, 'hires resize method')
  check('upResize', resizeMethods.value, 'resize method')
  if (s.initImage && !listed(s.initImage, choices('LoadImage', 'image'))) {
    notes.push(`the picture ${s.initImage}, which isn't in this backend's input folder`)
    delete s.initImage
  }
  Object.assign(target, s)
  if (to === 'img2img') {
    target.hires = false
    if (s.initImage) initPreview.value = `${BASE}/api/view?filename=${encodeURIComponent(s.initImage)}&type=input`
  }
  tab.value = to
  if (!s.checkpoint) notes.push(`a checkpoint (it stays ${target.checkpoint || 'unset'})`)
  if (notes.length) toast(`Sent to ${to}, but the form has no place for: ${notes.join('; ')}.`, 'warn', 20000)
  return notes
}
function sendInfo(to: 'txt2img' | 'img2img') {
  if (pngInfo.value) fill(pngInfo.value, to)
}
/** a picture dropped on txt2img or img2img (not on img2img's own picture) fills that tab from it, as
 * Forge does with a picture dropped on the prompt */
async function dropOnTab(e: DragEvent) {
  e.preventDefault()
  const f = e.dataTransfer?.files?.[0]
  if (!f || tab.value === 'pnginfo') return
  try {
    const r = await readInfo(f)
    if (!Object.keys(r.settings).length) return toast(`${f.name}: ${r.leftOut[0] ?? r.kind}`, 'error')
    pngInfo.value = r
    pngPreview.value = URL.createObjectURL(f)
    if (!fill(r, tab.value).length) toast(`filled from ${f.name}`, 'ok', 3000)
  } catch (err: any) {
    toast(`${f.name}: ${err.message}`, 'error')
  }
}

function drop(e: DragEvent, then: (f: FileList | null | undefined) => void) {
  e.preventDefault()
  then(e.dataTransfer?.files)
}

let poll = 0
async function idlePoll() {
  if (!running.value) {
    try {
      status.value = await ui('/status')
    } catch {
      status.value = null
    }
  }
  poll = window.setTimeout(idlePoll, 5000)
}
onMounted(() => {
  refresh()
  idlePoll()
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  clearTimeout(poll)
  window.removeEventListener('keydown', onKey)
})
const shown = computed(() => results.value[selected.value])
/** rustyui's first editor, the node graph, which main.ts opens for #graph */
function openGraph() {
  location.hash = 'graph'
  location.reload()
}
</script>

<template>
  <div class="forge">
    <header class="quick">
      <label class="quick-ckpt">
        <span>Stable Diffusion checkpoint</span>
        <div class="with-tool">
          <select v-model="cur.checkpoint">
            <option v-for="c in checkpoints" :key="c" :value="c">{{ c }}</option>
          </select>
          <button class="tool" title="Refresh the lists" @click="refresh">🔄</button>
        </div>
      </label>
      <div class="quick-skip"><FSlider label="Clip skip" :min="1" :max="4" :step="1" v-model="cur.clipSkip" /></div>
      <div class="quick-links">
        <a :href="BASE + '/'" title="ComfyUI's node editor">Node editor</a>
        <a href="#graph" title="rustyui's first editor" @click.prevent="openGraph">Graph</a>
      </div>
    </header>

    <nav class="tabs">
      <button :class="{ on: tab === 'txt2img' }" @click="tab = 'txt2img'">txt2img</button>
      <button :class="{ on: tab === 'img2img' }" @click="tab = 'img2img'">img2img</button>
      <button :class="{ on: tab === 'pnginfo' }" @click="tab = 'pnginfo'">PNG Info</button>
    </nav>

    <section v-if="tab !== 'pnginfo'" class="gen" @dragover.prevent @drop="dropOnTab">
      <div class="toprow">
        <div class="prompts">
          <textarea v-model="cur.prompt" rows="3" :placeholder="'Prompt\n(Press Ctrl+Enter to generate, Esc to interrupt)'"></textarea>
          <textarea v-model="cur.negative" rows="3" placeholder="Negative prompt"></textarea>
        </div>
        <div class="actions">
          <div class="generate-box">
            <template v-if="running">
              <button class="interrupt" @click="interrupt">Interrupt</button>
              <button class="skip" @click="skip">Skip</button>
            </template>
            <button v-else class="generate" @click="generate">Generate</button>
          </div>
          <div class="tools">
            <button class="tool" title="Read the settings back from the picture shown" @click="reuseLast">↙️</button>
            <button class="tool" title="Clear the prompt" @click="cur.prompt = ''">🗑️</button>
            <button v-if="status?.writer" class="tool" title="The prompt writer" @click="showIdea = !showIdea">✍️</button>
          </div>
          <div v-if="showIdea" class="idea">
            <input v-model="idea" placeholder="an idea for the writer" @keydown.enter="writePrompt" />
            <button :disabled="writing" @click="writePrompt">{{ writing ? 'Writing…' : 'Write' }}</button>
          </div>
        </div>
      </div>

      <nav class="subtabs">
        <button :class="{ on: sub === 'generation' }" @click="sub = 'generation'">Generation</button>
        <button :class="{ on: sub === 'lora' }" @click="sub = 'lora'">Lora</button>
      </nav>

      <div class="cols">
        <div v-if="sub === 'generation'" class="settings">
          <div v-if="tab === 'img2img'" class="block drop" @dragover.prevent @drop.stop="drop($event, pickInit)">
            <img v-if="initPreview" :src="initPreview" alt="" />
            <label class="drop-hint">
              {{ initPreview ? 'Drop another picture, or click' : 'Drop a picture here, or click' }}
              <input type="file" accept="image/*" @change="pickInit(($event.target as HTMLInputElement).files)" />
            </label>
          </div>
          <div class="block">
            <div class="row">
              <label class="field">
                <span>Sampling method</span>
                <select v-model="cur.sampler"><option v-for="s in samplers" :key="s" :value="s">{{ samplerLabel(s) }}</option></select>
              </label>
              <label class="field">
                <span>Schedule type</span>
                <select v-model="cur.scheduler"><option v-for="s in schedulers" :key="s" :value="s">{{ schedulerLabel(s) }}</option></select>
              </label>
            </div>
            <FSlider label="Sampling steps" :min="1" :max="150" :step="1" v-model="cur.steps" />
          </div>
          <div v-if="tab === 'txt2img'" class="block accordion" :class="{ open: t2i.hires }">
            <label class="acc-head">
              <input type="checkbox" v-model="t2i.hires" />
              <span>Hires. fix</span>
              <span v-if="t2i.hires" class="acc-note">from {{ t2i.width }}x{{ t2i.height }} to {{ hrTarget[0] }}x{{ hrTarget[1] }}</span>
            </label>
            <div v-if="t2i.hires" class="acc-body">
              <div class="row">
                <label class="field">
                  <span>Upscaler</span>
                  <select v-model="t2i.hrUpscaler"><option v-for="u in upscalers" :key="u" :value="u">{{ u }}</option></select>
                </label>
                <FSlider label="Hires steps" :min="0" :max="150" :step="1" v-model="t2i.hrSteps" />
                <FSlider label="Denoising strength" :min="0" :max="1" :step="0.01" v-model="t2i.hrDenoise" />
              </div>
              <div class="row">
                <FSlider label="Upscale by" :min="1" :max="4" :step="0.05" v-model="t2i.hrScale" />
                <FSlider label="Resize width to" :min="0" :max="4096" :step="8" v-model="t2i.hrResizeX" />
                <FSlider label="Resize height to" :min="0" :max="4096" :step="8" v-model="t2i.hrResizeY" />
              </div>
              <div class="row">
                <FSlider label="Hires CFG Scale (0: the same)" :min="0" :max="30" :step="0.1" v-model="t2i.hrCfg" />
              </div>
              <div class="row">
                <label class="field">
                  <span>Hires sampling method</span>
                  <select v-model="t2i.hrSampler">
                    <option value="">Use same sampler</option>
                    <option v-for="s in samplers" :key="s" :value="s">{{ samplerLabel(s) }}</option>
                  </select>
                </label>
                <label class="field">
                  <span>Hires schedule type</span>
                  <select v-model="t2i.hrScheduler">
                    <option value="">Use same scheduler</option>
                    <option v-for="s in schedulers" :key="s" :value="s">{{ schedulerLabel(s) }}</option>
                  </select>
                </label>
                <label v-if="!(t2i.hrUpscaler in LATENT_UPSCALERS)" class="field">
                  <span>Resize method</span>
                  <select v-model="t2i.hrResize"><option v-for="m in resizeMethods" :key="m" :value="m">{{ m }}</option></select>
                </label>
              </div>
            </div>
          </div>
          <div class="block accordion" :class="{ open: cur.up }">
            <label class="acc-head">
              <input type="checkbox" v-model="cur.up" />
              <span>Upscale after</span>
              <span v-if="cur.up" class="acc-note">from {{ upFrom[0] }}x{{ upFrom[1] }} to {{ upTarget[0] }}x{{ upTarget[1] }}</span>
            </label>
            <div v-if="cur.up" class="acc-body">
              <div class="row">
                <label class="field">
                  <span>Upscaler</span>
                  <select v-model="cur.upModel"><option v-for="u in modelUpscalers" :key="u" :value="u">{{ u }}</option></select>
                </label>
                <label class="field">
                  <span>Resize method</span>
                  <select v-model="cur.upResize"><option v-for="m in resizeMethods" :key="m" :value="m">{{ m }}</option></select>
                </label>
              </div>
              <div class="row">
                <FSlider label="Upscale by" :min="1" :max="4" :step="0.05" v-model="cur.upScale" />
                <FSlider label="Resize width to" :min="0" :max="8192" :step="8" v-model="cur.upResizeX" />
                <FSlider label="Resize height to" :min="0" :max="8192" :step="8" v-model="cur.upResizeY" />
              </div>
            </div>
          </div>
          <div class="block">
            <div class="row size">
              <div class="size-sliders">
                <FSlider label="Width" :min="64" :max="2048" :step="8" v-model="cur.width" />
                <FSlider label="Height" :min="64" :max="2048" :step="8" v-model="cur.height" />
              </div>
              <button class="tool swap" title="Switch width and height" @click="swapSize">⇅</button>
              <div class="size-sliders">
                <FSlider label="Batch count" :min="1" :max="100" :step="1" v-model="cur.batchCount" />
                <FSlider label="Batch size" :min="1" :max="8" :step="1" v-model="cur.batchSize" />
              </div>
            </div>
            <FSlider label="CFG Scale" :min="1" :max="30" :step="0.1" v-model="cur.cfg" />
            <FSlider v-if="tab === 'img2img'" label="Denoising strength" :min="0" :max="1" :step="0.01" v-model="i2i.denoise" />
            <div class="row seed">
              <label class="field grow">
                <span>Seed</span>
                <input type="number" v-model.number="cur.seed" />
              </label>
              <button class="tool" title="A random seed every time (-1)" @click="cur.seed = -1">🎲</button>
              <button class="tool" title="The last run's seed" @click="lastSeed !== null && (cur.seed = lastSeed)">♻️</button>
            </div>
          </div>
        </div>
        <div v-else class="settings">
          <div class="block">
            <input class="filter" v-model="loraFilter" placeholder="Search LoRAs" />
            <div class="loras">
              <button v-for="l in loraList" :key="l" class="lora" :title="l" @click="addLora(l)">{{ l.replace(/\.(safetensors|pt|ckpt)$/i, '') }}</button>
            </div>
          </div>
        </div>

        <div class="output">
          <div class="gallery block">
            <img v-if="shown" :src="shown.url" alt="" />
            <div v-else class="empty">{{ running ? '' : 'Your pictures land here' }}</div>
            <div v-if="running" class="progress">
              <div class="bar" :style="{ width: (progress * 100).toFixed(1) + '%' }"></div>
              <span>{{ progressText }}</span>
            </div>
          </div>
          <div v-if="results.length > 1" class="thumbs">
            <img v-for="(r, i) in results" :key="r.url" :src="r.url" :class="{ on: i === selected }" alt="" @click="selected = i" />
          </div>
          <div class="out-buttons">
            <a v-if="shown" class="tool" :href="shown.url" target="_blank" title="Open the picture">📂</a>
            <button class="tool" :disabled="!shown" title="Send to img2img" @click="sendToImg2img">🖼️</button>
          </div>
          <pre v-if="shown" class="infotext">{{ shown.info }}</pre>
        </div>
      </div>
    </section>

    <section v-else class="pnginfo">
      <div class="cols">
        <div class="settings">
          <div class="block drop" @dragover.prevent @drop="drop($event, readPng)">
            <img v-if="pngPreview" :src="pngPreview" alt="" />
            <label class="drop-hint">
              Drop a PNG here, or click
              <input type="file" accept="image/png" @change="readPng(($event.target as HTMLInputElement).files)" />
            </label>
          </div>
        </div>
        <div class="output">
          <div v-if="pngInfo" class="block">
            <div class="kind">{{ pngInfo.kind }}</div>
            <pre class="infotext">{{ pngInfo.text }}</pre>
            <div v-if="pngInfo.leftOut.length" class="notes">
              <span>The form has no place for:</span>
              <ul><li v-for="n in pngInfo.leftOut" :key="n">{{ n }}</li></ul>
            </div>
            <div v-if="pngInfo.unrunnable.length" class="notes">
              <span>This backend can't run the picture's graph as it is:</span>
              <ul><li v-for="n in pngInfo.unrunnable" :key="n">{{ n }}</li></ul>
            </div>
            <div class="row">
              <button class="send" @click="sendInfo('txt2img')">Send to txt2img</button>
              <button class="send" @click="sendInfo('img2img')">Send to img2img</button>
            </div>
            <div class="row">
              <button
                class="send replay"
                :disabled="running || !listed(pngInfo.settings.checkpoint, checkpoints)"
                :title="pngInfo.leftOut.length ? 'Fill ' + pngInfo.tab + ' and Generate: the form leaves out what is listed above, so the picture will differ' : 'Fill ' + pngInfo.tab + ' and Generate, seed kept: the same picture'"
                @click="replay"
              >
                Replay
              </button>
              <button
                v-if="pngInfo.graph"
                class="send replay"
                :disabled="running || pngInfo.unrunnable.length > 0"
                title="Queue the picture's own graph as it is"
                @click="replayExactly"
              >
                Replay exactly
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="toasts">
      <div v-for="t in toasts" :key="t.id" class="toast" :class="t.kind">{{ t.text }}</div>
    </div>
  </div>
</template>
