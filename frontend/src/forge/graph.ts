// The Forge-style editor's settings as a ComfyUI API graph (what POST /prompt takes), built from
// the nodes rustyui runs exactly: a checkpoint, the prompt's <lora:name:weight> tags as a LoRA
// chain (A1111's syntax, the tags taken out of the text), clip skip as CLIPSetLastLayer, BREAK as
// ConditioningConcat, KSampler, a hires fix through a model upscaler (ImageUpscaleWithModel, then a
// resize to the target, as common SDXL hires workflows do it) or in latent space, and an upscale
// after the last decode.

export interface Settings {
  checkpoint: string
  clipSkip: number
  prompt: string
  negative: string
  sampler: string
  scheduler: string
  steps: number
  cfg: number
  width: number
  height: number
  seed: number
  batchCount: number
  batchSize: number
  hires: boolean
  hrUpscaler: string
  hrSteps: number
  hrDenoise: number
  hrScale: number
  hrResizeX: number
  hrResizeY: number
  /** how the model-upscaled picture is resized to the hires target (ImageScale's method) */
  hrResize: string
  /** the hires pass's own sampler, schedule and CFG: '' and 0 for the first pass's */
  hrSampler: string
  hrScheduler: string
  hrCfg: number
  /** an upscale after the last decode: through a model, then resized to a size given as the hires
   * fix gives its own, from the size the picture has by then */
  up: boolean
  upModel: string
  upScale: number
  upResizeX: number
  upResizeY: number
  upResize: string
  /** img2img: the uploaded picture's name in the input folder, and how much of it is redrawn */
  initImage?: string
  denoise: number
}

/** Forge's latent upscale modes rustyui runs, by the names Forge gives them */
export const LATENT_UPSCALERS: Record<string, string> = {
  Latent: 'bilinear',
  'Latent (nearest-exact)': 'nearest-exact',
}

/** ComfyUI's samplers and schedulers by the names A1111 and Forge give them */
export const SAMPLER_LABELS: Record<string, string> = {
  euler: 'Euler', euler_ancestral: 'Euler a', euler_cfg_pp: 'Euler CFG++', euler_ancestral_cfg_pp: 'Euler a CFG++', heun: 'Heun', lms: 'LMS',
  dpm_2: 'DPM2', dpm_2_ancestral: 'DPM2 a', dpm_fast: 'DPM fast', dpm_adaptive: 'DPM adaptive', dpmpp_2s_ancestral: 'DPM++ 2S a',
  dpmpp_sde: 'DPM++ SDE', dpmpp_2m: 'DPM++ 2M', dpmpp_2m_cfg_pp: 'DPM++ 2M CFG++', dpmpp_2m_sde: 'DPM++ 2M SDE', dpmpp_2m_sde_gpu: 'DPM++ 2M SDE GPU',
  dpmpp_2m_sde_heun: 'DPM++ 2M SDE Heun', dpmpp_3m_sde: 'DPM++ 3M SDE', ddim: 'DDIM', uni_pc: 'UniPC', lcm: 'LCM', er_sde: 'ER SDE',
}
export const SCHEDULER_LABELS: Record<string, string> = {
  normal: 'Normal', karras: 'Karras', exponential: 'Exponential', sgm_uniform: 'SGM Uniform', simple: 'Simple', ddim_uniform: 'DDIM',
  beta: 'Beta', kl_optimal: 'KL Optimal', linear_quadratic: 'Linear Quadratic',
}
export const samplerLabel = (s: string) => SAMPLER_LABELS[s] ?? s
export const schedulerLabel = (s: string) => SCHEDULER_LABELS[s] ?? s

export interface LoraUse {
  file: string
  /** the model's strength and the text encoder's (0: the LoRA on the model only) */
  model: number
  clip: number
}

// a tag with the ", " or spaces before it, or at the very start with those after it
const LORA_TAG = /^<lora:([^:>]+)((?::[^:>]*)*)>(?:,[ \t]*|[ \t]+)?|(?:,[ \t]*|[ \t]+)?<lora:([^:>]+)((?::[^:>]*)*)>/i

/** The prompt without its <lora:...> tags, and the LoRAs they name (matched to the files by name,
 * case and extension aside). Each tag goes out with the separator beside it and nothing else in the
 * text changes, so a graph's prompt read back with its LoRAs appended as ", <lora:...>" comes out
 * as it went in. A1111's weights: <lora:name:w> for both, <lora:name:te:unet> (or te=, unet=) for
 * the text encoder's and the model's own. A name no file has is an error, as Forge says it. */
export function parseLoras(prompt: string, files: string[]): { text: string; loras: LoraUse[] } {
  const loras: LoraUse[] = []
  const base = (f: string) => f.replace(/\.(safetensors|pt|ckpt)$/i, '').toLowerCase()
  let text = prompt
  for (let m = LORA_TAG.exec(text); m; m = LORA_TAG.exec(text)) {
    const name = (m[1] ?? m[3]).trim()
    const args = (m[2] ?? m[4]).split(':').slice(1).map((a) => a.trim())
    const named = Object.fromEntries(args.filter((a) => a.includes('=')).map((a) => a.split('=').map((x) => x.trim())))
    const pos = args.filter((a) => !a.includes('='))
    const te = Number(named.te ?? (pos[0] || 1))
    const unet = Number(named.unet ?? (pos[1] || te))
    if (!Number.isFinite(te) || !Number.isFinite(unet)) throw new Error(`the LoRA ${name}'s weight isn't a number`)
    const want = name.toLowerCase()
    const file = files.find((f) => base(f) === want) ?? files.find((f) => base(f.split('/').pop()!) === want)
    if (!file) throw new Error(`no LoRA called ${name}`)
    loras.push({ file, model: unet, clip: te })
    text = text.slice(0, m.index) + text.slice(m.index + m[0].length)
  }
  return { text, loras }
}

/** A size made bigger as Forge's hires fix makes it: "Resize width/height to" when set (one of them
 * alone keeps the aspect), else the size times "Upscale by", each truncated */
export function scaledSize(w: number, h: number, by: number, toX: number, toY: number): [number, number] {
  if (toX > 0 || toY > 0) {
    if (toX > 0 && toY > 0) return [toX, toY]
    if (toX > 0) return [toX, Math.trunc((h * toX) / w)]
    return [Math.trunc((w * toY) / h), toY]
  }
  return [Math.trunc(w * by), Math.trunc(h * by)]
}

/** The hires fix's target size */
export const hiresSize = (s: Settings) => scaledSize(s.width, s.height, s.hrScale, s.hrResizeX, s.hrResizeY)

/** The size the upscale after the last decode resizes to, from the size the picture has by then */
export function upscaleSize(s: Settings): [number, number] {
  const [w, h] = s.hires ? hiresSize(s) : [s.width, s.height]
  return scaledSize(w, h, s.upScale, s.upResizeX, s.upResizeY)
}

/** "Upscale by" when it gives `to` from `from` as Forge truncates it, else "Resize width/height to" */
export function scaleFor(from: [number, number], to: [number, number]): { by?: number; toX: number; toY: number } {
  const [w, h] = from
  const by = w && h ? [to[0] / w, to[1] / h].map((x) => Math.round(x * 100) / 100).find((x) => Math.trunc(w * x) === to[0] && Math.trunc(h * x) === to[1]) : undefined
  return by ? { by, toX: 0, toY: 0 } : { toX: to[0], toY: to[1] }
}

type Graph = Record<string, { class_type: string; inputs: Record<string, unknown> }>

/** One prompt's graph: `seed` is this run's (batch count runs one graph each, seed + i) */
export function buildGraph(s: Settings, seed: number, loraFiles: string[], prefix: string): Graph {
  const g: Graph = {}
  let n = 0
  const add = (class_type: string, inputs: Record<string, unknown>) => {
    const id = String(++n)
    g[id] = { class_type, inputs }
    return id
  }
  const ckpt = add('CheckpointLoaderSimple', { ckpt_name: s.checkpoint })
  let model: [string, number] = [ckpt, 0]
  let clip: [string, number] = [ckpt, 1]
  const pos = parseLoras(s.prompt, loraFiles)
  const neg = parseLoras(s.negative, loraFiles)
  for (const l of [...pos.loras, ...neg.loras]) {
    if (l.clip === 0) {
      model = [add('LoraLoaderModelOnly', { lora_name: l.file, strength_model: l.model, model }), 0]
      continue
    }
    const id = add('LoraLoader', { lora_name: l.file, strength_model: l.model, strength_clip: l.clip, model, clip })
    model = [id, 0]
    clip = [id, 1]
  }
  if (s.clipSkip > 1) clip = [add('CLIPSetLastLayer', { stop_at_clip_layer: -s.clipSkip, clip }), 0]
  // BREAK: each part encoded, then joined end to end (one space either side of it goes with it, as
  // the reader joins them, so the parts come back as they were)
  const encode = (text: string): [string, number] => {
    const parts = text.split(/\s?\bBREAK\b\s?/)
    let cond: [string, number] = [add('CLIPTextEncode', { text: parts[0], clip }), 0]
    for (const p of parts.slice(1)) {
      const next = add('CLIPTextEncode', { text: p, clip })
      cond = [add('ConditioningConcat', { conditioning_to: cond, conditioning_from: [next, 0] }), 0]
    }
    return cond
  }
  const positive = encode(pos.text)
  const negative = encode(neg.text)
  let latent: [string, number]
  let denoise = 1
  if (s.initImage) {
    // img2img: the picture resized to the size ("Just resize"), encoded, partly redrawn
    const img = add('LoadImage', { image: s.initImage })
    const sized = add('ImageScale', { image: [img, 0], upscale_method: 'lanczos', width: s.width, height: s.height, crop: 'disabled' })
    latent = [add('VAEEncode', { pixels: [sized, 0], vae: [ckpt, 2] }), 0]
    denoise = s.denoise
  } else {
    latent = [add('EmptyLatentImage', { width: s.width, height: s.height, batch_size: s.batchSize }), 0]
  }
  const sample = (lat: [string, number], steps: number, d: number, hr = false) =>
    add('KSampler', {
      model, seed, steps, cfg: (hr && s.hrCfg) || s.cfg, sampler_name: (hr && s.hrSampler) || s.sampler, scheduler: (hr && s.hrScheduler) || s.scheduler,
      positive, negative, latent_image: lat, denoise: d,
    })
  let z: [string, number] = [sample(latent, s.steps, denoise), 0]
  if (s.hires) {
    const [tw, th] = hiresSize(s)
    const steps = s.hrSteps > 0 ? s.hrSteps : s.steps
    const mode = LATENT_UPSCALERS[s.hrUpscaler]
    let lat: [string, number]
    if (mode) {
      lat = [add('LatentUpscale', { samples: z, upscale_method: mode, width: tw, height: th, crop: 'disabled' }), 0]
    } else {
      const decoded = add('VAEDecode', { samples: z, vae: [ckpt, 2] })
      const um = add('UpscaleModelLoader', { model_name: s.hrUpscaler })
      const up = add('ImageUpscaleWithModel', { upscale_model: [um, 0], image: [decoded, 0] })
      const sized = add('ImageScale', { image: [up, 0], upscale_method: s.hrResize || 'lanczos', width: tw, height: th, crop: 'disabled' })
      lat = [add('VAEEncode', { pixels: [sized, 0], vae: [ckpt, 2] }), 0]
    }
    z = [sample(lat, steps, s.hrDenoise, true), 0]
  }
  let image: [string, number] = [add('VAEDecode', { samples: z, vae: [ckpt, 2] }), 0]
  if (s.up) {
    const [w, h] = upscaleSize(s)
    const um = add('UpscaleModelLoader', { model_name: s.upModel })
    const up = add('ImageUpscaleWithModel', { upscale_model: [um, 0], image })
    image = [add('ImageScale', { image: [up, 0], upscale_method: s.upResize || 'lanczos', width: w, height: h, crop: 'disabled' }), 0]
  }
  add('SaveImage', { images: image, filename_prefix: prefix })
  return g
}

/** Whether two API graphs run the same: node by node from their outputs, ids aside, and each
 * SaveImage's filename prefix aside. `only` gets the nodes of `a` that `b` has no match for. */
export function sameGraph(a: Graph, b: Graph, only?: (cls: string) => void): boolean {
  const table = new Map<string, number>()
  const sig = (g: Graph) => {
    const memo = new Map<string, number>()
    const node = (id: string, depth = 0): number => {
      const n = g[id]
      if (memo.has(id)) return memo.get(id)!
      if (!n || depth > 256) return -1
      const ins = Object.keys(n.inputs ?? {})
        .filter((k) => !(n.class_type === 'SaveImage' && k === 'filename_prefix'))
        .sort()
        .map((k) => {
          const v = n.inputs[k]
          return Array.isArray(v) && v.length === 2 && typeof v[1] === 'number' ? `${k}=#${node(String(v[0]), depth + 1)}.${v[1]}` : `${k}=${JSON.stringify(v)}`
        })
      const key = `${n.class_type}{${ins.join(',')}}`
      if (!table.has(key)) table.set(key, table.size)
      memo.set(id, table.get(key)!)
      return table.get(key)!
    }
    const outs = Object.keys(g).filter((id) => /^(Save|Preview)/.test(g[id].class_type))
    return { outs: outs.map((id) => node(id)).sort(), all: new Set(memo.values()), memo }
  }
  const [x, y] = [sig(a), sig(b)]
  if (only) for (const [id, k] of x.memo) if (!y.all.has(k)) only(a[id].class_type)
  return JSON.stringify(x.outs) === JSON.stringify(y.outs)
}

/** A1111's infotext for one run, as Forge prints it under the picture */
export function infotext(s: Settings, seed: number): string {
  const fields: [string, string | number][] = [
    ['Steps', s.steps],
    ['Sampler', samplerLabel(s.sampler)],
    ['Schedule type', schedulerLabel(s.scheduler)],
    ['CFG scale', s.cfg],
    ['Seed', seed],
    ['Size', `${s.width}x${s.height}`],
  ]
  if (s.checkpoint) fields.push(['Model', s.checkpoint.replace(/\.(safetensors|ckpt)$/i, '')])
  if (s.initImage) fields.push(['Denoising strength', s.denoise])
  if (s.clipSkip > 1) fields.push(['Clip skip', s.clipSkip])
  if (s.hires) {
    const [tw, th] = hiresSize(s)
    fields.push(['Denoising strength', s.hrDenoise], ['Hires upscale', s.hrScale], ['Hires resize', `${tw}x${th}`], ['Hires steps', s.hrSteps], ['Hires upscaler', s.hrUpscaler])
    if (s.hrSampler) fields.push(['Hires sampler', samplerLabel(s.hrSampler)])
    if (s.hrScheduler) fields.push(['Hires schedule type', schedulerLabel(s.hrScheduler)])
    if (s.hrCfg) fields.push(['Hires CFG Scale', s.hrCfg])
  }
  if (s.up) {
    const [w, h] = upscaleSize(s)
    fields.push(['Postprocess upscale to', `${w}x${h}`], ['Postprocess upscaler', s.upModel])
  }
  const neg = s.negative.trim() ? `\nNegative prompt: ${s.negative.trim()}` : ''
  return `${s.prompt.trim()}${neg}\n${fields.map(([k, v]) => `${k}: ${v}`).join(', ')}`
}
