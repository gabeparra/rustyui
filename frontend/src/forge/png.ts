// PNG Info: a picture's text chunks, and the generation settings in them, from either kind of
// picture: A1111/Forge/Civitai's "parameters" infotext, or ComfyUI's own graph ("prompt", which
// every ComfyUI and rustyui render carries).

import type { ObjectInfo } from '../api'
import { LATENT_UPSCALERS, SAMPLER_LABELS, SCHEDULER_LABELS, scaleFor, type Settings } from './graph.ts'

/** Every tEXt, zTXt (as raw bytes' text) and iTXt chunk of a PNG, by keyword */
export async function pngText(buf: ArrayBuffer): Promise<Record<string, string>> {
  const b = new Uint8Array(buf)
  const sig = [137, 80, 78, 71, 13, 10, 26, 10]
  if (sig.some((v, i) => b[i] !== v)) throw new Error('not a PNG')
  const out: Record<string, string> = {}
  const dv = new DataView(buf)
  const latin1 = new TextDecoder('latin1')
  const utf8 = new TextDecoder('utf-8')
  let p = 8
  while (p + 8 <= b.length) {
    const len = dv.getUint32(p)
    const type = latin1.decode(b.subarray(p + 4, p + 8))
    const data = b.subarray(p + 8, p + 8 + len)
    if (type === 'tEXt') {
      const z = data.indexOf(0)
      out[latin1.decode(data.subarray(0, z))] = latin1.decode(data.subarray(z + 1))
    } else if (type === 'iTXt') {
      const z = data.indexOf(0)
      const key = latin1.decode(data.subarray(0, z))
      const compressed = data[z + 1] === 1
      let q = z + 3
      q = data.indexOf(0, q) + 1 // language tag
      q = data.indexOf(0, q) + 1 // translated keyword
      let text = data.subarray(q)
      if (compressed) text = new Uint8Array(await new Response(new Blob([text]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer())
      out[key] = utf8.decode(text)
    } else if (type === 'zTXt') {
      const z = data.indexOf(0)
      const key = latin1.decode(data.subarray(0, z))
      const raw = data.subarray(z + 2)
      const text = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer())
      out[key] = latin1.decode(text)
    } else if (type === 'IEND') {
      break
    }
    p += 12 + len
  }
  return out
}

const reverse = (m: Record<string, string>) => Object.fromEntries(Object.entries(m).map(([k, v]) => [v.toLowerCase(), k]))
const SAMPLERS = reverse(SAMPLER_LABELS)
const SCHEDULERS = reverse(SCHEDULER_LABELS)

// infotext keys that don't change the picture
const BOOKKEEPING = /^(Model hash|Lora hashes|TI hashes|Hashes|Version|VAE hash|Civitai resources|Civitai metadata|Created Date|Style Selector.*|Steps|Sampler|Schedule type|CFG scale|Seed|Size|Model|Clip skip|Denoising strength|Hires .*|Postprocess .*)$/

/** A1111's infotext (prompt, "Negative prompt: ...", then "Key: value, ..." on the last line) as
 * settings, with the keys the form has no place for; "Sampler: DPM++ 2M Karras" in older ones
 * carries the schedule in the name */
export function fromParameters(text: string): GraphReading {
  const lines = text.trim().split('\n')
  const last = lines.length > 1 && /(^|, )Steps: /.test(lines[lines.length - 1]) ? lines.pop()! : ''
  const neg = lines.findIndex((l) => l.startsWith('Negative prompt:'))
  const prompt = (neg < 0 ? lines : lines.slice(0, neg)).join('\n').trim()
  const negative = neg < 0 ? '' : [lines[neg].slice('Negative prompt:'.length), ...lines.slice(neg + 1)].join('\n').trim()
  const kv: Record<string, string> = {}
  for (const m of last.matchAll(/\s*([\w ./+()-]+):\s*("(?:\\.|[^"])*"|[^,]*)(?:,|$)/g)) kv[m[1].trim()] = m[2].trim().replace(/^"|"$/g, '')
  const out: Partial<Settings> = { prompt, negative, hires: false, up: false }
  const left: string[] = []
  const size = (v: string) => v.split('x').map(Number) as [number, number]
  const sampler = (label: string) => {
    let l = label.toLowerCase()
    let sched: string | undefined
    for (const [k, v] of Object.entries(SCHEDULERS)) if (l.endsWith(` ${k}`) && SAMPLERS[l.slice(0, -k.length - 1)]) [l, sched] = [l.slice(0, -k.length - 1), v]
    if (!SAMPLERS[l]) left.push(`the sampler ${label}`)
    return { name: SAMPLERS[l], sched }
  }
  const scheduler = (label: string) => {
    if (!SCHEDULERS[label.toLowerCase()] && label.toLowerCase() !== 'automatic') left.push(`the schedule type ${label}`)
    return SCHEDULERS[label.toLowerCase()]
  }
  if (kv['Steps']) out.steps = Number(kv['Steps'])
  if (kv['CFG scale']) out.cfg = Number(kv['CFG scale'])
  if (kv['Seed']) out.seed = Number(kv['Seed'])
  if (kv['Size']) [out.width, out.height] = size(kv['Size'])
  if (kv['Clip skip']) out.clipSkip = Number(kv['Clip skip'])
  if (kv['Model']) out.checkpoint = kv['Model']
  if (kv['Sampler']) {
    const r = sampler(kv['Sampler'])
    out.sampler = r.name
    out.scheduler = r.sched
  }
  if (kv['Schedule type']) out.scheduler = scheduler(kv['Schedule type']) ?? out.scheduler
  if (kv['Hires upscale'] || kv['Hires upscaler'] || kv['Hires resize']) {
    out.hires = true
    out.hrResizeX = out.hrResizeY = 0
    out.hrSampler = out.hrScheduler = ''
    out.hrCfg = 0
    if (kv['Hires upscale']) out.hrScale = Number(kv['Hires upscale'])
    if (kv['Hires resize']) [out.hrResizeX, out.hrResizeY] = size(kv['Hires resize'])
    if (kv['Hires steps']) out.hrSteps = Number(kv['Hires steps'])
    if (kv['Hires upscaler']) out.hrUpscaler = kv['Hires upscaler']
    if (kv['Denoising strength']) out.hrDenoise = Number(kv['Denoising strength'])
    if (kv['Hires sampler'] && kv['Hires sampler'] !== 'Use same sampler') out.hrSampler = sampler(kv['Hires sampler']).name
    if (kv['Hires schedule type'] && kv['Hires schedule type'] !== 'Use same scheduler') out.hrScheduler = scheduler(kv['Hires schedule type'])
    if (kv['Hires CFG Scale'] && Number(kv['Hires CFG Scale']) !== out.cfg) out.hrCfg = Number(kv['Hires CFG Scale'])
    for (const k of ['Hires prompt', 'Hires negative prompt', 'Hires checkpoint']) if (kv[k] && kv[k] !== 'Use same checkpoint') left.push(`${k}: ${kv[k].slice(0, 60)}`)
  } else if (kv['Denoising strength']) {
    out.denoise = Number(kv['Denoising strength'])
  }
  if (kv['Postprocess upscaler']) {
    Object.assign(out, { up: true, upModel: kv['Postprocess upscaler'], upScale: Number(kv['Postprocess upscale by'] ?? 1), upResizeX: 0, upResizeY: 0 })
    if (kv['Postprocess upscale to']) [out.upResizeX, out.upResizeY] = size(kv['Postprocess upscale to'])
  }
  for (const [k, v] of Object.entries(kv)) if (!BOOKKEEPING.test(k)) left.push(`${k}: ${v.length > 60 ? v.slice(0, 60) + '…' : v}`)
  for (const k of Object.keys(out) as (keyof Settings)[]) if (out[k] === undefined) delete out[k]
  return { settings: out, leftOut: left, tab: kv['Denoising strength'] && !out.hires ? 'img2img' : 'txt2img' }
}

type Node = { class_type: string; inputs: Record<string, any> }
type Graph = Record<string, Node>
type Found = [string, Node]

/** A ComfyUI graph as the form's settings, with what of it the form has no place for */
export interface GraphReading {
  settings: Partial<Settings>
  /** the graph's pieces the form leaves out, in words; empty when the form carries all of it */
  leftOut: string[]
  /** txt2img, or img2img when the sampler starts from a loaded picture */
  tab: 'txt2img' | 'img2img'
}

const isLink = (v: unknown): v is [string | number, number] => Array.isArray(v) && v.length === 2 && typeof v[1] === 'number'
const bare = (name: unknown) => String(name).replace(/\.(safetensors|ckpt|pt|pth|bin|gguf)$/i, '')
const LATENT_LABELS: Record<string, string> = Object.fromEntries(Object.entries(LATENT_UPSCALERS).map(([k, v]) => [v, k]))

/** A ComfyUI API graph (a PNG's "prompt" chunk) read back from its SaveImage: the picture's decode,
 * its sampler, a hires pass before it (through an upscale model or in latent space), the LoRA chain
 * to the checkpoint, clip skip, and the prompts (ConditioningConcat as BREAK). Whatever else the
 * picture passes through is named in `leftOut`, so the form never quietly drops part of a render. */
export function readComfyGraph(json: string): GraphReading {
  const g: Graph = JSON.parse(json)
  const left: string[] = []
  const s: Partial<Settings> = {}
  const reading: GraphReading = { settings: s, leftOut: left, tab: 'txt2img' }
  const used = new Set<string>()
  const at = (link: unknown): Found | null => (isLink(link) && g[String(link[0])] ? [String(link[0]), g[String(link[0])]] : null)
  const is = (link: unknown, ...classes: string[]): Found | null => {
    const f = at(link)
    return f && classes.includes(f[1].class_type) ? f : null
  }
  const use = (...fs: (Found | null | undefined)[]) => fs.forEach((f) => f && used.add(f[0]))
  // a widget's value, through the primitive node it may be linked from (PrimitiveInt, a seed node, ...)
  const val = (v: unknown, depth = 0): any => {
    const f = at(v)
    if (!f) return v
    const k = ['value', 'seed', 'noise_seed', 'text', 'string', 'int', 'float', 'number'].find((k) => k in (f[1].inputs ?? {}))
    if (!k || depth > 8) return undefined
    use(f)
    return val(f[1].inputs[k], depth + 1)
  }
  const sinks = Object.entries(g).filter(([id]) => !Object.values(g).some((n) => Object.values(n.inputs ?? {}).some((v) => isLink(v) && String(v[0]) === id)))
  const outputs = sinks.filter(([, n]) => /^(Save|Preview)|VideoCombine/.test(n.class_type))
  const save = outputs.find(([, n]) => n.class_type === 'SaveImage') ?? outputs.find(([, n]) => n.class_type === 'PreviewImage')
  if (!save) {
    left.push('the graph saves no picture (no SaveImage)')
    return reading
  }
  use(save)
  const others = outputs.filter((o) => o !== save)
  if (others.length) left.push(`another output: ${others.map(([, n]) => n.class_type).join(', ')}`)

  let img = save[1].inputs.images
  // an upscale after the last decode (a final model upscale), its size worked out further down
  const fScale = is(img, 'ImageScale')
  const fUp = is(fScale ? fScale[1].inputs.image : img, 'ImageUpscaleWithModel')
  let upTo: [number, number] | null = null
  s.up = false
  if (fUp) {
    const loader = is(fUp[1].inputs.upscale_model, 'UpscaleModelLoader')
    use(fScale, fUp, loader)
    img = fUp[1].inputs.image
    if (fScale && loader) {
      Object.assign(s, { up: true, upModel: loader[1].inputs.model_name, upResize: fScale[1].inputs.upscale_method })
      upTo = [fScale[1].inputs.width, fScale[1].inputs.height]
      if (fScale[1].inputs.crop !== 'disabled') left.push(`the final resize's crop (${fScale[1].inputs.crop})`)
    } else {
      left.push(`the upscale after the last decode at the model's own size (${bare(loader?.[1].inputs.model_name)}; the form resizes it to a size)`)
    }
  } else if (fScale) {
    left.push(`the resize after the last decode (${fScale[1].inputs.upscale_method} to ${fScale[1].inputs.width}x${fScale[1].inputs.height})`)
    use(fScale)
    img = fScale[1].inputs.image
  }
  const dec = is(img, 'VAEDecode')
  if (!dec) {
    left.push(`the picture comes from ${at(img)?.[1].class_type ?? 'nowhere'}, which the form doesn't make`)
    return reading
  }
  use(dec)
  const vae = at(dec[1].inputs.vae)
  if (vae?.[1].class_type === 'VAELoader') {
    left.push(`the VAE ${bare(vae[1].inputs.vae_name)} (VAELoader; the form decodes with the checkpoint's)`)
    use(vae)
  }
  const last = is(dec[1].inputs.samples, 'KSampler')
  if (!last) {
    left.push(`the sampler is ${at(dec[1].inputs.samples)?.[1].class_type ?? 'missing'} (the form's is KSampler)`)
    return reading
  }
  use(last)
  const ks = last[1].inputs

  // a hires pass: the first sampler's latent upscaled, or decoded, upscaled by a model, resized and
  // encoded again; else img2img's picture, loaded and resized
  let base = ks
  let hires: Record<string, any> | null = null
  let target: [number, number] | null = null
  const lu = is(ks.latent_image, 'LatentUpscale')
  const enc = is(ks.latent_image, 'VAEEncode')
  if (lu && is(lu[1].inputs.samples, 'KSampler')) {
    const first = is(lu[1].inputs.samples, 'KSampler')!
    use(lu, first)
    hires = ks
    base = first[1].inputs
    target = [lu[1].inputs.width, lu[1].inputs.height]
    const label = LATENT_LABELS[lu[1].inputs.upscale_method]
    if (label) s.hrUpscaler = label
    else left.push(`the hires pass's latent upscale ${lu[1].inputs.upscale_method} (the form has ${Object.keys(LATENT_UPSCALERS).join(', ')})`)
    if (lu[1].inputs.crop !== 'disabled') left.push(`the hires pass's crop (${lu[1].inputs.crop})`)
  } else if (enc) {
    const sc = is(enc[1].inputs.pixels, 'ImageScale')
    const up = is(sc ? sc[1].inputs.image : enc[1].inputs.pixels, 'ImageUpscaleWithModel')
    const d2 = up && is(up[1].inputs.image, 'VAEDecode')
    const first = d2 && is(d2[1].inputs.samples, 'KSampler')
    const load = !up && is(sc ? sc[1].inputs.image : enc[1].inputs.pixels, 'LoadImage')
    if (first && up) {
      const loader = is(up[1].inputs.upscale_model, 'UpscaleModelLoader')
      use(enc, sc, up, d2, first, loader)
      hires = ks
      base = first[1].inputs
      s.hrUpscaler = loader?.[1].inputs.model_name
      if (sc) {
        target = [sc[1].inputs.width, sc[1].inputs.height]
        s.hrResize = sc[1].inputs.upscale_method
        if (sc[1].inputs.crop !== 'disabled') left.push(`the hires pass's crop (${sc[1].inputs.crop})`)
      } else {
        left.push(`the hires pass at the upscale model's own size (the form resizes to a target)`)
      }
    } else if (load) {
      use(enc, sc, load)
      reading.tab = 'img2img'
      s.initImage = val(load[1].inputs.image)
      s.denoise = val(ks.denoise)
      if (sc) {
        s.width = sc[1].inputs.width
        s.height = sc[1].inputs.height
        if (sc[1].inputs.upscale_method !== 'lanczos') left.push(`the picture's resize method ${sc[1].inputs.upscale_method} (the form resizes with lanczos)`)
      } else {
        left.push(`the picture at its own size (the form resizes it to the width and height)`)
      }
    }
  }
  if (reading.tab === 'txt2img') {
    const empty = is(base.latent_image, 'EmptyLatentImage')
    if (empty) {
      use(empty)
      s.width = val(empty[1].inputs.width)
      s.height = val(empty[1].inputs.height)
      s.batchSize = val(empty[1].inputs.batch_size)
      if (val(base.denoise) !== 1) left.push(`denoise ${val(base.denoise)} on an empty latent`)
    } else {
      left.push(`the sampler starts from ${at(base.latent_image)?.[1].class_type ?? 'nothing'} (the form starts from an empty latent, or img2img's picture)`)
    }
  }
  s.seed = val(base.seed)
  s.steps = val(base.steps)
  s.cfg = val(base.cfg)
  s.sampler = val(base.sampler_name)
  s.scheduler = val(base.scheduler)
  if (hires) {
    s.hires = true
    s.hrSteps = val(hires.steps)
    s.hrDenoise = val(hires.denoise)
    // Forge's "Use same sampler / scheduler" when they match, the hires pass's own when not
    s.hrSampler = val(hires.sampler_name) === s.sampler ? '' : val(hires.sampler_name)
    s.hrScheduler = val(hires.scheduler) === s.scheduler ? '' : val(hires.scheduler)
    s.hrCfg = val(hires.cfg) === s.cfg ? 0 : val(hires.cfg)
    if (val(hires.seed) !== s.seed) left.push(`the hires pass's own seed ${val(hires.seed)} (the form uses the first pass's, ${s.seed})`)
    if (JSON.stringify(hires.model) !== JSON.stringify(base.model)) left.push(`the hires pass's own model`)
    if (JSON.stringify([hires.positive, hires.negative]) !== JSON.stringify([base.positive, base.negative])) left.push(`the hires pass's own prompts`)
    if (target) {
      const r = scaleFor([s.width ?? 0, s.height ?? 0], target)
      Object.assign(s, { hrResizeX: r.toX, hrResizeY: r.toY }, r.by ? { hrScale: r.by } : {})
    }
  } else if (reading.tab === 'txt2img') {
    s.hires = false
  }
  if (upTo) {
    const from: [number, number] = hires && target ? target : [s.width ?? 0, s.height ?? 0]
    const r = scaleFor(from, upTo)
    Object.assign(s, { upResizeX: r.toX, upResizeY: r.toY }, r.by ? { upScale: r.by } : {})
  }

  // the model chain from the first sampler back to its loader, LoRA by LoRA
  const loras: { id: string; name: string; model: number; clip?: number }[] = []
  let ckpt: Found | null = null
  let m = at(base.model)
  for (let i = 0; m && i < 64; i++) {
    const [id, n] = m
    if (n.class_type === 'LoraLoader' || n.class_type === 'LoraLoaderModelOnly') {
      use(m)
      loras.unshift({ id, name: bare(val(n.inputs.lora_name)), model: val(n.inputs.strength_model) })
    } else if (n.class_type === 'CheckpointLoaderSimple') {
      use(m)
      ckpt = m
      break
    } else if (n.class_type === 'UNETLoader') {
      use(m)
      left.push(`the model ${bare(n.inputs.unet_name)} (a diffusion model file, UNETLoader; the form loads checkpoints)`)
      break
    }
    m = at(n.inputs.model)
  }
  if (ckpt) s.checkpoint = val(ckpt[1].inputs.ckpt_name)

  // the prompts, and the text encoder's chain from the first prompt back to its loader
  const encoders: Found[] = []
  const text = (link: unknown, depth = 0): string => {
    const f = at(link)
    if (!f || depth > 64) return ''
    const n = f[1]
    if (n.class_type === 'CLIPTextEncode') {
      use(f)
      encoders.push(f)
      const t = val(n.inputs.text)
      return typeof t === 'string' ? t : ''
    }
    if (n.class_type === 'ConditioningConcat') {
      use(f)
      return `${text(n.inputs.conditioning_to, depth + 1)} BREAK ${text(n.inputs.conditioning_from, depth + 1)}`
    }
    // anything else on the conditioning (ControlNet, hooks, ...) is named below; its text is behind it
    if (isLink(n.inputs.positive) && isLink(n.inputs.negative)) return text(isLink(link) && link[1] === 1 ? n.inputs.negative : n.inputs.positive, depth + 1)
    for (const k of ['conditioning', 'conditioning_to', 'conditioning_1', 'cond']) if (isLink(n.inputs[k])) return text(n.inputs[k], depth + 1)
    return ''
  }
  const positive = text(base.positive)
  s.negative = text(base.negative)
  if (encoders.some(([, n]) => /\bBREAK\b/.test(String(val(n.inputs.text))))) left.push(`"BREAK" written in a prompt (a plain word to ComfyUI; the form splits the prompt there)`)
  if (encoders.some(([, n]) => /<lora:/i.test(String(val(n.inputs.text))))) left.push(`a <lora:...> tag written in a prompt (plain text to ComfyUI; the form loads it as a LoRA)`)
  let c = encoders.length ? at(encoders[0][1].inputs.clip) : null
  for (let i = 0; c && i < 64; i++) {
    const [id, n] = c
    if (n.class_type === 'LoraLoader') {
      use(c)
      const l = loras.find((x) => x.id === id)
      if (l) l.clip = val(n.inputs.strength_clip)
      else left.push(`the LoRA ${bare(n.inputs.lora_name)} on the text encoder only`)
    } else if (n.class_type === 'CLIPSetLastLayer') {
      use(c)
      s.clipSkip = -val(n.inputs.stop_at_clip_layer)
    } else if (n.class_type === 'CheckpointLoaderSimple') {
      if (ckpt && id !== ckpt[0]) left.push(`a second checkpoint for the text encoder (${bare(n.inputs.ckpt_name)})`)
      use(c)
      break
    } else if (n.class_type === 'CLIPLoader') {
      use(c)
      left.push(`the text encoder ${bare(n.inputs.clip_name)} (CLIPLoader, type ${n.inputs.type}; the form uses the checkpoint's)`)
      break
    }
    c = at(n.inputs.clip)
  }
  if (encoders.length && s.clipSkip === undefined) s.clipSkip = 1
  // LoRA tags after the text, each with ", " before it, which parseLoras takes back out exactly;
  // <lora:name:te:unet> when the strengths differ, te 0 for a LoRA on the model only
  const tag = (l: (typeof loras)[number]) => (l.clip === l.model ? `<lora:${l.name}:${l.model}>` : `<lora:${l.name}:${l.clip ?? 0}:${l.model}>`)
  s.prompt = positive + loras.map((l, i) => `${positive || i ? ', ' : ''}${tag(l)}`).join('')

  // every other node the picture passes through
  const reach = new Set<string>()
  const stack = [save[0]]
  while (stack.length) {
    const id = stack.pop()!
    if (reach.has(id) || !g[id]) continue
    reach.add(id)
    for (const v of Object.values(g[id].inputs ?? {})) if (isLink(v)) stack.push(String(v[0]))
  }
  const unread = new Map<string, number>()
  for (const id of reach) if (!used.has(id)) unread.set(g[id].class_type, (unread.get(g[id].class_type) ?? 0) + 1)
  for (const [cls, k] of unread) left.push(k > 1 ? `${cls} (${k})` : cls)
  for (const k of Object.keys(s) as (keyof Settings)[]) if (s[k] === undefined) delete s[k]
  return reading
}

/** What stops this backend running a graph as it is: node classes it has no definition for, and
 * choices (a checkpoint, a LoRA, a picture in the input folder) it doesn't list */
export function unrunnable(json: string, info: ObjectInfo): string[] {
  const g: Graph = JSON.parse(json)
  const out: string[] = []
  const missing = new Set<string>()
  for (const n of Object.values(g)) {
    const def = info[n.class_type]
    if (!def) {
      missing.add(n.class_type)
      continue
    }
    for (const [k, v] of Object.entries(n.inputs ?? {})) {
      if (typeof v !== 'string') continue
      const spec = def.input?.required?.[k] ?? def.input?.optional?.[k]
      const choices = Array.isArray(spec?.[0]) ? spec![0] : spec?.[0] === 'COMBO' ? spec?.[1]?.options : null
      if (Array.isArray(choices) && !choices.includes(v)) out.push(`${n.class_type}'s ${k} "${v}" isn't one this backend has`)
    }
  }
  if (missing.size) out.unshift(`rustyui here has no ${[...missing].join(', ')}`)
  return out
}
