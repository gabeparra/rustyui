// Opening things: a ComfyUI workflow (the format LiteGraph itself saves, so ComfyUI's files open
// as they are), a ComfyUI API graph (laid out in columns by depth), and PNGs from ComfyUI or
// rustyui (their "workflow" text chunk, else the "prompt" one). Node types the backend does not
// run are kept as red stand-ins, so nothing in the file is lost.
import { LiteGraph, type LGraph, type LGraphNode } from '@comfyorg/litegraph'
import { CONTROL, info, inputsOf, registerMissing, VIRTUAL } from './registry'
import { clearTextareas } from './textarea'

export type Opened = { kind: 'workflow' | 'api'; missing: string[] }

/** PNG text chunks (tEXt, and uncompressed iTXt) by keyword */
export function pngText(buf: ArrayBuffer): Record<string, string> {
  const bytes = new Uint8Array(buf)
  const sig = [137, 80, 78, 71, 13, 10, 26, 10]
  if (!sig.every((b, i) => bytes[i] === b)) throw new Error('not a PNG')
  const view = new DataView(buf)
  const out: Record<string, string> = {}
  const latin1 = new TextDecoder('latin1')
  const utf8 = new TextDecoder('utf-8')
  let p = 8
  while (p + 8 <= bytes.length) {
    const len = view.getUint32(p)
    const type = latin1.decode(bytes.subarray(p + 4, p + 8))
    const data = bytes.subarray(p + 8, p + 8 + len)
    if (type === 'tEXt') {
      const z = data.indexOf(0)
      out[latin1.decode(data.subarray(0, z))] = latin1.decode(data.subarray(z + 1))
    } else if (type === 'iTXt') {
      const z = data.indexOf(0)
      const compressed = data[z + 1]
      let q = z + 3
      q = data.indexOf(0, q) + 1 // language tag
      q = data.indexOf(0, q) + 1 // translated keyword
      if (!compressed) out[latin1.decode(data.subarray(0, z))] = utf8.decode(data.subarray(q))
    } else if (type === 'IEND') break
    p += 12 + len
  }
  return out
}

/** A workflow in ComfyUI's format (LiteGraph's), not the first rustyui editor's own ("rustyui": 1) */
export function isWorkflow(j: any): boolean {
  return !!j && typeof j === 'object' && Array.isArray(j.nodes) && !('rustyui' in j) && ('links' in j || 'last_node_id' in j || 'version' in j)
}

function isApiGraph(j: any): boolean {
  return j && typeof j === 'object' && !Array.isArray(j) && Object.values(j).length > 0 && Object.values(j).every((n: any) => n && typeof n.class_type === 'string')
}

/** Loads a ComfyUI workflow into the graph, stand-ins for unknown node types. */
export function loadWorkflow(graph: LGraph, wf: any): Opened {
  const missing: string[] = []
  for (const n of wf.nodes ?? []) {
    if (n.type && !info[n.type] && !VIRTUAL.has(n.type) && !LiteGraph.registered_node_types[n.type]) {
      registerMissing(n.type)
      missing.push(n.type)
    } else if (n.type && !info[n.type] && LiteGraph.registered_node_types[n.type] && !VIRTUAL.has(n.type)) {
      missing.push(n.type)
    }
  }
  clearTextareas()
  graph.configure(wf)
  return { kind: 'workflow', missing: [...new Set(missing)] }
}

/** Lays an API graph out: each node in the column of its longest path from a source. */
export function loadApiGraph(graph: LGraph, api: Record<string, any>): Opened {
  clearTextareas()
  graph.clear()
  const missing: string[] = []
  const nodes = new Map<string, LGraphNode>()
  for (const [id, n] of Object.entries(api)) {
    if (!info[n.class_type]) {
      registerMissing(n.class_type)
      missing.push(n.class_type)
    }
    const node = LiteGraph.createNode(n.class_type)
    if (!node) continue
    node.id = Number(id) || node.id
    if (n._meta?.title) node.title = n._meta.title
    graph.add(node)
    nodes.set(id, node)
    for (const w of (node.widgets ?? []) as any[]) {
      const v = n.inputs?.[w.name]
      if (v !== undefined && !Array.isArray(v)) w.value = v
      if (w.name === CONTROL) w.value = 'fixed' // a graph read back renders the same picture again
    }
  }
  // links, and for a missing node slots made from its inputs
  for (const [id, n] of Object.entries(api)) {
    const to = nodes.get(id)
    if (!to) continue
    for (const [name, v] of Object.entries(n.inputs ?? {})) {
      if (!Array.isArray(v) || v.length !== 2 || !nodes.has(String(v[0]))) continue
      const from = nodes.get(String(v[0]))!
      let slot = to.inputs.findIndex((i) => i.name === name)
      if (slot < 0) {
        to.addInput(name, '*')
        slot = to.inputs.length - 1
      }
      while (from.outputs.length <= Number(v[1])) from.addOutput(`out ${from.outputs.length}`, '*')
      from.connect(Number(v[1]), to, slot)
    }
  }
  // columns by depth
  const depth = new Map<string, number>()
  const depthOf = (id: string, stack = new Set<string>()): number => {
    if (depth.has(id)) return depth.get(id)!
    if (stack.has(id)) return 0
    stack.add(id)
    let d = 0
    for (const v of Object.values(api[id]?.inputs ?? {})) {
      if (Array.isArray(v) && v.length === 2 && api[String(v[0])]) d = Math.max(d, depthOf(String(v[0]), stack) + 1)
    }
    depth.set(id, d)
    return d
  }
  const columns = new Map<number, string[]>()
  for (const id of nodes.keys()) {
    const d = depthOf(id)
    columns.set(d, [...(columns.get(d) ?? []), id])
  }
  let x = 40
  for (const d of [...columns.keys()].sort((a, b) => a - b)) {
    let y = 60
    let widest = 0
    for (const id of columns.get(d)!) {
      const node = nodes.get(id)!
      node.pos = [x, y]
      y += node.size[1] + 70
      widest = Math.max(widest, node.size[0])
    }
    x += widest + 70
  }
  graph.setDirtyCanvas(true, true)
  return { kind: 'api', missing: [...new Set(missing)] }
}

/** Anything the editor opens: a workflow or API graph as JSON, or a PNG with either inside. */
export async function openFile(graph: LGraph, file: File): Promise<Opened> {
  const buf = await file.arrayBuffer()
  if (file.type === 'image/png' || file.name.toLowerCase().endsWith('.png')) {
    const text = pngText(buf)
    if (text.workflow) {
      const wf = JSON.parse(text.workflow)
      if (isWorkflow(wf)) return loadWorkflow(graph, wf)
    }
    if (text.prompt) return loadApiGraph(graph, JSON.parse(text.prompt))
    if (text.parameters) throw new Error('an A1111/Forge picture holds generation parameters, not a ComfyUI graph, so there is nothing to open')
    throw new Error('this PNG holds no graph')
  }
  return openJson(graph, JSON.parse(new TextDecoder().decode(buf)))
}

export function openJson(graph: LGraph, j: any): Opened {
  if (isWorkflow(j)) return loadWorkflow(graph, j)
  if (isApiGraph(j)) return loadApiGraph(graph, j)
  if (j?.prompt && isApiGraph(j.prompt)) return loadApiGraph(graph, j.prompt)
  throw new Error('neither a workflow nor an API graph')
}

/** The starting graph: an SDXL text-to-image of rustyui's own. */
export function defaultGraph(graph: LGraph) {
  clearTextareas()
  graph.clear()
  const make = (type: string, pos: [number, number]) => {
    const n = LiteGraph.createNode(type)
    if (!n) throw new Error(`the backend has no ${type}`)
    n.pos = pos
    graph.add(n)
    return n
  }
  const set = (n: LGraphNode, name: string, v: unknown) => {
    const w = (n.widgets ?? []).find((w: any) => w.name === name) as any
    if (w) w.value = v
  }
  const ckpt = make('CheckpointLoaderSimple', [40, 300])
  const pos = make('CLIPTextEncode', [380, 80])
  const neg = make('CLIPTextEncode', [380, 340])
  const latent = make('EmptyLatentImage', [380, 600])
  const sampler = make('KSampler', [760, 120])
  const decode = make('VAEDecode', [1110, 120])
  const save = make('SaveImage', [1340, 120])
  pos.title = 'Prompt'
  neg.title = 'Negative prompt'
  set(pos, 'text', 'a lighthouse on a rocky shore at dawn, soft golden light, waves, highly detailed')
  set(neg, 'text', 'text, watermark, blurry, lowres')
  set(latent, 'width', 1024)
  set(latent, 'height', 1024)
  set(sampler, 'seed', Math.floor(Math.random() * 2 ** 48))
  set(sampler, 'steps', 25)
  set(sampler, 'cfg', 6)
  ckpt.connect(0, sampler, 0)
  ckpt.connect(1, pos, 0)
  ckpt.connect(1, neg, 0)
  pos.connect(0, sampler, 1)
  neg.connect(0, sampler, 2)
  latent.connect(0, sampler, 3)
  sampler.connect(0, decode, 0)
  ckpt.connect(2, decode, 1)
  decode.connect(0, save, 0)
  graph.setDirtyCanvas(true, true)
}

/** After queueing: each seed moves as its "control after generate" says. */
export function advanceSeeds(graph: LGraph) {
  for (const node of graph._nodes as LGraphNode[]) {
    const ws = (node.widgets ?? []) as any[]
    ws.forEach((w, i) => {
      if (w.name !== CONTROL || i === 0) return
      const seed = ws[i - 1]
      const max = Math.min(seed.options?.max ?? Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER)
      const min = seed.options?.min ?? 0
      if (w.value === 'increment') seed.value = Math.min(max, seed.value + 1)
      else if (w.value === 'decrement') seed.value = Math.max(min, seed.value - 1)
      else if (w.value === 'randomize') seed.value = min + Math.floor(Math.random() * Math.min(2 ** 48, max - min))
    })
  }
  graph.setDirtyCanvas(true, false)
}

/** the inputs a node type takes, for menus */
export function describe(type: string): string {
  const def = info[type]
  if (!def) return type
  return `${def.display_name}: ${inputsOf(def).map(([n]) => n).join(', ')}`
}
