// The graph as ComfyUI's API "prompt" (what POST /prompt takes), as ComfyUI's graphToPrompt
// builds it: every node that runs, its value widgets by input name, and each linked input as
// [source node id, output slot]. Links are followed through bypassed nodes and legacy Reroute
// nodes; a PrimitiveNode gives its value; muted nodes and notes are left out. Only what an output
// node (Save Image) needs is sent, since only that runs, and that part is checked first: nodes the
// backend does not run, inputs left unconnected, choices it does not have.
import { LGraphEventMode, type LGraph, type LGraphNode } from '@comfyorg/litegraph'
import { comboValues, info, inputsOf, isValueInput, VIRTUAL } from './registry'

export interface PromptNode {
  class_type: string
  inputs: Record<string, unknown>
  _meta: { title: string }
}

export interface Built {
  prompt: Record<string, PromptNode>
  /** why it cannot run as it is, one line each; empty when it can */
  problems: string[]
}

type Ref = [string, number]
const isRef = (v: unknown): v is Ref => Array.isArray(v) && v.length === 2 && typeof v[0] === 'string'

/** What feeds input `slot` of `node`: [id, output] of the real source, a primitive's value, or undefined */
function source(graph: LGraph, node: LGraphNode, slot: number, seen = new Set<string>()): unknown {
  const input = node.inputs[slot]
  if (input?.link == null) return undefined
  const link = graph.links.get(input.link)
  if (!link) return undefined
  const from = graph.getNodeById(link.origin_id)
  if (!from) return undefined
  const key = `${from.id}:${link.origin_slot}`
  if (seen.has(key)) return undefined
  seen.add(key)
  if (from.type === 'PrimitiveNode') return from.widgets?.[0]?.value
  if (from.type === 'Reroute') return source(graph, from, 0, seen)
  if (from.mode === LGraphEventMode.BYPASS) {
    // a bypassed node passes on its input of the same type
    const type = from.outputs[link.origin_slot]?.type
    const i = from.inputs.findIndex((inp) => inp.type === type && inp.link != null)
    return i >= 0 ? source(graph, from, i, seen) : undefined
  }
  return [String(from.id), link.origin_slot]
}

const NOT_RUN = 'is not something rustyui runs (yet)'

export function graphToPrompt(graph: LGraph): Built {
  const all: Record<string, PromptNode> = {}
  /** nodes the backend does not run, by id */
  const missing = new Map<string, { title: string; type: string }>()
  const nodes = [...(graph._nodes as LGraphNode[])].sort((a, b) => Number(a.id) - Number(b.id))
  for (const node of nodes) {
    const type = node.type ?? ''
    if (VIRTUAL.has(type) || node.mode === LGraphEventMode.NEVER || node.mode === LGraphEventMode.BYPASS) continue
    const def = info[type]
    if (!def || (node as any).missing) {
      missing.set(String(node.id), { title: node.title, type })
      continue
    }
    const names = new Set(inputsOf(def).map(([n]) => n))
    const inputs: Record<string, unknown> = {}
    for (const w of (node.widgets ?? []) as any[]) {
      if (names.has(w.name)) inputs[w.name] = w.value
    }
    node.inputs.forEach((input, i) => {
      if (!names.has(input.name)) return
      const v = source(graph, node, i)
      if (v !== undefined) inputs[input.name] = v
    })
    all[String(node.id)] = { class_type: type, inputs, _meta: { title: node.title } }
  }

  // what the outputs need, walking back along the links
  const problems: string[] = []
  const need = new Set<string>()
  const walk = (id: string) => {
    if (need.has(id)) return
    need.add(id)
    const m = missing.get(id)
    if (m) problems.push(`${m.title}: ${m.type} ${NOT_RUN}`)
    for (const v of Object.values(all[id]?.inputs ?? {})) if (isRef(v)) walk(v[0])
  }
  for (const [id, n] of Object.entries(all)) if (info[n.class_type]?.output_node) walk(id)
  if (!need.size && missing.size) {
    // its outputs are among the nodes rustyui does not run (a video workflow, say)
    const types = [...new Set([...missing.values()].map((m) => m.type))]
    const more = types.length > 8 ? ` and ${types.length - 8} more` : ''
    problems.push(`nothing to render here: its outputs need ${types.slice(0, 8).join(', ')}${more}, and those are not something rustyui runs (yet)`)
  } else if (!need.size) {
    problems.push('nothing to render: the graph needs a Save Image node')
  }

  const prompt: Record<string, PromptNode> = {}
  for (const id of need) {
    const n = all[id]
    if (!n) continue
    // a link from a node that does not run (muted, missing) is dropped, as ComfyUI drops it; one
    // from a missing node is already listed, so its input is not called unconnected as well
    const fedByMissing = new Set<string>()
    for (const [k, v] of Object.entries(n.inputs)) {
      if (!isRef(v) || v[0] in all) continue
      if (missing.has(v[0])) fedByMissing.add(k)
      delete n.inputs[k]
    }
    const optional = new Set(Object.keys(info[n.class_type].input.optional ?? {}))
    for (const [name, spec] of inputsOf(info[n.class_type])) {
      const v = n.inputs[name]
      if (!spec || fedByMissing.has(name)) continue
      if (isValueInput(spec)) {
        const values = comboValues(spec)
        if (values && !isRef(v) && v !== undefined && !values.includes(String(v))) {
          problems.push(`${n._meta.title}: ${name} "${v}" is not one the backend has`)
        }
      } else if (v === undefined && !optional.has(name)) {
        problems.push(`${n._meta.title}: ${name} is not connected`)
      }
    }
    prompt[id] = n
  }
  return { prompt, problems }
}

/** The prompt text going into a sampler's positive (through Concat and the like), for the queue's titles. */
export function titleOf(prompt: Record<string, PromptNode>): string | undefined {
  const textBehind = (ref: unknown, depth: number): string | undefined => {
    if (!isRef(ref) || depth > 4) return undefined
    const n = prompt[ref[0]]
    if (!n) return undefined
    if (typeof n.inputs.text === 'string') return n.inputs.text
    for (const v of Object.values(n.inputs)) {
      const t = textBehind(v, depth + 1)
      if (t) return t
    }
    return undefined
  }
  for (const n of Object.values(prompt)) {
    const t = textBehind(n.inputs.positive, 0)
    if (t) return t.slice(0, 140)
  }
  const any = Object.values(prompt).find((n) => typeof n.inputs.text === 'string')
  return any ? (any.inputs.text as string).slice(0, 140) : undefined
}
