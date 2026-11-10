// Node types from the backend's /object_info, registered with LiteGraph the way ComfyUI does it:
// link inputs become slots, value inputs become widgets in the node definition's order, and a
// seed gets ComfyUI's "control after generate" right after it (so a ComfyUI workflow's
// widgets_values land on the same widgets).
import { LiteGraph, LGraphNode } from '@comfyorg/litegraph'
import type { InputSpec, NodeDef, ObjectInfo } from '../api'
import { addTextarea, dropTextareas } from './textarea'

export const CONTROL = 'control after generate'
const VALUE_TYPES = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO', 'COMFY_DYNAMICCOMBO_V3'])
/** nodes that exist only in the editor and never reach the backend */
export const VIRTUAL = new Set(['Note', 'MarkdownNote', 'Reroute', 'PrimitiveNode'])

export let info: ObjectInfo = {}

/** pictures each output node last made, by node id, drawn inside the node */
export const previews = new Map<string, HTMLImageElement[]>()

/** An input's type; one that accepts several ("FLOAT,INT", LTXVEmptyLatentAudio's frame_rate) is
 * a widget of its first, as in ComfyUI */
function typeOf(spec: InputSpec): string | unknown[] {
  const t = spec[0]
  return typeof t === 'string' && t.includes(',') ? t.split(',')[0] : t
}

export function isValueInput(spec: InputSpec | undefined): boolean {
  return !!spec && (Array.isArray(spec[0]) || VALUE_TYPES.has(typeOf(spec) as string))
}

/**
 * A combo's choices in any of ComfyUI's forms: `[[...]]`, the newer `["COMBO", {options}]`, or a
 * dynamic combo's option keys (`COMFY_DYNAMICCOMBO_V3`, SaveVideo's format and codec; the inputs an
 * option adds are not shown yet). Undefined for anything that is not a combo.
 */
export function comboValues(spec: InputSpec | undefined): string[] | undefined {
  if (!spec) return undefined
  const [type, opts = {}] = spec
  if (Array.isArray(type)) return type.map(String)
  if (type !== 'COMBO' && type !== 'COMFY_DYNAMICCOMBO_V3') return undefined
  return ((opts.options ?? []) as any[]).map((o) => (o && typeof o === 'object' && 'key' in o ? String(o.key) : String(o)))
}

/** The node definition's inputs in order, required then optional */
export function inputsOf(def: NodeDef): [string, InputSpec][] {
  const req = def.input.required ?? {}
  const opt = def.input.optional ?? {}
  const order = [...(def.input_order?.required ?? Object.keys(req)), ...(def.input_order?.optional ?? Object.keys(opt))]
  return order.map((n) => [n, (req[n] ?? opt[n]) as InputSpec])
}

function decimals(step: number): number {
  const s = String(step)
  return s.includes('.') ? s.split('.')[1].length : 0
}

function addInput(node: LGraphNode, name: string, spec: InputSpec) {
  const opts = spec[1] ?? {}
  const type = typeOf(spec)
  const values = comboValues(spec)
  if (values) {
    node.addWidget('combo', name, String(opts.default ?? values[0] ?? ''), null, { values })
  } else if (type === 'INT') {
    node.addWidget('number', name, opts.default ?? 0, null, {
      min: opts.min ?? 0,
      max: opts.max ?? Number.MAX_SAFE_INTEGER,
      step2: opts.step ?? 1,
      precision: 0,
    } as any)
    if (opts.control_after_generate || name === 'seed' || name === 'noise_seed') {
      node.addWidget('combo', CONTROL, 'randomize', null, { values: ['fixed', 'increment', 'decrement', 'randomize'] })
    }
  } else if (type === 'FLOAT') {
    const step = opts.round ?? opts.step ?? 0.01
    node.addWidget('number', name, opts.default ?? 0, null, {
      min: opts.min ?? -Number.MAX_VALUE,
      max: opts.max ?? Number.MAX_VALUE,
      step2: opts.step ?? 0.01,
      precision: Math.min(4, decimals(step)),
    } as any)
  } else if (type === 'STRING') {
    if (opts.multiline) addTextarea(node, name, opts.default ?? '')
    else node.addWidget('text', name, opts.default ?? '', null, {})
  } else if (type === 'BOOLEAN') {
    node.addWidget('toggle', name, opts.default ?? false, null, { on: opts.label_on ?? 'true', off: opts.label_off ?? 'false' })
  } else {
    node.addInput(name, type as string)
  }
}

/** Pictures in the node's free space below its widgets (Save Image's last outputs). */
function drawPreviews(this: LGraphNode, ctx: CanvasRenderingContext2D) {
  const imgs = previews.get(String(this.id))
  if (!imgs?.length || this.flags.collapsed) return
  // below the last widget (LiteGraph has laid them out by now), else below the slots
  const last = this.widgets?.at(-1) as any
  const slots = Math.max(this.inputs.length, this.outputs.length) * LiteGraph.NODE_SLOT_HEIGHT
  const top = last?.y != null ? last.y + (last.computedHeight ?? LiteGraph.NODE_WIDGET_HEIGHT) + 6 : slots + 8
  const [w, h] = [this.size[0] - 16, this.size[1] - top - 8]
  const img = imgs[imgs.length - 1]
  if (h < 24 || !img.complete || !img.naturalWidth) return
  const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight)
  const [dw, dh] = [img.naturalWidth * scale, img.naturalHeight * scale]
  ctx.drawImage(img, 8 + (w - dw) / 2, top + (h - dh) / 2, dw, dh)
}

function makeClass(type: string, def: NodeDef) {
  return class extends LGraphNode {
    static title = def.display_name || type
    static category = def.category
    static nodeDef = def
    constructor(title?: string) {
      super(title ?? def.display_name ?? type)
      this.comfyClass = type
      this.serialize_widgets = true
      for (const [name, spec] of inputsOf(def)) if (spec) addInput(this, name, spec)
      def.output.forEach((t, i) => this.addOutput(def.output_name?.[i] ?? t, t))
      const size = this.computeSize()
      this.size = [Math.max(size[0], type === 'SaveImage' ? 300 : 240), Math.max(size[1], type === 'SaveImage' ? 320 : type === 'CLIPTextEncode' ? 160 : 0)]
    }
    onRemoved() {
      dropTextareas(this)
    }
    onDrawBackground(ctx: CanvasRenderingContext2D) {
      drawPreviews.call(this, ctx)
    }
  }
}

/** Registers every node type the backend runs (again, after Models), replacing the old ones. */
export function registerNodes(objectInfo: ObjectInfo) {
  info = objectInfo
  for (const [type, def] of Object.entries(objectInfo)) {
    LiteGraph.registerNodeType(type, makeClass(type, def) as any)
  }
  registerVirtual()
}

/**
 * The editor-only nodes ComfyUI workflows carry, so they open with their places and values:
 * notes, legacy Reroute nodes, and PrimitiveNode (one value feeding widget inputs, its own
 * "control after generate" with it). None of them reaches the backend.
 */
function registerVirtual() {
  if (LiteGraph.registered_node_types.Reroute) return
  const note = (type: string) =>
    class extends LGraphNode {
      static title = type === 'MarkdownNote' ? 'Markdown Note' : 'Note'
      static category = 'utils'
      constructor(title?: string) {
        super(title ?? type)
        this.serialize_widgets = true
        this.color = '#432'
        this.bgcolor = '#653'
        addTextarea(this, 'text', '')
        this.size = [400, 160]
      }
      onRemoved() {
        dropTextareas(this)
      }
    }
  LiteGraph.registerNodeType('Note', note('Note') as any)
  LiteGraph.registerNodeType('MarkdownNote', note('MarkdownNote') as any)

  class Reroute extends LGraphNode {
    static title = 'Reroute'
    static category = 'utils'
    constructor(title?: string) {
      super(title ?? 'Reroute')
      this.addInput('', '*')
      this.addOutput('', '*')
      this.size = [75, 26]
      this.serialize_widgets = true
    }
  }
  LiteGraph.registerNodeType('Reroute', Reroute as any)

  class PrimitiveNode extends LGraphNode {
    static title = 'Primitive'
    static category = 'utils'
    constructor(title?: string) {
      super(title ?? 'Primitive')
      this.addOutput('connect to widget input', '*')
      this.serialize_widgets = true
    }
    onConfigure(data: any) {
      // a widget for the value it was saved with (ComfyUI makes it from the input it feeds)
      const [v, control] = data.widgets_values ?? []
      if (v === undefined || this.widgets?.length) return
      if (typeof v === 'number') this.addWidget('number', 'value', v, null, { precision: Number.isInteger(v) ? 0 : 3, step2: Number.isInteger(v) ? 1 : 0.01 } as any)
      else if (typeof v === 'boolean') this.addWidget('toggle', 'value', v, null, {})
      else this.addWidget('text', 'value', String(v), null, {})
      if (['fixed', 'increment', 'decrement', 'randomize'].includes(control)) {
        this.addWidget('combo', CONTROL, control, null, { values: ['fixed', 'increment', 'decrement', 'randomize'] })
      }
    }
  }
  LiteGraph.registerNodeType('PrimitiveNode', PrimitiveNode as any)
}

/**
 * A stand-in for a node type the backend does not run (yet): it keeps the workflow's slots and
 * values so the graph reads and saves unchanged, and shows red.
 */
export function registerMissing(type: string) {
  if (LiteGraph.registered_node_types[type]) return
  class Missing extends LGraphNode {
    static title = type
    static category = 'missing'
    constructor(title?: string) {
      super(title ?? type)
      this.comfyClass = type
      this.serialize_widgets = true
      ;(this as any).missing = true
      this.color = '#7a2828'
      this.bgcolor = '#4a1f1f'
    }
    saved: any = null
    onConfigure(data: any) {
      this.saved = data
    }
    onSerialize(out: any) {
      // the workflow's own widget values, which this stand-in has no widgets for
      if (this.saved?.widgets_values) out.widgets_values = this.saved.widgets_values
    }
  }
  LiteGraph.registerNodeType(type, Missing as any)
}
