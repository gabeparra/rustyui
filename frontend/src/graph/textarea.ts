// Multiline text widgets (prompts) as real <textarea>s over the canvas, as ComfyUI has them:
// LiteGraph draws the node, and each frame the textarea is moved and scaled onto the widget's
// place in it. A textarea whose node was not drawn this frame (scrolled away, collapsed) hides.
import type { LGraphCanvas, LGraphNode } from '@comfyorg/litegraph'

let overlay: HTMLElement | null = null
let canvas: LGraphCanvas | null = null
let frame = 0
const lastSeen = new Map<HTMLTextAreaElement, number>()

/** Hooks the textareas to a canvas and the element they float in (sized like the canvas). */
export function attachTextareas(c: LGraphCanvas, layer: HTMLElement) {
  canvas = c
  overlay = layer
  const draw = c.draw.bind(c)
  c.draw = (fg?: boolean, bg?: boolean) => {
    const foreground = Boolean(fg) || Boolean((c as any).dirty_canvas)
    if (foreground) frame++
    draw(fg, bg)
    if (foreground) {
      for (const [el, seen] of lastSeen) el.style.display = seen === frame ? '' : 'none'
    }
  }
}

export function addTextarea(node: LGraphNode, name: string, value: string) {
  const el = document.createElement('textarea')
  el.className = 'rui-textarea'
  el.spellcheck = false
  el.value = value
  el.placeholder = name
  el.style.display = 'none'
  // wheel and pointer stay in the textarea (keys reach the window, for Ctrl+Enter)
  for (const ev of ['pointerdown', 'wheel', 'contextmenu'] as const) {
    el.addEventListener(ev, (e) => e.stopPropagation())
  }
  el.addEventListener('input', () => node.graph?.setDirtyCanvas(true, false))
  overlay?.append(el)
  lastSeen.set(el, -1)

  const widget: any = node.addCustomWidget({
    type: 'customtext',
    name,
    value,
    options: { multiline: true },
    draw(this: any, _ctx: CanvasRenderingContext2D, n: LGraphNode, width: number, y: number) {
      if (!canvas) return
      const ds = canvas.ds
      const s = ds.scale
      const left = (n.pos[0] + 8 + ds.offset[0]) * s
      const top = (n.pos[1] + y + ds.offset[1]) * s
      // the height LiteGraph gave it: the node's free space, as ComfyUI's prompt boxes have
      const h = Math.max(40, (this.computedHeight ?? 64) - 6)
      el.style.transform = `translate(${left}px, ${top}px) scale(${s})`
      el.style.width = `${Math.max(40, width - 16)}px`
      el.style.height = `${h}px`
      lastSeen.set(el, frame)
    },
    computeLayoutSize() {
      return { minHeight: 64, maxHeight: 1e6, minWidth: 200, maxWidth: 1e6 }
    },
    onRemove() {
      el.remove()
      lastSeen.delete(el)
    },
  } as any)
  // the widget's value is the textarea's, whichever side sets it (typing, or a loaded workflow)
  Object.defineProperty(widget, 'value', {
    get: () => el.value,
    set: (v: unknown) => {
      el.value = v == null ? '' : String(v)
    },
    configurable: true,
  })
  widget.element = el
  return widget
}

/** Removes the textareas of a node that left the graph. */
export function dropTextareas(node: LGraphNode) {
  for (const w of (node.widgets ?? []) as any[]) {
    if (w.element instanceof HTMLTextAreaElement) {
      w.element.remove()
      lastSeen.delete(w.element)
    }
  }
}

/** Every textarea, for clearing the graph */
export function clearTextareas() {
  for (const el of lastSeen.keys()) el.remove()
  lastSeen.clear()
}
