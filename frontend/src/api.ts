// Talking to rustyui-backend. ComfyUI's API is under /api; the editor's own routes (its status
// and the history of everything the backend has run) are under /rustyui. Both are below where the
// backend is: the editor is served at <base>/simple/, <base> empty on the backend's own address
// and the proxy's path when a reverse proxy serves it below one (e.g. /rustyui).
export const BASE = location.pathname.replace(/\/simple(\/.*)?$/, '')

export interface RequestOptions {
  method?: string
  body?: unknown
}

export async function request(url: string, opts: RequestOptions = {}): Promise<any> {
  const init: RequestInit = { method: opts.method ?? 'GET' }
  if (opts.body !== undefined) {
    init.body = typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  const r = await fetch(url, init)
  const text = await r.text()
  let body: any = text
  try {
    body = JSON.parse(text)
  } catch {
    // plain text
  }
  if (!r.ok) {
    const e = body && body.error
    const msg = (e && (e.message || e.details || e)) || (typeof body === 'string' && body) || r.statusText
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return body
}

export const api = (path: string, opts?: RequestOptions) => request(BASE + '/api' + path, opts)
export const ui = (path: string, opts?: RequestOptions) => request(BASE + '/rustyui' + path, opts)

/** a video output (SaveVideo's MP4), played rather than shown */
export const isVideo = (o: { filename: string }) => /\.(mp4|webm)$/i.test(o.filename)

export const viewUrl = (o: { filename: string; subfolder?: string }) =>
  `${BASE}/api/view?filename=${encodeURIComponent(o.filename)}&subfolder=${encodeURIComponent(o.subfolder ?? '')}`

/** One input of a node in /object_info: a type name (or a list of choices) and its options */
export type InputSpec = [string | unknown[], Record<string, any>?]

export interface NodeDef {
  name: string
  display_name: string
  category: string
  description?: string
  input: { required?: Record<string, InputSpec>; optional?: Record<string, InputSpec>; hidden?: Record<string, unknown> }
  input_order?: { required?: string[]; optional?: string[] }
  output: string[]
  output_name?: string[]
  output_node?: boolean
}

export type ObjectInfo = Record<string, NodeDef>

export interface Status {
  engine: string
  loaded: boolean
  running: number
  pending: number
  executing: { prompt_id: string; node: string; class_type: string; value: number; max: number } | null
  history_version: number
  /** whether this backend has the prompt writer (an LLM, `serve --writer`) */
  writer?: boolean
}

export interface HistoryItem {
  id: string
  number: number
  title: string | null
  queued: number
  started?: number
  finished?: number
  status: 'queued' | 'running' | 'success' | 'error' | 'lost'
  outputs: { node: string; filename: string; subfolder: string }[]
  error: string | null
  seconds: number | null
}
