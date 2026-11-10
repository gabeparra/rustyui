import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'

// `bun run dev` serves the editor on 127.0.0.1:5173 with hot reload and passes the API on to the
// backend on 127.0.0.1:8190. The backend refuses another origin, which the dev server is, so the
// proxy drops the browser's Origin header; in exchange the dev server refuses, as the backend
// would, what a web page elsewhere could send through it.
const backend = 'http://127.0.0.1:8190'

const thisBoxOnly: Plugin = {
  name: 'rustyui-this-box-only',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const host = (req.headers.host ?? '').replace(/:\d+$/, '').replace(/^\[|\]$/g, '')
      const origin = req.headers.origin
      const site = req.headers['sec-fetch-site']
      const loopback = host === 'localhost' || host === '::1' || host.startsWith('127.')
      const foreign = (origin && origin !== `http://${req.headers.host}`) || (site && site !== 'same-origin' && site !== 'none')
      if (!loopback || foreign) {
        res.statusCode = 403
        res.end('the rustyui dev server answers this box only\n')
        return
      }
      next()
    })
  },
}

const proxied = {
  target: backend,
  changeOrigin: true,
  configure: (proxy: any) => proxy.on('proxyReq', (req: any) => req.removeHeader('origin')),
}

// served by the backend at /simple/ (ComfyUI's editor, forked in ../ui, is the one at /), or by a
// reverse proxy below a path of its own: the files are relative to the page so either works
export default defineConfig({
  base: './',
  plugins: [thisBoxOnly, vue()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: { '/api': proxied, '/rustyui': proxied } },
  preview: { host: '127.0.0.1' },
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 4000 },
})
