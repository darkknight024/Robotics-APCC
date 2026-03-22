import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
/** Scatter/line only — avoids probe-image-size → Node `stream` (breaks Vite dev). */
const plotlyCartesian = path.resolve(
  __dirname,
  'node_modules/plotly.js/dist/plotly-cartesian.min.js',
)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // plotly.js → has-hover expects Node's `global` (dev + browser bundle)
  define: {
    global: 'globalThis',
  },
  resolve: {
    alias: {
      // plotly.js (via image trace) imports `buffer/` — map to npm buffer
      buffer: 'buffer',
      'buffer/': 'buffer',
      // react-plotly.js requires `plotly.js/dist/plotly` — full `plotly.js` pulls probe-image-size / Node stream
      'plotly.js/dist/plotly': plotlyCartesian,
    },
  },
  optimizeDeps: {
    include: ['plotly.js/dist/plotly'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
      },
    },
  },
})
