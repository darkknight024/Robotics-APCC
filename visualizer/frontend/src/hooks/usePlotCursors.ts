import { useCallback, useRef } from 'react'
// Same entry as react-plotly.js — Vite aliases to cartesian bundle (no Node image-trace deps)
import * as Plotly from 'plotly.js/dist/plotly'

/** plotId -> Plotly graph div (HTMLElement from react-plotly onInitialized) */
const registry = new Map<string, HTMLElement>()

export function registerPlotDiv(plotId: string, graphDiv: HTMLElement) {
  registry.set(plotId, graphDiv)
}

export function unregisterPlotDiv(plotId: string) {
  registry.delete(plotId)
}

export function broadcastCursorUpdate(waypointIndex: number) {
  for (const div of registry.values()) {
    try {
      void Plotly.relayout(div, {
        'shapes[0].x0': waypointIndex,
        'shapes[0].x1': waypointIndex,
      } as Record<string, unknown>)
    } catch {
      /* ignore */
    }
  }
}

export function usePlotCursors() {
  const counter = useRef(0)
  const makeId = useCallback((prefix: string) => {
    counter.current += 1
    return `${prefix}_${counter.current}`
  }, [])

  return { registerPlotDiv, unregisterPlotDiv, broadcastCursorUpdate, makePlotId: makeId }
}
