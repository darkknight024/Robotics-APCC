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

/** Waypoint-index plots use `waypointIndex`; `topp_*` plot IDs use `timeS` when provided. */
export function broadcastCursorUpdate(waypointIndex: number, timeS?: number) {
  for (const [plotId, div] of registry.entries()) {
    const x = plotId.startsWith('topp_') && timeS !== undefined ? timeS : waypointIndex
    try {
      void Plotly.relayout(div, {
        'shapes[0].x0': x,
        'shapes[0].x1': x,
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
