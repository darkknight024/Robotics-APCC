import { useEffect } from 'react'
import type { ComponentType } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { PlotParams } from 'react-plotly.js'
import ReactPlotlyImport from 'react-plotly.js'
import { registerPlotDiv, unregisterPlotDiv } from '../../hooks/usePlotCursors'

/** Vite/Rolldown CJS interop can leave `default` nested — React needs a function/class, not `{ default: ... }`. */
function resolveReactPlotly(): ComponentType<PlotParams> {
  let m: unknown = ReactPlotlyImport
  for (let i = 0; i < 4; i++) {
    if (typeof m === 'function') return m as ComponentType<PlotParams>
    if (m && typeof m === 'object' && 'default' in m) {
      m = (m as { default: unknown }).default
    } else {
      break
    }
  }
  throw new Error('react-plotly.js: could not resolve a React component export')
}

const Plot = resolveReactPlotly()

type Props = {
  plotId: string
  title: string
  x: number[]
  series: { name: string; y: number[]; color?: string }[]
  xLabel?: string
  yLabel?: string
  height?: number
}

export function TimelinePlot({
  plotId,
  title,
  x,
  series,
  xLabel = 'Waypoint',
  yLabel = '',
  height = 200,
}: Props) {
  const data: Data[] = series.map((s) => ({
    x,
    y: s.y,
    name: s.name,
    type: 'scatter',
    mode: 'lines',
    line: { color: s.color || '#94a3b8', width: 1.5 },
  }))

  const layout: Partial<Layout> = {
    title: { text: title, font: { size: 11, color: '#a1a1b5' } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#12121a',
    font: { color: '#a1a1b5', size: 10 },
    margin: { l: 48, r: 12, t: 36, b: 40 },
    showlegend: series.length <= 8,
    legend: { orientation: 'h', y: -0.22, font: { size: 9 } },
    xaxis: { title: { text: xLabel }, gridcolor: '#2d2d3d', zeroline: false },
    yaxis: { title: { text: yLabel }, gridcolor: '#2d2d3d', zeroline: false },
    shapes: [
      {
        type: 'line',
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: '#ef4444', width: 2 },
      },
    ],
    height,
    autosize: true,
  }

  useEffect(() => {
    return () => unregisterPlotDiv(plotId)
  }, [plotId])

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%', minHeight: height }}
      onInitialized={(_fig, graphDiv) => {
        registerPlotDiv(plotId, graphDiv)
      }}
    />
  )
}
