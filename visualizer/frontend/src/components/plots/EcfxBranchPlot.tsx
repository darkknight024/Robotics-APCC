import { useEffect, useMemo } from 'react'
import type { ComponentType } from 'react'
import type { Data, Layout } from 'plotly.js'
import type { PlotParams } from 'react-plotly.js'
import ReactPlotlyImport from 'react-plotly.js'
import { registerPlotDiv, unregisterPlotDiv } from '../../hooks/usePlotCursors'
import type { KinematicsRunResult } from '../../types/data'
import { ecfxDiscreteColor } from './ecfxColors'

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

type CfKey = 'cf1' | 'cf4' | 'cf6'

type Props = {
  plotId: string
  title: string
  result: KinematicsRunResult
  jointIndex: number
  cfKey: CfKey
  height?: number
}

export function EcfxBranchPlot({
  plotId,
  title,
  result,
  jointIndex,
  cfKey,
  height = 200,
}: Props) {
  const nWp = result.n_waypoints
  const dense = nWp > 200

  const { xs, ys, colors, sizes, symbols, lineWidths, lineColors, opacities } = useMemo(() => {
    const xs: number[] = []
    const ys: number[] = []
    const colors: string[] = []
    const sizes: number[] = []
    const symbols: string[] = []
    const lineWidths: number[] = []
    const lineColors: string[] = []
    const opacities: number[] = []
    const wpx = result.waypoint_ecfx
    if (!wpx) {
      return { xs, ys, colors, sizes, symbols, lineWidths, lineColors, opacities }
    }
    for (let i = 0; i < nWp; i++) {
      const wp = wpx[i]
      if (!wp?.solutions?.length) continue
      const sel = wp.selected_index
      for (const sol of wp.solutions) {
        const jdeg = sol.joint_angles_deg[jointIndex] ?? 0
        const cf = sol.ecfx[cfKey]
        const c = ecfxDiscreteColor(cf)
        const isSel = sol.branch_index === sel
        const isLs = sol.is_ls
        xs.push(i)
        ys.push(jdeg)
        colors.push(c)
        const baseOp = isLs ? 0.45 : 1
        const op = dense && !isSel ? 0.2 * baseOp : baseOp
        opacities.push(op)
        symbols.push(isLs ? 'circle-open' : 'circle')
        sizes.push(isSel ? 11 : 6)
        lineWidths.push(isSel ? 2 : isLs ? 1.5 : 0)
        lineColors.push(isLs ? c : isSel ? '#0f172a' : 'rgba(0,0,0,0)')
      }
    }
    return { xs, ys, colors, sizes, symbols, lineWidths, lineColors, opacities }
  }, [result.waypoint_ecfx, nWp, jointIndex, cfKey, dense])

  const data: Data[] = [
    {
      x: xs,
      y: ys,
      type: 'scatter',
      mode: 'markers',
      marker: {
        color: colors,
        size: sizes,
        opacity: opacities,
        symbol: symbols,
        line: {
          color: lineColors,
          width: lineWidths,
        },
      },
    },
  ]

  const layout: Partial<Layout> = {
    title: { text: title, font: { size: 11, color: '#a1a1b5' } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#12121a',
    font: { color: '#a1a1b5', size: 10 },
    margin: { l: 48, r: 12, t: 36, b: 40 },
    showlegend: false,
    xaxis: { title: { text: 'Waypoint' }, gridcolor: '#2d2d3d', zeroline: false },
    yaxis: {
      title: { text: `J${jointIndex + 1} (deg)` },
      gridcolor: '#2d2d3d',
      zeroline: false,
    },
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

  if (xs.length === 0) return null

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
