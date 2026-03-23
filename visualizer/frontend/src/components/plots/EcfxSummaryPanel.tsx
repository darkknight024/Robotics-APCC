import { useMemo } from 'react'
import type { KinematicsRunResult } from '../../types/data'
import { ecfxDiscreteColor } from './ecfxColors'

function labelKey(ecfx: { cf1: number; cf4: number; cf6: number; cfx: number }): string {
  return `(${ecfx.cf1},${ecfx.cf4},${ecfx.cf6})`
}

export function EcfxSummaryPanel({
  result,
  timelineIndex,
}: {
  result: KinematicsRunResult
  timelineIndex: number
}) {
  const { columns, rows } = useMemo(() => {
    const wpx = result.waypoint_ecfx
    if (!wpx) return { columns: [] as string[], rows: [] as number[][] }
    const labels = new Set<string>()
    for (const wp of wpx) {
      if (!wp?.solutions) continue
      for (const s of wp.solutions) {
        labels.add(labelKey(s.ecfx))
      }
    }
    const columns = Array.from(labels).sort()
    const rows: number[][] = []
    for (let i = 0; i < result.n_waypoints; i++) {
      const wp = wpx[i]
      const counts = columns.map((lab) => {
        if (!wp?.solutions) return 0
        return wp.solutions.filter((s) => labelKey(s.ecfx) === lab).length
      })
      rows.push(counts)
    }
    return { columns, rows }
  }, [result.waypoint_ecfx, result.n_waypoints])

  if (columns.length === 0) return null

  return (
    <div className="overflow-x-auto border border-border rounded-md bg-surface-2/50">
      <table className="w-full text-xxs font-mono">
        <thead>
          <tr className="text-text-muted border-b border-border">
            <th className="text-left p-1 sticky left-0 bg-surface-2">wp</th>
            {columns.map((c) => (
              <th key={c} className="p-1 text-center min-w-[4rem]">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className={
                i === timelineIndex ? 'bg-accent-blue/15 ring-1 ring-accent-blue/40' : 'hover:bg-surface-1/80'
              }
            >
              <td className="p-1 sticky left-0 bg-inherit">{i}</td>
              {r.map((cell, j) => {
                const lab = columns[j]!
                const m = lab.match(/\((-?\d+),(-?\d+),(-?\d+)\)/)
                const cf1 = m ? parseInt(m[1]!, 10) : 0
                const bg = ecfxDiscreteColor(cf1)
                return (
                  <td
                    key={j}
                    className="p-1 text-center"
                    style={{
                      backgroundColor: cell > 0 ? `${bg}33` : undefined,
                    }}
                  >
                    {cell || '—'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
