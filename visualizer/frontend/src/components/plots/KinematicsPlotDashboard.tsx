import { useMemo } from 'react'
import type { KinematicsRunResult } from '../../types/data'
import { PlotGroup } from './PlotGroup'
import { TimelinePlot } from './TimelinePlot'

const COLORS = ['#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#fb923c']

export function KinematicsPlotDashboard({ result }: { result: KinematicsRunResult }) {
  const x = useMemo(
    () => Array.from({ length: result.n_waypoints }, (_, i) => i),
    [result.n_waypoints],
  )

  const jointSeries = useMemo(() => {
    const jd = result.joints_deg
    if (!jd || jd.length === 0) return []
    const nJ = jd[0].length
    return Array.from({ length: nJ }, (_, j) => ({
      name: `J${j + 1}`,
      y: jd.map((row) => row[j] ?? 0),
      color: COLORS[j % COLORS.length],
    }))
  }, [result.joints_deg])

  const tcpSeries = useMemo(() => {
    const t = result.tcp_xyz
    if (!t || t.length === 0) return []
    return [
      { name: 'x', y: t.map((p) => p[0] * 1000), color: '#38bdf8' },
      { name: 'y', y: t.map((p) => p[1] * 1000), color: '#a78bfa' },
      { name: 'z', y: t.map((p) => p[2] * 1000), color: '#f472b6' },
    ]
  }, [result.tcp_xyz])

  return (
    <div className="space-y-1">
      <PlotGroup title="Joint angles (deg)">
        <TimelinePlot
          plotId="kin_joints"
          title=""
          x={x}
          series={jointSeries}
          xLabel="Index"
          yLabel="deg"
          height={220}
        />
      </PlotGroup>
      <PlotGroup title="TCP position (mm)">
        <TimelinePlot
          plotId="kin_tcp"
          title=""
          x={x}
          series={tcpSeries}
          xLabel="Index"
          yLabel="mm"
          height={200}
        />
      </PlotGroup>
    </div>
  )
}
