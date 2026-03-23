import { useMemo, useState } from 'react'
import type { KinematicsRunResult } from '../../types/data'
import { useAnalysisStore } from '../../stores/analysisStore'
import { EcfxBranchPlot } from './EcfxBranchPlot'
import { EcfxSummaryPanel } from './EcfxSummaryPanel'
import { PlotGroup } from './PlotGroup'
import { TimelinePlot } from './TimelinePlot'

const COLORS = ['#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#fb923c']

export function KinematicsPlotDashboard({ result }: { result: KinematicsRunResult }) {
  const timelineIndex = useAnalysisStore((s) => s.timelineIndex)
  const [showEcfx, setShowEcfx] = useState(false)
  const hasEcfx =
    result.kind === 'ik' &&
    result.solver === 'eaik' &&
    Array.isArray(result.waypoint_ecfx) &&
    result.waypoint_ecfx.some((w) => w && w.solutions && w.solutions.length > 0)
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
      {hasEcfx && (
        <div className="flex items-center gap-2 py-1">
          <label className="flex items-center gap-2 text-xxs text-text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={showEcfx}
              onChange={(e) => setShowEcfx(e.target.checked)}
              className="accent-accent-blue"
            />
            ECFX details (branch plots + summary)
          </label>
        </div>
      )}
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
      {showEcfx && hasEcfx && (
        <PlotGroup title="ECFX — cf1 / J1">
          <EcfxBranchPlot
            plotId="ecfx_cf1"
            title="J1 vs waypoint (color = cf1)"
            result={result}
            jointIndex={0}
            cfKey="cf1"
            height={200}
          />
        </PlotGroup>
      )}
      {showEcfx && hasEcfx && (
        <PlotGroup title="ECFX — cf4 / J4">
          <EcfxBranchPlot
            plotId="ecfx_cf4"
            title="J4 vs waypoint (color = cf4)"
            result={result}
            jointIndex={3}
            cfKey="cf4"
            height={200}
          />
        </PlotGroup>
      )}
      {showEcfx && hasEcfx && (
        <PlotGroup title="ECFX — cf6 / J6">
          <EcfxBranchPlot
            plotId="ecfx_cf6"
            title="J6 vs waypoint (color = cf6)"
            result={result}
            jointIndex={5}
            cfKey="cf6"
            height={200}
          />
        </PlotGroup>
      )}
      {showEcfx && hasEcfx && (
        <PlotGroup title="ECFX summary (counts per label)">
          <EcfxSummaryPanel result={result} timelineIndex={timelineIndex} />
        </PlotGroup>
      )}
    </div>
  )
}
