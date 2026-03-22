import { useEffect, useMemo } from 'react'
import type { FeasibilityRunResult } from '../../types/data'
import { postFeasibilityScene } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'
import { PlotGroup } from './PlotGroup'
import { TimelinePlot } from './TimelinePlot'

export function FeasibilityPlotDashboard({ result }: { result: FeasibilityRunResult }) {
  const sessionId = useAnalysisStore((s) => s.sessionId)
  const selectedTrajectoryIndex = useAnalysisStore((s) => s.selectedTrajectoryIndex)
  const setSelectedTrajectoryIndex = useAnalysisStore((s) => s.setSelectedTrajectoryIndex)

  useEffect(() => {
    if (!sessionId) return
    void postFeasibilityScene(sessionId, selectedTrajectoryIndex).catch(() => {})
  }, [sessionId, selectedTrajectoryIndex])

  const traj = result.trajectory_results[selectedTrajectoryIndex]
  const nTraj = result.num_trajectories ?? result.trajectory_results.length

  const x = useMemo(
    () => (traj ? Array.from({ length: traj.num_waypoints }, (_, i) => i) : []),
    [traj],
  )

  const singularitySeries = useMemo(() => {
    if (!traj) return []
    return [
      { name: 'min σ', y: traj.min_singular_value ?? [], color: '#f472b6' },
      { name: 'cond #', y: traj.condition_number ?? [], color: '#a78bfa' },
    ]
  }, [traj])

  const manipSeries = useMemo(() => {
    if (!traj) return []
    const ns = traj.near_singularity ?? []
    return [
      { name: 'manipulability', y: traj.manipulability ?? [], color: '#38bdf8' },
      ...(ns.length
        ? [{ name: 'near singularity (0/1)', y: ns.map((b) => (b ? 1 : 0)), color: '#f97316' }]
        : []),
    ]
  }, [traj])

  const c0x = useMemo(() => {
    if (!traj?.joint_space_distances?.length) return []
    return traj.joint_space_distances.map((_, i) => i)
  }, [traj])

  const c0series = useMemo(() => {
    if (!traj?.joint_space_distances) return []
    return [{ name: 'joint space dist (rad)', y: traj.joint_space_distances, color: '#34d399' }]
  }, [traj])

  const toppT = traj?.topp_series?.t_samples_s ?? []
  const toppJointSeries = useMemo(() => {
    if (!traj?.topp_series?.q_rad?.[0]) return []
    const q = traj.topp_series.q_rad
    const cols = ['#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#fb923c']
    return Array.from({ length: Math.min(6, q[0]?.length ?? 0) }, (_, j) => ({
      name: `J${j + 1}`,
      y: q.map((row) => row[j] ?? 0),
      color: cols[j % cols.length],
    }))
  }, [traj])

  if (!traj) {
    return <p className="text-xxs text-text-muted p-2">No trajectory data.</p>
  }

  return (
    <div className="space-y-2">
      {nTraj > 1 && (
        <div className="flex items-center gap-2 text-xxs">
          <span className="text-text-muted">Trajectory</span>
          <select
            className="input-field text-xs font-mono flex-1"
            value={selectedTrajectoryIndex}
            onChange={(e) => setSelectedTrajectoryIndex(Number(e.target.value))}
          >
            {Array.from({ length: nTraj }, (_, i) => (
              <option key={i} value={i}>
                {i + 1} / {nTraj}
              </option>
            ))}
          </select>
        </div>
      )}

      <PlotGroup title="Feasibility — Singularity" defaultOpen>
        <TimelinePlot
          plotId={`feas_singularity_${selectedTrajectoryIndex}`}
          title="Minimum singular value & condition number"
          x={x}
          series={singularitySeries}
          xLabel="Waypoint"
          yLabel="value"
          height={180}
        />
      </PlotGroup>

      <PlotGroup title="Feasibility — Manipulability" defaultOpen>
        <TimelinePlot
          plotId={`feas_manip_${selectedTrajectoryIndex}`}
          title="Manipulability (and near-singularity flags)"
          x={x}
          series={manipSeries}
          xLabel="Waypoint"
          yLabel="value"
          height={180}
        />
      </PlotGroup>

      {c0x.length > 0 && (
        <PlotGroup title="Feasibility — C0 joint distance" defaultOpen>
          <TimelinePlot
            plotId={`feas_c0_${selectedTrajectoryIndex}`}
            title="Joint-space step distance"
            x={c0x}
            series={c0series}
            xLabel="Segment"
            yLabel="rad"
            height={160}
          />
        </PlotGroup>
      )}

      {toppT.length > 0 && toppJointSeries.length > 0 && (
        <PlotGroup title="TOPP-RA (time)" defaultOpen>
          <TimelinePlot
            plotId={`topp_joints_${selectedTrajectoryIndex}`}
            title="Joint positions vs time"
            x={toppT}
            series={toppJointSeries}
            xLabel="t (s)"
            yLabel="rad"
            height={220}
          />
        </PlotGroup>
      )}
    </div>
  )
}
