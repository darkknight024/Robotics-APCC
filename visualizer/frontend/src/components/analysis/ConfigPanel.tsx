import { useAnalysisStore } from '../../stores/analysisStore'

export function ConfigPanel() {
  const mode = useAnalysisStore((s) => s.analysisMode)
  const runConfig = useAnalysisStore((s) => s.runConfig)
  const setRunConfig = useAnalysisStore((s) => s.setRunConfig)
  const detection = useAnalysisStore((s) => s.detectionResult)
  const setStep = useAnalysisStore((s) => s.setStep)

  const maxTraj = Math.max(0, (detection?.num_trajectories ?? 1) - 1)
  const feas = runConfig.feasibility ?? {}

  return (
    <div className="p-3 space-y-3 text-xs">
      <div className="space-y-1">
        <label className="text-xxs text-text-muted uppercase">Configuration</label>
        <div className="flex gap-2">
          {(['eaik', 'pin'] as const).map((s) => (
            <button
              key={s}
              type="button"
              className={`flex-1 py-1.5 rounded text-xxs font-mono border ${
                runConfig.solver === s ? 'border-accent-blue bg-accent-blue/15' : 'border-border'
              }`}
              onClick={() => setRunConfig({ solver: s })}
            >
              {s === 'eaik' ? 'EAIK' : 'Pinocchio'}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-1">
        <label className="text-xxs text-text-muted uppercase">EE link (URDF frame)</label>
        <input
          className="input-field w-full text-xs font-mono"
          placeholder="ee_link"
          value={runConfig.ee_frame_name}
          onChange={(e) => setRunConfig({ ee_frame_name: e.target.value || 'ee_link' })}
        />
        <p className="text-xxs text-text-muted">Must match a link name in the robot URDF (validated on load).</p>
      </div>

      {mode !== 'feasibility' && (
        <div className="space-y-1">
          <label className="text-xxs text-text-muted uppercase">Trajectory index</label>
          <input
            type="number"
            min={0}
            max={maxTraj}
            className="input-field w-full text-xs font-mono"
            value={runConfig.trajectory_index}
            onChange={(e) => setRunConfig({ trajectory_index: Math.max(0, Number(e.target.value) || 0) })}
          />
          <p className="text-xxs text-text-muted">0-based (multi-T0 files)</p>
        </div>
      )}

      {mode === 'feasibility' && (
        <div className="space-y-2 max-h-[55vh] overflow-y-auto pr-1 border border-border rounded-md p-2">
          <div className="space-y-1">
            <label className="text-xxs text-text-muted uppercase">TCP speed (mm/s)</label>
            <input
              type="number"
              min={1}
              className="input-field w-full text-xs font-mono"
              value={runConfig.speed_mm_s ?? 100}
              onChange={(e) => setRunConfig({ speed_mm_s: Math.max(1, Number(e.target.value) || 100) })}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xxs text-text-muted">Max IK failures / trajectory</label>
            <input
              type="number"
              min={0}
              className="input-field w-full text-xs font-mono"
              value={feas.max_ik_failures_per_trajectory ?? 1}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    max_ik_failures_per_trajectory: Math.max(0, Number(e.target.value) || 0),
                  },
                })
              }
            />
          </div>

          <p className="text-xxs text-text-muted uppercase pt-1">Singularity</p>
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.singularity?.enabled !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    singularity: { ...feas.singularity, enabled: e.target.checked },
                  },
                })
              }
            />
            Enabled
          </label>
          <label className="text-xxs text-text-muted">σ_min threshold</label>
          <input
            type="number"
            step={0.001}
            className="input-field w-full text-xs font-mono"
            value={feas.singularity?.threshold ?? 0.01}
            onChange={(e) =>
              setRunConfig({
                feasibility: {
                  ...feas,
                  singularity: { ...feas.singularity, threshold: Number(e.target.value) || 0.01 },
                },
              })
            }
          />
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.singularity?.check_j5_only !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    singularity: { ...feas.singularity, check_j5_only: e.target.checked },
                  },
                })
              }
            />
            Check J5 only (classified mode)
          </label>

          <p className="text-xxs text-text-muted uppercase pt-1">Manipulability</p>
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.manipulability?.enabled !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    manipulability: { ...feas.manipulability, enabled: e.target.checked },
                  },
                })
              }
            />
            Enabled
          </label>
          {(['warning', 'translational_warning', 'rotational_warning', 'directional_warning'] as const).map((k) => (
            <div key={k} className="space-y-0.5">
              <label className="text-xxs text-text-muted">{k}</label>
              <input
                type="number"
                step={0.0001}
                className="input-field w-full text-xs font-mono"
                value={(feas.manipulability?.[k] as number) ?? 0.001}
                onChange={(e) =>
                  setRunConfig({
                    feasibility: {
                      ...feas,
                      manipulability: { ...feas.manipulability, [k]: Number(e.target.value) || 0 },
                    },
                  })
                }
              />
            </div>
          ))}

          <p className="text-xxs text-text-muted uppercase pt-1">Continuity (C0/C1)</p>
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.continuity?.enabled !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    continuity: { ...feas.continuity, enabled: e.target.checked },
                  },
                })
              }
            />
            Enabled
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xxs text-text-muted">pose_scale_m_per_rad</label>
              <input
                type="number"
                step={0.01}
                className="input-field w-full text-xs font-mono"
                value={feas.continuity?.pose_scale_m_per_rad ?? 0.1}
                onChange={(e) =>
                  setRunConfig({
                    feasibility: {
                      ...feas,
                      continuity: { ...feas.continuity, pose_scale_m_per_rad: Number(e.target.value) || 0.1 },
                    },
                  })
                }
              />
            </div>
            <div>
              <label className="text-xxs text-text-muted">safety_factor</label>
              <input
                type="number"
                step={0.01}
                className="input-field w-full text-xs font-mono"
                value={feas.continuity?.safety_factor ?? 1.05}
                onChange={(e) =>
                  setRunConfig({
                    feasibility: {
                      ...feas,
                      continuity: { ...feas.continuity, safety_factor: Number(e.target.value) || 1 },
                    },
                  })
                }
              />
            </div>
          </div>

          <p className="text-xxs text-text-muted uppercase pt-1">Waypoint density</p>
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.waypoint_density?.enabled !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    waypoint_density: { ...feas.waypoint_density, enabled: e.target.checked },
                  },
                })
              }
            />
            Enabled
          </label>
          <label className="text-xxs text-text-muted">max_gap_mm</label>
          <input
            type="number"
            className="input-field w-full text-xs font-mono"
            value={feas.waypoint_density?.max_gap_mm ?? 5}
            onChange={(e) =>
              setRunConfig({
                feasibility: {
                  ...feas,
                  waypoint_density: { ...feas.waypoint_density, max_gap_mm: Number(e.target.value) || 5 },
                },
              })
            }
          />
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.waypoint_density?.interpolate_sparse === true}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    waypoint_density: { ...feas.waypoint_density, interpolate_sparse: e.target.checked },
                  },
                })
              }
            />
            Interpolate sparse segments
          </label>

          <p className="text-xxs text-text-muted uppercase pt-1">EAIK multi-solution</p>
          <label className="flex items-center gap-2 text-xxs">
            <input
              type="checkbox"
              checked={feas.eaik_multi_solution?.enabled !== false}
              onChange={(e) =>
                setRunConfig({
                  feasibility: {
                    ...feas,
                    eaik_multi_solution: { ...feas.eaik_multi_solution, enabled: e.target.checked },
                  },
                })
              }
            />
            Enabled
          </label>
        </div>
      )}

      <button type="button" className="btn-primary text-xxs w-full py-2" onClick={() => setStep('run')}>
        Continue to run
      </button>
    </div>
  )
}
