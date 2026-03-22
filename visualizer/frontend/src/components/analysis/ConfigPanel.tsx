import { useAnalysisStore } from '../../stores/analysisStore'

export function ConfigPanel() {
  const runConfig = useAnalysisStore((s) => s.runConfig)
  const setRunConfig = useAnalysisStore((s) => s.setRunConfig)
  const detection = useAnalysisStore((s) => s.detectionResult)
  const setStep = useAnalysisStore((s) => s.setStep)

  const maxTraj = Math.max(0, (detection?.num_trajectories ?? 1) - 1)

  return (
    <div className="p-3 space-y-3 text-xs">
      <div className="space-y-1">
        <label className="text-xxs text-text-muted uppercase">Solver</label>
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
        <label className="text-xxs text-text-muted uppercase">EE frame</label>
        <input
          className="input-field w-full text-xs font-mono"
          value={runConfig.ee_frame_name}
          onChange={(e) => setRunConfig({ ee_frame_name: e.target.value })}
        />
      </div>
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
      <button type="button" className="btn-primary text-xxs w-full py-2" onClick={() => setStep('run')}>
        Continue to run
      </button>
    </div>
  )
}
