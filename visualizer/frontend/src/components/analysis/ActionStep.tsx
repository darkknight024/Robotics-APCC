import { useAnalysisStore } from '../../stores/analysisStore'
import type { AnalysisMode } from '../../types/data'

export function ActionStep() {
  const detection = useAnalysisStore((s) => s.detectionResult)
  const mode = useAnalysisStore((s) => s.analysisMode)
  const setMode = useAnalysisStore((s) => s.setAnalysisMode)
  const setStep = useAnalysisStore((s) => s.setStep)

  const canIk = detection?.has_task_space
  const canFk = detection?.has_joint_space

  const pick = (m: AnalysisMode) => {
    setMode(m)
    setStep('config')
  }

  return (
    <div className="p-3 space-y-3 text-xs">
      <p className="text-text-muted text-xxs">Choose analysis mode (IK/FK or full feasibility).</p>
      <div className="space-y-2">
        {canIk && (
          <button
            type="button"
            className={`w-full text-left px-3 py-2 rounded-md border text-xxs ${
              mode === 'feasibility' ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:bg-surface-3'
            }`}
            onClick={() => pick('feasibility')}
          >
            <div className="font-medium text-text-primary">Feasibility pipeline</div>
            <div className="text-text-muted">IK, TOPP-RA, continuity, singularity, manipulability</div>
          </button>
        )}
        {canIk && (
          <button
            type="button"
            className={`w-full text-left px-3 py-2 rounded-md border text-xxs ${
              mode === 'ik_only' ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:bg-surface-3'
            }`}
            onClick={() => pick('ik_only')}
          >
            <div className="font-medium text-text-primary">Run IK</div>
            <div className="text-text-muted">Solve IK on task-space poses</div>
          </button>
        )}
        {canFk && (
          <button
            type="button"
            className={`w-full text-left px-3 py-2 rounded-md border text-xxs ${
              mode === 'fk_only' ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:bg-surface-3'
            }`}
            onClick={() => pick('fk_only')}
          >
            <div className="font-medium text-text-primary">Run FK</div>
            <div className="text-text-muted">Forward kinematics on joint columns</div>
          </button>
        )}
        {!canIk && !canFk && (
          <p className="text-amber-500 text-xxs">No task or joint pose detected. Re-upload a valid CSV.</p>
        )}
      </div>
    </div>
  )
}
