import { useAnalysisStore } from '../../stores/analysisStore'

export function ResultsStep() {
  const { runResult } = useAnalysisStore()
  const setStep = useAnalysisStore((s) => s.setStep)

  if (!runResult) {
    return <p className="p-3 text-xs text-text-muted">No results yet.</p>
  }

  const successRate =
    runResult.kind === 'ik' && runResult.ik_success
      ? `${runResult.ik_success.filter(Boolean).length} / ${runResult.n_waypoints} IK ok`
      : `${runResult.n_waypoints} FK samples`

  return (
    <div className="p-3 space-y-2 text-xs">
      <p className="text-text-primary font-medium">Run complete</p>
      <p className="text-xxs text-text-muted font-mono">{successRate}</p>
      <p className="text-xxs text-text-muted">Kinematics plots are in the right panel. Scrub the timeline below the 3D view.</p>
      <button type="button" className="btn-secondary text-xxs w-full py-1.5" onClick={() => setStep('run')}>
        Run again
      </button>
    </div>
  )
}
