import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { configureSession } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'
import { RobotSelector } from './RobotSelector'

export function RobotStep() {
  const {
    sessionId,
    selectedRobot,
    useBaseFrame,
    selectedKnife,
    detectionResult,
  } = useAnalysisStore()
  const [busy, setBusy] = useState(false)

  const apply = async () => {
    if (!sessionId || !selectedRobot) {
      toast.error('Select a robot')
      return
    }
    setBusy(true)
    const json = await configureSession(sessionId, {
      use_base_frame: useBaseFrame,
      knife_name: useBaseFrame ? null : selectedKnife,
      robot_name: selectedRobot,
    })
    setBusy(false)
    if (!json.ok || !json.data) {
      toast.error(json.error || 'Configure failed')
      return
    }
    const n = json.data.transformed_waypoints_preview?.length ?? 0
    toast.success(
      n > 0
        ? `Preview loaded (${n} waypoints). Check the 3D view.`
        : 'Robot loaded (no TCP path for this CSV).',
    )
  }

  const hasTask = detectionResult?.has_task_space ?? true

  return (
    <div className="p-3 space-y-3">
      <RobotSelector skipInitialViserLoad />
      <p className="text-xxs text-text-muted">
        {hasTask
          ? 'Loads the URDF and draws the TCP path in the 3D view (green).'
          : 'Joint-only CSV: loads the robot model without a task-space path preview.'}
      </p>
      <button
        type="button"
        className="btn-primary text-xxs w-full py-2 flex items-center justify-center gap-2"
        disabled={busy || !sessionId}
        onClick={() => void apply()}
      >
        {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
        Load robot & preview
      </button>
    </div>
  )
}
