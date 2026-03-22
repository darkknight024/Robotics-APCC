import { useCallback, useEffect, useState } from 'react'
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import { toast } from 'sonner'
import { detectColumns } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'
import type { DetectionResult } from '../../types/data'

const ROLE_OPTIONS = [
  { value: 'ignore', label: 'Ignore' },
  { value: 'x', label: 'x (mm)' },
  { value: 'y', label: 'y (mm)' },
  { value: 'z', label: 'z (mm)' },
  { value: 'qw', label: 'qw' },
  { value: 'qx', label: 'qx' },
  { value: 'qy', label: 'qy' },
  { value: 'qz', label: 'qz' },
  { value: 'j1_deg', label: 'j1_deg' },
  { value: 'j2_deg', label: 'j2_deg' },
  { value: 'j3_deg', label: 'j3_deg' },
  { value: 'j4_deg', label: 'j4_deg' },
  { value: 'j5_deg', label: 'j5_deg' },
  { value: 'j6_deg', label: 'j6_deg' },
  { value: 'speed', label: 'speed' },
]

export function DetectStep() {
  const { sessionId, setDetectionResult, setStep } = useAnalysisStore()
  const [loading, setLoading] = useState(true)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [mapper, setMapper] = useState<Record<string, string>>({})

  const runDetect = useCallback(
    async (columnMap?: Record<string, string>) => {
      if (!sessionId) return
      setLoading(true)
      const json = await detectColumns(sessionId, columnMap)
      setLoading(false)
      if (!json.ok || !json.data) {
        toast.error(json.error || 'Detection failed')
        return
      }
      setResult(json.data)
      setDetectionResult(json.data)
      if (json.data.warnings?.length) {
        json.data.warnings.forEach((w) => {
          if (w) toast.message(w, { icon: <AlertCircle className="w-3.5 h-3.5" /> })
        })
      }
    },
    [sessionId, setDetectionResult],
  )

  useEffect(() => {
    if (sessionId) void runDetect()
  }, [sessionId, runDetect])

  const unknowns = result?.unknown_columns ?? []
  const needsMap = unknowns.length > 0

  const applyMapping = () => {
    const map: Record<string, string> = {}
    for (const col of unknowns) {
      const role = mapper[col] || 'ignore'
      map[col] = role
    }
    void runDetect(map)
  }

  const continueNext = () => {
    if (needsMap) {
      toast.error('Map all unknown columns or set them to Ignore')
      return
    }
    if (result?.has_task_space) setStep('frame')
    else setStep('robot')
  }

  if (!sessionId) {
    return <p className="p-3 text-xs text-text-muted">No session — upload a CSV first.</p>
  }

  return (
    <div className="p-3 space-y-3 text-xs">
      {loading && (
        <div className="flex items-center gap-2 text-text-muted">
          <Loader2 className="w-4 h-4 animate-spin" />
          Analyzing columns…
        </div>
      )}

      {!loading && result && (
        <>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-text-primary">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
              Detection
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-xxs font-mono">
              <span className="text-text-muted">Task space</span>
              <span>{result.has_task_space ? 'yes' : 'no'}</span>
              <span className="text-text-muted">Joint space</span>
              <span>{result.has_joint_space ? 'yes' : 'no'}</span>
              <span className="text-text-muted">Trajectories</span>
              <span>{result.num_trajectories}</span>
              <span className="text-text-muted">Waypoints / traj</span>
              <span>{result.num_waypoints_per_trajectory.join(', ') || '—'}</span>
            </div>
          </div>

          <div>
            <div className="text-xxs text-text-muted mb-1">Mapped columns</div>
            <div className="max-h-28 overflow-y-auto rounded border border-border px-2 py-1.5 font-mono text-xxs space-y-0.5">
              {Object.entries(result.detected_columns).map(([k, v]) => (
                <div key={k}>
                  <span className="text-text-muted">{k}</span> → <span className="text-text-primary">{v}</span>
                </div>
              ))}
            </div>
          </div>

          {needsMap && (
            <div className="space-y-2 border-t border-border pt-3">
              <div className="flex items-center gap-1.5 text-amber-500">
                <AlertCircle className="w-3.5 h-3.5" />
                Unknown columns — assign a role
              </div>
              {unknowns.map((col) => (
                <div key={col} className="flex items-center gap-2">
                  <span className="font-mono text-xxs flex-1 truncate">{col}</span>
                  <select
                    className="select-field text-xxs flex-shrink-0 max-w-[140px]"
                    value={mapper[col] || ''}
                    onChange={(e) =>
                      setMapper((m) => ({ ...m, [col]: e.target.value || 'ignore' }))
                    }
                  >
                    <option value="">—</option>
                    {ROLE_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
              <button type="button" className="btn-secondary text-xxs w-full py-1.5" onClick={applyMapping}>
                Apply mapping & re-detect
              </button>
            </div>
          )}

          <button
            type="button"
            className="btn-primary text-xxs w-full py-2 mt-2"
            onClick={continueNext}
            disabled={loading}
          >
            Continue
          </button>
        </>
      )}
    </div>
  )
}
