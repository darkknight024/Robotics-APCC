import { useState } from 'react'
import { Loader2, Terminal } from 'lucide-react'
import { toast } from 'sonner'
import { getRunResults, runFk, runIk } from '../../lib/api'
import { useJobWebSocket } from '../../hooks/useWebSocket'
import { useAnalysisStore } from '../../stores/analysisStore'
import type { KinematicsRunResult } from '../../types/data'

export function RunPanel() {
  const sessionId = useAnalysisStore((s) => s.sessionId)
  const mode = useAnalysisStore((s) => s.analysisMode)
  const runConfig = useAnalysisStore((s) => s.runConfig)
  const setRunResult = useAnalysisStore((s) => s.setRunResult)
  const setLastJobId = useAnalysisStore((s) => s.setLastJobId)
  const setStep = useAnalysisStore((s) => s.setStep)
  const { connect } = useJobWebSocket()

  const [busy, setBusy] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])

  const body = {
    solver: runConfig.solver,
    ee_frame_name: runConfig.ee_frame_name,
    trajectory_index: runConfig.trajectory_index,
  }

  const pollFallback = async (sid: string, jid: string) => {
    for (let i = 0; i < 40; i++) {
      await new Promise((r) => setTimeout(r, 500))
      const r = await getRunResults(sid, jid)
      if (r.ok && r.data?.result) {
        setRunResult(r.data.result)
        setStep('results')
        setBusy(false)
        return
      }
    }
    toast.error('Timed out waiting for results')
    setBusy(false)
  }

  const run = async () => {
    if (!sessionId) {
      toast.error('No session')
      return
    }
    setBusy(true)
    setLogLines([])
    const api = mode === 'ik_only' ? runIk : mode === 'fk_only' ? runFk : null
    if (!api) {
      toast.error('Select a mode first')
      setBusy(false)
      return
    }
    const res = await api(sessionId, body)
    if (!res.ok || !res.data?.job_id) {
      toast.error(res.error || 'Failed to start job')
      setBusy(false)
      return
    }
    const jobId = res.data.job_id
    setLastJobId(jobId)

    connect(sessionId, jobId, {
      onLogLine: (line) => setLogLines((lines) => [...lines, line]),
      onServerError: (message) => {
        toast.error(message)
        setBusy(false)
      },
      onDone: (result: KinematicsRunResult) => {
        setRunResult(result)
        setStep('results')
        setBusy(false)
      },
      onTransportError: () => {
        toast.error('WebSocket error — polling results')
        void pollFallback(sessionId, jobId)
      },
    })
  }

  return (
    <div className="p-3 space-y-3 text-xs">
      <p className="text-text-muted text-xxs">
        Runs {mode === 'ik_only' ? 'inverse' : 'forward'} kinematics on the server. Progress streams over WebSocket.
      </p>
      <button
        type="button"
        className="btn-primary text-xxs w-full py-2 flex items-center justify-center gap-2"
        disabled={busy || !sessionId}
        onClick={() => void run()}
      >
        {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
        Run
      </button>
      {logLines.length > 0 && (
        <div className="rounded border border-border bg-surface-2 p-2 max-h-32 overflow-y-auto font-mono text-xxs text-text-muted">
          <div className="flex items-center gap-1 text-text-secondary mb-1">
            <Terminal className="w-3 h-3" />
            Log
          </div>
          {logLines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  )
}
