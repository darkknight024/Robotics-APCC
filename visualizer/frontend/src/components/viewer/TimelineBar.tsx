import { useEffect, useRef } from 'react'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { useTimeline } from '../../hooks/useTimeline'
import { useAnalysisStore } from '../../stores/analysisStore'

function waypointCountFromStore(): number {
  const s = useAnalysisStore.getState()
  const rr = s.runResult
  if (!rr) return 0
  if (rr.kind === 'feasibility') {
    const tr = rr.trajectory_results[s.selectedTrajectoryIndex]
    const d = tr?.dense_n_samples ?? 0
    if (s.feasibilityPlayback === 'dense' && d > 0) return d
    return tr?.num_waypoints ?? 0
  }
  return rr.n_waypoints
}

export function TimelineBar() {
  const runResult = useAnalysisStore((s) => s.runResult)
  const selectedTrajectoryIndex = useAnalysisStore((s) => s.selectedTrajectoryIndex)
  const feasibilityPlayback = useAnalysisStore((s) => s.feasibilityPlayback)
  const setFeasibilityPlayback = useAnalysisStore((s) => s.setFeasibilityPlayback)
  const { timelineIndex, nWaypoints, scrubTo } = useTimeline()
  const scrubRef = useRef(scrubTo)
  scrubRef.current = scrubTo
  const playing = useRef(false)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      if (timer.current) clearInterval(timer.current)
    }
  }, [])

  useEffect(() => {
    if (runResult && nWaypoints > 0) {
      scrubTo(0)
    }
  }, [runResult, nWaypoints, scrubTo, selectedTrajectoryIndex, feasibilityPlayback])

  const togglePlay = () => {
    playing.current = !playing.current
    if (timer.current) {
      clearInterval(timer.current)
      timer.current = null
    }
    if (playing.current) {
      timer.current = setInterval(() => {
        const nw = waypointCountFromStore()
        if (nw <= 1) return
        const s = useAnalysisStore.getState()
        const next = (s.timelineIndex + 1) % nw
        scrubRef.current(next)
      }, 280)
    }
  }

  if (nWaypoints <= 0) return null

  const trFeas =
    runResult?.kind === 'feasibility'
      ? runResult.trajectory_results[selectedTrajectoryIndex]
      : null
  const hasDense = (trFeas?.dense_n_samples ?? 0) > 0

  return (
    <div className="border-t border-border bg-surface-1 px-3 py-2 flex flex-col gap-2 shrink-0">
      {hasDense && (
        <div className="flex items-center gap-2 text-xxs">
          <span className="text-text-muted whitespace-nowrap">Playback</span>
          <select
            className="select-field text-xxs font-mono flex-1 py-1"
            value={feasibilityPlayback}
            onChange={(e) => {
              setFeasibilityPlayback(e.target.value as 'dense' | 'sparse')
              scrubTo(0)
            }}
          >
            <option value="dense">TOPP trajectory (dense)</option>
            <option value="sparse">Input waypoints (sparse)</option>
          </select>
        </div>
      )}
      <div className="flex items-center gap-3 w-full">
      <span className="text-xxs font-mono text-text-muted">
        {timelineIndex + 1} / {nWaypoints}
      </span>
      <input
        type="range"
        min={0}
        max={nWaypoints - 1}
        value={timelineIndex}
        onChange={(e) => scrubTo(Number(e.target.value))}
        className="flex-1 accent-accent-blue h-1"
      />
      <div className="flex items-center gap-1">
        <button
          type="button"
          className="btn-ghost p-1"
          onClick={() => scrubTo(timelineIndex - 1)}
          title="Prev"
        >
          <SkipBack className="w-3.5 h-3.5" />
        </button>
        <button type="button" className="btn-ghost p-1" onClick={togglePlay} title="Play">
          {playing.current ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        </button>
        <button
          type="button"
          className="btn-ghost p-1"
          onClick={() => scrubTo(timelineIndex + 1)}
          title="Next"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>
      </div>
      </div>
    </div>
  )
}
