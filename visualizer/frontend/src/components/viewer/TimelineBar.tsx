import { useEffect, useRef } from 'react'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { broadcastCursorUpdate } from '../../hooks/usePlotCursors'
import { useTimeline } from '../../hooks/useTimeline'
import { postTimelineIndex } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'

export function TimelineBar() {
  const runResult = useAnalysisStore((s) => s.runResult)
  const { timelineIndex, nWaypoints, scrubTo } = useTimeline()
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
  }, [runResult, nWaypoints, scrubTo])

  const togglePlay = () => {
    playing.current = !playing.current
    if (timer.current) {
      clearInterval(timer.current)
      timer.current = null
    }
    if (playing.current && nWaypoints > 1) {
      timer.current = setInterval(() => {
        const s = useAnalysisStore.getState()
        const nw = s.runResult?.n_waypoints ?? 0
        if (nw === 0) return
        const next = (s.timelineIndex + 1) % nw
        s.setTimelineIndex(next)
        broadcastCursorUpdate(next)
        if (s.sessionId) void postTimelineIndex(s.sessionId, next)
      }, 280)
    }
  }

  if (nWaypoints <= 0) return null

  return (
    <div className="border-t border-border bg-surface-1 px-3 py-2 flex items-center gap-3 shrink-0">
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
  )
}
