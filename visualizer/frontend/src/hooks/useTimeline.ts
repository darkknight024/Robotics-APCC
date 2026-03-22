import { useCallback } from 'react'
import { broadcastCursorUpdate } from './usePlotCursors'
import { postTimelineIndex } from '../lib/api'
import { useAnalysisStore } from '../stores/analysisStore'

/**
 * Global timeline index for kinematics runs: scrubber, plots (via broadcastCursorUpdate), Viser.
 */
export function useTimeline() {
  const runResult = useAnalysisStore((s) => s.runResult)
  const sessionId = useAnalysisStore((s) => s.sessionId)
  const timelineIndex = useAnalysisStore((s) => s.timelineIndex)
  const setTimelineIndex = useAnalysisStore((s) => s.setTimelineIndex)
  const nWaypoints = runResult?.n_waypoints ?? 0

  const scrubTo = useCallback(
    (index: number) => {
      if (!runResult || nWaypoints === 0) return
      const i = Math.max(0, Math.min(index, nWaypoints - 1))
      setTimelineIndex(i)
      broadcastCursorUpdate(i)
      if (sessionId) {
        void postTimelineIndex(sessionId, i)
      }
    },
    [runResult, nWaypoints, sessionId, setTimelineIndex],
  )

  return {
    timelineIndex,
    setTimelineIndex,
    nWaypoints,
    scrubTo,
  }
}
