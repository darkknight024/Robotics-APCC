import { useCallback, useMemo } from 'react'
import { broadcastCursorUpdate } from './usePlotCursors'
import { postTimelineIndex } from '../lib/api'
import { useAnalysisStore } from '../stores/analysisStore'

/**
 * Global timeline index: scrubber, plots (via broadcastCursorUpdate), Viser.
 * Feasibility runs use selectedTrajectoryIndex + waypoint index; TOPP cursor time is derived.
 */
export function useTimeline() {
  const runResult = useAnalysisStore((s) => s.runResult)
  const sessionId = useAnalysisStore((s) => s.sessionId)
  const timelineIndex = useAnalysisStore((s) => s.timelineIndex)
  const setTimelineIndex = useAnalysisStore((s) => s.setTimelineIndex)
  const selectedTrajectoryIndex = useAnalysisStore((s) => s.selectedTrajectoryIndex)
  const feasibilityPlayback = useAnalysisStore((s) => s.feasibilityPlayback)
  const timelinePlaying = useAnalysisStore((s) => s.timelinePlaying)

  const nWaypoints = useMemo(() => {
    if (!runResult) return 0
    if (runResult.kind === 'feasibility') {
      const tr = runResult.trajectory_results[selectedTrajectoryIndex]
      const dense = tr?.dense_n_samples ?? 0
      if (feasibilityPlayback === 'dense' && dense > 0) return dense
      return tr?.num_waypoints ?? 0
    }
    return runResult.n_waypoints ?? 0
  }, [runResult, selectedTrajectoryIndex, feasibilityPlayback])

  const scrubTo = useCallback(
    (index: number) => {
      if (!runResult || nWaypoints === 0) return
      const i = Math.max(0, Math.min(index, nWaypoints - 1))
      setTimelineIndex(i)

      let timeS: number | undefined
      if (runResult.kind === 'feasibility') {
        const tr = runResult.trajectory_results[selectedTrajectoryIndex]
        const times = tr?.dense_time_ms
        if (
          feasibilityPlayback === 'dense' &&
          times &&
          times.length === nWaypoints &&
          i >= 0 &&
          i < times.length
        ) {
          timeS = times[i] / 1000
        } else {
          const dur = tr?.topp_result?.duration_s
          if (typeof dur === 'number' && dur > 0 && nWaypoints > 1) {
            timeS = (i / (nWaypoints - 1)) * dur
          }
        }
      }

      broadcastCursorUpdate(i, timeS)
      if (sessionId) {
        const pb =
          runResult.kind === 'feasibility'
            ? feasibilityPlayback === 'dense'
              ? 'dense'
              : 'sparse'
            : 'auto'
        void postTimelineIndex(sessionId, i, selectedTrajectoryIndex, pb, timelinePlaying)
      }
    },
    [
      runResult,
      nWaypoints,
      sessionId,
      setTimelineIndex,
      selectedTrajectoryIndex,
      feasibilityPlayback,
      timelinePlaying,
    ],
  )

  return {
    timelineIndex,
    setTimelineIndex,
    nWaypoints,
    scrubTo,
  }
}
