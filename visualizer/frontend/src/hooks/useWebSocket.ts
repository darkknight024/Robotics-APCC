import { useCallback, useRef } from 'react'
import { streamWebSocketUrl } from '../lib/api'
import type { AnalysisRunResult } from '../types/data'

type StreamPayload =
  | { type: 'log'; line?: string }
  | { type: 'progress' }
  | { type: 'error'; message?: string }
  | { type: 'done'; result?: AnalysisRunResult }

export type JobStreamHandlers = {
  onLogLine?: (line: string) => void
  onDone?: (result: AnalysisRunResult) => void
  /** Server or worker reported failure */
  onServerError?: (message: string) => void
  /** Browser could not use WebSocket — caller may poll GET /api/results */
  onTransportError?: () => void
}

/**
 * WebSocket stream for `/ws/stream/{session_id}/{job_id}` job progress.
 */
export function useJobWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)

  const close = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const connect = useCallback(
    (sessionId: string, jobId: string, handlers: JobStreamHandlers) => {
      close()
      const ws = new WebSocket(streamWebSocketUrl(sessionId, jobId))
      wsRef.current = ws

      ws.onmessage = (ev) => {
        let msg: StreamPayload
        try {
          msg = JSON.parse(ev.data as string)
        } catch {
          return
        }
        if (msg.type === 'log' && msg.line) {
          handlers.onLogLine?.(msg.line)
        }
        if (msg.type === 'error') {
          handlers.onServerError?.(msg.message || 'Run failed')
          close()
        }
        if (msg.type === 'done' && msg.result) {
          handlers.onDone?.(msg.result)
          close()
        }
      }
      ws.onerror = () => {
        handlers.onTransportError?.()
      }
      ws.onclose = () => {
        wsRef.current = null
      }
      return ws
    },
    [close],
  )

  return { connect, close }
}
