import { create } from 'zustand'
import type { TeleopMode, TeleopMetrics } from '../types/data'

interface TeleopState {
  mode: TeleopMode
  setMode: (mode: TeleopMode) => void

  isConnected: boolean
  setConnected: (connected: boolean) => void

  isRecording: boolean
  setRecording: (recording: boolean) => void

  waypointCount: number
  setWaypointCount: (count: number) => void

  selectedJoint: number
  setSelectedJoint: (joint: number) => void

  stepSize: number
  setStepSize: (size: number) => void

  metrics: TeleopMetrics | null
  setMetrics: (metrics: TeleopMetrics | null) => void

  currentQ: number[]
  setCurrentQ: (q: number[]) => void

  currentTcp: number[]
  setCurrentTcp: (tcp: number[]) => void
}

export const useTeleopStore = create<TeleopState>((set) => ({
  mode: 'task_space',
  setMode: (mode) => set({ mode }),

  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),

  isRecording: false,
  setRecording: (recording) => set({ isRecording: recording }),

  waypointCount: 0,
  setWaypointCount: (count) => set({ waypointCount: count }),

  selectedJoint: 1,
  setSelectedJoint: (joint) => set({ selectedJoint: joint }),

  stepSize: 5.0,
  setStepSize: (size) => set({ stepSize: size }),

  metrics: null,
  setMetrics: (metrics) => set({ metrics }),

  currentQ: [0, 0, 0, 0, 0, 0],
  setCurrentQ: (q) => set({ currentQ: q }),

  currentTcp: [0, 0, 0, 1, 0, 0, 0],
  setCurrentTcp: (tcp) => set({ currentTcp: tcp }),
}))
