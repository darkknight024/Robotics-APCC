import { create } from 'zustand'
import type {
  RobotOption,
  KnifeOption,
  AnalysisStep,
  AnalysisMode,
  DetectionResult,
  KinematicsRunResult,
  RunConfig,
} from '../types/data'

interface AnalysisState {
  currentStep: AnalysisStep
  setStep: (step: AnalysisStep) => void

  robots: RobotOption[]
  setRobots: (robots: RobotOption[]) => void
  selectedRobot: string | null
  setSelectedRobot: (name: string) => void

  knives: KnifeOption[]
  setKnives: (knives: KnifeOption[]) => void
  selectedKnife: string | null
  setSelectedKnife: (name: string | null) => void

  detectionResult: DetectionResult | null
  setDetectionResult: (result: DetectionResult | null) => void

  useBaseFrame: boolean
  setUseBaseFrame: (val: boolean) => void

  analysisMode: AnalysisMode | null
  setAnalysisMode: (mode: AnalysisMode) => void

  sessionId: string | null
  setSessionId: (id: string | null) => void

  runConfig: RunConfig
  setRunConfig: (c: Partial<RunConfig>) => void

  runResult: KinematicsRunResult | null
  setRunResult: (r: KinematicsRunResult | null) => void

  lastJobId: string | null
  setLastJobId: (id: string | null) => void

  timelineIndex: number
  setTimelineIndex: (i: number) => void
}

const defaultRunConfig: RunConfig = {
  solver: 'eaik',
  ee_frame_name: 'ee_link',
  trajectory_index: 0,
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentStep: 'upload',
  setStep: (step) => set({ currentStep: step }),

  robots: [],
  setRobots: (robots) => set({ robots }),
  selectedRobot: null,
  setSelectedRobot: (name) => set({ selectedRobot: name }),

  knives: [],
  setKnives: (knives) => set({ knives }),
  selectedKnife: null,
  setSelectedKnife: (name) => set({ selectedKnife: name }),

  detectionResult: null,
  setDetectionResult: (result) => set({ detectionResult: result }),

  useBaseFrame: false,
  setUseBaseFrame: (val) => set({ useBaseFrame: val }),

  analysisMode: null,
  setAnalysisMode: (mode) => set({ analysisMode: mode }),

  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),

  runConfig: { ...defaultRunConfig },
  setRunConfig: (c) =>
    set((s) => ({ runConfig: { ...s.runConfig, ...c } })),

  runResult: null,
  setRunResult: (r) => set({ runResult: r }),

  lastJobId: null,
  setLastJobId: (id) => set({ lastJobId: id }),

  timelineIndex: 0,
  setTimelineIndex: (i) => set({ timelineIndex: i }),
}))
