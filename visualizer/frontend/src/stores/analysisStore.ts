import { create } from 'zustand'
import type { RobotOption, KnifeOption, AnalysisStep, AnalysisMode, DetectionResult } from '../types/data'

interface AnalysisState {
  // Current step
  currentStep: AnalysisStep
  setStep: (step: AnalysisStep) => void

  // Robot & knife selections
  robots: RobotOption[]
  setRobots: (robots: RobotOption[]) => void
  selectedRobot: string | null
  setSelectedRobot: (name: string) => void

  knives: KnifeOption[]
  setKnives: (knives: KnifeOption[]) => void
  selectedKnife: string | null
  setSelectedKnife: (name: string | null) => void

  // Data detection
  detectionResult: DetectionResult | null
  setDetectionResult: (result: DetectionResult | null) => void

  // Frame config
  useBaseFrame: boolean
  setUseBaseFrame: (val: boolean) => void

  // Analysis mode
  analysisMode: AnalysisMode | null
  setAnalysisMode: (mode: AnalysisMode) => void

  // Session
  sessionId: string | null
  setSessionId: (id: string) => void
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
}))
