import { create } from 'zustand'
import type {
  RobotOption,
  KnifeOption,
  AnalysisStep,
  AnalysisMode,
  DetectionResult,
  AnalysisRunResult,
  RunConfig,
  FeasibilityPlaybackMode,
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

  runResult: AnalysisRunResult | null
  setRunResult: (r: AnalysisRunResult | null) => void

  selectedTrajectoryIndex: number
  setSelectedTrajectoryIndex: (i: number) => void

  lastJobId: string | null
  setLastJobId: (id: string | null) => void

  timelineIndex: number
  setTimelineIndex: (i: number) => void

  /** After feasibility: dense = TOPP CSV samples, sparse = original IK waypoints. */
  feasibilityPlayback: FeasibilityPlaybackMode
  setFeasibilityPlayback: (m: FeasibilityPlaybackMode) => void
}

const defaultRunConfig: RunConfig = {
  solver: 'eaik',
  ee_frame_name: 'ee_link',
  trajectory_index: 0,
  speed_mm_s: 100,
  feasibility: {
    max_ik_failures_per_trajectory: 1,
    singularity: { enabled: true, threshold: 0.01, mode: 'unified', check_j5_only: true, j5_threshold_deg: 0.76 },
    manipulability: {
      enabled: true,
      warning: 0.001,
      translational_warning: 0.001,
      rotational_warning: 0.001,
      directional_warning: 0.01,
    },
    continuity: { enabled: true, pose_scale_m_per_rad: 0.1, safety_factor: 1.05, default_speed_mm_s: 100 },
    waypoint_density: {
      enabled: true,
      check_frequency_hz: 50,
      max_gap_mm: 5,
      interpolate_sparse: false,
      default_speed_mm_s: 100,
    },
    reachability: {},
    eaik_multi_solution: { enabled: true, weights: { c0: 10, singularity: 1, manipulability: 0.5 } },
    topp_ra: {},
    ranking: { safety_bin_size: 10, smoothness_weight: 1, dexterity_weight: 1 },
  },
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
  setRunResult: (r) =>
    set({
      runResult: r,
      selectedTrajectoryIndex: 0,
      feasibilityPlayback:
        r && r.kind === 'feasibility' && (r.trajectory_results[0]?.dense_n_samples ?? 0) > 0
          ? 'dense'
          : 'sparse',
    }),

  selectedTrajectoryIndex: 0,
  setSelectedTrajectoryIndex: (i) =>
    set((s) => {
      let pb = s.feasibilityPlayback
      const r = s.runResult
      if (r?.kind === 'feasibility') {
        const tr = r.trajectory_results[i]
        if (pb === 'dense' && (tr?.dense_n_samples ?? 0) === 0) pb = 'sparse'
      }
      return { selectedTrajectoryIndex: i, feasibilityPlayback: pb }
    }),

  lastJobId: null,
  setLastJobId: (id) => set({ lastJobId: id }),

  timelineIndex: 0,
  setTimelineIndex: (i) => set({ timelineIndex: i }),

  feasibilityPlayback: 'sparse',
  setFeasibilityPlayback: (m) => set({ feasibilityPlayback: m }),
}))
