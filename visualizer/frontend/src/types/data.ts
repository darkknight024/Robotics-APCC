// TypeScript types for all API responses

export interface APIResponse<T> {
  ok: boolean
  data?: T
  error?: string
}

export interface RobotOption {
  name: string
  description: string
  urdf_path: string
  reach_m: number
  payload_kg: number
  velocity_limits_rad_s: number[]
  acceleration_limits_rad_s2: number[]
}

export interface KnifeOption {
  name: string
  description: string
  translation_mm: [number, number, number]
  quaternion: [number, number, number, number] // w, x, y, z
}

export interface DetectionResult {
  has_task_space: boolean
  has_joint_space: boolean
  num_trajectories: number
  num_waypoints_per_trajectory: number[]
  detected_columns: Record<string, string>
  unknown_columns: string[]
  warnings?: string[]
}

export interface WaypointStatus {
  index: number
  reachable: boolean
  manipulability: number
  condition_number: number
  c0_violation: boolean
  c1_violation: boolean
  ecfx?: ECFXLabel
  ik_method: string
}

export interface ECFXLabel {
  cf1: number
  cf4: number
  cf6: number
  cfx: number
}

export interface SolutionBranch {
  branch_index: number
  ecfx: ECFXLabel
  joint_angles_deg: number[]
  is_ls: boolean
  fk_error_mm: number
}

export interface PlotData {
  group: string
  plot_id: string
  title: string
  x: number[]
  series: PlotSeries[]
  x_label: string
  y_label: string
}

export interface PlotSeries {
  name: string
  y: number[]
  color?: string
  dash?: 'solid' | 'dash' | 'dot'
  mode?: 'lines' | 'markers' | 'lines+markers'
  opacity?: number
}

export interface TeleopRobotState {
  q: number[]
  tcp: [number, number, number, number, number, number, number]
  metrics: TeleopMetrics
}

export interface TeleopMetrics {
  manipulability: number
  condition_number: number
  min_singular_value: number
  joint_limit_fractions: number[]
  ik_method: string
}

export type AnalysisStep = 'upload' | 'detect' | 'frame' | 'robot' | 'action' | 'config' | 'run' | 'results'
export type AnalysisMode = 'ik_only' | 'fk_only' | 'compare' | 'feasibility'
export type TeleopMode = 'task_space' | 'joint_space'
