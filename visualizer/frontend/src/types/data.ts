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

/** Per-waypoint multi-branch payload (EAIK IK only). */
export interface WaypointECFXData {
  solutions: SolutionBranch[]
  selected_index: number
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

/** Result payload from POST /api/run-ik or run-fk (result field when done) */
export interface KinematicsRunResult {
  kind: 'ik' | 'fk'
  solver: string
  ee_frame_name: string
  trajectory_index: number
  n_waypoints: number
  joints_deg: number[][]
  tcp_xyz: number[][]
  tcp_quat?: number[][]
  ik_success?: boolean[]
  waypoint_colors_hex?: string[]
  /** EAIK multi-solution / ECFX (omitted for Pin). */
  waypoint_ecfx?: (WaypointECFXData | null)[] | null
}

export interface FeasibilityToppSeries {
  t_samples_s: number[]
  q_rad: number[][]
  qdot_rad_s: number[][]
  qddot_rad_s2: number[][]
}

export interface FeasibilityTrajectoryResult {
  trajectory_index: number
  num_waypoints: number
  reachable_flags: boolean[]
  tcp_xyz_m: number[][]
  joint_angles_deg: number[][]
  manipulability: number[]
  min_singular_value: number[]
  condition_number: number[]
  near_singularity: boolean[]
  joint_space_distances: number[]
  per_joint_jumps: number[][]
  singularity_threshold_used?: number
  c0_segment_violation?: boolean[] | null
  topp_result?: { duration_s?: number | null; n_samples?: number; error?: string } | null
  topp_series?: FeasibilityToppSeries | null
  /** Absolute path on server to TOPP `final_trajectory_*.csv` (dense playback). */
  final_trajectory_csv?: string | null
  dense_n_samples?: number
  /** Milliseconds from TOPP CSV (same length as dense samples) for real-time playback. */
  dense_time_ms?: number[]
}

/** Nested groups aligned with `config/batch_feasibility_config.yaml` (API merges into FeasibilityConfig). */
export interface FeasibilityConfigPayload {
  max_ik_failures_per_trajectory?: number
  singularity?: {
    enabled?: boolean
    threshold?: number
    mode?: string
    check_j5_only?: boolean
    j5_threshold_deg?: number
  }
  manipulability?: {
    enabled?: boolean
    warning?: number
    translational_warning?: number
    rotational_warning?: number
    directional_warning?: number
  }
  continuity?: {
    enabled?: boolean
    pose_scale_m_per_rad?: number
    safety_factor?: number
    default_speed_mm_s?: number
  }
  waypoint_density?: {
    enabled?: boolean
    check_frequency_hz?: number
    max_gap_mm?: number
    interpolate_sparse?: boolean
    default_speed_mm_s?: number
  }
  reachability?: { generate_graphs?: boolean }
  eaik_multi_solution?: {
    enabled?: boolean
    max_waypoints_in_graph?: number
    weights?: { c0?: number; singularity?: number; manipulability?: number }
  }
  topp_ra?: { generate_graphs?: boolean }
  ranking?: { safety_bin_size?: number; smoothness_weight?: number; dexterity_weight?: number }
}

/** Result from POST /api/run-feasibility */
export interface FeasibilityRunResult {
  kind: 'feasibility'
  toolpath_name?: string
  num_trajectories: number
  trajectory_results: FeasibilityTrajectoryResult[]
  robot_name?: string
}

export type AnalysisRunResult = KinematicsRunResult | FeasibilityRunResult

export interface RunConfig {
  solver: 'eaik' | 'pin'
  ee_frame_name: string
  trajectory_index: number
  /** Phase 4 feasibility */
  speed_mm_s?: number
  feasibility?: FeasibilityConfigPayload
}

export type FeasibilityPlaybackMode = 'sparse' | 'dense'

export type AnalysisStep = 'upload' | 'detect' | 'frame' | 'robot' | 'action' | 'config' | 'run' | 'results'
export type AnalysisMode = 'ik_only' | 'fk_only' | 'compare' | 'feasibility'
export type TeleopMode = 'task_space' | 'joint_space'
