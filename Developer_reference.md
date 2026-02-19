# Developer Reference

> Robotics-APCC — Kinematic Solver Comparison, Reachability Testing & Feasibility Analysis

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Module (`core/`)](#2-core-module)
3. [Utilities Module (`utils/`)](#3-utilities-module)
4. [Test Scripts (`tests/`)](#4-test-scripts)
5. [Root-Level Analysis Scripts](#5-root-level-analysis-scripts)
6. [Configuration Files](#6-configuration-files)
7. [Data Flow & Architecture](#7-data-flow--architecture)
8. [Dependencies](#8-dependencies)

---

## 1. Project Structure

```
Robotics-APCC_duplicate_agent/
├── core/                          # Kinematic solver backends & feasibility engine
│   ├── __init__.py                # Factory function, exports, backward-compat aliases
│   ├── base_solvers.py            # Abstract base classes for all solvers
│   ├── eaik_fk_solver.py          # EAIK analytical FK solver
│   ├── eaik_ik_solver.py          # EAIK analytical IK solver
│   ├── pin_fk_solver.py           # Pinocchio numerical FK solver
│   ├── pin_ik_solver.py           # Pinocchio numerical IK solver
│   └── feasibility_checks.py      # Feasibility analyzer (manipulability, singularity, etc.)
│
├── utils/                         # Shared utilities (loaders, plotters, math)
│   ├── __init__.py                # Central re-exports for all utils
│   ├── config_loader.py           # YAML config loading & typed config objects
│   ├── urdf_loader.py             # URDF parsing for both backends
│   ├── csv_loader_robostudio.py   # RobotStudio CSV loader
│   ├── csv_loader_toolpath.py     # Toolpath CSV loader
│   ├── transform_handler.py       # Coordinate frame transformations
│   ├── math.py                    # Joint-space math utilities
│   ├── generate_plot_fk.py        # FK comparison plot generators
│   ├── generate_plot_ik.py        # IK comparison & outcome plot generators
│   ├── feasibility_plot.py        # Feasibility & debug visualization
│   ├── generate_combinatorial_plots.py  # Ranking bar charts
│   ├── generate_knife_poses.py    # Knife pose perturbation generator
│   └── time_weighted_aggregation.py     # Time-weighted metric aggregation
│
├── tests/                         # Test & validation scripts
│   ├── test_solvers.py            # FK/IK comparison against RobotStudio data
│   ├── test_reachability.py       # Reachability analysis for toolpaths
│   ├── test_toolpaths.py          # Toolpath IK comparison with recorded joints
│   ├── tolerance_check.py         # Threshold-based pass/fail on raw_comparison.csv
│   ├── run_experiments.py         # Automated experiment runner with benchmarking
│   └── configs/                   # Per-script YAML configurations
│       ├── test_solvers_config.yaml
│       ├── reachability_config.yaml
│       ├── test_toolpaths_config.yaml
│       ├── tolerance_config.yaml
│       └── experiments_config.yaml
│
├── config/                        # Global configuration files
│   ├── ik_config.yaml             # IK solver parameters (shared by both backends)
│   ├── robots_config.yaml         # Robot database (URDF paths, limits, reach)
│   ├── knife_config.yaml          # Knife pose definitions (T_B_K)
│   ├── batch_feasibility_config.yaml
│   ├── combinatorial_search_config.yaml
│   ├── scoring_weights.yaml
│   ├── generated_knife_poses.yaml
│   └── sparse_generated_knife_poses.yaml
│
├── feasibility_analysis.py        # Single-toolpath feasibility analysis
├── feasibility_analysis_batch.py  # Batch feasibility across combinations
├── combinatorial_search.py        # Full combinatorial ranking search
│
├── Assets/                        # Robot URDF files & meshes
├── Robot_APCC/                    # Experiment inputs, results, benchmarks
│   ├── Experiments/               # Input trajectories per experiment
│   ├── Results/                   # Test outputs per experiment
│   └── Benchmarks/                # Ground-truth CSVs for regression testing
│
└── requirements.txt
```

---

## 2. Core Module

### 2.1 `core/base_solvers.py` — Abstract Base Classes

Defines the interface contract that all solver backends must implement.

#### Dataclasses

| Class | Fields | Purpose |
|-------|--------|---------|
| `FKResult` | `position_m: ndarray`, `quaternion: ndarray`, `rotation_matrix: ndarray` | Result of a single FK computation |
| `BaseIKConfig` | `ee_frame_name: str = "ee_link"` | Base config inherited by all IK config classes |

#### Classes

**`BaseFKSolver(ABC)`** — Abstract FK solver interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `ee_frame_name` | `@property -> str` | Name of the end-effector frame |
| `solver_name` | `@property -> str` | Human-readable solver name for reports |
| `solve` | `(q: ndarray) -> FKResult` | Compute FK for a single joint config |
| `solve_batch` | `(joint_positions: ndarray) -> (ndarray, ndarray)` | FK for multiple configs; returns (positions, quaternions) |
| `get_jacobian` | `(q: ndarray, local_frame: bool = True) -> ndarray` | Compute 6xn Jacobian [angular; linear] |

**`BaseIKSolver(ABC)`** — Abstract IK solver interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `solver_name` | `@property -> str` | Human-readable solver name for reports |
| `solve` | `(target_position: ndarray, target_quaternion: ndarray, q_init: ndarray = None) -> (bool, ndarray, dict)` | Solve IK for a single target pose |
| `solve_with_retries` | `(target_position, target_quaternion, q_init, num_random_retries) -> (bool, ndarray, dict)` | Solve IK with fallback/retry strategies |
| `_quat_to_rotation` | `@staticmethod (quat: ndarray) -> ndarray` | Convert [qw,qx,qy,qz] to 3x3 rotation matrix |

---

### 2.2 `core/eaik_fk_solver.py` — EAIK FK Solver

**`EAIKFKSolver(BaseFKSolver)`** — FK using EAIK analytical solver with numerical Jacobian

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(robot_model: RobotModel)` | Initialize with parsed RobotModel |
| `solve` | `(q: ndarray) -> FKResult` | FK via `eaik_robot.fwdkin()` + ee_transform correction |
| `get_jacobian` | `(q: ndarray, local_frame: bool = True) -> ndarray` | 6xn Jacobian via central finite differences |

**Module-level helpers:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_rotation_matrix_to_quaternion` | `(R: ndarray) -> ndarray` | Convert 3x3 rotation to [qw,qx,qy,qz] |
| `_log_rotation` | `(R: ndarray) -> ndarray` | Logarithmic map of rotation matrix (axis-angle) |

---

### 2.3 `core/eaik_ik_solver.py` — EAIK IK Solver

**`EAIKConfig(BaseIKConfig)`** — Configuration dataclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ee_frame_name` | `str` | `"ee_link"` | End-effector frame name |
| `solution_selection` | `str` | `"closest"` | Strategy: `"closest"` or `"min_norm"` |

**`EAIKIKSolver(BaseIKSolver)`** — Analytical IK returning all solutions, filtering by joint limits

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(robot_model: RobotModel, config: EAIKConfig = None)` | Initialize with RobotModel and optional config |
| `solve` | `(target_position, target_quaternion, q_init) -> (bool, ndarray, dict)` | Analytical IK with joint-limit filtering and solution selection |
| `solve_with_retries` | `(target_position, target_quaternion, q_init, num_random_retries) -> (bool, ndarray, dict)` | Delegates to `solve()` (analytical is deterministic) |
| `_within_joint_limits` | `(q, tolerance=1e-6) -> bool` | Check if config is within limits |
| `_get_violated_joints` | `(q, tolerance=1e-6) -> list` | Return indices of joints violating limits |
| `_select_closest` | `(solutions, q_ref) -> int` | Pick solution closest to q_ref (angle-wrapped) |
| `_select_min_norm` | `(solutions) -> int` | Pick solution with smallest joint norm |
| `_select_least_violation` | `(solutions, q_init) -> ndarray` | Fallback: pick solution with minimum limit violation |

**`info` dict keys returned by `solve()`:**

| Key | Type | Description |
|-----|------|-------------|
| `n_solutions` | `int` | Total analytical solutions found |
| `n_valid` | `int` | Solutions passing joint limits |
| `is_ls` | `bool` | Whether least-squares (approximate) solution was used |
| `selected_index` | `int/None` | Index of selected solution |
| `converged` | `bool` | Whether a valid solution was found |
| `reason` | `str` | `"converged"`, `"no_solutions"`, or `"no_valid_solutions_within_limits"` |
| `solve_method` | `str` | `"converged"`, `"no_solutions"`, or `"joint_limits"` |
| `violated_joints` | `list/None` | Joint indices that violated limits |

---

### 2.4 `core/pin_fk_solver.py` — Pinocchio FK Solver

**`PinocchioFKSolver(BaseFKSolver)`** — FK using Pinocchio library

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model: pin.Model, data: pin.Data, ee_frame_name: str = "ee_link")` | Initialize with Pinocchio model/data |
| `solve` | `(q: ndarray) -> FKResult` | FK via `pin.forwardKinematics` |
| `solve_batch` | `(joint_positions: ndarray) -> (ndarray, ndarray)` | Batch FK (loop-based) |
| `get_jacobian` | `(q: ndarray, local_frame: bool = True) -> ndarray` | Analytical Jacobian via `pin.computeFrameJacobian` |

---

### 2.5 `core/pin_ik_solver.py` — Pinocchio IK Solver

**`PinocchioIKConfig(BaseIKConfig)`** — Configuration dataclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ee_frame_name` | `str` | `"ee_link"` | End-effector frame name |
| `max_iterations` | `int` | `50` | Maximum solver iterations |
| `tolerance` | `float` | `1e-4` | Convergence threshold |
| `rot_weight` | `float` | `0.2` | Rotation error weight |
| `trans_weight` | `float` | `1.0` | Translation error weight |
| `lambda0` | `float` | `1e-3` | Initial LM damping |
| `lambda_max` | `float` | `1e1` | Maximum damping |
| `max_step` | `float` | `0.2` | Max step size (rad) |
| `backtrack` | `bool` | `True` | Enable backtracking line search |
| `use_initial_guess` | `bool` | `True` | Try initial guess strategy |
| `use_neutral` | `bool` | `True` | Try neutral config strategy |
| `use_random` | `bool` | `True` | Try random configs strategy |
| `num_random_retries` | `int` | `3` | Number of random retries |

**`PinocchioIKSolver(BaseIKSolver)`** — Damped least-squares IK with adaptive damping

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model: pin.Model, data: pin.Data, config: PinocchioIKConfig = None)` | Initialize with Pinocchio model/data |
| `solve` | `(target_position, target_quaternion, q_init) -> (bool, ndarray, dict)` | Single-attempt damped LS solve |
| `solve_with_retries` | `(target_position, target_quaternion, q_init, num_random_retries) -> (bool, ndarray, dict)` | Multi-strategy solve respecting `use_*` config flags |
| `_solve_damped` | `(target_pose: pin.SE3, q_init) -> (bool, ndarray, dict)` | Core damped least-squares solver loop |

**`info` dict keys (Pinocchio `solve_with_retries`):**

| Key | Type | Description |
|-----|------|-------------|
| `solve_method` | `str` | `"initial_guess"`, `"neutral"`, `"random"`, or `"failed"` |
| `iterations` | `int` | Number of iterations used |
| `residual_norm` | `float` | Final residual |
| `converged` | `bool` | Whether solver converged |
| `reason` | `str` | `"converged"`, `"max_iter_exceeded"`, or `"backtracking_failed; increased damping"` |
| `sigma_min` | `float` | Minimum singular value of Jacobian |
| `clip_count` | `int` | Number of joint-limit clips |

---

### 2.6 `core/feasibility_checks.py` — Feasibility Analysis Engine

**`FeasibilityResult`** — Per-waypoint feasibility metrics dataclass

| Field | Type | Description |
|-------|------|-------------|
| `is_reachable` | `bool` | Whether IK succeeded |
| `manipulability` | `float` | Yoshikawa manipulability index |
| `min_singular_value` | `float` | σ_min (singularity proximity) |
| `max_singular_value` | `float` | σ_max |
| `condition_number` | `float` | κ = σ_max / σ_min |
| `near_singularity` | `bool` | σ_min < threshold |
| `joint_positions_rad` | `ndarray/None` | Solved joint angles |
| `ik_debug_info` | `dict/None` | Debug data for failed waypoints |
| `target_position` | `ndarray/None` | Target position |
| `target_quaternion` | `ndarray/None` | Target quaternion |
| `joint_velocity_ratio` | `float/None` | Max |dq/dt| / limit ratio |
| `distance_to_joint_limits` | `float/None` | Min distance across all joints |
| `joint_space_distance` | `float/None` | Distance from previous waypoint |

**Standalone functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_manipulability` | `(jacobian: ndarray, characteristic_length_m: float = 1.0) -> float` | Normalized Yoshikawa manipulability |
| `compute_singularity_proximity` | `(jacobian: ndarray) -> float` | Minimum singular value |
| `compute_condition_number` | `(jacobian: ndarray) -> float` | Condition number with NaN safety |
| `compute_max_singular_value` | `(jacobian: ndarray) -> float` | Maximum singular value |
| `check_reachability` | `(ik_solver, target_position, target_quaternion, q_init) -> (bool, ndarray/None, dict)` | IK reachability check wrapper |

**`FeasibilityAnalyzer`** — Full trajectory feasibility analyzer

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(robot_model_or_limits, ik_solver, fk_solver, characteristic_length_m, singularity_threshold, velocity_limits_rad_s, joint_jump_limit_rad, max_ik_failures_per_trajectory)` | Initialize with solver-agnostic robot data |
| `analyze_waypoint` | `(target_position, target_quaternion, q_init) -> FeasibilityResult` | Full analysis of one waypoint |
| `analyze_trajectory` | `(positions, quaternions, timestamps, speed_mm_s, speeds_mm_s) -> dict` | Trajectory-level analysis with C0/C1 checks, 4-level ranking scores |

---

### 2.7 `core/__init__.py` — Factory & Exports

**`create_solvers(urdf_path, solver="eaik", ik_config=None, ee_frame_name="ee_link")`**

Factory function that returns `(fk_solver, ik_solver, robot_data)`:
- `solver="eaik"` → `(EAIKFKSolver, EAIKIKSolver, RobotModel)`
- `solver="pin"` → `(PinocchioFKSolver, PinocchioIKSolver, (pin.Model, pin.Data))`

**Backward-compatibility aliases:** `IKSolver = PinocchioIKSolver`, `IKConfig = PinocchioIKConfig`, `FKSolver = PinocchioFKSolver`

---

## 3. Utilities Module

### 3.1 `utils/config_loader.py` — Configuration Loading

**Dataclasses:**

| Class | Fields | Purpose |
|-------|--------|---------|
| `KnifePose` | `name`, `description`, `translation_m: ndarray`, `quaternion: ndarray` | Knife pose in robot base frame |
| `RobotConfig` | `name`, `urdf_path`, `reach_m`, `velocity_limits_rad_s`, `acceleration_limits_rad_s2`, `joint_jump_limit_rad` | Robot configuration from database |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_yaml` | `(config_path: str) -> dict` | Load any YAML file |
| `load_ik_config` | `(config_path: str) -> dict` | Load raw IK parameters dict |
| `load_ik_config_as_object` | `(config_path: str = None, solver: str = "eaik") -> EAIKConfig / PinocchioIKConfig` | Load typed IK config for the given solver backend |
| `load_knife_config` | `(config_path: str) -> dict[str, KnifePose]` | Load knife poses (mm→m conversion) |
| `load_robots_config` | `(config_path: str = None) -> dict[str, RobotConfig]` | Load robot database |
| `get_robot_by_name` | `(robot_name: str, robots_config_path: str = None) -> RobotConfig` | Lookup a single robot by name |
| `load_toolpath_config` | `(config_path: str) -> dict` | Load toolpath processing config with robot name resolution |
| `load_feasibility_config` | `(config_path: str) -> dict` | Load feasibility check flags & thresholds |
| `load_robostudio_test_config` | `(config_path: str) -> dict` | Load RobotStudio test trajectory config |
| `get_default_ik_config` | `() -> dict` | Return copy of `_DEFAULT_IK_CONFIG` |

---

### 3.2 `utils/urdf_loader.py` — URDF Parsing

**`RobotModel`** — Dataclass for EAIK-loaded robot

| Field | Type | Description |
|-------|------|-------------|
| `eaik_robot` | `EAIK.Robot` | EAIK robot for FK/IK |
| `n_joints` | `int` | Number of actuated joints |
| `joint_names` | `list[str]` | Actuated joint names |
| `lower_position_limit` | `ndarray` | Lower joint limits (rad) |
| `upper_position_limit` | `ndarray` | Upper joint limits (rad) |
| `ee_frame_name` | `str` | End-effector frame used |
| `ee_transform_4x4` | `ndarray` | 4x4 transform from last actuated link to EE |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_robot_model` | `(urdf_path, solver="eaik", ee_frame_name="ee_link")` | Dispatcher: loads via EAIK or Pinocchio |
| `load_robot_model_eaik` | `(urdf_path, ee_frame_name="ee_link") -> RobotModel` | Parse URDF with urchin + create EAIK robot |
| `load_robot_model_pin` | `(urdf_path) -> (pin.Model, pin.Data)` | Parse URDF with Pinocchio |
| `resolve_urdf_path` | `(urdf_path: str) -> Path` | Resolve path with fuzzy matching |
| `_find_similar_urdf` | `(requested_path, original_path) -> Path` | Fuzzy-match URDF files |
| `_get_search_directories` | `(requested_path) -> list[Path]` | Build list of URDF search directories |
| `_score_urdf_match` | `(filename_lower, key_parts, prefers_ee) -> int` | Score filename match quality |
| `_urdf_to_sp_conv` | `(axis_trafo, axis, parent_p) -> (ndarray, ndarray)` | Convert urchin joint to EAIK convention |
| `_clean_axis` | `(axis, tol=1e-4) -> ndarray` | Snap near-zero axis components to zero |
| `_find_ee_link` | `(robot, joints, ee_frame_name) -> link/None` | Find EE link traversing fixed joints |

---

### 3.3 `utils/csv_loader_robostudio.py` — RobotStudio Data Loading

**`RobotStudioData`** — Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `tcp_positions_m` | `ndarray (n,3)` | TCP positions in meters |
| `tcp_quaternions` | `ndarray (n,4)` | Quaternions [qw,qx,qy,qz] |
| `joint_positions_rad` | `ndarray (n,6)` | Joint angles in radians |
| `num_waypoints` | `int` | Number of waypoints |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_robostudio_full` | `(csv_path: str) -> RobotStudioData` | Load full RobotStudio CSV (TCP + joints) |
| `load_robostudio_joints_only` | `(csv_path: str) -> dict` | Load only joint positions |
| `validate_robostudio_csv` | `(csv_path, require_joints, require_tcp) -> (bool, str/None)` | Validate CSV columns |
| `find_robostudio_csvs` | `(folder_path: str) -> list` | Find all CSVs in a folder |

---

### 3.4 `utils/csv_loader_toolpath.py` — Toolpath Data Loading

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_toolpath_trajectories` | `(csv_path, max_trajectories=None) -> (list[ndarray], list[ndarray])` | Load trajectories + per-waypoint speeds from CSV |
| `get_trajectory_count` | `(csv_path: str) -> int` | Count trajectories without full load |
| `extract_toolpath_speed` | `(csv_path: str) -> float` | Extract commanded speed from column 8 |
| `validate_toolpath_csv` | `(csv_path: str) -> tuple` | Validate toolpath format |
| `_parse_waypoint` | `(row) -> (list/None, float/None)` | Parse single waypoint row |
| `_finalize_trajectory` | `(trajectories, speeds, current_trajectory, current_speeds, max_trajectories) -> None` | Append completed trajectory with dedup |
| `_remove_duplicate_waypoints` | `(trajectory, speeds, tolerance=1e-6) -> (ndarray, ndarray)` | Remove consecutive duplicates |

---

### 3.5 `utils/transform_handler.py` — Coordinate Frame Transforms

| Function | Signature | Description |
|----------|-----------|-------------|
| `quat_to_rotation_matrix` | `(quaternion: ndarray) -> ndarray` | [qw,qx,qy,qz] → 3x3 rotation |
| `rotation_matrix_to_quaternion` | `(R: ndarray) -> ndarray` | 3x3 rotation → [qw,qx,qy,qz] |
| `pose_to_matrix` | `(translation, quaternion) -> ndarray` | Build 4x4 homogeneous transform |
| `matrix_to_pose` | `(T: ndarray) -> (ndarray, ndarray)` | Extract (translation, quaternion) from 4x4 |
| `invert_transform` | `(T: ndarray) -> ndarray` | Invert 4x4 homogeneous transform |
| `transform_t_p_k_to_t_k_p` | `(T_P_K: ndarray) -> ndarray` | Invert per-waypoint: knife→part to part→knife |
| `transform_t_k_p_to_t_b_p` | `(T_K_P, knife_translation_m, knife_quaternion) -> ndarray` | Transform to base frame using knife pose |
| `transform_trajectory_to_base_frame` | `(trajectory_t_p_k, knife_translation_m, knife_quaternion) -> ndarray` | Full trajectory transform: T_P_K → T_B_P |
| `transform_trajectories_to_base_frame` | `(trajectories_t_p_k, knife_translation_m, knife_quaternion) -> list[ndarray]` | Batch trajectory transform |

---

### 3.6 `utils/math.py` — Joint-Space Mathematics

| Function | Signature | Description |
|----------|-----------|-------------|
| `shortest_angular_distance` | `(q1: float, q2: float) -> float` | Shortest angular distance with wrapping |
| `compute_joint_space_distance` | `(q1, q2) -> float` | Euclidean distance with angle wrapping |
| `compute_distance_to_joint_limits` | `(q, lower_limits, upper_limits) -> float` | Min distance to any joint limit |
| `compute_joint_velocity_ratio` | `(q_prev, q_current, dt, velocity_limits_rad_s) -> float` | Max joint velocity ratio (C1 metric) |
| `compute_joint_velocity_metrics` | `(joint_angles_rad, timestamps, velocity_limits_rad_s) -> dict` | Full velocity statistics |
| `compute_joint_limit_violations` | `(joint_angles_rad, lower_limits, upper_limits) -> dict` | Joint limit violation stats |
| `compute_normalized_joint_energy` | `(joint_angles_rad, timestamps, velocity_limits_rad_s) -> float` | Normalized energy (smoothness cost) |
| `compute_safety_tier` | `(max_condition_number, safety_bin_size=10.0) -> int` | Safety tier by binning condition number |

---

### 3.7 `utils/generate_plot_fk.py` — FK Comparison Plots

| Function | Signature | Description |
|----------|-----------|-------------|
| `plot_position_comparison` | `(ref_mm, computed_mm, output_path, title, ref_label, computed_label, adaptive_scale)` | 3-panel X/Y/Z position overlay |
| `plot_position_deltas` | `(ref_mm, computed_mm, output_path, title, adaptive_scale)` | 3-panel X/Y/Z delta plot |
| `plot_quaternion_comparison` | `(ref_quat, computed_quat, output_path, title, ref_label, computed_label, adaptive_scale)` | 4-panel qw/qx/qy/qz overlay |
| `plot_euclidean_error` | `(ref_mm, computed_mm, output_path, title, adaptive_scale)` | Per-waypoint Euclidean distance error |

---

### 3.8 `utils/generate_plot_ik.py` — IK Comparison & Outcome Plots

| Function | Signature | Description |
|----------|-----------|-------------|
| `plot_joint_comparison` | `(ref_deg, computed_deg, output_path, title, ref_label, computed_label, adaptive_scale, mask, joint_limits_deg)` | 2x3 subplot J1-J6 overlay |
| `plot_joint_deltas` | `(ref_deg, computed_deg, output_path, title, adaptive_scale, mask)` | 2x3 subplot |ref-computed| per joint |
| `plot_ik_success_failure` | `(ik_success, output_path, title, traj_index)` | Green/red scatter: success vs failure |
| `plot_ik_solve_methods` | `(solve_methods, ik_success, output_path, title, traj_index)` | Pinocchio: scatter by init method (initial_guess/neutral/random/failed) |
| `plot_eaik_solve_outcome` | `(solve_methods, ik_success, output_path, title, traj_index)` | EAIK: scatter by outcome (converged/joint_limits/no_solutions) |
| `plot_joint_limits_violated_per_waypoint` | `(violated_joints_per_wp, ik_success, robot_model, output_path, title, traj_index)` | EAIK: which joints violated limits |

---

### 3.9 `utils/feasibility_plot.py` — Feasibility Visualization

**Per-waypoint plots:**

| Function | Description |
|----------|-------------|
| `plot_singularity_per_waypoint` | Min singular value vs waypoint with threshold line |
| `plot_reachability_per_waypoint` | Binary reachable/unreachable per waypoint |
| `plot_manipulability_per_waypoint` | Manipulability index per waypoint |
| `plot_continuity_analysis` | 4-panel: Cartesian speed, joint velocities, velocity ratios, cumulative time |

**Per-trajectory aggregation plots:**

| Function | Description |
|----------|-------------|
| `plot_reachability_rate_per_trajectory` | Bar chart: reachability % per trajectory |
| `plot_manipulability_per_trajectory` | Bar chart: avg & min manipulability |
| `plot_singularity_per_trajectory` | Bar chart: avg & min singular values |
| `plot_continuity_summary` | Bar chart: max velocity ratio per trajectory |
| `plot_reachability_summary` | Bar chart: reachable waypoint counts |

**Debug plots:**

| Function | Description |
|----------|-------------|
| `plot_ik_failure_analysis` | Spatial failure analysis |
| `plot_joint_limit_analysis` | Joint limit proximity for failures |
| `plot_per_waypoint_ik_debug` | Single-waypoint IK failure breakdown |
| `plot_joint_configurations_vs_limits` | Joint angles vs limit bands |

**4-Level feasibility plots:**

| Function | Description |
|----------|-------------|
| `plot_feasibility_levels` | Summary: L1 feasibility, L2 safety, L3 smoothness, L4 dexterity |
| `plot_feasibility_levels_detailed` | Per-waypoint 4-level data |
| `plot_combination_feasibility_levels` | Multi-trajectory combination feasibility |

---

### 3.10 `utils/generate_combinatorial_plots.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `generate_ranking_plot` | `(results, output_path, title, metric_key, metric_label, ...)` | Ranked bar chart for combinatorial search results |

---

### 3.11 `utils/time_weighted_aggregation.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_time_weighted_average` | `(values, durations) -> float` | Time-weighted average to avoid sampling bias |
| `compute_time_weighted_manipulability` | `(manipulability_values, durations) -> float` | Time-weighted mean manipulability |
| `compute_time_weighted_smoothness` | `(velocity_ratios, durations) -> float` | Time-weighted mean squared velocity ratio |
| `extract_segment_durations_from_result` | `(trajectory_result) -> ndarray` | Extract dt array from trajectory result dict |
| `aggregate_metrics_time_weighted` | `(trajectory_result) -> dict` | Full time-weighted aggregation of all metrics |

---

### 3.12 `utils/generate_knife_poses.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_quaternion` | `(q) -> ndarray` | Normalize quaternion to unit length |
| `quaternion_multiply` | `(q1, q2) -> ndarray` | Hamilton product of two quaternions |
| `generate_perturbed_quaternion` | `(ref_quat, angle_deg, seed) -> ndarray` | Generate random rotation perturbation |
| `linspace` | `(start, stop, num) -> list` | Simple linspace without numpy |
| `main` | `() -> None` | Generate perturbed knife pose YAML files |

---

## 4. Test Scripts

### 4.1 `tests/test_solvers.py` — FK/IK Solver Comparison

**Purpose:** Compare FK and IK results from a kinematic solver against RobotStudio ground truth data. Processes CSV files containing recorded TCP positions, quaternions, and joint angles.

**Config file:** `tests/configs/test_solvers_config.yaml`

**CLI arguments:** `--config`, `--input`, `--output`, `--urdf`, `--ik-config`, `--solver {pin,eaik}`, `--ee-frame`, `--adaptive-scale`

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `save_individual_analysis` | `(output_path, csv_name, n_waypoints, fk_stats, ik_stats, fk_errors_mm, joint_errors_deg, pos_deltas_mm)` | Save per-CSV analysis text report |
| `save_global_analysis` | `(output_path, all_results, urdf_path, input_path, solver_name, ee_frame_name)` | Save global summary with dynamic solver/EE names |
| `process_single_csv` | `(csv_path, fk_solver, ik_solver, robot_data, output_dir, adaptive_scale, generate_fk_plots, generate_ik_plots) -> dict` | Process one CSV: FK analysis, IK analysis, plots, raw CSV export |
| `main` | `()` | Entry point: parse args, create solvers, process all CSVs |

**Outputs per CSV:** `fk_position_comparison.png`, `fk_position_deltas.png`, `fk_quaternion_comparison.png`, `fk_euclidean_error.png`, `ik_joint_comparison.png`, `ik_joint_deltas.png`, `ik_success_failure.png`, `ik_solve_methods.png` (Pinocchio) / `ik_solve_outcome.png` (EAIK), `raw_comparison.csv`, `analysis.txt`

**Global outputs:** `global_analysis.txt`

---

### 4.2 `tests/test_reachability.py` — Reachability Analysis

**Purpose:** Check if all waypoints in toolpath trajectories are reachable by the robot for each robot/knife/toolpath combination.

**Config file:** `tests/configs/reachability_config.yaml`

**CLI arguments:** `--config`, `--robot`, `--urdf`, `--knife-pose`, `--toolpaths-folder`, `--output`, `--solver {pin,eaik}`, `--ee-frame`

**Dataclasses:**

| Class | Description |
|-------|-------------|
| `TrajectoryResult` | Per-trajectory result: flags, methods, joint angles, target poses |
| `ToolpathResult` | Per-toolpath result aggregating multiple TrajectoryResults |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_trajectory_reachability` | `(trajectory_t_b_p, ik_solver, trajectory_index) -> TrajectoryResult` | Check reachability per waypoint, storing joint angles |
| `plot_ik_solve_methods_with_exclusions` | `(solve_methods, ik_success, ik_config, output_path, title, traj_index)` | Pinocchio: method plot with red exclusion bands |
| `save_reachability_csv` | `(traj_result, output_path)` | Export raw per-waypoint data to CSV |
| `generate_report` | `(all_results, output_path)` | Generate reachability_analysis.txt |
| `process_combination` | `(robot_name, urdf_path, knife_name, ..., solver_type, ee_frame_override) -> ToolpathResult` | Process one robot/knife/toolpath combination |
| `main` | `()` | Entry point: parse args, resolve robots/knives, process all combos |

**Outputs per trajectory:** `reachability_per_waypoint_T{n}.png`, `ik_success_failure_T{n}.png`, `ik_solve_methods_T{n}.png` / `ik_solve_outcome_T{n}.png`, `raw_reachability_T{n}.csv`

---

### 4.3 `tests/test_toolpaths.py` — Toolpath Joint Comparison

**Purpose:** Compare IK-computed joint angles against RobotStudio-recorded joint angles for toolpath trajectories. Supports parallel processing.

**Config file:** `tests/configs/test_toolpaths_config.yaml`

**CLI arguments:** `--config`, `--solver {pin,eaik}`, `--output`

**Dataclasses:**

| Class | Description |
|-------|-------------|
| `TrajectoryComparisonTask` | All data needed for one trajectory comparison task |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_robostudio_joints_csv` | `(csv_path) -> dict[int, ndarray]` | Load RobotStudio recorded joints |
| `validate_toolpath_robostudio_pair` | `(toolpath_name, trajectories, robostudio_csv) -> (bool, str, dict)` | Validate compatibility |
| `process_single_trajectory` | `(task: TrajectoryComparisonTask) -> dict` | Process one trajectory (parallel-safe) |
| `save_joints_csv` | `(joint_positions_rad, success_flags, solve_methods, output_path)` | Save CSV with IK status |
| `process_batch` | `(config_path) -> dict` | Run batch comparison with parallel workers |
| `main` | `()` | Entry point |

**Outputs per trajectory:** `joint_comparison.png`, `joint_deltas.png`, `raw_comparison.csv`

---

### 4.4 `tests/tolerance_check.py` — Threshold Validation

**Purpose:** Post-hoc check that FK/IK errors in `raw_comparison.csv` files stay within tolerance thresholds.

**Config file:** `tests/configs/tolerance_config.yaml`

**Dataclasses:**

| Class | Description |
|-------|-------------|
| `Violation` | Single threshold violation with type, detail, value |
| `ToolpathResult` | Aggregated violations for one toolpath |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_config` | `(config_path) -> dict` | Load tolerance config |
| `quaternion_angular_error_deg` | `(qw1, qx1, ..., qz2) -> float` | Angular difference between two quaternions (degrees) |
| `analyze_toolpath` | `(csv_path, toolpath_name, fk_threshold_mm, fk_rot_threshold_deg, ik_threshold_deg) -> ToolpathResult` | Analyze one raw_comparison.csv |
| `discover_toolpaths` | `(input_folder) -> list[Path]` | Find all subfolders containing raw_comparison.csv |
| `generate_report` | `(results, fk_threshold_mm, ..., input_folder) -> str` | Generate text report |
| `run_tolerance_check` | `(input_folder, report_output, fk_threshold_mm, ...) -> int` | Programmatic entry point |
| `main` | `()` | CLI entry point |

---

### 4.5 `tests/run_experiments.py` — Automated Experiment Runner

**Purpose:** Execute all experiments defined in `experiments_config.yaml`, calling the appropriate test script with CLI overrides. Optionally compares outputs against ground-truth benchmarks.

**Config file:** `tests/configs/experiments_config.yaml`

**CLI arguments:** `--config`, `--experiment`, `--solver {pin,eaik}`, `--dry-run`

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `resolve_urdf` | `(robot_name) -> str` | Look up URDF path from robots_config.yaml |
| `build_command` | `(test_script, run_cfg, experiment_cfg) -> (list, str)` | Build subprocess command + output path |
| `find_output_csvs` | `(output_dir) -> list[Path]` | Recursively find raw CSV files |
| `compare_csvs` | `(output_csv, gt_csv) -> dict` | Cell-by-cell CSV comparison (numeric + categorical) |
| `write_comparison_report` | `(report_path, run_label, comparisons)` | Write detailed diff report |
| `run_ground_truth_comparison` | `(output_dir, gt_dir, run_label) -> (bool/None, str)` | Compare all output CSVs against ground truth |
| `run_experiment` | `(experiment, solver_filter, dry_run, enable_benchmarking) -> list` | Execute all runs for one experiment |
| `print_summary` | `(all_results)` | Print final table: Exec status + Benchmark status |
| `main` | `()` | Entry point |

**Summary columns:** `Run`, `Exec` (OK/FAILED), `Benchmark` (PASS/FAIL/-), `Time`, `Details`

---

## 5. Root-Level Analysis Scripts

### 5.1 `feasibility_analysis.py` — Single-Toolpath Feasibility

**Purpose:** Analyze kinematic feasibility of a single toolpath: reachability, manipulability, singularity, C0/C1 continuity.

**Config file:** `config/batch_feasibility_config.yaml`

**Dataclasses:**

| Class | Description |
|-------|-------------|
| `ContinuityResult` | C1 continuity analysis result: passed, max velocities, violations |

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_segment_times` | `(trajectory_m, joint_angles_rad, speed_mm_s, speeds_mm_s, velocity_limits_rad_s, pose_scale_m_per_rad) -> (ndarray, ndarray)` | Speed-driven segment time estimation |
| `analyze_continuity` | `(trajectory_m, joint_angles_rad, speed_mm_s, ...) -> ContinuityResult` | C1 velocity limit analysis |
| `analyze_trajectory_feasibility` | `(trajectory_t_b_p, analyzer, trajectory_name, ...) -> dict` | Full single-trajectory feasibility |
| `process_toolpath` | `(toolpath_path, urdf_path, knife_translation_m, ..., solver_type) -> dict` | Process one toolpath with all analysis |
| `generate_analysis_report` | `(results, output_path)` | Write human-readable report |

---

### 5.2 `feasibility_analysis_batch.py` — Batch Feasibility

**Purpose:** Run feasibility analysis across all robot/knife/toolpath combinations defined in config, with parallel execution support.

**Config file:** `config/batch_feasibility_config.yaml`

**Dataclasses:**

| Class | Description |
|-------|-------------|
| `FeasibilityTask` | All parameters for one feasibility run |

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `run_single_analysis` | `(task: FeasibilityTask) -> dict` | Run feasibility for one combination (parallel-safe) |
| `generate_batch_summary` | `(results, output_path)` | Generate summary report |
| `process_batch` | `(config_path, output_base, num_workers, level1_only, detailed_per_trajectory_report) -> dict` | Main batch processor |

---

### 5.3 `combinatorial_search.py` — Combinatorial Ranking

**Purpose:** Exhaustive search across all (robot, knife_pose, toolpath) combinations with 4-level hierarchical feasibility ranking.

**Config file:** `config/combinatorial_search_config.yaml`

**Ranking hierarchy:**
1. **Level 1 — Feasibility:** reachability OK, C0 OK, C1 OK (binary)
2. **Level 2 — Safety:** `safety_tier = ceil(max_condition_number / bin_size)` (lower is better)
3. **Level 3 — Smoothness:** `mean_squared_velocity_ratio` (lower is better, power proxy)
4. **Level 4 — Dexterity:** `mean_manipulability` (higher is better)

**Key Dataclasses:**

| Class | Description |
|-------|-------------|
| `FeasibilityTask` | Task definition for parallel execution |
| `TrajectoryMetrics` | Per-trajectory computed metrics |
| `CombinationResult` | Result for one (robot, knife, toolpath) |
| `AggregatedKnifePoseResult` | Aggregated across toolpaths for (robot, knife) |
| `RobotRankingResult` | Robot-level ranking using best knife pose |

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_sort_key` | `(is_valid, safety_tier, smoothness_cost, dexterity_score) -> tuple` | Lexicographic sort key |
| `extract_trajectory_metrics` | `(trajectory_result) -> TrajectoryMetrics` | Extract metrics from raw result |
| `aggregate_trajectory_metrics` | `(metrics) -> dict` | Aggregate across trajectories |
| `aggregate_across_toolpaths` | `(results) -> dict` | Aggregate across toolpaths |
| `run_single_analysis` | `(task) -> CombinationResult` | Run one combination |
| `build_robot_ranking` | `(all_robot_results) -> list[RobotRankingResult]` | Build final robot rankings |
| `process_ranking_batch` | `(config_path, output_base, num_workers, ...) -> dict` | Main entry point |

**Outputs:** Per-robot CSVs, per-robot Markdown tables, knife pose detail reports, global ranking CSV, JSON metadata.

---

## 6. Configuration Files

### 6.1 `config/ik_config.yaml`

Shared IK solver parameters loaded by `load_ik_config_as_object()`.

| Section | Keys | Description |
|---------|------|-------------|
| `ik_parameters` | `max_iterations`, `tolerance`, `rot_weight`, `trans_weight`, `lambda0`, `lambda_max`, `max_step`, `backtrack` | Pinocchio damped LS tuning |
| | `ee_frame_name` | End-effector frame (`"ee_link"` or `"Link_6"`) |
| | `solution_selection` | EAIK strategy: `"closest"` or `"min_norm"` |
| | `use_initial_guess`, `use_neutral`, `use_random`, `num_random_retries` | Pinocchio retry strategy flags |

### 6.2 `config/robots_config.yaml`

Central robot database referenced by all scripts.

| Section | Keys | Description |
|---------|------|-------------|
| `constants` | `joint_jump_limit_rad` | C0 continuity threshold |
| `robots[]` | `name`, `description`, `urdf_path`, `reach_m`, `payload_kg`, `velocity_limits_rad_s`, `acceleration_limits_rad_s2` | Per-robot definition |

**Robots defined:** IRB 1300-7/1.4, IRB 1300-10/1.15, IRB 1300-11/0.9

### 6.3 `config/knife_config.yaml`

Knife pose definitions (T_B_K) in robot base frame.

| Section | Keys | Description |
|---------|------|-------------|
| `poses.<name>` | `description`, `translation_mm.{x,y,z}`, `rotation.{w,x,y,z}` | Knife pose with mm-to-m conversion |

### 6.4 `tests/configs/test_solvers_config.yaml`

Used by `test_solvers.py`.

| Key | Description |
|-----|-------------|
| `robot_name` | Robot name from robots_config |
| `input_folder` | Folder with RobotStudio CSVs |
| `output_folder` | Results output folder |
| `options.solver` | `"pin"` or `"eaik"` |
| `options.adaptive_scale` | Uniform vs adaptive plot Y-axis |
| `options.generate_fk_plots` | Enable FK plot generation |
| `options.generate_ik_plots` | Enable IK plot generation |

### 6.5 `tests/configs/reachability_config.yaml`

Used by `test_reachability.py`.

| Key | Description |
|-----|-------------|
| `robots_to_use[]` | List of robot names |
| `knife_poses_to_use[]` | List of knife pose names |
| `toolpaths_folder` | Input toolpath CSVs |
| `output_folder` | Results output |
| `options.solver` | `"pin"` or `"eaik"` |

### 6.6 `tests/configs/test_toolpaths_config.yaml`

Used by `test_toolpaths.py`.

| Key | Description |
|-----|-------------|
| `robots_to_use[]` | List of robot names |
| `knife_poses_to_use[]` | List of knife pose names |
| `toolpaths_folder` | Input toolpath CSVs |
| `robostudio_joints_folder` | Matching RobotStudio joint CSVs |
| `output_folder` | Results output |
| `options.solver` | `"pin"` or `"eaik"` |
| `options.save_joint_csv` | Save computed joints CSV |
| `options.generate_plots` | Enable plot generation |
| `options.num_workers` | Parallel workers (0 = all CPUs) |

### 6.7 `tests/configs/tolerance_config.yaml`

Used by `tolerance_check.py`.

| Key | Description |
|-----|-------------|
| `input_folder` | Folder containing toolpath subfolders with raw_comparison.csv |
| `report_output` | Path for tolerance report |
| `thresholds.fk_euclidean_error_mm` | Max FK position error (mm) |
| `thresholds.fk_rotation_error_deg` | Max FK rotation error (deg) |
| `thresholds.ik_joint_error_deg` | Max IK joint error (deg) |

### 6.8 `tests/configs/experiments_config.yaml`

Used by `run_experiments.py`. Defines automated experiment runs.

| Key | Description |
|-----|-------------|
| `enable_benchmarking` | Boolean to disable ground-truth comparison globally |
| `experiments[]` | List of experiment definitions |
| `experiments[].name` | Experiment name (e.g., `"Experiment_7"`) |
| `experiments[].test_script` | Script to run: `"test_solvers"`, `"test_reachability"`, `"test_toolpaths"` |
| `experiments[].robot` | Robot name (inherited by runs) |
| `experiments[].ee_frame` | End-effector frame (inherited by runs) |
| `experiments[].input` | Input data path (inherited by runs) |
| `experiments[].output_base` | Output base path |
| `experiments[].runs[]` | List of runs per experiment |
| `experiments[].runs[].run_name` | Run identifier (used as output subfolder) |
| `experiments[].runs[].solver` | `"eaik"` or `"pin"` |
| `experiments[].runs[].ground_truth` | Path to benchmark directory for regression testing |

Run-level keys override experiment-level keys: `solver`, `robot`, `ee_frame`, `input`, `output`, `knife_pose`, `ground_truth`.

### 6.9 `config/batch_feasibility_config.yaml`

Used by `feasibility_analysis.py` and `feasibility_analysis_batch.py`.

| Key | Description |
|-----|-------------|
| `robots_to_use[]` | Robot names |
| `knife_poses_to_use[]` | Knife pose names |
| `toolpaths_folder` | Input toolpaths |
| `output_folder` | Output directory |
| `checks.*` | Enable/disable: manipulability, singularity, reachability, condition_number, continuity |
| `thresholds.*` | Singularity/manipulability warning levels |
| `options.*` | `solver`, speed, workers, plot flags |

### 6.10 `config/combinatorial_search_config.yaml`

Used by `combinatorial_search.py`. Same structure as batch_feasibility_config with additional ranking options.

---

## 7. Data Flow & Architecture

### Solver Selection Flow

```
config YAML  ──┐
                ├──> load_ik_config_as_object(solver="pin"|"eaik")
CLI --solver ──┘         │
                         ├── solver="pin"  → PinocchioIKConfig
                         └── solver="eaik" → EAIKConfig
                                    │
                         create_solvers(urdf, solver, ik_config)
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    solver="eaik"         solver="pin"
                         │                     │
              load_robot_model_eaik    load_robot_model_pin
                         │                     │
               EAIKFKSolver            PinocchioFKSolver
               EAIKIKSolver           PinocchioIKSolver
               RobotModel             (pin.Model, pin.Data)
```

### Experiment Runner Flow

```
experiments_config.yaml
        │
  run_experiments.py
        │
        ├── For each experiment:
        │     ├── For each run:
        │     │     ├── build_command() → CLI args
        │     │     ├── subprocess.run(test_script)
        │     │     └── if ground_truth configured:
        │     │           ├── find_output_csvs()
        │     │           ├── compare_csvs() per file
        │     │           └── write_comparison_report()
        │     └── collect (exec_status, benchmark_status)
        └── print_summary()
```

### Coordinate Frame Transform Chain

```
T_P_K (toolpath in knife-part frame, mm)
  │  transform_t_p_k_to_t_k_p() — invert per waypoint
  ▼
T_K_P (part in knife frame)
  │  transform_t_k_p_to_t_b_p(knife_translation_m, knife_quaternion)
  ▼
T_B_P (part in robot base frame, meters)
  │  [x, y, z, qw, qx, qy, qz] per waypoint
  ▼
IK solver input
```

---

## 8. Dependencies

From `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `numpy` | Core numerical computation |
| `scipy` | Rotation utilities |
| `matplotlib` | Plot generation |
| `pandas` | CSV handling |
| `pyyaml` | YAML config loading |
| `pinocchio` | Pinocchio solver backend (optional) |
| `eaik` | EAIK analytical solver backend (optional) |
| `urchin` | URDF parsing for EAIK |
| `tqdm` | Progress bars for batch processing |
