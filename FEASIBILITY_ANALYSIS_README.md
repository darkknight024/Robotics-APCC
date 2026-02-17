# Feasibility Analysis

Comprehensive guide to the kinematic feasibility analysis tools for validating robot toolpath trajectories.

---

## Overview

The feasibility analysis system evaluates whether robot trajectories are kinematically feasible by checking:

- **Reachability** – Can the robot reach all waypoints? (IK solvability)
- **Manipulability** – Motion capability at each waypoint (Yoshikawa index)
- **Singularity proximity** – Distance from singular configurations
- **C¹ continuity** – Joint velocity limits compliance

---

## Core Components

### 1. `feasibility_analysis.py` – Single Toolpath

Analyzes one toolpath CSV for kinematic feasibility. Loads trajectories in T_P_K (plate frame), transforms to robot base frame, runs IK on each waypoint, computes metrics, and generates reports and plots.

**Key function:** `process_toolpath()` – orchestrates loading, transformation, analysis, and output.

### 2. `feasibility_analysis_batch.py` – Batch Processing

Runs feasibility analysis across multiple robots, knife poses, and toolpaths. Builds a task list and executes via `process_toolpath()` for each combination. Supports parallel execution.

**Key function:** `process_batch()` – discovers combinations, dispatches to workers, writes batch summary.

### 3. `core/feasibility_checks.py` – Analysis Logic

Provides the core feasibility logic:

- **FeasibilityAnalyzer** – Main analyzer class
  - `analyze_waypoint()` – Single waypoint IK + Jacobian metrics
  - `analyze_trajectory()` – Full trajectory with feasibility flags, safety, smoothness, dexterity
- **FeasibilityResult** – Dataclass with per-waypoint results (reachability, manipulability, condition number, singularity, joint velocity ratio, etc.)
- **compute_manipulability()** – Yoshikawa index: √det(J × J^T), normalized by robot reach
- **compute_singularity_proximity()** – Minimum singular value of Jacobian
- **compute_condition_number()** – κ = σ_max / σ_min
- **check_reachability()** – IK solver with retries

---

## IK Solver (`core/ik_solver.py`)

**Algorithm:** Damped least-squares (Levenberg–Marquardt style)

- Weighted SE(3) error (rotation + translation)
- Adaptive damping based on minimum singular value
- Backtracking line search
- Joint limit clipping

**Methods:**

- `solve()` – Single target pose
- `solve_with_retries()` – Uses neutral and random initial configs if first solve fails

**Config:** `config/ik_config.yaml` – `max_iterations`, `tolerance`, `rot_weight`, `trans_weight`, `lambda0`, `lambda_max`, `max_step`, `backtrack`, `ee_frame_name`.

---

## FK Solver (`core/fk_solver.py`)

**Responsibilities:**

- Compute end-effector pose from joint angles
- Provide Jacobian for IK and manipulability
- `solve()` – Single configuration → position, quaternion, rotation matrix
- `solve_batch()` – Multiple configurations
- `get_jacobian()` – 6×n Jacobian (local or world frame)

---

## Running the Scripts

### Single Toolpath (`feasibility_analysis.py`)

```bash
python feasibility_analysis.py --toolpath path/to/toolpath.csv --knife-pose pose_1

# Full options
python feasibility_analysis.py \
    --toolpath path/to/toolpath.csv \
    --urdf Assets/Robot\ APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf \
    --knife-config config/knife_config.yaml \
    --knife-pose pose_1 \
    --output output/feasibility/ \
    --reach 1.4 \
    --singularity-threshold 0.01 \
    --speed 100 \
    --no-continuity   # Skip C1 continuity analysis
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--toolpath`, `-t` | Required | Toolpath CSV file |
| `--urdf`, `-u` | IRB_1300_1400_URDF_with_fixture.urdf | Robot URDF path |
| `--knife-config`, `-k` | config/knife_config.yaml | Knife poses YAML |
| `--knife-pose` | pose_1 | Knife pose name |
| `--output`, `-o` | output/feasibility/ | Output directory |
| `--reach`, `-r` | 1.4 | Robot reach in meters |
| `--singularity-threshold` | 0.01 | Singularity warning threshold |
| `--speed` | 100 | End-effector speed in mm/s |
| `--no-continuity` | False | Skip continuity analysis |

### Batch Processing (`feasibility_analysis_batch.py`)

```bash
python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml

# With parallel workers
python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml --workers 4

# Custom output directory
python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml --output output/my_batch
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config`, `-c` | config/batch_feasibility_config.yaml | Path to batch config YAML |
| `--output`, `-o` | (from config) | Override output directory |
| `--workers`, `-w` | 1 | Number of parallel workers |

---

## Output Structure

### Single Toolpath

```
output/feasibility/robot_model_name/toolpath_name/knife_pose_name/
├── trajectory_1/
│   ├── reachability.png
│   ├── manipulability.png
│   ├── singularity.png
│   └── continuity.png
├── trajectory_2/
│   └── ...
├── aggregated_reachability_rate.png
├── aggregated_manipulability.png
├── aggregated_singularity.png
├── aggregated_continuity.png
├── feasibility_levels_comprehensive.png
├── reachability_summary.png
└── analysis_report.txt
```

### Batch Processing

```
output/feasibility_batch/
├── robot_name__knife_name__toolpath_name/
│   ├── trajectory_1/
│   │   ├── reachability.png
│   │   ├── manipulability.png
│   │   ├── singularity.png
│   │   └── continuity.png
│   ├── ...
│   ├── aggregated_*.png
│   └── analysis_report.txt
└── batch_summary.txt
```

### Report Contents

`analysis_report.txt` includes per-trajectory:

- Reachability (reachable count, unreachable waypoints)
- IK failure details (indices, positions, residuals, singular values)
- Singularity analysis
- Manipulability (mean, min)
- Continuity (pass/fail, max joint velocities, violations)

---

## Solver Comparison Scripts

The repo includes scripts to compare Pinocchio FK/IK against RobotStudio test data:

### 1. Test Trajectory Comparison

```bash
python solver_comparison_test_trajectory.py --input path/to/csv_or_folder
```

**Config:** `config/robostudio_test_config.yaml`

**Output:** `analysis.txt`, `global_analysis.txt`, FK/IK comparison plots.

### 2. Toolpath Trajectory Comparison

```bash
python solver_comparison_toolpath_trajectory.py --config config/toolpath_config.yaml
```

**Config:** `config/toolpath_config.yaml` – robots, knife poses, toolpaths folder, RobotStudio joints folder.

Transforms T_P_K → T_B_P, runs IK, compares with recorded joints.

---

## Configuration Files

### `config/batch_feasibility_config.yaml`

Main config for feasibility and batch runs:

```yaml
robots_to_use: ["IRB 1300-7/1.4"]
knife_poses_to_use: ["pose_1"]
toolpaths_folder: "Assets/Robot APCC/Toolpaths/Successful"
output_folder: "output/feasibility_batch"

checks:
  manipulability: true
  singularity: true
  reachability: true
  condition_number: false
  continuity: true

thresholds:
  singularity_warning: 0.01
  manipulability_warning: 0.001

performance:
  max_ik_failures_per_trajectory: 1   # Early terminate trajectory after N failures

continuity:
  enabled: true
  pose_scale_m_per_rad: 0.1
  safety_factor: 1.05
  default_speed_mm_s: 100.0
```

### `config/ik_config.yaml`

IK solver parameters (tolerance, iterations, damping, weights, end-effector frame).

### `config/robots_config.yaml`

Robot definitions: URDF paths, reach, velocity limits. Referenced by name in batch configs.

### `config/knife_config.yaml`

Knife poses (T_B_K transforms: translation in mm, quaternion [w,x,y,z]).

---

## Coordinate Frames

- **T_P_K** – Knife trajectory in plate frame (input CSV)
- **T_B_K** – Knife pose in robot base frame (from `knife_config.yaml`)
- **T_B_P** – Plate pose in base frame (derived)

Transformation: `T_B_P = T_B_K @ inv(T_P_K)`

---

## References

- [MASTER_README.md](MASTER_README.md) – Repo overview and installation
- [COMBINATORIAL_SEARCH_README.md](COMBINATORIAL_SEARCH_README.md) – Ranking and combinatorial search
