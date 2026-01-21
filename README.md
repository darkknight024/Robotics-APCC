# Robotics-APCC

> Kinematic Analysis & Validation Toolkit for ABB IRB 1300 Robots

A comprehensive toolkit for validating robot kinematics using Pinocchio against ABB RobotStudio, with feasibility analysis for toolpath trajectories.

---

## Installation

### 1. Install Miniconda/Anaconda

If you don't have conda installed, download and install from:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended, lightweight)
- [Anaconda](https://www.anaconda.com/download/)

### 2. Create and Activate Environment

```bash
# Create environment with Python 3.12
conda create -n robotics python=3.12 -y

# Activate the environment
conda activate robotics
```

### 3. Install Pinocchio

Pinocchio cannot be installed via pip on Windows. Use conda:

```bash
conda install pinocchio -c conda-forge
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

**Required packages in requirements.txt:**
- numpy
- pandas
- matplotlib
- pyyaml
- scipy

---

## Running the Scripts

### 1. Combinatorial Search & Ranking

Perform a full combinatorial search across robots, knife poses, and toolpaths to find the optimal knife placement.

```bash
# Basic usage
python combinatorial_search.py --config config/batch_feasibility_config.yaml

# Advanced usage with multiple workers and custom weights
python combinatorial_search.py \
    --config config/batch_feasibility_config.yaml \
    --weights config/scoring_weights.yaml \
    --workers 8 \
    --output output/feasibility_ranking

# Generate plots and reports for every trajectory in the toolpath (slow to run)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --workers 8 --detailed_per_trajectory_report
```

**Algorithm Details:**
- **Combinatorial Expansion**: Generates the full cross-product of (Robot x Knife Pose x Toolpath).
- **Metric Extraction**: Processes each combination to extract `IK_failure_rate`, `singularity_rate`, and `manipulability` indices.
- **Normalization**: Scales metrics to [0, 1] range (0=best, 1=worst) using per-robot min/max bounds to ensure fair comparison.
- **Weighted Scoring**: Combines normalized metrics into a single `normalized_score` based on user-defined weights.
- **Ranking**: Aggregates scores across all toolpaths to rank knife poses from most feasible to least.

---

### 2. Solver Comparison - Test Trajectory

Compare Pinocchio FK/IK with RobotStudio test data containing both joint angles and task-space poses.

```bash
# Using config file (recommended)
# Ensure to update the config/robostudio_test_config.yaml with the correct input folder that contains all the test csv files that Praneeth generated from RobotStudio
#Example: input_folder = Experiment_8/Experiment_8/square_profile_trajectories_sampled
python solver_comparison_test_trajectory.py

# Override input path
python solver_comparison_test_trajectory.py --input path/to/csv_or_folder

# Full CLI mode (no config)
python solver_comparison_test_trajectory.py \
    --input data/test.csv \
    --urdf path/to/robot.urdf
```

**Config:** `config/robostudio_test_config.yaml`
```yaml
robot_name: "IRB 1300-7/1.4"       # References robots_config.yaml
input_folder: "input/robostudio_trajectories"
output_folder: "output/test_comparison"
```

**Outputs:**
- `analysis.txt` - Per-CSV FK/IK error statistics
- `global_analysis.txt` - Aggregated statistics across all CSVs
- FK/IK comparison plots

---

### 3. Solver Comparison - Toolpath Trajectory

Transform toolpath poses (T_P_K) to base frame, run IK, compare with RobotStudio recorded joints.

```bash
python solver_comparison_toolpath_trajectory.py --config config/toolpath_config.yaml
```

**Config:** `config/toolpath_config.yaml`
```yaml
robots_to_use:
  - "IRB 1300-7/1.4"              # References robots_config.yaml

knife_poses_to_use:
  - "pose_1"                       # References knife_config.yaml

toolpaths_folder: "input/toolpaths"
robostudio_joints_folder: "input/robostudio_joints"
output_folder: "output/toolpath_comparison"
```

> **Note:** RobotStudio joint CSVs must have the **same filename** as corresponding toolpath CSVs.

---

### 4. Feasibility Analysis - Single Toolpath

Analyze kinematic feasibility including reachability, manipulability, singularity proximity, and C1 continuity.

```bash
python feasibility_analysis.py \
    --toolpath path/to/toolpath.csv \
    --knife-pose pose_1 \
    --speed 100
```

**Outputs:**
- `analysis_report.txt` - Human-readable feasibility report
- Per-trajectory plots: reachability, manipulability, singularity, continuity

---

### 5. Feasibility Analysis - Batch

Analyze multiple toolpaths across multiple robots and knife poses.

```bash
python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml
```

**Config:** `config/batch_feasibility_config.yaml`
```yaml
robots_to_use:
  - "IRB 1300-7/1.4"

knife_poses_to_use:
  - "pose_1"

toolpaths_folder: "Assets/Robot APCC/Toolpaths/Successful"
output_folder: "output/feasibility_batch"
```

---

### 6. Generate Knife Poses

Generate a grid of candidate knife poses (nominal grid + out-of-reach samples) to be evaluated by the combinatorial search.

```bash
# Basic usage (uses defaults: 5x5x3 grid, 2 orientations, 6 out-of-reach)
python utils/generate_knife_poses.py

# Custom grid density and output path
python utils/generate_knife_poses.py \
    --num_x 10 \
    --num_y 10 \
    --num_z 5 \
    --num_ori 4 \
    --num_out_of_reach 6 \
    --output_path config/sparse_generated_knife_poses.yaml
```

**CLI Arguments:**
- `--num_x`: Number of x-axis divisions (default: 5)
- `--num_y`: Number of y-axis divisions (default: 5)
- `--num_z`: Number of z-axis divisions (default: 3)
- `--num_ori`: Number of orientation variations per grid point (default: 2)
- `--num_out_of_reach`: Number of out-of-reach poses per robot (max 6, default: 6)
- `--output_path`: Path to save the generated poses (default: `config/generated_knife_poses.yaml`)

---

## Project Structure

```
Robotics-APCC/
|
|-- core/                              # Core kinematic solvers
|   |-- ik_solver.py                   # Pinocchio IK solver (damped least squares)
|   |-- fk_solver.py                   # Pinocchio FK solver + Jacobian
|   |-- feasibility_checks.py          # Manipulability, singularity, reachability
|   +-- robot_loader.py                # URDF loader
|
|-- utils/                             # Utility modules
|   |-- transform_handler.py           # T_P_K to T_B_P frame transformations
|   |-- csv_loader_toolpath.py         # Toolpath CSV loader (T_P_K format)
|   |-- csv_loader_robostudio.py       # RobotStudio CSV loader
|   |-- config_loader.py               # YAML config loading + robot resolution
|   |-- generate_plot_ik.py            # IK comparison plots
|   |-- generate_plot_fk.py            # FK comparison plots
|   +-- feasibility_plot.py            # Feasibility + continuity plots
|
|-- config/                            # Configuration files
|   |-- robots_config.yaml             # CENTRAL robot definitions
|   |-- knife_config.yaml              # Static knife poses
|   |-- generated_knife_poses.yaml     # AUTO-GENERATED grid of knife poses
|   |-- ik_config.yaml                 # IK solver parameters
|   |-- scoring_weights.yaml           # Weights for combinatorial ranking
|   |-- robostudio_test_config.yaml    # Test trajectory settings
|   |-- toolpath_config.yaml           # Toolpath processing settings
|   +-- batch_feasibility_config.yaml  # Batch & Ranking settings
|
|-- Assets/Robot APCC/                # Robot assets
|   |-- IRB-1300-*/urdf/               # URDF files
|   +-- Toolpaths/                     # Toolpath CSVs
|
|-- solver_comparison_test_trajectory.py
|-- solver_comparison_toolpath_trajectory.py
|-- feasibility_analysis.py
|-- feasibility_analysis_batch.py
+-- combinatorial_search.py            # Main ranking & optimization entry point
```

---

## Configuration Files

### 1. Central Robot Config (`config/robots_config.yaml`)

All robots are defined here once. Other configs reference robots by name.

```yaml
robots:
  - name: "IRB 1300-7/1.4"
    urdf_path: "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF.urdf"
    reach_m: 1.4
    velocity_limits_rad_s: [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]
    
  - name: "IRB 1300-10/1.15"
    urdf_path: "Assets/Robot APCC/IRB-1300 1150 URDF/urdf/..."
    reach_m: 1.15
    velocity_limits_rad_s: [4.887, 3.979, 5.864, 8.727, 7.245, 12.566]
```

### 2. Knife Poses (`config/knife_config.yaml`)

Defines the T_B_K transform (knife pose in robot base frame).

```yaml
poses:
  pose_1:
    description: "Standard knife pose"
    translation_mm:
      x: -367.773
      y: -915.815
      z: 520.4
    rotation:  # Quaternion [w, x, y, z]
      w: 0.00515984
      x: 0.712632
      y: -0.701518
      z: 0.000396522
```

### 3. IK Solver Parameters (`config/ik_config.yaml`)

```yaml
ik_parameters:
  max_iterations: 1000
  tolerance: 1e-4
  ee_frame_name: "ee_link"  # Must match URDF end-effector frame
```

### 4. Batch & Ranking Config (`config/batch_feasibility_config.yaml`)

Controls which robots, toolpaths, and knife poses are processed in batch or ranking modes.

```yaml
robots_to_use:
  - "IRB 1300-7/1.4"
toolpaths_folder: "input/toolpaths"
output_folder: "output/feasibility_ranking"
checks:
  manipulability: true
  singularity: true
  reachability: true
  continuity: true
```

### 5. Scoring Weights (`config/scoring_weights.yaml`)

Defines how different kinematic heuristics are weighted to compute the final feasibility score.

```yaml
weights:
  w_IK_failure_rate: 50.0       # Weight for reachability
  w_singularity_rate: 25.0      # Weight for singularity avoidance
  w_min_manipulability: 10.0    # Weight for worst-case dexterity
  w_mean_manipulability: 10.0   # Weight for average dexterity
  w_mean_min_singular_value: 5.0 # Weight for singularity proximity
```

### 6. Generated Knife Poses (`config/generated_knife_poses.yaml`)

Contains the auto-generated grid of candidate knife poses to be evaluated.

```yaml
poses:
  nominal_x-467.8_y-1015.8_z420.4_oriA:
    description: "Nominal grid pose"
    translation_mm: {x: -467.7, y: -1015.8, z: 420.4}
    rotation: {w: 0.0019, x: 0.6982, y: -0.7147, z: 0.0392}
```

---

## Coordinate Frames

```
T_P_K  ->  Knife trajectory in Plate frame (input from toolpath CSV)
T_K_P  ->  Plate pose in Knife frame (inverse of T_P_K)
T_B_K  ->  Knife pose in robot Base frame (from knife_config.yaml)
T_B_P  ->  Plate (end-effector target) in Base frame
```

**Transformation Chain:**
```
T_B_P = T_B_K @ inv(T_P_K)
```

---

## Output Structure

### 1. Combinatorial Search & Ranking
```
output/feasibility_ranking/
├── per_robot/
│   ├── IRB_1300-7_1.4_knifepose_ranking.csv
│   ├── IRB_1300-7_1.4_detailed_results.json
│   └── IRB_1300-7_1.4_ranking_plot.png
├── robot_name__knife_name__toolpath_name/
│   └── summary.json
├── global_ranking.csv
├── batch_ranking_summary.json
└── feasibility_ranking_report.md
```

### 2. Test Trajectory Comparison
```
output/test_comparison/
|-- csv_file_1/
|   |-- analysis.txt
|   |-- fk_position_comparison.png
|   |-- fk_position_deltas.png
|   |-- ik_joint_comparison.png
|   +-- ik_joint_deltas.png
|-- csv_file_2/
|   +-- ...
+-- global_analysis.txt
```

### Feasibility Analysis
```
output/feasibility/
|-- trajectory_1/
|   |-- reachability.png
|   |-- manipulability.png
|   |-- singularity.png
|   +-- continuity.png
+-- analysis_report.txt
```

---

## Quick Reference

| What | Where |
|------|-------|
| Define robots | `config/robots_config.yaml` |
| Define knife poses | `config/knife_config.yaml` |
| IK solver tuning | `config/ik_config.yaml` |
| Test comparison config | `config/robostudio_test_config.yaml` |
| Toolpath batch config | `config/toolpath_config.yaml` |
| Feasibility batch config | `config/batch_feasibility_config.yaml` |
