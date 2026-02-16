# Robotics-APCC

> Kinematic Analysis & Validation Toolkit for ABB IRB 1300 Robots

A toolkit for validating robot kinematics using Pinocchio against ABB RobotStudio, with feasibility analysis and combinatorial search for optimal knife placement.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Documentation Guide](#documentation-guide)
- [Data Flow](#data-flow)
- [Coordinate Frames](#coordinate-frames)
- [Quick Reference](#quick-reference)

---

## Installation

### 1. Install Miniconda/Anaconda

If you don't have conda installed:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended, lightweight)
- [Anaconda](https://www.anaconda.com/download/)

### 2. Create and Activate Environment

```bash
conda create -n robotics python=3.12 -y
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

**Required packages:** numpy, pandas, matplotlib, pyyaml, scipy, tqdm

---

## Repository Structure

```
Robotics-APCC/
├── core/                              # Core kinematic solvers
│   ├── ik_solver.py                   # Pinocchio IK (damped least-squares)
│   ├── fk_solver.py                   # Pinocchio FK + Jacobian
│   ├── feasibility_checks.py         # Manipulability, singularity, reachability
│   └── robot_loader.py                # URDF loader
│
├── utils/                             # Utility modules
│   ├── transform_handler.py           # T_P_K → T_B_P frame transforms
│   ├── csv_loader_toolpath.py         # Toolpath CSV loader
│   ├── csv_loader_robostudio.py       # RobotStudio CSV loader
│   ├── config_loader.py               # YAML config loading
│   ├── feasibility_plot.py            # Feasibility & continuity plots
│   └── generate_plot_ik.py / generate_plot_fk.py
│
├── config/                            # Configuration
│   ├── robots_config.yaml             # Robot definitions (URDF, limits)
│   ├── knife_config.yaml              # Static knife poses
│   ├── generated_knife_poses.yaml     # Generated grid of knife poses
│   ├── sparse_generated_knife_poses.yaml
│   ├── ik_config.yaml                 # IK solver parameters
│   ├── scoring_weights.yaml           # Combinatorial ranking weights
│   ├── batch_feasibility_config.yaml  # Batch feasibility settings
│   ├── combinatorial_search_config.yaml
│   ├── robostudio_test_config.yaml
│   └── toolpath_config.yaml
│
├── feasibility_analysis.py           # Single toolpath feasibility
├── feasibility_analysis_batch.py      # Batch feasibility analysis
├── combinatorial_search.py            # Combinatorial search & ranking
├── solver_comparison_test_trajectory.py
├── solver_comparison_toolpath_trajectory.py
│
├── Assets/Robot APCC/                 # Robot assets
│   ├── IRB-1300-*/urdf/               # URDF files
│   └── Toolpaths/                     # Toolpath CSVs
│
├── MASTER_README.md                   # This file – overview & installation
├── FEASIBILITY_ANALYSIS_README.md     # Feasibility scripts & IK/FK
└── COMBINATORIAL_SEARCH_README.md     # Combinatorial search & ranking
```

---

## Documentation Guide

| Document | Purpose |
|----------|---------|
| **[MASTER_README.md](MASTER_README.md)** | Repo overview, installation, structure (this file) |
| **[FEASIBILITY_ANALYSIS_README.md](FEASIBILITY_ANALYSIS_README.md)** | Single toolpath and batch feasibility analysis. Covers `feasibility_analysis.py`, `feasibility_analysis_batch.py`, `core/feasibility_checks.py`, IK/FK solvers, CLI args, output layout, and configs. |
| **[COMBINATORIAL_SEARCH_README.md](COMBINATORIAL_SEARCH_README.md)** | Combinatorial search and ranking. Covers `combinatorial_search.py`, metrics, scoring, output structure, and configs. |

---

## Data Flow

```
INPUT SOURCES
├── CSV Trajectory Files (T_P_K, positions in mm)
├── Robot URDF Files
└── Config YAML (knife poses, robots, thresholds)

        ↓
        
TRANSFORMATION
├── Load toolpath CSV (T_P_K format)
├── Parse trajectories (separated by "T0")
├── Convert mm → meters
└── Transform T_P_K → T_B_P using knife pose (T_B_K)

        ↓
        
KINEMATIC ANALYSIS (per waypoint)
├── IK solve (damped least-squares via Pinocchio)
├── Jacobian computation
├── Manipulability (Yoshikawa: √det(J×J^T))
├── Singular value / condition number
└── C¹ continuity (joint velocity limits)

        ↓
        
OUTPUT
├── Feasibility: analysis_report.txt, plots, per-trajectory metrics
└── Combinatorial: ranking CSVs, JSON, reports, plots
```

### Processing Pipeline (Single Trajectory)

1. **Transformation** – Apply knife pose to get T_B_P poses  
2. **Kinematic analysis** – IK per pose, manipulability, singularity  
3. **Continuity** – C¹ velocity checks (if all reachable)  
4. **Visualization & reporting** – Plots and text reports  

---

## Coordinate Frames

| Frame | Description |
|-------|-------------|
| **T_P_K** | Knife trajectory in plate frame (input CSV) |
| **T_K_P** | Plate in knife frame (inverse of T_P_K) |
| **T_B_K** | Knife pose in robot base (from knife_config.yaml) |
| **T_B_P** | Plate (end-effector target) in base frame |

**Transformation chain:**

```
CSV (T_P_K)  →  T_B_P = T_B_K @ inv(T_P_K)  →  IK Solver  →  Joint Angles
```

**Conventions:** Meters (internally), millimeters (CSV), radians (angles). Quaternions: [w, x, y, z].

---

## Key Algorithms

### IK Solver (core/ik_solver.py)

- **Method:** Damped least-squares (Levenberg–Marquardt style)
- **Error:** Weighted SE(3) (rotation + translation)
- **Damping:** Adaptive based on Jacobian singular values
- **Retries:** Neutral config + random configs if initial solve fails

### Yoshikawa Manipulability

- **Formula:** m = √det(J × J^T), Jacobian normalized by robot reach
- **Interpretation:** m → 0 near singularity; higher m = more dexterity

### C¹ Continuity

- **Metric:** Unified pose distance d = √(d_linear² + (scale × d_angle)²)
- **Timing:** From speed (CSV) and joint velocity limits
- **Check:** max(|Δq_j|/dt) / velocity_limit_j ≤ 1.0

---

## Quick Reference

| Task | Command |
|------|---------|
| Single toolpath analysis | `python feasibility_analysis.py --toolpath <csv> --knife-pose pose_1` |
| Batch feasibility | `python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml --workers 4` |
| Combinatorial search | `python combinatorial_search.py --config config/combinatorial_search_config.yaml --workers 8` |
| Generate knife poses | `python utils/generate_knife_poses.py` |
| Solver comparison (test) | `python solver_comparison_test_trajectory.py --input <path>` |
| Solver comparison (toolpath) | `python solver_comparison_toolpath_trajectory.py --config config/toolpath_config.yaml` |

| Config | Purpose |
|--------|---------|
| `config/robots_config.yaml` | Robot definitions |
| `config/knife_config.yaml` | Knife poses |
| `config/ik_config.yaml` | IK solver tuning |
| `config/batch_feasibility_config.yaml` | Batch feasibility |
| `config/combinatorial_search_config.yaml` | Combinatorial search |
| `config/scoring_weights.yaml` | Ranking weights |

---

## References

- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) – Rigid-body dynamics and kinematics
- [ABB IRB 1300](https://new.abb.com/industrial-robots) – Robot specifications
- Yoshikawa (1985), "Manipulability of Robotic Mechanisms"
