## Combinatorial Search (Updated)

This document describes the current combinatorial search pipeline that ranks robot + knife pose combinations across a batch of toolpaths using a 4-level lexicographic “waterfall” key.

### Inputs
- **Robot models**: URDF paths and robot configs from `config/combinatorial_search_config.yaml`.
- **Knife poses**: Loaded from `config/sparse_generated_knife_poses.yaml` (fallback to `config/knife_config.yaml`).
- **Toolpaths**: CSV files from the configured toolpath folder(s).
- **Feasibility config**: `config/batch_feasibility_config.yaml` (singularity threshold, continuity settings).

### Pipeline Overview
1. **Build tasks** for every `(robot, knife pose, toolpath)` combination.
2. **Run per-toolpath analysis** via `process_toolpath`, producing per-trajectory metrics.
3. **Aggregate per-trajectory metrics** into a single combination result.
4. **Aggregate across toolpaths** to get a single result per `(robot, knife pose)`.
5. **Lexicographically rank** knife poses per robot using the 4-level feasibility tuple.
6. **Select best knife per robot** and rank robots using the same lexicographic tuple.
7. **Emit outputs** (CSV, JSON, Markdown, plots).

### Per-Waypoint Metrics (Computed in `core/feasibility_checks.py`)
Each waypoint computes:
- IK solution (`q`), reachability, Jacobian condition number, manipulability.
- Joint-space delta (`Δq`) between waypoints.
- Velocity ratios `|dq/dt| / limit` for C1 feasibility.

### Per-Trajectory Metrics (Computed in `feasibility_analysis.py`)
For a trajectory:
- **Level 1 (Validity)**: all reachability + C0 + C1 checks must pass.
- **Level 2 (Safety Tier)**: `ceil(max_condition_number / bin_size)`.
- **Level 3 (Smoothness Cost)**: normalized joint energy using velocity ratios.
- **Level 4 (Dexterity Score)**: mean manipulability.

### Aggregation Strategy
Aggregation uses conservative (worst‑case) logic:

- **Within a toolpath (across trajectories)**:
  - `is_valid`: all trajectories must be valid.
  - `safety_tier`: max (worst) tier.
  - `smoothness_cost`: max (worst) cost.
  - `dexterity_score`: mean across trajectories.

- **Across toolpaths (per robot + knife pose)**:
  - `is_valid`: all toolpaths must be valid.
  - `safety_tier`: max (worst) tier across toolpaths.
  - `smoothness_cost`: max (worst) cost across toolpaths.
  - `dexterity_score`: mean across toolpaths.

### Lexicographic Ranking Key
All ranking uses a strict, ordered tuple:

```
(invalid_flag, safety_tier, smoothness_cost, -dexterity_score)
```

Interpretation:
- **`invalid_flag`**: `0` if valid, `1` if invalid → all valid combinations rank ahead of invalid.
- **`safety_tier`**: lower is better.
- **`smoothness_cost`**: lower is better.
- **`dexterity_score`**: higher is better (negated for ascending sort).

This tuple is computed for each `(robot, knife pose)` and for each best‑knife‑per‑robot.

### Outputs
The batch run emits:
- **Per-robot**: `knife_pose_ranking.csv`, `knife_pose_ranking.md`, `metadata.json`, `detailed_results.json`.
- **Global**: `global_ranking.csv`, `robot_ranking.csv`, `batch_ranking_summary.json`, `feasibility_ranking_report.md`.
- **Plots**: aggregated reachability/manipulability/singularity and 4‑level feasibility plots per combination.

### Summary JSON (Per Combination)
Each combination’s `summary.json` includes:
- Aggregated 4‑level metrics and `feasibility_tuple` with per‑field comments.
- Raw kinematic metrics (IK failure, singularity rate, manipulability stats).
