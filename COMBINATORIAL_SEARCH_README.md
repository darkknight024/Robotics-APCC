# Combinatorial Search

Guide to the combinatorial feasibility search and knife-pose ranking system.

---

## Overview

The combinatorial search evaluates **R robots × K knife poses × T toolpaths** to find the best robot and knife pose combination. It generates feasibility metrics, normalizes them per robot, applies weighted scoring, and produces ranked reports.

---

## Core Component: `combinatorial_search.py`

Main entry point for the combinatorial search. It:

1. **Builds combinations** – Cross-product of robots, knife poses, and toolpaths
2. **Processes each combination** – Uses `process_toolpath()` from `feasibility_analysis.py` (no plots by default)
3. **Aggregates metrics** – Worst-case across trajectories and toolpaths per knife pose
4. **Normalizes** – Per-robot scaling to [0, 1] (0=best, 1=worst)
5. **Scores** – Weighted sum with configurable weights
6. **Ranks** – Sorts knife poses within each robot; ranks robots by raw metrics
7. **Generates reports** – CSV, JSON, Markdown, and plots

---

## Running the Script

```bash
# Basic usage
python combinatorial_search.py --config config/combinatorial_search_config.yaml

# With parallel workers (recommended: 4–8)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --workers 8

# Choose IK solver (overrides config file)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --solver eaik
python combinatorial_search.py --config config/combinatorial_search_config.yaml --solver pin

# Feasibility-only mode: IK reachability check only (no ranking, no continuity)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --feasibility_only
python combinatorial_search.py --config config/combinatorial_search_config.yaml --feasibility_only --solver eaik

# Custom output
python combinatorial_search.py --config config/combinatorial_search_config.yaml --output output/ranking

# Custom knife pose set
python combinatorial_search.py --config config/combinatorial_search_config.yaml --knife-config config/generated_knife_poses.yaml

# Generate per-trajectory plots (slower)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --detailed_per_trajectory_report

# Enable plots (disabled by default for speed)
python combinatorial_search.py --config config/combinatorial_search_config.yaml --plots

# Validate existing outputs
python combinatorial_search.py --validate --output output/feasibility_ranking

# Debug logging
python combinatorial_search.py --config config/combinatorial_search_config.yaml --debug
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config`, `-c` | combinatorial_search_config.yaml | Batch config YAML |
| `--output`, `-o` | output/feasibility_ranking | Output directory |
| `--workers`, `-w` | 1 | Parallel workers |
| `--knife-config` | sparse_generated_knife_poses.yaml | Knife poses YAML |
| `--solver` | From config (default: `pin`) | IK solver: `pin` (Pinocchio) or `eaik` (EAIK analytical). Overrides config file. |
| `--feasibility_only` | False | IK feasibility check only (no ranking). Outputs `Feasibility-report.md`. |
| `--debug` | False | Enable debug logging |
| `--validate` | False | Only validate existing outputs |
| `--detailed_per_trajectory_report` | False | Per-trajectory plots |
| `--plots` | False | Save PNG plots |

---

## Trajectory Metrics and Ranking

### Per-Trajectory Metrics

From feasibility analysis:

- **IK failure rate** – Fraction of unreachable waypoints
- **Singularity rate** – Fraction of waypoints near singularity
- **Min manipulability** – Worst-case Yoshikawa index
- **Mean manipulability** – Average dexterity
- **Mean min singular value** – Average σ_min (singularity proximity)

### 4-Level Feasibility

Trajectories are evaluated with lexicographic ordering:

1. **Level 1 (feasibility gate)** – Reachability OK, C0 OK, C1 OK
2. **Level 2 (safety tier)** – Condition number binned (lower = safer)
3. **Level 3 (smoothness cost)** – Normalized joint energy (lower = smoother)
4. **Level 4 (dexterity)** – Mean manipulability (higher = better)

### Aggregation

- **Trajectory → toolpath:** Worst-case for failures (max IK failure rate, max singularity rate), min for min manipulability, mean for averages
- **Toolpath → knife pose:** Same worst-case aggregation across toolpaths

---

## Scoring Logic

### Normalization (per robot)

Metrics are normalized to [0, 1] where **0 = best**, **1 = worst**:

- **Lower-is-better** (e.g. IK failure rate, singularity rate): `(v - min) / (max - min)`
- **Higher-is-better** (manipulability, singular values): inverted so best → 0

### Weighted Score

```
raw_score = w_IK × IK_rate + w_sing × sing_rate + w_min_manip × norm_min_manip
          + w_mean_manip × norm_mean_manip + w_SV × norm_mean_min_SV
normalized_score = raw_score / total_weight
if IK_failure_rate > 0:
    normalized_score += 1.0   # Penalty for infeasibility
```

### Default Weights (`config/scoring_weights.yaml`)

| Weight | Value | Metric |
|--------|-------|--------|
| w_IK_failure_rate | 50 | Reachability |
| w_singularity_rate | 25 | Singularity avoidance |
| w_min_manipulability | 10 | Worst-case dexterity |
| w_mean_manipulability | 10 | Average dexterity |
| w_mean_min_singular_value | 5 | Singularity proximity |

### Verdict Thresholds

| Score Range | Verdict |
|-------------|---------|
| 0.00 – 0.25 | Recommended |
| 0.25 – 0.50 | Borderline |
| 0.50 – 0.75 | Poor |
| ≥ 0.75 or IK failure > 0 | Infeasible |

### Robot Ranking

Robot ranking uses **raw metrics** of the best knife pose per robot (not normalized scores), because normalization is per-robot and not comparable across robots.

---

## Output Structure

Each run creates a timestamped folder:

```
output/feasibility_ranking/<HH_MM_SS>/
├── per_robot/
│   └── <robot_name>/                    # e.g., IRB_1300-7_1.4/
│       ├── knife_pose_ranking.csv       # Main ranking table
│       ├── knife_pose_ranking.md        # Markdown version
│       ├── metadata.json                # Summary stats
│       ├── detailed_results.json        # Full data
│       ├── ranking_plot.png             # Bar chart
│       └── knife_poses/
│           └── <knife_pose_id>/
│               ├── toolpath_details.csv # Per-toolpath breakdown
│               └── details.json
├── <robot>__<knife>__<toolpath>/        # Per-combination
│   └── summary.json
├── global_ranking.csv                   # All (robot, knife) combos
├── robot_ranking.csv                    # Best pose per robot
├── batch_ranking_summary.json
└── feasibility_ranking_report.md        # Human-readable report
```

### Output File Descriptions

| File | Description |
|------|-------------|
| `knife_pose_ranking.csv` | Main ranking with Rank, Knife Pose ID, Score, metrics, Verdict |
| `knife_pose_ranking.md` | Markdown table of the ranking |
| `metadata.json` | Counts, verdict breakdown, best/worst pose IDs |
| `detailed_results.json` | Raw and normalized metrics for each knife pose |
| `ranking_plot.png` | Bar chart of top/bottom knife poses |
| `toolpath_details.csv` | Per-toolpath metrics for a knife pose |
| `details.json` | Full metrics, rank, and per-toolpath results per knife pose |
| `global_ranking.csv` | All (robot, knife) combos with scores |
| `robot_ranking.csv` | One row per robot (best knife pose) for cross-robot comparison |
| `batch_ranking_summary.json` | Run summary: counts, timestamps, top poses |
| `feasibility_ranking_report.md` | Text report with top/bottom poses and failure analysis |

---

## IK Solver Selection

The `--solver` CLI flag selects which IK backend to use. It overrides the `solver` field in the config YAML.

| Solver | Backend | Description |
|--------|---------|-------------|
| `pin` | Pinocchio | Numerical damped least-squares IK with multi-strategy retries |
| `eaik` | EAIK | Analytical IK returning all solutions, filtered by joint limits |

Both solvers implement the same `BaseIKSolver` / `BaseFKSolver` interfaces. All downstream metrics (manipulability, singularity proximity, condition number, smoothness, dexterity) are computed from the solver-provided Jacobian and joint solutions, so results are comparable across backends.

---

## Feasibility-Only Mode (`--feasibility_only`)

When `--feasibility_only` is set, the script runs a lightweight IK reachability check instead of the full 4-level ranking pipeline:

- **What it checks**: Whether each waypoint in every toolpath is individually IK solvable for each (robot, knife pose) combination.
- **What it skips**: C0/C1 continuity analysis, Level 2–4 scoring (safety tier, smoothness, dexterity), ranking, and weighted scoring.
- **Verdict**: A combination is **feasible** (`Yes`) only if 100% of waypoints across all toolpaths are IK reachable. Even one failed waypoint marks it as **infeasible** (`No`).

### Output

Instead of the full ranking output, feasibility-only mode produces a single report:

```
output/feasibility_ranking/<timestamp>/
└── Feasibility-report.md
```

The report contains:
- Total robot models, knife poses, toolpaths, and combinations
- IK solver used
- A table of every (robot, knife pose) combination with IK feasibility rate and verdict
- Per-robot breakdown with best IK rate

---

## Configuration Files

### `config/combinatorial_search_config.yaml`

Same structure as batch feasibility config:

```yaml
robots_to_use: ["IRB 1300-7/1.4", "IRB 1300-10/1.15"]
knife_poses_to_use: ["pose_1"]
toolpaths_folder: "Assets/Robot APCC/Toolpaths/Sample"
output_folder: "output/combinatorial_search"

checks:
  manipulability: true
  singularity: true
  reachability: true
  continuity: true

thresholds:
  singularity_warning: 0.01
  manipulability_warning: 0.001

performance:
  max_ik_failures_per_trajectory: 1

continuity:
  enabled: true
  pose_scale_m_per_rad: 0.1
  safety_factor: 1.05
  default_speed_mm_s: 100.0
```

### `config/scoring_weights.yaml`

Weights for the weighted score (see Scoring Logic above).

### `config/sparse_generated_knife_poses.yaml` (default)

Generated grid of knife poses. Can be replaced by `config/generated_knife_poses.yaml` or a custom file via `--knife-config`.

---

## Generating Knife Poses

```bash
python utils/generate_knife_poses.py --num_x 10 --num_y 10 --num_z 5 --output_path config/my_poses.yaml
```

Output is a YAML of candidate knife poses used by the combinatorial search.

---

## References

- [FEASIBILITY_ANALYSIS_README.md](FEASIBILITY_ANALYSIS_README.md) – Feasibility scripts and metrics
- [MASTER_README.md](MASTER_README.md) – Repo overview and installation
