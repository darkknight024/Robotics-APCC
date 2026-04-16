# Experiment 23 — Feature 3 D1 Full Pipeline

## Overview

`run_experiment_23_full.py` is the **single entry point** for running, calibrating, and verifying
Feature 3 Deliverable 1 against all Experiment 23 RobotStudio data.

### Developer Story

Our solver predicts the TCP speed profile a robot will execute given a toolpath with per-waypoint
zone configurations. To know if the prediction is correct, we compare against RobotStudio — which
runs the identical motion planner as the real ABB OmniCore controller.

Experiment 23 provides three classes of trajectories:

| Category | Count | Frame | Purpose |
|---|---|---|---|
| **Straight line** | 4 files (v100–v1000) | Robot base | Calibrate `a_tcp`, `T_settle` |
| **Corner** | 30 files (5 angles × 6 zones) | Robot base | Calibrate blend speed model |
| **Siping** | 32 files (4 basenames × 2 speeds × 4 zones) | Zund knife → base | End-to-end validation |

This script automates the full workflow:

1. **Phase 1 — Run**: Execute Feature 3 D1 on every toolpath, generating speed profiles,
   joint trajectories, blend geometry, and comparison-ready CSVs.
2. **Phase 2 — Calibrate**: Extract robot dynamic parameters (`a_tcp`, blend model, joint limits)
   from RobotStudio Signal Analyser data — the system identification step.
3. **Phase 3 — Verify**: Compare every solver output trajectory against its RS counterpart,
   computing RMS speed error, TCP position deviation, and joint utilisation metrics.

## Quick Start

```bash
# From the repository root
cd <repo_root>

# Full pipeline (run + calibrate + verify)
conda run -n robotics python tests/run_experiment_23_full.py

# Preview what will run (no execution)
conda run -n robotics python tests/run_experiment_23_full.py --dry-run

# Calibration only (uses RS data directly, no solver run needed)
conda run -n robotics python tests/run_experiment_23_full.py --phase calibrate

# Verification only (requires a previous run)
conda run -n robotics python tests/run_experiment_23_full.py --phase verify --run-dir Results/12_30_00_04_10_26

# Force re-run (ignore existing results)
conda run -n robotics python tests/run_experiment_23_full.py --force
```

## Output Structure

Each run creates a timestamped directory under `Robot_APCC/Experiments/Experiment_23/Results/`:

```
Results/
  HH_MM_SS_MM_DD_YY/                   ← timestamped run folder
  ├── straight_line/
  │   ├── straight_line_v100_fine_<csv_stem>/
  │   │   └── trajectory_1/
  │   │       ├── trajectory_1_result.csv
  │   │       ├── f3_d1_report.json
  │   │       └── *.png (speed profile, joint plots, blend geometry)
  │   ├── straight_line_v300_fine_<csv_stem>/
  │   ├── straight_line_v500_fine_<csv_stem>/
  │   └── straight_line_v1000_fine_<csv_stem>/
  ├── corner/
  │   └── corner_v500_<zone>_<csv_stem>/
  ├── siping_toolpath/
  │   └── siping_<speed>_<zone>_<csv_stem>/
  ├── calibration/
  │   ├── calibration_report.json       ← all calibrated parameters
  │   ├── a_tcp_calibration.png
  │   ├── blend_model_calibration.png
  │   ├── joint_limits_calibration.png
  │   └── calibration_offsets.png
  └── verification/
      ├── verification_report.json      ← per-trajectory comparison metrics
      ├── speed_rms_error_summary.png
      ├── duration_comparison.png
      └── position_deviation_summary.png
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--phase {all,run,calibrate,verify}` | `all` | Which pipeline phase(s) to execute |
| `--dry-run` | off | List all tasks without executing |
| `--force` | off | Re-run even if results already exist |
| `--run-dir <path>` | (new timestamp) | Use an existing run directory |

## Related Scripts

| Script | Purpose |
|---|---|
| `tests/test_calibration.py` | Validate calibrated parameters against physical bounds |
| `tests/test_tcp_v_a_profile.py` | Per-trajectory speed/acceleration overlay plots |
| `tests/test_segment.py` | TCP path comparison (solver vs RS vs input waypoints) |
| `tests/test_joint_velocity_comparison.py` | Joint velocity and utilisation comparison |
| `tests/calibration_analysis.py` | Standalone calibration analysis (legacy) |

## Architecture

All core computation lives in `core/blend_zone/`:

- `pipeline.py` — Feature 3 D1 pipeline (`run_feature3_d1`)
- `calibration.py` — System identification from RS data
- `verification.py` — Solver vs RS comparison metrics
- `speed_profile.py` — TCP speed profile prediction
- `reporting.py` — CSV export in RS-compatible format
- `blend_geometry.py` — Quadratic Bézier blend arc geometry
- `zone_resolver.py` — ABB zone lookup and overlap reduction
- `path_sampler.py` — Dense SE(3) path with blend arcs

Test scripts (`tests/`) are thin wrappers that call `core/` functions and
render results — they never reimplement the math.

## Calibrated Parameters

| Parameter | Source | Method | Status |
|---|---|---|---|
| `a_tcp` (mm/s²) | Straight-line V1 | P95 \|accel\| during ramp | Calibratable |
| `a_tcp_decel` (mm/s²) | Straight-line V1 | P95 \|accel\| during decel | Calibratable |
| `T_settle` (s) | Multi-fine-stop | Dwell time at v≈0 | NOT calibratable from Exp23 |
| Blend speed model | Corner V2 | Min speed at apex vs √(a·ρ) | Validatable |
| Joint velocity limits | All RS data | Peak dθ/dt from central diff | Estimable |
| Joint acceleration limits | All RS data | Peak d²θ/dt² (noisy) | Marginal |

## Success Criteria (from Proposal)

> TCP speed profile predicted by our solver must match RobotStudio's as closely as practicable.
>
> Quantitative target: RMS speed error < 20 mm/s at 300 mm/s commanded,
> < 50 mm/s at 800 mm/s commanded.
