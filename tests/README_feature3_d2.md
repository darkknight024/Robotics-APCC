# Feature 3 D2 / Experiment 24 Validation README

This README explains how to run the Feature 3 D2 validation scripts for
Experiment 24. The validation work lives in `tests/experiment24_validation.py`,
but that file is a shared utility module. The normal executable entrypoints are:

- `tests/test_exp24_cross_validation.py`
- `tests/test_speed_profile_vs_rs.py`
- `tests/run_experiment_23_full.py` for selected geometry/full-pipeline modes

Run commands from the repository root:

```bash
cd <repo-root>
```

Use the project Python environment:

```bash
conda activate robotics
python <script>
```

`pytest` may not be installed on the site machine, so each test file also has a
`main()` path and can be run directly as a Python script.

## Required Config/Data

Robot and solver config:

- `config/robots_config.yaml`
- `config/ik_config.yaml`
- `config/batch_feasibility_config.yaml`
- `config/knife_config.yaml`

Important frame convention:

- v1 joint-sweep validation uses `Link_6`, matching the v1 logged TCP.
- v2 orientation-corner validation uses `ee_link`, matching the v2 logged TCP.
- v3 siping toolpaths are raw `T_P_K` and are transformed to `T_B_P` with the
  existing Zund knife pose from `config/knife_config.yaml`.
- v3 RobotStudio result poses are native `T_P_K`; the validation transforms
  them to `T_B_P` for FK/Jacobian comparisons, while native logged speed and
  acceleration are kept in plots as reference.

Experiment 24 data roots:

- v1: `Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v1`
- v2: `Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v2_orientation_varying_corners_24ms`
- v3 results: `Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v3_siping_recordings_at_controlled_spacing`
- v3 toolpaths: `Robot_APCC/Experiments/Experiement_24/Toolpaths/v3_siping_recordings_at_controlled_spacing`

All validation runs write a new timestamped folder under:

```text
Robot_APCC/Experiments/Experiement_24/Results/MM_DD_YY_HH_MM_SS
```

## Run All Experiment 24 Dynamics Validations

```bash
python tests/test_speed_profile_vs_rs.py
```

This runs:

- v1 speed reconstruction validation
- v2 orientation-varying corner velocity validation
- v3 controlled-spacing siping D2 validation

Each section creates its own timestamped result folder.

## v1: Individual Joint Speed/Acceleration Validation

Run acceleration-focused v1 cross-validation:

```bash
python tests/test_exp24_cross_validation.py
```

What it validates:

- Loads all v1 joint-sweep RobotStudio CSVs.
- Reconstructs TCP velocity/acceleration from RobotStudio joint states:
  `v_tcp = J(q) qdot`
  `a_tcp = J(q) qddot + Jdot(q, qdot) qdot`
- Uses the correct v1 TCP frame (`Link_6`).
- Reports J1-J3 translational-axis agreement and J5 plateau/ramp diagnostics.

Typical outputs:

- `summary.txt`
- `trajectory_metrics.csv`
- `trajectory_metrics.json`
- `plots/acceleration_relative_error_by_joint.png`
- `plots/speed_relative_error_by_joint.png`
- `plots/<configuration>_j<joint>_traj_<n>_overlay.png`

## v2: Orientation-Varying Corner Validation

Run via:

```bash
python tests/test_speed_profile_vs_rs.py
```

or call the utility from Python if you only want v2:

```bash
python - <<'PY'
from pathlib import Path
from tests.experiment24_validation import create_exp24_results_dir, evaluate_exp24_v2_orientation_dataset

repo = Path.cwd()
out = create_exp24_results_dir("exp24_v2_orientation_only", repo)
evaluate_exp24_v2_orientation_dataset(out, repo)
print(out)
PY
```

What it validates:

- Processes all 30 v2 orientation-varying corner recordings.
- Uses `ee_link`, matching v2 logged TCP positions.
- Reconstructs `qdot/qddot` from joint positions because v2 lacks joint velocity
  and acceleration columns.
- Compares Jacobian-derived speed/acceleration to RobotStudio signals.

Expected interpretation:

- TCP speed reconstruction should agree well.
- Acceleration is less reliable because it is a second derivative of 24 ms
  joint-position samples.

Outputs:

- `v2_orientation_summary.txt`
- `v2_orientation_metrics.csv`
- `v2_orientation_metrics.json`
- `v2_orientation_plots/*_overlay.png`

## v2 Geometry-Only Bézier Validation

Use this when you only want to check pose/path geometry for the orientation
corner data, not velocity or acceleration:

```bash
python tests/run_experiment_23_full.py \
  --exp24-v2-geometry
```

What it does:

- Reconstructs the 3 programmed waypoints from v2 metadata and known 400 mm
  corner-leg geometry.
- Runs the existing Feature 3 Bézier blend comparison against RobotStudio.
- Confirms pose-only blend geometry.

Useful flags:

- `--dry-run`: list all 30 files without running.
- `--run-dir <name>`: write to a specific folder under Experiment 24 `Results/`.
- `--blend-threshold <mm>`: change pass/fail threshold; default is `1.0`.
- `--lite`: skip heavier per-case plots.

Outputs:

- `generated_waypoints/*.csv`
- `orientation_corners/.../blend_arc_metrics.json`
- `orientation_corners/.../blend_arc_wp*_comparison.png`
- `blend_deviation_report/flagged_toolpaths.txt`
- `exp24_v2_geometry_summary.json`

## v3: Controlled-Spacing Siping D2 Validation

Run all 16 v3 controlled-spacing siping cases:

```bash
python tests/run_experiment_23_full.py \
  --exp24-v3-siping
```

What it validates:

- Loads raw v3 toolpaths from `Toolpaths/v3_siping_recordings_at_controlled_spacing`.
- Transforms raw `T_P_K` toolpaths to `T_B_P` using the Zund knife pose.
- Runs Feature 3 D2 solver replay.
- Loads RobotStudio v3 results and transforms their native `T_P_K` poses to
  `T_B_P` for FK/Jacobian and solver comparisons.
- Computes direct RobotStudio joint-state Jacobian reconstruction:
  `v_tcp = J(q) qdot`
  `a_tcp = J(q) qddot + Jdot(q, qdot) qdot`
- Computes solver-vs-RobotStudio speed, acceleration, orientation speed, pose,
  quaternion, and raw-waypoint tracking metrics.

Useful flags:

- `--run-dir <name>`: write to a specific folder under Experiment 24 `Results/`.
- `--corner-debug`: generate focused 3D corner plots for selected high-curvature waypoints.
- `--max-debug-corners <N>`: number of debug corners per trajectory, default `8`.

Example with corner debug:

```bash
python tests/run_experiment_23_full.py \
  --exp24-v3-siping \
  --corner-debug \
  --max-debug-corners 8
```

To run a single v3 file from Python:

```bash
python - <<'PY'
from pathlib import Path
from tests.experiment24_validation import create_exp24_results_dir, evaluate_exp24_v3_siping_dataset

repo = Path.cwd()
rs = repo / "Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v3_siping_recordings_at_controlled_spacing/v3_siping_recordings_at_controlled_spacing/20260608_mc_plaque_9mm_corner_radius_8mm_spacing_v200.csv"
out = create_exp24_results_dir("exp24_v3_single_case", repo)
evaluate_exp24_v3_siping_dataset(out, repo, csv_paths=[rs], corner_debug=True, max_debug_corners=8)
print(out)
PY
```

Outputs:

- `v3_siping_summary.txt`
- `v3_siping_metrics.csv`
- `v3_siping_metrics.json`
- `v3_siping/<case>/v3_solver_vs_rs_dynamics.png`
- `v3_siping/<case>/v3_solver_vs_rs_full_pose.png`
- `v3_siping/<case>/v3_solver_vs_rs_pose_by_waypoint_index.png`
- `v3_siping/<case>/solver/trajectory_1/trajectory_1_result.csv`
- If `--corner-debug` is enabled:
  `v3_siping/<case>/corner_debug/corner_wp*_blend_geometry_3d.png`

Important v3 metrics:

- `direct_jac_*`: checks Jacobian math using RobotStudio joint states.
- `solver_*`: checks full Feature 3 D2 replay from the raw toolpath.
- `raw_to_solver_rms_error_mm`: distance from transformed raw waypoints to solver path.
- `raw_to_rs_rms_error_mm`: distance from transformed raw waypoints to RobotStudio path.
- `pose_*`: solver pose distance to transformed RobotStudio path.
- `quat_*`: solver-vs-RobotStudio quaternion component delta after sign alignment.

## Experiment 23 V6 Cross-Validation

This is still useful for checking D2 inside the Experiment 23 corner-speed
validation set:

```bash
python tests/test_exp24_cross_validation.py
```

That script runs v1 acceleration validation and then Feature 3 D2 against
Experiment 23 V6. V6 outputs are written under:

```text
Robot_APCC/Experiments/Experiment_23/Results/cross_validation/MM_DD_YY_HH_MM_SS
```

You can also run V6 directly:

```bash
python tests/run_experiment_23_full.py \
  --phase run \
  --v6_only \
  --with_speed_fit \
  --feature3-version d2 \
  --force
```

