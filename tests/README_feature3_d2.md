# Feature 3 D2 Test README

Feature 3 D2 validates Jacobian-based TCP dynamics using Experiment 24 joint-sweep data and the existing Experiment 23 path/speed comparison harness.

## Experiment 24 Validation

### TCP Acceleration Cross-Validation

```bash
python tests/test_exp24_cross_validation.py
```

What it does:

- Loads Experiment 24 RobotStudio CSVs from `Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v1`.
- Reconstructs TCP speed and acceleration from joint speeds using the geometric Jacobian.
- Uses URDF frame `Link_6`, matching the TCP frame logged in Experiment 24.
- Compares reconstructed TCP acceleration against `linear_acceleration_mm_s_2`.
- Writes a new timestamped result folder under `Robot_APCC/Experiments/Experiement_24/Results/MM_DD_YY_HH_MM_SS`.

Outputs:

- `summary.txt`
- `trajectory_metrics.csv`
- `trajectory_metrics.json`
- `plots/*.png`

### Speed Reconstruction Validation

```bash
python tests/test_speed_profile_vs_rs.py
```

What it does:

- Uses the same Experiment 24 dataset and Jacobian reconstruction path.
- Compares reconstructed TCP speed against RobotStudio `speed_mm_per_s`.
- Produces the same timestamped result folder structure and plots.

## Experiment 23 Full Pipeline

Use this runner when validating D2 inside the full blend-zone solver:

```bash
python tests/run_experiment_23_full.py \
  --toolpath v2/corner \
  --speed v20 \
  --with_speed_fit \
  --feature3-version d2 \
  --force
```

Important CLI flags:

- `--feature3-version d1|d2`: chooses scalar D1 dynamics or D2 Jacobian dynamics.
- `--toolpath <path>`: single CSV, folder, or glob relative to `Experiment_23/Toolpaths_And_Waypoints`.
- `--speed <tag>`: filters by speed tag, e.g. `v20`, `v500`, `v800`.
- `--zone <tag>`: filters by zone tag, e.g. `z1`, `z10`.
- `--with_speed_fit`: enables solver-vs-RobotStudio speed metrics and plots.
- `--lite`: writes a smaller artifact set for faster iteration.
- `--force`: reruns even if prior result CSVs exist.
- `--rs-version v2|v3|v4|v5`: selects a RobotStudio result version, mainly for siping re-recordings.

