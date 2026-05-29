# Feature 3 Deliverable 1 — Blend-Zone Speed-Profile Prediction

*Last updated: 2026-04-21.  Runtime validated on ABB IRB 1300-7/1.4.*

## TL;DR

Given a toolpath (waypoints + per-waypoint `zonedata` + target TCP speed),
Feature 3 D1 predicts what the robot will actually do:

1. The **SE(3) path** it traces (straight legs rounded by blend arcs).
2. The **TCP speed profile** along that path (centripetal ceiling +
   acceleration / deceleration ramps + apex dip at tight zones).
3. The **joint angles** that realise the path (via EAIK, least-cost branch).
4. A **side-by-side comparison** of every output against RobotStudio.

All of this is exercised end-to-end in
`tests/run_experiment_23_full.py`.

---

## User Story


What the user sees when they run Experiment 23:

```bash
conda activate robotics
python tests/run_experiment_23_full.py \
    --toolpath "v2/corner/corner_30_deg_*.csv" \
    --speed v20 \
    --force
```

For each toolpath the run produces, under
`Robot_APCC/Experiments/Experiment_23/Results/<timestamp>/`:

```
v2/corner/corner_30_deg_v20_z50/
├── trajectory_1/
│   └── trajectory_1_result.csv          ← solver output (RS-style columns
│                                          without the `rs_` prefix)
└── trajectory_1_result/
    ├── rs_comparison_speed.png          ← solver vs RS TCP-speed overlay
    ├── rs_comparison_tcp_deviation.png  ← X/Y/Z + quat + Euclidean dev
    ├── rs_comparison_path_3d.png        ← 3-D XY-facing overlay
    ├── rs_comparison_joints.png
    ├── rs_comparison_metrics.json       ← every number in one JSON
    ├── blend_arc_wp*_comparison.png     ← per-blend shape overlay
    ├── blend_full_path_comparison.png
    └── blend_arc_metrics.json

verification_summary/
└── summary_table.txt                    ← one-page scorecard per toolpath
```

### Scorecard columns (cheat-sheet)

| Column      | Meaning                                                     | Good (needs sign off from APPC team|
|-------------|-------------------------------------------------------------|-----:|
| `RMS`       | time-averaged \|v_sol − v_rs\| over the active window       | < 5  |
| `MaxErr`    | worst pointwise speed error anywhere (inc. ramps)           | < 20 |
| `MaxCr`     | worst speed error in the **cruise** window (excludes ramps) | < 3  |
| `DurΔ`      | solver_duration − rs_duration (ms)                          | < 100 ms |
| `MeanD/MaxD/P95` | point-to-polyline Euclidean TCP deviation (mm)         | MaxD < 1 |
| `ApexSpd`   | worst speed error in a ±300 ms window around the corner     | < 10 |
| `ApexPos`   | worst point-to-polyline TCP deviation in the same window    | < 1  |

Pass criterion today: `MaxD ≤ 1 mm` and `RMS ≤ 15 mm/s`.

Most-recent 30° v20 run (`04_21_26_13_04_10`) — every zone passes:

| Zone | RMS  | MaxErr | MaxCr | DurΔ  | MeanD | MaxD  | ApexSpd | ApexPos |
|------|-----:|-------:|------:|------:|------:|------:|--------:|--------:|
| z0   | 0.90 | 20.00  | 0.85  | -278  | 0.006 | 0.038 |  8.73   | 0.005   |
| z1   | 0.90 | 20.00  | 1.07  | -376  | 0.010 | 0.107 |  7.53   | 0.010   |
| z5   | 0.94 | 20.00  | 1.74  | -303  | 0.012 | 0.170 |  1.70   | 0.066   |
| z10  | 0.93 | 20.00  | 1.62  | -279  | 0.018 | 0.198 |  2.34   | 0.157   |
| z50  | 0.91 | 20.00  | 1.63  | -270  | 0.062 | 0.817 |  1.63   | 0.783   |

The `20.00` `MaxErr` column is the ramp-shape mismatch (S-curve vs
trapezoid).  `MaxCr` and `ApexPos` are the fair numbers for blend quality.

---

## Developer Story

### Entry points

| Call                                               | What it does                                   |
|----------------------------------------------------|------------------------------------------------|
| `pipeline.run_feature3_d1(...)`                    | Toolpath → `Feature3D1Result`                  |
| `reporting.export_robotstudio_csv(...)`            | Writes RS-compatible solver CSV                |
| `verification.verify_trajectory(sol, rs)`          | Solver ↔ RS metrics + PNGs                     |
| `blend_comparison.compare_blend_arcs(...)`         | Per-blend-arc Fréchet / Hausdorff / deviation  |
| `calibration.run_calibration(...)`                 | Identifies `a_tcp`, `T_settle` etc from RS CSVs    |

### Module map (pipeline order)

```
    load_toolpath_f3 ─────────────────────────────┐
                                                  ▼
    zone_resolver.resolve_zone_list   (M1)  ── zone params
    zone_resolver.apply_overlap_reduction       (effective zones)
                                                  ▼
    blend_geometry.compute_blend_geometries (M2)  cubic Bézier, k=0.78
                                                  ▼
    orientation_zone.populate_orientation_zones(M3) SLERP onsets
                                                  ▼
    path_sampler.sample_blended_path        (M4)  DensePath
                                                  ▼
    EAIK analyze_trajectory                       (q_star, cfx)
                                                  ▼
    speed_profile.predict_speed_profile     (M5)  v_actual(s)
                                                  ▼
    joint_velocity.compute_joint_velocities (M6)  dq/dt from J⁺·ω_e
                                                  ▼
    reporting.generate_f3_report +
    reporting.export_robotstudio_csv              JSON + CSV artefacts
```

### Key calibrated constants — `config/robots_config.yaml`

| Parameter           | Value (IRB 1300-7/1.4) | Source                              |
|---------------------|-----------------------:|-------------------------------------|
| `a_tcp_mm_s2`       | 6128                   | V1 straight-line P95 \|a\|          |
| `a_accel_mm_s2`     | 2800 (≈ 0.46 · a_tcp)  | Distance-based ramp fit             |
| `a_decel_mm_s2`     | 2800                   | Distance-based ramp fit             |
| `rho_min_scale`     | 1.00                   | Direct Bézier fit against RS blends |
| `blend_shape_k`     | 0.78                   | Joint Nelder-Mead fit, all corners  |
| `a_n_blend_mm_s2`   | 3100                   | Observed v_rs_dip=12.47 mm/s at
|                     |                        | z1 apex where ρ_min≈0.05 mm         |
| `T_settle_s`        | 0.2                    | Multi-stop V1 dwell time            |

#### Why `a_n_blend = 3100`?

IRC5 is a jerk-limited S-curve planner — inside a blend arc the
controller enforces a **lower normal (centripetal) acceleration** than
the straight-leg tangential limit `a_tcp`.  Our
`v_blend(s) = √(a_n_blend · ρ(s))` captures that effect directly.
`a_n_blend = 3100 mm/s²` inverts the observed RS dip at z1:

```
ρ_min (z1, 30°, k=0.78) ≈ 0.050 mm     (analytic, blend_geometry.py)
v_rs_dip (30_deg_corner_z1.csv, t≈21412 ms) = 12.47 mm/s
⇒ a_n_blend = v² / ρ = 12.47² / 0.050 ≈ 3110 mm/s²
```

For larger zones ρ_min grows quadratically, so `v_blend > v_cmd` and the
limit simply doesn't bind — exactly what we observe.

### Does our solver reproduce the dip?

Yes.  Spot-check from the latest run:

```
z1 solver   min TCP speed = 12.403 mm/s  @ t = 19737 ms
z1 RS       min TCP speed = 12.470 mm/s  @ t = 21412 ms
                                                (matches within 0.07 mm/s)
```

The amplitude matches.  What we **don't** match perfectly is the **shape
and timing** of the dip: RS uses an S-curve (cubic jerk profile), we use
a trapezoid.  Closing that gap would require a jerk limit `j_tcp` from
RS (site team — differentiate `linear_acceleration_mm_s_2` over ramp
samples or expose the controller's `TuneServo`/`JerkMax`).  See
`FEATURE3_CONTEXT.md` §9 for the complete next-step list.

### Solver result CSV schema (2026-04-21)

The solver writes columns **without** the RS `rs_` prefix so it is
syntactically distinguishable from a recording:

```
time_ms, j1_deg..j6_deg, speed_mm_per_s,
cf1, cf4, cf6, cfx,
x_mm, y_mm, z_mm, qw, qx, qy, qz,
linear_acceleration_mm_s_2, is_at_waypoint
```

`calibration.load_rs_csv` (alias `load_trajectory_csv`) reads both the
prefixed RS layout and the new unprefixed solver layout — the rest of
the pipeline is agnostic.

### Smoke test

```bash
python tests/run_experiment_23_full.py \
    --toolpath v2/corner/corner_30_deg_v20_z50.csv --force
# ≈ 45 s on an 8-core laptop; produces the scorecard shown above.
```

### Code locations

| Concept                    | File                                            |
|----------------------------|-------------------------------------------------|
| Zone-spec parsing          | `zone_resolver.py`                              |
| Bézier geometry + ρ_min    | `blend_geometry.py`                             |
| Dense SE(3) sampling       | `path_sampler.py`                               |
| Speed profile + ceilings   | `speed_profile.py`                              |
| Calibration from RS CSVs   | `calibration.py`                                |
| Solver↔RS verification     | `verification.py`                               |
| Per-arc Fréchet/Hausdorff  | `blend_comparison.py`                           |
| Result CSV writer          | `reporting.py`                                  |
| End-to-end pipeline        | `pipeline.py`  (`run_feature3_d1`)              |
| Batch runner + summary     | `tests/run_experiment_23_full.py`               |
| Narrative + site handover  | `FEATURE3_CONTEXT.md`                           |
