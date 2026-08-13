# Feature 3 Deliverable 2 — Module Pipeline Reference

> **Purpose:** Module-by-module reference for Feature 3 Deliverable 2: **geometry**
> (zone blend + orientation schedule) and **velocity profiling** (joint-feasible
> `v*(s)`, `q(s)`, `q̇(s)`).  
> **Last updated:** 2026-08-13  
> **Velocity-profiling entry point:** `tests/test_optimal_velocity_profile.py`  
> **Geometry entry point:** `run_feature3()` in `core/blend_zone/pipeline.py`  
> **Related:** `[OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md](OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md)`
> (stage math), `[PATH_PARAMETERISATION.md](PATH_PARAMETERISATION.md)` (choice of `s`),
> `[plots_readme.md](plots_readme.md)` (every plot/CSV).

---

## Table of contents

1. [Current architecture (read this first)](#0-current-architecture-read-this-first)
2. [What this pipeline answers](#1-what-this-pipeline-answers)
3. [How to run](#2-how-to-run)
4. [Feature 2 vs Feature 3 — two pipelines](#3-feature-2-vs-feature-3--two-pipelines)
5. [End-to-end F3 D2 velocity-profiling pipeline](#4-end-to-end-f3-d2-velocity-profiling-pipeline)
6. [Input layer — toolpath parsing](#5-input-layer--toolpath-parsing)
7. [M1 — Zone Resolver](#6-m1--zone-resolver)
8. [M2 — Blend Geometry](#7-m2--blend-geometry)
9. [M3 — Orientation Zone](#8-m3--orientation-zone)
10. [M4 — Path Sampler](#9-m4--path-sampler)
11. [Feature 2 IK (between M4 and velocity profiling)](#10-feature-2-ik-between-m4-and-velocity-profiling)
12. [Velocity profiling — `core/optimal_velocity](#11-velocity-profiling--coreoptimal_velocity)`
13. [Legacy M5/M6 inside `run_feature3()](#12-legacy-m5m6-inside-run_feature3)`
14. [Pipeline output](#13-pipeline-output)
15. [Configuration map](#14-configuration-map)
16. [Feasibility under zone data and TCP velocity](#15-feasibility-under-zone-data-and-tcp-velocity)
17. [Quick API reference](#16-quick-api-reference)
18. [Known limitations (current software)](#17-known-limitations-current-software)
19. [Module index](#18-module-index)

---

## 0. Current architecture (read this first)

Deliverable 2 used to mean “M5 arc-length reachability + M6 Jacobian `q̇`”
inside `run_feature3()`. That path **still exists** for Feature-3 feasibility
and the older speed-prediction report, but it is **not** the velocity profiler
we run against RobotStudio today.


| Layer                       | What it does                                                                                       | Code                                                                     | Invoked by                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Geometry (M1–M4)**        | Zones, Bézier fly-bys, ABB C³ orientation schedule, plate-frame dense path, IK                     | `core/blend_zone/`                                                       | `run_feature3()` and, via `load_joint_path_from_toolpath()`, the velocity profiler |
| **Velocity profiling (D2)** | Uniform resample → LSQ quintic `q(s)` → MVC ceilings → Heun TOPP → `v*(s)`, `q̇(s)`, `q̈(s)` vs RS | `core/optimal_velocity/`                                                 | `tests/test_optimal_velocity_profile.py`                                           |
| **Legacy M5/M6**            | Task-space ceiling stack + Jacobian twist inversion                                                | `core/blend_zone/speed_profile.py`, `core/checks/task_space_velocity.py` | `run_feature3()` only — **not** the RS-benchmarked profiler                        |


The rest of this document keeps M1–M4 as the geometry reference (they are
current). Sections 11–12 split the two speed engines. **Known limitations in
§17 are for the current velocity profiler**, not the legacy M5 path.

---

## 1. What this pipeline answers

Given a **toolpath** (waypoint poses in `T_P_K`, zone data, commanded cut
speeds), Feature 3 D2 estimates:


| Output                           | Meaning                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Blended TCP path**             | SE(3) trajectory the robot executes (straights + cubic Bézier fly-bys), built in the programmed **plate** frame then mapped to base. |
| **Joint positions** `q(s)`       | IK on the dense blended path, then an LSQ quintic spline (the path TOPP differentiates).                                             |
| **Cut speed** `v*(s)`            | Knife-relative linear speed after joint limits, the command governor, and (optionally) the RS zone cap.                              |
| **Joint rates** `q̇(s)`, `q̈(s)` | Chain rule: `q̇ = q' ṡ`, `q̈ = q' s̈ + q'' ṡ²`.                                                                                      |
| **Duration**                     | `t = ∫ ds/ṡ` on the path parameter (robot-base arc by default).                                                                      |


**Feasibility** on a velocity-profile run:

- Joint limits: `max |q̇|/q̇_max ≤ 1` and `max |q̈|/q̈_max ≤ 1` (`limits_check` in the mode summary).
- Command tracking: `v`* vs toolpath col-8 on unmasked samples (`command_tracking` in the summary).
- RS bench: `|v* − v_RS|` on samples **not** in transient / v_cmd ramp / v_cap lookup (`run_feasibility_summary.txt`).

A path can be geometrically reachable and still fail command tracking or RS
bench. That is expected: joint acceleration, the governor, and unmodeled IRC5
corner derate all pull `v`* below `v_cmd`.

---

## 2. How to run

From the repo root. Plate-frame blend and the ABB C³ orientation schedule are
**on by default** — do not pass `--no-plate-frame-blend` unless A/B-testing.

**Full v7 cropped set, commanded mode only:**

```bash
cd /home/koushik/Nike/Robotics-APCC
python tests/test_optimal_velocity_profile.py \
  --dataset v7_cropped \
  --out Robot_APCC/Experiments/Experiement_24/Results/v7_cropped_latest
```

**Same, all three modes (commanded / constant / optimal):**

```bash
python tests/test_optimal_velocity_profile.py \
  --dataset v7_cropped \
  --time-optimal \
  --out Robot_APCC/Experiments/Experiement_24/Results/v7_cropped_latest_to
```

**Single toolpath:**

```bash
python tests/test_optimal_velocity_profile.py \
  --toolpath Robot_APCC/Experiments/Experiement_24/Toolpaths/v7_sidewall_wrapped_toolpath/cropped_toolpath_by_segment/sidewall_wrapped_toolpath_cropped_traj_1.csv \
  --rs-csv "Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/cropped_toolpath/sidewall_wrapped_toolpath_cropped_traj_1.csv" \
  --out Robot_APCC/Experiments/Experiement_24/Results/traj_1_latest
```

Leave `--ds-mm`, `--uniform-resample-mm`, `--resid-tol-deg`, and
`--ceiling-smooth-mm` at defaults (`0.25`, `0.25`, `0.05°`, `2.5 mm`) unless
you are running a sensitivity. `--se3-arc-length` is experimental and **off**.

**Where to look after a run**


| Artifact                                        | What it is                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| `<toolpath>/commanded/tcp_velocity_profile.png` | Cut speed, `‖ω‖`, accel. **x = robot-base arc**, **y = tool-frame** |
| `<toolpath>/commanded/summary.txt`              | Command tracking, RS `                                              |
| `<run>/run_feasibility_summary.txt`             | Per-toolpath waypoint counts, RS bench, traversal times             |
| `<run>/batch_fk_check.csv`                      | FK(spline) vs blended-arc residual                                  |
| `<toolpath>/M_orientation_phasing/`             | `dθ/ds_tool`, gain, cancellation                                    |
| `<toolpath>/I_spline_fk_check/`                 | Spline vs Feature-3 pose residual                                   |


`batch_summary.txt` is **not** written (it duplicated the two files above).

**Geometry-only / legacy M5** (not the RS-benchmarked profiler):

```bash
python feasibility_analysis.py --feature3 --toolpath <csv>
# or batch: feature3_d1.enabled: true in config/batch_feasibility_config.yaml
```

---

## 3. Feature 2 vs Feature 3 — two pipelines

```
┌──────────────────────────────── F2 FEASIBILITY (with TOPP-RA) ────────────────────────────────┐
│ toolpath CSV → load_toolpath_trajectories_ext() → [optional densify]                        │
│   → IK on waypoints → TOPP-RA parameterize_trajectory() → q(t), q̇(t)                      │
│   → compute_task_space_velocity(TOPP q̇) → check_speed_limits(vs CSV)                         │
│   → check_c1_continuity(TOPP q̇, q̈)                                                         │
│ Output: reachability, C0/C1, TOPP duration, TCP speed violations                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────── F3 D2 velocity profiling (no TOPP-RA) ──────────────────────┐
│ toolpath CSV → load_joint_path_from_toolpath() → M1–M4 plate-frame blend + ABB C³          │
│   → IK on dense blended path → LSQ quintic q(s) → MVC + Heun TOPP                           │
│ Output: v*(s) in tool frame, q(s)/q̇(s)/q̈(s) in memory, RS-matching metrics               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

`run_feature3()` still has a **legacy** M5/M6 speed path (task-space ceilings +
Jacobian `q̇`). That is not the RS-benchmarked profiler; see §12.


| Question                                  | Answered by F2 + TOPP-RA          | Answered by F3 D2 profiler                                              |
| ----------------------------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| Can the robot reach all programmed poses? | Yes (Phase 1 IK)                  | Yes (IK on blended dense path)                                          |
| Minimum time under joint q̇/q̈ limits?    | Yes (`ToppraResult.duration_s`)   | Yes (`optimal` mode; `commanded` is additionally `v ≤ v_cmd`)           |
| Is TCP speed within CSV process limit?    | Post-hoc check on TOPP trajectory | Built-in: `v* ≤ v_cmd` in commanded mode (governor may overshoot ≤ 15%) |
| What path does the robot actually follow? | Straight/densified polyline       | Zone-respecting Bézier blends in `T_P_K`                                |
| What speed will IRC5/RobotStudio achieve? | Not modeled                       | Joint MVC + governor vs RS recordings; **no** IRC5 corner derate        |
| Joint velocities at planned speed?        | From TOPP `qdot_t`                | Chain rule `q̇ = q' ṡ` (not Jacobian twist inversion)                   |


---

## 4. End-to-end F3 D2 velocity-profiling pipeline

```
toolpath CSV  (T_P_K poses, v_cmd, zone cols)
        │
        ▼
load_joint_path_from_toolpath()          utils/optimal_velocity/toolpath_load.py
        │  knife → base, then run_feature3()
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  M1 zone_resolver → M2 blend_geometry → M3 orientation_zone          │
│  M4 sample_blended_path_plate_frame()  (ABB C³ schedule, default)    │
│  IK (EAIK) on dense poses → q_raw                                    │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
run_diagnostics()                        core/optimal_velocity/pipeline.py
  0  validate / de-dup
  1b uniform resample (0.25 mm position arc)
  1  LSQ quintic q(s)
  2  MVC: v_vel, v_accel, v_secant  (+ min-preserving smooth)
  3  commanded cap  ṡ ≤ v_cmd / g   (governor)
  4  Heun TOPP → ṡ*(s), t(s)
  5  realize q̇ = q' ṡ, q̈ = q' s̈ + q'' ṡ², v_tool = g · ṡ
        │
        ▼
plots + summaries
  commanded/tcp_velocity_profile.png     (y tool-frame, x base arc)
  commanded/summary.txt                  command tracking + RS |err|
  run_feasibility_summary.txt            batch rollup
  batch_fk_check.csv                     I_spline_fk_check
  M_orientation_phasing/                 geometry gate
```

`--time-optimal` repeats stages 2–5 for `constant` and `optimal` (no `v_cmd` cap)
into sibling folders.

**D2 activation of geometry:** `plate_frame_blend: true` (default),
`ori_schedule_mode: "abb"` (default). Velocity-profile runs set
`cfg.feature3_d1.ds_mm` from `--ds-mm` (default **0.25 mm**, not the YAML 1.0).

The **legacy** `run_feature3()` M5/M6 chain (task-space ceilings → Jacobian
`q̇`) is §12. It is not what `test_optimal_velocity_profile.py` runs.

---

## 5. Input layer — toolpath parsing

**Module:** `utils/csv_loader_toolpath.py`  
**Function:** `load_toolpath_f3()` / `prepare_toolpath_load_result_for_feature3()`

### Role

Parse the toolpath CSV into per-trajectory waypoint arrays with zone specs and commanded speeds. Optionally apply knife→base frame transform before the pipeline runs.

### Input


| Source               | Fields                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Preset zone CSV**  | `x,y,z,qw,qx,qy,qz, speed_mm_s, zone_number [, fine]`                                                          |
| **Custom zone CSV**  | `x,y,z,qw,qx,qy,qz, pzone_tcp, pzone_ori, zone_ori [, v_cmd]`                                                  |
| **Header-based CSV** | `rs_x_mm, rs_y_mm, rs_z_mm, rs_qw…, speed_mm_s, zone, fine`                                                    |
| **Exp24 / v7**       | headerless: pose + col-8 `v_cmd` + cols 9–14 `pzone_tcp, pzone_ori, pzone_eax, zone_ori, zone_leax, zone_reax` |


`pzone_eax` / `zone_leax` / `zone_reax` are stored and unused (no external axes). See §17.11.

### Output — `ToolpathLoadResultF3`


| Field        | Type                   | Description                                                      |
| ------------ | ---------------------- | ---------------------------------------------------------------- |
| `waypoints`  | `List[(N, 7) ndarray]` | `[x_m, y_m, z_m, qw, qx, qy, qz]` per trajectory                 |
| `v_cmd`      | `List[(N,) ndarray]`   | Commanded TCP speed per waypoint (mm/s)                          |
| `zone_specs` | `List[List]`           | Zone name strings or `(pzone_tcp, pzone_ori, zone_ori)` triplets |
| `metadata`   | `dict`                 | `zone_extracted`, `speed_extracted`, source path                 |


### Config / constants


| Config key                       | File                            | Default  | Used for                            |
| -------------------------------- | ------------------------------- | -------- | ----------------------------------- |
| `feature3_d1.custom_zone`        | `batch_feasibility_config.yaml` | `false`  | Preset vs custom zone column layout |
| `feature3_d1.default_zone`       | `batch_feasibility_config.yaml` | `"fine"` | Fallback when zone column missing   |
| `feature3_d1.default_v_cmd_mm_s` | `batch_feasibility_config.yaml` | `300.0`  | Fallback when speed column missing  |


---

## 6. M1 — Zone Resolver

**File:** `core/blend_zone/zone_resolver.py`  
**Label:** M1

### Role and responsibilities

1. Parse any zone specification (preset name `z0`…`z100`, `fine`, or custom triplet) into structured `ZoneParams`.
2. Apply **ABB overlap reduction** so adjacent fly-by zones never exceed half the inter-waypoint segment length.
3. Single source of truth for zone radii — downstream modules never parse raw zone strings.

### Functions


| Function                                    | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| `resolve_zone_spec(spec)`                   | One waypoint → `ZoneParams`          |
| `resolve_zone_list(specs)`                  | Waypoint list → `List[ZoneParams]`   |
| `apply_overlap_reduction(zones, waypoints)` | Mutate effective radii per ABB rules |


### Input interface


| Parameter     | Type             | Source                                                            |
| ------------- | ---------------- | ----------------------------------------------------------------- |
| `zone_specs`  | `List[str        | Tuple[float,float,float]]`                                        |
| `waypoints_m` | `(N, 7) ndarray` | Programmed poses (m) — used for overlap reduction segment lengths |


### Output interface — `ZoneParams`


| Field              | Unit | Description                                              |
| ------------------ | ---- | -------------------------------------------------------- |
| `finep`            | bool | True = stop point (no fly-by blend)                      |
| `pzone_tcp_mm`     | mm   | Programmed TCP position zone radius                      |
| `pzone_ori_mm`     | mm   | Programmed orientation zone (TCP travel)                 |
| `zone_ori_deg`     | deg  | Programmed orientation zone (angle)                      |
| `eff_pzone_tcp_mm` | mm   | **Effective** TCP radius after overlap reduction         |
| `eff_pzone_ori_mm` | mm   | **Effective** orientation radius after overlap reduction |
| `source`           | str  | `'fine'`, `'z10'`, `'custom'`, …                         |


### Config / constants


| Name               | Location           | Value      | Notes                                                    |
| ------------------ | ------------------ | ---------- | -------------------------------------------------------- |
| `PREDEFINED_ZONES` | `zone_resolver.py` | ABB table  | `fine`, `z0`…`z200` → `(pzone_tcp, pzone_ori, zone_ori)` |
| `ZONE_NUMBER_MAP`  | `zone_resolver.py` | int → name | e.g. `5 → "z5"`                                          |


No YAML config — table is hardcoded from ABB RAPID manual.

---

## 7. M2 — Blend Geometry

**File:** `core/blend_zone/blend_geometry.py`  
**Label:** M2

### Role and responsibilities

1. Convert effective TCP zone radii into **physical fly-by geometry** at each non-fine interior waypoint.
2. Build symmetric **cubic Bézier** blend arcs: entry/exit points, inner controls, arc length, minimum radius of curvature `ρ_min`.
3. Pure geometry — no speed, time, or IK.

### Functions


| Function                                              | Purpose                            |
| ----------------------------------------------------- | ---------------------------------- |
| `compute_blend_geometry(...)`                         | Single corner → `BlendArcGeometry` |
| `compute_blend_geometries(waypoints, zones, shape_k)` | Full path → `List[BlendArcGeometry |


### Input interface


| Parameter     | Type               | Source                                                            |
| ------------- | ------------------ | ----------------------------------------------------------------- |
| `waypoints_m` | `(N, 7) ndarray`   | Programmed poses                                                  |
| `zone_params` | `List[ZoneParams]` | M1 output (uses `eff_pzone_tcp_mm`)                               |
| `shape_k`     | `float`            | `robots_config.yaml` → `blend_shape_k` or `DEFAULT_BLEND_SHAPE_K` |


### Output interface — `BlendArcGeometry`


| Field                                                 | Unit      | Description                                   |
| ----------------------------------------------------- | --------- | --------------------------------------------- |
| `waypoint_idx`                                        | int       | Programmed corner index                       |
| `entry_point_mm`, `exit_point_mm`                     | (3,) mm   | Blend arc endpoints on adjacent segments      |
| `inner_p1_mm`, `inner_p2_mm`                          | (3,) mm   | Cubic Bézier control points                   |
| `shape_k`                                             | —         | Shape parameter used                          |
| `corner_angle_rad`                                    | rad       | Deflection angle θ between segment directions |
| `r_tcp_eff_mm`                                        | mm        | Effective zone radius used                    |
| `arc_length_mm`                                       | mm        | Total blend arc length                        |
| `rho_min_mm`                                          | mm        | Minimum radius of curvature (apex)            |
| `centripetal_normal`                                  | (3,) unit | Normal at apex (populated for M5)             |
| `r_ori_eff_mm`, `ori_onset_in_mm`, `ori_onset_out_mm` | mm        | Filled by M3                                  |


`None` at fine points and path endpoints (no fly-by arc).

### Config / constants


| Name                    | Location                             | Default | Role                          |
| ----------------------- | ------------------------------------ | ------- | ----------------------------- |
| `DEFAULT_BLEND_SHAPE_K` | `blend_geometry.py`                  | `0.78`  | Cubic inner-control placement |
| `blend_shape_k`         | `robots_config.yaml` → `calibration` | `0.78`  | Overrides default when `> 0`  |


**Bézier layout:**

```
P0 = entry,  P3 = exit
P1 = P0 + k·r·dir_in,  P2 = P3 − k·r·dir_out
```

---

## 8. M3 — Orientation Zone

**File:** `core/blend_zone/orientation_zone.py`  
**Label:** M3

### Role and responsibilities

1. Compute where **orientation SLERP** begins on each segment (ABB formula).
2. Write effective orientation onset distances back into `BlendArcGeometry` (documented exception to M2 immutability).
3. Ensures M4 samples orientation transitions at the correct arc-length offset — not at blend arc entry.

### Functions


| Function                                                    | Purpose                                   |
| ----------------------------------------------------------- | ----------------------------------------- |
| `compute_effective_orientation_zone(waypoints, idx, zone)`  | One waypoint → `EffectiveOrientationZone` |
| `populate_orientation_zones(blend_geoms, zones, waypoints)` | Batch update of all `BlendArcGeometry`    |


### Input interface


| Parameter     | Type                   | Source                                               |
| ------------- | ---------------------- | ---------------------------------------------------- |
| `waypoints_m` | `(N, 7) ndarray`       | Poses + quaternions                                  |
| `zone_params` | `List[ZoneParams]`     | M1 (`pzone_ori`, `zone_ori_deg`, `eff_pzone_ori_mm`) |
| `blend_geoms` | `List[BlendArcGeometry | None]`                                               |


### Output interface — `EffectiveOrientationZone`


| Field                                       | Description                                            |
| ------------------------------------------- | ------------------------------------------------------ |
| `r_ori_eff_mm`                              | Effective orientation onset distance from waypoint     |
| `governed_by`                               | `'pzone_ori'` or `'zone_ori'` — which constraint bound |
| `delta_theta_in_rad`, `delta_theta_out_rad` | Orientation change on adjacent segments                |
| `segment_len_in_mm`, `segment_len_out_mm`   | Adjacent segment lengths                               |


**Side effect:** `blend_geoms[i].r_ori_eff_mm`, `ori_onset_in_mm`, `ori_onset_out_mm` updated.

### Config / constants


| Name                    | Source                | Notes                                                                     |
| ----------------------- | --------------------- | ------------------------------------------------------------------------- |
| ABB orientation formula | `orientation_zone.py` | `r_ori_eff = min(pzone_ori, (zone_ori_rad/Δθ)·L)`, floored at `pzone_tcp` |
| Zone fields             | M1 `ZoneParams`       | No separate YAML                                                          |


---

## 9. M4 — Path Sampler

**File:** `core/blend_zone/path_sampler.py`  
**Label:** M4

### Role and responsibilities

1. Assemble **dense SE(3) path** along straights + Bézier blend arcs.
2. **Default:** `sample_blended_path_plate_frame()` — build in programmed
  `T_P_K`, phase the ABB C³ orientation schedule on the **cut arc**, map back
   to base, thin to uniform base-arc spacing by **selecting** fine samples
   (never interpolating — that would flatten C³).
3. Carry commanded speed, blend flags, and Bézier parameter `t`.

Opt out of plate-frame construction with `--no-plate-frame-blend` (A/B only).
Opt out of the ABB schedule with `ori_schedule_mode="legacy"` (hold–SLERP–hold).

### Functions


| Function                               | Purpose                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------- |
| `sample_blended_path_plate_frame(...)` | **Default** dense path: plate-frame blend + ABB C³ schedule + map to base |
| `sample_blended_path(...)`             | Legacy base-frame assembly (`--no-plate-frame-blend`)                     |


### Input interface


| Parameter      | Type                   | Source            |
| -------------- | ---------------------- | ----------------- |
| `waypoints_m`  | `(N, 7) ndarray`       | Programmed poses  |
| `zones`        | `List[ZoneParams]`     | M1                |
| `blend_geoms`  | `List[BlendArcGeometry | None]`            |
| `v_cmd_per_wp` | `(N,) ndarray`         | mm/s per waypoint |
| `ds_mm`        | `float`                | Sample spacing    |


### Output interface — `DensePath`


| Field          | Type         | Unit     | Description                                  |
| -------------- | ------------ | -------- | -------------------------------------------- |
| `poses`        | `(M, 7)`     | m + quat | Dense SE(3) samples                          |
| `arc_lengths`  | `(M,)`       | mm       | Cumulative arc-length from start             |
| `is_blend_arc` | `(M,) bool`  | —        | True on blend arc samples                    |
| `segment_ids`  | `(M,) int`   | —        | Programmed segment index                     |
| `v_cmd_at_s`   | `(M,) float` | mm/s     | Commanded speed at each sample               |
| `blend_t`      | `(M,) float` | —        | Bézier parameter t ∈ [0,1]; NaN on straights |
| `blend_wp_idx` | `(M,) int`   | —        | Owning blend waypoint; −1 on straights       |


### Config / constants


| Config key                      | File                                                 | Default                     | Role                                                                      |
| ------------------------------- | ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------- |
| `feature3_d1.ds_mm`             | `batch_feasibility_config.yaml` / `Feature3D1Config` | YAML `5.0`, dataclass `1.0` | Profiler CLI overrides with `--ds-mm` (**0.25**)                          |
| `feature3_d1.plate_frame_blend` | same                                                 | `true`                      | Build in `T_P_K`; `--no-plate-frame-blend` turns off                      |
| `feature3_d1.ori_schedule_mode` | same                                                 | `"abb"`                     | ABB C³ dual-schedule; `"legacy"` = hold–SLERP–hold                        |
| `_PLATE_OVERSAMPLE`             | `path_sampler.py`                                    | `8`                         | Fine grid in plate frame; thinned by **selecting** samples onto `--ds-mm` |


Orientation-zone half-widths are distances in **path-parameter millimetres**
(the same `phase` the C³ schedule lives in), not a fraction of the programmed
chord. Each waypoint’s zone is also stretched to cover its own blend-arc
samples so the schedule never freezes on the corner quaternion. See
`[OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md](OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md)`
§1c.

---

## 10. Feature 2 IK (between M4 and velocity profiling)

**File:** `core/feasibility/analyzer.py` — `FeasibilityAnalyzer`  
**Label:** F2 (not M-numbered, but required for D2)

### Role and responsibilities

1. Solve inverse kinematics at **every** `DensePath` pose.
2. Select a continuous joint branch (least-cost / multi-solution weights).
3. Gate geometric feasibility — unreachable poses abort before velocity profiling.

The velocity profiler then **re-fits** `q(s)` as an LSQ quintic on that `q_raw`
(`core/optimal_velocity/pipeline.py`). The spline, not the raw IK polyline, is
what TOPP differentiates.

### Input interface


| Parameter     | Type             | Source                        |
| ------------- | ---------------- | ----------------------------- |
| `positions`   | `(M, 3) ndarray` | `dense_path.poses[:, :3]` (m) |
| `quaternions` | `(M, 4) ndarray` | `dense_path.poses[:, 3:7]`    |


### Output interface


| Field                               | Type             | Description                                      |
| ----------------------------------- | ---------------- | ------------------------------------------------ |
| `joint_angles_rad`                  | `(M, 6) ndarray` | `q_raw` — joint positions along the blended path |
| `feasibility_flags.reachability_ok` | `bool`           | False → pipeline abort                           |
| `per_waypoint_results`              | list             | Per-sample IK diagnostics                        |


### Config / constants


| Config key                       | File                            | Role                                     |
| -------------------------------- | ------------------------------- | ---------------------------------------- |
| `solver`                         | `batch_feasibility_config.yaml` | `"pin"` or `"eaik"` (profiler uses EAIK) |
| `singularity.threshold`          | same                            | Singularity detection                    |
| `eaik_multi_solution.weights`    | same                            | Branch selection                         |
| `max_ik_failures_per_trajectory` | same                            | Failure tolerance                        |
| `ik_config.yaml`                 | `config/`                       | EE frame name, IK parameters             |


---

## 11. Velocity profiling — `core/optimal_velocity`

**Entry:** `tests/test_optimal_velocity_profile.py`  
**Orchestrator:** `utils/optimal_velocity/runner.py`  
**Solver:** `core/optimal_velocity/pipeline.py` → `run_diagnostics()`  
**Stage math:** `[OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md](OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md)`

This is the **current** Feature 3 D2 speed engine. It is **not** M5/M6.

### Role and responsibilities

1. Take Feature-3 `q_raw`, poses, plate XYZ, and `v_cmd(s)`.
2. Uniform-resample onto a 0.25 mm **position** arc, fit LSQ quintics `q(s)`.
3. Build MVC ceilings (`v_vel`, `v_accel`, `v_secant`) from joint limits.
4. Cap commanded mode at `ṡ ≤ v_cmd / g` with the empirical speed governor.
5. Integrate Heun TOPP → `ṡ*(s)`, `t(s)`, then realize `q̇`, `q̈`, `v_tool = g·ṡ`.

### Stages (inside `run_diagnostics`)


| Stage | What                                                | Defaults                                |
| ----- | --------------------------------------------------- | --------------------------------------- |
| 0     | Validate / de-dup coincident samples                | —                                       |
| 1b    | Uniform resample on **position** arc                | `--uniform-resample-mm 0.25`            |
| 1     | LSQ quintic `q(s)` with residual bisection          | `--resid-tol-deg 0.05`                  |
| 2     | MVC: `v_vel`, `v_accel`, `v_secant`                 | secant window 2.5 mm                    |
| 2b    | Min-preserving ceiling smooth                       | `--ceiling-smooth-mm 2.5`               |
| 3     | Commanded cap `ṡ ≤ v_cmd/g` + governor              | `--cmd-accel-max 8000`, 1.15× overshoot |
| 4     | Heun TOPP                                           | `--path-jerk-max` default **0** (off)   |
| 5     | Realize `q̇ = q'ṡ`, `q̈ = q's̈ + q''ṡ²`, `v* = g·ṡ` | reporting frame = **tool**              |


`--time-optimal` repeats stages 2–5 for `constant` and `optimal` (no `v_cmd`
cap) into sibling folders.

### CLI flags that matter


| Flag                     | Default            | Notes                                                 |
| ------------------------ | ------------------ | ----------------------------------------------------- |
| `--ds-mm`                | `0.25`             | Overrides YAML `feature3_d1.ds_mm` (1.0) for this run |
| `--uniform-resample-mm`  | `0.25`             | `0` keeps Feature-3 sampling                          |
| `--ceiling-smooth-mm`    | `2.5`              | Never raises a ceiling; widens notches                |
| `--cmd-accel-max`        | `8000`             | Empirical governor, not robot physics                 |
| `--path-jerk-max`        | `0`                | Path-jerk slew **off**                                |
| `--cap-mode`             | `pointwise_spline` | `ṡ_target = v_cmd / g_spline`                         |
| `--se3-arc-length`       | **off**            | Experimental weighted SE(3) `s`                       |
| `--no-plate-frame-blend` | **off**            | A/B only — restores `g` wobble                        |
| `--no_vcap`              | **off**            | Disables RS spacing×zone lookup cap                   |


### Output — `ProfileResult` (`core/optimal_velocity/types.py`)

Computed **in memory** and plotted. There is **no** controller-ready export
of `q(t)` / `q̇(t)` (see §17.4).


| Field             | Unit          | Description                                         |
| ----------------- | ------------- | --------------------------------------------------- |
| `s_eval`          | mm            | Path parameter (robot-base position arc by default) |
| `q`               | rad           | Spline joint positions                              |
| `q_dot`, `q_ddot` | rad/s, rad/s² | Realized rates                                      |
| `t`               | s             | Time axis from `∫ ds/ṡ`                             |
| `v_star`          | mm/s          | Cut speed in the **tool** frame                     |
| `s_dot_path`      | mm/s          | Path speed `ṡ` (drives `q̇` and `ω`)                |
| `plate_gain`      | —             | `g = ds_tool / ds_base`                             |
| `metrics`         | dict          | Limit utilisation, duration, command tracking       |


### Joint limits used by the profiler

Loaded from `config/robots_config.yaml` → IRB 1300-7/1.4 `calibration.joint_dynamics`
(`utils/optimal_velocity/toolpath_load.py`), **not** from
`JointLimits.exp24_neutral()` (that helper still has the old v8 J4=144 deg/s²
numbers and is not the production path).


| Joint | q̇_max (deg/s) | q̈_accel (deg/s²) | q̈_decel (deg/s²) |
| ----- | -------------- | ----------------- | ----------------- |
| J1    | 280.0          | 1586              | 1283              |
| J2    | 180.0          | 4268              | 4343              |
| J3    | 250.0          | 5311              | 5305              |
| J4    | 500.0          | 67164             | 57539             |
| J5    | 415.8          | 23221             | 23449             |
| J6    | 720.0          | 60116             | 56318             |


`q̇_max` is Exp24 v1 (trusted). `q̈_`* are **v9 RS p99 observed peaks**, not
ABB catalog values. The bound TOPP uses is the symmetric
`min(accel, decel)` (`JointLimits.q_ddot_max`). See §17.3.

---

## 12. Legacy M5/M6 inside `run_feature3()`

Used only by `run_feature3()` / `feasibility_analysis.py --feature3`.
**Not** invoked by `test_optimal_velocity_profile.py`.

### M5 — `core/blend_zone/speed_profile.py`

Task-space ceiling stack (blend centripetal / corner-dip, Jacobian `v_joint`,
scalar `ω_max`) then forward/backward squared-speed reachability on the
blended arc. Output: `SpeedProfileResult.v_actual` [mm/s].

Optional `v_topp_ceiling` exists on the function signature but is **never
passed** by `pipeline.py`. TOPP-RA (`core/topp_check.py`) remains a Feature-2
module.

### M6 — `core/checks/task_space_velocity.py`

On the F3 path: `ω_e` from dense quaternions × `v_actual`, then
`q̇ = J⁻¹ · twist`. On the F2 path (not F3): `V = J @ q̇` from TOPP-RA
`qdot_t`. Two different code paths in one file.

### When to use which speed engine


| Use case                                                                  | Pipeline                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| RS-matching cut speed + joint-feasible `v*(s)` on a **zone-blended** path | D2 profiler (`test_optimal_velocity_profile.py`) |
| Feature-3 feasibility report / capability probe (`v_max_cruise`)          | `run_feature3()` M5                              |
| Minimum time under joint q̇/q̈ on a **waypoint polyline**                 | F2 + `parameterize_trajectory()`                 |
| Estimate `q̇` at M5 TCP speed                                             | F3 M6 (Jacobian inversion)                       |


---

## 13. Pipeline output

### Velocity profiler (current D2)

Artifacts under `--out` (see `[plots_readme.md](plots_readme.md)` for every PNG):


| File                                            | Content                                                                           |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| `<toolpath>/commanded/tcp_velocity_profile.png` | Key overlay. **x = robot-base arc**, **y = tool-frame** (title lists both totals) |
| `<toolpath>/commanded/summary.txt`              | Command tracking, RS `                                                            |
| `<toolpath>/commanded/D_optimal_profile/`       | `q(s)`, `q̇(t)`, `q̈(t)` plots — **not** a playback CSV                           |
| `<run>/run_feasibility_summary.txt`             | Per-toolpath waypoint counts, RS bench, traversal times                           |
| `<run>/batch_fk_check.csv`                      | `I_spline_fk_check` rollup                                                        |
| `<toolpath>/M_orientation_phasing/`             | Geometry gate                                                                     |
| `<toolpath>/I_spline_fk_check/`                 | FK(spline) vs blended-arc residual                                                |


`batch_summary.txt` is **not** written.

`ProfileResult.q`, `.q_dot`, `.q_ddot`, `.t` exist in memory after
`run_diagnostics()`. Feature 2 has `export_final_trajectory_csv` /
`export_dense_ik_trajectory_csv`; the D2 profiler does **not** call them.
No RAPID, ROS bag, or timestamped joint CSV for controller playback.

### Legacy `run_feature3()` — `Feature3D1Result`


| Field                   | Description                            |
| ----------------------- | -------------------------------------- |
| `feasible`              | IK reachability gate                   |
| `q_star`                | Joint positions along the blended path |
| `speed_profile`         | M5 — includes `v_actual`               |
| `joint_velocity_result` | M6 — includes `q_dot`                  |
| `dense_path`            | M4 blended SE(3) path                  |


Optional artefacts: `trajectory_N_result.csv` (RS-format replay from M5/M6),
`f3_d1_report.json`, diagnostic PNGs. That CSV is the **legacy** speed
engine, not the Heun-TOPP profile.

---

## 14. Configuration map

### Velocity profiler (D2)


| Module         | Primary config                                    | Keys / flags                                                                                               |
| -------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Geometry M1–M4 | `batch_feasibility_config.yaml`                   | `custom_zone`, `default_zone`, `ds_mm` (overridden by `--ds-mm`), `plate_frame_blend`, `ori_schedule_mode` |
| M2 shape       | `robots_config.yaml`                              | `calibration.blend_shape_k`                                                                                |
| IK             | `batch_feasibility_config.yaml`, `ik_config.yaml` | `solver`, `eaik_multi_solution`                                                                            |
| Joint limits   | `robots_config.yaml`                              | `calibration.joint_dynamics` (IRB 1300-7/1.4)                                                              |
| Knife          | `config/knife_config.yaml`                        | `zundV1` → `T_B_K` (required for plate-frame blend)                                                        |
| Profiler knobs | CLI                                               | `--uniform-resample-mm`, `--ceiling-smooth-mm`, `--cmd-accel-max`, `--se3-arc-length`, …                   |


### Legacy `run_feature3()` M5/M6


| Module | Keys                                                                                                                                                        |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M5     | `SpeedCalibration` in `robots_config.yaml` + `feature3_d1` toggles (`enable_joint_velocity_ceiling`, `k_corner_dip`, `T_settle_s`, `use_jacobian_dynamics`) |
| M6     | `joint_dynamics.q_dot_max` via `final_vel_lims`                                                                                                             |


`feature3_d1.enabled: true` makes `feasibility_analysis_batch.py` run **only**
F3 (`run_feature3`) and skip the F2/TOPP pipeline for that job.

### Feature 2 + TOPP-RA (separate — not invoked by the D2 profiler)


| Phase           | Module                        | Config                                                                               |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------ |
| Phase 1 IK      | `FeasibilityAnalyzer`         | `solver`, `singularity.`*, `eaik_multi_solution.*`                                   |
| Phase 2 TOPP-RA | `core/topp_check.py`          | `topp_ra.enabled`; limits from `velocity_limits_rad_s`, `acceleration_limits_rad_s2` |
| Phase 3 TCP     | `compute_task_space_velocity` | CSV speed                                                                            |
| Phase 4 C1      | `check_c1_continuity`         | `continuity.enable_c1`                                                               |


---

## 15. Feasibility under zone data and TCP velocity

### What “feasible” means on a profiler run

Three independent gates (a path can pass IK and still fail the others):


| Gate             | Where                                | Pass criterion                                                                        |
| ---------------- | ------------------------------------ | ------------------------------------------------------------------------------------- |
| Geometric        | IK on blended poses                  | Every dense sample reachable                                                          |
| Joint limits     | `limits_check` in mode `summary.txt` | `max                                                                                  |
| Command tracking | `command_tracking` in `summary.txt`  | Unmasked `v`* vs col-8 (ramps excluded from the worst-point name, not from the curve) |
| RS bench         | `run_feasibility_summary.txt`        | `                                                                                     |


### How zone data affects the result


| Zone effect        | Module  | Impact                                                                                 |
| ------------------ | ------- | -------------------------------------------------------------------------------------- |
| Larger `pzone_tcp` | M1 → M2 | Shorter straights, rounder corners — different IK branch and `q'(s)`                   |
| `fine` stops       | M1      | Path through the programmed waypoint; dwell uses `T_settle_s` only in **legacy M5**    |
| Overlap reduction  | M1      | Shrinks effective radius on short segments                                             |
| Orientation onset  | M3 → M4 | C³ zone width in **arc mm**; too-short zones used to freeze θ on the corner quaternion |


### How TCP velocity affects the result


| Speed effect         | Module                   | Impact                                                              |
| -------------------- | ------------------------ | ------------------------------------------------------------------- |
| `v_cmd` per waypoint | Input → profiler stage 3 | Upper bound on `v`* in commanded mode                               |
| Joint MVC            | Stage 2                  | Pose-dependent cap — may bind before `v_cmd`                        |
| Governor             | Stage 3                  | Rate-limits `ṡ_target`; 1.15× overshoot allowed                     |
| Ceiling smooth       | Stage 2b                 | Widens notches; never raises a ceiling                              |
| Chain-rule `q̇`      | Stage 5                  | `q̇ ∝ ṡ`; higher command → higher joint rates until a ceiling binds |


Zone data primarily changes **geometry** (and thus IK / `q'`). Commanded speed
changes **timing** along a geometrically fixed path.

---

## 16. Quick API reference

### Current D2 profiler

```python
from utils.optimal_velocity.toolpath_load import load_joint_path_from_toolpath
from core.optimal_velocity.pipeline import run_diagnostics

ctx = load_joint_path_from_toolpath("path/to/toolpath.csv")  # ds_mm=0.25, plate-frame blend on
res = run_diagnostics(
    ctx.q_raw, ctx.poses, ctx.limits,
    v_cmd=ctx.v_cmd,
    v_cmd_s_mm=ctx.s_cmd_mm,
    v_cmd_at_s=ctx.v_cmd_at_s,
    plate_xyz=ctx.plate_xyz,
    knife_translation_m=ctx.knife_translation_m,
    knife_quaternion_wxyz=ctx.knife_quaternion_wxyz,
)
# In memory (not written as a playback file):
q, q_dot, q_ddot, t = res.q, res.q_dot, res.q_ddot, res.t
v_tool = res.v_star   # mm/s, tool frame
```

`utils.optimal_velocity.runner.process_one_toolpath` is the same path the CLI
uses (RS overlay, plots, summaries). Production fleet runs:

```bash
python tests/test_optimal_velocity_profile.py --dataset v7_cropped --time-optimal \
  --out Robot_APCC/Experiments/Experiement_24/Results/v7_cropped_latest_to
```

### Legacy geometry + M5 (`run_feature3`)

```python
from core.blend_zone import run_feature3
from utils.config_loader import load_batch_config

cfg = load_batch_config("config/batch_feasibility_config.yaml")
cfg.feature3_d1.enabled = True

result = run_feature3(
    toolpath_csv="path/to/toolpath.csv",
    urdf_path="Assets/.../robot.urdf",
    config=cfg,
    robot_model_name="IRB 1300-7/1.4",
    jacobian_dynamics_override=True,
)
q_positions = result.q_star
q_velocities = result.joint_velocity_result.q_dot   # Jacobian inversion
v_actual = result.speed_profile.v_actual            # M5, not Heun TOPP
```

### Feature 2 + TOPP-RA (separate — no zone blending)

```python
from utils.feasibility.pipeline_runner import run_feasibility_pipeline
from utils.feasibility.pipeline_types import FeasibilityPipelineInputs

# feature3_d1.enabled must be False
inputs = FeasibilityPipelineInputs(
    toolpath_path="path/to/toolpath.csv",
    urdf_path="Assets/.../robot.urdf",
    config=cfg,
    output_dir="output/feasibility",
    robot_model_name="IRB 1300-7/1.4",
    robot_reach_m=1.4,
    velocity_limits_rad_s=velocity_limits,
    speed_mm_s=100.0,
)
result = run_feasibility_pipeline(inputs)
```

---

## 17. Known limitations (current software)

These apply to the **velocity profiler** (`core/optimal_velocity`, defaults as
of 2026-08-13) plus the geometry it consumes. They are modelling / product
gaps, not open bugs in the stages that *are* implemented. Stage-wise knobs
are also listed in `[OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md](OPTIMAL_VELOCITY_PIPELINE_STAGEWISE.md)` §11.

### 17.1 Consecutive waypoints with the same position and different orientation

Default path parameter is **position-only base-TCP arc**
(`core/path_parameterization/position_arc.py`): `s = Σ‖Δp‖`.

If two successive samples share a tooltip position (`Δp = 0`) but rotate
(`Δθ > 0`):

- `Δs = 0` → `dq/ds` is ill-conditioned / infinite. TOPP has no well-defined
`ṡ` on that interval.
- Commanded linear speed `v_cmd` [mm/s] does not constrain a stationary
tooltip. Physically `v_tcp = 0` while the wrist still has to move.
- Cut-arc parameterisation (`s_tool`) has the same stall: pure reorientation
about the contact point advances neither `s_base` nor `s_tool`.

Weighted SE(3) arc **Approach B** in `[PATH_PARAMETERISATION.md](PATH_PARAMETERISATION.md)`
handles this:

```
Δτ = √(‖Δp‖² + (λ Δθ)²) = λ Δθ    when Δp = 0
```

Then `v_tcp = 0`, `v_cmd / ‖dp/dτ‖ → ∞` (linear command does not bind),
`ω_tcp = ṡ/λ`, and joint limits still bind through `dq/dτ ≠ 0`. If a *whole*
segment is pure rotation the per-segment λ estimator is undefined; the
fallback is the tooltip-to-wrist lever (~158 mm/rad, CLI default
`--se3-lambda-fixed 172.7`).

**Status:** B is implemented behind `--se3-arc-length` and is **off**.
Production siping paths always translate, so this has not blocked v7; it
**will** break a dwell-and-reorient or tooltip-fixed wrist flip. Do not treat
the design doc’s “we picked B” as the running default — the running default
is still position-only (Approach E in that document’s rejected list).

### 17.2 Ringing / wavering TCP angular velocity

`tcp_velocity_profile.png` panel 2 (`‖ω_BP‖`) still shows ringing (Gibbs like)  
texture that RobotStudio’s logged `orientation_speed` does not.

**Fixed (temp - needs another look):** within-segment `g` wobble from
base-frame blending, and orientation-zone chord/arc mismatch that froze θ on
the corner quaternion (zone half-widths are now arc millimetres and cover
each blend arc).

**Still present:**

- Commanded mode sets `ṡ = v_cmd / g`, so `ω = (dθ/ds_tool)·v_tool`. Any
residual `dθ/ds` imprint from the quintic, the C³ septic, or waypoint-rate
curvature leaks into `ω` at the command frequency.
- Tight 0.3 mm blends inject waypoint-frequency joint-accel notches; those
modulate `ṡ` and therefore `ω` even when linear `v`* looks acceptable.
- RS orientation-speed is logged ~every 24 ms and cannot resolve 1–2 mm
features; some of the solver “ringing” is denser sampling, not extra motion.
- `--gain-smooth-segment-aware` was a filter for manufactured `g` scatter
and is largely superseded by plate-frame blend.

Look at group T / H and `M_orientation_phasing/` before retuning TOPP.

### 17.3 Unmodeled / non-catalog joint acceleration (IRB 1300-7/1.4)

The profiler is kinematic TOPP: `|q̇_j| ≤ q̇_max[j]`, `|q̈_j| ≤ min(q̈_accel, q̈_decel)[j]`.

- `q̇_max` — Exp24 v1, treated as reliable.
- `q̈_accel` / `q̈_decel` — **p99 of observed |q̈| on Exp24 v9 snake-orientation
RS runs**, not ABB catalog, not payload- or pose-dependent, not a motor
thermal model. Symmetric bound = `min(accel, decel)`.
- Old v8 values (especially **J4 = 144 deg/s²**) collapsed time-optimal /
no-dip speeds. `JointLimits.exp24_neutral()` in `types.py` **still stores
those v8 numbers**; production loads YAML v9 instead. Do not call the helper
and assume it matches the robot.
- `feature3_d1.joint_accel_limit_scale` exists for **legacy M5** only.
- `T_settle_s = 0.2 s` is an estimate, never measured on site (legacy M5
fine-point dwell).
- IRB 1300-10/1.15 and 11/0.9 `acceleration_limits_rad_s2` are placeholders
(`FIX ME` in `robots_config.yaml`). Only 7/1.4 is production.

### 17.4 No controller-ready `q(t)`, `q̇(t)` export

This was my North Star. The profiler **computes** a time-parameterized joint trajectory
(`ProfileResult.q`, `.q_dot`, `.q_ddot`, `.t`) and plots it (groups D, G).
It does **not** write a playback file.

Feature 2 exports (`export_final_trajectory_csv`,
`export_dense_ik_trajectory_csv`) are not called. There is no RAPID
(`MoveAbsJ` / `MoveL` with speed data), ROS bag, or timestamped joint CSV
from this pipeline. Using the profile on the controller is a separate
deliverable.

### 17.5 RobotStudio corner slowdown is unmodeled

The solver slows only when a **joint ceiling** (or the governor / v_cap
lookup) binds. IRC5 also derates on zone radius and turn angle
(“CornerPathReduction”). On v7 traj_7 / 9 / 10 around `s ≈ 85–92 mm` the
solver holds ~41 mm/s where RS holds ~37. Legacy M5’s `k_corner_dip` is
**not** wired into the D2 profiler as we have removed any task space constraints (max tangential or radial acceleration). We have not modeled joint acceleration well to handle this well. 

### 17.6 Ceiling smoothing widens notches

`smooth_ceiling_min_preserving` with a 2.5 mm window never raises a ceiling,
so the profile stays feasible, but a two-sample accel dip becomes a ~3 mm
trench. On v7 traj_1 this is the largest commanded-mode tracking loss
(`v*/v_cmd ≈ 0.65` at `s ≈ 226 mm`, secant-accel (s_dot^2) minimum carried in from a  
neighbour). `--ceiling-smooth-mm` controls it; `0` disables.

### 17.7 `ṡ = v_cmd / g` is ill-conditioned

Frame gain `g = ds_tool/ds_base` on v7 spans ~0.09–1.23. Relative error in  
`g` lands 1:1 on the path-space target (clip ~1e4 mm/s behind a small `g`).  
Parameterizing by the **cut** arc would make `ṡ_tool ≤ v_cmd` exact, but the  
cut arc stalls on pure reorientation (§17.1). `--se3-arc-length` is the  
candidate that stalls in neither factor; it is not the default. We need a better way to model the T_P_K frame as an extension of TOPP-RA, rather than mapping velocities in Base and tool frame back and forth.

### 17.8 Spline rounding vs blended arc (`I_spline_fk_check`)

Pass budget: `|Δp| ≤ 0.2 mm`, `|Δθ| ≤ 0.1 rad`. Fleet runs often FAIL a
majority of toolpaths (e.g. 14/18). Tight corners lose curvature in the
quintic; that is a separate accuracy bottleneck from TOPP. Switching path
parameter does not fix it. See group A / `I_spline_fk_check/`.

### 17.9 No torque, no hardware jerk, path jerk off

Limits are kinematic (q̇, q̈). There is no torque / `M(q)q̈` TOPP.  
`--path-jerk-max` default is **0** (off): bang-bang `s̈` corners are not  
slew-limited. Joint jerk panels (`--jerk`) are diagnostic S–G derivatives,  
not constraints. ABB jerk / S-curve is only indirectly present via the  
empirical governor.

### 17.10 ABB orientation model is a reconstruction

The C³ septic dual-schedule (`_abb_orientation_schedule`) matches RS
stop-point SLERP outside zones to ~0.003° on v7, but it is **not** extracted
from IRC5 firmware. `ori_schedule_mode="legacy"` is hold–SLERP–hold.
Fix-1/2/3 (tool-arc SLERP, Step 5b smooth, ISA re-phase) are **inactive
under the ABB schedule** (superseded; still reachable via flags).

### 17.11 Governor is empirical

`cmd_accel_max = 8000 mm/s²`, 1.5 mm low-pass, 1.15× overshoot: fitted to  
observed RS ramps on one dataset family. Not a documented IRC5 parameter.  
`--cmd-accel-max 0` disables it (useful for geometry A/B, not for RS match).

### 17.12 RS v_cap lookup is a table, not a model

Spacing×zone cruising cap comes from
`velocity_zone_lookup_table_interp.csv`. Failed lookups are **excluded**
from RS bench (`--no_vcap` disables the cap entirely). The table is not a
substitute for IRC5 zone physics.

### 17.13 Uniform resample is position-arc, not SE(3)

`--uniform-resample-mm` regrids on `Σ‖Δp‖`. On a rotation-dominated stretch
it still clusters samples. Combined with §17.1, dense orientation change
between coincident positions cannot be represented.

---

## 18. Module index

### Feature 3 D2 — geometry + velocity profiler


| Label             | File                                                    | Primary types                                        |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| Input             | `utils/csv_loader_toolpath.py`                          | `ToolpathLoadResultF3`                               |
| Load for profiler | `utils/optimal_velocity/toolpath_load.py`               | `ToolpathContext`, `load_joint_path_from_toolpath()` |
| M1                | `core/blend_zone/zone_resolver.py`                      | `ZoneParams`                                         |
| M2                | `core/blend_zone/blend_geometry.py`                     | `BlendArcGeometry`                                   |
| M3                | `core/blend_zone/orientation_zone.py`                   | `EffectiveOrientationZone`                           |
| M4                | `core/blend_zone/path_sampler.py`                       | `DensePath`, `sample_blended_path_plate_frame()`     |
| F2 IK             | `core/feasibility/analyzer.py`                          | `FeasibilityAnalyzer`                                |
| Path parameter    | `core/path_parameterization/position_arc.py`            | default `s`; SE(3) behind `--se3-arc-length`         |
| Profiler          | `core/optimal_velocity/pipeline.py`                     | `run_diagnostics()`, `ProfileResult`                 |
| MVC / TOPP        | `core/optimal_velocity/mvc_ceilings.py`, `heun_topp.py` | ceilings, `ṡ*(s)`                                    |
| CLI               | `tests/test_optimal_velocity_profile.py`                | fleet / single-toolpath entry                        |
| Plots / summaries | `utils/optimal_velocity/plotting.py`, `reporting.py`    | PNG groups, `run_feasibility_summary.txt`            |


### Legacy inside `run_feature3()` (not the RS-benchmarked profiler)


| Label        | File                                                    | Primary types                            |
| ------------ | ------------------------------------------------------- | ---------------------------------------- |
| M5           | `core/blend_zone/speed_profile.py`                      | `SpeedCalibration`, `SpeedProfileResult` |
| M5-D2        | `core/calibration/tcp_dynamics.py`, `joint_dynamics.py` | `JointDynamicsCalibration`               |
| M6           | `core/checks/task_space_velocity.py`                    | `JointVelocityResult`                    |
| Orchestrator | `core/blend_zone/pipeline.py`                           | `Feature3D1Result`, `run_feature3()`     |


### Feature 2 feasibility + TOPP-RA (not part of F3 D2)


| Label           | File                                   | Primary types                               |
| --------------- | -------------------------------------- | ------------------------------------------- |
| F2 orchestrator | `utils/feasibility/pipeline_runner.py` | `run_feasibility_pipeline()`                |
| TOPP-RA         | `core/topp_check.py`                   | `ToppraResult`, `parameterize_trajectory()` |
| Phase 3 TCP     | `core/checks/task_space_velocity.py`   | `compute_task_space_velocity()`             |
| Phase 4 C1      | `core/checks/c1_continuity.py`         | `check_c1_continuity()`                     |
| Entry CLI       | `feasibility_analysis.py`              | `--feature3` selects legacy F3 vs F2        |


---

*End of Feature 3 D2 Module Pipeline Reference.*