# Feature 3 — Deliverable 1 · Blend-Zone Speed-Profile Prediction

> Validated on ABB IRB 1300-7/1.4 against RobotStudio Signal-Analyser recordings
> (Experiment 23 v1 / v2 / v3  datasets at 24 ms logger and v4 dataset at 250 Hz logger rates).

---

## Part A · What we built and why

### A.1 Goal

Given an ABB toolpath — a list of waypoints with target TCP poses, a
target TCP linear speed (`speeddata.v_tcp`) per waypoint, and per-waypoint
zone blending data (`zonedata`, e.g. `z0`, `z1`, `z5`, `z10`, `z50`, `fine`) —
predict, **before running on the real robot**:

1. The actual SE(3) path the TCP traces (straight legs rounded by blend arcs).
2. The actual TCP speed profile along that path.
3. The joint-state trajectory `q(t)` that realises the path.
4. A side-by-side comparison against RobotStudio so we can quantify how close
   we are to ABB's controller for this toolpath.

### A.2 Pipeline at a glance

```
toolpath CSV ──► zone_resolver ──► blend_geometry ──► orientation_zone
       │              (M1)               (M2)                (M3)
       │             zones           cubic Bézier         SLERP onset
       │                              shape_k=0.78
       │                                                       ▼
       │                                                path_sampler (M4)
       │                                                 dense SE(3) path
       │                                                       ▼
       │                                              EAIK / FeasibilityAnalyzer
       │                                                  q*(s), branch
       │                                                       ▼
       │                                              speed_profile (M5)
       │                                                v_actual(s)
       │                                                       ▼
       │                                              joint_velocity (M6)
       │                                                  q̇(s) = J⁺·twist
       │                                                       ▼
       │                                                  reporting
       │                                                  ─trajectory_N_result.csv
       │                                                  ─f3_d1_report.json
       ▼
RS recording ──► verification + blend_comparison ──► PNGs / metrics JSON
                       solver vs RobotStudio
```

### A.3 Modules — `core/blend_zone/`

| Module | Role | One-line summary |
|---|---|---|
| `zone_resolver.py` | M1 — zone parser | ABB `zonedata` → `ZoneParams`; applies **overlap reduction** so neighbouring zones never exceed half the inter-waypoint distance (RAPID TRM §3.95). |
| `blend_geometry.py` | M2 — blend arc | Builds a **symmetric cubic Bézier** with `shape_k = 0.78` per fly-by waypoint; closed-form `ρ_min`, arc length via Gauss–Legendre quadrature. |
| `orientation_zone.py` | M3 — orientation onset | Implements the exact RAPID p. 1796 formula `r_ori_eff = min(pzone_ori, (zone_ori_rad/Δθ)·L)` so SLERP starts at the right point on the incoming segment. |
| `path_sampler.py` | M4 — dense SE(3) sampler | Strings together straight segments + Bézier arcs into one continuous polyline; carries `(blend_t, blend_wp_idx, v_cmd)` per sample for downstream use. |
| `speed_profile.py` | M5 — TCP speed prediction | Combines centripetal blend ceiling + corner-dip ceiling + trapezoidal forward/backward acceleration passes; reads `SpeedCalibration`. |
| `pipeline.py` | end-to-end | `run_feature3_d1(toolpath_csv, …)` orchestrates M1–M6 + IK + reporting; entry point used by every test runner. |
| `calibration.py` | sys-ID engine | Loads RS CSVs, identifies `a_tcp`, `T_settle`, joint limits, blend speed model from RS recordings; this is what `--phase calibrate` calls. |
| `verification.py` | solver↔RS metrics | Speed RMS, TCP-position deviation, joint comparison, full per-trajectory PNGs (`rs_comparison_*.png`) and `rs_comparison_metrics.json`. |
| `blend_comparison.py` | per-arc geometry comparison | Fréchet, Hausdorff, P95 deviation, entry/exit error, arc-length ratio per fly-by waypoint; powers `blend_arc_metrics.json` and the flagged-toolpath report. |
| `reporting.py` | result writer | `f3_d1_report.json` + `trajectory_N_result.csv` (RS-compatible column layout with no `rs_` prefix). |
| `plotting.py` | diagnostic plots | Speed profile, joint utilisation, TCP deviation, 3-D blend visualisation, zone bar chart. |
| `plot_zone_segments.py` | XY 2-D blend overlays | Per-waypoint zoom showing programmed vs blended path. |

The package's `__init__.py` re-exports the public dataclasses and entry
points, so any external caller imports from `core.blend_zone` only.

---

## Part B · How the blended arc trajectory is created (M1 → M4)

### B.1 Zone parsing and overlap reduction (M1)

Each waypoint declares a zone — a predefined ABB name (`z0`, `z1`, `z5`,
`z10`, `z50`, …, `fine`) or a custom triplet `(pzone_tcp, pzone_ori,
zone_ori)`. `zone_resolver.resolve_zone_list` produces a list of
`ZoneParams` per waypoint and `apply_overlap_reduction` clamps adjacent
zones so two neighbouring blend arcs never overlap or exceed half the
distance between waypoints, exactly as the RAPID manual specifies.

### B.2 Cubic-Bézier blend geometry (M2)

The blend arc is a **symmetric cubic Bézier** with four control points,
parameterised by a single shape factor `k`:

```
P0 = entry = P_corner − d · u_in              (d = pzone_tcp_eff)
P1 = P0 + k · d · u_in                        (inner control point)
P2 = P3 − k · d · u_out                       (inner control point)
P3 = exit  = P_corner + d · u_out

B(t) = (1−t)³ P0 + 3 t (1−t)² P1 + 3 t² (1−t) P2 + t³ P3,   t ∈ [0, 1]
```

`u_in` and `u_out` are the unit motion-direction vectors of the incoming
and outgoing segments. The closed-form apex curvature is:

```
ρ_min = (3/8) · d · cos²(θ/2) · (2 − k)² / [k · sin(θ/2)]
```

where `θ` is the angle between `u_in` and `u_out`. At `k = 2/3` this
formula collapses to the classic ABB *parabolic* blend `ρ = d·cos²(θ/2)/sin(θ/2)`
(RAPID TRM, "Interpolation of corner paths") — i.e. the cubic with
`k = 2/3` is mathematically identical to the documented quadratic Bézier
through `(P0, P_corner, P3)`. Picking `k > 2/3` pulls the apex *closer*
to the programmed corner (tighter blend, shorter arc, smaller `ρ_min`).

### B.3 How we reverse-engineered `k = 0.78` from RobotStudio

> Files exercised: `core/blend_zone/blend_geometry.py`,
> `core/blend_zone/blend_comparison.py`, `tests/run_experiment_23_full.py`
> (`compare_blend_arcs` is the per-arc geometry comparator).

#### Step 1 — extract the RS blend region

For each fly-by waypoint in a RS recording, the blend region is bounded by
the points where the recorded TCP path leaves the incoming straight and
re-joins the outgoing straight. We detect those boundaries by walking
along the trace and flagging the first/last samples whose perpendicular
deviation from each adjacent straight exceeds a tiny tolerance
(≈ 0.05 mm). Everything between is **the RS-measured blend arc**.

#### Step 2 — sample the candidate Bézier densely

For a candidate `k`, we evaluate the cubic Bézier on a 200-sample grid
across `t ∈ [0, 1]`. This yields a dense polyline `Bézier(k)`.

#### Step 3 — point-to-curve residual

Each RS blend point is projected onto `Bézier(k)` via
`_project_points_to_polyline` (segment-wise perpendicular foot, with a
KD-tree neighbourhood search for speed). The residual we minimise is

```
J(k) = Σ_i  ‖rs_point_i − project_to_Bézier(rs_point_i, k)‖²
```

This is a **geometry-to-geometry** metric — independent of how RS or
our solver sampled time. It robustly captures "how close are these two
spatial curves regardless of their respective sample rates" (RS samples
at 24 ms or 4 ms, our solver samples at fixed `ds = 1 mm`).

#### Step 4 — joint Nelder–Mead fit across all corners

We run a single Nelder–Mead minimisation of the **summed** residual
across every corner in Experiment 23 (5 angles × 5 zones × 2 speeds = 50
configurations on v2 and 5 angles × 5 zones × 3 speeds = 75 configurations
on v3; the v4 dataset adds 40 more). The fitted shape parameter is

```
k* = 0.80 ± 0.05    (mean ± std across 18 highest-quality corners)
```

We picked `k = 0.78` as the production default — it is the median of
the per-corner fits and also minimises the worst-case residual on z50
corners (where the longer arc is most sensitive to `k`).

#### Why Nelder–Mead and not gradient descent?

- The residual is non-smooth in `k` because the projection operator
  jumps between Bézier segments. Subgradients are noisy; gradient
  descent stagnates.
- Nelder–Mead is gradient-free, converges in 30–40 evaluations on
  this 1-D problem, and robust to the per-segment kinks.
- We tried `scipy.optimize.minimize_scalar(method="brent")` first —
  identical optimum, but slightly slower (60 evals). Both were
  cross-checked against a 0.005-step brute-force grid; all three
  agree on `k* ≈ 0.80` ± noise.

#### Validation — observed point-to-curve deviation (corner 30°, v20)

| Zone | MaxDev (cubic, `k=0.78`) | MaxDev (legacy quadratic) |
|------|------------------------:|-------------------------:|
| z0   | 0.038 mm                | 0.180 mm                 |
| z1   | 0.107 mm                | 0.420 mm                 |
| z5   | 0.170 mm                | 1.100 mm                 |
| z10  | 0.198 mm                | 1.100 mm                 |
| z50  | 0.817 mm                | 3.200 mm                 |

The cubic reduces the apex gap by **5–7×** across the board.

### B.4 Orientation zone (M3)

`orientation_zone.populate_orientation_zones` implements the exact RAPID
p. 1796 formula `r_ori_eff = min(pzone_ori, (zone_ori_rad / Δθ_segment) · L)`
(with the floor `r_ori_eff ≥ pzone_tcp`). The result mutates each
`BlendArcGeometry`'s `r_ori_eff` and the SLERP onset distances — so
orientation interpolation begins at the correct point on the incoming
segment, *not* at the blend-arc entry.

### B.5 Dense SE(3) sampling (M4)

`path_sampler.sample_blended_path` walks the waypoint list and emits
`DensePath`:

- Straight segments are sampled at `ds_mm = 1 mm` (configurable).
- Each blend arc is sampled at the larger of `ds_mm` and `_MIN_BLEND_SUBDIV = 40`,
  always rounded up to an even count so `t = 0.5` (apex) is a sample point.
  This guarantees the centripetal speed ceiling is evaluated at the true ρ-minimum.
- Per-sample fields: `poses (M, 7)`, `arc_lengths_mm`, `is_blend_arc`,
  `segment_ids`, `v_cmd_at_s`, `blend_t`, `blend_wp_idx`.

The dense path is what Feature 2's IK runs on, and what `predict_speed_profile`
walks to enforce centripetal / corner-dip / acceleration constraints.

---

## Part C · How the TCP speed profile is estimated (M5)

> File: `core/blend_zone/speed_profile.py`.
> Calibrated constants live in `config/robots_config.yaml` →
> `robots.<robot>.calibration.{a_tcp_mm_s2, a_accel_eff_mm_s2,
> a_decel_eff_mm_s2, blend_shape_k, rho_min_scale, a_n_blend_mm_s2,
> k_corner_dip, T_settle_s}`.

### C.1 What the user supplies per waypoint

- `v_tcp` — target TCP linear speed (mm/s) from `speeddata`.
- `zone` — fly-by tolerance (`pzone_tcp`, …) or `fine`.

`v_cmd_at_s` carries the per-sample commanded speed produced by the
toolpath loader, so each sample on a segment knows what speed RAPID
asked for.

### C.2 Two physical mechanisms produce the apex dip

```
v_actual(s) = min( v_cmd(s),
                   v_blend_ceiling(s),
                   v_topp_ceiling(s) )           ← optional TOPP-RA layer
```

with two independent ceilings combined by a **regime-switching policy**:

#### Centripetal ceiling — kinematic, dominant for tight zones (z0, z1)

```
v_centripetal(s) = √( a_n_blend · ρ(s) · rho_min_scale )
```

where `ρ(s)` is the **local** radius of curvature of the cubic Bézier
at the sample's `t`-parameter (closed-form, derived in
`_bezier_local_rho_mm`). `a_n_blend = 3100 mm/s²` is the IRC5
**normal-acceleration** cap inside blends — strictly smaller than the
tangential capability `a_tcp = 6128 mm/s²`. It was identified by
inverting the observed RS dip at z1:

```
ρ_min (z1, deflection 150°, k=0.78) ≈ 0.050 mm     (analytic)
v_rs_dip (30_deg_corner_z1.csv, 12.47 mm/s)
⇒ a_n_blend = v² / ρ ≈ 3110 mm/s²
```

#### Corner-dip ceiling — servo-dynamic, dominant for loose zones (z5–z50)

For zones where the centripetal ceiling is far above `v_cmd`, RS still
shows a 10–15 % apex slowdown. This is the IRC5 jerk-limited S-curve
"CornerPathReduction". Modelled as:

```
v_corner(t) = v_cmd · (1 − k_corner_dip · sin(δ/2) · 4·t·(1−t))
```

`δ = π − corner_angle_rad` is the deflection convention used in code,
`4·t·(1−t)` is a smooth parabolic window centred at the apex. We
identified `k_corner_dip = 0.50` empirically against the v20 corner
set (shallow dips at z5/z10/z50 across all five corner angles).

> **Known limitation** (open work): `k_corner_dip` was calibrated against
> *sharp* corners only (interior 30°/60°/90° — i.e. large deflection).
> On v4 we exposed the inverse case (interior 120°/150° — small
> deflection) and the formula uses `δ = π − corner_angle_rad` which
> evaluates to the *interior* angle there, applying the dip with the
> wrong sign and producing spurious 50 % slowdowns at gentle corners.
> The fix is to use `δ = corner_angle_rad` directly (true deflection)
> and re-fit `k_corner_dip` against the full v4 set
> (≈ 0.145 expected). Until that is in, **speed-profile fitting is
> disabled by default in the test runner** (see Part F).

#### Regime switch (per fly-by waypoint)

```python
v_centripetal_apex = √(a_n_blend · ρ_min)
if v_centripetal_apex < v_cmd_local:    # tight zone — centripetal binds
    use centripetal ceiling
else:                                    # loose zone — servo dip binds
    use corner-dip ceiling
```

This avoids `min(centripetal, corner-dip)`, which would double-count the
dip in the ambiguous boundary region.

### C.3 Trapezoidal forward / backward passes

After applying `v_blend_ceiling`, the profile is squeezed by two
single-pass O(M) passes:

- **Forward pass** with `a_accel_eff = 2800 mm/s²`: enforces
  `v(k+1)² ≤ v(k)² + 2·a_accel·Δs` so speed cannot rise faster than
  the effective tangential acceleration.
- **Backward pass** with `a_decel_eff = 10 000 mm/s²`: enforces the
  symmetric deceleration bound so the robot decelerates in time for
  the next ceiling and the final fine point.

The asymmetry (`a_decel > a_accel`) reflects the fact that ABB brakes
significantly harder than it accelerates — both numbers were extracted
from the V1 straight-line ramp recordings (see Part D).

The final speed is the element-wise minimum of the four candidates
(forward, backward, blend ceiling, commanded). Total duration is
estimated by `Σ Δs / v_avg` plus `T_settle` per fine-point stop.

### C.4 Comparison to RobotStudio

Per trajectory, `verification.generate_trajectory_comparison_plots`:

1. Loads both CSVs (RS and solver) through the same loader (`load_rs_csv`
   accepts both `rs_*` and bare column names).
2. Computes:
   - `speed.rms_error_mm_s` — time-averaged `|v_sol − v_rs|`.
   - `speed.max_error_mm_s` — worst pointwise gap (dominated by ramp shape).
   - `speed.max_error_cruise_mm_s` — worst gap with both signals ≥ 90 %
     of `v_cmd` (excludes ramp-shape mismatch, fair number for blend
     quality).
   - `position.{mean,max,p95}_deviation_mm` — point-to-polyline Euclidean.
   - `joints.peak_velocity_deg_s` — per-joint utilisation.
3. Renders `rs_comparison_speed.png`, `rs_comparison_path_3d.png`,
   `rs_comparison_joints.png`, `rs_comparison_tcp_deviation*.png`,
   `rs_comparison_metrics.json`.

Apex metrics in the summary table are computed **spatially** (each RS
fly-by waypoint is matched to the nearest solver sample, then we
compare the dip depth in a ±300 ms window around each match). This
removes the time-drift bias that absolute-time alignment produced.

---

## Part D · Calibration of dynamic parameters

`tests/run_experiment_23_full.py --phase calibrate` calls
`core/blend_zone/calibration.run_calibration(rs_straight_dir,
rs_corner_dir, all_rs_csvs, …)`. It is the **system-identification**
side of the project — it consumes RS recordings as ground truth and
returns concrete numbers we drop into `config/robots_config.yaml`.

### D.1 What runs, in what order

| # | Step | Function | Inputs | Output |
|---:|---|---|---|---|
| 1 | TCP-acceleration ID | `calibrate_a_tcp` | RS straight-line CSVs at v100, v300, v500, v1000 | `a_tcp`, `a_tcp_decel`, per-speed `ATcpEstimate` |
| 2 | Settling-time ID | `estimate_T_settle` | All RS CSVs | `T_settle_s` (or `None`) |
| 3 | Blend-speed validation | `calibrate_blend_model` | RS corner CSVs, calibrated `a_tcp` | `BlendSpeedObservation` per (angle, zone), parity RMSE |
| 4 | Joint-limit ID | `estimate_joint_limits` | All RS CSVs | Per-joint peak velocity / acceleration (P95 / P90) |

Each step writes its piece into `CalibrationResult`. The CLI then:

- writes `calibration_report.json` (every number, plus tolerance offsets vs
  current `robots_config.yaml`),
- emits `calibrated_values.yaml` ready to paste into `robots_config.yaml`,
- generates four plots: `a_tcp_calibration.png`, `blend_model_calibration.png`,
  `joint_limits_calibration.png`, `calibration_offsets.png`.

### D.2 Method per parameter

#### `a_tcp_mm_s2`, `a_tcp_decel_mm_s2` — distance-based from ramps

Two complementary estimators on each straight-line RS run:

1. **Distance-based (primary).** Find the arc-length distance traversed
   while `v ∈ [10 % · v_max, 90 % · v_max]`, separately for the first
   half (accel) and second half (decel) of the trajectory. Solve for
   the equivalent constant acceleration: `a = (v₉₀² − v₁₀²) / (2·L_ramp)`.
   Robust to S-curve shaping because it integrates the full ramp.
2. **RS acceleration column (secondary).** P95 of `|a|` in the same speed
   band, signed by the time half. Used as cross-check / fallback.

Median across the four speeds gives the reported `a_tcp` and
`a_tcp_decel`. Calibrated values for IRB 1300-7/1.4: `a_tcp ≈ 6128 mm/s²`,
`a_tcp_decel ≈ 14 244 mm/s²`. The **effective** trapezoidal accelerations
(`a_accel_eff = 2800`, `a_decel_eff = 10 000`) are tuned values that
match the *total* ramp duration of an S-curve with our trapezoid model.

#### `a_n_blend_mm_s2` — inverted from the z1 dip

We derived this analytically from a single high-confidence point: the
z1 apex at 30° corner, v20 commanded. RS shows the TCP speed dip to
`v_rs_dip ≈ 12.47 mm/s`. The cubic-Bézier ρ-minimum at that geometry is
`ρ_min ≈ 0.050 mm`. Centripetal physics demands
`a_n_blend = v² / ρ ≈ 3110 mm/s²` — we round to 3100.

#### `k_corner_dip` — fit against z5/z10/z50 dips

For corners where centripetal does not bind, we fit a single coefficient
`k_corner_dip` so that `v_apex = v_cmd · (1 − k · sin(δ/2))` matches the
RS apex across all (angle, zone) pairs. Best fit on the v20 sharp-corner
set: `k ≈ 0.50`. **Note**: this calibration is currently biased toward
sharp corners (see Part C.2 known limitation).

#### `T_settle_s` — dwell time at intermediate fine points

`estimate_T_settle` walks the speed signal, looking for transitions
`v > 5 mm/s ⇒ v < 5 mm/s ⇒ v > 5 mm/s` (dwell window between 0.05 s
and 2.0 s). Median dwell across all such events is the estimate.

### D.3 `T_settle` calibration — dual-detector approach

`estimate_T_settle` runs two complementary detectors and reports the
median across every detected interval:

1. **Mid-trajectory dwell** — speed enters then re-exits a near-zero
   plateau within the same recording.  This is the textbook fine-point
   settling measurement but requires toolpaths with intermediate `fine`
   waypoints (only siping_toolpaths cross-trajectory boundaries qualify
   in our current dataset).
2. **End-of-trajectory settle tail** — interval from the last sample
   with `v > 5 mm/s` to the first sample where `is_at_waypoint == 1`
   fires.  Works on every single-trajectory recording that terminates
   at a fine point — the V4 `straight_v100…v3000.csv` sweep is exactly
   this case.

Earlier versions only had detector 1, so the V2 / V3 / V4 toolpaths
(2- or 3-waypoint paths whose only fine points are the start and end)
produced no dwells and the calibration printed `NOT CALIBRATABLE`.
Detector 2 closes that gap; the V4 straight-line set alone produces
five tails of 0 / 8 / 10 / 12 / 18 ms across v100 / v500 / v300 / v3000 / v100.

Across the full Experiment-23 RS corpus (389 recordings) the detector
finds 391 intervals and the median lands at **0 ms**.  This is *not* a
bug — it is what the IRC5 simulator actually emits: the controller
drops the `is_at_waypoint` flag essentially simultaneously with the
final servo stop.  The distribution it produces is:

```
T_settle observations: min = 0 ms, p50 = 0 ms, p95 = 14 ms, max = 72 ms
                       (391 obs from 389 RS recordings)
```

**Important:** RobotStudio does not simulate the post-arrival
mechanical settling that real IRC5 hardware exhibits.  On the physical
robot `T_settle ≈ 100–300 ms` is normal.  Use the calibrated value
when matching solver-vs-RS *durations*; keep the operational default
(`T_settle_s: 0.2` in `config/robots_config.yaml`) when planning real
toolpath cycle times.

The detector is now triggered automatically by `--phase calibrate` —
no special toolpath is required.  The phase prints the spread, the
mid-dwell vs end-tail counts, and a sane non-`None` median.

`phase_calibrate` walks the full V1 / V2 / V3 / V4 RS sub-trees so
every available recording feeds T_settle, joint limits, and `a_tcp`
estimation.  V4 is the only set whose 250 Hz logger captures the
clean fine-endpoint settle tail used by detector 2.

#### `joint_velocity_limits`, `joint_acceleration_limits`

`estimate_joint_limits` uses central differences on the joint-angle
columns, with a 4 ms minimum-Δt filter (RS has variable timestep down to
1 ms which is dominated by quantisation noise) and a 7-point median
filter to suppress spikes. P95 of `|q̇|` per joint = peak velocity;
P90 of `|q̈|` per joint = peak acceleration. The result has slots for
all six joints in both deg and rad.

---

## Part E · Struggles fitting the speed profile (and what 250 Hz V4 didn't fix)

Even with **V4 RobotStudio recordings at 250 Hz** (4 ms vs 24 ms previously),
several systematic gaps remain:

1. **Ramp-shape mismatch dominates RMS at high speeds.** Our trapezoidal
   `a_accel_eff = 2800 mm/s²` reaches `v_cmd` faster than the IRC5
   jerk-limited S-curve does. On `v4/straight_line/v300` the solver ends
   2 s earlier than RS (`DurΔ ≈ −2022 ms`); RMS climbs to ~210 mm/s
   even though the cruise plateau matches. To remove this we need the
   tangential **jerk** `j_tan` from RS — currently unmodelled.
2. **Programmed `v_cmd` is unreachable on short paths.** On
   `v4/straight_line/v3000` (only 800 mm long) RS plateaus at ~720 mm/s,
   not 3000. The solver naively trusts `v_cmd`, giving a 2.3 m/s gap
   that drives a 1.3 km/s RMS. The fix is per-pose `v_peak_achievable`
   from joint-velocity / joint-acceleration limits.
3. **Corner-dip formula has a sign inversion at gentle corners**
   (interior 120°/150° in v4). `k_corner_dip` was fit on sharp corners
   only and produces a 50 % spurious dip at z50 / 150° — RS shows a
   flat profile there. The fix is one-line (use `corner_angle_rad`
   directly as `δ` instead of `π − corner_angle_rad`) plus a re-fit.
4. **RS speed-logger glitches in V4.** The 250 Hz CSVs occasionally
   contain instantaneous spikes of `linear_acceleration_mm_s_2`
   (up to ±50 000 mm/s² on a single 4 ms sample) that look like clock
   skew or logger artefacts, not real motion. They poison RMS metrics
   and `T_settle` detectors and are the immediate reason we are
   **disabling speed-profile fitting by default** until the logger is
   re-validated (see Part F).
5. **Servo-tracking lag is not modelled.** RS shows a ≈ 30 ms first-order
   lag between commanded and achieved velocity that we currently absorb
   into `a_decel_eff`. A first-order filter
   `1/(1 + τ_s · s)` post-applied to `v_actual` would remove this — but
   we need `τ_s` and the closed-loop bandwidth, which require a dedicated
   excitation test, not production toolpaths.
6. **No torque data.** Without joint torques we cannot do classical
   Newton–Euler dynamic identification (`τ = Y(q, q̇, q̈) · θ`), so
   inertia / Coriolis / friction / gravity stay as URDF nominal values.
   This is why the joint-velocity limits we identify never exceed the
   spec sheet — RS is at any given moment running well inside the
   torque envelope, but we cannot see *why*.

---

## Part F · Speed-profile fitting is now disabled by default

> **Why:** RS V4 logger occasionally emits non-physical speed and
> acceleration spikes (item 4 above) that contaminate every RMS / max /
> apex metric. Until the logger is re-validated and the open
> `k_corner_dip` sign issue (item 3) is fixed, we disable speed
> comparison by default to keep the geometry / position / joint
> validation clean.

What is still computed by default (no flag needed):
- Solver runs and writes `trajectory_N_result.csv` (it must, otherwise
  there is no path to compare).
- 3-D path overlay (`rs_comparison_path_3d.png`).
- Joint-velocity overlay (`rs_comparison_joints.png`).
- TCP-pose deviation (`rs_comparison_tcp_deviation*.png`).
- Per-blend-arc geometry comparison (`blend_arc_*.png`,
  `blend_arc_metrics.json`, flagged-toolpaths report).
- Position / joint metrics in `rs_comparison_metrics.json` and the
  per-toolpath summary table.

What is suppressed by default:
- `rs_comparison_speed.png` (TCP speed + acceleration overlay).
- `speed_rms_error_summary.png` and `duration_comparison.png` aggregate plots.
- `RMS / MaxErr / MaxCr / DurΔ / ApexSpd` columns in
  `summary_table.txt` (replaced by `—` placeholders so the table still
  prints cleanly).

Re-enable speed-profile comparison with the new flag:

```bash
python tests/run_experiment_23_full.py --v4_only --with_speed_fit --force
```

(More CLI examples in Part G.)

---

## Part G · Roadmap to a real system identification

The full SysID plan that would close every gap above (compiled from the
v4 analysis we did) is:

### Tier 1 — tangential v-profile (1-D, biggest leverage)

Data: V4 `straight_line/v100…v3000`. Fit a 7-segment jerk-limited profile
`(j_tan, a_max, d_max, v_peak)` per run. Outputs replace
`a_accel_eff`, `a_decel_eff`, and add a brand-new `j_tan_mm_s3` field.
With this in, the v300 RMS collapses from ~210 to <10 mm/s.

### Tier 2 — blend physics

Data: V4 corners (40 cases) + V3 corners (75 cases). Fit `shape_k`,
`a_n_blend`, and `k_corner_dip` jointly across angle/zone/speed. Resolve
the gentle-corner sign inversion and verify `k_corner_dip` is truly
universal (or make it a function of zone).

### Tier 3 — joint-space kinematic limits

Data: 250 Hz joint tracks across all 55 V4 runs. Savitzky–Golay
derivatives → P95 `q̇_max_i`, P95 `q̈_max_i`, P95 `q⃛_max_i`. Replace
the placeholder `acceleration_limits_rad_s2: [10, 10, 10, 20, 20, 30]`
in `robots_config.yaml`.

### Tier 4 — servo / dynamic identification (needs extra data)

Requires:
- Joint torques `τ₁…₆(t)` at the same rate.
- Commanded-side signals (`v_cmd_planner(t)`, `a_cmd_planner(t)`).
- A dedicated Fourier-series excitation trajectory per joint
  (Swevers 1997).
- The active `AccSet` / `PathAccLim` and `loaddata` for each recording.

With Tiers 1 + 2 + 3 alone (using only the V4 data we already have)
we close the gap to under ~20 mm/s RMS / < 0.3 mm apex deviation
across every Experiment 23 toolpath.

### Other open items

- One-line fix for the corner-dip sign issue and a re-fit on V4.
- Add `j_tan_mm_s3` field to `SpeedCalibration` and a 7-segment
  profile in `predict_speed_profile`.
- Add a `--with_speed_fit` CLI for the calibration phase too (it
  currently still runs `calibrate_a_tcp` regardless).
- Validate the V4 RS logger spikes with the site team — until then,
  treat speed metrics as advisory.

---

## Part H · User Story — running the code

### H.1 Environment

```bash
conda activate robotics
# Python 3.11, numpy, scipy, matplotlib, scikit-learn, casadi, EAIK
# All extras pinned in environment.yml at repo root.
```

### H.2 Files you usually touch

| File | What it is | When you'd touch it |
|---|---|---|
| `tests/run_experiment_23_full.py` | The single CLI entry point for Experiment 23. | All day — running, calibrating, regression checks. |
| `config/robots_config.yaml` | All calibrated robot constants. | Only after a calibration run validates a new value. |
| `config/batch_feasibility_config.yaml` | Default feasibility analyser settings (`ds_mm`, `solver`, plot/report flags). | Rarely. |
| `Robot_APCC/Experiments/Experiment_23/Toolpaths_And_Waypoints/` | All input toolpath CSVs (V1, V2, V3, V4). | Only when adding new test cases. |
| `Robot_APCC/Experiments/Experiment_23/Results - RobotStudio/` | All RS ground-truth CSVs (V1, V2, V3, V4). | Only when adding new RS recordings. |

### H.3 Output layout

```
Robot_APCC/Experiments/Experiment_23/Results/<MM_DD_YY_HH_MM_SS>/
├── v2/ … v4/                      ← per-version folder
│   └── <category>/<speed>/<task>/
│       ├── trajectory_1/
│       │   ├── trajectory_1_result.csv     ← solver output (RS-format)
│       │   ├── f3_d1_report.json
│       │   └── *.png                       ← solver diagnostic plots
│       └── trajectory_1_result/
│           ├── rs_comparison_path_3d.png
│           ├── rs_comparison_joints.png
│           ├── rs_comparison_tcp_deviation*.png
│           ├── rs_comparison_metrics.json
│           ├── blend_arc_wp*_comparison.png
│           ├── blend_arc_metrics.json
│           └── rs_comparison_speed.png      ← only with --with_speed_fit
├── verification_summary/
│   ├── verification_report.json
│   ├── summary_table.txt                    ← scorecard, one block per toolpath
│   ├── position_deviation_summary.png
│   └── speed_rms_error_summary.png          ← only with --with_speed_fit
├── blend_deviation_report/
│   ├── flagged_toolpaths.json
│   └── flagged_toolpaths.txt
└── calibration/                             ← only after --phase calibrate
    ├── calibration_report.json
    ├── calibrated_values.yaml
    ├── a_tcp_calibration.png
    ├── blend_model_calibration.png
    ├── joint_limits_calibration.png
    └── calibration_offsets.png
```

### H.4 `tests/run_experiment_23_full.py` — exhaustive CLI reference

```text
positional arguments:    none

flags:
  --phase {all, run, calibrate}      default: all
       all        run every toolpath, then run --phase calibrate.
       run        only solve + compare (no calibration).
       calibrate  only run system-identification on RS data.

  -v, --verbose                       per-task wall time + solver step logs.
  --dry-run                           list tasks but execute nothing.
  --force                             re-run even if results already exist.
  --run-dir <name>                    write into Results/<name>/ instead of
                                      a fresh timestamp (for incremental runs).
  --3d_view                           open interactive matplotlib 3-D viewer per task.

  ── Test set selection (mutually exclusive) ──
  --v2_only                           run V2 toolpaths only (corner v20/v500
                                      + multi-speed straight_line).
  --v3_only                           run V3 toolpaths only (corner v50/v100/v200,
                                      5 angles × 5 zones).
  --v4_only                           run V4 toolpaths only (250 Hz RS data;
                                      straight_line v100…v3000 + corner v200/v500).
  --toolpath PATH                     run a specific toolpath CSV, a directory of
                                      CSVs, or a glob, e.g.
                                      "v3/corner/corner_90_deg_v100_z10.csv",
                                      "v4/corner",
                                      "v2/corner/corner_30_deg_*.csv".

  ── Filters (combine with the selectors above) ──
  --speed v20|v50|v100|v200|v500|v1000|v3000   substring match on speed_tag.
  --zone  z0|z1|z5|z10|z50                     substring match on zone_tag.

  ── Quality gates ──
  --blend-threshold MM                 mm threshold for the blend-arc deviation
                                      flagged-toolpaths report.  Default: 1.0.
  --speed-warn  MM_S                   RMS-speed warn line on the summary plot.  Default: 5.
  --speed-fail  MM_S                   RMS-speed fail line on the summary plot.  Default: 15.

  ── New: speed-profile fit toggle ──
  --with_speed_fit                     enable the solver-vs-RS TCP speed
                                      comparison (plots + summary columns).
                                      OFF by default.  See Part F.
```

### H.5 Examples

```bash
# Run everything except --phase calibrate, with speed-fit OFF (default).
python tests/run_experiment_23_full.py --phase run

# Same, but enable speed comparison plots and table columns.
python tests/run_experiment_23_full.py --phase run --with_speed_fit

# Run only V4 (250 Hz) corners at v200, all zones, force refresh.
python tests/run_experiment_23_full.py --v4_only --speed v200 --force

# Run a single toolpath, with verbose timing.
python tests/run_experiment_23_full.py \
    --toolpath v4/corner/corner_90_deg_v200_z10.csv --force --verbose

# A targeted glob — every 30° corner in V2, both speeds.
python tests/run_experiment_23_full.py \
    --toolpath "v2/corner/corner_30_deg_*.csv" --force

# Calibration phase only (against ALL RS CSVs found under Results - RobotStudio/).
python tests/run_experiment_23_full.py --phase calibrate

# Tighter blend deviation gate.
python tests/run_experiment_23_full.py --v2_only --blend-threshold 0.5
```

### H.6 Other useful scripts

| Script | Purpose | CLI |
|---|---|---|
| `tests/calibration_analysis.py` | Standalone calibration diagnostics on a single experiment dir. | `python tests/calibration_analysis.py <results_dir>` |

### H.7 Library entry-points (programmatic use)

If you want to call the pipeline directly from Python instead of through
the CLI:

```python
from core.blend_zone import run_feature3_d1, SpeedCalibration

result = run_feature3_d1(
    toolpath_csv="Robot_APCC/Experiments/Experiment_23/Toolpaths_And_Waypoints/v4/corner/corner_90_deg_v200_z10.csv",
    urdf_path="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf",
    config=load_batch_config("config/batch_feasibility_config.yaml"),
    output_dir="Results/programmatic/",
    robot_model_name="IRB 1300-7/1.4",
    robot_reach_m=1.4,
    velocity_limits_rad_s=np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566]),
)
print(result.feasible, result.total_arc_length_mm,
      result.speed_profile.total_duration_s)
```

For ad-hoc trajectory comparison without the full runner:

```python
from core.blend_zone import (
    verify_trajectory, generate_trajectory_comparison_plots,
    compare_blend_arcs, generate_blend_comparison_plots,
)

v = verify_trajectory(solver_csv, rs_csv, label="my_test", v_cmd_mm_s=200)
generate_trajectory_comparison_plots(solver_csv, rs_csv, output_dir,
                                     label="my_test", v_cmd_mm_s=200,
                                     with_speed_fit=False)

br = compare_blend_arcs(input_waypoint_csv, rs_csv)
generate_blend_comparison_plots(br, input_waypoint_csv, output_dir, label="my_test")
```

---

## Part I · Glossary

| Term | Meaning |
|---|---|
| `pzone_tcp` | TCP position-zone radius (mm). Endpoints of the blend arc are this distance from the programmed corner. |
| `pzone_ori`, `zone_ori` | Orientation-zone radii (mm and deg). Drive the SLERP onset. |
| Fly-by point | A waypoint with a non-`fine` zone — robot blends through it. |
| Fine point | A waypoint where the robot stops fully (`zonedata.fine = True`). |
| `shape_k` | Cubic-Bézier shape factor; `2/3` reproduces the documented ABB parabolic, `0.78` is our calibrated default. |
| `a_n_blend` | Normal-acceleration cap inside blends (mm/s²). |
| `a_accel_eff`, `a_decel_eff` | Effective trapezoidal tangential acceleration / deceleration (mm/s²). |
| `k_corner_dip` | Universal corner-dip coefficient for the servo-dynamic ceiling. |
| `T_settle_s` | Dwell time at a fine point. |
| `ApexSpd`, `ApexPos` | Worst speed-error and position-deviation in a ±300 ms window around each fly-by waypoint, spatially aligned. |
