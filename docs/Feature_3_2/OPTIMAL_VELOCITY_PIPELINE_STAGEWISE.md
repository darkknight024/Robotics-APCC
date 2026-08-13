# Optimal-Velocity Profiling Pipeline — Stage-wise Reference

> **Last updated:** 2026-08-13  
> Entry point: `tests/test_optimal_velocity_profile.py --dataset v7_cropped [--time-optimal]`  
> Implementation: `utils/optimal_velocity/runner.py` → `core/optimal_velocity/pipeline.py:run_diagnostics()`  
> Forensics: `utils/dump_velocity_trace.py` (per-variable CSV) + `utils/plot_velocity_trace.py` (multi-panel plot)  
> Geometry diagnostics: `utils/optimal_velocity/orientation_phasing.py` → `M_orientation_phasing/`  
> Module map + **known limitations:** [`FEATURE3_D2_MODULE_PIPELINE.md`](FEATURE3_D2_MODULE_PIPELINE.md) (§2 how to run, §17 limitations)  
> Plot catalog: [`plots_readme.md`](plots_readme.md)
>
> Defaults: plate-frame blend **on**, `ori_schedule_mode="abb"`, `--ds-mm 0.25`,
> `--uniform-resample-mm 0.25`, `--resid-tol-deg 0.05`, `--ceiling-smooth-mm 2.5`.
> `--se3-arc-length` is experimental / **off**. Do not pass `--no-plate-frame-blend`
> unless A/B-testing.
>
> Run-root artefacts: `run_feasibility_summary.txt`, `batch_fk_check.csv`.
> `batch_summary.txt` is **not** written.

---

## 0. Mission, Frames, Constraints, Goals

### 0.1 Objective

Given a programmed cutting toolpath (a sequence of waypoints of the **knife** in the
moving **plate** frame, each with a commanded knife-relative cut speed), generate a
time-parameterized robot trajectory that:

1. tracks the **commanded tool (knife-relative) speed** wherever physically possible,
2. never violates per-joint **velocity** and **acceleration** limits,
3. is **smooth** (no acceleration chatter / sawtooth joint velocities), and
4. finishes as fast as those constraints allow.

Success is measured against RobotStudio (RS) recordings of the same toolpath:
execution duration, TCP speed profile, joint velocities and joint accelerations.

### 0.2 The three reference frames

| Symbol | Meaning | Fixed? |
|---|---|---|
| `B` | Robot base frame | fixed |
| `P` | Plate frame — origin at the end-effector plate tip (TCP), moves with the robot | moving |
| `K` | Knife frame — a **static** knife mounted at a known pose in `B` (`config/knife_config.yaml`, `zundV1`) | fixed |

The **only moving body is the plate** (robot end effector). The knife is fixed in the
world. The cut happens where the plate sweeps the workpiece across the knife; what the
process cares about is the **knife-relative speed of the plate**, i.e. motion described
in `T_P,K` (knife pose w.r.t. the plate frame).

Transform used constantly: the lever arm from the plate tip to the knife tip, expressed
in the base frame,

```
r(s) = p_BK − p_BP(s)        [mm]
```

with `p_BK` from the knife config and `p_BP(s)` the TCP position along the path.

### 0.3 Inputs

| Input | Source | Frame | Notes |
|---|---|---|---|
| Waypoint poses `T_P,K` | toolpath CSV (headerless rows, mm + quat) | plate frame `P` | programmed path of the knife over the plate |
| Commanded speed `v_cmd` | toolpath CSV column 8 | plate frame `P` | authored knife-relative cut speed `[mm/s]` |
| Zone columns | toolpath CSV cols 9–14 | — | `pzone_tcp, pzone_ori, pzone_eax, zone_ori, zone_leax, zone_reax` (ABB TRM 1794–1799; eax stored, unused without external axes) |
| Knife pose `T_B,K` | `config/knife_config.yaml` (`zundV1`) | base frame `B` | static mount, gives `p_BK`, `R_BK` |
| Robot joint limits | robot config (`JointLimits`) | joint space | `q̇_max[j]` `[rad/s]`, `q̈_max[j]` `[rad/s²]` |
| RobotStudio recording | RS CSV export | plate frame `P` (tool frame) | ground truth: `speed_mm_per_s`, joint speeds, time |

### 0.4 Constraints

- **Joint velocity limits**: `|q̇_j(t)| ≤ q̇_max[j]` for all 6 joints, all t.
- **Joint acceleration limits**: `|q̈_j(t)| ≤ q̈_max[j]` for all 6 joints, all t.
- **Never exceed the commanded cut speed** pointwise (overshoot allowance ≤ 15%,
  modeling the controller's second-order speed governor — see §6).
- **Smoothness**: realized joint accelerations must not chatter at the limits
  (this was the sawtooth defect; RS is the reference for "smooth").

### 0.5 Goals (acceptance criteria)

| Goal | Metric |
|---|---|
| Track command | `v_tool(t)/v_cmd(t)` median ≈ 1.0, sag only where limits/governor bind |
| Feasible | `max_j |q̇_j|/q̇_max ≤ 1`, `max_j |q̈_j|/q̈_max ≤ 1` (verified a posteriori) |
| Smooth | joint-velocity roughness (RMS 2nd difference) at RS's log cadence ≈ RS's own |
| Fast | commanded-mode duration within ~5% of RS duration |
| Geometry (Fix 1–3) | `M_orientation_phasing`: pivot density ≤ ~1.15× authored, `g_min ≥ 0.15`, HIGH-risk corners = 0 |

### 0.6 Notation used throughout

| Symbol | Meaning | Units |
|---|---|---|
| `s` | path parameter (position arc, or weighted SE(3) arc — see §2) | mm |
| `s_pos` | position arc length (base-frame TCP) | mm |
| `s_plate` / `s_tool` | plate-frame arc length (knife-relative) | mm |
| `ṡ, s̈` | path speed / path acceleration (`ds/dt`, `d²s/dt²`) | mm/s, mm/s² |
| `q(s)` | joint configuration as a function of the path parameter | rad |
| `q' = dq/ds`, `q'' = d²q/ds²` | path derivatives of the joint path | rad/mm, rad/mm² |
| `g(s) = ds_plate/ds` | frame gain — plate arc per unit parameter | — |
| `v_tool` | knife-relative (plate-frame) speed = what RS logs as `speed_mm_per_s` | mm/s |
| `v_lim_joint(s)` | path-speed ceiling from joint limits | mm/s |
| `τ(s)` | authored command target in path space (`v_cmd/g`) | mm/s |
| `u = ṡ²` | TOPP integration variable | mm²/s² |
| `θ(s)` | cumulative plate reorientation | rad |
| `dθ/ds_tool` | orientation density — degrees of reorientation per mm of cut | deg/mm |

---

## 1. Stage 0 — Toolpath load and dense path generation

**Module**: `utils/optimal_velocity/toolpath_load.py:load_joint_path_from_toolpath()`

### Purpose
Convert the discrete, headerless toolpath CSV (knife poses in the plate frame +
per-waypoint commanded speed) into a **dense, uniformly sampled motion description in
the robot base frame**, ready for differentiation and joint-limit analysis.

### What happens
1. Parse the CSV → waypoint poses `T_P,K` and the commanded-speed schedule `v_cmd(s_pos)`.
2. Transform waypoints into the robot base frame (`waypoints_base`, via the `zundV1`
   calibration chain: plate frame → end-effector → base).
3. **Feature-3 corner blending**: generate dense blended samples through the waypoints
   (shortcuts corners within blend zones, exactly like a controller's zone blending).
   Orientation is scheduled on the **tool-frame cut arc** when the knife is present
   (Fix 1, §1c) — not the base-frame position arc.
4. **Forward kinematics per dense sample** → joint configurations `q_raw` (M × 6 rad),
   TCP positions `p_BP` (M × 3 mm), orientations (quaternions, wxyz).
5. Knife-relative plate positions `plate_xyz` (M × 3 mm) — the knife tip in the plate
   frame — for the frame-gain computation (§4).
6. Load the static knife pose in `B` from `config/knife_config.yaml` (`zundV1`).

### Outputs (all length M, raw/dense sampling)
`q_raw`, `poses` (p_BP + quat), `plate_xyz`, `waypoints_plate`, `waypoints_base`,
`s_cmd_mm` + `v_cmd_at_s` (commanded speed on the position-arc grid), `knife_translation_m`,
`limits`, `zone_params`, `blend_geoms`.

### Downstream use
- Stage 1 consumes `q_raw`/`poses` to build the path parameter and splines.
- §4 consumes `plate_xyz` + knife pose for the frame gain.
- §6 consumes the commanded schedule for the cap.
- `M_orientation_phasing` consumes `zone_params`/`blend_geoms`/`quat_slerp_raw` for
  the geometry gate (§9b).

---

## 1b. Uniform-arc resampling (optional, ON by default at 0.25 mm)

**Module**: `core/path_parameterization/uniform_resample.py`

### Purpose
The Feature-3 dense blend samples at a spacing that collapses inside corner blends
and stretches on straightaways. That non-uniform `Δs` leaks into the secant
ceiling's window (`h = max(window, factor · median Δs)`) and the spline weighting.
This stage resamples `(q, pose, plate)` onto a **uniform position-arc grid**
(linear position, SLERP orientation) before any differentiation.

### Waypoint diagnostics (unaffected)
Programmed waypoints are **not** the sampling grid. They are kept as a separate
index/arc map (`waypoint_arc_map`) that projects each waypoint onto the (now
uniform) solver grid by nearest TCP. Per-waypoint target / solver / RS comparison
uses that map via `np.interp` — independent of sampling density. Disable with
`--uniform-resample-mm 0`.

---

## 1c. Feature-3 orientation schedule — ABB dual-schedule blend (C³)

**Module**: `core/blend_zone/path_sampler.py:_abb_orientation_schedule()`
**Config**: `Feature3D1Config.ori_schedule_mode` — `"abb"` (default) or `"legacy"`.

### Why the old schedule was replaced
Feature-3 used to hold the orientation on either side of a fly-by and slew between the
holds (`hold–SLERP–hold`, phased on the tool-frame cut arc — the former "Fix 1"). Two
things are wrong with that against the controller it is modelling:

* **ABB never dwells.** Outside a waypoint's orientation zone the controller follows
  that segment's *stop-point* SLERP — the schedule it would use if the waypoint were a
  stop point — and only inside the zone does it cross-fade to the next segment's
  schedule. Measured on the 43 Hz recordings, RobotStudio tracks the stop-point
  schedule to **0.002–0.003°** outside the zones. A hold puts the whole `Δθ` of a
  segment into the middle of it instead, which spikes `dθ/ds` exactly where the frame
  gain is weakest.
* **A hold edge is not differentiable.** `dθ/ds` steps at each hold boundary, so the
  second derivative is impulsive there. The spline and TOPP stages downstream
  differentiate this schedule, so those steps become real artefacts.

### What the schedule is now
Each programmed segment gets an **affine** map from the path arc to its own segment
fraction, pinned to the arc stations of its two waypoints (a fly-by's station is the
apex of its blend arc — the point of the actual path closest to the programmed corner).
From those two ingredients:

* **Outside every orientation zone** — `q(s) = SLERP(q_j, q_{j+1}, f_j(s))`, the
  segment's exact stop-point schedule, advancing uniformly with path progress.
* **Inside waypoint `j`'s zone `[A, D]`** — the incoming and outgoing schedules, i.e.
  the *same* affine maps simply evaluated beyond their own segment, cross-faded by the
  C³ septic kernel `h(u) = 35u⁴ − 84u⁵ + 70u⁶ − 20u⁷`. The zone half-widths come from
  M3's `r_ori_eff` (floored at `pzone_tcp`, capped at each segment midpoint), and are
  additionally widened to cover waypoint `j`'s Bézier arc outright — see below.

### The zone must be measured in arc, and must cover the blend arc

Both boundaries live in the same units as the phase, which is **path arc in mm**:

$$ A = T_j - \min(r^{\text{in}}_j,\; \tfrac12\,\mathrm{span}_{j-1}), \qquad
   D = T_j + \min(r^{\text{out}}_j,\; \tfrac12\,\mathrm{span}_j) $$

with `span_j = T_{j+1} − T_j`. That looks pedantic and is not. The geometry offers two
length scales — arc travelled along the path, and chord between programmed corners —
and they are not interchangeable: rounding a corner makes the arc between two waypoint
stations **strictly shorter** than the chord they span. Converting a radius to a
fraction of the chord (`r/L`) and re-applying that fraction to the phase span shrinks
the zone by `span/L`, which on v7 traj_1 reached 0.93. The schedule builder therefore
no longer takes the waypoint positions as an argument at all: with no chord in scope,
the two scales cannot be mixed.

Independently of the radii, waypoint `j`'s zone is stretched to contain every sample of
`j`'s own blend arc. Those samples carry `seg_ids == j−1` with a phase past `T_j`, so
the base layer can only clamp their segment fraction to 1 — which freezes the schedule
on the corner quaternion. What that cost before the fix, on traj_1:

* 3 samples out of 8594 froze at the exit of waypoint 38's blend arc, bracketed by
  0.85° and 0.94° steps against a 0.023° neighbourhood — a 1.8° step in `θ(s)`.
* Through the ~94 mm lever arm those steps became **1.37 mm and 1.52 mm jumps in the
  base-frame position** of the dense path, against a 0.031 mm median.
* Thinning to a uniform base stride cannot repair a hole in the fine grid, so the dense
  path shipped with `ds_base ∈ [0.016, 1.52] mm` against a 0.25 mm target.
* Finite differences across those steps collapsed `‖dp/ds‖` and `‖θ'×r‖` by ~15×, so
  the reported frame gain fell to 0.027 where the authored segment gain is 0.160.
* Commanded mode then asked for `ṡ = v_cmd/g ≈ 630 mm/s`, saturated joint acceleration
  (`util_tot` = 1.000), and delivered 3.3 mm/s of cut speed where RobotStudio holds
  15.3 mm/s — the velocity notch at `s ≈ 223 mm`.

Only two of 8594 steps were wrong, and every percentile-based check in the suite passed
straight through them. That is why the grid check in §1d is scored on min/max.

### Fleet effect (v7 cropped, 18 trajectories, commanded mode)

| | before | after |
|---|---|---|
| worst dense-path base-arc step (target 0.25 mm) | 0.46–0.50 mm on 15/18 | **0.278–0.289 mm on 18/18** |
| `g_min` | — | **improved on 13/18** (traj_10 0.075 → 0.167, traj_9 0.063 → 0.096) |
| traversal time vs RS, mean abs error | 3.06 % | **0.52 %** |
| accel-transient fraction of path (mean) | 34.6 % | 30.3 % |
| RS point failures on samples eligible in both runs | 159 | **145** |

Every trajectory got faster and closer to RobotStudio; the notches were costing real
cycle time, not just plot quality. Note the two second-order consequences:

* The **benchmark got stricter**. Fewer notches means fewer accel-transient regions,
  so ~4 % more of each path becomes bench-eligible. Per-trajectory failure counts read
  worse in places (traj_9: 0/768 → 19/758) purely because samples that always disagreed
  with RS are no longer hidden behind the transient mask. Compared on the samples
  eligible in *both* runs, the fleet improves.
* Where it does regress (traj_7 / 9 / 10) the failures cluster in `s ≈ 85–92 mm` and
  have one sign: the solver now runs **~41 mm/s where RS runs ~37**. That is RS's own
  corner slowdown, which nothing in this pipeline models — see §12.

`h` and its first three derivatives vanish at both ends, and the functions being
cross-faded are the very ones used on either side, so the blend meets the base layer
with **C³ contact** at `A` and `D`; everything in between is analytic in the path arc.
The schedule is therefore safe to differentiate three times in parameter space, with
no reliance on differentiating sampled geometry.

**XYZ / `arc_lengths` are never modified** — quaternions only.

### Two design points worth recording

*Interpolating in the plate frame would give the same rotations.* With a fixed knife,
`q_PK = q_BP* ⊗ q_BK`; inversion and constant right-multiplication are isometries of
the quaternion sphere, so the geodesic between two plate poses maps onto the geodesic
between the corresponding knife-in-plate poses. Only the **phase** along that geodesic
is frame dependent.

*Phasing on the cut arc is not well posed here, which is why the path arc is used.* In
a cancellation region the knife tip barely advances and what motion it has is produced
by the very rotation being scheduled; making rotation proportional to tip travel is a
positive feedback loop. Iterating that fixed point diverges — on traj_1 the cut arc
grew 123 → 190 mm over five rounds — while path-arc phasing lands closest to the
programmed cut length (123.1 mm against an authored 120.4 mm).

### Consequences elsewhere in the pipeline
* **Step 5b (global orientation smooth) is skipped** under `ori_schedule_mode="abb"`.
  The schedule is already C³, and a global smoother perturbs orientation everywhere,
  including the regions where ABB guarantees exact stop-point tracking.
* **Step 5b2 (Fix-3 cancellation re-phase) is skipped** for the same reason: it builds
  a monotone `t(s)` numerically and anchors the fly-by quaternions, which breaks both
  the C³ construction and the stop-point regions. `ori_rephase_force_under_abb` re-
  enables it if ever needed.

### Measured effect (v7 cropped, traj_1 / traj_7 / traj_15)

| | hold–SLERP–hold | ABB C³ schedule | RobotStudio |
|---|---|---|---|
| stop-point deviation outside zones (median) | 0.64° | **0.041 / 0.042 / 0.000°** | 0.002–0.003° |
| dwell fraction near a fly-by | 0.135 | **0.010 / 0.019 / 0.040** | 0.031 / 0.048 / 0.080 |
| `dθ/ds` coefficient of variation | 0.61 | **0.65 / 0.59 / 1.00** | 1.12 / 1.16 / 1.62 |
| `dθ/ds` peak / mean | 2.53 | **2.49 / 2.20 / 3.05** | 4.82 / 3.91 / 5.38 |
| cut arc vs authored | +3.3% | **+2.3 / +0.8 / +1.3%** | −1.5 / −2.0 / −0.8% |

Orientation density is now *more* uniform than RobotStudio's on all three (RS reorients
in shorter bursts), and the schedule no longer dwells at fly-bys.

### Verification
`tests/test_orientation_schedule_continuity.py` drives the schedule builder with a
uniformly sampled synthetic corner and checks that `max|dᵏθ/dsᵏ|` for k = 1…3 is
independent of sample spacing — a break of order k shows up as an impulse that doubles
every time the grid halves. It includes a negative control: the legacy hold schedule
must fail the same bound (it does). Third-derivative continuity cannot be established
from a real toolpath, whose sampler changes stride at every blend entry.

`test_blend_arc_never_escapes_its_own_orientation_zone` pins the coverage rule with
zones deliberately narrower than the arc they blend: with the rule removed it sees a
6.57° step against a 0.007° median, and with it in place the schedule is smooth.

---

## 1d. Feature-3 dense path is built in the programmed plate frame

**Module**: `core/blend_zone/path_sampler.py:sample_blended_path_plate_frame()`
**Config**: `feature3_d1.plate_frame_blend` (default **on**; needs a knife pose)
**CLI opt-out**: `--no-plate-frame-blend`

### The frame the move is programmed in

The knife is bolted down and the plate rides on the flange, so in RAPID terms the knife
is a *stationary tool* and the plate is the work object. `MoveL` interpolates the tool
relative to the work object — `T_P_K` — not the flange pose `T_B_P`. A programmed
segment is therefore straight in the **plate** frame, its `pzone_tcp` corner cut is a
plate-frame cut, and the orientation blend is phased on the **cut arc**.

The sampler used to do all of that in the base frame and convert at the end. Two things
went wrong with that, and both landed on the same symptom.

### Symptom 1 — the tip bows, so the frame gain swings inside a segment

Interpolating `p_BP` linearly makes `p_PK` non-linear, because the lever arm from the
plate origin to the knife sweeps as the plate turns. The knife tip therefore bows off
the authored chord, and the frame gain

$$ g \;=\; \frac{ds_{\text{tool}}}{ds_{\text{base}}} $$

swings by 10–14 % *within* a single segment where the authored move holds it to
< 0.5 %. Everything measured per unit tool arc inherits that swing:

$$ \frac{d\theta}{ds_{\text{tool}}} \;=\; \frac{d\theta/ds_{\text{base}}}{g}, \qquad
   \omega \;=\; \frac{d\theta}{ds_{\text{tool}}}\, v_{\text{tool}} $$

and commanded mode inverts it a second time through `ṡ = v_cmd / g`. That is the
waypoint-frequency wobble seen in `M2_dtheta_ds_tool` and in the `|ω_BP|` panel of
`tcp_velocity_profile.png`. The giveaway that it was geometry and not the schedule: the
realised cut arc came out **longer** than the authored polyline, which corner rounding
cannot do.

### Symptom 2 — the sample grid alternates density at waypoint frequency

Blend arcs carry a forced minimum subdivision, so a 0.6 mm corner got ~30 samples at
~0.02 mm while the neighbouring straight ran at ~0.12 mm. Interquartile stride spread
inside a segment was **0.79** of the median step. Every finite difference taken across
that 5× jump — `dθ/ds`, the gain, the joint-spline knot spacing — picked up a
waypoint-frequency component with no geometric meaning.

### What the sampler does now

1. Convert the programmed waypoints to `T_P_K = T_B_P⁻¹ · T_B_K`.
2. Re-run overlap reduction, Bézier corner construction and orientation-zone
   population against **plate-frame** segment lengths.
3. Assemble straights and corners in the plate frame, on a grid `_PLATE_OVERSAMPLE = 8`
   times finer than requested.
4. Phase the §1c orientation schedule on the resulting **cut arc**.
5. Map back: `q_BP = q_BK ⊗ q_PK*`, `p_BP = p_BK − R_BP · p_PK`.
6. Thin to uniform base-arc spacing by **selecting** fine samples, never by
   interpolating between them — interpolation would flatten the C³ schedule to
   piecewise-linear and put the ripple straight back.

Because the cut arc now advances uniformly inside a segment, `dθ/ds_tool` is the
authored constant by construction rather than something that has to be filtered.

### Measured effect (v7 cropped, ds = 0.25 mm)

| | base-frame | plate-frame | RobotStudio |
|---|---|---|---|
| tip deviation from authored chord, traj_1 / 7 / 11 | 0.37 / 0.71 / 0.88 mm | **0.300 mm** (= the zone radius, i.e. corner rounding only) | 0.85 mm |
| cut arc vs authored, traj_1 / 7 / 11 | +2.3 / +0.8 / +1.4 % | **−0.3 / −0.2 / −0.3 %** | −1.5 / −2.0 / −1.7 % |
| base-arc stride spread (IQR / median) | 0.78–0.81 | **0.000–0.009** | 1.1–2.5 |
| base-arc stride spread (**max − min**) / median | 0.78–0.81 | **0.24** | — |
| pointwise `g` ripple, traj_1 / 15 | 1.19 / 0.79 | **0.23 / 0.09** | 0.24 / 0.10 |
| `M2` solver peak / authored peak, traj_1 | 1.02× | **1.00×** | — |
| `ω` ripple in the waypoint band, traj_1 / 11 / 15 | 0.173 / 0.244 / 0.162 | **0.123 / 0.177 / 0.130** | — |
| `ω_max`, traj_7 / 11 / 15 | 224 / 300 / 156 °/s | **200 / 271 / 141 °/s** | — |

`g_min` is unchanged or better on all four paths, so the change does not cost
feasibility, and the commanded-mode duration is within 0.1 % (traj_1: 4.742 s → 4.737 s).

### Verification
`tests/test_plate_frame_blend.py` drives a synthetic cut that is straight in the plate
frame with a steady plate twist, so the authored answer is known exactly. It checks
that the tip stays on the chord away from corners (0.0020 mm vs 0.213 mm base-frame),
that the base-arc stride is uniform, that `dθ/ds_tool` inside a segment is flat to < 5 %,
that the plate↔base mapping round-trips, and that the cut arc never exceeds the authored
polyline. Two of the tests carry a negative control against the base-frame sampler so
they cannot pass for the wrong reason.

The stride check is scored on **min/max against the target stride** (0.7–1.3 × `ds_mm`),
not on percentiles. It used to compare p1 to p99, and in that form it passed on a path
carrying a 6 × stride hole — two bad steps in 8594 do not move a percentile, but they do
corrupt every finite difference taken across them.

---

## 2. Stage 0b — Path parameterization

**Module**: `core/optimal_velocity/validate.py:step0_validate()`,
`core/path_parameterization/se3_arc_length.py`

### Purpose
Choose the scalar parameter `s` along which everything else (splines, ceilings, TOPP)
is differentiated and integrated. **This is a modeling choice, not physics** — and it
is one of the two places where orientation-vs-translation weighting enters.

### Math
Degenerate (zero-motion) raw samples are pruned first. Then:

**Position-only mode** (λ = 0, default when SE(3) is off):

```
s_pos[i] = Σ_{k<i} ‖p_BP[k+1] − p_BP[k]‖            [mm]
```

**Weighted SE(3) mode** (`--se3-arc-length`):

```
s[i] = Σ_{k<i} √( ‖Δp[k]‖² + λ²·Δθ[k]² )            [mm]
Δθ[k] = 2·acos(|q[k+1]·q[k]|)                        [rad]  (quaternion geodesic)
```

`λ [mm/rad]` converts orientation change to equivalent millimetres. In `auto` mode it
is estimated from the dense pose data by `resolve_lambda()` so that the rotational
contribution is commensurate with the translational spread.

Both arcs are kept: `s` (active parameter) and `s_pos` (position arc). The ratio
`dp/ds = ds_pos/ds ≤ 1` appears later whenever a *position-space* budget is applied
in the active parameter.

### Outputs
`s_mm` (active parameter, pruned), `q_kept`, `pos_kept`, `quat_kept`, `s_pos_mm`,
`dp_ds` per raw sample.

### Downstream use
The active parameter defines the domain of every spline, ceiling, and the TOPP
integration. `dp/ds` is needed in §6 to express a **base-frame** acceleration budget
while integrating in SE(3) parameter space.

---

## 3. Stage 1 — Joint-path splines

**Module**: `core/optimal_velocity/differentiation.py:step1_differentiate()`

### Purpose
Replace the noisy discrete joint samples by smooth analytic functions so that path
derivatives (which multiply the speed profile into joint rates) are well defined and
noise-free.

### Math
Per joint `j = 1..6`, an LSQ quintic spline `q_j(s)` is fit to the pruned samples with
an IK-residual tolerance; the smoothing level is auto-tuned (`_tune_lsq_spline`).
Analytic derivatives:

```
q'_j(s)  = dq_j/ds          [rad/mm]
q''_j(s) = d²q_j/ds²        [rad/mm²]
```

evaluated on a uniform grid `s_eval` (default spacing ~0.25 mm).

**Chain rule** (used everywhere downstream):

```
q̇_j(t) = q'_j(s)·ṡ                    (joint velocity)
q̈_j(t) = q'_j(s)·s̈ + q''_j(s)·ṡ²      (joint acceleration)
```

The second term, `q''·ṡ²`, is the **centripetal** term: it costs joint acceleration
even at constant path speed wherever the joint path is curved.

### Outputs
`s_eval`, `q'(s)`, `q''(s)` (N_eval × 6), the spline objects, smoothing diagnostics.

### Downstream use
§5 (ceilings) and §7 (TOPP) consume `q'`, `q''` on `s_eval`; §8 realizes joint
profiles with the same chain rule. **Verified clean**: the spline geometry carries
~±1% texture — it was exonerated as the jaggedness source (the texture was in `ṡ`).

---

## 4. Stage 3 — Frame conversion (tool frame ↔ base frame)

**Module**: `core/path_parameterization/frame_conversion.py`,
`core/path_parameterization/twist.py`, `utils/toolpath_speed_frames.py`

### Purpose
The authored speed `v_cmd` is **knife-relative** (plate frame), but the robot's joints
produce motion in the **base frame**. This stage derives the scalar, position-dependent
conversion between the two — the *frame gain*.

### The exact rigid-body identity

The knife is fixed, so `T_B,K = T_B,P · T_P,K` is constant in time. Differentiating
that constraint gives the velocity of the plate material point at the knife tip:

```
v_tip = v_BP + ω_BP × r ,     r = p_BK − p_BP
```

and the tool-frame cut speed is its magnitude:

```
v_tool = ‖v_tip‖ = ‖ ṗ_BP + ω_BP × r ‖
```

With the twist-per-parameter from pose splines (`fit_pose_twist_splines` /
`eval_pose_twist`):

```
g_spline(s) = ‖ p'(s) + θ'(s) × r(s) ‖      (spline-adjoint gain)
```

so `v_tool = g·ṡ` and the commanded path-speed target is `ṡ_target = v_cmd/g`.

**Where each term comes from** (geometry only — no RS, no physics constants):

| Symbol | Meaning | How we get it from the toolpath |
|---|---|---|
| `p(s)` | plate origin `p_BP` in base [mm] | Feature-3 dense poses (XYZ) |
| `p'(s)` | plate linear rate per unit `s` [mm/mm] | derivative of LSQ quintic position splines |
| `θ'(s)` | plate angular rate per unit `s` [rad/mm] | `θ' = 2·vec(q'⊗q̄)` from hemisphere-unwrapped quaternion splines, projected onto the unit sphere |
| `r(s)` | lever arm `p_BK − p_BP` [mm] | knife config `p_BK` (zundV1) minus `p(s)` |
| `ω_BP` | plate angular velocity [rad/s] | `θ'·ṡ` |
| `v_BP` | plate-origin velocity [mm/s] | `p'·ṡ` |

Two independent estimators exist and **disagree pointwise by ±30–60% at blend
corners** (both integrate to the same total plate arc within ~0.3%):

| Estimator | Definition | Character |
|---|---|---|
| `g_fd` | `Δs_plate/Δs` per raw step (`plate_arc_and_gain`) | faithful, but carries sub-mm FD needle texture |
| `g_spline` | adjoint norm from LSQ pose splines | smooth, misses real sub-spline-scale corner curvature |

The commanded target in path space is `τ(s) = v_cmd(s)/g(s)`; the reported tool speed
is `v_tool = g_report·ṡ`. **Capping with one estimator and reporting with the other
prints their disagreement directly onto the reported speed** — a past root cause of
the TCP-profile sawtooth. The pipeline now uses `g_spline` for *both* directions.

### Why `g` collapses (cancellation)

`g = ‖p' + θ'×r‖` is small exactly where translation and the lever-arm rotation term
**oppose each other with matching magnitude** (`p' ≈ −θ'×r`). That is not a joint-limit
phenomenon — it is pure geometry of the synthesized path. If the orientation schedule
piles `Δθ` into a short tool-arc band (high `dθ/ds_tool`), the cancellation deepens
and `g` needles appear. This is what Fix 1–3 attack upstream of TOPP.

### Outputs
`s_plate` (plate arc), `g_fd`, `g_spline` on raw and eval grids, plus the plate twist
components in base and knife frames for the comparison plots.

### Downstream use
§6 divides authored speeds by `g` to get path-space targets; §8 multiplies the solved
`ṡ` by the same `g` to report tool-frame speed.

---

## 5. Stage 2 — Joint-limit ceilings in path space

**Module**: `core/optimal_velocity/mvc_ceilings.py`

### Purpose
Translate the joint-space limits into the **maximum path speed** `v_lim_joint(s)` that
keeps every joint inside its velocity and acceleration envelopes at each point of the
path.

### Math
**Velocity ceiling** (from `|q̇_j| = |q'_j|·ṡ ≤ q̇_max[j]`):

```
v_vel(s) = min_j  q̇_max[j] / |q'_j(s)|
```

**Acceleration ceiling** (from `q̈_j = q'_j·s̈ + q''_j·ṡ² ≤ q̈_max[j]`): per node, a
bisection on `ṡ` such that the maximally-feasible `s̈` remains non-negative:

```
v_acc(s) = max { v : ∃ s̈ ≥ 0 with |q'_j s̈ + q''_j v²| ≤ q̈_max[j] ∀j }
```

**Secant ceiling**: joint-space acceleration feasibility evaluated on *raw-sample
secants* over a window, which recovers corner curvature the smoothing spline
cannot represent:

```
h = max(window_mm, sample_factor · median Δs)
v ≤ sqrt( q̈_max_j · h² / |q(s+h)−2q(s)+q(s−h)|_j )   (min over joints)
```

then median-filtered over `median_windows` half-windows so single-sample IK
jitter cannot punch notches into `v_lim`.  Defaults (after the denoise pass):
`window_mm = 2.5`, `sample_factor = 5`, `median_windows = 2`.  Disable with
`--no-secant-cap`.

**Combination and smoothing**:

```
v_lim_joint(s) = smooth_min( min(v_vel, v_acc, v_secant) )
```

`smooth_ceiling_min_preserving()` erodes → low-passes → clamps to never exceed the
raw ceiling, removing mm-scale binding-joint switching texture that TOPP would
otherwise bang in and out of (safe by construction: result ≤ raw ceiling everywhere).

### Outputs
`v_vel`, `v_acc`, `v_secant`, `v_lim_joint` (per node on `s_eval` and on the dense
MVC grid), plus `binding_joint`/`binding_kind` diagnostics.

### Downstream use
`v_lim_joint` is the hard ceiling in §7; the MVC-grid version enforces cell-min
conservatism so TOPP can never tunnel through a sub-grid-sample constraint.

---

## 6. Stage 4 — Command target and the speed governor

**Module**: `core/optimal_velocity/pipeline.py` (`_segment_zoh_target_raw`,
`_governor_rate_limit`)

### Purpose
Express the authored knife-relative speed schedule as a **path-space target** the
profiler may approach but (apart from the bounded governor allowance) not exceed —
and make that target *trackable* by a real controller.

### Cap construction (three modes, default `pointwise_spline`)

| Mode | Target | Character |
|---|---|---|
| `segment` | `τ_seg = v_cmd,seg · L_param,seg / L_plate,seg` (ZOH per programmed segment) | staircase: step at every waypoint → **sawtooth joint velocities (defect)** |
| `pointwise` | `τ(s) = v_cmd(s)/g_fd(s)` | FD needles |
| `pointwise_spline` | `τ(s) = v_cmd(s)/g_spline(s)` | continuous; keeps real needle geometry |

### The governor (model of the controller's second-order speed governor)

A pure pointwise target still swings several-fold within a few millimetres at
gain-needle valleys; chasing it at full joint-accel capability is what produced the
residual chatter. The governor shapes the target with three stages:

1. **Short centred low-pass** (`smooth_mm = 1.5 mm`) — rounds accel corners;
2. **Pointwise clamp at `1.15 ×` the raw target** — the smoothing may lift the target
   inside needle valleys by at most 15% (RS's own logs run 1–3% above command
   transiently);
3. **Bounded-acceleration rate limit** in `u = v_base²` space:

```
forward pass:   u[i+1] = min(u_raw[i+1], u[i] + 2·a·Δs_pos)
backward pass:  u[i]   = min(u_fwd[i],  u[i+1] + 2·a·Δs_pos)
ṡ_gov = √u_gov / (dp/ds)          (back into the active parameter)
```

with the **physical** budget `cmd_accel_max = 8000 mm/s²` applied in *base-frame*
position space (under SE(3), dividing by `dp/ds` back-converts; applying the budget
per SE(3) parameter would silently shrink it several-fold exactly inside pivots).
`--cmd-accel-max 0` disables the governor so the geometry-limited profile is visible.

Sag below command occurs **only** where the raw target moves faster than this budget
allows, or where joint ceilings bind. Calibrated so the sag-depth distribution matches
RS's own (p5 ≈ 0.75–0.79 vs RS 0.77).

### Outputs
`v_target_path` (post-governor target), `v_cmd_for_cap` (per eval node), MVC variant.

### Downstream use
§7 computes the effective ceiling `v_lim = min(v_lim_joint, governed cap)`; §8 reports
`v_target_path` in the trace so every sag can be attributed to *governor* vs *joint
ceiling* vs *RS zone cap*.

---

## 7. Stage 5 — Time-optimal path parameterization (TOPP)

**Module**: `core/optimal_velocity/heun_topp.py:step3_time_optimal()`

### Purpose
Find the fastest feasible speed profile along the fixed path, subject to the combined
ceiling and joint-acceleration feasibility at every point.

### Math
Integrated in `u = ṡ²` (path energy), for which dynamics are linear:
`du/ds = 2·s̈`. Forward pass accelerates as hard as joint limits allow without
exceeding the ceiling; backward pass enforces deceleration feasibility into every
constraint. Heun (trapezoidal) integration per cell; cell-min MVC conservatism for
the joint ceiling; optional jerk slew (`path_jerk_max`, default off — it created
artificial dips).

```
u_fwd[i+1] = min( u[i] + (A0+A1)·Δs ,  ceiling[i+1] )
A chosen so that |q'·s̈ + q''·u| ≤ q̈_max  at both cell ends
```

### Outputs
`ṡ(t) = s_dot_path`, `s̈(t)`, time map `t(s) = ∫ ds/ṡ`, duration.

### Downstream use
§8 realizes all physical profiles from `(q'(s), q''(s), ṡ, s̈)`.

---

## 8. Stage 6 — Realization, frame reporting, RS comparison

**Module**: `core/optimal_velocity/pipeline.py` (reporting tail),
`utils/optimal_velocity/plotting.py`, `utils/optimal_velocity/benchmarking.py`

### Math
```
q̇(t) = q'(s(t))·ṡ(t)                       [rad/s]
q̈(t) = q'(s(t))·s̈(t) + q''(s(t))·ṡ(t)²     [rad/s²]
v_tool(t) = g_report(s(t))·ṡ(t)            [mm/s]   (same gain used for the cap)
```

Plate twist components are reported in both frames: base frame `(v_BP, ω_BP)` and
knife frame via the SE(3) adjoint. Region masks: `cruise` (on ceiling), `transient`
(accel/decel), `boundary`, plus RS-side transient exclusion for fair benchmarking.

### Frame invariance of the twist (important)

A rigid-body 6-vector twist is **not** frame-invariant as a whole:

| Component | Frame behaviour | Why |
|---|---|---|
| Linear `\|v\|` | **frame-dependent** | picks up the lever-arm term `ω×r`; `\|v_knife\| = g·ṡ` ≠ `\|v_base\|` |
| Angular `\|ω\|` | **frame-invariant** | a body has one rotation rate regardless of the fixed observer frame |

So `stage7_twist.png` legitimately shows **different linear magnitudes** in the two
frames (their ratio is exactly `g(s)`) and **identical angular magnitude** (plotted
twice — base and tool — only to make the invariance explicit). For `traj_1` the
plate swings about the knife on a long lever arm, so `\|v_base\|` (≈400 mm/s,
dominated by `ω×r`) runs well above the knife-relative cut speed `\|v_knife\|`
(≈120 mm/s). This is correct physics, not a unit bug.

### Stage-4 vs Stage-7 y-axis (important)

These two panels intentionally use **different quantities/units on the y-axis** and
must not be expected to line up:

| Plot | y-axis | Space |
|---|---|---|
| `stage4_ceilings.png` | `v_vel / v_acc / v_secant / v_lim_joint` | **path space** `[mm/s of s_act]` — how fast the *parameter* may advance |
| `stage7_tool_speed.png` | `v_tool = g_report·ṡ` | **tool (knife) frame** `[mm/s]` — the realized knife-relative cut speed |

They are related by the frame gain: `v_tool(s) = g(s)·ṡ(s)` and the path ceiling
`v_lim_joint` bounds `ṡ`. A pointwise frame conversion (`×g(s)`) maps Stage-4
ceilings into tool space. For **optimal / constant** modes there is no authored
`v_cmd`, so Stage 7 shows only the physical result of running at the joint ceiling;
for **commanded** mode the tool-frame `v_cmd` cap and RS overlay are drawn on the
same tool-frame axis.

### Uniform-arc sampling (important)

With SE(3) on, the solver integrates over `s_eval` which is **uniform in `s_act`
(Δs_act cv = 0)** — confirmed in the dumps. Consequently `dq/ds_act` and `q̇` are
sampled uniformly in the variable the math is done in. `s_pos` is *intentionally
non-uniform* under SE(3) (cv ≈ 0.23): that is the re-weighting doing its job, not a
resampling failure. The residual "ripple" in `q̇` traces is **real bang-bang
switching between joint acceleration limits** (visible as saturated plateaus in
`stage7_qddot_vs_s.png`), which uniform sampling cannot remove — it is physics, and
it is *why* the tool speed dips at reorientation corners.

### Modes (three `run_diagnostics` calls per toolpath)

| Mode | Ceiling | Meaning |
|---|---|---|
| `commanded` | `min(v_lim_joint, governed v_cmd cap, governed RS zone cap)` | emulate RS |
| `constant` | `min(v_lim_joint, governed v_const)` | best constant authored speed |
| `time_optimal` | `v_lim_joint` (+ raw RS zone cap) | joint-limit-bound minimum time |

### Outputs
Per-mode result dirs with `optimal_velocity_profile_report.json`, `summary.txt`,
`tcp_velocity_profile.png`, `transient_decision_variables.csv`, and the figure sets
`A_geometry_spline … K_base_frame_command` (incl. `G3/G4` joint velocity/acceleration
vs RS, `D2/D3` joint profiles vs time, twist-component plots in both frames).

---

## 9. Forensic instrumentation — every variable, stamped

### `utils/dump_velocity_trace.py`
Runs the pipeline in-memory for all three modes and writes, per mode, an 84-column
CSV where **every intermediate variable** above is present on the eval grid, stamped
by `t_s` (time), `s_param_mm` (position arc), `s_act_mm` (active SE(3) parameter),
`s_plate_mm` (plate arc), `seg_id`, and `wp_near_idx`:

- parameters: `s_param_mm`, `s_act_mm`, `s_plate_mm`, `theta_ori_rad`, `dtheta_ds_rad_mm`
- gains: `g_fd`, `g_spline`, `g_report`, `g_seg_mean`, and the adjoint decomposition
  (`dp_ds_norm`, `lever_norm`, `align_cos`, `theta_ds_norm_rad_mm`)
- ceilings: `v_vel_path`, `v_acc_path`, `v_secant_path`, `v_lim_joint_path_raw`,
  `v_lim_joint_path_smooth`, plus per-joint `v_vel_j*` / `v_acc_iso_j*`
- command chain: `v_cmd_tool_mm_s`, `zoh_target_path_mm_s`, `v_target_path_mm_s`
  (post-governor), `v_cap_final_path_mm_s`
- TOPP: `s_dot_mm_s`, `s_ddot_mm_s2`, `u_mm2_s2`
- realized: `v_tcp_tool_mm_s` (+ spline-gain variant), `accel_tool_mm_s2`,
  `q1..q6`, `dqds1..6`, `d2qds2_1..6`, `qdot1..6`, `qddot1..6`
- attribution: `binding_joint`, `binding_kind`, `cruise`, `transient`, `boundary`,
  `accel_transient`, `qdot_util(_joint)`, `qddot_util(_joint)`, `path_jerk_util`

Plus `trace_raw_samples.csv` (raw parameterization samples) and `trace_analysis.txt`
(roughness, variance attribution, dominant texture wavelength, binding-joint switches).

### `utils/plot_velocity_trace.py`
Renders the whole chain as six stacked panels from a trace CSV — reported tool speed,
gain estimators, gain decomposition, path speed/targets/ceilings, orientation rate &
path acceleration, joint ceilings — with programmed-segment boundaries marked and
**per-dip attribution** (each dip in the reported speed is assigned to the first
upstream variable that moved: joint velocity limit, joint acceleration limit, secant,
governor budget, or unexplained).

---

## 9b. Geometry gate — `M_orientation_phasing` (Fix 1–3 acceptance)

**Module**: `utils/optimal_velocity/orientation_phasing.py`

Every `process_one_toolpath` run writes `M_orientation_phasing/` at the toolpath
folder. It compares orientation phasing along the **tool-frame arc** across three
sources that must agree on the same task:

1. authored toolpath waypoints (`T_P,K`),
2. Feature-3 dense blended path (post Step-5b smooth + Fix-3 re-phase),
3. RobotStudio executed recording.

| # | Artifact | What it proves |
|---|---|---|
| M1 | `M1_theta_vs_tool_arc.png/.csv` | θ(s_tool) overlay + residual vs authored |
| M2 | `M2_dtheta_ds_tool.png/.csv` | Orientation density; WP verticals + `r_ori_eff` bands |
| M3 | `M3_gain_vs_tool_arc.png/.csv` | Solver `g_spline` vs RS samplewise gain |
| M4 | `M4_cancellation_isa.png/.csv` | cos∠(p′,−θ′×r), \|θ′×r\|/\|p′\|, ISA→knife distance |
| M5 | `M5_step5b_before_after.png` + JSON | Pre/post smooth density (each on its own tip arc) + knots |
| M6 | `M6_per_corner_table.csv` + `summary.txt` | Per-WP density ratio, `g_min`, `r_ori_eff`, risk |

**Acceptance targets (traj_7-class):** pivot peak density ≤ ~1.15× authored,
`g_min ≥ 0.15` (or ≥ 0.85 × RS), M1 residual p95 ≤ ~2°, HIGH-risk corners = 0.

---

## 10. Feature-3 orientation fixes (Fix 1–3, superseded by §1c)

These are **solver-geometry** changes upstream of TOPP. They exist because the
commanded profile was hitting `g` needles that joint limits did not explain: RS
tracked the authored orientation density (~1.00×) while our dense path spiked to
2.13×, collapsing `g` and forcing `v_cmd/g` (and joint rates) far above what RS
needed. All three fixes leave **XYZ / `arc_lengths` unchanged** — quaternions only.

> **Status.** Fixes 1–3 were built on top of the `hold–SLERP–hold` schedule and are
> **inactive by default** now that the ABB dual-schedule blend (§1c) is the default:
> Fix 1's schedule is what §1c replaces, and Fixes 2 and 3 are skipped under it
> (each perturbs the regions where ABB tracks stop-point SLERP exactly, and neither
> is C³). They remain reachable via `ori_schedule_mode="legacy"` and
> `ori_rephase_force_under_abb`, and are documented here because the reasoning
> behind them — cancellation, density, the tool/base arc distinction — still
> explains why the schedule matters at all.

### Fix 1 — tool-arc orientation schedule (superseded by §1c)
Synchronize orientation to the **tool-frame cut arc** (ABB rule) instead of the
base-frame position arc, honoring `r_ori_eff` as hold–SLERP–hold.

### Fix 2 — shape-preserving Step 5b smooth
**Module**: `core/blend_zone/orientation_smooth.py`

The global rotvec smooth now uses a finer knot floor (`2 mm`, `L/80`) so it can track
pivot-scale features, and a **global peak density guard** (`max |dr/ds|_smooth ≤
osc_factor × max |dr/ds|_raw`) instead of a samplewise envelope. A strict pointwise
envelope forced the knot search back to ~L/8 (~30 mm) and smeared the pivot phasing
Fix 1 had just repaired.

### Fix 3 — cancellation / ISA re-phase
**Module**: `core/blend_zone/orientation_rephase.py`

After Fix 1+2, residual `g_spline` needles remain where `p' ≈ −θ'×r`. Fix 3 lifts the
**same spline-adjoint `g` TOPP uses** by re-timing orientation **inside programmed
segments** (waypoint quaternions stay anchors):

1. Evaluate cancellation metrics with the same pose-twist splines TOPP uses.
2. While `min g < g_floor` (default 0.15): locate the worst sample's segment, propose
   WP-anchored schedules (anti-cancellation weights, ease-in/out, smoothstep) on that
   segment ± neighbors, and accept the first candidate that raises **global** `min g`
   without exceeding a density cap (2.5× uniform SLERP).
3. Optional endpoint-SLERP fallback with a hard max waypoint orientation-error budget
   (`max_wp_err_deg`, default 2.5°).

Config: `feature3_d1.ori_rephase_enabled` (default True), `ori_rephase_g_floor`,
`ori_rephase_max_rounds`, `ori_rephase_max_wp_err_deg`,
`ori_rephase_allow_endpoint_fallback`.

### Measured effect (commanded mode, `--cmd-accel-max 0`)

| | traj_7 before → after Fix 1–3 | traj_1 after Fix 1–3 |
|---|---|---|
| Pivot peak `dθ/ds_tool` | 2.13× → **1.37× authored** | **1.02×** (≈ RS 1.01×) |
| `g_min` | 0.065 → **0.159** (RS 0.146) | **0.116** (RS 0.099) |
| HIGH-risk corners (M6) | 7 → **0** | 4 (local windows; global g already > RS) |
| `ω_max` | 359 → **236 deg/s** | **291 deg/s** |
| max \|q̇\| util | 0.52 → **0.32** | **0.34** |

---

## 11. Known modeling layers & limitations

Full catalog (export gap, pure-rotation `s`, unmodeled accel, `ω` ringing, …):
[`FEATURE3_D2_MODULE_PIPELINE.md`](FEATURE3_D2_MODULE_PIPELINE.md) **§17**.

These are **choices, not measurements** — the places to question first when results
disappoint:

1. **Path parameter** (§2): production is **position-only** `s = Σ‖Δp‖`. Weighted
   SE(3) (`--se3-arc-length`) is experimental. Consecutive same-position /
   different-orientation waypoints collapse `Δs → 0` on the default.
2. **Frame gain estimator** (§4): `g_fd` vs `g_spline` disagree ±30–60% pointwise at
   blend corners; both are only as faithful as our Feature-3 blend is to RS's own
   zone blending.
3. **Governor budget & overshoot** (§6): `cmd_accel_max = 8000 mm/s²`,
   `1.5 mm` low-pass, `1.15×` clamp — calibrated to RS's *observed* ramps on one
   dataset family; they are the controller model, not robot physics.
4. **Ceiling smoothing window** (§5): `2.5 mm` min-preserving — trades a little
   conservatism for chatter-free ceilings.
5. **Uniform resample spacing** (§1b): default `0.25 mm`; set to 0 to keep the
   raw Feature-3 sampling. Resample is on the **position** arc, not SE(3).
6. **Secant denoise** (§5): `window / sample_factor / median_windows`
   (defaults `2.5 / 5 / 2`) — kill IK-noise notches without erasing real
   corner valleys.
7. **Fix-3 re-phase** (§10): re-times orientation within segments to lift `g`; it can
   trade a small waypoint-orientation residual (≤ `max_wp_err_deg`) and a slightly
   longer tool arc for a higher `g_min`. It does **not** change XYZ, and it stops at
   `g_floor` — it is not a global trajectory optimizer. Inactive under the ABB
   schedule.
8. **Frame the position path is straight in** — *fixed*, see §1d. The sampler used to
   build straight lines and Bézier blends in the base frame; it now builds them in the
   programmed plate frame. The residual limitation is only that the knife pose
   `T_B_K` must be calibrated: without one the solver falls back to base-frame blending
   and the wobble described in §1d returns.
9. **Ceiling smoothing widens a narrow ceiling dip into a speed trench** (§5). The
   min-preserving smoother erodes over a 2.5 mm window and then averages, so a
   two-sample acceleration ceiling shows up as a ~3 mm plateau at ~0.5× the local
   ceiling. It can never raise a ceiling, so it is always feasible, but it is now the
   largest remaining source of commanded-mode tracking loss on v7 traj_1: the worst
   `v*/v_cmd` is 0.65 at `s = 226 mm`, attributed to a secant-acceleration minimum
   carried in from a neighbouring sample. `--ceiling-smooth-mm` controls it.
10. **Joint `q̈` limits** are v9 RS p99 peaks, not ABB catalog; bound =
    `min(accel, decel)`. No torque TOPP. `--path-jerk-max` default 0.
11. **TCP angular ringing** on `tcp_velocity_profile.png` panel 2 is residual
    `dθ/ds` / `q'` texture under `ṡ = v_cmd/g`, not the old within-segment `g`
    wobble (that was plate-frame blend + zone-arc coverage).
12. **No playback export** of `q(t)`, `q̇(t)` — computed in `ProfileResult`,
    plotted in groups D/G, not written as RAPID / joint CSV.
13. **RS corner derate** unmodeled (traj_7/9/10 `s ≈ 85–92 mm`: solver ~41 vs
    RS ~37 mm/s).
14. **No servo lag** (~30 ms RS first-order). ABB orientation is a reconstructed
    C³ septic, not firmware.

Everything else (splines, joint ceilings, TOPP) is deterministic math with verified
implementations.

---

## 12. Next steps

See also the full limitation catalog:
[`FEATURE3_D2_MODULE_PIPELINE.md`](FEATURE3_D2_MODULE_PIPELINE.md) §17
(pure-rotation `s`, `ω` ringing, non-catalog `q̈`, no `q(t)` export, …).

- **Model RS's corner slowdown** — now the largest remaining disagreement, and the only
  one with a consistent sign. On traj_7 / 9 / 10 the solver holds ~41 mm/s through
  `s ≈ 85–92 mm` where RobotStudio holds ~37; every other failing sample on the fleet is
  scattered noise. Nothing in this pipeline reduces speed for a corner *as such* — the
  profile only slows when a joint ceiling binds — whereas ABB derates on zone radius and
  turn angle. Extract the implied derate from the recordings as a function of
  (`r_ori_eff`, turn angle) before adding any model.
- **Fleet validation**: confirm M6 HIGH-risk → 0 and `g_min` ≥ RS across the set. `g_min`
  now beats the previous run on 13/18 (§1c) but has not been checked against RS's own.
- **Ceiling-smoothing width**: now the top commanded-mode tracking loss (§11.9). Try
  scaling the window with the local ceiling gradient, or eroding only across genuine
  multi-sample minima, so a two-sample notch stops costing 3 mm of cut speed.
- **Parameterize by something that never stalls**: `ṡ = v_cmd/g` is exact — the report
  divides by the same `g` it multiplied by — but it is badly conditioned, with `g`
  spanning 0.09–1.23 and a path-space target reaching 425 mm/s behind a `1e4` clip.
  A relative error in `g` lands 1:1 on the target. Parameterizing by the cut arc would
  remove the division (`ṡ_tool ≤ v_cmd`) and move `g` onto the constraint side where a
  small `g` correctly tightens the joint bound, but the cut arc stalls during pure
  reorientation about the contact point. `--se3-arc-length` stalls in neither factor and
  is the parameterization to evaluate here.
- **Fix-3 tuning**: expose `g_floor` / density cap / WP-error budget per dataset;
  consider a global (multi-segment) re-phase only if single-segment passes plateau.
- **Governor re-calibration**: with geometry fixed, re-fit `cmd_accel_max` against RS
  so the governor models the controller rather than hiding geometry.
- **Acceptance test**: thin pytest that recomputes M metrics after Feature-3 on a
  golden traj_7/13 and fails on regression.
