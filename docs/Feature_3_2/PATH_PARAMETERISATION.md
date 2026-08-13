# Path Parameterisation for Time-Optimal Velocity Profiling

*A pragmatic comparison of four parameterisations for the joint-feasible TCP velocity profiler on the ABB IRB-1300 siping stack.*

> **Last updated:** 2026-08-13 — design comparison. Production default is still
> position-only `s = Σ‖Δp‖`; Approach B is `--se3-arc-length` (off).
> Limitations of the running stack:
> [`FEATURE3_D2_MODULE_PIPELINE.md`](FEATURE3_D2_MODULE_PIPELINE.md) §17.

---

## TL;DR

We run TOPP on our blended-arc joint path `q*(s)` for the siping toolpaths. All hardware limits (joint position, velocity, acceleration, jerk) live in **time**, but TOPP naturally works in a **path parameter** `s`, converting between the two via the chain rule. The choice of `s` shapes every downstream quantity we report to Nike and every conversion we owe.

We considered four options:

- **A. Joint-space arc-length** — `s = Σ ‖Δq‖`. Cleanest math: `‖dq/ds‖ = 1` by construction, so the derivatives that feed the velocity-limit curve are perfectly conditioned. Costs a Jacobian evaluation for every reporting quantity.
- **B. Weighted SE(3) arc-length** — `s = √(‖Δp‖² + λ²·Δθ²)` with `λ` estimated per segment from the path itself. Reports linear TCP velocity directly from pose ratios, no Jacobian required.
- **C. Screw / se(3) Lie-algebra arc-length** — `s = ‖log(T_k⁻¹ T_{k+1})‖`. Elegant Lie-theoretic formulation, but carries an implicit `λ = 1 mm/rad` — a poor fit for our robot's ~158 mm TCP lever arm — and needs Adjoint machinery for every reporting conversion.
- **D. Decoupled position + orientation** — parameterise each independently, then merge. Reveals whether a bottleneck is translation-driven or rotation-driven, but the merged second derivative has cross-terms and master-parameter switching that add engineering surface without accuracy gain.

**We designed B**, with `λ` estimated per segment. It inherits directly from the displacement-space TOPP we started with in Feature 3.1, gives us `v_tcp_linear` in mm/s from pose ratios (no Jacobian), handles the pure-rotation edge case gracefully, and extends cleanly if we later need to profile angular velocity too.

> **Production status (2026-08-13).** The running profiler still parameterises
> by **position-only** base-TCP arc `s = Σ‖Δp‖` (`core/path_parameterization/position_arc.py`).
> Approach B is implemented behind `--se3-arc-length` and is **off** by default.
> Pure-rotation segments (`Δp = 0`, `Δθ ≠ 0`) therefore still collapse `Δs → 0`
> unless that flag is set. See §12 and
> [`FEATURE3_D2_MODULE_PIPELINE.md`](FEATURE3_D2_MODULE_PIPELINE.md) §17.1.

This document walks through the math, quantifies the trade-offs, and calls out where B will fail so we know when to switch.

---

## Table of Contents

1. [The Parameterisation Problem](#1-the-parameterisation-problem)
2. [The Universal Chain Rule](#2-the-universal-chain-rule)
3. [Approach A — Joint-Space Arc-Length](#3-approach-a--joint-space-arc-length)
4. [Approach B — Weighted SE(3) Arc-Length](#4-approach-b--weighted-se3-arc-length)
5. [Approach C — Screw / se(3) Lie-Algebra Arc-Length](#5-approach-c--screw--se3-lie-algebra-arc-length)
6. [Approach D — Decoupled Position + Orientation](#6-approach-d--decoupled-position--orientation)
7. [Comparison Table](#7-comparison-table)
8. [Which Fits Siping](#8-which-fits-siping)
9. [A vs B — Head to Head](#9-a-vs-b--head-to-head)
10. [Why We Chose B](#10-why-we-chose-b)
11. [Known Limitations of B](#11-known-limitations-of-b)
12. [Special Case — Successive Waypoints, Same Position, Different Orientation](#12-special-case--successive-waypoints-same-position-different-orientation)
13. [Alternatives Considered But Rejected](#13-alternatives-considered-but-rejected)

---

## 1. The Parameterisation Problem

We track the tooltip of the end-effector fixture — the TCP — and it is offset from J6 by `[99.8, 123, 83] mm`, so the tooltip sits roughly 158 mm from the J6 axis. Every motion command is expressed as a tooltip pose in SE(3). The velocity profiler receives a blended-arc geometric path in SE(3), runs IK with continuous EAIK branch selection to obtain `q*(s)`, and must decide the time-optimal timing law `s(t)` that respects all hardware limits.

Every hardware limit lives in **time**:

$$|\dot q_j| \le \dot q_{j,\max}, \qquad |\ddot q_j| \le \ddot q_{j,\max}, \qquad |\dddot q_j| \le \dddot q_{j,\max}$$

But the path itself is a function of a **spatial** parameter `s`. TOPP bridges the two via the chain rule (see §2 below), which is standard [1], [2]. The subtlety is that the chain rule leaves the **choice of `s`** free: any strictly increasing parameter along the path is admissible. Position arc-length is the historical default, but it isn't the only option, and it collapses for motions that reorient the tool without translating it. Since our TCP is offset from J6, we can command exactly that — hold the tooltip fixed and rotate the tool around it — which makes the choice a live question, not a theoretical one.

The four candidates differ in:

- **What `s` measures** — pure joint motion, position-weighted SE(3) motion, screw magnitude, or decoupled position/orientation.
- **How well conditioned `dq/ds` and `d²q/ds²` are.** These feed the velocity-limit curve; noisy or spiky derivatives mean noisy or spiky speed ceilings.
- **What machinery each reporting quantity needs.** Feasibility at `v_cmd`, best constant speed, `v_tcp` linear, `ω_tcp` angular, and the RS-comparison plot in millimetres all involve converting from `ṡ` back to task-space, and the cost varies dramatically.

---

## 2. The Universal Chain Rule

For any strictly monotone parameter `τ` along the path, standard TOPP-style manipulations [1, Ch. 9], [2, Ch. 4] give:

$$\dot q_j = \frac{dq_j}{d\tau}\,\dot\tau \tag{1}$$

$$\ddot q_j = \frac{dq_j}{d\tau}\,\ddot\tau + \frac{d^2 q_j}{d\tau^2}\,\dot\tau^{\,2} \tag{2}$$

$$\dddot q_j = \frac{dq_j}{d\tau}\,\dddot\tau + 3\,\frac{d^2 q_j}{d\tau^2}\,\dot\tau\,\ddot\tau + \frac{d^3 q_j}{d\tau^3}\,\dot\tau^{\,3} \tag{3}$$

The path derivatives `dq/dτ`, `d²q/dτ²`, `d³q/dτ³` are **fixed geometric properties**; the timing terms `ṡ`, `s̈`, `s⃛` are what TOPP chooses. Everything in the four approaches below is a specific choice of what `τ` means and what units the derivatives carry.

---

## 3. Approach A — Joint-Space Arc-Length

### Definition

$$\Delta \tau_A^{[k]} = \left\| q^{[k+1]} - q^{[k]} \right\|_2, \qquad \tau_A = \sum \Delta\tau_A^{[k]} \tag{4}$$

Units of `τ_A` are radians (a joint-space distance).

### Key property

By construction, the joint-space tangent is unit-norm:

$$\left\| \frac{dq}{d\tau_A} \right\|_2 = 1 \quad \text{everywhere} \tag{5}$$

Each component `|dq_j/dτ_A| ∈ [0, 1]`. Derivatives are always well conditioned — no spikes, no zero-denominator pathologies, and completely immune to waypoint-density variation.

### Velocity ceiling

From (1) with (5):

$$\dot\tau_A \le \frac{\dot q_{j,\max}}{|dq_j/d\tau_A|}, \qquad v_{\mathrm{vel},A} = \min_j \frac{\dot q_{j,\max}}{|dq_j/d\tau_A|} \quad [\mathrm{rad}/\mathrm{s}] \tag{6}$$

Because `|dq_j/dτ_A| ≤ 1`, we have `v_{vel,A} ≥ max_j q̇_{j,max}`. Robust by construction.

### Acceleration ceiling (Bobrow / Pfeiffer–Johanni)

Let `c_j = dq_j/dτ_A`, `h_j = d²q_j/dτ_A²`, `u = ṡ²`. From (2):

$$L_j(u) = \frac{-\ddot q_{j,\max} - h_j\,u}{c_j}, \qquad U_j(u) = \frac{\ddot q_{j,\max} - h_j\,u}{c_j} \tag{7}$$

(swap roles when `c_j < 0`). A candidate `u` is feasible when

$$\max_j L_j(u) \;\le\; \min_j U_j(u) \tag{8}$$

which is the standard velocity-limit-curve condition [3, Bobrow 1985], [4, Pfeiffer & Johanni 1987]. Since each `L_j` and `U_j` is linear in `u`, both envelopes are piecewise linear, and the largest feasible `u` is where some pair `L_i = U_j` crosses. This can be found as a **1-D linear program over the O(n²) candidate joint-pair intersections** — 36 for a 6R arm — which is cleaner and exacter than bisection.

Then `v_{accel,A} = √u_max` and `v_{lim,A} = min(v_{vel,A}, v_{accel,A})`.

### Deliverable conversions

Each of these needs the geometric Jacobian evaluated along the path.

$$v_{tcp} = \left\| J_p(q)\,\frac{dq}{d\tau_A} \right\| \cdot \dot\tau_A \quad [\mathrm{mm}/\mathrm{s}] \tag{9}$$

$$\omega_{tcp} = \left\| J_\omega(q)\,\frac{dq}{d\tau_A} \right\| \cdot \dot\tau_A \quad [\mathrm{rad}/\mathrm{s}] \tag{10}$$

The `v_cmd` constraint (spatially varying):

$$\dot\tau_A \le \frac{v_{cmd}}{\left\| J_p(q)\,dq/d\tau_A \right\|} \tag{11}$$

Best constant TCP linear speed:

$$v_{flat} = \min_\tau \Big( v_{\mathrm{lim},A}(\tau) \cdot \left\| J_p(q)\,dq/d\tau_A \right\| \Big) \tag{12}$$

RS x-axis (position arc-length):

$$s_{pos}(\tau_A) = \int_0^{\tau_A} \left\| J_p(q)\,dq/d\tau_A \right\| \, d\tau_A \tag{13}$$

**Every reporting deliverable except the time-optimal duration itself needs the Jacobian.** Pinocchio gives it to us in constant time, but every evaluation is a new place for a bug or FK-consistency drift.

### Literature

- Kunz & Stilman [5] apply TOPP in joint-space parameterisation and note the well-conditioned derivatives it produces.
- Pham's TOPP-RA formulation [6] is parameterisation-agnostic; joint-space is a valid choice.
- Verscheure et al. [7] present the convex `u = ṡ²` formulation used above.

---

## 4. Approach B — Weighted SE(3) Arc-Length

### Definition

$$\Delta \tau_B^{[k]} = \sqrt{ \left\| \Delta p_{tcp}^{[k]} \right\|^2 + \lambda^2 \, \Delta\theta^{[k]\,2} } \tag{14}$$

with `‖Δp_tcp‖` the Euclidean position change (mm), `Δθ = 2·arccos(|q_k · q_{k+1}|)` the geodesic rotation on `SO(3)` (rad), and `λ` (mm/rad) a scale factor. Units of `τ_B` are millimetres of blended SE(3) travel.

### Estimating `λ` per segment

Rather than pick a constant, we estimate `λ` from each segment's data. The physically meaningful choice is the median lever arm:

$$\lambda = \operatorname{median}_{\{k:\, \Delta\theta^{[k]} > \varepsilon\}} \frac{\| \Delta p_{tcp}^{[k]} \|}{\Delta\theta^{[k]}} \quad [\mathrm{mm}/\mathrm{rad}] \tag{15}$$

For our siping paths this comes out roughly 80–160 mm/rad — close to the tooltip's distance from the wrist axes, which is what we'd expect from the geometry.

### Chain-rule quantities

`dq/dτ_B` is not unit-norm; it varies along the path. Otherwise the ceilings have the same structure as Approach A, with `c_j = dq_j/dτ_B` [rad/mm] and `h_j = d²q_j/dτ_B²` [rad/mm²]:

$$v_{\mathrm{vel},B} = \min_j \frac{\dot q_{j,\max}}{|dq_j/d\tau_B|}, \qquad v_{\mathrm{accel},B} = \sqrt{u_{\max}} \tag{16}$$

with the same LP for `u_max`, and `v_{lim,B} = min(v_{vel,B}, v_{accel,B})`.

### The conversion that makes B attractive

From (14), the position and rotation contributions to arc-length rate are:

$$\left\|\frac{dp}{d\tau_B}\right\| = \frac{\|\Delta p_{tcp}\|}{\Delta\tau_B} \in [0,1], \qquad \left\|\frac{d\theta}{d\tau_B}\right\| = \frac{\Delta\theta}{\Delta\tau_B} \tag{17}$$

so the deliverables reduce to arithmetic on quantities already in the dense-path object:

$$v_{tcp} = \left\| \frac{dp}{d\tau_B} \right\| \cdot \dot\tau_B, \qquad \omega_{tcp} = \left\| \frac{d\theta}{d\tau_B} \right\| \cdot \dot\tau_B \tag{18}$$

$$\dot\tau_B \le \frac{v_{cmd}}{\|dp/d\tau_B\|}, \qquad v_{flat} \le \min_\tau v_{\mathrm{lim},B}(\tau) \cdot \| dp/d\tau_B \| \tag{19}$$

For the RS-comparison plot in millimetres, one small simplification: since by definition `‖dp/dτ_B‖ · dτ_B = ‖dp‖`, the integral is trivial and reduces to cumulative Euclidean displacement:

$$s_{pos} = \sum \|\Delta p_{tcp}\| \tag{20}$$

which we already compute for other purposes. **Nothing in this section needs the Jacobian.**

### Pure-rotation behaviour

When the tooltip is held fixed and the tool rotates, `‖Δp_tcp‖ = 0`, so `Δτ_B = λ·Δθ > 0`. The position and rotation arc-length rates become `‖dp/dτ_B‖ → 0` and `‖dθ/dτ_B‖ → 1/λ`. Then `v_tcp = 0` (correct — tooltip stationary), the linear `v_cmd` ceiling becomes infinite (correct — `v_cmd` limits linear speed, not rotation), and the joint constraints continue to bind through the wrist joints via `dq/dτ_B`. Handled cleanly, no special-case logic.

### Literature

- Park [8] proves SE(3) admits no bi-invariant positive-definite metric — the weighted choice is a pragmatic compromise, and Park explicitly discusses this class of metrics.
- Žefran & Kumar [9] cover interpolation and metrics on SE(3), including weighted variants.
- Kunz [10] discusses SE(3)-parameterised TOPP for real-time control.

---

## 5. Approach C — Screw / se(3) Lie-Algebra Arc-Length

### Definition

Between consecutive poses, take the matrix logarithm:

$$\xi^{[k]} = \log\!\big( T_k^{-1} T_{k+1} \big) \in \mathfrak{se}(3), \qquad \xi = \begin{bmatrix} \omega \\ v \end{bmatrix} \tag{21}$$

with `ω ∈ ℝ³` (rad) and `v ∈ ℝ³` (mm), then

$$\Delta \tau_C^{[k]} = \| \xi^{[k]} \|_2 = \sqrt{\|\omega\|^2 + \|v\|^2} \tag{22}$$

The translational component `v` is **not** `Δp`. Under the constant-screw motion between waypoints assumed by this parameterisation, `v = A⁻¹(ω)·Δp` where `A(ω)` is the left Jacobian of `SO(3)`; for small rotations `A ≈ I` and `v ≈ Δp`, and for large rotations the two differ [11, Murray, Li & Sastry, Ch. 2].

### The unit-consistency problem

Equation (22) mixes rad² and mm². It carries an **implicit `λ = 1 mm/rad`**, and there is no way around that inside the pure Lie-algebra norm. For our robot with the ~158 mm tooltip lever arm this weighting is unphysical: a 1 rad rotation produces about 158 mm of tooltip travel, so the `‖v‖` term dominates while `‖ω‖` barely registers. The screw arc-length ends up nearly the position arc-length in disguise, with a small correction from `A⁻¹`.

If we weight the norm as `√(‖ω‖² + (1/λ²)‖v‖²)`, we recover exactly the free parameter of Approach B, but with the extra `A⁻¹` machinery in every conversion. So C is either dimensionally inconsistent (unweighted) or algebraically equivalent to B (weighted) with a rougher implementation.

### Chain-rule quantities and TCP conversion

Same structure as A and B via (1)–(3). For the twist-to-TCP-velocity conversion: under the constant-screw model that `τ_C` implicitly uses, the body twist along the path is `V_b = ξ_local · τ̇`, where `ξ_local` is the local (per-interval) log twist. The spatial twist is then `V_s = Ad(T) · V_b`, and the tooltip point velocity — since the tooltip is offset from the body-frame origin — is

$$\dot p_{tcp} = v_{\mathrm{spatial}} + \omega_{\mathrm{spatial}} \times p_{tcp}^{(\mathrm{world})} \tag{23}$$

with the standard body-vs-spatial care from [12, Lynch & Park Ch. 3]. This is more machinery than either A or B, for no benefit in our setting.

### Literature

- Murray, Li & Sastry [11] — definitive treatment of screws and log/exp maps.
- Lynch & Park [12], Ch. 3 — twists, Adjoint, body-vs-spatial frames.
- Park [8] — the metric non-uniqueness on SE(3) applies just as much to se(3).

---

## 6. Approach D — Decoupled Position + Orientation

### Idea

Parameterise position and orientation separately, then merge into a single scalar for TOPP:

$$s_{pos} = \sum \|\Delta p_{tcp}\| \; [\mathrm{mm}], \qquad \sigma_{rot} = \sum \Delta\theta \; [\mathrm{rad}] \tag{24}$$

The two sensitivities are

$$\frac{dq_j}{d s_{pos}} \; [\mathrm{rad}/\mathrm{mm}], \qquad \frac{dq_j}{d\sigma_{rot}} \; [\mathrm{rad}/\mathrm{rad}] \tag{25}$$

### The merge

Along a fixed geometric path, both `s_pos` and `σ_rot` are functions of a master parameter (say `s = s_pos`), and their ratio `ρ(s) = dσ_rot/ds_pos` is a fixed geometric function of position. Then

$$\dot q = \left( \frac{dq_p}{d s_{pos}} + \rho(s)\,\frac{dq_\omega}{d\sigma_{rot}} \right) \dot s_{pos} \tag{26}$$

collapses the two-input system to one, and we can define the effective tangent

$$\frac{dq}{ds_{\mathrm{eff}}} = \frac{dq_p}{d s_{pos}} + \rho(s)\,\frac{dq_\omega}{d\sigma_{rot}} \tag{27}$$

The catch is the second derivative. Differentiating (27) with respect to `s`, applying the product rule to the second term and then the chain rule `d/ds[dq_ω/dσ] = ρ · d²q_ω/dσ²`:

$$\frac{d^2 q}{d s^2} = \frac{d^2 q_p}{d s_{pos}^2} + \rho'(s)\,\frac{dq_\omega}{d\sigma_{rot}} + \rho^2(s)\,\frac{d^2 q_\omega}{d\sigma_{rot}^2} \tag{28}$$

The `ρ'` cross-term couples the two parameterisations wherever the rotation-to-translation ratio changes — which is at every corner of a siping path. And when the tooltip is held fixed (`ds_pos → 0`), `ρ → ∞`, so we have to switch the master parameter to `σ_rot`, matching values and derivatives across the switch. Doable, but a lot of engineering surface for what is essentially a diagnostic decomposition.

### Where D is genuinely useful

Even without adopting D as the parameterisation, the decomposition (27) tells us **whether a bottleneck is translation-driven or rotation-driven**. If at the slowest corner `‖dq_p/ds_pos‖ ≫ ρ · ‖dq_ω/dσ_rot‖`, the corner is a positional-curvature problem (widen the blend); if the inequality flips, it's an orientation-rate problem (slow the reorientation schedule). Worth computing as a post-hoc diagnostic on top of whichever parameterisation we run.

### Literature

- Hauser & Ng-Thow-Hing [13] discuss separate position/orientation timing in the context of shortcut smoothing.
- Pham [6] establishes the single-parameter TOPP framework that (27) is engineered to fit into.
- Siciliano et al. [2], Sec. 4.3.3, treats decoupled position/orientation trajectory planning.

---

## 7. Comparison Table

| Property | A. Joint-space | B. Weighted SE(3) | C. Screw se(3) | D. Decoupled |
|---|---|---|---|---|
| **Parameter units** | rad | mm (weighted) | mixed (rad²+mm²) | mm (master), rad (secondary) |
| **Free parameter** | none | `λ` (data-estimated) | implicit `λ = 1` | none, but switching logic |
| **‖dq/dτ‖** | `= 1` (unit norm) | varies | varies | varies + cross terms |
| **Derivative conditioning** | best | good | mixed-unit hazard | mixed at switches |
| **Handles pure rotation** | yes | yes (via `λΔθ`) | yes but underweighted | yes (switch master) |
| **`v_tcp` linear (mm/s)** | needs Jacobian | pose ratios | Adjoint + FK | direct or FK |
| **`ω_tcp` angular (rad/s)** | needs Jacobian | quaternion ratios | Adjoint | `ρ·ṡ` or direct |
| **`v_cmd` ceiling** | Jacobian, spatially varying | pose ratios, spatially varying | Adjoint | direct on `s_pos` master |
| **Best constant TCP speed** | Jacobian | pose ratios | Adjoint | direct on `s_pos` master |
| **RS x-axis (mm)** | FK integration | `Σ‖Δp‖` (trivial) | FK integration | native on `s_pos` master |
| **TOPP core** | standard | standard | standard | switching + cross terms |
| **New infrastructure required** | Jacobian at every eval | none | Adjoint + log/exp | switching + cross terms |

Cross-checked against Bobrow [3], Pfeiffer & Johanni [4], Verscheure [7], Pham [6], and Lynch & Park Ch. 9 [1]. The pure-rotation column is a genuine differentiator: A and B handle it structurally, C underweights it, D requires a master switch.

---

## 8. Which Fits Siping

Siping has a specific character worth exploiting:

- **Orientation changes gradually with displacement.** The tool rotation is spread over long straight runs, not concentrated at points, so `ρ(s) = dσ/ds_pos` is small and slowly varying almost everywhere.
- **Waypoint spacing is a design lever we control.** Zone data is set so the effective blend radius after ABB's overlap reduction is what we want. We can densify or thin waypoints without changing the geometric path.
- **The tooltip pose is tracked to tight geometric tolerance** (`|Δp| < 1 mm`, `|Δθ| < 0.1 rad` in the FK-vs-blend check).
- **Nike's deliverable is the linear TCP velocity in mm/s.** Angular velocity is a diagnostic today; it might become a first-class report later.

Given all four, B is the natural fit. The parameter is in millimetres, aligning with what we report; `λ` is well-behaved because rotation-per-mm is small and roughly uniform; the pose-ratio conversions are trivial from data we already have; and the RS comparison x-axis is `Σ‖Δp‖`, which we already compute for other purposes.

A also works — it's the safer general-purpose choice — but every reporting quantity is now a Jacobian evaluation, and for siping specifically the extra machinery buys nothing we can measure.

C and D are ruled out for this application. C has the wrong implicit weighting for our lever arm; D adds cross-term complexity we don't need when `ρ` is small.

---

## 9. A vs B — Head to Head

If A were universally better, we'd take the Jacobian tax and move on. It isn't, but it's close, and it's worth being honest about where.

**Where A wins.**

- Derivative conditioning. `‖dq/dτ_A‖ = 1` by construction, so the acceleration-limit curve is stable regardless of pathological path shapes or near-singular configurations.
- No free parameter. `λ` disappears entirely, along with the sensitivity checks it obliges.
- Robust across arbitrary toolpaths. Rotation-dominated segments, near-singular configurations, sparse waypoints — A doesn't care.
- Multi-segment consistency. Every segment's `τ_A` has the same physical meaning; concatenating is trivial.

**Where B wins for us.**

- Every reporting quantity avoids the Jacobian. `v_tcp`, `ω_tcp`, `v_cmd` ceiling, best constant speed, RS x-axis — all reduce to arithmetic on pose ratios we already have.
- `v_cmd` is a linear TCP-speed limit in mm/s. B's ceiling is `v_cmd / ‖dp/dτ_B‖`, computed directly from pose data. A's ceiling is `v_cmd / ‖J_p·dq/dτ_A‖`, which requires the Jacobian *and* the FK-consistency of the spline joint path. Every place we need Jacobian consistency is a new validation surface.
- `τ_B` has millimetre units, aligning with everything Nike (and RobotStudio) reports in.
- On siping paths, the corner-rounding problem in the spline is the real accuracy bottleneck. Switching parameterisation doesn't fix it. So the "A is cleaner" argument evaporates when the actual error source is elsewhere.

**Where A fails for us.**

The Jacobian dependency creates a validation web:

1. `q(τ)` is a fitted quintic spline. Small residuals against the raw IK path are unavoidable.
2. `J_p(q_spline(τ))` inherits those residuals with a compressive/amplifying factor that varies with configuration.
3. Every reporting quantity — `v_tcp`, `ω_tcp`, `v_cmd` ceiling, `v_flat`, `s_pos` — is `J_p·(dq/dτ)`, so a small `q` error can become a several-percent `v_tcp` error near singular configurations. There is no clean way to bound this without adding a monitor for `min(σ_J)` along the path.

B replaces the Jacobian chain with pose ratios computed on the *ground-truth* dense pose data (not on `q_spline`), so the reporting is decoupled from spline-fit accuracy. That decoupling is the concrete engineering advantage.

**Transition effort if we later switch to A.**

If we ever hit a toolpath where B's `λ` estimation is unstable (pure or near-pure rotation dominating the segment), the switch to A is bounded and clear:

- Add a `world_jacobian(q)` layer for evaluation (Pinocchio provides it directly).
- Replace the five conversions with their Jacobian-based versions (equations 9–13).
- Add a Jacobian-consistency monitor: compare `J_p·q̇` at the profile's sampled configurations to `Δp/Δt` on the dense poses. Flag any sample where they disagree above threshold.
- Keep the TOPP core untouched — it's parameterisation-agnostic.

Roughly one week of implementation and validation. We're not painting ourselves into a corner.

---

## 10. Why We Chose B

Two contextual reasons drove this, on top of the technical arguments above.

**We inherited displacement-space TOPP.** The velocity-profiling feature grew out of the Feature 3.1 module, which already worked on displacement arc-length. TOPP was operating on `q(s)` where `s` was the arc displacement along the blended arc, and we already tracked TCP orientation tightly through IK on the SE(3) blended path. Continuing in displacement space — with `λ` extending the metric to handle the rotation term properly — keeps the whole upstream chain (blend construction, dense sampling, IK, spline fitting) untouched. Approach A would have required reparameterising everything from scratch and re-validating a lot of code that's already working.

**Our deliverable is linear TCP speed in mm/s.** Nike asks "can we run this sipe at 50 mm/s?" and "how long will it take?" — both linear-speed questions. `τ_B` in millimetres and `ṡ_B` interpretable as blended TCP travel per second makes the direct answer a one-line conversion instead of a Jacobian evaluation. If angular velocity ever becomes a first-class report, B extends cleanly via (18) — the same pose-ratio machinery.

There's a third reason on top of those: on siping paths, the accuracy bottleneck is the spline rounding of tight corners (0.7 mm position residual and the corresponding curvature loss at the apex), not the parameterisation. Switching to A doesn't help with that; the fix has to be in how we compute `d²q/dτ²` at corners (secant cap, raw-curvature injection, or corner-aware knot placement). Given equal error contribution, we chose the parameterisation with less validation surface.

That design choice is **not yet the running default** — see the status box in the TL;DR. Until `--se3-arc-length` is validated, production is still `τ_E`.

---

## 11. Known Limitations of B

**`λ` is data-dependent.** `λ` is estimated per segment. Two different siping segments give two different `λ` values, so raw `τ_B` is not directly comparable across segments. This does not affect any deliverable — all reporting quantities are converted to task-space units — and there is no continuity issue at segment boundaries because inter-sipe motion is planned separately by a different feature and is not profiled here. But it's worth documenting for anyone reading raw TOPP internals.

**`λ`-sensitivity of the speed profile.** Running the same segment at `λ`, `λ/2`, `2λ` produces slightly different profiles. On siping paths this is small (typically well under 5% duration change), but it should be checked per segment as a cheap sanity metric.

**Prism application (adjacent Nike program).** Prism has sharp 180° turns, but the reorientation is continuously spatially distributed along an arc with displacement. That's structurally similar to siping — `ρ(s)` is small and smooth, `λ` estimation is stable — so B should carry over cleanly. Worth validating with a representative Prism toolpath before committing.

**The stalled acute-angle program (per Jared).** We don't know the orientation profile. If the acute-angle corners are position-only with the tool holding orientation, B handles it fine — same as siping, tighter blend geometry. If the corners involve step orientation changes (a discrete axis flip at the corner), `Δθ` spikes locally, `λ` estimation becomes noisy, and both A and B need re-evaluation for that path. Approach A is the safer starting point for that program when it revives.

**Corner rounding is not fixed by parameterisation choice.** The 0.7 mm residual and the corresponding curvature loss at corner apices is a spline-fit issue and applies to any parameterisation. It's a separate work item — secant cap tuning or raw-curvature injection at corners — and B does not help with it.

---

## 12. Special Case — Successive Waypoints, Same Position, Different Orientation

Suppose four consecutive waypoints have identical tooltip position `p` but different orientations. The commanded motion is a pure reorientation around the fixed tooltip. Because the TCP is offset from J6, this is not a trivial motion — J1–J6 must coordinate to hold the tooltip fixed while the tool rotates around it, which involves substantial joint motion (particularly at the wrist, but the arm joints participate too).

For each transition `k → k+1`:

$$\| \Delta p_{tcp} \| = 0, \qquad \Delta\theta > 0 \tag{29}$$

so from (14):

$$\Delta\tau_B = \sqrt{ 0 + \lambda^2 \Delta\theta^2 } = \lambda \, \Delta\theta \tag{30}$$

The parameter advances at a well-defined rate proportional to the rotation. Downstream:

- **Position tangent** `‖dp/dτ_B‖ = 0`, so `v_tcp = 0` at these samples — correct, the tooltip is stationary in space.
- **Rotation tangent** `‖dθ/dτ_B‖ = 1/λ`, so `ω_tcp = ṡ/λ`. Reporting is well-defined and non-zero.
- **`v_cmd` ceiling** `v_cmd / ‖dp/dτ_B‖ → ∞`. The linear-speed constraint does not bind pure rotation, which is physically correct: `v_cmd` in mm/s limits linear TCP motion, not tool rotation. If we ever add a rotational command limit `ω_cmd`, it enters as `ω_cmd·λ` in the same slot — cleanly extensible.
- **Joint constraints** continue to bind through the wrist joints via `dq/dτ_B`, which is nonzero.
- **Best constant TCP linear speed** through this segment is `v_flat = 0` — you cannot maintain a nonzero linear speed through a pure reorientation. Consistent with reality.

The parameterisation handles this case cleanly, without special-case logic. The only caveat: our `λ` estimate (15) uses samples with `Δθ > ε`; if the *entire* segment is pure rotation, all those samples have `‖Δp‖ = 0`, so the median lever arm is undefined. In that degenerate case we fall back to a default `λ` (say the fixed tooltip-to-wrist distance, ~158 mm/rad for our fixture; CLI `--se3-lambda-fixed` default 172.7). This is a soft fallback rather than a failure mode — the profile is still well-defined and physically correct — but the specific numerical value of `ṡ_B` depends on the chosen fallback.

**Production (2026-08-13):** this section describes Approach B. The running default is still position-only `s = Σ‖Δp‖`, which **does not** handle this case (`Δs = 0`). Turn on `--se3-arc-length` to get the behaviour above. For siping this never arises; do not author same-position / different-orientation waypoints until B is the default.

---

## 13. Alternatives Considered But Rejected

For completeness — and to head off "did you consider X?" questions — we looked at three other parameterisations before settling on A/B/C/D as the comparison set. None survived a technical filter.

**Position-only arc-length: `τ_E = Σ‖Δp_tcp‖`.** This is what Feature 3.1 originally used and what B extends. It's a special case of B with `λ → 0` (rotation contributes nothing to the parameter). It collapses catastrophically on pure-rotation segments (`Δτ_E = 0`, division-by-zero in `dq/dτ`) and silently under-weights rotation-dominated segments where joint work is real but positional travel is small. B is strictly a generalisation and dominates it.

**This is still the production default.** Enable `--se3-arc-length` to run B. Until that is validated on siping + a pure-reorient fixture path, do not author consecutive waypoints at the same tooltip position.

**Normalised path parameter `τ ∈ [0, 1]` (waypoint index or fractional).** Used in some TOPP-RA reference implementations for algorithmic convenience. Loses the physical meaning of derivatives (they're now in rad-per-normalised-unit rather than rad/mm), so every reporting quantity needs an extra conversion through the path length. Adds paperwork without insight.

**Inertia-weighted joint-space arc-length: `Δτ² = Δq^T M(q) Δq`.** Uses the configuration-space mass matrix as the metric. This is well-motivated for **torque-limited** planning (where the natural "distance" is kinetic energy at unit speed), but our limits are **kinematic** (velocity, acceleration, jerk on joint coordinates). Inertia weighting distorts the derivatives relative to the constraint box in a way that doesn't help — it introduces "mass-preferred" directions the constraints don't care about, and requires configuration-dependent `M(q)` evaluation everywhere. If we were doing torque-limited TOPP, this would be the right choice; we aren't.

The general principle we settled on: on SE(3) with no canonical metric [8], any parameterisation choice is a modelling assumption, and the honest question is which assumption produces the smallest validation burden while matching the deliverable. For our stack that's B; for a general-purpose profiler it would be A.

---

## References

[1] Lynch, K.M. and Park, F.C., *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017. Ch. 9 — Time-scaling of trajectories and time-optimal path parameterisation.

[2] Siciliano, B., Sciavicco, L., Villani, L., Oriolo, G., *Robotics: Modelling, Planning and Control*, Springer, 2009. Ch. 4 — Trajectory planning.

[3] Bobrow, J.E., Dubowsky, S., Gibson, J.S., "Time-optimal control of robotic manipulators along specified paths," *International Journal of Robotics Research*, 4(3):3–17, 1985. Original derivation of the velocity limit curve.

[4] Pfeiffer, F., Johanni, R., "A concept for manipulator trajectory planning," *IEEE Journal of Robotics and Automation*, 3(2):115–123, 1987. Phase-plane formulation used above.

[5] Kunz, T., Stilman, M., "Time-optimal trajectory generation for path following with bounded acceleration and velocity," *Robotics: Science and Systems*, 2012. Joint-space parameterisation for TOPP.

[6] Pham, Q.-C., "A general, fast, and robust implementation of the time-optimal path parameterization algorithm," *IEEE Transactions on Robotics*, 30(6):1533–1540, 2014.

[7] Verscheure, D., Demeulenaere, B., Swevers, J., De Schutter, J., Diehl, M., "Time-optimal path tracking for robots: a convex optimization approach," *IEEE Transactions on Automatic Control*, 54(10):2318–2327, 2009.

[8] Park, F.C., "Distance metrics on the rigid-body motions with applications to mechanism design," *ASME Journal of Mechanical Design*, 117(1):48–54, 1995. Non-existence of a bi-invariant positive-definite metric on SE(3).

[9] Žefran, M., Kumar, V., "Interpolation schemes for rigid body motions," *Computer-Aided Design*, 30(3):179–189, 1998. Weighted metrics on SE(3).

[10] Kunz, T., *Time-Optimal Path Following in Robot Motion Planning*, Ph.D. thesis, Georgia Institute of Technology, 2016. SE(3)-parameterised TOPP.

[11] Murray, R.M., Li, Z., Sastry, S.S., *A Mathematical Introduction to Robotic Manipulation*, CRC Press, 1994. Ch. 2–3 — screws, exponential and log maps, Adjoint.

[12] Lynch, K.M., Park, F.C., *Modern Robotics*, Ch. 3 — twists and body-vs-spatial velocity conventions.

[13] Hauser, K., Ng-Thow-Hing, V., "Fast smoothing of manipulator trajectories using optimal bounded-acceleration shortcuts," *IEEE International Conference on Robotics and Automation (ICRA)*, 2010. Position-orientation timing considerations.
