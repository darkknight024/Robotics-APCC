# Singularity Analysis — Conversations and Updates

Singularity analysis for **6-DOF spherical-wrist robots** (ABB IRB 1300 family).

Branch: **singularity** - will be merged to main after this branch is tested for singularities.

Two analysis modes are available (configured via `singularity_analysis.mode` in
`reachability_config.yaml` or via CLI `--singularity-mode`):

| Mode | Class | Module | Description |
|---|---|---|---|
| `classified` | `SingularityAnalyzer` | `core/singularity_analysis.py` | Decomposes into shoulder / elbow / wrist sub-types |
| `unified` | `UnifiedSingularity` | `core/unified_singularity.py` | Full-Jacobian σ_min only (no type split) |
| `none` | — | — | Skips singularity analysis entirely |

Orchestrated from `tests/test_reachability_singularity.py` and
`feasibility_analysis.py`.

---

## 0. Unified Singularity (Legacy Approach)

### What it does

`UnifiedSingularity` computes three metrics from the **full 6×6 Jacobian** without
attempting to identify *which* type of singularity is occurring:

| Metric | Formula | Interpretation |
|---|---|---|
| σ_min | Smallest singular value of J | → 0 at any singularity |
| Condition number κ | σ_max / σ_min | → ∞ at any singularity |
| Manipulability w | √det(JJᵀ) | → 0 at any singularity |

A waypoint is flagged `near_singularity` when σ_min < `singularity_threshold`
(default 0.01).

### Why it was the original approach

The unified approach was the first implementation (inside `core/feasibility_checks.py`).
It is computationally cheap and sufficient when the only question is *"is this
waypoint near any singularity?"*  It cannot answer *which kind* of singularity,
so operators receive no guidance on which joint motions to avoid.

### Why we moved to classified analysis

Physical testing with ABB RobotWare revealed that the robot controller treats
different singularity types differently:

1. **Wrist singularity** has a hard-coded ±0.76° dead-band on joint 5.
   A unified σ_min threshold cannot replicate this geometric boundary because the
   sub-Jacobian's scale depends on link lengths, not just angular proximity.
2. **Shoulder singularity** causes the trajectory planner to abort at a specific
   condition number of the arm sub-Jacobian — information invisible to the full
   Jacobian's σ_min.
3. Operators need to know *which* joint to adjust, not just that "something is
   singular."

The classified `SingularityAnalyzer` addresses all three points by decomposing the
Jacobian into per-type sub-matrices and applying independently calibrated thresholds.

### When to use unified mode

- Quick screening where per-type information is not needed.
- Backward-compatible analysis matching earlier reports.
- As a cross-check against the classified analyzer's aggregate σ_min.

---

## 1. Singularity Types (Classified Mode)

A 6R robot with a spherical wrist has exactly **three fundamental singularity types**.
Each is detected from the 6×6 Jacobian J (convention: rows = `[angular(3); linear(3)]`,
columns = joints 1–6) and the joint-angle vector q.

### 1.1 Wrist Singularity

Occurs when **joints 4 and 6 axes align** (joint 5 ≈ 0 or π).

Two detection modes are supported (controlled by `check_j5_only` flag on
`_classify_wrist`, default `True`):

**J5-only mode** (default, `check_j5_only=True`):

| What | Formula |
|------|--------|
| Detection metric | abs(q₅) (absolute joint 5 angle) |
| Threshold | 0.76° = 0.01326 rad (matching ABB RobotWare dead-band) |
| Flagged when | abs(q₅) < 0.76° |

This is a fast geometric check that exactly replicates the empirically observed
ABB RobotWare singularity boundary.

**Sub-Jacobian mode** (`check_j5_only=False`):

| What | Formula |
|------|--------|
| Sub-Jacobian | J_w = J[0:3, 3:6] (angular rows, wrist columns) |
| Detection metric | σ_min(J_w) |
| Flagged when | σ_min(J_w) < τ_wrist |

Additional metrics stored: det(J_w), q₅ angle, angular distance of q₅ from 0 / π.

### 1.2 Shoulder Singularity

Occurs when the **wrist center lies on the joint-1 (base Z) axis**.

| What | Formula |
|---|---|
| Sub-Jacobian | J_s = J[3:6, 0:3] (linear rows, arm columns) |
| Detection metric | σ_min(J_s) |
| Flagged when | σ_min(J_s) < τ_shoulder |

Additional metric: det(J_s), and optionally the XY distance of the wrist center
from the base Z-axis (when the FK solver is available).

**Known limitation:** The FK-based XY distance metric currently uses the TCP position
instead of the true wrist center (joint 5 origin). This is informational only and does
not affect the classification decision (which relies on σ_min).


### 1.3 Elbow Singularity

This is work in progress and currently deprioritized. Will pick this up when we need this. 

Occurs when the **Wrist Center Point (WCP) lies exactly on the plane defined by axes 2 and 3**. Physically, the arm is fully extended to the workspace boundary or folded completely back on itself. 

*Crucial Note:* Evaluating the linear velocity columns of the Jacobian at the TCP (J[3:6, 1:3]) is mathematically brittle, as the wrist configuration will artificially warp the vectors. Elbow singularities are purely positional and must be evaluated at the WCP.

| What | Formula |
|---|---|
| Metric Source | Forward Kinematics (FK) evaluation of the WCP, or the Jacobian evaluated strictly at the WCP. |
| Detection metric | Distance d from the WCP to the Axis 2/3 plane, or q₃ proximity to its mechanical limit. |
| Flagged when | d < τ_elbow_dist or abs(q₃ − q₃,limit) < τ_elbow_angle |

Additional metric stored: If using a WCP-evaluated Jacobian, collinearity = 1 − |cos θ| where θ is the angle between the Axis 2 and Axis 3 linear velocity column vectors at the WCP.

**Known limitation:** The current code evaluates collinearity from the TCP Jacobian
(`J[3:6, 1]` and `J[3:6, 2]`), which is warped by the wrist configuration and can
produce false positives/negatives. The σ_min of the same sub-matrix is used for the
decision, which inherits the same warping issue.

### 1.4 Compound Types

Multiple types can be active simultaneously. The classifier outputs one of:
`none`, `shoulder`, `elbow`, `wrist`, `shoulder+elbow`, `shoulder+wrist`,
`elbow+wrist`, `shoulder+elbow+wrist`.

### 1.5 Thresholds

Default τ = 0.1 for all three types (configurable in `reachability_config.yaml`).
A waypoint is classified as **singular** (`is_singular = True`) when *any* type is active.

*Proposal for improvement*

Thresholds cannot be universal because the sub-Jacobians operate in different units (e.g., J_w maps angular to angular and is unitless; J_s maps angular to linear and scales with mm or m). 

Each singularity type requires a mathematically independent, empirically calibrated threshold:
* τ_wrist: In `check_j5_only` mode, this is replaced by the 0.76° geometric boundary. In sub-Jacobian mode, calibrate against the empirical q₅ ≈ ±0.76° boundary.
* τ_shoulder: Calibrated against the condition number degradation curve.
* τ_elbow: Calibrated against the WCP proximity to the Axis 2/3 plane.

A waypoint is classified as **singular** (`is_singular = True`) when *any* type breaches its specific threshold.

---

## 2. Solver Compatibility

The analyser operates on the **6×6 Jacobian** and **joint angles** — it is solver-agnostic. Regardless of how IK is solved, analyzing a singularity requires evaluating the 6×6 Jacobian matrix J(q). There are two ways to compute this:

| Solver | Jacobian source | Convention |
|---|---|---|
| **EAIK** (analytical IK) | `eaik_fk_solver.get_jacobian(q)` — Calculated using exact spatial derivatives (cross products of joint axes and distances). This is mathematically pure and exact at all configurations. | `[angular; linear]` natively |
| **Pinocchio** (numerical IK) | `pin_fk_solver.get_jacobian(q)` — `pinocchio.computeFrameJacobian` Calculated using finite differences. You perturb each joint by a tiny amount Δq, compute the Forward Kinematics, and measure the change in pose Δx. Therefore, J ≈ Δx/Δq. | Pinocchio returns `[linear; angular]`; the FK solver **swaps rows** to output `[angular; linear]` to be consistent for downstream operations|

Both solvers produce an identical `[angular(3); linear(3)]` Jacobian by the time it reaches
`SingularityAnalyzer.analyze()`, so all downstream metrics and classification are consistent.

### Comparing solvers

Running the same trajectory with both backends and comparing the CSV / plots is a practical way
to validate IK solutions — differences in the singular value spectrum or type classification
indicate that the two solvers found different IK branches.

---

## 3. Configuration

### reachability_config.yaml

```yaml
singularity_analysis:
  mode: "classified"          # "classified", "unified", or "none"
  export_singularity_graphs: true
  unified_threshold: 0.01     # σ_min (smallest singular value) threshold for unified mode
  thresholds:                 # per-type thresholds for classified mode
    wrist: 0.1
    shoulder: 0.1
    elbow: 0.1
```

### feasibility_analysis.py CLI

```bash
python feasibility_analysis.py --toolpath <csv> --urdf <urdf> \
    --singularity-mode unified   # or "none" to skip
    --singularity-threshold 0.01
```

---

## 4. Empirical Calibration & Solver Validation Proposal (later)

### 4.1 Empirical Threshold Calibration
The default thresholds must be replaced with empirical limits derived from physical RobotStudio testing. 
1. **Wrist Threshold (τ_wrist):** Physical testing proved RobotWare uses a static boundary of q₅ ≈ ±0.76°. We must run the identical FK waypoints through `SingularityAnalyzer.analyze()` and map the output `wrist_sigma_min` at exactly q₅ = ±0.76°. That exact σ_min value becomes our hardcoded τ_wrist.
2. **Shoulder Threshold (τ_shoulder):** By iterating through the `MoveL` signal analyzer captures, we will calculate the condition number of the decoupled J_s matrix at every timestamp leading up to the failure. This will isolate the exact mathematical threshold where RobotStudio's trajectory planner aborts, establishing τ_shoulder.

### 4.2 Numerical vs. Analytical Jacobian Discrepancy
The current architecture assumes that the Jacobian matrices extracted from both solver backends produce identical downstream metrics for singularity classification. This assumption introduces a mathematical risk near singularity boundaries due to how the Jacobians are computed.

* **The Architectural Reality:** * While EAIK is an *analytical IK solver*, our pipeline extracts its Jacobian using a *numerical approximation* (finite differences: Δx/Δq). 
  * While Pinocchio serves as our *numerical IK solver*, its `computeFrameJacobian` function evaluates the Jacobian *analytically* using exact spatial derivatives.
* **The Problem:** Near a kinematic singularity, the mapping between joint space and task space becomes highly non-linear. The finite difference approximation used by the EAIK pipeline will degrade rapidly, causing its output singular value spectrum (`sv_0` ... `sv_5`) to diverge from the true mathematical state of the robot.
* **Validation Action:** We must run a differential analysis script comparing the full singular value spectrum outputs of both pipelines for the waypoints strictly inside the ±5° wrist sweep.
* **Resolution Path:** If the finite-difference Jacobian diverges significantly from the exact spatial Jacobian at the threshold boundary, we must standardize our singularity analyzer to strictly evaluate the exact analytical Jacobian (via Pinocchio) for all classifications, ensuring the mathematical condition numbers are strictly precise regardless of which IK solver generates the target joint state.

---

## 5. Email Conversations so far
Here is a brief summary of the conversation you had with the Nike team regarding "singularities":

### 5.1 Software Updates and Improvements**

  * A new software branch was created to estimate and classify the three types of singularities: shoulder, elbow, and wrist, by decomposing the estimated Jacobian into sub-jacobians. Branch name: **singularity**.

### 5.2 Data Requested and Stress Testing

  * **Data Requested:** You requested stress test data that involves applying small joint perturbations from a singular joint space, computing Forward Kinematics (FK), and then feeding these poses into RobotStudio's Inverse Kinematics (IK) solver to record singularity/reachability errors.
  * **Wrist Singularity Data:** Initial stress testing with the MoveL command in RobotStudio found the "Near Singularity" error occurs at approximately **-0.70° to 0.70°** for Joint 5 (J5). This was confirmed to be consistent across randomized base joint configurations, leading to a suggested conservative bound of **-1° to 1° for J5**.
  * **Shoulder Singularity Data:** Experiments were conducted where trajectories ended in a near-singularity error when the robot's Wrist Center Point (WCP) aligned with axis 1.
  * **RobotStudio Software Limitations:** It was noted that RobotStudio does not provide errors that distinguish between all three singularity types, only having a built-in "Near Wrist singularity" indicator for J5.

---