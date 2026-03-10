# Singularity Analysis — Technical Reference

Singularity analysis for **6-DOF spherical-wrist robots** (ABB IRB 1300 family).
Implemented in `core/singularity_analysis.py`, visualised by `utils/feasibility_plot.py`,
orchestrated from `tests/test_reachability_singularity.py`.

---

## 1. Singularity Types

A 6R robot with a spherical wrist has exactly **three fundamental singularity types**.
Each is detected from the 6×6 Jacobian $J$ (convention: rows = `[angular(3); linear(3)]`,
columns = joints 1–6) and the joint-angle vector $q$.

### 1.1 Wrist Singularity

Occurs when **joints 4 and 6 axes align** (joint 5 ≈ 0 or π).

| What | Formula |
|---|---|
| Sub-Jacobian | $J_w = J_{[0:3,\; 3:6]}$ (angular rows, wrist columns) |
| Detection metric | $\sigma_{\min}(J_w)$ |
| Flagged when | $\sigma_{\min}(J_w) < \tau_{\text{wrist}}$ |

Additional metrics stored: $\det(J_w)$, $q_5$ angle, angular distance of $q_5$ from 0 / π.

### 1.2 Shoulder Singularity

Occurs when the **wrist center lies on the joint-1 (base Z) axis**.

| What | Formula |
|---|---|
| Sub-Jacobian | $J_s = J_{[3:6,\; 0:3]}$ (linear rows, arm columns) |
| Detection metric | $\sigma_{\min}(J_s)$ |
| Flagged when | $\sigma_{\min}(J_s) < \tau_{\text{shoulder}}$ |

Additional metric: $\det(J_s)$, and optionally the XY distance of the wrist center
from the base Z-axis (when the FK solver is available).


### 1.3 Elbow Singularity

This is work in progress and currently deprioritized. Will pick this up when we need this. 

Occurs when the **Wrist Center Point (WCP) lies exactly on the plane defined by axes 2 and 3**. Physically, the arm is fully extended to the workspace boundary or folded completely back on itself. 

*Crucial Note:* Evaluating the linear velocity columns of the Jacobian at the TCP ($J_{[3:6, 1:3]}$) is mathematically brittle, as the wrist configuration will artificially warp the vectors. Elbow singularities are purely positional and must be evaluated at the WCP.

| What | Formula |
|---|---|
| Metric Source | Forward Kinematics (FK) evaluation of the WCP, or the Jacobian evaluated strictly at the WCP. |
| Detection metric | Distance $d$ from the WCP to the Axis 2/3 plane, or $q_3$ proximity to its mechanical limit. |
| Flagged when | $d < \tau_{\text{elbow\_dist}}$ or $\|q_3 - q_{3,\text{limit}}\| < \tau_{\text{elbow\_angle}}$ |

Additional metric stored: If using a WCP-evaluated Jacobian, collinearity $= 1 - |\cos\theta|$ where $\theta$ is the angle between the Axis 2 and Axis 3 linear velocity column vectors at the WCP.


### 1.4 Compound Types

Multiple types can be active simultaneously. The classifier outputs one of:
`none`, `shoulder`, `elbow`, `wrist`, `shoulder+elbow`, `shoulder+wrist`,
`elbow+wrist`, `shoulder+elbow+wrist`.

### 1.5 Thresholds

Default $\tau = 0.1$ for all three types (configurable in `reachability_config.yaml`).
A waypoint is classified as **singular** (`is_singular = True`) when *any* type is active.

*Proposal for improvement*

Thresholds cannot be universal because the sub-Jacobians operate in different units (e.g., $J_w$ maps angular to angular and is unitless; $J_s$ maps angular to linear and scales with mm or m). 

Each singularity type requires a mathematically independent, empirically calibrated threshold:
* $\tau_{\text{wrist}}$: Calibrated against the empirical $q_5 \approx \pm 0.76^\circ$ boundary.
* $\tau_{\text{shoulder}}$: Calibrated against the condition number degradation curve.
* $\tau_{\text{elbow}}$: Calibrated against the WCP proximity to the Axis 2/3 plane.

A waypoint is classified as **singular** (`is_singular = True`) when *any* type breaches its specific threshold.

---

## 2. Solver Compatibility

The analyser operates on the **6×6 Jacobian** and **joint angles** — it is solver-agnostic.

| Solver | Jacobian source | Convention |
|---|---|---|
| **EAIK** (analytical IK) | `eaik_fk_solver.get_jacobian(q)` — numerical Jacobian via finite differences | `[angular; linear]` natively |
| **Pinocchio** (numerical IK) | `pin_fk_solver.get_jacobian(q)` — `pinocchio.computeFrameJacobian` | Pinocchio returns `[linear; angular]`; the FK solver **swaps rows** to output `[angular; linear]` |

Both solvers produce an identical `[angular(3); linear(3)]` Jacobian by the time it reaches
`SingularityAnalyzer.analyze()`, so all downstream metrics and classification are consistent.

### Comparing solvers

Running the same trajectory with both backends and comparing the CSV / plots is a practical way
to validate IK solutions — differences in the singular value spectrum or type classification
indicate that the two solvers found different IK branches.

---

## 3. Singular Value Spectrum

### 3.1 What it is

For each waypoint, the full Jacobian is decomposed via SVD:

$$J = U \, \Sigma \, V^T, \qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_6 \ge 0$$

The **spectrum** is the ordered set $\{\sigma_1, \dots, \sigma_6\}$.

### 3.2 Key quantities derived from the spectrum

| Quantity | Formula | Interpretation |
|---|---|---|
| $\sigma_{\min}$ | $\sigma_6$ | Proximity to singularity (→ 0 at singularity) |
| Condition number | $\kappa = \sigma_1 / \sigma_6$ | Isotropy of motion capability (→ ∞ at singularity) |
| Manipulability | $w = \sqrt{\det(JJ^T)} = \prod \sigma_i$ | Volume of the velocity ellipsoid |

### 3.3 Why plot all 6 values

- $\sigma_{\min}$ alone tells you *how close* to singularity but not *which DOFs* are degraded.
- The full spectrum reveals how many singular values are simultaneously small, distinguishing
  a single borderline DOF from a near-complete rank collapse.
- On a log-scale plot, separation between the curves quantifies directional imbalance.

### 3.4 Solver comparison use-case

Different IK solutions (branches) for the same end-effector pose yield different Jacobians.
Overlaying the spectrum from EAIK and Pinocchio highlights:
- Whether both solvers chose the same IK branch.
- Which solution has better manipulability / is further from singularity.

---

## 4. Plots

All plots are exported per-trajectory when `export_singularity_graphs: true`.
A CSV singularity report is **always** saved regardless of this flag.

### 4.1 Overall σ\_min

Line plot of $\sigma_{\min}(J)$ per waypoint with a horizontal threshold line.
Gives a single-number summary of *how close* each waypoint is to any singularity.
Threshold values appear as bold red ticks on the y-axis.

### 4.2 Singularity Type Classification

Colour-coded bar strip — one colour per waypoint indicating the classified type
(`none`, `wrist`, `shoulder`, …). Instantly shows *where* and *what kind* of singularity
occurs along the trajectory.

### 4.3 Sub-Jacobian σ\_min

Three overlaid lines (wrist / shoulder / elbow) showing the $\sigma_{\min}$ of each sub-Jacobian.
Horizontal dashed lines mark the per-type thresholds (shared or individual).
This is the main diagnostic plot — it tells you *which* type is degrading and by how much.

### 4.4 Sub-Jacobian Determinants

Three vertically stacked subplots:

| Subplot | Metric |
|---|---|
| Wrist | $\det(J_{[0:3,\;3:6]})$ |
| Shoulder | $\det(J_{[3:6,\;0:3]})$ |
| Elbow | Collinearity $1 - |\cos\theta|$ of J2 / J3 columns |

Determinants complement $\sigma_{\min}$: they can change sign (indicating configuration flips)
and are sensitive to near-rank-deficiency in a different way than the smallest singular value.

### 4.5 Joint Angles

All 6 joint angles (degrees) plotted over the trajectory. Useful for correlating
singularity events with specific joint configurations (e.g., J5 crossing 0° → wrist singularity).

### 4.6 Dashboard (2×3)

Combined view of all the above in one image:

| | Col 0 | Col 1 | Col 2 |
|---|---|---|---|
| **Row 0** | Type Classification | Overall σ\_min | Sub-Jacobian σ\_min |
| **Row 1** | Joint Angles (deg) | Singular Value Spectrum | Sub-Jacobian Determinants |

---

## 5. CSV Column Reference

Each row = one waypoint. File: `T{n}_singularity_report.csv`.

| Column | Type | Description |
|---|---|---|
| `waypoint_index` | int | Zero-based waypoint number |
| `singularity_type` | str | Classified type (`none`, `wrist`, `shoulder+elbow`, …) or `unreachable` |
| `is_singular` | bool / str | `True` if any type active, `False` if none active, `unreachable` if waypoint was not reachable |
| `active_types` | str | Semicolon-separated list of active types (e.g. `shoulder;wrist`) |
| `overall_sigma_min` | float | $\sigma_{\min}(J)$ — smallest singular value of full Jacobian |
| `overall_condition_number` | float | $\kappa = \sigma_{\max}/\sigma_{\min}$ (∞ when $\sigma_{\min} = 0$) |
| `overall_manipulability` | float | $w = \sqrt{\det(JJ^T)}$ |
| `sv_0` … `sv_5` | float | Full singular value spectrum (descending: `sv_0` = $\sigma_{\max}$) |
| `wrist_sigma_min` | float | $\sigma_{\min}(J_{[0:3,3:6]})$ — wrist sub-Jacobian |
| `wrist_det_wrist_jacobian` | float | $\det(J_{[0:3,3:6]})$ |
| `wrist_j5_angle_rad` | float | Joint 5 angle (rad) |
| `wrist_j5_distance_to_singularity_rad` | float | Angular distance of $q_5$ from nearest singular value (0 or π) |
| `shoulder_sigma_min` | float | $\sigma_{\min}(J_{[3:6,0:3]})$ — shoulder sub-Jacobian |
| `shoulder_det_arm_jacobian` | float | $\det(J_{[3:6,0:3]})$ |
| `elbow_sigma_min` | float | $\sigma_{\min}([J_{[3:6,1]} \mid J_{[3:6,2]}])$ — elbow sub-matrix |
| `elbow_j2_j3_collinearity` | float | $1 - |\cos\theta|$ between J2 and J3 linear-velocity columns (0 = collinear = singular) |

**Unreachable waypoints**: `singularity_type` = `unreachable`, `is_singular` = `unreachable`,
all numeric fields default to 0 / empty. This lets CSV readers determine reachability and
singularity status from a single file.

## 6. Empirical Calibration & Solver Validation Proposal (later)

### 6.1 Empirical Threshold Calibration
The default thresholds must be replaced with empirical limits derived from physical RobotStudio testing. 
1. **Wrist Threshold ($\tau_{\text{wrist}}$):** Physical testing proved RobotWare uses a static boundary of $q_5 \approx \pm 0.76^\circ$. We must run the identical FK waypoints through `SingularityAnalyzer.analyze()` and map the output `wrist_sigma_min` at exactly $q_5 = \pm 0.76^\circ$. That exact $\sigma_{\min}$ value becomes our hardcoded $\tau_{\text{wrist}}$.
2. **Shoulder Threshold ($\tau_{\text{shoulder}}$):** We will iterate through the `MoveL` signal analyzer captures. By calculating the analytical condition number of the decoupled $J_s$ matrix at every timestamp leading up to the failure, we will isolate the exact mathematical threshold where RobotStudio's trajectory planner aborts, establishing $\tau_{\text{shoulder}}$.

### 6.2 Numerical vs. Analytical Jacobian Discrepancy
The current architecture assumes that EAIK's numerical Jacobian (via finite differences) and Pinocchio's analytical Jacobian produce identical downstream metrics. This assumption breaks down near singularity boundaries.

* **The Problem:** Near a singularity, the $SE(3)$ mapping becomes highly non-linear. The finite difference approximation ($\frac{\Delta x}{\Delta q}$) used by EAIK will rapidly degrade and diverge from Pinocchio's true analytical derivative.
* **Validation Action:** We must run a differential analysis script comparing the `overall_sigma_min` and `sv_0` ... `sv_5` outputs of both solvers for the waypoints inside the $\pm 5^\circ$ wrist sweep.
* **Resolution Path:** If EAIK's numerical approximation diverges significantly from Pinocchio at the threshold boundary, we must either:
  a) Implement a dynamic, infinitely smaller step size for EAIK's finite differences as it approaches boundaries (computationally expensive).
  b) Strictly utilize Pinocchio's analytical Jacobian for all singularity classification, relegating EAIK purely to positional IK solving.
