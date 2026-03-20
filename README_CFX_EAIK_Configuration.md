# ABB Robot Configuration Data (confdata) & EAIK Solution Mapping

> **Audience:** This document is written for a coding agent that must replicate RobotStudio's configuration-selection behaviour using the EAIK analytical IK solver.  
> **Source authority:** ABB RobotWare-OS RAPID Reference Manual (3HAC 16581-1 Rev. J) and ABB RobotStudio SDK `ConfigurationData` struct.

---

## Table of Contents

1. [What is `confdata`?](#1-what-is-confdata)
2. [The Four Configuration Parameters](#2-the-four-configuration-parameters)
3. [The `cfx` Field: All 8 Configurations Explained](#3-the-cfx-field-all-8-configurations-explained)
4. [Meaning in Task Space vs Joint Space](#4-meaning-in-task-space-vs-joint-space)
5. [How to Compute cf1 / cf4 / cf6 from Joint Angles](#5-how-to-compute-cf1--cf4--cf6-from-joint-angles)
6. [How to Compute cfx (0–7) from Joint Angles](#6-how-to-compute-cfx-07-from-joint-angles)
7. [Why IRB 1400 Uses Only cf1, cf4, cf6 (no cfx)](#7-why-irb-1400-uses-only-cf1-cf4-cf6-no-cfx)
8. [Robot Model Reference Table](#8-robot-model-reference-table)
9. [EAIK Returns 8 Solutions: Mapping to ECFX](#9-eaik-returns-8-solutions-mapping-to-ecfx)
10. [ECFX for IRB 1400: Exact Formulas](#10-ecfx-for-irb-1400-exact-formulas)
11. [Full Worked Example](#11-full-worked-example)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. What is `confdata`?

When a robot's tool-centre-point (TCP) is commanded to a specific pose in Cartesian space, the robot's inverse-kinematics engine typically produces **multiple valid joint-angle solutions** — for a 6-axis robot up to 8 solutions. Each solution places the TCP at the same position and orientation but with the arm in a physically different posture.

**`confdata`** (configuration data) is ABB's mechanism to **unambiguously select one of those solutions**. It is stored inside every `robtarget` in RAPID code and consumed by the motion controller to pick exactly the intended arm posture.

In RobotStudio's SDK it is the `ConfigurationData` struct:

```
confdata = [ cf1, cf4, cf6, cfx ]
```

---

## 2. The Four Configuration Parameters

| Parameter | Data type | Meaning (rotating axis) | Meaning (linear axis) |
|-----------|-----------|-------------------------|-----------------------|
| `cf1`     | `num` (integer) | Quadrant of **axis 1** | Metre interval of axis 1 |
| `cf4`     | `num` (integer) | Quadrant of **axis 4** | Metre interval of axis 4 |
| `cf6`     | `num` (integer) | Quadrant of **axis 6** | Metre interval of axis 6 |
| `cfx`     | `num` (integer) | **Robot-model-dependent** (see §3 and §7) | Metre interval of axis 2 |

### Quadrant Definition (the core formula)

For any **rotating** joint with angle `θ` (in degrees):

```
quadrant(θ) = floor(θ / 90)
```

Examples:

| θ (degrees) | quadrant |
|-------------|----------|
| 0° to 89.9° | 0 |
| 90° to 179.9° | 1 |
| −1° to −89.9° | −1 |
| −90° to −179.9° | −2 |
| 180° to 269.9° | 2 |

> **Important:** `floor` is the mathematical floor (round towards −∞), not truncation.  
> In code: `cf = int(math.floor(theta_deg / 90.0))` (Python) or `(int)std::floor(theta_deg / 90.0)` (C++).

---

## 3. The `cfx` Field: All 8 Configurations Explained

For **IRB 140, 6600, 6650, 7600** (and generally for robots where cfx is active as a posture selector), `cfx` takes integer values **0 through 7**. These 8 values correspond to the 8 distinct arm postures that can place the wrist centre at the same point in space.

The 8 postures are defined by three binary flags:

| Bit | Flag | Value 0 | Value 1 |
|-----|------|---------|---------|
| Bit 2 (weight 4) | Wrist centre relative to **axis 1** | **In front of** axis 1 | **Behind** axis 1 |
| Bit 1 (weight 2) | Wrist centre relative to **lower arm** | **In front of** lower arm | **Behind** lower arm |
| Bit 0 (weight 1) | Sign of **axis 5** angle | **Positive** (θ₅ ≥ 0) | **Negative** (θ₅ < 0) |

Combining these three binary flags:

```
cfx = 4 × B2  +  2 × B1  +  B0
```

Full table:

| cfx | Wrist vs Axis 1 | Wrist vs Lower Arm | θ₅ sign |
|-----|-----------------|--------------------|---------|
| 0 | In front | In front | Positive |
| 1 | In front | In front | Negative |
| 2 | In front | Behind | Positive |
| 3 | In front | Behind | Negative |
| 4 | Behind | Behind (†) | Positive |
| 5 | Behind | In front (†) | Negative |
| 6 | Behind | Behind | Positive |
| 7 | Behind | Behind | Negative |

> (†) The ABB manual shows cfx 4 and 5 as "Behind axis 1, In front of lower arm", cfx 6 and 7 as "Behind axis 1, Behind lower arm". Verify against the exact diagrams on pages 1094–1095 of 3HAC 16581-1 for your specific robot model.

---

## 4. Meaning in Task Space vs Joint Space

### Task-Space Interpretation

"Wrist centre" refers to the intersection of axes 4, 5, and 6 (the spherical wrist centre). The three binary flags describe the **geometry of the arm in Cartesian space**:

- **Wrist in front of / behind axis 1:** Imagine a vertical plane through axis 1 in the direction the robot base is facing. If the wrist centre lies on the same side as the robot's forward direction, it is "in front"; if it has crossed to the opposite side, it is "behind".
- **Wrist in front of / behind lower arm:** This is the classic **elbow-up vs elbow-down** distinction. "In front of lower arm" = elbow-down (the lower arm extends generally toward the wrist); "behind lower arm" = elbow-up (the lower arm bends backwards over the shoulder).
- **θ₅ sign:** For a spherical-wrist robot, each wrist-centre solution admits two wrist orientations related by flipping axis 5. This is the standard "wrist-flip" pair.

### Joint-Space Interpretation

Each of the three bits maps to the sign/range of specific joint angles:

| cfx bit | Joint-space indicator |
|---------|-----------------------|
| B2 (axis 1 side) | Determined by the direction axis 1 points relative to the target. When the robot "reaches backward" (axis 1 rotates past ±90° to face the target from the other side), B2 = 1. |
| B1 (elbow) | Equivalent to the **sign of θ₃** in many robots. `θ₃ > 0` → elbow-down (B1 = 0); `θ₃ < 0` → elbow-up (B1 = 1). (Convention may vary by robot model; validate against your DH parameters.) |
| B0 (wrist flip) | Directly the sign of θ₅: B0 = 0 if θ₅ ≥ 0, B0 = 1 if θ₅ < 0. |

---

## 5. How to Compute cf1 / cf4 / cf6 from Joint Angles

Given a joint-angle solution `[θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]` in **degrees**:

```
cf1 = floor(θ₁ / 90)
cf4 = floor(θ₄ / 90)
cf6 = floor(θ₆ / 90)
```

These are plain quadrant numbers — signed integers. They are **independent of robot model** for rotating axes.

### Example

A solution where `θ₁ = 110°`, `θ₄ = −45°`, `θ₆ = 5°`:

```
cf1 = floor(110 / 90) = floor(1.22) = 1
cf4 = floor(−45 / 90) = floor(−0.5) = −1
cf6 = floor(5 / 90)   = floor(0.055) = 0

→ confdata = [1, −1, 0, cfx]
```

This matches the RAPID example in the manual:
> `VAR confdata conf15 := [1, -1, 0, 0]` — axis 1 in quadrant 1 (90°–180°), axis 4 in quadrant −1 (0° to −90°), axis 6 in quadrant 0 (0°–90°).

---

## 6. How to Compute cfx (0–7) from Joint Angles

This applies only to robots **where cfx encodes the overall arm posture** (IRB 140, 6600, 6650, 7600 etc. — see §8).

### Step-by-step

**Step 1 — Compute wrist centre position** from FK of joints 1–3.

For a robot with DH parameters, the wrist centre `p_wc` is:

```
p_wc = T_base_to_joint3 × [0, 0, d6, 1]ᵀ
```

where `d6` is the distance from joint 3 origin to wrist centre along axis 6.

**Step 2 — B2: Wrist relative to axis 1**

Project the wrist centre onto the XY plane (axis 1 rotates about Z). Compute the angle from axis 1's current pointing direction to the wrist:

```python
# axis1_dir = [cos(θ₁), sin(θ₁), 0]   (unit vector axis 1 faces)
# p_wc_xy   = [p_wc.x, p_wc.y]
dot = p_wc_xy · axis1_dir
B2 = 1 if dot < 0 else 0
```

Alternatively (practical shortcut): when the robot "flips" to reach the point from the back, θ₁ is typically displaced by ~180° from what a forward reach would use. So:

```
B2 = 1  if the robot used a "backward" axis-1 solution
B2 = 0  if the robot used a "forward" axis-1 solution
```

In practice this is identified by the sign of the cross product or the relationship between axis 1 and axes 2–3.

**Step 3 — B1: Elbow configuration (wrist relative to lower arm)**

For standard 6R ABB robots, this corresponds to the **sign of the "elbow angle"**, which is directly related to `θ₃`:

```
B1 = 1 if θ₃ < 0   (elbow-up / "behind lower arm")
B1 = 0 if θ₃ ≥ 0   (elbow-down / "in front of lower arm")
```

> **Verify this for your specific robot model** — the DH convention can flip the sign. Test with known configurations from the manual diagrams.

**Step 4 — B0: Axis-5 sign**

```
B0 = 1 if θ₅ < 0
B0 = 0 if θ₅ ≥ 0
```

**Step 5 — Assemble cfx**

```
cfx = 4 × B2  +  2 × B1  +  B0
```

---

## 7. Why IRB 1400 Uses Only cf1, cf4, cf6 (no cfx)

### The Short Answer

For the IRB 1400, the **quadrant values of axes 1, 4, and 6 already uniquely identify every one of the 8 possible arm configurations**. The `cfx` posture-selector is redundant because knowing which 90° sector those three axes are in implicitly encodes all three binary flags (B2, B1, B0) described in §3.

### The Evidence from the Manual

The ABB RAPID Reference Manual (3HAC 16581-1, p. 1096) states explicitly:

> **Robot configuration data for IRB 1400, 2400, 3400, 4400, 6400**  
> *Only the three configuration parameters `cf1`, `cf4`, and `cf6` are used.*

In contrast, for the IRB 140 (p. 1094):

> **Robot configuration data for IRB 140, 6600, 6650, 7600**  
> `cf1` is the quadrant number for axis 1.  
> `cf4` is the quadrant number for axis 4.  
> `cf6` is the quadrant number for axis 6.  
> `cfx` is used to select one of eight possible robot configurations numbered from 0 through 7.

For IRB 140, **all four fields are needed**. For IRB 1400, **only three**.

### Why Does This Geometric Difference Arise?

The distinction comes from the **kinematic geometry and joint limits** of each robot:

1. **IRB 140 (small, agile):** Joint 1 has a wide range (±360° or more). The robot can point axis 1 in the same quadrant while physically placing the arm in two entirely different postures (e.g., elbow-up vs elbow-down can both leave axis 1 in quadrant 0). Therefore, `cf1` alone does **not** fully resolve the arm configuration. The separate `cfx` (0–7) integer is needed to explicitly state the posture.

2. **IRB 1400 (medium-payload, industrial):** Due to the kinematic proportions of the IRB 1400's links and its practical joint limits, each combination of `(cf1, cf4, cf6)` maps to **exactly one physically reachable arm posture**. The information encoded by `cfx`'s three bits (B2, B1, B0) is implicitly captured by the combination of axis-1, axis-4, axis-6 quadrant values. No additional integer is required.

   Concretely: when the IRB 1400's axis 1 is in quadrant 1 (90°–180°), its physical geometry constrains the elbow and wrist-flip solutions to specific axis-4 and axis-6 quadrants. There is no valid configuration where two different cfx values correspond to the same `(cf1, cf4, cf6)` triple.

3. **The `cfx` field in IRB 1400's RAPID data is still present** (it is part of the `confdata` data type definition), but it is **ignored by the IRB 1400 motion controller**. When programming, its value does not affect which solution is selected. RobotStudio typically stores it as `0`.

### Summary Table

| Robot | cf1 | cf4 | cf6 | cfx used as posture? |
|-------|-----|-----|-----|----------------------|
| IRB 140, 6600, 6650, 7600 | ✓ | ✓ | ✓ | ✓ (posture 0–7) |
| **IRB 1400, 2400, 3400, 4400, 6400** | ✓ | ✓ | ✓ | ✗ (ignored) |
| IRB 5400 | ✓ (axis 1) | ✓ (axis 4) | ✓ (axis 6) | ✓ (quadrant of axis 5) |

---

## 8. Robot Model Reference Table

| Robot Model | cf1 | cf4 | cf6 | cfx |
|-------------|-----|-----|-----|-----|
| IRB 140, 6600, 6650, 7600 | Axis 1 quadrant | Axis 4 quadrant | Axis 6 quadrant | Posture selector 0–7 |
| **IRB 1400, 2400, 3400, 4400, 6400** | Axis 1 quadrant | Axis 4 quadrant | Axis 6 quadrant | **Not used** |
| IRB 340 | Not used | Axis 4 quadrant | Not used | Not used |
| IRB 260, 660 | Not used | Not used | Axis 6 quadrant | Not used |
| IRB 5400 | Axis 1 quadrant | Axis 4 quadrant | Axis 6 quadrant | Axis 5 quadrant |
| IRB 5404, 5406 | Rotating axis 1 | Not used | Not used | Rotating axis 2 |
| IRB 5413, 5414, 5423 | Linear axis 1 | Rotating axis 4 | Not used | Linear axis 2 |
| IRB 840 | Linear axis 1 | Rotating axis 4 | Not used | Linear axis 2 |

---

## 9. EAIK Returns 8 Solutions: Mapping to ECFX

This section is the **core of the EAIK-to-RobotStudio replication effort**. It is split into two fully independent phases:

- **Phase 1 — ECFX Labelling:** Attach a configuration label to every EAIK solution so it speaks the same language as RobotStudio's `confdata`.
- **Phase 2 — Solution Selection:** Decide, from all labelled solutions, which one RobotStudio would actually choose — and replicate that decision rule exactly.

---

### What EAIK Gives You (the raw material)

EAIK's `calculate_IK()` returns an `IK_Solution` struct:

```cpp
struct IK_Solution {
    std::vector<std::vector<double>> Q;   // up to 8 solutions; each element is [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] in RADIANS
    std::vector<bool> is_LS_vec;          // parallel array: true = least-squares/approximate, false = exact
};
```

Key facts about this output:
- Angles are in **radians** — must be converted to degrees before quadrant maths.
- Order is **deterministic** (same input → same solution order) but has **no quadrant-based or industrial-convention ordering**.
- EAIK attaches **no configuration label** to any solution. That is entirely your job.

---

### Phase 1: ECFX Labelling

**Goal:** For each of the up to 8 EAIK solutions, compute a `(cf1, cf4, cf6, cfx)` tuple — called the **ECFX label** — that is directly comparable to ABB's `confdata`.

This phase is **pure computation** with no selection logic involved. You are simply annotating each solution.

#### The ECFX Formula (for rotating axes, IRB 1400 and similar)

```
For solution[i] = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]  (radians from EAIK):

θ₁_deg = θ₁ × (180 / π)
θ₄_deg = θ₄ × (180 / π)
θ₆_deg = θ₆ × (180 / π)

cf1_i = floor(θ₁_deg / 90)
cf4_i = floor(θ₄_deg / 90)
cf6_i = floor(θ₆_deg / 90)
cfx_i = 0   ← for IRB 1400 this field is unused; set to 0 by convention

ECFX_label[i] = (cf1_i, cf4_i, cf6_i, cfx_i)
```

> **`floor` = mathematical floor (towards −∞), not truncation.**  
> Python: `math.floor(x)` | C++: `(int)std::floor(x)`

#### Output of Phase 1

Least-squares solutions are stripped by the pipeline before this point. You now have a table of only exact solutions:

| Solution index | θ₁° | θ₄° | θ₆° | cf1 | cf4 | cf6 | cfx |
|---------------|-----|-----|-----|-----|-----|-----|-----|
| 0 | 20.1 | −44.7 | 5.2 | 0 | −1 | 0 | 0 |
| 1 | 20.1 | 135.2 | −174.8 | 0 | 1 | −2 | 0 |
| … | … | … | … | … | … | … | … |

Each row is an independently valid IK solution tagged with RobotStudio's language. **No solution has been chosen yet.** Phase 1 is now complete.

#### Uniqueness guarantee

For a given target pose and robot (IRB 1400), the `(cf1, cf4, cf6)` triple will be **unique across all 8 solutions**. No two rows share the same triple. This is what makes quadrant-based selection unambiguous — if you are looking for `cf1=0, cf4=−1, cf6=0`, at most one solution satisfies it.

---

### Phase 2: Solution Selection — ConfigurationMode

#### The ABB API Evidence

The `RsMoveInstruction` class in the RobotStudio SDK exposes:

```csharp
public Task<JumpResult> JumpToAsync(bool updateController, ConfigurationMode configurationMode)
```

> *"Specifies how the arm configuration stored in the target shall be used. In this case this method returns true only if the mechanism can move to the specified target with its specified arm configuration. Only valid for RsRobTargets."*

This is the **direct API proof** of how RobotStudio moves to a target waypoint. The `ConfigurationMode` parameter is **passed per-instruction** and controls exactly which IK solution is selected from all valid candidates. It is not a global setting — it is a per-move decision.

The `ConfigurationMode` enum (ABB.Robotics.RobotStudio.Stations):

```csharp
public enum ConfigurationMode
{
    Compliant,  // match stored config within ±1 for Cf1, Cf4, Cf6; Cfx must be equal
    Exact,      // must match stored configuration exactly
    Ignore      // stored configuration is ignored entirely
}
```

---

#### The Three Modes — What RobotStudio Does and How EAIK Replicates It

---

##### Mode: `Compliant` *(closest to RobotStudio's default path-following behaviour)*

**What RobotStudio does:**  
The solution's ECFX values must be within ±1 of the **previous waypoint's** ECFX values on cf1, cf4, cf6, and exactly equal on cfx. This is directly the `IsCompatible()` check from `ConfigurationData`:

```csharp
public static bool IsCompatible(ConfigurationData cnf1, ConfigurationData cnf2)
// Returns: True if Cf1, Cf4, Cf6 differ by at most one and Cfx are equal.
```

The ±1 tolerance exists because a joint can legitimately cross a quadrant boundary during smooth continuous motion. Compliant mode keeps the robot in a geometrically similar posture to where it was, while allowing gradual quadrant drift.

**What it means physically:** The robot will not make large arm-posture jumps between waypoints. It stays in roughly the same configuration family as the previous move.

**How EAIK replicates it:**

```
INPUTS: candidates     = Phase 1 ECFX-labelled solutions (exact only, LS already stripped)
        previous_q     = joint angles selected at previous waypoint  [θ₁..θ₆] radians
        previous_ecfx  = ECFX label of previous_q: (cf1_p, cf4_p, cf6_p, cfx_p)

IF first waypoint (no previous_q):
    → _select_min_norm(candidates)   # see below; nothing to compare against

ELSE:
    STEP 1 — Filter by Compliant rule:
        compliant = [ s for s in candidates
                      if abs(s.cf1 - cf1_p) <= 1
                      and abs(s.cf4 - cf4_p) <= 1
                      and abs(s.cf6 - cf6_p) <= 1
                      and s.cfx == cfx_p ]

    STEP 2 — Select from filtered set:
        if compliant is not empty:
            → _select_closest(compliant, previous_q)   # argmin joint-space distance
        else:
            → _select_min_norm(candidates)             # fallback: no compliant solution exists
                                                       # (log a warning — configuration jump occurred)

    STEP 3 — Update anchor:
        anchor = selected.q
        anchor_ecfx = selected.ecfx
```

---

##### Mode: `Exact` *(documented; NOT used in this pipeline)*

**What RobotStudio does:**  
The solution's ECFX values must exactly match the stored confdata in the robtarget on all four fields. Returns false/failure if no such solution exists.

**Why not used here:**  
Exact mode requires a stored target confdata to match against — it is used when replaying pre-programmed RAPID targets where the confdata was explicitly written at teach time. In this pipeline, both EAIK and RobotStudio are computing solutions autonomously from the same Cartesian waypoints; there is no pre-stored confdata to match against. Exact mode is not applicable.

---

##### Mode: `Ignore` *(two sub-behaviours controlled by `solution_selection` in config)*

**What RobotStudio does:**  
The stored target configuration is completely ignored. RobotStudio picks freely among all valid IK solutions with no ECFX filtering.

**EAIK replication — two options selectable via `solution_selection` in `ik_config.yaml`:**

**`Ignore / closest`** — Replicates path-continuous free selection:

```
IF first waypoint:
    → _select_min_norm(candidates)

ELSE:
    → _select_closest(candidates, previous_q)
       = argmin over candidates of:  sqrt( Σ_j (q[j] - previous_q[j])² )
    → anchor = selected.q
```

This is the chained nearest-neighbour rule: pick whichever exact solution is closest in joint space to the previous waypoint's selected joints, with no quadrant filtering. Keeps the path smooth in joint space.

**`Ignore / min_norm`** — Stateless neutral-pose selection:

```
FOR every waypoint (including first):
    → _select_min_norm(candidates)
       = argmin over candidates of:  Σ_j |q[j]|
         (solution whose joint angles are collectively closest to zero)

No anchor. No chaining. Stateless per-waypoint.
```

This selects the most neutral, zero-biased posture at every waypoint independently. Useful for offline analysis or when path continuity is not a concern.

---

#### First Waypoint Rule (All Modes)

For the very first waypoint in a path, there is no previous q to compare against, and no previous ECFX to filter by. All modes that would otherwise use `previous_q` fall through to `_select_min_norm`:

```
IF first_waypoint:
    selected = _select_min_norm(candidates)
    anchor   = selected.q
    anchor_ecfx = selected.ecfx
```

From waypoint 1 onward, the selected solution becomes the anchor for the next step.

---

#### Mode Summary Table

| Mode | Config value | First waypoint | Subsequent waypoints | ECFX filter? |
|------|-------------|----------------|----------------------|--------------|
| **Compliant** | `configuration_mode: Compliant` | `_select_min_norm` | Filter ±1, then `_select_closest` | Yes — ±1 on cf1/cf4/cf6, exact cfx |
| **Exact** | N/A — not used | — | — | Yes — exact match (requires stored confdata) |
| **Ignore/closest** | `configuration_mode: Ignore` + `solution_selection: closest` | `_select_min_norm` | `_select_closest(previous_q)` | No |
| **Ignore/min_norm** | `configuration_mode: Ignore` + `solution_selection: min_norm` | `_select_min_norm` | `_select_min_norm` | No |

---

#### Updated `ik_config.yaml` (EAIK section)

```yaml
ik_parameters:
  # ...existing parameters...

  # =========================================================================
  # --- EAIK-specific parameters ---
  
  # ConfigurationMode — controls which IK solution is selected from all exact candidates.
  # Mirrors ABB RobotStudio's ConfigurationMode enum (RsMoveInstruction.JumpToAsync).
  #
  #   "Compliant" - Solution must be within ±1 quadrant of the previous waypoint's
  #                 cf1, cf4, cf6, and cfx must be exactly equal.
  #                 Replicates ABB Compliant mode (IsCompatible check).
  #                 First waypoint always uses min_norm (no previous config available).
  #
  #   "Ignore"    - Stored configuration is ignored entirely.
  #                 Sub-behaviour controlled by solution_selection below.
  #
  # Note: "Exact" mode is NOT supported — it requires a pre-stored confdata target
  # which does not exist in autonomous IK solving.
  configuration_mode: "Compliant"

  # Solution selection strategy for Ignore mode (ignored when configuration_mode is Compliant):
  #   "closest"  - Pick solution closest to previous q (chained nearest-neighbour).
  #                Best for trajectory continuity. First waypoint uses min_norm.
  #   "min_norm" - Always pick solution with smallest sum of |joint angles|.
  #                Stateless, neutral-pose biased. Used for every waypoint including first.
  solution_selection: "closest"

  # FK verification tolerances
  fk_pos_tolerance_m: 1.0e-3
  fk_rot_tolerance_deg: 0.02
```

---

#### Clean Pseudocode Reference

```python
def _select_min_norm(candidates):
    # Pick solution with smallest sum of absolute joint angles (most neutral posture)
    return min(candidates, key=lambda s: sum(abs(q) for q in s.q))

def _select_closest(candidates, previous_q):
    # Pick solution with smallest Euclidean distance in joint space from previous_q
    return min(candidates, key=lambda s: sum((s.q[j] - previous_q[j])**2 for j in range(6)))

def _pick_best(candidates, previous_q, previous_ecfx, configuration_mode, solution_selection, is_first):

    if is_first:
        return _select_min_norm(candidates)

    if configuration_mode == "Compliant":
        compliant = [
            s for s in candidates
            if abs(s.cf1 - previous_ecfx.cf1) <= 1
            and abs(s.cf4 - previous_ecfx.cf4) <= 1
            and abs(s.cf6 - previous_ecfx.cf6) <= 1
            and s.cfx == previous_ecfx.cfx
        ]
        if compliant:
            return _select_closest(compliant, previous_q)
        else:
            log_warning("No compliant solution found — configuration jump at this waypoint")
            return _select_min_norm(candidates)

    elif configuration_mode == "Ignore":
        if solution_selection == "closest":
            return _select_closest(candidates, previous_q)
        else:  # "min_norm"
            return _select_min_norm(candidates)

def select_solutions_for_path(waypoints, configuration_mode, solution_selection):
    anchor_q    = None
    anchor_ecfx = None
    results     = []

    for k, waypoint in enumerate(waypoints):
        candidates = compute_ecfx_labels(EAIK.calculate_IK(waypoint))  # Phase 1; LS already stripped

        if not candidates:
            raise Exception(f"No exact IK solution for waypoint {k}")

        is_first = (k == 0)
        best = _pick_best(candidates, anchor_q, anchor_ecfx,
                          configuration_mode, solution_selection, is_first)

        results.append(best)
        anchor_q    = best.q
        anchor_ecfx = best.ecfx   # update for next waypoint's Compliant filter

    return results
```

---

## 10. ECFX for IRB 1400: Exact Formulas

For the **IRB 1400** (and IRB 2400, 3400, 4400, 6400), `cfx` is not used. Only `cf1`, `cf4`, `cf6` are needed.

### Input

```
solution[i] = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]   (in radians, from EAIK)
```

### Step 1: Convert to Degrees

```
θ₁_deg = θ₁ × (180 / π)
θ₄_deg = θ₄ × (180 / π)
θ₆_deg = θ₆ × (180 / π)
```

### Step 2: Apply Quadrant Formula

```
cf1 = floor(θ₁_deg / 90)
cf4 = floor(θ₄_deg / 90)
cf6 = floor(θ₆_deg / 90)
```

Use **mathematical floor** (towards −∞):

```python
import math
cf1 = math.floor(theta1_deg / 90.0)
cf4 = math.floor(theta4_deg / 90.0)
cf6 = math.floor(theta6_deg / 90.0)
```

```cpp
#include <cmath>
int cf1 = (int)std::floor(theta1_deg / 90.0);
int cf4 = (int)std::floor(theta4_deg / 90.0);
int cf6 = (int)std::floor(theta6_deg / 90.0);
```

### Step 3: Build the ECFX Label

```
ECFX_label = (cf1, cf4, cf6, cfx=0)
```

This triple **uniquely identifies** one of the exact solutions for the IRB 1400. Pass this label and the raw `q` vector into `_pick_best()` as described in §9 Phase 2.

### Notes on `cfx` Field for IRB 1400

- For IRB 1400: always set `cfx = 0` in the ECFX label. It is not used in filtering or selection.
- When writing a `robtarget` back to RobotStudio for IRB 1400, set `cfx = 0`.

---

## 11. Full Worked Example

### Setup

Robot: IRB 1400  
Target pose → EAIK returns 8 solutions (radians):

| # | θ₁ (rad) | θ₄ (rad) | θ₆ (rad) | θ₁° | θ₄° | θ₆° | cf1 | cf4 | cf6 |
|---|----------|----------|----------|-----|-----|-----|-----|-----|-----|
| 0 | 0.35 | −0.78 | 0.09 | 20.1° | −44.7° | 5.2° | 0 | −1 | 0 |
| 1 | 0.35 | 2.36 | −3.05 | 20.1° | 135.2° | −174.8° | 0 | 1 | −2 |
| 2 | 0.35 | −0.78 | −3.05 | 20.1° | −44.7° | −174.8° | 0 | −1 | −2 |
| 3 | 0.35 | 2.36 | 0.09 | 20.1° | 135.2° | 5.2° | 0 | 1 | 0 |
| 4 | −2.79 | 0.78 | −3.05 | −159.8° | 44.7° | −174.8° | −2 | 0 | −2 |
| 5 | −2.79 | −2.36 | 0.09 | −159.8° | −135.2° | 5.2° | −2 | −2 | 0 |
| 6 | −2.79 | 0.78 | 0.09 | −159.8° | 44.7° | 5.2° | −2 | 0 | 0 |
| 7 | −2.79 | −2.36 | −3.05 | −159.8° | −135.2° | −174.8° | −2 | −2 | −2 |

### Target from RobotStudio

```rapid
robtarget p1 := [[x,y,z],[q1,q2,q3,q4],[0,-1,0,0],[9E9,9E9,9E9,9E9]];
```

Confdata = `[0, −1, 0, 0]` → target_cf1=0, target_cf4=−1, target_cf6=0.

### Matching

Search the table above for (cf1=0, cf4=−1, cf6=0) → **Solution #0**.

Selected joint angles: `[0.35, θ₂, θ₃, −0.78, θ₅, 0.09]` rad.

---

## 12. Implementation Checklist

```
□ EAIK returns Q in radians — convert to degrees before applying quadrant formula.
□ Use mathematical floor(), not truncation/round/int-cast alone.
□ Compute cf1, cf4, cf6 from θ₁, θ₄, θ₆ only (axes 2, 3, 5 not used for IRB 1400).
□ For IRB 1400: always set cfx = 0 in the ECFX label.
□ Least-squares solutions (is_LS_vec == true) are stripped BEFORE Phase 1 labelling.
□ Set configuration_mode in ik_config.yaml: "Compliant" or "Ignore".
□ If Ignore, also set solution_selection: "closest" or "min_norm".
□ "Exact" mode is NOT implemented — it requires a pre-stored confdata input.

For Compliant mode:
□ Store anchor_ecfx (cf1, cf4, cf6, cfx) from the previous selected solution.
□ Filter candidates: |cf1_i - cf1_prev| ≤ 1 AND |cf4_i - cf4_prev| ≤ 1
                     AND |cf6_i - cf6_prev| ≤ 1 AND cfx_i == cfx_prev
□ Select from filtered set using _select_closest(previous_q).
□ If filter yields empty set: fall back to _select_min_norm and log a warning.

For Ignore/closest:
□ Select using _select_closest(previous_q) — no ECFX filtering.

For Ignore/min_norm:
□ Select using _select_min_norm — no anchor, stateless.

First waypoint (all modes):
□ No previous_q exists → always use _select_min_norm regardless of mode.
□ Set anchor_q = selected.q and anchor_ecfx = selected.ecfx for waypoint 1 onward.

Comparison with RobotStudio:
□ After selection, compare ECFX label of selected solution against RobotStudio's
  stored confdata for the same waypoint.
□ Mismatch on first waypoint → starting posture assumption differs; try adjusting
  the initial anchor (e.g., use known pre-path MoveJ joint values instead of zeros).
□ Mismatch mid-path in Compliant mode → likely a configuration jump where no
  compliant solution existed; check if RobotStudio issued a config-change warning.
```

---

## Appendix: Quadrant Boundary Reference

| Quadrant n | Range (degrees) |
|------------|-----------------|
| −3 | −270° to −180° (exclusive lower) |
| −2 | −180° to −90° |
| −1 | −90° to 0° |
| 0 | 0° to 90° |
| 1 | 90° to 180° |
| 2 | 180° to 270° |
| 3 | 270° to 360° |

> Boundary angles (0°, 90°, 180°, −90°, etc.) belong to the **higher** quadrant because `floor(90/90) = floor(1.0) = 1`.

---

*Document synthesised from: ABB 3HAC 16581-1 Rev. J (RAPID Reference Manual), ABB RobotStudio SDK `ConfigurationData` struct, ABB RobotStudio SDK `RsMoveInstruction` class (`JumpToAsync` / `ConfigurationMode`), and EAIK codebase (OstermD/EAIK).*
