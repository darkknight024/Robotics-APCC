# Debug Plot Visual Reference

## Quick Guide to New Debug Plots

### 📊 Trajectory-Level Plot: `ik_failure_analysis.png`

**2 Plots Side-by-Side**

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  Distance from Robot Base           │  3D Trajectory                      │
│  (All Waypoints)                    │  (with Failed Waypoints)            │
│                                     │                                     │
│  • Green line: trajectory path      │  • Green path: reachable waypoints  │
│  • Red X: failed waypoints          │  • Red X: failed waypoints          │
│  • Shows spatial context            │  • Blue dot: robot base             │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

**Purpose**: Understand WHERE in workspace failures occur

---

### 🔬 Per-Waypoint Debug Plot: `ik_debug_<traj>_<wp>.png`

**5-Panel Detailed Analysis**

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Panel 1: IK Convergence (FULL WIDTH)                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Residual Norm vs Iteration                                              │
│  • Shows HOW solver converged (or failed to converge)                   │
│  • Green line: tolerance threshold                                       │
│  • Red X: where solver stopped                                           │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  Panel 2: Min Singular Value        │  Panel 3: Condition Number          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  σ_min vs Iteration                 │  κ = σ_max/σ_min vs Iteration      │
│  • Singularity detection            │  • Problem conditioning             │
│  • Orange line: threshold (0.01)    │  • Higher = worse conditioned       │
│  • Status: SINGULAR or OK           │  • Shows if near singularity        │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  Panel 4: Damping Parameter         │  Panel 5: Failure Summary           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  λ vs Iteration                     │  Text Panel:                        │
│  • Solver adaptation strategy       │  • Target position & quaternion     │
│  • Blue line: λ₀ (initial)          │  • IK solver status                 │
│  • Red line: λ_max (limit)          │  • Jacobian metrics                 │
│  • High damping = struggling        │  • Joint limit violations           │
│                                     │  • Distance from prev config        │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

**Purpose**: Understand WHY this specific waypoint failed

---

## Reading the Plots

### Panel 1: Convergence Curve

**Good Convergence (but timeout)**
```
Residual
  10⁰  ●
       │ ●
       │  ●
  10⁻¹ │   ●
       │    ●●
  10⁻² │      ●●
       │        ●●
  10⁻³ │          ●● ← Got close!
       │            ×  ← Stopped here (max iter)
  10⁻⁴ ├─ ─ ─ ─ ─ ─ ← Tolerance
       └──────────────
        Iterations
```
→ **Diagnosis**: Solver working well, just needs more iterations or relaxed tolerance

**Stuck at Singularity**
```
Residual
  10⁰  ●
       │ ●●
  10⁻¹ │   ●
       │    ●●●●●●●●●●●●●●●●●● ← Stuck!
  10⁻² │                      ×
       └──────────────
        Iterations
```
→ **Diagnosis**: Hit singularity, residual plateaued

**Divergence**
```
Residual
  10¹        ●
       ●    ●
  10⁰   ● ●
       ●  ← Getting worse!
  10⁻¹ ●
       └──────────────
        Iterations
```
→ **Diagnosis**: Bad initialization or infeasible pose

### Panel 2: Singularity Detection

**Not Singular**
```
σ_min
  10⁻¹ ●●●●●●●●●●●●●●●
       │                    All above threshold
  10⁻² ├─ ─ ─ ─ ─ ─ ← Threshold (0.01)
       │
  10⁻³ │
       └──────────────
        Iterations
```
→ **Diagnosis**: Singularity NOT the problem

**Near Singularity**
```
σ_min
  10⁻¹ ●●
       │  ●●
  10⁻² ├─ ●●●─ ─ ─ ← Threshold (0.01)
       │    ●●●●●●  ← Below threshold!
  10⁻³ │
       └──────────────
        Iterations
```
→ **Diagnosis**: Singularity IS the problem

### Panel 4: Damping Behavior

**Well-Behaved Problem**
```
Damping
  10¹  ├─ ─ ─ ─ ─ ─ ← λ_max
       │
  10⁰  │
       │
  10⁻¹ │
       │
  10⁻² │
       │
  10⁻³ ├●●●●●●●●●●●● ← Stayed low
       └──────────────
        Iterations
```
→ **Diagnosis**: Problem well-conditioned, damping not limiting

**Struggling Solver**
```
Damping
  10¹  ├●●●●●●●●●●●● ← Hit max!
       │
  10⁰  │
       │  ●●●
  10⁻¹ │ ●
       │●
  10⁻² │
  10⁻³ ├
       └──────────────
        Iterations
```
→ **Diagnosis**: High damping needed, problem ill-conditioned

---

## Naming Convention

### File Naming Pattern
```
ik_debug_<trajectory_number>_<waypoint_index>.png
         └─────┬──────┘      └────────┬──────────┘
               │                      │
          1-indexed                 0-indexed
        (matches trajectory_N)   (array index)
```

### Examples
- `ik_debug_18_0.png` → Trajectory 18, waypoint index 0
- `ik_debug_18_5.png` → Trajectory 18, waypoint index 5
- `ik_debug_3_127.png` → Trajectory 3, waypoint index 127

---

## Common Patterns

### Pattern 1: False Negative (Should be Reachable)
**Indicators:**
- ✓ Panel 1: Residual decreasing smoothly, gets < 1e-3
- ✓ Panel 2: σ_min > 0.01 (not singular)
- ✓ Panel 4: Damping stays low
- ✓ Panel 5: "max_iter_exceeded"

**Action**: Increase max_iterations or tolerance

### Pattern 2: True Singularity
**Indicators:**
- ✗ Panel 1: Residual plateaus at high value
- ✗ Panel 2: σ_min < 0.01 (singular!)
- ✗ Panel 3: Condition number very high (>1000)
- ✗ Panel 4: Damping hits max

**Action**: Modify trajectory to avoid singular configuration

### Pattern 3: Joint Limit Violation
**Indicators:**
- ✗ Panel 1: Residual moderate, not converging
- ✓ Panel 2: σ_min OK
- ✗ Panel 5: Shows joint limit violations

**Action**: Adjust knife pose or trajectory

### Pattern 4: Configuration Jump
**Indicators:**
- ✗ Panel 1: High initial residual, diverges
- ✗ Panel 5: Large "distance from prev config"

**Action**: Increase waypoint density

---

## Quick Diagnosis Checklist

For each failed waypoint plot:

1. **Check Panel 1 (Convergence)**:
   - [ ] Is residual decreasing?
   - [ ] Final residual < 1e-3?
   - [ ] Plateau or continuing to improve?

2. **Check Panel 2 (Singularity)**:
   - [ ] Is σ_min < 0.01?
   - [ ] Stable or changing?

3. **Check Panel 4 (Damping)**:
   - [ ] Hitting max (10)?
   - [ ] Staying low (< 0.1)?

4. **Check Panel 5 (Summary)**:
   - [ ] Any joint limit violations?
   - [ ] Large config space jump?
   - [ ] Reasonable workspace distance?

5. **Identify Pattern**:
   - [ ] Matches Pattern 1 (false negative)?
   - [ ] Matches Pattern 2 (singularity)?
   - [ ] Matches Pattern 3 (joint limits)?
   - [ ] Matches Pattern 4 (big jump)?

---

## Where to Start

1. **First**: Look at `ik_failure_analysis.png`
   - Understand spatial distribution
   - Are failures clustered or scattered?

2. **Then**: Pick a few failed waypoints
   - Start with ones you KNOW should work
   - Open their `ik_debug_<traj>_<wp>.png`
   - Follow checklist above

3. **Finally**: Identify common patterns
   - Do all fail for same reason?
   - Systematic issue or random?

---

## Pro Tips

💡 **Compare neighboring waypoints**: If WP 5 fails but WP 4 and 6 succeed, look for what's different

💡 **Check convergence rate**: Slow convergence (many iterations) suggests ill-conditioning

💡 **Watch damping evolution**: Oscillating damping suggests solver hunting for solution

💡 **Look for correlations**: Does high damping correlate with singularity? With plateau in residual?

💡 **Use Panel 5 as cheat sheet**: All key numbers in one place

---

## Need Help?

If plots show unexpected behavior:
1. Check `UPDATED_DEBUG_FEATURES.md` for interpretation guide
2. Compare with known successful waypoints
3. Look for patterns across multiple failures
4. Consider adjusting IK solver parameters

Happy debugging! 🎯
