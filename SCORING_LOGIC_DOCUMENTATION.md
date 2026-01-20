# Combinatorial Search Scoring Logic Documentation

**Last Updated:** 2026-01-20

This document explains the complete scoring logic for the combinatorial search, including all mathematical formulas, normalization strategies, and aggregation methods.

---

## Table of Contents

1. [Overview](#overview)
2. [Metric Extraction](#metric-extraction)
3. [Aggregation Strategy](#aggregation-strategy)
4. [Normalization Methods](#normalization-methods)
5. [Score Computation](#score-computation)
6. [Robot Ranking](#robot-ranking)
7. [Output Reports](#output-reports)

---

## Overview

The combinatorial search evaluates **R robots × K knife poses × T toolpaths** to find the optimal robot and knife pose combination. The scoring system uses:

- **Raw metrics**: Actual measured values from kinematic analysis
- **Normalized metrics**: Scaled to [0, 1] where 0=best, 1=worst
- **Weighted score**: Final score combining all metrics with configurable weights

---

## Metric Extraction

### Per-Waypoint Metrics

For each waypoint in a trajectory:
- **IK solvability**: Binary (reachable/unreachable)
- **Manipulability**: `w = sqrt(det(J * J^T))` where J is the Jacobian
- **Min Singular Value**: Smallest singular value of Jacobian (singularity proximity)

### Per-Trajectory Metrics

For a trajectory with N waypoints:

```
IK_failure_rate = (N_unreachable / N_total)

singularity_rate = (N_near_singularity / N_total)

mean_manipulability = mean(manipulability[i] for i in reachable waypoints)

min_manipulability = min(manipulability[i] for i in reachable waypoints)

mean_min_singular_value = mean(min_SV[i] for i in reachable waypoints)
```

---

## Aggregation Strategy

### Level 1: Trajectories → Toolpath

A toolpath contains multiple trajectories. We aggregate using **worst-case** for failures and **conservative** for quality:

```python
# For M trajectories in a toolpath:
max_IK_failure_rate = max(trajectory_i.IK_failure_rate for i in 1..M)
max_singularity_rate = max(trajectory_i.singularity_rate for i in 1..M)
min_min_manipulability = min(trajectory_i.min_manipulability for i in 1..M)
mean_mean_manipulability = mean(trajectory_i.mean_manipulability for i in 1..M)
mean_mean_min_singular_value = mean(trajectory_i.mean_min_SV for i in 1..M)
```

**Rationale:**
- **MAX for failures**: If ANY trajectory fails, the toolpath is problematic
- **MIN for min_manipulability**: Bottleneck determines feasibility
- **MEAN for averages**: Overall quality across all trajectories

### Level 2: Toolpaths → Knife Pose

A (robot, knife_pose) combination is tested on multiple toolpaths. Same aggregation strategy:

```python
# For T toolpaths tested with one knife pose:
max_IK_failure_rate = max(toolpath_j.max_IK_failure_rate for j in 1..T)
max_singularity_rate = max(toolpath_j.max_singularity_rate for j in 1..T)
min_min_manipulability = min(toolpath_j.min_min_manipulability for j in 1..T)
mean_mean_manipulability = mean(toolpath_j.mean_mean_manipulability for j in 1..T)
mean_mean_min_singular_value = mean(toolpath_j.mean_mean_SV for j in 1..T)
```

**Rationale:**
- If knife pose fails on ANY toolpath, it's not universally good
- Bottleneck across all toolpaths determines overall quality

---

## Normalization Methods

All normalized metrics map to [0, 1] where **0 = best, 1 = worst**.

### Method 1: Lower is Better (Direct Normalization)

Used for: IK failure rate, singularity rate

```python
def normalize_metric_lower_better(values):
    min_val = min(values)
    max_val = max(values)
    
    if max_val - min_val < 1e-10:
        return [0.0] * len(values)  # All same → all best
    
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    return normalized
```

**Example:**
```
Raw IK failure rates: [0.0, 0.1, 0.5]
min_val = 0.0, max_val = 0.5
Normalized: [(0.0-0.0)/0.5, (0.1-0.0)/0.5, (0.5-0.0)/0.5] = [0.0, 0.2, 1.0]
```

### Method 2: Higher is Better (INVERTED Normalization)

Used for: Manipulability metrics, singular values

```python
def normalize_metric_higher_better(values):
    min_val = min(values)
    max_val = max(values)
    
    if max_val - min_val < 1e-10:
        return [0.0] * len(values)  # All same → all best
    
    # INVERT: highest raw value → 0 (best), lowest raw value → 1 (worst)
    normalized = [(max_val - v) / (max_val - min_val) for v in values]
    return normalized
```

**Example:**
```
Raw manipulability: [0.01, 0.05, 0.10]
min_val = 0.01, max_val = 0.10
Normalized: [(0.10-0.01)/0.09, (0.10-0.05)/0.09, (0.10-0.10)/0.09] = [1.0, 0.556, 0.0]
                   ↑ worst              ↑ middle              ↑ best
```

**Why Invert?**
- Keeps all normalized metrics on same scale: 0=best, 1=worst
- Simplifies weighted scoring: just sum up all normalized metrics

### Per-Robot Normalization (CRITICAL)

**Normalization is performed SEPARATELY for each robot!**

```python
# For Robot A with knife poses K1, K2, K3:
raw_manip = [0.05, 0.08, 0.10]  # Raw values
normalized = [1.0, 0.4, 0.0]     # Normalized relative to Robot A's range

# For Robot B with knife poses K4, K5, K6:
raw_manip = [0.02, 0.03, 0.04]  # Raw values (different range!)
normalized = [1.0, 0.5, 0.0]     # Normalized relative to Robot B's range
```

**Implication:**
- Normalized scores are ONLY valid for comparing knife poses within the SAME robot
- Cross-robot comparison MUST use raw metrics
- Robot ranking uses raw metrics, NOT normalized scores

---

## Score Computation

### Weighted Score Formula

```python
def compute_weighted_score(
    raw_IK_failure_rate,      # RAW [0,1] - NOT normalized
    norm_singularity_rate,    # Normalized [0,1]
    norm_min_manipulability,  # Normalized [0,1] (inverted)
    norm_mean_manipulability, # Normalized [0,1] (inverted)
    norm_mean_min_SV,        # Normalized [0,1] (inverted)
    weights
):
    # Base weighted sum
    raw_score = (
        weights['w_IK_failure_rate'] * raw_IK_failure_rate +
        weights['w_singularity_rate'] * norm_singularity_rate +
        weights['w_min_manipulability'] * norm_min_manipulability +
        weights['w_mean_manipulability'] * norm_mean_manipulability +
        weights['w_mean_min_singular_value'] * norm_mean_min_SV
    )
    
    # Normalize by total weight
    total_weight = sum(weights.values())
    normalized_score = raw_score / total_weight
    
    # CRITICAL: Add +1.0 penalty for ANY IK failure
    if raw_IK_failure_rate > 0:
        normalized_score += 1.0
    
    return normalized_score
```

### Default Weights

```python
DEFAULT_WEIGHTS = {
    'w_IK_failure_rate': 50.0,
    'w_singularity_rate': 25.0,
    'w_min_manipulability': 10.0,
    'w_mean_manipulability': 10.0,
    'w_mean_min_singular_value': 5.0,
}
# Total: 100.0
```

### Feasibility Penalty (CRITICAL)

**Any IK failure results in +1.0 penalty:**

```python
if raw_IK_failure_rate > 0:
    normalized_score += 1.0
```

**Effect:**
- All **feasible** combinations (IK failure = 0%) have scores in [0, ~0.5]
- All **infeasible** combinations (IK failure > 0%) have scores in [1.0, 2.0]
- **Guarantees**: `best_infeasible_score > worst_feasible_score`

### Example Calculations

**Example 1: Excellent Feasible Pose**
```
Raw IK failure: 0%
Normalized metrics: [sing=0.0, min_manip=0.0, mean_manip=0.0, SV=0.0]

raw_score = 50*0.0 + 25*0.0 + 10*0.0 + 10*0.0 + 5*0.0 = 0.0
normalized = 0.0 / 100 = 0.00
final = 0.00 + 0 = 0.00 (no penalty) ✅ BEST POSSIBLE
```

**Example 2: Feasible but Poor Quality**
```
Raw IK failure: 0%
Normalized metrics: [sing=1.0, min_manip=1.0, mean_manip=1.0, SV=1.0]

raw_score = 50*0.0 + 25*1.0 + 10*1.0 + 10*1.0 + 5*1.0 = 50.0
normalized = 50.0 / 100 = 0.50
final = 0.50 + 0 = 0.50 (no penalty) ⚠️ BORDERLINE
```

**Example 3: Infeasible (1% IK failure)**
```
Raw IK failure: 0.01 (1%)
Normalized metrics: [sing=0.0, min_manip=0.0, mean_manip=0.0, SV=0.0]

raw_score = 50*0.01 + 25*0.0 + 10*0.0 + 10*0.0 + 5*0.0 = 0.5
normalized = 0.5 / 100 = 0.005
final = 0.005 + 1.0 = 1.005 (penalty) ❌ INFEASIBLE
```

### Score Interpretation

| Score Range | Verdict | Meaning |
|------------|---------|---------|
| 0.00 - 0.25 | ✅ Recommended | Feasible + excellent quality |
| 0.25 - 0.50 | ⚠️ Borderline | Feasible but some issues |
| 0.50 - 0.75 | ❗ Poor | Feasible but poor quality |
| 0.75 - 1.00 | ❌ Infeasible (edge) | Feasible but terrible quality |
| ≥ 1.00 | ❌ Infeasible | Has IK failures |

---

## Robot Ranking

### Strategy

Robot ranking takes the **best knife pose** from each robot and compares them using **raw metrics** (not normalized scores).

```python
# For each robot:
best_knife_pose = min(knife_poses, key=lambda k: k.normalized_score)

# Cross-robot sort by RAW metrics (lexicographic):
robots.sort(key=lambda r: (
    r.max_IK_failure_rate,          # Primary (lower better)
    r.max_singularity_rate,         # Secondary (lower better)
    -r.mean_mean_manipulability,    # Tertiary (higher better)
    -r.min_min_manipulability,      # Quaternary (higher better)
    -r.mean_mean_min_singular_value # Quinary (higher better)
))
```

### Why Not Use Normalized Scores?

**Normalized scores are per-robot and NOT comparable across robots!**

Example:
```
Robot A: Raw manip range [0.05, 0.10] → normalized [1.0, 0.0]
Robot B: Raw manip range [0.02, 0.03] → normalized [1.0, 0.0]

Best pose for Robot A: raw=0.10, norm=0.0
Best pose for Robot B: raw=0.03, norm=0.0

Both have norm=0.0, but Robot A is BETTER (0.10 > 0.03 in raw terms)!
```

### Ranking Example

```
Robot A best: IK=0%, sing=5%, manip=0.08
Robot B best: IK=0%, sing=2%, manip=0.06
Robot C best: IK=1%, sing=0%, manip=0.10

Ranking:
1. Robot B (IK=0% [tie], sing=2% [wins])
2. Robot A (IK=0% [tie], sing=5% [loses to B])
3. Robot C (IK=1% [fails IK check])
```

---

## Output Reports

### CSV Column Structure

All CSV outputs now include BOTH raw and normalized metrics:

```csv
Rank, Knife Pose ID, Score, Verdict,
IK Failure Rate (raw), Singularity Rate (raw), Min Manipulability (raw), ...,
IK Failure Rate (norm), Singularity Rate (norm), Min Manipulability (norm), ...
```

### Interpretation Guide

**For Within-Robot Comparison:**
- Use `Score` column (computed from normalized metrics)
- Use normalized columns to see relative performance

**For Cross-Robot Comparison:**
- IGNORE normalized columns
- Use raw metric columns only
- Compare absolute values directly

### Files Generated

1. **Global Ranking CSV** (`global_ranking.csv`)
   - All (robot, knife) combinations
   - Sorted by score
   - Contains both raw and normalized metrics

2. **Robot Ranking CSV** (`robot_ranking.csv`)
   - One row per robot (best knife pose)
   - Sorted by raw metrics (cross-robot comparison)
   - Contains raw metrics only

3. **Per-Robot CSV** (`per_robot/{robot}/knife_pose_ranking.csv`)
   - All knife poses for one robot
   - Sorted by score
   - Contains both raw and normalized metrics

---

## Summary

### Key Takeaways

1. **Aggregation**: Worst-case for failures, conservative for quality
2. **Normalization**: Per-robot, 0=best, 1=worst (inverted for "higher is better" metrics)
3. **Scoring**: Weighted sum + 1.0 penalty for any IK failure
4. **Robot Ranking**: Uses raw metrics, NOT normalized scores
5. **Output**: All reports now show BOTH raw and normalized values

### Critical Warnings

⚠️ **DO NOT compare normalized metrics across different robots!**

⚠️ **Always use raw metrics for cross-robot comparison!**

⚠️ **IK failure penalty dominates all other metrics!**

---

**End of Documentation**
