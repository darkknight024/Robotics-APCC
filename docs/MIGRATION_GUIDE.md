# Migration Guide - Code Review Fixes

## Overview
This guide helps you migrate existing code and data to use the updated field names and new time-weighted aggregation module.

---

## Breaking Changes

### 1. Field Name Changes (PEP 8 Standardization)

All dataclasses in `combinatorial_search.py` have been updated to use lowercase snake_case naming:

| Old Name | New Name | Classes Affected |
|----------|----------|------------------|
| `IK_failure_rate` | `ik_failure_rate` | TrajectoryMetrics, CombinationResult, AggregatedKnifePoseResult, RobotRankingResult |
| `max_IK_failure_rate` | `max_ik_failure_rate` | CombinationResult, AggregatedKnifePoseResult, RobotRankingResult |
| `n_waypoints` | `num_waypoints` | TrajectoryMetrics |
| `n_trajectories` | `num_trajectories` | CombinationResult |
| `n_toolpaths` | `num_toolpaths` | AggregatedKnifePoseResult, RobotRankingResult |
| `n_successful` | `num_successful` | AggregatedKnifePoseResult |
| `n_knife_poses_evaluated` | `num_knife_poses_evaluated` | RobotRankingResult |

### 2. Velocity Limits Now Required

`analyze_continuity()` in `feasibility_analysis.py` now **requires** `velocity_limits_rad_s` parameter:

**Before:**
```python
# Used hardcoded defaults if None
analyze_continuity(trajectory, joint_angles)
```

**After:**
```python
# Must provide velocity limits explicitly
velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
analyze_continuity(trajectory, joint_angles, velocity_limits_rad_s=velocity_limits)
```

**Action Required:**
1. Add `velocity_limits_rad_s` to all robot configs in `config/robots_config.yaml`
2. Pass velocity limits explicitly in all calls to `analyze_continuity()`

---

## Quick Fixes

### Python Code Migration

**Find and Replace (case-sensitive):**

```python
# Field access
.IK_failure_rate      → .ik_failure_rate
.max_IK_failure_rate  → .max_ik_failure_rate
.n_waypoints          → .num_waypoints
.n_trajectories       → .num_trajectories
.n_toolpaths          → .num_toolpaths
.n_successful         → .num_successful
.n_knife_poses_evaluated → .num_knife_poses_evaluated
```

**Example Migration:**

```python
# OLD CODE:
if result.max_IK_failure_rate > 0.1:
    print(f"High IK failure: {result.max_IK_failure_rate:.2%}")

# NEW CODE:
if result.max_ik_failure_rate > 0.1:
    print(f"High IK failure: {result.max_ik_failure_rate:.2%}")
```

### JSON/CSV Output Migration

If you have existing result files, field names in JSON exports have changed:

**OLD JSON:**
```json
{
  "IK_failure_rate": 0.05,
  "n_waypoints": 100,
  "max_IK_failure_rate": 0.1
}
```

**NEW JSON:**
```json
{
  "ik_failure_rate": 0.05,
  "num_waypoints": 100,
  "max_ik_failure_rate": 0.1
}
```

**Migration Script:** (Python)
```python
import json

def migrate_json_file(old_path, new_path):
    """Migrate old JSON to new field names"""
    with open(old_path, 'r') as f:
        data = json.load(f)
    
    # Recursive field name migration
    def migrate_dict(d):
        if not isinstance(d, dict):
            return d
        
        new_dict = {}
        for key, value in d.items():
            # Migrate key names
            new_key = key
            if 'IK_failure' in key:
                new_key = key.replace('IK_failure', 'ik_failure')
            elif key.startswith('n_'):
                new_key = 'num_' + key[2:]
            
            # Recursively migrate nested dicts
            if isinstance(value, dict):
                new_dict[new_key] = migrate_dict(value)
            elif isinstance(value, list):
                new_dict[new_key] = [migrate_dict(v) if isinstance(v, dict) else v for v in value]
            else:
                new_dict[new_key] = value
        
        return new_dict
    
    migrated_data = migrate_dict(data)
    
    with open(new_path, 'w') as f:
        json.dump(migrated_data, f, indent=2)

# Usage:
migrate_json_file('output/old_results.json', 'output/new_results.json')
```

---

## New Features

### 1. Time-Weighted Aggregation Module

**NEW MODULE:** `utils/time_weighted_aggregation.py`

Provides time-weighted averaging functions per algorithm specification.

**Usage Example:**

```python
from utils.time_weighted_aggregation import (
    compute_time_weighted_average,
    compute_time_weighted_manipulability,
    compute_time_weighted_smoothness,
    aggregate_metrics_time_weighted
)

# Time-weighted manipulability (Level 4)
manipulability_values = np.array([0.1, 0.12, 0.09, 0.11])
segment_durations = np.array([0.5, 1.0, 0.3, 0.8])  # seconds

dexterity_score = compute_time_weighted_manipulability(
    manipulability_values, segment_durations
)

# Time-weighted smoothness (Level 3)
smoothness_cost = compute_time_weighted_smoothness(
    joint_angles_rad, timestamps, velocity_limits_rad_s
)

# Aggregate across multiple trajectories
trajectory_results = [...]  # List of trajectory result dicts
aggregated = aggregate_metrics_time_weighted(trajectory_results)

print(f"Time-weighted dexterity: {aggregated['dexterity_score']:.6f}")
print(f"Time-weighted smoothness: {aggregated['smoothness_cost']:.6f}")
```

### 2. Configurable Safety Bin Size

**NEW CONFIG:** `config/batch_feasibility_config.yaml`

```yaml
ranking:
  safety_bin_size: 10.0  # Adjustable condition number binning
  smoothness_weight: 1.0  # Reserved for future
  dexterity_weight: 1.0   # Reserved for future
```

**Usage:**

```python
from utils import load_feasibility_config

config = load_feasibility_config('config/batch_feasibility_config.yaml')
ranking_config = config.get('ranking', {})
safety_bin_size = ranking_config.get('safety_bin_size', 10.0)

# Use in ranking
from utils.math import compute_safety_tier
safety_tier = compute_safety_tier(max_condition_number, safety_bin_size)
```

### 3. Duplicate Waypoint Filtering

**NEW FEATURE:** Automatic duplicate removal during CSV loading

```python
from utils import load_toolpath_trajectories

# Duplicates are now automatically filtered during loading
trajectories, speeds = load_toolpath_trajectories('toolpath.csv')

# No manual filtering needed - duplicates removed in preprocessing
```

**Logging:**
```
DEBUG: Removed 3 duplicate waypoint(s) from trajectory
```

---

## Testing Your Migration

### 1. Run Unit Tests

```bash
# Test time-weighted averaging
python -m pytest tests/test_time_weighted_aggregation.py

# Test field name consistency
python -m pytest tests/test_dataclass_fields.py

# Test duplicate filtering
python -m pytest tests/test_csv_loader.py
```

### 2. Integration Test

```bash
# Run combinatorial search on test dataset
python combinatorial_search.py \
    --config config/test_config.yaml \
    --output output/migration_test \
    --workers 1

# Verify no errors and check output format
```

### 3. Compare Results

```python
# Compare old vs new rankings
import json

with open('output/old/batch_ranking_summary.json') as f:
    old_results = json.load(f)

with open('output/new/batch_ranking_summary.json') as f:
    new_results = json.load(f)

# Rankings may differ slightly due to time-weighting improvements
print("Old top robot:", old_results['robot_ranking'][0]['robot_name'])
print("New top robot:", new_results['robot_ranking'][0]['robot_name'])
```

---

## Common Issues & Solutions

### Issue 1: AttributeError on Old Field Names

**Error:**
```
AttributeError: 'CombinationResult' object has no attribute 'max_IK_failure_rate'
```

**Solution:**
Update field access to use new name:
```python
# OLD: result.max_IK_failure_rate
# NEW: result.max_ik_failure_rate
```

### Issue 2: Velocity Limits ValueError

**Error:**
```
ValueError: velocity_limits_rad_s is required for continuity analysis
```

**Solution:**
Add velocity limits to robot config or pass explicitly:
```python
velocity_limits = get_robot_config(robot_name).velocity_limits_rad_s
analyze_continuity(trajectory, joint_angles, velocity_limits_rad_s=velocity_limits)
```

### Issue 3: JSON Key Not Found

**Error:**
```
KeyError: 'n_waypoints'
```

**Solution:**
Update JSON parsing to use new field names, or migrate old JSON files using the migration script above.

---

## Rollback Instructions

If you need to temporarily rollback to the old version:

```bash
# 1. Checkout previous commit (before code review fixes)
git checkout <previous_commit_hash>

# 2. Or revert specific files
git checkout HEAD~1 -- combinatorial_search.py
git checkout HEAD~1 -- feasibility_analysis.py
git checkout HEAD~1 -- utils/csv_loader_toolpath.py

# 3. Re-run your analysis
python combinatorial_search.py --config config/my_config.yaml
```

**Warning:** Rollback loses improvements from fixes (time-weighting, angular wrapping, etc.)

---

## Support

If you encounter issues during migration:

1. Check `docs/CODE_REVIEW_FIXES_SUMMARY.md` for detailed fix descriptions
2. Review algorithm specification in `docs/combinatorial_context.md`
3. Run tests to isolate the issue
4. Check logs for specific error messages

---

## Checklist

Use this checklist to verify your migration is complete:

- [ ] Updated all field access from `IK_failure_rate` to `ik_failure_rate`
- [ ] Updated all field access from `max_IK_failure_rate` to `max_ik_failure_rate`
- [ ] Updated all `n_*` fields to `num_*`
- [ ] Added `velocity_limits_rad_s` to all robot configs
- [ ] Updated all `analyze_continuity()` calls to pass velocity limits
- [ ] Migrated or regenerated existing JSON result files
- [ ] Updated any post-processing scripts that parse output files
- [ ] Ran integration tests and verified no errors
- [ ] Compared old vs new rankings to verify improvements
- [ ] Updated documentation and README files

---

**Migration Complete!** 🎉

Your codebase now follows PEP 8 naming conventions, implements correct time-weighted averaging, and has improved robustness for edge cases.

---

*Generated: February 7, 2026*  
*Version: 2.0.0*
