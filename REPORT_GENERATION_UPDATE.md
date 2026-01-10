# Report Generation Update Summary

## Changes Made

The `combinatorial_search.py` script has been enhanced to generate comprehensive, structured reports following the exact specifications.

## New Features

### 1. Verdict System ✅

Implemented the exact verdict rules:

```python
if IK_failure_rate > 0:
    verdict = "❌ Infeasible"
elif final_score < 0.25:
    verdict = "✅ Recommended"
elif final_score < 0.50:
    verdict = "⚠️ Borderline"
elif final_score < 0.75:
    verdict = "❗ Poor"
else:
    verdict = "❌ Infeasible"
```

### 2. Per-Robot Folder Structure

**Old Structure:**
```
per_robot/
├── IRB_1300-7_1.4_knifepose_ranking.csv
├── IRB_1300-7_1.4_detailed_results.json
└── IRB_1300-7_1.4_ranking_plot.png
```

**New Structure:**
```
per_robot/
└── IRB_1300-7_1.4/                        # Robot folder
    ├── knife_pose_ranking.csv             # Main CSV report
    ├── knife_pose_ranking.md              # Markdown report
    ├── metadata.json                      # Summary statistics
    ├── detailed_results.json              # Full data dump
    ├── ranking_plot.png                   # Visual ranking
    └── knife_poses/                       # Per-knife-pose details
        ├── nominal_x-467.8_y-915.8_z520.4_oriA/
        │   ├── toolpath_details.csv       # Per-toolpath breakdown
        │   └── details.json               # Full details
        ├── nominal_x-467.8_y-915.8_z520.4_oriB/
        │   ├── toolpath_details.csv
        │   └── details.json
        └── ...
```

### 3. Enhanced CSV Format

**New Columns:**
- `Rank` - Rank within robot
- `Knife Pose ID` - Pose identifier
- `Score` - Normalized score (3 decimals)
- `IK Failure Rate` - Max IK failure rate (2 decimals)
- `Singularity Rate` - Max singularity rate (2 decimals)
- `Min Manipulability` - Worst-case manipulability (3 decimals)
- `Mean Manipulability` - Average manipulability (3 decimals)
- `Mean Min Singular Value` - Average min SV (3 decimals)
- `Verdict` - Feasibility verdict with emoji ✅⚠️❗❌

**Formatting Applied:**
- Scores: 3 decimal places (0.234)
- Rates: 2 decimal places (0.05)
- Manipulability/Singular Values: 3 decimal places (0.015)

### 4. Markdown Reports

New `knife_pose_ranking.md` file with:
- Robot name and generation timestamp
- Total knife poses evaluated
- Formatted ranking table
- Legend explaining verdict categories

Example:
```markdown
# Knife Pose Ranking for IRB 1300-7/1.4

Generated: 2026-01-09 22:35:12

Total Knife Poses Evaluated: 395

## Ranking Table

| Rank | Knife Pose ID | Score | IK Failure Rate | ... | Verdict |
|------|---------------|-------|-----------------|-----|---------|
| 1 | nominal_x-467.8_y-915.8_z520.4_oriA | 0.234 | 0.00 | ... | ✅ Recommended |
| 2 | nominal_x-467.8_y-915.8_z620.4_oriB | 0.387 | 0.00 | ... | ⚠️ Borderline |
...

## Legend

- ✅ **Recommended**: IK feasible (0% failure) and score < 0.25
- ⚠️ **Borderline**: IK feasible (0% failure) and 0.25 ≤ score < 0.50
- ❗ **Poor**: IK feasible (0% failure) and 0.50 ≤ score < 0.75
- ❌ **Infeasible**: IK failure rate > 0% or score ≥ 0.75
```

### 5. Metadata JSON

New `metadata.json` file with:
- Total knife poses evaluated
- Total toolpaths used
- Number of fully reachable poses (0% IK failure)
- Verdict breakdown (counts per category)
- Best knife pose details
- Worst knife pose details

Example:
```json
{
  "robot_name": "IRB 1300-7/1.4",
  "generated": "2026-01-09T22:35:12.123456",
  "total_knife_poses_evaluated": 395,
  "total_toolpaths": 2,
  "fully_reachable_poses": 312,
  "verdict_breakdown": {
    "recommended": 145,
    "borderline": 98,
    "poor": 69,
    "infeasible": 83
  },
  "best_knife_pose": {
    "id": "nominal_x-467.8_y-915.8_z520.4_oriA",
    "score": 0.234,
    "ik_failure_rate": 0.0,
    "verdict": "✅ Recommended"
  },
  "worst_knife_pose": {
    "id": "out_of_reach_back_high",
    "score": 0.950,
    "ik_failure_rate": 0.45,
    "verdict": "❌ Infeasible"
  }
}
```

### 6. Per-Knife-Pose Details

Each knife pose gets its own subfolder with:

**`toolpath_details.csv`** - Per-toolpath breakdown:
```csv
Toolpath,Success,IK Failure Rate,Singularity Rate,Min Manipulability,Mean Manipulability,Error
20250804_mc_HyperFree_1,Yes,0.00,0.05,0.015,0.035,
20250804_mc_HyperFree_2,Yes,0.02,0.08,0.012,0.029,
```

**`details.json`** - Complete knife pose data:
- Aggregated metrics
- Rank and verdict
- Per-toolpath results
- Error messages

### 7. Enhanced Global Ranking

Updated `global_ranking.csv` with:
- Proper column names (title case with spaces)
- Verdict column
- Better ordering (Global Rank first)
- All metrics formatted consistently

## New Functions Added

1. `_compute_verdict(ik_failure_rate, final_score)` - Compute verdict based on rules
2. `save_per_robot_markdown(results, output_path, robot_name)` - Generate markdown report
3. `save_per_robot_metadata(results, output_path, robot_name)` - Generate metadata JSON
4. `save_knife_pose_details(knife_pose_result, output_dir)` - Save per-knife-pose details

## Modified Functions

1. `save_per_robot_csv()` - Updated with new columns and verdict
2. `save_global_ranking_csv()` - Enhanced formatting and verdict column
3. `_process_robot_results()` - Creates robot folders and saves all report types

## Testing

To verify the new structure:

```bash
# Run the combinatorial search
python combinatorial_search.py --config config/batch_feasibility_config.yaml --workers 8

# Check the output structure
tree output/feasibility_ranking/<timestamp>/per_robot/

# Verify CSV format
head output/feasibility_ranking/<timestamp>/per_robot/IRB_1300-7_1.4/knife_pose_ranking.csv

# Check metadata
cat output/feasibility_ranking/<timestamp>/per_robot/IRB_1300-7_1.4/metadata.json
```

## Example Usage

```python
import pandas as pd
import json

# Load ranking
df = pd.read_csv('output/.../per_robot/IRB_1300-7_1.4/knife_pose_ranking.csv')

# Get recommended poses
recommended = df[df['Verdict'] == '✅ Recommended']
print(f"Found {len(recommended)} recommended poses")

# Load metadata
with open('output/.../per_robot/IRB_1300-7_1.4/metadata.json') as f:
    meta = json.load(f)
    
print(f"Total poses evaluated: {meta['total_knife_poses_evaluated']}")
print(f"Fully reachable: {meta['fully_reachable_poses']}")
print(f"Best pose: {meta['best_knife_pose']['id']}")

# Load knife pose details
with open('output/.../knife_poses/nominal_x-467.8_y-915.8_z520.4_oriA/details.json') as f:
    details = json.load(f)
    
print(f"Verdict: {details['verdict']}")
print(f"Score: {details['normalized_score']}")
```

## Benefits

1. **Organized Structure**: Clear hierarchy with robot-specific folders
2. **Multiple Formats**: CSV, JSON, and Markdown for different use cases
3. **Easy Navigation**: Subfolders for each knife pose make it easy to find details
4. **Rich Metadata**: Summary statistics provide quick insights
5. **Visual Indicators**: Emoji verdicts make results immediately understandable
6. **Consistent Formatting**: All numbers formatted to appropriate precision
7. **Complete Data**: Both human-readable reports and machine-readable JSON

## Backward Compatibility

- Global reports still generated at root level
- Per-combination folders unchanged
- All existing functionality preserved
- Only output structure enhanced

## Files Modified

1. `combinatorial_search.py` - Enhanced report generation functions
2. `OUTPUT_STRUCTURE.md` - Complete documentation of output structure (NEW)
3. `REPORT_GENERATION_UPDATE.md` - This summary document (NEW)
