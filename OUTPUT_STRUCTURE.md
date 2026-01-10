# Combinatorial Search Output Structure

## Overview

The combinatorial search generates a comprehensive set of reports organized by robot and knife pose. This document describes the complete output structure.

## Directory Structure

```
output/feasibility_ranking/<HH_MM_SS>/
├── per_robot/
│   └── <robot_name>/                      # e.g., IRB_1300-7_1.4/
│       ├── knife_pose_ranking.csv         # Main ranking CSV
│       ├── knife_pose_ranking.md          # Main ranking Markdown
│       ├── metadata.json                  # Summary statistics
│       ├── detailed_results.json          # Full data dump
│       ├── ranking_plot.png               # Visual ranking plot
│       └── knife_poses/                   # Per-knife-pose details
│           └── <knife_pose_id>/           # e.g., nominal_x-467.8_y-915.8_z420.4_oriA/
│               ├── toolpath_details.csv   # Per-toolpath breakdown
│               └── details.json           # Full knife pose details
├── <robot>__<knife>__<toolpath>/          # Per-combination folders
│   └── summary.json                       # Combination summary
├── global_ranking.csv                     # Cross-robot ranking
├── batch_ranking_summary.json             # Overall batch summary
└── feasibility_ranking_report.md          # Human-readable report
```

## File Descriptions

### Per-Robot Files

#### 1. `knife_pose_ranking.csv`

Main ranking table with the following columns:

| Column | Format | Description |
|--------|--------|-------------|
| Rank | Integer | Rank within this robot (1=best) |
| Knife Pose ID | String | Unique identifier for knife pose |
| Score | 3 decimals | Normalized score (0=best, 1=worst) |
| IK Failure Rate | 2 decimals | Max IK failure rate across toolpaths (0-1) |
| Singularity Rate | 2 decimals | Max singularity rate across toolpaths (0-1) |
| Min Manipulability | 3 decimals | Worst-case manipulability across toolpaths |
| Mean Manipulability | 3 decimals | Average manipulability across toolpaths |
| Mean Min Singular Value | 3 decimals | Average minimum singular value |
| Verdict | String | Feasibility verdict with emoji |

**Verdict Rules (implemented exactly):**
- If `IK_failure_rate > 0` → ❌ Infeasible
- Else if `final_score < 0.25` → ✅ Recommended
- Else if `final_score < 0.50` → ⚠️ Borderline
- Else if `final_score < 0.75` → ❗ Poor
- Else → ❌ Infeasible

**Example:**
```csv
Rank,Knife Pose ID,Score,IK Failure Rate,Singularity Rate,Min Manipulability,Mean Manipulability,Mean Min Singular Value,Verdict
1,nominal_x-467.8_y-915.8_z520.4_oriA,0.234,0.00,0.05,0.015,0.035,0.025,✅ Recommended
2,nominal_x-467.8_y-915.8_z620.4_oriB,0.387,0.00,0.12,0.012,0.028,0.021,⚠️ Borderline
3,nominal_x-467.8_y-815.8_z420.4_oriA,0.651,0.00,0.23,0.008,0.019,0.015,❗ Poor
4,out_of_reach_back_high,0.950,0.45,0.38,0.003,0.011,0.009,❌ Infeasible
```

#### 2. `knife_pose_ranking.md`

Markdown version of the ranking table with:
- Header with robot name and generation timestamp
- Total knife poses evaluated
- Full ranking table
- Legend explaining verdict categories

#### 3. `metadata.json`

Summary statistics in JSON format:

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

**Fields:**
- `total_knife_poses_evaluated`: Total number of knife poses tested
- `total_toolpaths`: Number of toolpaths used for evaluation
- `fully_reachable_poses`: Number of poses with 0% IK failure rate across all toolpaths
- `verdict_breakdown`: Count of poses in each verdict category
- `best_knife_pose`: Details of the top-ranked knife pose
- `worst_knife_pose`: Details of the lowest-ranked knife pose

#### 4. `detailed_results.json`

Complete data dump with all raw and normalized metrics for each knife pose. Used for programmatic access and reanalysis.

#### 5. `ranking_plot.png`

Visual bar chart showing:
- Top 10 best knife poses (green gradient)
- Top 10 worst knife poses (red gradient)

### Per-Knife-Pose Files

Each knife pose has a subfolder in `knife_poses/` containing:

#### `toolpath_details.csv`

Per-toolpath breakdown for this knife pose:

```csv
Toolpath,Success,IK Failure Rate,Singularity Rate,Min Manipulability,Mean Manipulability,Error
20250804_mc_HyperFree_1,Yes,0.00,0.05,0.015,0.035,
20250804_mc_HyperFree_2,Yes,0.02,0.08,0.012,0.029,
toolpath_unreachable,No,N/A,N/A,N/A,N/A,IK solver failed
```

#### `details.json`

Complete details for this knife pose including:
- Aggregated metrics across all toolpaths
- Rank and verdict
- Per-toolpath results with full metrics
- Error messages for failed toolpaths

### Global Files

#### `global_ranking.csv`

Cross-robot ranking with all robots and knife poses:

| Column | Description |
|--------|-------------|
| Global Rank | Rank across all robots |
| Robot Rank | Rank within the robot |
| Robot Name | Name of the robot |
| Knife Pose ID | Unique knife pose identifier |
| Score | Normalized score (3 decimals) |
| ... | Same metrics as per-robot CSV |
| Verdict | Feasibility verdict |
| N Toolpaths | Total toolpaths evaluated |
| N Successful | Number of successful toolpaths |

#### `batch_ranking_summary.json`

Overall batch processing summary:
- Total combinations processed
- Success/failure counts
- Top knife pose per robot
- Timestamp and metadata

#### `feasibility_ranking_report.md`

Human-readable markdown report with:
- Executive summary
- Scoring weights used
- Top 5 and bottom 5 knife poses per robot
- Failure analysis
- List of failed combinations

## Usage Examples

### Finding the Best Knife Pose for a Robot

```python
import pandas as pd

# Load the ranking
df = pd.read_csv('output/feasibility_ranking/22_35_12/per_robot/IRB_1300-7_1.4/knife_pose_ranking.csv')

# Get recommended poses
recommended = df[df['Verdict'] == '✅ Recommended']

# Best overall
best = df.iloc[0]
print(f"Best knife pose: {best['Knife Pose ID']}")
print(f"Score: {best['Score']}")
print(f"IK Failure Rate: {best['IK Failure Rate']}")
```

### Analyzing a Specific Knife Pose

```python
import json

# Load knife pose details
with open('output/.../knife_poses/nominal_x-467.8_y-915.8_z520.4_oriA/details.json') as f:
    details = json.load(f)

# Check per-toolpath performance
for tp in details['per_toolpath_results']:
    print(f"{tp['toolpath_name']}: {tp['metrics']}")
```

### Comparing Robots

```python
import pandas as pd

# Load global ranking
df = pd.read_csv('output/.../global_ranking.csv')

# Find best pose per robot
best_per_robot = df.groupby('Robot Name').first()
print(best_per_robot[['Knife Pose ID', 'Score', 'Verdict']])
```

## Formatting Standards

- **Scores**: 3 decimal places (e.g., 0.234)
- **Rates**: 2 decimal places (e.g., 0.05)
- **Manipulability**: 3 decimal places (e.g., 0.015)
- **Singular Values**: 3 decimal places (e.g., 0.025)
- **Verdicts**: Emoji + text (e.g., "✅ Recommended")

## Notes

1. **Timestamp Folders**: Each run creates a new timestamped folder to prevent overwriting previous results
2. **CSV Format**: All CSV files use standard format with headers for easy import into Excel/pandas
3. **JSON Format**: All JSON files are pretty-printed with 2-space indentation for readability
4. **Emoji Support**: Verdicts use emoji for visual clarity; ensure your viewer supports UTF-8
5. **Per-Combination Folders**: Individual combination folders contain the raw `summary.json` from `process_toolpath`
