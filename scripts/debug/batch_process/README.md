# Batch Processing Module

Provides per-trajectory wrapper functions for trajectory analysis in batch mode.

## Overview

The batch processing module contains helper functions that wrap the main analysis tools (`pose_analyzer`, `continuity_analyzer`) to work with individual trajectories extracted from CSV files.

This module is used by `pose_3d_batch.py` to process folders of CSV files, where each CSV can contain multiple trajectories separated by single-column delimiter rows.

## Module Components

### `pose_analyzer_batch.py`

Wraps pose analysis functions for single trajectories:

- `analyze_single_trajectory_delta()` - Analyzes delta (position/rotation changes) between consecutive poses
  - Shows position delta (ΔX, ΔY, ΔZ) in original and transformed trajectories
  - Shows rotation delta (angular changes) in original and transformed trajectories

## Usage Pattern

These functions are called by `pose_3d_batch.py` automatically. They are not meant to be used standalone.

Each function:
1. Takes a single trajectory (numpy array of shape n_poses×7)
2. Wraps it in a list for compatibility with main analysis functions
3. Generates analysis graphs
4. Saves output to specified directory with trajectory-specific naming

## File Naming Convention

All generated files follow this naming pattern:
```
{csv_stem}_Traj_{trajectory_index}_[1-{num_poses}]_{analysis_type}.png
```

Examples:
- `file1_Traj_1_[1-30]_comparison.png`
- `file1_Traj_2_[1-50]_delta_analysis.png`
- `file1_Traj_2_[1-50]_absolute_value_line_graph.png`

## Output Structure

When processing a folder of CSV files, output is organized as:

```
input_folder/
  ├── file1.csv
  ├── file1/
  │   ├── Traj_1_[1-30]/
  │   │   ├── file1_Traj_1_[1-30]_transformed.csv
  │   │   ├── file1_Traj_1_[1-30]_comparison.png
  │   │   ├── file1_Traj_1_[1-30]_delta_analysis.png
  │   │   └── file1_Traj_1_[1-30]_absolute_value_line_graph.png
  │   └── Traj_2_[1-50]/
  │       └── ...
  └── file2/
      └── ...
```

## Configuration

Uses the same `compare.yaml` configuration as `pose_3d_batch.py`. Key settings:

- `transformation.enabled` - Apply knife pose transformation
- `comparison.enabled` - Generate data comparison graphs
- `pose_analyzer.enabled` + `save_delta_graphs` - Generate delta analysis

## Error Handling

If any analysis fails for a single trajectory:
- Error is logged but processing continues
- Other trajectories in the batch are unaffected
- Missing output files indicate where failures occurred



