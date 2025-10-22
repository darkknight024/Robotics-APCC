# Graphing Utility - Joint & Speed Analysis

## Overview

The `graphing_utility.py` module (previously `speed_analyzer.py`) now provides comprehensive data collection and visualization for robot trajectory playback:

1. **End-effector speed analysis** (existing functionality)
2. **Joint position tracking** (NEW)

---

## Features

### 1. Speed Analysis
- Collects end-effector speed data at each simulation step
- Calculates average speeds between waypoints
- Compares calculated speeds with target speeds (from CSV 8th column)
- Generates 3 plots:
  - Waypoint average speeds (line graph with target comparison)
  - Step-by-step instantaneous speeds
  - Combined view

### 2. Joint Position Analysis (NEW)
- Collects joint positions (radians) for all 6 joints at each step
- Generates 6 individual plots (one per joint):
  - X-axis: Step number
  - Y-axis: Joint position in radians
  - Y-axis limits: Min/max from URDF joint limits (with 10% padding)
  - Red dashed lines show joint limits
  - Statistics box shows mean, max, min, and range
- **NEW: Combined plot** showing all 6 joints together:
  - X-axis: Waypoint index
  - Y-axis: Joint position in radians
  - 6 different colored lines (one per joint)
  - Shaded regions showing valid joint limit ranges
  - Legend with joint limits
  - Statistics for all joints

---

## Joint Limits (from URDF)

| Joint    | Min (rad) | Max (rad) | Min (deg) | Max (deg) |
|----------|-----------|-----------|-----------|-----------|
| Joint_1  | -3.142    | 3.142     | -180°     | 180°      |
| Joint_2  | -1.745    | 2.269     | -100°     | 130°      |
| Joint_3  | -3.665    | 1.134     | -210°     | 65°       |
| Joint_4  | -4.014    | 4.014     | -230°     | 230°      |
| Joint_5  | -2.269    | 2.269     | -130°     | 130°      |
| Joint_6  | -6.981    | 6.981     | -400°     | 400°      |

---

## Usage

### During Playback
The script automatically collects both speed and joint data during playback mode:

```bash
python irb_1300_example.py --mode p --csv your_trajectory.csv
```

### Output Files

After playback, the following files are generated:

**Speed Data (in `./speed_data/`):**
- `speed_<trajectory>_<timestamp>_steps.csv` - Per-step speed data
- `speed_<trajectory>_<timestamp>_waypoints.csv` - Per-waypoint speed data
- `waypoint_speeds.png` - Waypoint average speeds plot
- `step_speeds.png` - Instantaneous speeds plot
- `combined_speed_analysis.png` - Combined view

**Joint Data (in `./joint_data/`):**
- `joint_<trajectory>_<timestamp>_joints.csv` - Per-step joint positions
- `Joint_1_position.png` - Joint 1 position plot
- `Joint_2_position.png` - Joint 2 position plot
- `Joint_3_position.png` - Joint 3 position plot
- `Joint_4_position.png` - Joint 4 position plot
- `Joint_5_position.png` - Joint 5 position plot
- `Joint_6_position.png` - Joint 6 position plot
- `all_joints_combined.png` - **NEW: All 6 joints in one plot vs waypoints**

### Standalone Analysis

You can also analyze existing data files:

```bash
python graphing_utility.py ./speed_data/speed_trajectory_20250113_120000.csv
```

This will:
1. Load existing speed data (if `_steps.csv` and `_waypoints.csv` exist)
2. Load existing joint data (if `_joints.csv` exists)
3. Print statistics for both
4. Regenerate all plots

---

## Data Collectors

### SpeedDataCollector
Collects end-effector position and calculates speeds.

**Methods:**
- `record_step(ee_position, current_waypoint)` - Record EE position at each step
- `record_waypoint_transition(ee_position, waypoint_index, time)` - Mark waypoint changes
- `save_data(filepath)` - Save to CSV files

### JointDataCollector (NEW)
Collects joint positions for all 6 joints.

**Methods:**
- `record_step(joint_pos_tensor)` - Record joint positions at each step
- `save_data(filepath)` - Save to CSV file

---

## Integration in irb_1300_example.py

The playback mode now:

1. **Initializes both collectors:**
   ```python
   speed_collector = SpeedDataCollector(dt=sim_dt, target_speeds_mm_s=target_speeds_mm_s)
   joint_collector = JointDataCollector()
   ```

2. **Records data at each step:**
   ```python
   # Joint data collected at EVERY step (including settling phase)
   joint_collector.record_step(robot.data.joint_pos[0])
   
   # Speed data only after settling (tied to waypoint progression)
   if is_settled:
       speed_collector.record_step(ee_pose_w[0, 0:3], current_waypoint)
   ```

3. **Saves and analyzes after playback:**
   ```python
   speed_collector.save_data(speed_data_file)
   joint_collector.save_data(joint_data_file)
   analyze_trajectory_speed(speed_data_file)
   analyze_joint_data(joint_data_file)
   plot_speed_data(speed_data_file, output_dir=speed_data_dir)
   plot_joint_data(joint_data_file, output_dir=joint_data_dir)
   ```

---

## Example Output

### Terminal Output
```
[INFO] Playback finished. Processing data...
[SpeedDataCollector] Saved step data to: ./speed_data/speed_trajectory_20250113_120000_steps.csv
[SpeedDataCollector] Saved waypoint data to: ./speed_data/speed_trajectory_20250113_120000_waypoints.csv
[JointDataCollector] Saved joint data to: ./joint_data/joint_trajectory_20250113_120000_joints.csv

============================================================
TRAJECTORY SPEED ANALYSIS
============================================================
...

============================================================
JOINT POSITION ANALYSIS
============================================================
Joint_1:
  Mean: 0.523 rad (29.9°)
  Max: 1.234 rad (70.7°)
  Min: -0.456 rad (-26.1°)
  Range: 1.690 rad
  Limit: [-3.142, 3.142] rad
...

[GraphingUtility] Saved Joint_1 plot: ./joint_data/Joint_1_position.png
[GraphingUtility] Saved Joint_2 plot: ./joint_data/Joint_2_position.png
...

============================================================
DATA ANALYSIS COMPLETE
============================================================
```

---

## Benefits

1. **Speed Analysis**: Verify robot is moving at desired speeds
2. **Joint Tracking**: Monitor joint movements and ensure they stay within limits
3. **Debugging**: Identify problematic joints or waypoints
4. **Optimization**: Analyze joint usage and trajectory efficiency
5. **Safety**: Detect when joints approach their limits

---

## Notes

- Joint data is collected during playback mode at **every single step** (including settling phase)
- Speed data is collected only after settling (since it's tied to waypoint progression)
- Joint data is NOT collected during recording mode
- All angles are in radians (both in CSV and plots)
- Plots include degree conversions in statistics for convenience
- CSV files can be opened in Excel/Python for further analysis

