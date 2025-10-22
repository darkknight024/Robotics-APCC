# KUKA IRB-1300 Trajectory Validator

A modular, clean trajectory feasibility checker for the KUKA IRB-1300 robot in Isaac Lab. This script validates end-effector trajectories (CSV format) against various kinematic constraints.

## Overview

This validator checks trajectory waypoints for:
- **Workspace Reachability**: Whether the end-effector position is within the robot's workspace
- **IK Feasibility**: Whether inverse kinematics can find a valid joint configuration (kinematic feasibility)
- **Joint Limits**: Whether computed joint angles are within allowed ranges
- **Manipulability**: Measure of robot dexterity at each pose (manipulability estimation)
- **Singularity Detection**: Identifies kinematic singularities using condition number and joint analysis (singularity checks)

## Key Features

- **Modular Design**: Separate classes for each validation check
- **Selective Testing**: Run individual tests or all tests together
- **Detailed Reports**: Console output and file-based reports with statistics
- **Color-Coded Visualization**: Optional trajectory playback with feasibility indicators
- **KUKA IRB-1300 Support**: Configured for KUKA robot with `ee_pose` end-effector

## Architecture

### Core Classes

1. **`TrajectoryLoader`**: Loads and validates CSV trajectory files
2. **`WorkspaceChecker`**: Validates workspace reachability
3. **`IKChecker`**: Computes inverse kinematics and checks feasibility
4. **`JointLimitChecker`**: Validates joint angles against limits
5. **`ManipulabilityAnalyzer`**: Computes and analyzes manipulability indices
6. **`FeasibilityReport`**: Aggregates results and generates reports
7. **`TrajectoryVisualizer`**: Visualizes trajectories with color-coded markers

### Robot Configuration

- **Robot**: KUKA IRB-1300/0.9 (900mm reach)
- **End-Effector**: `ee_pose` (fixed joint offset from `Link_6`)
- **Joints**: `Joint_1` through `Joint_6`
- **Home Position**: All joints at 0° except `Joint_5` (90° down)

## Installation

Ensure Isaac Lab is installed and configured. This script requires:
- Isaac Lab/Isaac Sim
- PyTorch
- Pandas
- NumPy

## Usage

### Basic Usage (All Tests)

```bash
python Assests/kuka_trajectory_validator.py --csv path/to/trajectory.csv
```

### Selective Testing

Run specific tests using the `--tests` flag:

```bash
# Run only workspace and IK checks
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --tests workspace ik

# Run only joint limits check
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --tests joint_limits

# Run workspace, manipulability, and singularity checks
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks workspace manipulability singularity
```

### Available Test Options

- `workspace`: Check if waypoints are within robot's reachable workspace
- `ik`: Attempt inverse kinematics for each waypoint (kinematic feasibility)
- `joint_limits`: Verify joint angles stay within limits
- `manipulability`: Compute manipulability index (manipulability estimation)
- `singularity`: Detect kinematic singularities via Jacobian analysis (singularity checks)
- `all`: Run all tests (default)

### Additional Options

```bash
# Specify custom report filename
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --report_file my_report.txt

# Change number of environments (usually keep at 1)
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --num_envs 1

# Headless mode (no visualization prompt)
python Assests/kuka_trajectory_validator.py --csv trajectory.csv --headless
```

### Full Example

```bash
python Assests/kuka_trajectory_validator.py \
  --csv trajectories/shoe_assembly.csv \
  --tests workspace ik joint_limits manipulability \
  --report_file results/shoe_assembly_report.txt \
  --num_envs 1
```

## CSV Format

The trajectory CSV should contain end-effector poses (position + quaternion):

```csv
pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z
0.5,0.2,0.3,1.0,0.0,0.0,0.0
0.5,0.2,0.35,0.707,0.0,0.707,0.0
...
```

- **Position**: (x, y, z) in meters, relative to robot base
- **Orientation**: Quaternion (w, x, y, z)
- **Total columns**: 7

## Output

All outputs are automatically saved to `results/validation_test/` directory with timestamps.

### Console Output

Real-time progress during validation:
```
[INFO] Loading trajectory from: trajectory.csv
[INFO] Loaded 500 waypoints
[INFO] Initialized 4 checkers:
  - WorkspaceChecker
  - IKChecker
  - JointLimitChecker
  - ManipulabilityChecker
[INFO] Moving to home position...
[INFO] Validating trajectory with 4 checks...
  Validated 500/500 waypoints...
[INFO] Validation complete!
```

### Summary Report

```
================================================================================
TRAJECTORY FEASIBILITY REPORT
================================================================================

Trajectory: trajectory.csv
Generated: 2025-10-12 14:30:45

Total Waypoints:         500
Feasible Waypoints:      487 (97.4%)
Infeasible Waypoints:    13

--------------------------------------------------------------------------------
INFEASIBILITY BREAKDOWN
--------------------------------------------------------------------------------
Out-of-Workspace:        5
IK Failures:             2
Joint Limit Violations:  6
Singularity Detections:  12

--------------------------------------------------------------------------------
MANIPULABILITY ANALYSIS
--------------------------------------------------------------------------------
Min:                     0.000123
Max:                     0.045678
Mean:                    0.012345

What is Manipulability?
  - Measure of how well the robot can move in all directions
  - Higher values = better dexterity
  - Values near 0 = singularity (restricted motion)

--------------------------------------------------------------------------------
SINGULARITY ANALYSIS
--------------------------------------------------------------------------------
Singularities Detected:  12
Max Condition Number:    245.67
Mean Condition Number:   35.42

What are Singularities?
  - Configurations where robot loses degrees of freedom
  - Detected via Jacobian condition number
  - Types: Wrist, Shoulder, Elbow singularities
  - Should be avoided for smooth motion
```

### Report File

Detailed report saved to `results/validation_test/<trajectory_name>_feasibility_report_<timestamp>.txt` containing:
- Summary statistics
- Breakdown by failure type
- Detailed waypoint-by-waypoint results
- Manipulability analysis with explanation

### Graphs (NEW!)

Automatically generated graphs for each test:

#### Individual Test Graphs

1. **Workspace Analysis** (`workspace_analysis_<timestamp>.png`)
   - X-axis: Waypoint index
   - Y-axis: Distance from robot base (meters)
   - Color: Green (valid), Red (out of workspace)
   - Shows max/min workspace limits

2. **IK Feasibility** (`ik_analysis_<timestamp>.png`)
   - X-axis: Waypoint index
   - Y-axis: Success (1) or Failure (0)
   - Color: Green (success), Red (failure)

3. **Joint Limit Violations** (`joint_limits_analysis_<timestamp>.png`)
   - X-axis: Waypoint index
   - Y-axis: Maximum violation magnitude (radians)
   - Color: Green (valid), Red (violated)

4. **Manipulability Index** (`manipulability_analysis_<timestamp>.png`)
   - X-axis: Waypoint index
   - Y-axis: Manipulability (log scale)
   - Shows manipulability throughout trajectory

5. **Singularity Detection** (`singularity_analysis_<timestamp>.png`) **NEW!**
   - X-axis: Waypoint index
   - Y-axis: Jacobian condition number (log scale)
   - Color: Green (no singularity), Red (singularity detected)
   - Shows warning (orange) and critical (red) threshold lines

#### Combined Overview Graph

`combined_validation_<timestamp>.png` - Multi-panel view with all enabled tests in one figure.

All graphs are saved to: `results/validation_test/`

### Visualization

After validation, you'll be prompted:
```
Visualize trajectory? (y/n):
```

Visualization features:
- **Green spheres**: Feasible waypoints
- **Red spheres**: IK/joint limit failures
- **Black spheres**: Out-of-workspace waypoints
- **Robot animation**: Plays through trajectory
- **Frame markers**: Current EE pose and goal pose

## Configuration Updates

### Updated Robot Configuration

The script is compatible with the latest `irb_1300_cfg.py` which uses separate actuator groups:
- `main_axes`: Joints 1-3 (base, shoulder, elbow) with 150 Nm torque
- `wrist_axes`: Joints 4-6 (wrist joints) with 10-20 Nm torque

This more realistic configuration won't affect validation checks, which focus on:
- Kinematic limits (position-based)
- Workspace reachability  
- IK solvability
- Manipulability

## Robot Specifications

### KUKA IRB-1300/0.9

- **Reach**: 900mm (0.9m)
- **Payload**: 7kg
- **Repeatability**: ±0.03mm
- **Degrees of Freedom**: 6

### Joint Limits

| Joint | Min (rad) | Max (rad) | Min (deg) | Max (deg) |
|-------|-----------|-----------|-----------|-----------|
| Joint_1 | -3.142 | 3.142 | -180° | 180° |
| Joint_2 | -1.745 | 2.269 | -100° | 130° |
| Joint_3 | -3.665 | 1.134 | -210° | 65° |
| Joint_4 | -4.014 | 4.014 | -230° | 230° |
| Joint_5 | -2.269 | 2.269 | -130° | 130° |
| Joint_6 | -6.981 | 6.981 | -400° | 400° |

### Default Home Position

```python
Joint_1: 0.0°    (base rotation)
Joint_2: 0.0°    (shoulder)
Joint_3: 0.0°    (elbow)
Joint_4: 0.0°    (wrist roll)
Joint_5: 90.0°   (wrist pitch - pointing down)
Joint_6: 0.0°    (wrist yaw)
```

## Workspace Limits

- **Maximum reach from base**: 0.9m (900mm nominal reach)
- **Minimum reach**: 0.15m (avoid base singularity)

Configurable in lines ~159-160 of the script.

## Manipulability Index

Yoshikawa's manipulability measure: `m = sqrt(det(J * J^T))`

### Interpretation for KUKA IRB-1300

- **m > 0.01**: Good maneuverability
- **m = 0.001 - 0.01**: Acceptable range
- **m < 0.0001**: Near singularity, restricted motion
- **m ≈ 0**: Singularity, loss of degree of freedom

## Troubleshooting

### Common Issues

1. **"ee_pose not found"**
   - Ensure USD file has `ee_pose` link (not `ee_link`)
   - Check URDF-to-USD conversion preserved end-effector

2. **All waypoints fail workspace check**
   - Check trajectory is in correct coordinate frame (robot base frame)
   - Adjust `MAX_WORKSPACE_REACH` in script if needed

3. **High IK failure rate**
   - Trajectory may contain unreachable poses
   - Check for singularities or extreme orientations
   - Increase IK solver iterations if needed

4. **Joint limit violations**
   - Trajectory pushes robot beyond physical limits
   - Review joint constraints in your path planner

## Code Structure

```
kuka_trajectory_validator.py
├── Constants & Configuration
│   ├── Joint limits
│   ├── Home position
│   └── Visualization colors
├── Checker Classes
│   ├── TrajectoryLoader
│   ├── WorkspaceChecker
│   ├── IKChecker
│   ├── JointLimitChecker
│   └── ManipulabilityAnalyzer
├── Reporting
│   └── FeasibilityReport
├── Visualization
│   └── TrajectoryVisualizer
├── Scene Configuration
│   └── KUKASceneCfg
└── Main Orchestration
    └── main()
```

## Customization

### Adjust Workspace Limits

Edit lines ~159-160:
```python
WORKSPACE_MAX_REACH = 0.9  # Maximum reach in meters
WORKSPACE_MIN_REACH = 0.15  # Minimum reach in meters
```

### Modify Home Position

Edit lines ~153-156:
```python
DEFAULT_HOME_POSE = torch.tensor(
    [x, y, z, qw, qx, qy, qz],  # Position + quaternion
    dtype=torch.float32
)
```

### Change Joint Limits

Edit the `JOINT_LIMITS` dictionary (lines ~140-147).

### Adjust IK Solver

Modify `DifferentialIKControllerCfg` parameters in `TrajectoryValidator.__init__()` (lines ~816-820):
- `ik_method`: "dls" (damped least squares) or "svd"
- `use_relative_mode`: True/False
- Additional solver parameters

### Change Output Directory

Modify `get_output_dir()` function (lines ~168-173) to change where results are saved.

## Performance

- **Validation Speed**: ~100-200 waypoints/second
- **Memory**: Minimal (single environment)
- **GPU**: Optional but recommended for faster IK computation

## Notes

- This script targets the `ee_pose` end-effector, which is at a fixed offset from `Link_6`
- All checks are kinematic only (no dynamics or collision with environment)
- Collision detection is not implemented (future feature)
- Uses Isaac Lab's `DifferentialIKController` for IK computation

## License

Copyright (c) 2022-2025, The Isaac Lab Project Developers.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

## Support

For issues or questions:
1. Check Isaac Lab documentation
2. Review KUKA IRB-1300 specifications
3. Verify trajectory CSV format
4. Check console output for specific error messages

