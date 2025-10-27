# Batch Processing System for Kinematic Feasibility Analysis

## Overview

This document describes the batch processing system for conducting comprehensive kinematic feasibility analysis of ABB IRB 1300 robots across multiple configurations.

The batch processing system enables parametric sweep experiments across:
- **Robot Models**: Multiple URDF configurations (e.g., IRB-1300 900, IRB-1300 1150)
- **Toolpaths**: Different trajectory CSV files from the Successful directory
- **Knife Poses**: Various knife configurations defined in `knife_poses.yaml`

## Architecture

### Modular Design

The batch processing system is built with high modularity for maintainability and extensibility:

```
batch_analyze_trajectories.py (Main orchestrator)
├── batch_processor.py (Discovers components)
├── results_handler.py (Results management)
├── csv_handling.py (CSV reading)
├── math_utils.py (Mathematical utilities)
├── handle_transforms.py (Coordinate transforms)
└── graph_utils.py (Visualization)
```

### Module Responsibilities

#### `batch_processor.py`
Discovers and catalogs available experiment components:
- `discover_robot_models()`: Finds all URDF files in Robot APCC directory
- `discover_toolpaths()`: Discovers CSV files in Toolpaths/Successful
- `load_knife_poses()`: Loads knife pose configurations from YAML
- `generate_output_dirname()`: Creates standardized output folder names
- `summarize_batch_experiment()`: Computes experiment statistics

#### `results_handler.py`
Manages structured storage of results:
- `ExperimentResult`: Data container for single experiment
- `ResultsManager`: YAML serialization and loading
- `create_experiment_result_from_analysis()`: Converts analysis output to result format

#### `batch_analyze_trajectories.py`
Main batch processing script with three nested loops:
1. **Outer Loop**: Robot models
2. **Middle Loop**: Toolpaths
3. **Inner Loop**: Knife poses

## Knife Pose Configuration

### File Location
```
Assests/Robot APCC/knife_poses.yaml
```

### YAML Format
```yaml
poses:
  pose_1:
    description: "Standard knife pose (from calibration data)"
    translation:
      x: -367.773  # mm
      y: -915.815  # mm
      z: 520.4     # mm
    rotation:
      w: 0.00515984   # quaternion w
      x: 0.712632     # quaternion x
      y: -0.701518    # quaternion y
      z: 0.000396522  # quaternion z
```

### Adding New Poses

To add a new knife pose configuration, append to the `poses` section:

```yaml
  pose_2:
    description: "Alternative knife configuration"
    translation:
      x: -370.0
      y: -910.0
      z: 525.0
    rotation:
      w: 0.005
      x: 0.713
      y: -0.701
      z: 0.001
```

**Important Notes:**
- All translations are in **millimeters** (automatically converted to meters)
- Quaternions should be **unit quaternions** (norm ≈ 1)
- Quaternion format: **[w, x, y, z]** (w is the scalar component)

## Running Batch Analysis

### Basic Usage

```bash
cd /path/to/Robotics-APCC
python scripts/batch_analyze_trajectories.py
```

### With Custom Output Directory

```bash
python scripts/batch_analyze_trajectories.py -o /custom/output/path
```

### Advanced Options

```bash
# Disable visualization plots (faster processing)
python scripts/batch_analyze_trajectories.py --no-visualize

# Set IK parameters
python scripts/batch_analyze_trajectories.py \
    --max-iterations 2000 \
    --tolerance 1e-5

# All options combined
python scripts/batch_analyze_trajectories.py \
    -o results \
    --max-iterations 2000 \
    --tolerance 1e-4 \
    --no-visualize
```

## Output Structure

Results are organized hierarchically by experiment parameters:

```
results/
├── batch_experiment_summary.yaml          # Overall batch summary
├── IRB-1300_900__20250820_mc_HyperFree_AF1__pose_1/
│   ├── experiment_results.yaml            # Detailed results (YAML)
│   ├── experiment.csv                     # Waypoint-by-waypoint data
│   └── plots/
│       ├── joint_angles_plot.png
│       ├── manipulability_plot.png
│       ├── reachability_plot.png
│       ├── singularity_measure_plot.png
│       ├── trajectory_3d_comparison.png
│       └── trajectory_analysis_results.csv
├── IRB-1300_900__20250820_mc_HyperFree_AF1__pose_2/
│   └── (same structure)
├── IRB-1300_900__20250804_mc_HyperFree_1__pose_1/
│   └── (same structure)
└── IRB-1300_1150__20250820_mc_HyperFree_AF1__pose_1/
    └── (same structure)
```

### Output Naming Convention

Output directories follow the pattern:
```
{robot_name}__{toolpath_name}__{pose_name}
```

Examples:
- `IRB-1300_900__20250820_mc_HyperFree_AF1__pose_1`
- `IRB-1300_1150__20250804_mc_HyperFree_1__pose_2`

All spaces are replaced with underscores for filesystem compatibility.

## Results Format

### Batch Experiment Summary (`batch_experiment_summary.yaml`)

```yaml
timestamp: "2025-10-27T14:23:45.123456"
total_experiments: 48
completed_experiments: 45
failed_experiments: 3
configuration:
  max_ik_iterations: 1000
  ik_tolerance: 1e-4
  visualize_plots: true
robot_models:
  - "IRB-1300 900"
  - "IRB-1300 1150"
toolpaths:
  - "20250820_mc_HyperFree_AF1"
  - "20250804_mc_HyperFree_1"
  # ... more toolpaths
knife_poses:
  - "pose_1"
results:
  - robot_name: "IRB-1300 900"
    toolpath_name: "20250820_mc_HyperFree_AF1"
    pose_name: "pose_1"
    status: "completed"
    summary:
      total_waypoints: 3219
      reachable_waypoints: 3150
      reachability_rate_percent: 97.8
      avg_manipulability: 0.0456
      avg_min_singular_value: 0.00234
```

### Experiment Results (`experiment_results.yaml`)

```yaml
experiment_metadata:
  robot_name: "IRB-1300 900"
  toolpath_name: "20250820_mc_HyperFree_AF1"
  pose_name: "pose_1"
  timestamp: "2025-10-27T14:25:12.456789"

summary:
  total_waypoints: 3219
  reachable_waypoints: 3150
  unreachable_waypoints: 69
  reachability_rate_percent: 97.8
  avg_manipulability: 0.0456
  min_manipulability: 0.0012
  max_manipulability: 0.0892
  avg_min_singular_value: 0.00234
  avg_condition_number: 425.3
  max_condition_number: 1850.2
  avg_position_error_m: 0.000142
  max_position_error_m: 0.001234

detailed_results:
  - trajectory_id: 1
    waypoint_index: 0
    x_m: 0.5234
    y_m: -0.9158
    z_m: 0.5204
    reachable: true
    manipulability: 0.0456
    min_singular_value: 0.00234
    condition_number: 425.3
    actual_ee_x_m: 0.5235
    actual_ee_y_m: -0.9160
    actual_ee_z_m: 0.5205
    position_error_m: 0.0001
    q1_rad: 0.1234
    q2_rad: -1.5678
    # ... more joint angles
  # ... more waypoints
```

### Experiment CSV (`experiment.csv`)

Flat CSV format with one row per waypoint:

```
trajectory_id,waypoint_index,x_m,y_m,z_m,reachable,manipulability,min_singular_value,...
1,0,0.5234,-0.9158,0.5204,True,0.0456,0.00234,...
1,1,0.5240,-0.9165,0.5210,True,0.0458,0.00236,...
1,2,0.5245,-0.9172,0.5216,False,,,
```

## Algorithm Flow

### Batch Processing Loop Structure

```python
for robot in robots:
    load_robot_model(robot.urdf)
    
    for toolpath in toolpaths:
        load_trajectory(toolpath.csv)
        
        for pose in knife_poses:
            apply_knife_transformation(pose)
            run_kinematic_analysis()
            save_results()
            generate_visualizations()
```

### Kinematic Analysis Steps

For each waypoint in each trajectory:

1. **Inverse Kinematics**
   - Warm-start with previous solution
   - Fallback to neutral configuration
   - Final fallback to random configurations
   - Damped least-squares solver with adaptive damping

2. **Singularity Analysis** (for reachable waypoints)
   - Compute Jacobian at solution
   - Calculate Yoshikawa manipulability
   - Compute minimum singular value
   - Calculate condition number

3. **Error Metrics**
   - Position error (target vs achieved)
   - Joint angle continuity

4. **Results Storage**
   - Per-waypoint results
   - Summary statistics
   - YAML and CSV output
   - Visualization plots

## Data Units and Conventions

### Internal Representation
- **Positions**: meters (m)
- **Angles**: radians (rad)
- **Quaternions**: unit quaternions [w, x, y, z]

### Input Data
- **CSV trajectories**: millimeters (mm) → converted to meters automatically
- **Knife pose translations**: millimeters (mm) → converted to meters by loader

### Output Data
- **YAML results**: meters, radians
- **CSV results**: meters, radians
- **Plot labels**: appropriate unit suffixes (m, rad, %, etc.)

## Error Handling and Robustness

### Multi-Level IK Fallback
1. **Primary**: Warm-start with previous joint configuration
2. **Secondary**: Neutral configuration
3. **Tertiary**: Random configurations (up to 3 attempts)

### Robust Kinematics Solver
- Damped Least-Squares (DLS) with adaptive damping
- Singular value decomposition (SVD) monitoring
- Backtracking line search for convergence assurance
- Joint limit enforcement after each step

### Batch Error Recovery
- Individual experiment failures don't stop batch processing
- Failed experiments logged in `batch_experiment_summary.yaml`
- Partial results preserved

## Performance Considerations

### Computational Complexity
- **Workspace**: O(r × t × p) where r=robots, t=toolpaths, p=poses
- **Per-waypoint**: O(1000) IK iterations × waypoints
- **Memory**: One robot model loaded per outer loop iteration

### Optimization Tips
1. **Disable visualization** for faster processing (use `--no-visualize`)
2. **Reduce IK iterations** for quick prototyping
3. **Increase IK tolerance** for faster convergence (trade-off: accuracy)
4. **Process robot models sequentially** (good for parallelization in future)

### Typical Runtime Estimates
- Single experiment (1 robot × 1 toolpath × 1 pose): 5-15 minutes
- Full batch (2 robots × 16 toolpaths × 2 poses): 4-8 hours
- Depends on: toolpath length, hardware, IK parameters

## Extending the System

### Adding New Robot Models

1. Create directory: `Assests/Robot APCC/IRB-1300-XXX URDF/`
2. Create `urdf/` subdirectory
3. Place URDF file (preferably named `*_ee.urdf`)
4. Batch system will auto-discover it

### Adding New Toolpaths

1. Place CSV file in: `Assests/Robot APCC/Toolpaths/Successful/`
2. File must contain valid trajectory data (x, y, z, qw, qx, qy, qz)
3. Batch system will auto-discover it

### Adding New Knife Poses

1. Edit: `Assests/Robot APCC/knife_poses.yaml`
2. Add new `pose_X` entry with translation and rotation
3. Batch system will automatically include it

### Custom Analysis Workflows

Modify `batch_analyze_trajectories.py`:
- Hook into results after analysis
- Custom aggregation logic
- Alternative output formats
- Post-processing analysis

## Troubleshooting

### Common Issues

**No experiments run**
```
Error discovering experiment components: No URDF files found
```
- Check `Robot APCC/` directory structure
- Ensure each robot folder has `urdf/` subdirectory
- Verify URDF files exist and are named correctly

**Knife poses not loading**
```
FileNotFoundError: Knife poses YAML file not found
```
- Verify `Assests/Robot APCC/knife_poses.yaml` exists
- Check for typos in YAML file path
- Validate YAML syntax

**All trajectories unreachable**
- Verify knife pose translation/rotation values
- Check robot URDF is correct for the toolpath
- Increase IK iterations and relax tolerance
- Visualize the toolpath vs robot workspace

**Out of memory**
- Process robot models in separate batches
- Reduce number of toolpaths per batch
- Increase swap memory on system

## References

### Key Papers
- Yoshikawa, T. "Manipulability of robotic mechanisms." International Journal of Robotics Research, 1985.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. "Robotics: Modelling, Planning and Control." Springer, 2009.

### Libraries
- **Pinocchio**: https://github.com/stack-of-tasks/pinocchio
- **PyYAML**: https://pyyaml.org/
- **NumPy**: https://numpy.org/
- **Pandas**: https://pandas.pydata.org/

## Support and Contribution

For issues, improvements, or extensions:
1. Document the problem clearly
2. Provide example configuration files
3. Include error messages and logs
4. Maintain backward compatibility

---

**Last Updated**: October 2025
**Version**: 1.0
