# IK Unreachability Debug Features

## Overview

Enhanced the feasibility analysis system to provide comprehensive debugging information when IK reachability fails. This helps identify why certain waypoints are reported as unreachable, particularly important when analyzing trajectories that are known to be successful in practice.

## What's New

### 1. Automatic Debug Folder Creation

For any trajectory that contains at least 1 unreachable waypoint, the system automatically creates:
```
output/<toolpath_name>/trajectory_N/unreachability_debug/
```

This folder contains:
- `ik_failure_analysis.png` - Comprehensive 6-panel failure analysis
- `joint_limit_analysis.png` - Joint limit proximity analysis

### 2. Enhanced Analysis Report

The `analysis_report.txt` now includes detailed failure information for each unreachable waypoint:

```
DETAILED FAILURE ANALYSIS:
  Failed waypoint indices: [5, 12, 23]
  
  Waypoint 5:
    Position: [0.5234, 0.3421, 0.8912] m
    Quaternion: [0.7071, 0.0000, 0.7071, 0.0000]
    Distance from origin: 1.1234 m
    IK Solver Status:
      - Iterations attempted: 50
      - Final residual norm: 0.003456
      - Failure reason: max_iter_exceeded
      - Min singular value: 0.002341
      - Max singular value: 45.234
      - Joint limit violations detected:
        J3: Upper limit exceeded by 5.23 deg
      - Distance from previous config: 0.3456 rad
```

### 3. Debug Visualization Plots

#### IK Failure Analysis Plot (6 panels)

**Panel 1: Reachability Status**
- Bar chart showing reachable/unreachable status for each waypoint
- Failed waypoints highlighted with red vertical lines

**Panel 2: IK Convergence Error**
- Scatter plot of final residual norm for failed waypoints
- Log scale to show convergence quality
- Lower values indicate the solver got closer to a solution

**Panel 3: IK Iterations**
- Number of iterations attempted before failure
- Max iterations (50) indicates solver timeout
- Low iterations may indicate early divergence

**Panel 4: Jacobian Singularity**
- Minimum singular value at the failed configuration
- Red dashed line shows singularity threshold (0.01)
- Values below threshold indicate near-singular configurations

**Panel 5: Workspace Distance**
- Distance from robot base for failed vs successful waypoints
- Helps identify if failures are at workspace boundaries
- Green dots show successful waypoints for comparison

**Panel 6: 3D Trajectory Visualization**
- 3D plot of trajectory path
- Failed waypoints marked with large red X
- Robot base shown as blue circle
- Helps visualize spatial distribution of failures

#### Joint Limit Analysis Plot (2 panels)

**Panel 1: Joint Positions at Failure**
- Bar chart showing all 6 joint angles at each failed waypoint
- Dashed lines indicate joint limits
- Helps identify which joints are problematic

**Panel 2: Joint Limit Distance Heatmap**
- Color-coded distance to nearest joint limit
- Red (0.0) = at limit, Green (0.5) = centered
- Each cell shows numerical value
- Quickly identifies joints near their limits

## Metrics Tracked for Failed Waypoints

### IK Solver Metrics
1. **Iterations**: Number of optimization iterations attempted (max 50)
2. **Residual Norm**: Final pose error (position + orientation weighted)
3. **Failure Reason**: 
   - `max_iter_exceeded` - Solver timeout
   - `backtracking_failed` - Step size reduction failed
   - Other solver-specific reasons

### Singularity Metrics
4. **Min Singular Value (σ_min)**: Minimum singular value of Jacobian
   - < 0.01: Near singularity
   - < 0.001: Very close to singularity
5. **Max Singular Value (σ_max)**: Maximum singular value of Jacobian
6. **Condition Number**: σ_max / σ_min (not shown in plots but available)

### Workspace Metrics
7. **Distance from Origin**: Euclidean distance from robot base
   - Helps identify workspace boundary issues
8. **Distance from Previous Config**: Joint space distance from last successful pose
   - Large values indicate discontinuous jumps

### Joint Limit Metrics
9. **Joint Limit Violations**: Which joints exceed limits and by how much
10. **Joint Limit Distances**: Normalized distance to nearest limit (0-0.5)
    - 0.0 = at limit, 0.5 = perfectly centered

## Common Failure Patterns

### Pattern 1: Near Singularity
- **Symptoms**: σ_min < 0.01, high condition number
- **Visualization**: Panel 4 shows low singular values
- **Cause**: Robot configuration close to singularity
- **Solution**: Modify trajectory or add waypoints to avoid singular regions

### Pattern 2: Joint Limit Violations
- **Symptoms**: Joint positions at/beyond limits
- **Visualization**: Panel 1 of joint limit analysis shows violations
- **Cause**: Target pose requires joint angles outside robot limits
- **Solution**: Adjust knife pose, modify trajectory, or use different robot

### Pattern 3: Workspace Boundary
- **Symptoms**: High distance from origin, high residual norm
- **Visualization**: Panel 5 shows failed points far from base
- **Cause**: Target pose outside robot's reachable workspace
- **Solution**: Move robot base closer or adjust trajectory

### Pattern 4: Large Configuration Jump
- **Symptoms**: High distance from previous config, solver divergence
- **Visualization**: Low iteration count with high residual
- **Cause**: Waypoint spacing too large for sequential IK
- **Solution**: Add intermediate waypoints or use better initial guess

### Pattern 5: Solver Convergence Issues
- **Symptoms**: Residual norm stuck at moderate value, max iterations
- **Visualization**: Panel 2 shows residual around 1e-3 to 1e-2
- **Cause**: Local minimum, poor conditioning, or tight tolerances
- **Solution**: Adjust IK solver parameters (tolerance, damping, max_step)

## Usage

The debug features activate automatically when running feasibility analysis:

```bash
python feasibility_analysis.py --toolpath <csv> --urdf <urdf> --knife-config <yaml>
```

If any trajectory has unreachable waypoints:
1. Console will show: "⚠ Trajectory N has X unreachable waypoints - generating debug analysis..."
2. Debug plots are automatically generated in `trajectory_N/unreachability_debug/`
3. Detailed failure information is added to `analysis_report.txt`

## Files Modified

1. **core/feasibility_checks.py**
   - Enhanced `FeasibilityResult` dataclass with debug fields
   - Modified `analyze_waypoint()` to capture failure diagnostics

2. **utils/feasibility_plot.py**
   - Added `plot_ik_failure_analysis()` - 6-panel failure visualization
   - Added `plot_joint_limit_analysis()` - Joint limit proximity analysis

3. **feasibility_analysis.py**
   - Automatic detection of failed trajectories
   - Debug folder creation
   - Enhanced report generation with failure details
   - Integration of debug plotting functions

4. **utils/__init__.py**
   - Exported new plotting functions

## Debugging Workflow

1. **Run feasibility analysis** on your trajectory
2. **Check console output** for warning messages about unreachable waypoints
3. **Open analysis_report.txt** to see detailed failure information
4. **Review debug plots** in `unreachability_debug/` folder:
   - Start with IK failure analysis for overview
   - Check joint limit analysis if near limits
5. **Identify pattern** using metrics above
6. **Take corrective action** based on failure type

## Technical Details

### Debug Information Structure

```python
ik_debug_info = {
    'ik_solver_info': {
        'iterations': int,          # Number of IK iterations
        'residual_norm': float,     # Final pose error
        'reason': str,              # Failure reason
        'sigma_min': float,         # Min singular value
        'sigma_max': float,         # Max singular value
        'converged': bool,          # Convergence status
        'clip_count': int          # Joint limit clipping count
    },
    'distance_from_origin_m': float,
    'distance_from_prev_config_rad': float,
    'joint_limit_violations': {
        'lower': [float] * 6,       # Lower limit violations (rad)
        'upper': [float] * 6,       # Upper limit violations (rad)
        'any_violation': bool
    },
    'joint_limit_distances': [float] * 6,  # Distance to nearest limit
    'final_q_rad': [float] * 6     # Final joint configuration
}
```

### IK Solver Parameters

The IK solver uses damped least-squares with:
- Max iterations: 50
- Tolerance: 1e-4
- Rotation weight: 0.2
- Translation weight: 1.0
- Initial damping (λ₀): 1e-3
- Max damping (λ_max): 10
- Max step size: 0.2 rad

These can be adjusted in `config/ik_config.yaml` if needed.

## Future Enhancements

Potential additions for even better debugging:
1. **Multiple IK solution attempts**: Try different initialization strategies
2. **Workspace visualization**: Show reachable workspace boundary
3. **Trajectory smoothing**: Suggest optimized waypoint spacing
4. **Alternative configurations**: Suggest joint angle modifications
5. **Collision detection**: Check if unreachability is due to collisions
6. **Historical comparison**: Compare with known successful trajectories

## Questions?

If you encounter unexpected unreachability issues:
1. Check the debug plots for visual clues
2. Review the detailed failure metrics in the report
3. Compare failed waypoints with nearby successful ones
4. Consider adjusting IK solver parameters
5. Verify URDF and knife configuration are correct
