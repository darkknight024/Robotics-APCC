# Robot Trajectory Metrics and Ranking Guide

This document provides a comprehensive guide to all trajectory metrics computed in the system and how they are used for ranking trajectories.

---

## Table of Contents

1. [Per-Waypoint Metrics](#per-waypoint-metrics)
2. [Per-Trajectory Metrics](#per-trajectory-metrics)
3. [Ranking System](#ranking-system)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Implementation Details](#implementation-details)

---

## Per-Waypoint Metrics

These metrics are computed for each waypoint in a trajectory and stored in `FeasibilityResult`:

### 1. **Joint Solution** (`joint_positions_rad`)
- **Type**: `np.ndarray` (n_joints,)
- **Purpose**: The IK solution (joint angles) for this waypoint
- **Critical**: Yes - Required for all subsequent computations
- **Data Structure**: Array of joint angles in radians

### 2. **IK Validity** (`is_reachable`)
- **Type**: `bool`
- **Purpose**: Binary flag indicating if waypoint is kinematically reachable
- **Critical**: Yes - Determines `reachability_ok` flag for trajectory
- **Data Structure**: Boolean (True/False)

### 3. **Condition Number** (`condition_number`)
- **Type**: `float` (κ = σ_max / σ_min)
- **Purpose**: **CRITICAL for Safety** - Measures proximity to singularity
- **Critical**: Yes - Used directly for `safety_score` (max across trajectory)
- **Formula**: `κ = σ_max / σ_min` where σ are singular values of Jacobian
- **Interpretation**: Lower is better (smaller condition number = safer, away from singularity)
- **Data Structure**: Float (infinity if near singular)

### 4. **Min Singular Value** (`min_singular_value`)
- **Type**: `float` (σ_min)
- **Purpose**: **CRITICAL** - Required to compute condition number
- **Critical**: Yes - Cannot compute condition number without this
- **Use**: Also used for per-waypoint safety checks (Safety Tiers)
- **Data Structure**: Float (non-negative)

### 5. **Max Singular Value** (`max_singular_value`)
- **Type**: `float` (σ_max)
- **Purpose**: Required to compute condition number
- **Critical**: Yes - Used with σ_min to compute condition number
- **Data Structure**: Float (non-negative)

### 6. **Manipulability Index** (`manipulability`)
- **Type**: `float` (Yoshikawa measure)
- **Purpose**: **CRITICAL for Dexterity** - Measures motion capability
- **Critical**: Yes - Used for `dexterity_score` (mean across trajectory)
- **Formula**: `w = sqrt(det(J_normalized * J_normalized^T))`
- **Interpretation**: Higher is better (more manipulability = more dexterous)
- **Note**: Should be Task-Oriented (Projected) if selection matrix is implemented
- **Data Structure**: Float (non-negative, dimensionless)

### 7. **Joint Velocity Ratio** (`joint_velocity_ratio`)
- **Type**: `float` (max of |dq/dt| / limit across joints)
- **Purpose**: **CRITICAL for C1 Feasibility** - Checks velocity limit compliance
- **Critical**: Yes - Determines `c1_ok` flag
- **Computation**: Requires timing information (timestamps or speed estimate)
- **Formula**: `max(|dq_i/dt| / limit_i)` for all joints i
- **Threshold**: Value > 1.0 indicates C1 violation
- **Data Structure**: Float (non-negative)

### 8. **Distance to Joint Limits** (`distance_to_joint_limits`)
- **Type**: `float`
- **Purpose**: Minimum distance to any joint limit across all joints
- **Critical**: No - Useful for analysis but not used in ranking
- **Data Structure**: Float (in radians)

### 9. **Joint Space Distance** (`joint_space_distance`)
- **Type**: `float`
- **Purpose**: Euclidean distance from previous waypoint in joint space
- **Critical**: Yes - Used for C0 continuity check
- **Computation**: Only computed when previous waypoint exists
- **Data Structure**: Float (in radians)

### 10. **Near Singularity Flag** (`near_singularity`)
- **Type**: `bool`
- **Purpose**: Boolean indicating proximity to singularity
- **Critical**: No - Derived from min_singular_value
- **Data Structure**: Boolean

---

## Per-Trajectory Metrics

These metrics are computed from per-waypoint data and used for ranking:

### Feasibility Flags (Booleans)

#### 1. **`reachability_ok`**
- **Type**: `bool`
- **Computation**: `reachable_count == n_waypoints`
- **Purpose**: All waypoints must be reachable
- **Critical**: Yes - Primary feasibility gate
- **Use in Ranking**: Trajectories with `reachability_ok = False` are filtered out

#### 2. **`c0_ok`**
- **Type**: `bool`
- **Computation**: `max_joint_jump < joint_jump_limit_rad`
- **Purpose**: C0 continuity - no excessive joint jumps
- **Critical**: Yes - Position continuity check
- **Use in Ranking**: Trajectories with `c0_ok = False` are filtered out
- **Configuration**: Requires `joint_jump_limit_rad` from `config/robots_config.yaml`

#### 3. **`c1_ok`**
- **Type**: `bool`
- **Computation**: `max_velocity_ratio <= 1.0`
- **Purpose**: C1 continuity - velocity limits respected
- **Critical**: Yes - Velocity continuity check
- **Use in Ranking**: Trajectories with `c1_ok = False` are filtered out
- **Configuration**: Requires `velocity_limits_rad_s` from robot config

### Ranking Scores (Floats)

#### 1. **`safety_score`**
- **Type**: `float`
- **Computation**: `max_condition_number` across all reachable waypoints
- **Purpose**: **Level 1: Safety Sort** - Worst-case singularity proximity
- **Critical**: Yes - Primary safety metric
- **Interpretation**: Lower is better (smaller condition number = safer)
- **Use in Ranking**: Primary sort key (ascending) - safest trajectories first
- **Data Structure**: Float (infinity if any waypoint is singular)

#### 2. **`dexterity_score`**
- **Type**: `float`
- **Computation**: `mean_manipulability` across all reachable waypoints
- **Purpose**: **Level 2: Dexterity Sort** - Overall motion efficiency
- **Critical**: Yes - Primary dexterity metric
- **Interpretation**: Higher is better (more manipulability = more dexterous)
- **Use in Ranking**: Secondary sort key (descending) - most dexterous trajectories first
- **Note**: This is the **average**, not max. A steady 0.5 is better than spikes to 1.0 with lows at 0.1.
- **Data Structure**: Float (non-negative)

#### 3. **`smoothness_score`**
- **Type**: `float`
- **Computation**: `mean_squared_velocity_ratio` = mean of (velocity_ratio²) across segments
- **Purpose**: **Level 3: Smoothness Sort** - Energy-based smoothness measure
- **Critical**: Yes - Primary smoothness metric
- **Interpretation**: Lower is better (less energy = smoother)
- **Formula**: `mean([(v_i / limit_i)² for all segments i])`
- **Use in Ranking**: Tertiary sort key (ascending) - smoothest trajectories first
- **Data Structure**: Float (non-negative)

### Additional Statistics (For Analysis)

#### Manipulability Statistics
- `mean_manipulability`: Average manipulability (same as `dexterity_score`)
- `min_manipulability`: Minimum manipulability across waypoints
- `max_manipulability`: Maximum manipulability across waypoints
- `std_manipulability`: Standard deviation of manipulability values

#### Singular Value Statistics
- `mean_min_singular_value`: Average minimum singular value
- `min_min_singular_value`: Minimum of minimum singular values
- `max_min_singular_value`: Maximum of minimum singular values
- `mean_max_singular_value`: Average maximum singular value
- `min_max_singular_value`: Minimum of maximum singular values
- `max_max_singular_value`: Maximum of maximum singular values
- `std_min_singular_value`: Standard deviation of minimum singular values

#### Condition Number Statistics
- `mean_condition_number`: Average condition number
- `min_condition_number`: Minimum condition number
- `max_condition_number`: Maximum condition number (same as `safety_score`)
- `std_condition_number`: Standard deviation of condition numbers

#### Joint Limit Statistics
- `mean_distance_to_joint_limits`: Average distance to joint limits
- `min_distance_to_joint_limits`: Minimum distance to joint limits (bottleneck)
- `joint_limit_violation_count`: Number of waypoints violating joint limits
- `joint_limit_violation_rate`: Percentage of waypoints violating joint limits
- `has_violations`: Boolean flag indicating if any violations exist
- `max_velocity_ratio`: Maximum velocity ratio across all segments

#### Path Length Metrics
- `total_joint_space_path_length`: Sum of all joint space distances
- `mean_joint_space_segment_length`: Average joint space segment length

---

## Ranking System

The ranking system uses a **3-level hierarchical sort**:

### Level 1: Feasibility Filtering
Trajectories must pass all feasibility gates:
- `reachability_ok = True` (all waypoints reachable)
- `c0_ok = True` (no excessive joint jumps)
- `c1_ok = True` (velocity limits respected)

Trajectories failing any gate are **filtered out** and not ranked.

### Level 2: Safety Sort (Primary)
Among feasible trajectories, sort by `safety_score` (ascending):
- Lower condition number = safer = better rank
- Trajectories with `safety_score = inf` are ranked last

### Level 3: Dexterity Sort (Secondary)
For trajectories with same `safety_score`, sort by `dexterity_score` (descending):
- Higher mean manipulability = more dexterous = better rank

### Level 4: Smoothness Sort (Tertiary)
For trajectories with same `safety_score` and `dexterity_score`, sort by `smoothness_score` (ascending):
- Lower mean squared velocity ratio = smoother = better rank

### Ranking Algorithm Pseudocode

```python
def rank_trajectories(trajectory_stats_list):
    # Step 1: Filter feasible trajectories
    feasible = [t for t in trajectory_stats_list 
                if all(t['feasibility_flags'].values())]
    
    # Step 2: Sort by safety_score (ascending)
    feasible.sort(key=lambda t: t['safety_score'])
    
    # Step 3: Group by safety_score and sort groups by dexterity_score (descending)
    from itertools import groupby
    ranked = []
    for safety_score, group in groupby(feasible, key=lambda t: t['safety_score']):
        group_list = list(group)
        group_list.sort(key=lambda t: -t['dexterity_score'])  # Negative for descending
        
        # Step 4: Within same safety and dexterity, sort by smoothness (ascending)
        for dexterity_score, sub_group in groupby(group_list, key=lambda t: t['dexterity_score']):
            sub_group_list = list(sub_group)
            sub_group_list.sort(key=lambda t: t['smoothness_score'])
            ranked.extend(sub_group_list)
    
    return ranked
```

---

## Configuration

### Robot Configuration (`config/robots_config.yaml`)

The system automatically loads robot-specific parameters from the configuration file:

```yaml
constants:
  joint_jump_limit_rad: 0.5  # Maximum allowed joint jump (C0 check)

robots:
  - name: "IRB 1300-7/1.4"
    velocity_limits_rad_s: [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]
    # ... other robot parameters
```

**Parameters Used:**
- `velocity_limits_rad_s`: Per-joint velocity limits (for C1 check and `smoothness_score`)
- `joint_jump_limit_rad`: Maximum allowed joint jump (for C0 check) - from `constants` section

### Loading Robot Configuration

```python
from utils import get_robot_by_name

# Load robot config
robot_config = get_robot_by_name("IRB 1300-7/1.4")

# Access parameters
velocity_limits = np.array(robot_config.velocity_limits_rad_s)
joint_jump_limit = robot_config.joint_jump_limit_rad
```

---

## Usage Examples

### Basic Usage with Robot Config

```python
from core import FeasibilityAnalyzer, load_robot_model
from core.ik_solver import IKSolver
from core.fk_solver import FKSolver
from utils import get_robot_by_name, load_ik_config_as_object
import numpy as np

# Load robot configuration
robot_config = get_robot_by_name("IRB 1300-7/1.4")

# Setup robot model
model, data = load_robot_model(robot_config.urdf_path)
ik_config = load_ik_config_as_object()
ik_solver = IKSolver(model, data, config=ik_config)
fk_solver = FKSolver(model, data, ee_frame_name=ik_config.ee_frame_name)

# Create analyzer with robot config parameters
analyzer = FeasibilityAnalyzer(
    model, data, ik_solver, fk_solver,
    characteristic_length_m=robot_config.reach_m,
    velocity_limits_rad_s=np.array(robot_config.velocity_limits_rad_s),
    joint_jump_limit_rad=robot_config.joint_jump_limit_rad
)

# Analyze trajectory
positions = np.array([[0.5, 0.0, 0.5], [0.6, 0.0, 0.5], ...])  # (n, 3)
quaternions = np.array([[1, 0, 0, 0], [1, 0, 0, 0], ...])  # (n, 4)

trajectory_stats = analyzer.analyze_trajectory(positions, quaternions)

# Access critical ranking metrics
flags = trajectory_stats['feasibility_flags']
if flags['reachability_ok'] and flags['c0_ok'] and flags['c1_ok']:
    safety = trajectory_stats['safety_score']      # Lower is better
    dexterity = trajectory_stats['dexterity_score']    # Higher is better
    smoothness = trajectory_stats['smoothness_score']  # Lower is better
    
    print(f"Safety Score: {safety:.3f}")
    print(f"Dexterity Score: {dexterity:.6f}")
    print(f"Smoothness Score: {smoothness:.6f}")
```

### Accessing Per-Waypoint Metrics

```python
# Access per-waypoint results
for i, result in enumerate(trajectory_stats['per_waypoint_results']):
    if result.is_reachable:
        print(f"Waypoint {i}:")
        print(f"  Condition Number: {result.condition_number:.3f}")
        print(f"  Manipulability: {result.manipulability:.6f}")
        print(f"  Min Singular Value: {result.min_singular_value:.6f}")
        if result.joint_velocity_ratio is not None:
            print(f"  Velocity Ratio: {result.joint_velocity_ratio:.3f}")
```

### Ranking Multiple Trajectories

```python
# Analyze multiple trajectories
trajectory_results = []
for trajectory in trajectories:
    stats = analyzer.analyze_trajectory(trajectory['positions'], trajectory['quaternions'])
    trajectory_results.append({
        'name': trajectory['name'],
        'stats': stats
    })

# Filter feasible trajectories
feasible = [t for t in trajectory_results 
            if all(t['stats']['feasibility_flags'].values())]

# Sort by ranking scores
feasible.sort(key=lambda t: (
    t['stats']['safety_score'],           # Level 1: Safety (ascending)
    -t['stats']['dexterity_score'],      # Level 2: Dexterity (descending)
    t['stats']['smoothness_score']        # Level 3: Smoothness (ascending)
))

# Print ranking
for rank, traj in enumerate(feasible, 1):
    print(f"Rank {rank}: {traj['name']}")
    print(f"  Safety: {traj['stats']['safety_score']:.3f}")
    print(f"  Dexterity: {traj['stats']['dexterity_score']:.6f}")
    print(f"  Smoothness: {traj['stats']['smoothness_score']:.6f}")
```

---

## Implementation Details

### Data Structures

#### `FeasibilityResult` (Per-Waypoint)
```python
@dataclass
class FeasibilityResult:
    is_reachable: bool
    manipulability: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray]
    joint_velocity_ratio: Optional[float]
    distance_to_joint_limits: Optional[float]
    joint_space_distance: Optional[float]
```

#### `analyze_trajectory()` Return Dictionary (Per-Trajectory)
```python
{
    # Basic statistics
    'n_waypoints': int,
    'reachable_count': int,
    'reachability_percent': float,
    'singularity_count': int,
    
    # CRITICAL: Ranking scores
    'feasibility_flags': {
        'reachability_ok': bool,
        'c0_ok': bool,
        'c1_ok': bool
    },
    'safety_score': float,        # max_condition_number
    'smoothness_score': float,    # mean_squared_velocity_ratio
    'dexterity_score': float,     # mean_manipulability
    
    # Statistics (for analysis)
    'mean_manipulability': float,
    'min_manipulability': float,
    'max_manipulability': float,
    'std_manipulability': float,
    # ... (see full list in Per-Trajectory Metrics section)
    
    'per_waypoint_results': List[FeasibilityResult]
}
```

### Mathematical Functions

Pure mathematical utilities are located in `utils/math.py`:
- `compute_joint_space_distance()` - Euclidean distance in joint space
- `compute_distance_to_joint_limits()` - Distance to joint limits
- `compute_joint_velocity_ratio()` - Velocity ratio computation
- `compute_joint_limit_violations()` - Joint limit violation detection

Robot-specific kinematic functions are in `core/feasibility_checks.py`:
- `compute_manipulability()` - Yoshikawa manipulability
- `compute_singularity_proximity()` - Minimum singular value
- `compute_max_singular_value()` - Maximum singular value
- `compute_condition_number()` - Condition number computation

### Configuration Loading

The system automatically loads:
1. **Robot-specific parameters** from `config/robots_config.yaml`:
   - `velocity_limits_rad_s` (per robot)
   - `reach_m` (per robot)
   
2. **Global constants** from `config/robots_config.yaml`:
   - `joint_jump_limit_rad` (shared across all robots)

These are accessed via `get_robot_by_name()` which returns a `RobotConfig` object with all parameters.

---

## Summary

### Critical Metrics for Ranking

**Per-Waypoint:**
1. `joint_positions_rad` - Joint solution
2. `is_reachable` - IK validity
3. `condition_number` - Safety metric
4. `manipulability` - Dexterity metric
5. `min_singular_value` - Required for condition number
6. `joint_velocity_ratio` - C1 feasibility

**Per-Trajectory:**
1. **Feasibility Flags**: `reachability_ok`, `c0_ok`, `c1_ok`
2. **Safety Score**: `max_condition_number` (lower is better)
3. **Dexterity Score**: `mean_manipulability` (higher is better)
4. **Smoothness Score**: `mean_squared_velocity_ratio` (lower is better)

### Ranking Order

1. Filter by feasibility flags
2. Sort by `safety_score` (ascending)
3. Sort by `dexterity_score` (descending)
4. Sort by `smoothness_score` (ascending)

---

**Last Updated**: 2026-01-20
