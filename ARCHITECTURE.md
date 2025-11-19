# System Architecture Documentation

Deep dive into how the Robotics APCC batch processing system works internally.

## Table of Contents

- [Data Flow](#data-flow)
- [Core Modules](#core-modules)
- [Key Algorithms](#key-algorithms)
- [Coordinate Frames](#coordinate-frames)
- [Processing Pipeline](#processing-pipeline)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT SOURCES                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 1. CSV Trajectory Files    │ 2. Robot URDF Files    │ 3. Config YAML │
│    (positions in mm)       │    (robot models)      │ (knife poses)  │
└──────────┬──────────────────────────┬──────────────────────┬─────────┘
           │                          │                      │
           ▼                          ▼                      ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────┐
│ CSV Handling        │   │ Batch Processor     │   │ Config Load  │
│ - Parse CSV         │   │ - Robot discovery   │   │ - Knife pose │
│ - Validate data     │   │ - Toolpath discover │   │ - Parameters │
│ - Convert mm→m      │   │ - Generate combos   │   └──────────────┘
│ Returns: trajectories   │   └─────────────────────┘
│ (n_poses, 7) in meters  │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ FOR EACH TRAJECTORY:                                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. TRANSFORMATION STAGE                                 │
│     ┌─────────────────────────────────────────────┐     │
│     │ transform_to_ee_poses_matrix_with_pose()    │     │
│     │ Input: trajectory (n_poses, 7) [meters]     │     │
│     │        knife_pose (translation, quaternion) │     │
│     │ Output: transformed trajectory [meters]     │     │
│     └─────────────────────────────────────────────┘     │
│                                                           │
│  2. KINEMATIC ANALYSIS STAGE                             │
│     ┌─────────────────────────────────────────────┐     │
│     │ analyze_trajectory()                         │     │
│     │ Input: URDF, transformed trajectory          │     │
│     │ Process:                                     │     │
│     │   - For each pose: IK solving via Pinocchio │     │
│     │   - Compute joint angles (if reachable)     │     │
│     │   - Calculate manipulability metrics        │     │
│     │ Output: joint_angles, reachability, manip   │     │
│     └─────────────────────────────────────────────┘     │
│                                                           │
│  3. CONTINUITY ANALYSIS STAGE (if reachable)             │
│     ┌─────────────────────────────────────────────┐     │
│     │ analyze_trajectory_continuity()              │     │
│     │ Input: trajectory, joint_angles, speed_mm/s │     │
│     │ Process:                                     │     │
│     │   - Compute unified pose distance metric    │     │
│     │   - Scale timing by joint velocity limits   │     │
│     │   - Check C¹ continuity (velocity checks)   │     │
│     │ Output: continuity_report.yaml              │     │
│     └─────────────────────────────────────────────┘     │
│                                                           │
│  4. VISUALIZATION STAGE                                  │
│     ┌─────────────────────────────────────────────┐     │
│     │ Generate plots:                              │     │
│     │ - 3D visualization (points in space)         │     │
│     │ - Data comparison (original vs transformed)  │     │
│     │ - Delta analysis (pose-to-pose changes)     │     │
│     └─────────────────────────────────────────────┘     │
│                                                           │
└──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ OUTPUT RESULTS                                            │
├──────────────────────────────────────────────────────────┤
│ results/csv_name/robot_model_pose/Traj_N_[1-M]/         │
│ ├── experiment_results.yaml (IK, reachability, manip)   │
│ ├── experiment.csv (joint angles for all poses)          │
│ ├── Traj_N_visualization.png (3D point cloud)            │
│ ├── pose_viz/                                             │
│ │   ├── Traj_N_comparison.png (original vs transformed)  │
│ │   ├── Traj_N_data_comparison.png (translation/rotation)│
│ │   └── Traj_N_delta_analysis.png (deltas between poses) │
│ └── continuity/                                           │
│     └── continuity_analysis.yaml (C¹ report)             │
└──────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Batch Orchestrator (`batch_trajectory_processor.py`)

**Responsibility:** Coordinate all processing stages

**Key Methods:**
- `__init__()` - Load config, setup output directories
- `process_all()` - Main 3-level nested loop orchestration
- `_discover_components()` - Find robots, CSVs, knife poses
- `_process_csv_file()` - Process single CSV file
- `_process_single_trajectory()` - Process single trajectory with all analysis stages
- `_run_visualization()` - Delegate to visualization modules

**Input:** Config YAML file  
**Output:** Organized results in `results/` directory  
**Dependencies:** All other modules

---

### 2. CSV Handling (`utils/csv_handling.py`)

**Responsibility:** Parse trajectory files with validation and unit conversion

**Key Functions:**
- `read_trajectories_from_csv()` - Main entry point
  - Reads CSV line by line
  - Validates data format (7 columns minimum)
  - Converts positions from mm to meters
  - Normalizes quaternions
  - Handles trajectory separators ("T0" markers)
  - Returns: `List[np.ndarray]` where each array is shape `(n_poses, 7)`

**CSV Format Expected:**
```
x_mm, y_mm, z_mm, qw, qx, qy, qz, [optional: speed_mm/s]
100.0, 200.0, 300.0, 1.0, 0.0, 0.0, 0.0, 150.0
...
T0   <- trajectory separator
200.0, 300.0, 400.0, 1.0, 0.0, 0.0, 0.0, 200.0
...
```

**Critical Transformations:**
- Position: mm → meters (÷ 1000)
- Quaternion: Auto-normalized to unit length

---

### 3. Coordinate Transformations (`utils/handle_transforms.py`)

**Responsibility:** Transform trajectories between coordinate frames

**Coordinate Frames Used:**
- **T_P_K** (Plate→Knife): Raw CSV data - knife pose in manufacturing plate coordinates
- **T_B_K** (Base→Knife): Target frame - knife pose in robot base coordinates
- **T_B_P** (Base→Plate): Intermediate - plate pose in robot base

**Key Functions:**
- `transform_to_ee_poses_matrix_with_pose()`
  - Input: trajectories in meters, knife_translation (m), knife_rotation (quat)
  - Process: Applies T_B_K transformation to all poses
  - Output: Transformed trajectory in robot base frame
  - Formula: **T_B_K = T_B_P × T_P_K × T_K_K (knife identity)**

**Transformation Pipeline:**
```
CSV Data (T_P_K)
     ↓
     × T_B_K transformation matrix
     ↓
Robot Base Frame (T_B_K)
     ↓ [To IK Solver]
```

---

### 4. Kinematic Analysis (`trajectory_processing/analyze_irb1300_trajectory.py`)

**Responsibility:** Compute joint angles and kinematic metrics

**Key Function:** `analyze_trajectory()`
- Input: URDF path, transformed trajectory (meters)
- Process:
  1. Load robot model via Pinocchio
  2. For each pose:
     - Call IK solver with damped Levenberg-Marquardt
     - Compute Jacobian at solution
     - Calculate Yoshikawa manipulability: m = √det(J × J^T)
     - Check reachability (IK converged within tolerance)
  3. Aggregate statistics
- Output: `experiment_results.yaml` with per-pose IK status, joint angles, manipulability

**IK Solver Details:**
- Algorithm: Damped Levenberg-Marquardt
- Parameters:
  - `max_iterations`: 2000 (configurable)
  - `tolerance`: 1e-4 m (translational error)
  - `lambda0`: 1e-3 (initial damping)
  - `lambda_max`: 10.0 (maximum damping)
- Rotational error weight: Configurable (default 0.2)

**Manipulability Metric:**
- **Yoshikawa Index**: m = √(det(J × J^T))
  - m = 1.0: Isotropic (equal motion in all directions)
  - m → 0: Singularity approaching
  - Used for normalized metric: m_norm = m / m_max_for_workspace

---

### 5. Continuity Analysis (`trajectory_processing/trajectory_continuity_analyzer.py`)

**Responsibility:** Check trajectory smoothness and joint velocity constraints

**Continuity Levels:**
- **C⁰ Continuity**: Position continuity (always satisfied for pose lists)
- **C¹ Continuity**: Velocity continuity (checked against joint limits)
- **C² Continuity**: Acceleration continuity (computed but not enforced since we don't have joint acceleration limits)

**Key Algorithm:** Unified Pose Timing Calculator
- Input:
  - Trajectory poses (m)
  - Joint configurations (rad)
  - Speed from CSV 8th column (mm/s)
- Process:
  1. Compute unified pose distance: d = √(d_linear² + (scale × d_angle)²)
  2. Compute timing based on speed and pose distances
  3. For each segment, enforce maximum joint velocity:
     - T_joint = max_j(|Δq_j| / vmax_j)
     - T_final = max(T_pose, T_joint)
  4. Check if any joint exceeds limits → C¹ violation
- Output: `continuity_analysis.yaml` with violation report

**Key Parameters:**
```yaml
continuity:
  pose_scale_m_per_rad: 0.1          # Rotation → distance scale
  safety_factor: 1.05                # 5% margin on limits
  velocity_limits_rad_s: [4.443, ...] # Per-joint velocity limits
```

---

### 6. Math Utilities (`utils/math_utils.py`)

**Responsibility:** Core quaternion and transformation operations

**Key Functions:**

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `quat_mul(q1, q2)` | Quaternion multiplication | Two quats [w,x,y,z] | Product quat |
| `quat_to_rot_matrix(q)` | Convert quat to 3×3 rotation | Quat [w,x,y,z] | 3×3 matrix |
| `quat_conjugate(q)` | Quaternion inverse | Quat [w,x,y,z] | Conjugate quat |
| `normalize_quat(q)` | Unit quaternion normalization | Quat [w,x,y,z] | Normalized quat |
| `pose_to_matrix()` | SE(3) transformation from pose | Translation + quat | 4×4 matrix |
| `matrix_to_pose()` | Extract pose from SE(3) matrix | 4×4 matrix | Translation + quat |

**Quaternion Convention:** All quaternions use [w, x, y, z] format (real part first)

---

### 7. Visualization (`trajectory_processing/trajectory_visualizer.py`)

**Responsibility:** Generate publication-quality plots

**Generated Plots:**

1. **3D Trajectory Visualization**
   - Shows pose positions as scatter points (no connecting lines)
   - Optional coordinate frame axes at each pose
   - Color-coded by trajectory

2. **Data Comparison Graphs**
   - 2×2 subplot grid
   - Top: Translation (X, Y, Z in meters)
   - Bottom: Rotation (QW, QX, QY, QZ components)
   - Both original and transformed shown

3. **Delta Analysis Graphs**
   - Pose-to-pose changes (differences between consecutive poses)
   - Position deltas (ΔX, ΔY, ΔZ in mm)
   - Rotation deltas (quaternion differences)

**Plot Quality:**
- Resolution: 150 DPI (default), 300 DPI (high quality)
- Format: PNG
- Size: Adaptive based on data

---

## Key Algorithms

### IK Solving Algorithm (Pinocchio)

**Problem:** Given end-effector pose T_B_K, find joint angles q

**Method:** Damped Levenberg-Marquardt (Non-linear Least Squares)

**Algorithm:**
```python
q = q_init  # Start from home configuration or previous solution
for iteration in range(max_iterations):
    # Compute error
    T_current = forward_kinematics(q)
    error = pose_error(T_target, T_current)
    
    # Compute Jacobian
    J = compute_jacobian(q)
    
    # Solve with damping (Levenberg-Marquardt)
    lambda = initial_lambda
    while True:
        # Solve (J^T J + lambda*I) dq = J^T error
        A = J.T @ J + lambda * np.eye(n_joints)
        b = J.T @ error
        dq = solve(A, b)
        
        q_new = q + dq
        error_new = pose_error(T_target, forward_kinematics(q_new))
        
        if ||error_new|| < ||error||:
            q = q_new
            lambda = lambda / 10  # Reduce damping
            break
        else:
            lambda = lambda * 10  # Increase damping
    
    if ||error|| < tolerance:
        return q  # Converged!
    
return None  # IK failed
```

**Reachability:** If IK converges within tolerance and max_iterations, pose is reachable

---

### Yoshikawa Manipulability Index

**Definition:** m = √(det(J × J^T))

**Interpretation:**
- Measures how well robot can move in all directions
- m = 1.0: Isotropic (equal dexterity in all directions)
- m → 0: Singularity (constrained motion)

**Normalized Manipulability:**
```
m_norm = m / m_max_workspace
```
- Accounts for workspace size
- `m_max_workspace` computed from random samples in workspace

**Use:** Quality metric for trajectory feasibility

---

### C¹ Continuity Verification

**Goal:** Ensure joint velocities don't exceed hardware limits

**Metric:** Unified pose distance
```
d_pose = sqrt(d_linear^2 + (pose_scale * d_angle)^2)
```

**Timing Calculation:**
```python
# From pose distances and CSV speed
time_from_speed = accumulated_distance / speed_mm_s

# From joint angle changes and joint velocity limits
for each joint j:
    time_from_joint_j = |Δq_j| / velocity_limit_j
time_from_joints = max(all_joints)

# Final segment time enforces both
segment_time = max(time_from_speed, time_from_joints)

# Check for violations
if any_joint_velocity > limit:
    C1_violation = True
```

**Result:** Pass/Fail report with detailed violation analysis

---

## Coordinate Frames

### Frame Hierarchy

```
World
  ├─ Robot Base (B)
  │  ├─ Link 1, Link 2, ..., Link 6
  │  └─ End-Effector (E)
  │
  ├─ Manufacturing Plate (P)
  │  └─ Knife/Tool (K) ← Position varies
  │
  └─ Calibration
     └─ T_B_K: Fixed knife pose in base frame
```

### Transformation Chain

**From CSV to Robot Control:**

```
CSV Data (in Plate frame)
    [x_p, y_p, z_p, qw_p, qx_p, qy_p, qz_p]
              ↓
       Pose Transformation
       T_P_K = [ R_P_K   p_p_k ]
               [   0        1   ]
              ↓
       Apply Calibration
       T_B_K = T_B_P × T_P_K × T_K_E
              ↓
    Robot Base Frame
    [x_b, y_b, z_b, qw_b, qx_b, qy_b, qz_b]
              ↓
       IK Solver
              ↓
    Joint Angles
    [q1, q2, q3, q4, q5, q6] (radians)
```

### Convention Notes

- **Origin:** Robot base (origin of URDF)
- **Axes:** Right-handed coordinate system
- **Units:** Meters (internally), Millimeters (CSV), Radians (angles)
- **Quaternions:** [w, x, y, z] format (real part first, NOT [x, y, z, w])

---

## Processing Pipeline

### Single Trajectory Processing Flow

```
┌─────────────────────────────────────────┐
│ Single Trajectory Input                  │
│ trajectory_m[n_poses, 7]                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Step 1: Transformation                  │
│ Apply knife pose via handle_transforms  │
│ Output: trajectory_base[n_poses, 7]     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Step 2: Kinematic Analysis              │
│ - Load URDF via Pinocchio               │
│ - Run IK for all poses                  │
│ - Compute manipulability                │
│ Output:                                 │
│ - joint_angles[n_poses, 6]              │
│ - reachability[n_poses] bool            │
│ - manipulability[n_poses]               │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        NO_REACH      ALL_REACH
        │             │
        ▼             ▼
      SKIP      ┌──────────────────────┐
            CONTINUITY│ Step 3: Continuity    │
            ANALYSIS  │ - Check C¹ limits     │
                      │ Output: violations    │
                      └──────────────┬───────┘
                                     │
                      ┌──────────────┴───────┐
                      │                      │
                    PASS                   FAIL
                      │                      │
                      └──────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────┐
│ Step 4: Visualization & Reporting       │
│ - Generate 3D plots                     │
│ - Generate data comparison              │
│ - Generate delta analysis               │
│ - Save all results to YAML/CSV/PNG      │
└─────────────────────────────────────────┘
```

### Nested Loop Structure

```
FOR each CSV file:
    load_csv()
    
    FOR each robot:
        load_urdf()
        
        FOR each knife pose:
            extract_knife_params()
            
            FOR each trajectory in CSV:
                # Single Trajectory Processing Flow (above)
                transform_trajectory()
                analyze_trajectory()  # IK solving
                IF all_reachable:
                    analyze_continuity()
                generate_visualizations()
                save_results()
    
    save_csv_summary()

save_batch_summary()
```

### Error Recovery

- **CSV Parse Error**: Skip to next CSV
- **URDF Load Error**: Skip to next robot
- **Transformation Error**: Skip trajectory with warning
- **IK Failure**: Mark poses as unreachable, skip continuity
- **Continuity Error**: Log warning, continue
- **Visualization Error**: Log warning, continue to next trajectory

---

## Performance Characteristics

### Time Complexity per Trajectory

| Step | Complexity | Time (100 poses) |
|------|-----------|-----------------|
| CSV parsing | O(n) | <1ms |
| Transformation | O(n) | <1ms |
| IK solving (n poses × 2000 iterations) | O(n × iter) | 10-30s |
| Continuity analysis | O(n) | <100ms |
| Visualization | O(n) | 1-2s |
| **Total** | | **15-40s** |

### Memory Usage

- Single trajectory: ~1-2 MB
- URDF loaded: ~5-10 MB
- Entire batch (100 trajectories): ~200-500 MB

### Parallelization Opportunity

- **Easy**: Process multiple CSVs in parallel
- **Medium**: Process trajectories in parallel (IK solving is bottleneck)
- **Hard**: Batch IK solving (requires Pinocchio changes)

---

## Configuration Impact

### Key Config Parameters

| Parameter | Impact | Default |
|-----------|--------|---------|
| `ik.max_iterations` | IK convergence time (slower = more reliable) | 2000 |
| `ik.tolerance` | IK precision (smaller = stricter) | 1e-4 m |
| `continuity.safety_factor` | Velocity limit margin (higher = more conservative) | 1.05 |
| `pose_scale_m_per_rad` | Rotation weight in unified metric | 0.1 m/rad |
| `visualization.scale` | Coordinate frame size (0 = disabled) | 0.01 m |

### Typical Processing Times

- **Small trajectory** (50 poses): 15-20s per robot configuration
- **Medium trajectory** (100 poses): 30-40s per robot configuration
- **Large trajectory** (200+ poses): 60-80s per robot configuration

---

## Debugging Tips

### Enabling Verbose Output

```bash
python batch_trajectory_processor.py -c config.yaml 2>&1 | tee processing.log
```

### Checking IK Solutions

Look in `experiment_results.yaml`:
```yaml
reachability_summary:
  total_poses: 100
  reachable: 95
  unreachable: 5
  reachability_percentage: 95.0
```

### Inspecting Continuity Violations

```yaml
c1_velocity_check:
  passed: true/false
  violations:
    - joint: 3
      max_velocity_rad_s: 8.727
      required_velocity_rad_s: 9.2
      margin_percent: 5.4
```

### Visualizing Results

Use debug tools: `python scripts/debug/pose_3d_visualizer.py ...`

---

## References

- **Pinocchio**: https://github.com/stack-of-tasks/pinocchio
- **ABB IRB 1300**: https://new.abb.com/industrial-robots
- **Yoshikawa Manipulability**: T. Yoshikawa (1985), "Manipulability of Robotic Mechanisms"

---

**Last Updated:** 2025-01-18  
**System Version:** 1.0

