# Robotics APCC Trajectory Processing System

A comprehensive batch processing system for analyzing 6-DOF robotic trajectories across multiple ABB IRB 1300 robot configurations, toolpaths, and knife poses. The system performs kinematic feasibility analysis, continuity checks, and generates publication-quality visualizations.

## Table of Contents

- [Quickstart](#quickstart)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Output Structure](#output-structure)
- [Running Individual Scripts](#running-individual-scripts)
- [Project Structure](#project-structure)
- [Getting Help](#getting-help)

---

## Quickstart

### Installation

1. **Clone and setup the project:**
   ```bash
   git clone <repo_url>
   cd Robotics-APCC
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **System Requirements:**
   - Python 3.7+
   - For macOS with Apple Silicon: Ensure compatible versions of `torch` and `pinocchio`
   - For Linux/Windows: Standard pip installation works

3. **Verify installation:**
   ```bash
   python -c "import pinocchio; import numpy; import pandas; print('✓ All dependencies installed')"
   ```

### Run Batch Processing

**Basic Usage:**
```bash
cd Robotics-APCC
python scripts/batch_trajectory_processor.py -c scripts/config_batch_processing.yaml
```

**Custom Output Directory:**
```bash
python scripts/batch_trajectory_processor.py -c scripts/config_batch_processing.yaml -o /path/to/output
```

**What happens:**
1. Loads robot URDF files from configured paths
2. Discovers all CSV trajectory files
3. For each CSV, trajectory, and robot combination:
   - Reads trajectory poses (in millimeters)
   - Transforms to robot base frame using knife pose
   - Analyzes kinematic reachability via IK solving
   - Generates analysis results and visualizations
   - Performs C¹ continuity checks (velocity limits)
4. Saves results in organized directory structure:
   ```
   results/
   ├── csv_filename_1/
   │   ├── Robot_Model_pose_1/
   │   │   ├── Traj_1_[1-50]/
   │   │   │   ├── experiment_results.yaml
   │   │   │   ├── experiment.csv
   │   │   │   ├── Traj_1_visualization.png
   │   │   │   |── pose_viz/
   |   |   |   |   └── Traj_1_[1-86]_comparison.png
   │   │   │   |   └── Traj_1_[1-86]_data_comparison.png
   │   │   │   └── continuity/
   |   |   |       └── Traj_1_continuity.png
   │   │   │       └── continuity_analysis.yaml
   │   │   └── Traj_2_[1-100]/
   │   └── csv_summary.yaml
   └── batch_processing_summary.yaml
   ```
### Sample Results

The batch processor generates comprehensive visualizations for each trajectory. Here are representative examples:

#### Joint Angles Analysis
![Joint Angles](./Assests/Robot%20APCC/Toolpaths/sample_results/joint_angles_plot.png)
*Example of joint angle trajectories over time, showing smooth transitions and constraint satisfaction*

#### 3D Pose Visualization Comparison
![3D Pose Comparison](./Assests/Robot%20APCC/Toolpaths/sample_results/pose_comparison_sample.png)
*Original vs. transformed trajectory poses in 3D space, with coordinate frame visualizations*

#### C¹ Continuity Analysis
![Continuity Analysis](Assests/Robot%20APCC/Toolpaths/sample_results/continuity_analysis_sample.png)
*Cartesian velocity profiles and joint velocity constraints, demonstrating C¹ continuity compliance*



---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Batch Trajectory Processor (Orchestration)             │
│  batch_trajectory_processor.py                          │
│  ├─ Discovers robots, toolpaths, knife poses           │
│  ├─ Manages nested loops (CSV → Robot → Pose → Traj)   │
│  └─ Aggregates results                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────────┐
        │                     │                  │
        ▼                     ▼                  ▼
   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
   │ Trajectory  │   │ Trajectory   │   │ Handle          │
   │ Analysis    │   │ Continuity   │   │ Transforms      │
   │ (IK, FK,    │   │ (C¹, C² chks)│   │ (Coordinate     │
   │ Manipulab.) │   │              │   │  transforms)    │
   └─────────────┘   └──────────────┘   └─────────────────┘
        │                    │                   │
        └────────────────────┴───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ Utilities (CSV, Math, Graphs)  │
        │ ├─ csv_handling.py             │
        │ ├─ math_utils.py               │
        │ ├─ graph_utils.py              │
        │ ├─ results_handler.py          │
        │ └─ batch_processor.py          │
        └────────────────────────────────┘
```

---

## Configuration

### Main Config File: `config_batch_processing.yaml`

The configuration file controls all aspects of batch processing. Here's what each section does:

#### **Robots Section**
```yaml
robots:
  - path: "Assests/Robot APCC/IRB-1300-1400-URDF"
    model: "IRB 1300-7/1.4"
    reach_m: 1.4
    velocity_limits_rad_s: [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]
    acceleration_limits_rad_s2: [10.0, 10.0, 10.0, 20.0, 20.0, 30.0] (dummy values-its not used right now)
```
- Defines which robot configurations to process
- Specifies kinematic limits for continuity checking
- Robot discovery is automatic if URDF files exist in `urdf/` subdirectories

#### **Toolpaths Section**
```yaml
toolpaths:
  - "Assests/Robot APCC/Toolpaths/debug/testings"
```
- Specifies CSV files or directories to process
- If directory: all `.csv` files are processed
- If file: only that specific file is processed

#### **Knife Poses Section**
```yaml
knife_poses:
  pose_1:
    description: "Standard knife pose (from calibration data)"
    translation:
      x: -367.773  # mm
      y: -915.815  # mm
      z: 520.4     # mm
    rotation:
      w: 0.00515984
      x: 0.712632
      y: -0.701518
      z: 0.000396522
```
- Defines end-effector to tool transformations
- All translations in **millimeters** (automatically converted to meters)
- Quaternions must be unit quaternions [w, x, y, z]
- Multiple poses can be defined for multi-configuration analysis

#### **Visualization Section**
```yaml
visualization:
  enabled: true
  plot_3d:
    enabled: true
    scale: 0.01        # Axis size in meters (0 = disabled)
    point_size: 20
  data_comparison:
    enabled: true
  delta_analysis:
    enabled: true
```
- Controls what plots are generated
- `scale`: Shows coordinate frame axes at poses (useful for orientation checking)
- Comparison graphs show original vs transformed trajectories

#### **Analysis Section**
```yaml
analysis:
  enabled: true
  ik:
    max_iterations: 2000
    tolerance: 1e-4
    rot_weight: 0.2
    trans_weight: 1.0
```
- IK solver parameters for kinematic feasibility
- Lower tolerance = more precise solutions (slower)
- Weights control translation vs rotation importance

#### **Continuity Section**
```yaml
continuity:
  enabled: true
  pose_scale_m_per_rad: 0.1
  safety_factor: 1.05
  generate_graphs: true
```
- Enables C¹ (velocity) continuity checks
- Uses robot velocity limits from robot configuration
- Speed extracted from CSV 8th column (first pose)
- Safety factor adds margin to limits (1.05 = 5% margin)

### CSV Format

**Expected CSV Structure:**
```
x_mm,  y_mm,  z_mm,  qw,   qx,   qy,   qz,  speed_mm_s
100.0, 200.0, 300.0, 1.0,  0.0,  0.0,  0.0, 150.0
101.0, 201.0, 301.0, 0.99, 0.05, 0.0,  0.0, 150.0
...
T0  (trajectory separator)
200.0, 300.0, 400.0, 1.0,  0.0,  0.0,  0.0, 200.0
...
```

**Key Points:**
- Columns 1-7: Position (mm) + Quaternion [w, x, y, z]
- Column 8 (optional): Speed in mm/s (used for continuity analysis)
- Single-column rows like "T0" separate trajectories
- Positions **must** be in millimeters (auto-converted to meters)
- Quaternions should be normalized to unit length

---

## Advanced Usage

### Custom Batch Processing

**Create a custom config file:**
```yaml
# my_config.yaml
robots:
  - path: "Assests/Robot APCC/IRB-1300-1400-URDF"
    model: "IRB 1300-7/1.4"
    reach_m: 1.4
    velocity_limits_rad_s: [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]

toolpaths:
  - "path/to/my/trajectories.csv"

knife_poses:
  pose_custom:
    translation: {x: -370.0, y: -910.0, z: 525.0}
    rotation: {w: 0.005, x: 0.713, y: -0.701, z: 0.001}

visualization:
  enabled: true
  plot_3d: {enabled: true, scale: 0.01}

analysis:
  enabled: true
  ik: {max_iterations: 2000, tolerance: 1e-4}

continuity:
  enabled: true
```

**Run with custom config:**
```bash
python scripts/batch_trajectory_processor.py -c my_config.yaml -o my_results/
```

### Debug and Interactive Visualization

For interactive trajectory exploration, see [Debug Tools](#debug-tools) section.

---

## Debug Tools

For advanced trajectory visualization and analysis, see: **[`scripts/debug/README.md`](scripts/debug/README.md)**

---

## Dependencies

See `requirements.txt` for complete list:

**Core Dependencies:**
- `pinocchio` (≥2.6.0) - Robotics kinematics and dynamics
- `numpy` (≥1.21.0) - Numerical computing
- `pandas` (≥1.3.0) - Data manipulation
- `matplotlib` (≥3.5.0) - Plotting and visualization
- `pyyaml` - Configuration file parsing
- `tqdm` - Progress bars

**Installation:**
```bash
pip install -r requirements.txt
```

**Platform Notes:**
- **macOS with Apple Silicon**: Ensure compatible `torch` and `pinocchio` versions
- **Linux/Windows**: Standard installation works
- **URDF Dependencies**: Pinocchio automatically handles URDF parsing

---

## Troubleshooting

### "URDF file not found"
- Ensure robot folder has `urdf/` subdirectory
- Check that URDF files have `.urdf` extension
- Verify path in `config_batch_processing.yaml`

### "IK failed for all poses"
- Check knife pose quaternion is normalized (norm = 1)
- Verify trajectories are within robot workspace
- Increase `max_iterations` and decrease `tolerance` in config

### "Continuity checks fail"
- Speed column (column 8) might be missing from CSV
- Check that joint angles were successfully computed (requires reachable IK)
- Verify velocity limits are realistic for the robot model

### Memory issues with large batches
- Process subsets of trajectories separately
- Use smaller CSV files
- Disable visualization to reduce memory usage

---

## Output Structure

All results are saved in `results/` directory:

```
results/
├── batch_processing_summary.yaml          # Overall batch statistics
├── csv_name_1/
│   ├── batch_summary.yaml                 # CSV-level summary
│   ├── Robot_Model_pose_1/
│   │   ├── Traj_1_[1-50]/
│   │   │   ├── experiment_results.yaml    # Analysis results (reachability, IK success, etc.)
│   │   │   ├── experiment.csv             # Joint angles for all poses
│   │   │   ├── Traj_1_visualization.png   # 3D trajectory plot
│   │   │   ├── pose_viz/
│   │   │   │   ├── Traj_1_comparison.png  # Original vs Transformed
│   │   │   │   ├── Traj_1_data_comparison.png
│   │   │   │   └── Traj_1_delta_analysis.png
│   │   │   └── continuity/
│   │   │       ├── continuity_analysis.yaml
│   │   │       └── continuity_analysis.png
│   │   └── Traj_2_[1-100]/
│   │       └── (same structure)
│   └── Robot_Model_pose_2/
│       └── (same structure)
└── csv_name_2/
    └── (same structure)
```

---

## Project Structure

### Root Level Files

**`batch_trajectory_processor.py`** - Main Orchestration Script
- Entry point for all batch processing
- Manages 3-level nested loops: CSV files → Robots → Knife poses → Trajectories
- Coordinates between analysis, continuity checking, and visualization modules
- Generates batch-level and CSV-level summary reports

**`config_batch_processing.yaml`** - Configuration File
- Defines robots to process (paths, velocity/acceleration limits)
- Specifies trajectory CSV locations
- Sets knife pose transformations
- Controls visualization, analysis, and continuity settings
- See [Configuration](#configuration) section for details

### Key Directories

#### `scripts/trajectory_processing/`
Core analysis algorithms:

**`analyze_irb1300_trajectory.py`** - Kinematic Analysis Engine
- Performs Inverse Kinematics (IK) on trajectory poses using Pinocchio
- Calculates joint angles for all reachable poses
- Computes manipulability index (Yoshikawa metric)
- Generates reachability statistics and analysis plots
- Exports joint configurations for continuity analysis

**`trajectory_continuity_analyzer.py`** - Continuity Verification
- Analyzes C⁰ continuity (position continuity)
- Analyzes C¹ continuity (velocity constraints with joint limits)
- Analyzes C² continuity (acceleration smoothness)
- Computes unified pose distance metrics
- Generates detailed continuity reports and graphs

**`trajectory_visualizer.py`** - Non-Interactive Visualization
- Creates 3D trajectory visualizations (point clouds, not lines)
- Generates data comparison graphs (original vs transformed)
- Creates delta analysis graphs (pose-to-pose changes)
- Exports all plots as high-resolution PNG files

#### `scripts/utils/`
Utility modules shared across the system:

**`batch_processor.py`** - Discovery and Configuration Loading
- Discovers robot URDF files in directory trees
- Loads knife pose configurations from YAML
- Generates output directory names

**`csv_handling.py`** - CSV Parsing and Validation
- Reads trajectory CSV files with proper validation
- Handles trajectory separators (single-column rows)
- Automatically converts positions from mm to meters for URDF compatibility
- Normalizes quaternions

**`handle_transforms.py`** - Coordinate Frame Transformations
- Transforms trajectories from plate frame (T_P_K) to base frame (T_B_K)
- Applies knife pose offset transformations
- Validates quaternion-based rotations

**`math_utils.py`** - Mathematical Utilities
- Quaternion to rotation matrix conversions
- Quaternion operations (conjugates, products)
- Common geometric calculations

**`graph_utils.py`** - Plotting and Visualization
- Generates all analysis plots (reachability, manipulability, etc.)
- Creates comparison graphs with multiple subplots
- Handles matplotlib backend and styling

**`results_handler.py`** - Results Management
- Formats and saves analysis results to YAML and CSV
- Generates experiment reports
- Aggregates batch-level statistics

**`trajectory_visualizer.py`** - Utility Visualization Functions
- Creates 3D plots with optional coordinate frames
- Generates comparison visualizations
- Exports figures to files

#### `Assests/Robot APCC/`
Robot and trajectory data:

**Robot URDF Directories:**
- `IRB-1300 1150 URDF/` - 1.15m reach robot configuration
- `IRB-1300 900 URDF/` - 0.9m reach robot configuration
- `IRB-1300-1400-URDF/` - 1.4m reach robot configuration

Each contains:
- `urdf/` - URDF files with end-effector definitions
- `config/` - Joint configuration YAML
- `meshes/` - STL visual/collision geometry
- `launch/` - ROS launch files

**`Toolpaths/` Directory:**
- `Successful/` - Production-ready trajectory CSV files
- `debug/` - Test trajectories for validation
- `converted/` - Converted trajectory formats
- `Toolpaths/converted/` - Older converted files

#### `scripts/debug/`
Advanced interactive tools (see [Debug Tools](#debug-tools)):

**`pose_3d_visualizer.py`** - Interactive Single-Trajectory Visualization
- Real-time 3D visualization with mouse controls
- Optional coordinate frame axes overlay
- Debug mode with pose indexing
- Side-by-side transformation comparison
- Keyboard controls (Ctrl+S to save)

**`pose_3d_batch.py`** - Batch Trajectory Processor
- Non-interactive batch processing of trajectory folders
- Extracts individual trajectories from CSVs
- Runs all analysis and visualization automatically
- Generates reports for each trajectory

**`continuity_analyzer.py`** - Standalone Continuity Analysis
- Detailed C¹ and C² continuity visualization
- Extracts speed from CSV 8th column
- Creates publication-quality continuity graphs

**`pose_analyzer.py`** - Delta Analysis Tool
- Analyzes pose-to-pose changes (deltas)
- Generates delta comparison graphs
- Useful for trajectory smoothness assessment

**Config Files:**
- `compare.yaml` - Configuration for comparison mode visualization
- `default.yaml` - Default visualization settings

---

## Running Individual Scripts

### `analyze_irb1300_trajectory.py` - Standalone Kinematic Analysis

Run IK solving and manipulability analysis on a single trajectory independently:

```bash
cd Robotics-APCC
python scripts/trajectory_processing/analyze_irb1300_trajectory.py \
  -o output_directory \
  --max-iterations 2000 \
  --tolerance 1e-4
```

**Parameters:**
- `-o, --output DIR` - Output directory for results (default: `output/`)
- `--max-iterations INT` - Maximum IK solver iterations (default: 1000)
- `--tolerance FLOAT` - IK convergence tolerance in meters (default: 1e-4)

**Note:** Modify the hardcoded paths in the script for URDF and CSV files (lines 57-58)

---

### `trajectory_visualizer.py` - Non-Interactive Batch Visualization

This script is designed to run within the batch processor pipeline. **For interactive visualization, use the debug tools instead:**

**Use [`pose_3d_visualizer.py`](scripts/debug/README.md) for:**
- Interactive 3D visualization with mouse controls
- Single or side-by-side trajectory comparison
- Manual transformation testing
- CSV export of transformed trajectories

See [`scripts/debug/README.md`](scripts/debug/README.md) for detailed usage instructions.

---

### `continuity_analyzer.py` - Trajectory Continuity Analysis

**Note:** This script requires pre-computed IK solutions (joint angles). It **cannot run standalone** as it depends on kinematic data from the main batch processor.

**To analyze continuity:**

1. **Run batch processing first** to compute IK solutions:
   ```bash
   python scripts/batch_trajectory_processor.py -c scripts/config_batch_processing.yaml
   ```

2. **Enable continuity analysis** in your config (`config_batch_processing.yaml`):
   ```yaml
   continuity:
     enabled: true
     generate_graphs: true
   ```

3. **Results are automatically generated** in `results/{csv_name}/{robot_model}_{pose}/continuity/`

---

## Getting Help

1. **Check Debug Tools Documentation:** [`scripts/debug/README.md`](scripts/debug/README.md)
2. **Review Configuration Example:** `scripts/config_batch_processing.yaml`
3. **Inspect Output YAML Files:** Each trajectory generates `experiment_results.yaml` with detailed analysis
4. **Check Console Output:** Batch processor prints progress and errors to stdout

---

**Last Updated:** 2025-01-18  
**System Version:** 1.0

