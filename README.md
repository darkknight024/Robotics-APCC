# Robotics-APCC: Kinematic Validation and Knife Pose Estimation

This repository provides tools for kinematic validation of robot models and trajectory analysis for optimal knife poses in manufacturing applications. It supports ABB IRB-1300 and KUKA robot models with comprehensive trajectory planning and analysis capabilities.

## Overview

The Robotics-APCC project focuses on:
- **Kinematic validation** of robot trajectories using forward and inverse kinematics
- **3D visualization** of robot models, trajectories, and coordinate frames
- **Kinematic analysis** including manipulability, reachability, and singularity detection
- **Knife pose optimization** for siping processes

## Repository Structure

This is the initial structure. This is expected to change after we bring in isaac sim stuff, and based on the recommendation from Sahil, Jared and anyone else who intend to use this repo.

```
├── Assests/
│   ├── Robot APCC/                 # Robot model definitions
│   │   ├── IRB-1300 1150 URDF/    # ABB IRB-1300 1150mm reach model
│   │   └── IRB-1300 900 URDF/      # ABB IRB-1300 900mm reach model
│   ├── Toolpaths/                  # Trajectory data files
│   │   ├── converted/             # Processed trajectories
│   │   └── Successful/            # Validated trajectory examples
│   └── graphing_utility.py        # Data collection and plotting utilities
├── scripts/
│   ├── unit_tests/                # Comprehensive unit tests
│   │   ├── test_trajectories.csv  # Test trajectory data
│   │   ├── test_math_utils.py     # Math utility function tests
│   │   ├── test_trajectory_transforms.py  # Transformation tests
│   │   └── run_all_tests.py       # Test runner script
│   ├── analyze_irb1300_trajectory.py  # Main kinematic analysis tool
│   └── utils/
│       ├── math_utils.py           # Centralized math utilities
│       ├── trajectory_transform.py        # Coordinate frame transformations
│       ├── trajectory_visualizer.py       # Advanced 3D trajectory visualization
│       └── visualize_frames.py            # Coordinate frame and URDF visualization
└── requirements.txt               # Python dependencies
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Robotics-APCC
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Required packages:**
   - `numpy` - Numerical computations
   - `pandas` - Data manipulation
   - `matplotlib` - Plotting and visualization
   - `pinocchio` - Robot kinematics and dynamics

## Scripts Overview

### 1. `visualize_frames.py`
**Purpose:** Visualizes coordinate frames, robot models, and knife poses in 3D space.

**Key Features:**
- Displays origin and transformed coordinate frames (RGB arrows)
- Loads and visualizes URDF robot models
- Shows knife tool position and orientation
- Interactive 3D matplotlib visualization

**Usage:**
```bash
python scripts/utils/visualize_frames.py [options]

# Example: Visualize knife pose with robot model
python scripts/utils/visualize_frames.py \
  --urdf "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf" 

# Command Line Arguments:
--urdf PATH              # Path to URDF file for robot visualization
--base-link NAME         # Base/root link name (optional)
--tx, --ty, --tz FLOAT   # Tool/frame translation (mm) (optional) - Default set in code from Jared's email
--qw, --qx, --qy, --qz FLOAT  # Frame quaternion (w,x,y,z) (optional) - Default set in code from Jared's email
--labels                 # Show joint name labels (optional)
```

### 2. `trajectory_visualizer.py`
**Purpose:** Advanced 3D visualization of trajectory data with multiple coordinate frame views for comprehensive trajectory analysis.

**Key Features:**
- **Three View Modes:** Visualize trajectories in different reference frames
  - **T_P_K (Plate Frame):** Raw CSV trajectories showing knife poses in plate frame
  - **T_K_P (Knife Frame):** Inverted trajectories showing plate poses in knife frame
  - **T_B_P (Base Frame):** Transformed trajectories showing plate poses in robot base frame with knife frame visualization
- Reads trajectory CSV files (x, y, z, qw, qx, qy, qz format)
- Interactive 3D visualization with coordinate frames at waypoints
- Customizable waypoint step for controlling visualization density
- Color-coded legend showing both coordinate axes (Red=X, Green=Y, Blue=Z) and individual trajectories
- Frame labels: P (Plate), K (Knife), B (Base) displayed directly on coordinate frames
- Supports multiple trajectories in single file
- Filtering options for trajectory selection

**Usage:**
```bash
python scripts/utils/trajectory_visualizer.py trajectory.csv [options]

# Example: Show all three views with controlled waypoint density
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view all --waypoint-step 10 --num-trajectories 3

# Example: Show only T_P_K view (knife in plate frame)
# Raw CSV visualization - shows knife motion relative to plate
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view pk --waypoint-step 5

# Example: Show T_K_P view (plate in knife frame)
# Shows required plate motion with static knife - for robot control
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view kp --waypoint-step 8

# Example: Show T_B_P view (robot base frame) with knife visualization
# Shows actual robot end-effector poses needed (solve IK for these)
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view bp --waypoint-step 10

# Example: Legacy robot-base mode (equivalent to --view bp)
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --robot-base --waypoint-step 15

# Example: Show only even-numbered trajectories with custom waypoint step
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --even --view all --waypoint-step 20

# Command Line Arguments:
csv_file                 # Path to CSV file with trajectory data
--view VIEW              # View mode: 'pk' (T_P_K), 'kp' (T_K_P), 'bp' (T_B_P), or 'all' (default: all)
--robot-base             # Apply robot-base transform (same as --view bp)
--num-trajectories N     # Number of trajectories to visualize (default: all)
--odd                    # Show only odd-numbered trajectories (0-based indexing)
--even                   # Show only even-numbered trajectories (0-based indexing)
--waypoint-step N        # Number of waypoints between coordinate frames (default: 15)

Legend:
    Red line = X axis, Green line = Y axis, Blue line = Z axis
    Colored lines = Individual trajectories (Trajectory 1, Trajectory 2, etc.)
    Frame names: P (Plate), K (Knife), B (Base) - labeled directly on each coordinate frame
```

### 3. `trajectory_transform.py`
**Purpose:** Transforms trajectory data between coordinate frames and exports to new CSV files.

**Key Features:**
- Reads trajectory CSV files with optional speed data
- Applies coordinate frame transformations
- Outputs transformed trajectories in mm or meters
- Preserves or separates multiple trajectories

**Usage:**
```bash
python scripts/utils/trajectory_transform.py input.csv output.csv [options]

# Example: Transform to meters with trajectory separation
python scripts/utils/trajectory_transform.py \
  input.csv output_meters.csv --meters --separated

# Command Line Arguments:
input.csv                # Input trajectory CSV file
output.csv               # Output transformed CSV file
--meters                 # Output positions in meters (default: mm)
--separated              # Preserve trajectory separators in output
```

### 4. `analyze_irb1300_trajectory.py`
**Purpose:** Comprehensive kinematic analysis of IRB-1300 robot trajectories using Pinocchio.

**Key Features:**
- Forward and inverse kinematics validation
- Manipulability index calculation (Yoshikawa measure)
- Singularity detection and proximity analysis
- Joint limit checking and trajectory feasibility
- Reachability analysis and workspace validation
- Statistical analysis and visualization

**Usage:**
```bash
python scripts/analyze_irb1300_trajectory.py [options]

# Example: Analyze trajectory with custom parameters
python scripts/analyze_irb1300_trajectory.py \
  --urdf "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf" \
  --csv "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --output-dir "results" --samples 100

# Command Line Arguments:
--urdf PATH              # Path to robot URDF file (default: IRB-1300 1150)
--csv PATH               # Path to trajectory CSV file (default: plq_curve.csv)
--output-dir DIR         # Output directory for results (default: output)
--samples N              # Number of trajectory samples to analyze (default: all)
--ik-tolerance FLOAT     # Inverse kinematics tolerance (default: 1e-4)
--ik-max-iter N          # Maximum IK iterations (default: 1000)
```

### 5. `toolpath_visualizer.py`
**Purpose:** Alternative trajectory visualization tool located in the Assests directory.

**Key Features:**
- 3D trajectory visualization
- CSV trajectory file parsing
- Basic plotting functionality

**Usage:**
```bash
python Assests/toolpath_visualizer.py trajectory.csv

# Command Line Arguments:
csv_path                 # Path to trajectory CSV file
```

### 6. `graphing_utility.py`
**Purpose:** Data collection and advanced plotting utilities for trajectory playback analysis.

**Key Features:**
- End-effector speed analysis and tracking
- Joint position monitoring and limit checking
- Waypoint transition detection
- Statistical analysis of trajectory execution
- Multiple plot types for performance analysis

**Usage:**
```python
# Import and use in trajectory playback scripts
from Assests.graphing_utility import SpeedDataCollector

# Initialize collector
collector = SpeedDataCollector(dt=0.01, target_speeds_mm_s=target_speeds)

# Record data during playback
collector.record_step(ee_position, current_waypoint)

# Generate plots and save data
collector.save_data("trajectory_analysis.csv")
collector.plot_all_metrics()
```

## Robot Models

### ABB IRB-1300 1150mm Reach
- **URDF Location:** `Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf`
- **Joint Limits:** 6 joints with standard ABB IRB-1300 ranges
- **End Effector:** Configured for knife/tool mounting

### ABB IRB-1300 900mm Reach
- **URDF Location:** `Assests/Robot APCC/IRB-1300 900 URDF/urdf/IRB-1300 900 URDF_ee.urdf`
- **Joint Limits:** 6 joints with ABB IRB-1300 900mm specifications
- **End Effector:** Configured for manufacturing applications

## Trajectory Data Format

Trajectory CSV files should contain:
- **Position:** x, y, z (millimeters)
- **Orientation:** qw, qx, qy, qz (unit quaternion, w,x,y,z order)
- **Optional:** speed (mm/s) in 8th column
- **Separators:** Single row with "T0" to separate multiple trajectories

Example CSV format:
```csv
x,y,z,qw,qx,qy,qz,speed_mm_s
-100.0,200.0,50.0,1.0,0.0,0.0,0.0,100.0
-90.0,190.0,55.0,0.995,0.0,0.0,0.105,95.0
T0
-80.0,180.0,60.0,0.990,0.0,0.0,0.141,90.0
```

## Coordinate Frame Transformations

The repository uses the following coordinate frame conventions:

1. **T_P_K (Plate Frame):** Knife pose expressed in plate coordinates (raw CSV data)
2. **T_K_P (Knife Frame):** Plate pose expressed in knife coordinates (inverse of T_P_K)
3. **T_B_P (Base Frame):** Plate pose expressed in robot base coordinates (for robot control)

**Transformation Chain:**
- **T_K_P = T_P_K^(-1)** (matrix inversion)
- **T_B_P = T_B_K × T_K_P** (coordinate transformation)

Key transformation parameters (from robot-base calibration):
- **Knife in Base (T_B_K):** Translation `[-367.773, -915.815, 520.4]` mm, Quaternion `[0.00515984, 0.712632, -0.701518, 0.000396522]` (w,x,y,z)

**Understanding the Views:**
- **T_P_K:** Shows how the knife would move relative to the plate if the knife were mobile
- **T_K_P:** Shows how the plate should move relative to the static knife (robot motion)
- **T_B_P:** Shows the actual robot end-effector poses needed to achieve the desired knife motion

## Unit Testing

The repository includes comprehensive unit tests for all mathematical transformations and trajectory processing functionality.

### Running Tests

```bash
# Run all unit tests
python scripts/unit_tests/run_all_tests.py

# Run individual test files
python scripts/unit_tests/test_trajectory_transforms.py
python scripts/unit_tests/test_math_utils.py

# Run tests with verbose output
python scripts/unit_tests/test_trajectory_transforms.py
```

### Test Coverage

The unit tests verify:
- **Round-trip composition**: `T_B_K_check = T_B_P @ T_P_K ≈ T_B_K`
- **CSV trajectory parsing**: Reading and validation of trajectory files
- **Trajectory filtering**: `--odd` and `--even` command line options
- **Coordinate transformations**: Robot-base frame transformations
- **Quaternion operations**: Multiplication, rotation matrices, normalization
- **Edge cases**: Empty data, invalid quaternions, error handling

## Usage Examples

### Basic Trajectory Visualization
```bash
# Visualize a trajectory in all three reference frames
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view all --waypoint-step 10 --num-trajectories 2

# Single view visualization with controlled density
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --view bp --waypoint-step 5
```

### Kinematic Validation
```bash
# Analyze trajectory feasibility for IRB-1300
python scripts/analyze_irb1300_trajectory.py \
  --csv "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --samples 50
```

### Frame Transformation and Visualization
```bash
# Transform trajectory and visualize with robot model
python scripts/utils/trajectory_transform.py input.csv transformed.csv
python scripts/utils/trajectory_visualizer.py transformed.csv --view bp --waypoint-step 10
python scripts/utils/visualize_frames.py --urdf "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf"
```

## Output Files

Analysis scripts generate:
- **CSV files:** Trajectory data, joint angles, analysis results
- **PNG files:** Manipulability plots, reachability maps, singularity analysis
- **Statistical reports:** Feasibility analysis and performance metrics

## Contributing

1. Follow the existing code structure and naming conventions
2. Add proper documentation and command-line argument descriptions
3. Include example usage in docstrings
4. Test with provided trajectory files before submitting

## License

This project is part of Nike's robotics research and development efforts for advanced manufacturing processes.
