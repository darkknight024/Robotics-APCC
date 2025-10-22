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
│   ├── tests/
│   │   ├── analyze_irb1300_trajectory.py  # Main kinematic analysis tool
│   │   └── visualize_trajectory.py        # Simple trajectory plotting
│   └── utils/
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
**Purpose:** 3D visualization of trajectory data with coordinate frame transformations. This is Work In Progress. Koushik is running some experiments to understand the transforms and framing conventions (grasshopper -> csv -> robot), and I probably need help from Sahil and Jared to understand the rotations in CSV.  

**Key Features:**
- Reads trajectory CSV files (x, y, z, qw, qx, qy, qz, speed)
- Transforms trajectories from tool frame to robot base frame
- Interactive 3D visualization with coordinate frames
- Supports multiple trajectories in single file

**Usage:**
```bash
python scripts/utils/trajectory_visualizer.py trajectory.csv [options]

# Example: Visualize trajectory with robot-base transform
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --robot-base --num-trajectories 20

# Command Line Arguments:
csv_file                 # Path to CSV file with trajectory data
--robot-base             # Apply tool->robot-base transformation and visualize the end effector poses for those trajectories
--num-trajectories N     # Number of trajectories to visualize (default: all)
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
python scripts/tests/analyze_irb1300_trajectory.py [options]

# Example: Analyze trajectory with custom parameters
python scripts/tests/analyze_irb1300_trajectory.py \
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

### 5. `visualize_trajectory.py`
**Purpose:** Simple 2D plotting of trajectory position and orientation data.

**Key Features:**
- Position plot (x, y, z vs time/index)
- Quaternion orientation plot (qw, qx, qy, qz vs time/index)
- Basic CSV trajectory file parsing

**Usage:**
```bash
python scripts/tests/visualize_trajectory.py trajectory.csv

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

1. **Tool Frame (T_P_K):** Knife/plate coordinate system
2. **Robot Base Frame (T_B_K):** Robot base coordinate system
3. **Transformation:** T_B_P = T_B_K × T_P_K (plate pose in robot base frame)

Key transformation parameters (from robot-base calibration):
- **Translation:** `[-367.773, -915.815, 520.4]` mm
- **Quaternion:** `[0.00515984, 0.712632, -0.701518, 0.000396522]` (w,x,y,z)

## Usage Examples

### Basic Trajectory Visualization
```bash
# Visualize a trajectory in 3D
python scripts/utils/trajectory_visualizer.py \
  "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv"
```

### Kinematic Validation
```bash
# Analyze trajectory feasibility for IRB-1300
python scripts/tests/analyze_irb1300_trajectory.py \
  --csv "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv" \
  --samples 50
```

### Frame Transformation and Visualization
```bash
# Transform trajectory and visualize with robot model
python scripts/utils/trajectory_transform.py input.csv transformed.csv
python scripts/utils/trajectory_visualizer.py transformed.csv --robot-base
python scripts/utils/visualize_frames.py --urdf "path/to/robot.urdf"
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
