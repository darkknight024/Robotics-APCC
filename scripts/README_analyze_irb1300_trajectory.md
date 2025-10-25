# ABB IRB 1300 Trajectory Analysis Script

This script analyzes the kinematic reachability and singularity proximity of an ABB IRB 1300 6-axis robot along a specified trajectory using Pinocchio.

## Features

The script provides comprehensive kinematic analysis including:

1. **Kinematic Feasibility Analysis**: Checks if target poses are reachable by the robot using inverse kinematics
2. **Singularity Analysis**: Computes manipulability indices, minimum singular values, and condition numbers
3. **Joint Space Continuity**: Analyzes joint angle changes between consecutive waypoints for smooth motion
4. **Orientation Constraints**: Evaluates trajectory continuity with respect to orientation requirements
5. **Visualization**: Generates comprehensive plots showing:
   - 3D trajectory comparison (target vs actual)
   - Reachability analysis
   - Manipulability and singularity measures
   - Joint angles with continuity markers

## Requirements

- pinocchio
- numpy
- pandas
- matplotlib
- tqdm

## Installation

Install the required dependencies:

```bash
pip install pinocchio numpy pandas matplotlib tqdm
```

## Usage

### Basic Usage

```bash
python analyze_irb1300_trajectory.py
```

This will analyze the default trajectory using the default robot model and configuration.

### Command Line Options

```bash
python analyze_irb1300_trajectory.py [options]

Options:
  -i, --input PATH        Path to CSV file containing trajectory (x, y, z, qw, qx, qy, qz)
  -u, --urdf PATH         Path to URDF file (default: configured URDF_PATH)
  -o, --output DIR        Output directory (default: configured OUTPUT_DIR)
  -b, --base              Assume input CSV is already in robot base frame
  --max-iterations INT    Max IK iterations (default: 1000)
  --tolerance FLOAT       IK tolerance (default: 1e-4)
  --visualize             Generate visualization plots (default: True)
```

### Examples

1. **Analyze a specific trajectory:**
```bash
python analyze_irb1300_trajectory.py -i path/to/trajectory.csv
```

2. **Use a different robot model:**
```bash
python analyze_irb1300_trajectory.py -u path/to/robot.urdf
```

3. **Save results to custom directory:**
```bash
python analyze_irb1300_trajectory.py -o my_analysis_results
```

4. **Skip visualization plots:**
```bash
python analyze_irb1300_trajectory.py --visualize False
```

5. **Use more strict IK tolerance:**
```bash
python analyze_irb1300_trajectory.py --tolerance 1e-5
```

## Input Data Format

The trajectory CSV file should contain columns in this order:
- `x, y, z`: Position in meters (or millimeters if using the trajectory transformation utilities)
- `qw, qx, qy, qz`: Quaternion orientation (w, x, y, z format)

### Coordinate Frame Options

The script supports two input formats:

1. **T_P_K format** (default): Knife poses in plate coordinate frame
   - Use without `-b` flag
   - Script will automatically transform to robot base frame

2. **T_B_P format**: Plate poses in robot base frame
   - Use with `-b` flag
   - No transformation needed

## Output

The script generates:

1. **Results CSV file** (`trajectory_analysis_results.csv`):
   - Reachability status for each waypoint
   - Joint angles for reachable poses
   - Singularity measures (manipulability, singular values, condition number)
   - Position errors
   - Joint continuity and orientation constraint analysis

2. **Visualization plots**:
   - 3D trajectory comparison
   - Reachability analysis
   - Manipulability indices
   - Singularity measures
   - Joint angles with continuity markers

3. **Console output** with comprehensive statistics:
   - Reachability percentages
   - Position accuracy metrics
   - Singularity analysis
   - Joint continuity issues
   - Orientation constraint violations

## Configuration

The script uses these default paths (relative to project root):

- **URDF**: `Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf`
- **CSV**: `Assests/Robot APCC/Toolpaths/converted/plq_curve.csv`
- **Output**: `output/`

## Algorithm Details

### Inverse Kinematics
- Uses damped least squares method
- Configurable maximum iterations and tolerance
- Tries multiple initial guesses (neutral, previous solution, random configurations)
- Respects joint limits

### Singularity Analysis
- **Manipulability Index**: Yoshikawa measure (√det(JJᵀ))
- **Minimum Singular Value**: Smallest singular value of Jacobian
- **Condition Number**: κ(J) = σ_max/σ_min

### Joint Continuity
- Analyzes joint angle changes between consecutive waypoints
- Identifies large jumps that may indicate motion planning issues
- Default threshold: 30 degrees

### Orientation Constraints
- Evaluates trajectory smoothness in joint space
- Identifies potential orientation discontinuities
- Default threshold: 5 degrees

## Integration with Existing Utilities

The script leverages existing utility modules:

- `math_utils.py`: Quaternion and rotation matrix operations
- `csv_handling.py`: Trajectory CSV parsing
- `handle_transforms.py`: Coordinate frame transformations

## Troubleshooting

1. **Import errors**: Ensure all utility modules are in the `utils/` subdirectory
2. **URDF loading errors**: Verify URDF file path and format
3. **CSV parsing errors**: Check CSV format and column order
4. **IK convergence issues**: Try adjusting `--max-iterations` or `--tolerance`
5. **Memory issues**: For very large trajectories, consider processing in chunks

## Examples in the Repository

See the existing toolpath files in:
- `Assests/Robot APCC/Toolpaths/converted/`
- `Assests/Robot APCC/Toolpaths/Successful/`

These contain example trajectories that can be analyzed with this script.
