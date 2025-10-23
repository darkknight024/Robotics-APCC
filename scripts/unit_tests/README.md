# Unit Tests for Trajectory Visualizer

This directory contains comprehensive unit tests for the trajectory visualization and transformation functionality.

## Files

### `test_trajectories.csv`
Test trajectory data in CSV format with multiple trajectories separated by "T0" lines. Contains:
- 4 trajectories with 5 waypoints each
- Various positions and orientations for comprehensive testing
- Format: `x,y,z,qw,qx,qy,qz` (position in mm, quaternion w,x,y,z)

### `test_trajectory_transforms.py`
Comprehensive unit test suite that tests:

#### Round-trip Composition Tests
- Verifies that `T_B_K_check = T_B_P @ T_P_K ≈ T_B_K`
- Tests with various pose configurations (translations, rotations, combined)
- Ensures numerical accuracy within specified tolerances (1mm for translations, 0.01 for quaternions)

#### CSV Reading Tests
- Tests parsing of trajectory CSV files
- Verifies trajectory count and waypoint counts
- Validates quaternion normalization

#### Trajectory Filtering Tests
- Tests `--odd` and `--even` command line filtering
- Verifies correct trajectory selection by index

#### Transformation with Filtering Tests
- Tests that robot-base transformation works correctly with filtered trajectories
- Ensures transformation is applied only to selected trajectories, not all trajectories

#### Quaternion Operations Tests
- Tests quaternion normalization
- Tests rotation matrix conversions
- Tests pose-to-matrix round-trip conversions

#### Edge Case Tests
- Tests with empty trajectory lists
- Tests with zero-length quaternions
- Verifies graceful error handling

## Running the Tests

```bash
# From the project root directory
python scripts/unit_tests/test_trajectory_transforms.py

# Or from the unit_tests directory
cd scripts/unit_tests
python test_trajectory_transforms.py
```

## Test Data Format

The test CSV file follows the same format as production trajectory files:
- **Position:** x, y, z in millimeters
- **Orientation:** qw, qx, qy, qz (unit quaternion, w,x,y,z order)
- **Trajectory separation:** Single row with "T0" separates multiple trajectories

Example CSV row:
```csv
x,y,z,qw,qx,qy,qz
-50.0,100.0,25.0,1.0,0.0,0.0,0.0
```

## Mathematical Verification

The primary test verifies the coordinate transformation math:
1. **T_B_K** (knife relative to base) is known from calibration
2. **T_P_K** (plate relative to knife) comes from trajectory data
3. **T_B_P** (plate relative to base) is computed using the transformation function
4. **Round-trip:** `T_B_K_check = T_B_P @ T_P_K` should equal `T_B_K`

This verifies that the transformation mathematics is implemented correctly.
