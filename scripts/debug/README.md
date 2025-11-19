# Debug Tools - Interactive Trajectory Visualization and Analysis

This directory contains advanced interactive visualization tools for detailed trajectory analysis, transformation verification, and continuity checking. These tools are complementary to the batch processor and provide real-time feedback for trajectory debugging.

## Table of Contents

- [Quick Start](#quick-start)
- [Tool Overview](#tool-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Output](#output)

---

## Quick Start

### Tool 1: Interactive Single Trajectory Viewer (`pose_3d_visualizer.py`)

**Basic usage:**
```bash
cd Robotics-APCC-copy
python scripts/debug/pose_3d_visualizer.py "<path/to/csv>" --config scripts/debug/compare.yaml
```

**Features:**
- Interactive 3D visualization (rotate, zoom, pan with mouse)
- Shows original trajectory points
- Compares original vs. transformed poses side-by-side
- Optional coordinate frame axes overlay
- CSV row indices in debug mode
- Save figures with Ctrl+S

**Mouse Controls:**
- **Drag**: Rotate view
- **Scroll**: Zoom in/out
- **Right-click + Drag**: Pan
- **Ctrl+S**: Save current view as PNG

### Tool 2: Batch Trajectory Processor (`pose_3d_batch.py`)

**Basic usage:**
```bash
cd Robotics-APCC-copy
python scripts/debug/pose_3d_batch.py "path/to/csv_folder" --config scripts/debug/compare.yaml
```

**Features:**
- Non-interactive batch processing of CSV files
- Automatically extracts all trajectories from each CSV
- Generates all visualizations and analysis automatically
- Saves results in organized folder structure
- No user interaction needed (great for overnight processing)

---

## Tool Overview

### `pose_3d_visualizer.py` - Interactive Single-Trajectory Viewer

**Purpose:** Real-time exploration of a single trajectory with full transformation visualization

**What it does:**
1. Reads a single CSV file
2. Extracts all trajectories (separated by "T0" markers)
3. Displays 3D scatter plot of pose positions
4. (Optional) Applies knife pose transformation
5. (Optional) Shows original and transformed side-by-side
6. (Optional) Generates delta analysis and continuity graphs
7. Allows interactive rotation/zoom and Ctrl+S saving

**Input:** Single CSV file or directory containing CSVs

**Output:** 
- PNG visualizations (saved on Ctrl+S or auto-save enabled)
- Transformed CSV (if enabled)
- Data comparison graphs (if enabled)
- Delta analysis graphs (if enabled)
- Continuity analysis graphs (if enabled)

**Best for:**
- Debugging individual trajectories
- Verifying transformation correctness
- Quick visual inspection
- Development and troubleshooting

---

### `pose_3d_batch.py` - Batch Trajectory Processor

**Purpose:** Automated batch processing of trajectory folders without user interaction

**What it does:**
1. Finds all CSV files in input folder
2. For each CSV file:
   - Extracts all trajectories (separated by "T0" markers)
   - Creates output folder: `input_folder/csv_name/`
   - For each trajectory:
     - Creates folder: `input_folder/csv_name/Traj_N_[1-M]/`
     - Runs transformation (if enabled)
     - Generates 3D visualization comparison
     - Creates data comparison graphs (if enabled)
     - Creates delta analysis graphs (if enabled)
     - Saves all outputs

**Input:** Directory containing CSV files

**Output:**
```
input_folder/
├── csv_file_1/
│   ├── Traj_1_[1-50]/
│   │   ├── Traj_1_[1-50]_comparison.png
│   │   ├── Traj_1_[1-50]_data_comparison.png
│   │   ├── Traj_1_[1-50]_delta_analysis.png
│   │   └── Traj_1_[1-50]_transformed.csv
│   ├── Traj_2_[1-100]/
│   │   └── (same structure)
│   └── ...
└── csv_file_2/
    └── (same structure)
```

**Best for:**
- Processing multiple trajectories overnight
- Batch analysis of trajectory folders
- Automated report generation
- No interactive input required

---

### `continuity_analyzer.py` - Standalone Continuity Analysis

**Purpose:** Detailed C¹ and C² continuity verification with speed analysis

**Features:**
- C¹ continuity: velocity limit enforcement
- C² continuity: acceleration smoothness (not implemented yet - waiting for acceleration limits and other dynamic properties of the robot)
- Speed extracted from CSV 8th column
- Detailed continuity reports
- Publication-quality graphs

**Usage:**
```bash
python scripts/debug/continuity_analyzer.py path/to/trajectory.csv
```

---

### `pose_analyzer.py` - Delta Analysis Tool

**Purpose:** Analyze pose-to-pose changes (deltas) in trajectories

**Features:**
- Computes position deltas (ΔX, ΔY, ΔZ) between consecutive poses
- Computes rotation deltas (quaternion changes)
- Generates comparison graphs
- Shows smoothness patterns

**Usage:**
```bash
python scripts/debug/pose_analyzer.py path/to/trajectory.csv
```

---

## Configuration

### Configuration File: `compare.yaml`

The configuration file controls all aspects of visualization and transformation. It's YAML-based and human-readable.

#### Main Configuration Sections

**1. Visualization Settings**
```yaml
visualization:
  scale: 0.001              # Coordinate frame axis size (0 = disabled, typical: 0.001-0.05)
  point_size: 20            # Marker point size in plot
  every: 1                  # Show axes every N poses (1 = every pose, 5 = every 5th)
  mode: "single"            # "single" or "comparison"
  show: true                # Display interactive plot
  debug: true               # Show CSV row indices on points
```

- `scale`: Larger values = bigger coordinate frames at each pose
- `every`: Use larger values for dense trajectories to avoid visual clutter
- `debug`: Useful for identifying specific poses in the trajectory

**2. Transformation Settings**
```yaml
transformation:
  enabled: true
  knife_pose:
    translation:
      x: -367.773           # mm (tool offset from base)
      y: -915.815           # mm
      z: 520.4              # mm
    quaternion:
      w: 0.00515984         # Rotation (normalized quaternion)
      x: 0.712632
      y: -0.701518
      z: 0.000396522
  save_transformed_csv: true  # Export transformed poses to CSV
```

**3. Comparison Settings**
```yaml
comparison:
  enabled: true             # Side-by-side visualization
  show_labels: true         # Show subplot labels
  create_bar_graph: true    # Generate data comparison plots
```

**4. Pose Analyzer Settings**
```yaml
pose_analyzer:
  enabled: true
  save_delta_graphs: true   # Generate delta analysis plots
```

**5. Continuity Analyzer Settings**
```yaml
continuity_analyzer:
  enabled: true
  save_c1_graphs: true      # Velocity continuity graphs
  save_c2_graphs: true      # Acceleration continuity graphs
```

**6. Output Settings**
```yaml
output:
  path: "/path/to/output"   # Where to save results (null = CSV directory)
  auto_save: true           # Automatically save plots
  dpi: 300                  # Figure quality
  format: "png"             # Output format
```

### Example Configuration Scenarios

**Scenario 1: Quick Visual Check (Minimal)**
```yaml
visualization:
  scale: 0.0                # No axes
  point_size: 20
  debug: false
transformation:
  enabled: false            # No transformation
comparison:
  enabled: false            # Single view only
pose_analyzer:
  enabled: false
continuity_analyzer:
  enabled: false
```

**Scenario 2: Full Analysis (Complete)**
```yaml
visualization:
  scale: 0.01               # Show frame axes
  every: 1                  # Every pose
  debug: true               # Show indices
transformation:
  enabled: true
  save_transformed_csv: true
comparison:
  enabled: true
  create_bar_graph: true
pose_analyzer:
  enabled: true
  save_delta_graphs: true
continuity_analyzer:
  enabled: true
  save_c1_graphs: true
  save_c2_graphs: true
```

**Scenario 3: Transformation Verification**
```yaml
visualization:
  scale: 0.01               # Show frames to verify rotation
  debug: true
transformation:
  enabled: true
comparison:
  enabled: true             # Side-by-side original vs transformed
  create_bar_graph: true    # See numeric differences
pose_analyzer:
  enabled: false
continuity_analyzer:
  enabled: false
```

---

## Usage Examples

### Example 1: Quick Visual Inspection

Check if a trajectory looks correct:

```bash
python scripts/debug/pose_3d_visualizer.py Assests/Robot\ APCC/Toolpaths/debug/testings/test_trajectory.csv
```

This uses `default_config.yaml` with minimal visualization. You can:
- Rotate/zoom to inspect trajectory shape
- Press Ctrl+S to save
- Close the window to exit

### Example 2: Verify Transformation

Ensure your knife pose transformation is correct:

```bash
# Using compare.yaml (which has transformation.enabled: true)
python scripts/debug/pose_3d_visualizer.py \
  Assests/Robot\ APCC/Toolpaths/debug/testings/test_trajectory.csv \
  --config scripts/debug/compare.yaml
```

This shows:
- **Left plot**: Original trajectory (in plate frame)
- **Right plot**: Transformed trajectory (in robot base frame)

If transformation looks wrong:
- Check knife pose quaternion is normalized (magnitude = 1)
- Verify translation values (should be in mm)
- Compare with reference transformation data

### Example 3: Batch Process Entire Folder

Process all trajectories in a folder overnight:

```bash
python scripts/debug/pose_3d_batch.py \
  Assests/Robot\ APCC/Toolpaths/debug/testings/ \
  --config scripts/debug/compare.yaml
```

This creates:
```
Assests/Robot APCC/Toolpaths/debug/testings/
├── test_trajectory_1/
│   ├── Traj_1_[1-50]/
│   │   ├── Traj_1_[1-50]_comparison.png
│   │   ├── Traj_1_[1-50]_data_comparison.png
│   │   └── Traj_1_[1-50]_delta_analysis.png
│   └── Traj_2_[1-86]/
│       └── ...
└── test_trajectory_2/
    └── ...
```

### Example 4: Debug Mode with Row Indices

Identify specific problematic poses:

```bash
# Enable debug mode to see CSV row numbers
python scripts/debug/pose_3d_visualizer.py trajectory.csv --config compare.yaml
# Activate debug: true in compare.yaml
# Each point will show its CSV row number (1-based)
```

This helps when you need to:
- Locate a specific pose in the CSV file
- Debug IK convergence issues
- Correlate visualization issues with CSV data

### Example 5: Custom Knife Pose Testing

Test different knife pose configurations:

```bash
# Create custom_config.yaml with your new knife pose:
# transformation:
#   knife_pose:
#     translation: {x: -370.0, y: -910.0, z: 525.0}
#     rotation: {w: 0.005, x: 0.713, y: -0.701, z: 0.001}

python scripts/debug/pose_3d_visualizer.py trajectory.csv --config custom_config.yaml
```

Compare results visually and numerically in the data comparison graphs.

---

## Output

### Generated Files

**Visualization Images:**
- `Traj_N_[1-M]_visualization.png` - Original trajectory 3D plot
- `Traj_N_[1-M]_comparison.png` - Original vs. Transformed side-by-side (20" wide)
- `Traj_N_[1-M]_data_comparison.png` - Translation (X,Y,Z) and Rotation (QW,QX,QY,QZ) graphs
- `Traj_N_[1-M]_delta_analysis.png` - Pose-to-pose changes

**Data Files:**
- `Traj_N_[1-M]_transformed.csv` - Transformed trajectory in robot base frame
- `Traj_N_[1-M]_comparison.yaml` - Summary of transformation results

**Quality Settings:**
- Default: 150 DPI (faster, smaller files)
- High quality: 300 DPI (set in config)
- All saved as PNG for publication quality

### Reading the Graphs

**Data Comparison Graphs (4 subplots):**
1. **Top-left**: Original Translation (X, Y, Z in meters)
2. **Bottom-left**: Original Rotation (QW, QX, QY, QZ quaternion components)
3. **Top-right**: Transformed Translation
4. **Bottom-right**: Transformed Rotation

Use these to verify:
- Transformation correctness (should see significant changes)
- Trajectory smoothness (lines should be relatively smooth)
- Reachability (QW, QX, QY, QZ should stay within [-1, 1])

**Delta Analysis Graphs (4 subplots):**
1. **Top-left**: Original Position Delta (ΔX, ΔY, ΔZ)
2. **Bottom-left**: Original Rotation Delta
3. **Top-right**: Transformed Position Delta
4. **Bottom-right**: Transformed Rotation Delta

Use these to identify:
- Discontinuities (sharp jumps)
- Singularities (extreme delta values)
- Smooth vs. jerky trajectories

---

## Troubleshooting

### "Module not found" errors
```python
ModuleNotFoundError: No module named 'csv_handling'
```
**Solution:** The scripts automatically add `utils/` to the Python path. If this fails:
```bash
export PYTHONPATH=/path/to/Robotics-APCC-copy/scripts/utils:$PYTHONPATH
python scripts/debug/pose_3d_visualizer.py ...
```

### Transformation looks wrong
1. Check knife pose quaternion is **normalized**: √(w² + x² + y² + z²) = 1
2. Check translation is in **millimeters** (not meters)
3. Verify rotation representation (should be [w, x, y, z] format)
4. Compare with known good transformations

### No output files generated
- Check `output.path` is set correctly in config (or leave null for CSV directory)
- Ensure write permissions to output directory
- Check console output for error messages

### 3D plot is cluttered
- Reduce `point_size` to make points smaller
- Increase `every` to show axes less frequently (e.g., `every: 5`)
- Disable axes with `scale: 0`

### Memory issues with large trajectories
- Process trajectories one at a time with `pose_3d_visualizer.py`
- Reduce `point_size` and `dpi`
- Disable comparison mode and extra graphs
- Close plots between runs

---

## Integration with Batch Processor

The debug tools are **complementary** to `batch_trajectory_processor.py`:

- **Batch Processor** (`batch_trajectory_processor.py`): Production batch analysis across many robots/trajectories
- **Debug Tools** (`pose_3d_visualizer.py`, `pose_3d_batch.py`): Development/debugging for individual trajectories

**Workflow:**
1. Use `pose_3d_visualizer.py` to verify a single trajectory
2. Use `pose_3d_batch.py` to process a folder of similar trajectories
3. Use `batch_trajectory_processor.py` for final production runs across all robots

---

## Advanced Tips

### Extracting specific trajectory from CSV

```python
# Manual extraction if needed:
python
>>> from pose_3d_visualizer import parse_csv_trajectories
>>> trajs, indices = parse_csv_trajectories('trajectory.csv')
>>> print(f"Found {len(trajs)} trajectories")
>>> trajectory_1 = trajs[0]  # First trajectory
>>> print(f"Trajectory 1 has {len(trajectory_1)} poses")
```

### Batch processing with custom parameters

Create a loop to test multiple configurations:

```bash
for config in default.yaml compare.yaml custom.yaml; do
  python scripts/debug/pose_3d_batch.py \
    Assests/Robot\ APCC/Toolpaths/debug/testings/ \
    --config scripts/debug/$config
done
```

### Comparing two knife poses

```bash
# Test knife pose 1
python scripts/debug/pose_3d_visualizer.py trajectory.csv --config knife_pose_1.yaml

# Test knife pose 2
python scripts/debug/pose_3d_visualizer.py trajectory.csv --config knife_pose_2.yaml

# Compare the data_comparison plots
```

---

## See Also

- **Main Documentation:** `../README.md` - Full system overview
- **Batch Processor:** `../batch_trajectory_processor.py` - Production processing
- **Configuration Example:** `compare.yaml` - Fully documented config file
- **Analysis Scripts:** `../trajectory_processing/` - Underlying analysis algorithms

---

**Last Updated:** 2025-01-18  
**Version:** 1.0  
**For Questions:** See main README.md troubleshooting section