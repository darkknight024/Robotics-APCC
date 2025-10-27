# Batch Processing Quick Start Guide

## 5-Minute Setup

### 1. Verify Configuration Files
Check that these files exist:
- ✓ `Assests/Robot APCC/knife_poses.yaml` - Knife pose definitions
- ✓ `Assests/Robot APCC/IRB-1300 900 URDF/urdf/` - Contains `*_ee.urdf`
- ✓ `Assests/Robot APCC/IRB-1300 1150 URDF/urdf/` - Contains `*_ee.urdf`
- ✓ `Assests/Robot APCC/Toolpaths/Successful/` - Contains CSV files

### 2. Install Dependencies
```bash
pip install pyyaml pinocchio numpy pandas matplotlib tqdm
```

### 3. Run Batch Analysis
```bash
cd /path/to/Robotics-APCC
python scripts/batch_analyze_trajectories.py
```

**That's it!** Results will be saved to `results/` directory.

---

## Common Usage Scenarios

### Quick Test (Single Experiment)
Temporarily modify `knife_poses.yaml` to have only one pose, and keep only one toolpath CSV:

```bash
python scripts/batch_analyze_trajectories.py --no-visualize
```

### Full Analysis with Custom Output
```bash
python scripts/batch_analyze_trajectories.py \
    -o /path/to/my_results \
    --max-iterations 2000
```

### Faster Processing (Skip Plots)
```bash
python scripts/batch_analyze_trajectories.py --no-visualize
```

### High-Quality Analysis (More Iterations)
```bash
python scripts/batch_analyze_trajectories.py \
    --max-iterations 3000 \
    --tolerance 1e-5
```

---

## Adding a New Knife Pose

Edit `Assests/Robot APCC/knife_poses.yaml`:

```yaml
poses:
  pose_1:
    description: "Standard knife pose"
    translation:
      x: -367.773
      y: -915.815
      z: 520.4
    rotation:
      w: 0.00515984
      x: 0.712632
      y: -0.701518
      z: 0.000396522
  
  # NEW POSE - ADD HERE
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

Batch system will automatically include all poses!

---

## Understanding Results

### Directory Structure
```
results/
├── batch_experiment_summary.yaml       # Overall summary
├── IRB-1300_900__toolpath_name__pose_1/
│   ├── experiment_results.yaml         # Detailed results
│   ├── experiment.csv                  # Waypoint data
│   └── plots/
│       ├── manipulability_plot.png
│       ├── reachability_plot.png
│       └── (more plots)
└── (more experiment folders)
```

### Key Metrics

**Reachability Rate** (%)
- Percentage of waypoints robot can reach
- Higher is better

**Manipulability** (unitless)
- Measure of robot dexterity
- Closer to 0 = near singularity (bad)
- Larger values = better workspace quality

**Condition Number**
- How "stretched" the workspace is
- Lower is better (< 100 is good)
- Higher values = ill-conditioned (bad)

**Position Error** (m)
- Difference between target and achieved position
- Smaller is better

---

## Interpreting Output

### Batch Summary YAML
```yaml
timestamp: "2025-10-27T14:23:45"
total_experiments: 48
completed_experiments: 45       # Successfully completed
failed_experiments: 3           # Errors occurred
```

### Experiment Results YAML
```yaml
summary:
  reachability_rate_percent: 97.8    # 97.8% of points reachable
  avg_manipulability: 0.0456         # Average dexterity
  avg_condition_number: 425.3        # Workspace quality
  avg_position_error_m: 0.000142     # ~0.14mm average error
```

### CSV Format
```
waypoint_index, reachable, manipulability, position_error_m, ...
0,              True,      0.0456,         0.0001, ...
1,              True,      0.0458,         0.0001, ...
2,              False,     ,               0.0234, ...
```

---

## Troubleshooting

### "No URDF files found"
**Problem**: Robot models not discovered
**Solution**: 
- Verify folder names end with "URDF" (e.g., "IRB-1300 900 URDF")
- Ensure `urdf/` subfolder exists in each robot directory
- Check URDF files exist and end with `.urdf`

### "Knife poses YAML file not found"
**Problem**: Configuration file missing
**Solution**:
- Create `Assests/Robot APCC/knife_poses.yaml`
- Verify YAML syntax is correct
- Check file path spelling and capitalization

### All Results are "Unreachable"
**Problem**: Robot cannot reach any waypoints
**Solutions**:
1. Verify knife pose translation/rotation are correct
2. Check toolpath is in valid robot workspace
3. Increase IK iterations: `--max-iterations 3000`
4. Relax tolerance: `--tolerance 1e-3`
5. Verify correct URDF file is being used

### Script Runs Very Slowly
**Problem**: Processing takes too long
**Solutions**:
1. Disable plots: `--no-visualize`
2. Reduce iterations: `--max-iterations 500`
3. Increase tolerance: `--tolerance 1e-3`
4. Test with subset of data first

### Out of Memory Error
**Problem**: System runs out of RAM
**Solutions**:
1. Edit script to process fewer toolpaths per batch
2. Add more swap space
3. Reduce number of knife poses temporarily
4. Process one robot at a time

---

## Performance Tips

### For Quick Prototyping
```bash
# Remove non-essential knife poses from YAML
# Keep only 1-2 toolpaths
# Reduce iterations and increase tolerance
python scripts/batch_analyze_trajectories.py \
    --no-visualize \
    --max-iterations 500 \
    --tolerance 1e-2
```

### For Production Analysis
```bash
# Use all data
# High iteration count for accuracy
# Keep visualization for inspection
python scripts/batch_analyze_trajectories.py \
    --max-iterations 2000 \
    --tolerance 1e-4
```

### Typical Run Times
| Configuration | Time |
|--------------|------|
| 1 robot, 1 toolpath, 1 pose | 5-10 min |
| 2 robots, 4 toolpaths, 1 pose | 45-60 min |
| 2 robots, 16 toolpaths, 2 poses | 4-8 hours |

---

## Next Steps

1. **Read Full Documentation**: `BATCH_PROCESSING_README.md`
2. **Understand Kinematics**: Review comments in `batch_analyze_trajectories.py`
3. **Analyze Results**: Check generated YAML and CSV files
4. **Extend System**: Modify modules for custom analysis

---

## Quick Command Reference

```bash
# Default settings
python scripts/batch_analyze_trajectories.py

# Custom output directory
python scripts/batch_analyze_trajectories.py -o my_results

# Skip visualization (faster)
python scripts/batch_analyze_trajectories.py --no-visualize

# Adjust IK solver parameters
python scripts/batch_analyze_trajectories.py \
    --max-iterations 2000 \
    --tolerance 1e-4

# All options together
python scripts/batch_analyze_trajectories.py \
    -o results \
    --max-iterations 2000 \
    --tolerance 1e-4 \
    --no-visualize
```

---

## Useful File Locations

```
Project Root
├── Assests/Robot APCC/
│   ├── knife_poses.yaml                  ← Edit knife poses here
│   ├── IRB-1300 900 URDF/urdf/           ← Robot URDF files
│   ├── IRB-1300 1150 URDF/urdf/
│   └── Toolpaths/Successful/             ← Toolpath CSV files
├── scripts/
│   ├── batch_analyze_trajectories.py     ← Main batch script
│   ├── utils/
│   │   ├── batch_processor.py            ← Discovery module
│   │   ├── results_handler.py            ← Results storage
│   │   └── (other utilities)
│   ├── BATCH_PROCESSING_README.md        ← Full documentation
│   └── BATCH_QUICK_START.md              ← This file
└── results/                              ← Output location (created automatically)
```

---

## Support

For detailed information, see: `scripts/BATCH_PROCESSING_README.md`

For specific issues with modules, check docstrings in:
- `scripts/utils/batch_processor.py`
- `scripts/utils/results_handler.py`
- `scripts/batch_analyze_trajectories.py`

---

**Happy batch analyzing!** 🤖

Last Updated: October 2025
