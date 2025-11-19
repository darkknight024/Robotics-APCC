#!/usr/bin/env python3
"""
Unified Batch Trajectory Processor - Orchestration Script

A clean orchestration agent for processing multiple trajectories across different robots,
toolpaths, and knife poses. This script acts as a coordinator, delegating actual analysis
to analyze_irb1300_trajectory.py and visualization to trajectory_visualizer.py.

Architecture:
- This script: Pure orchestration (discovery, loop, result organization)
- analyze_irb1300_trajectory.py: Handles kinematic feasibility analysis
- trajectory_visualizer.py: Handles non-interactive visualization

Configuration:
    Unified YAML config file contains:
    - robots: List of robot URDF directories to process
    - toolpaths: List of CSV trajectory files
    - knife_poses_yaml: Path to knife poses YAML file
    - output_dir: Output directory (optional, defaults to folder relative to input CSVs)

Usage:
    python batch_trajectory_processor.py -c config.yaml
    python batch_trajectory_processor.py -c config.yaml -o /custom/output
    python batch_trajectory_processor.py -c config.yaml --no-visualize

Output Structure:
    results/
    ├── csv_filename_1/
    │   ├── Traj_1_[1-50]/
    │   │   ├── experiment_results.yaml
    │   │   ├── experiment.csv
    │   │   ├── Traj_1_[1-50]_visualization.png
    │   │   ├── Traj_1_[1-50]_data_comparison.png
    │   │   └── Traj_1_[1-50]_transformed.csv
    │   ├── Traj_2_[1-100]/
    │   │   └── (same structure)
    │   └── csv_summary.yaml
    └── csv_filename_2/
        └── (same structure)

Requirements:
    - pinocchio
    - numpy
    - pandas
    - matplotlib
    - pyyaml
    - tqdm
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from typing import List, Optional

# Add utils directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from batch_processor import discover_robot_models
from csv_handling import read_trajectories_from_csv

# Import analysis functions
from trajectory_processing.analyze_irb1300_trajectory import analyze_trajectory
from trajectory_processing.trajectory_continuity_analyzer import analyze_trajectory_continuity


class BatchTrajectoryProcessor:
    """
    Orchestration agent for batch trajectory processing across multiple robots, toolpaths, and knife poses.
    
    This is the main entry point for the entire batch processing system. It coordinates:
    1. Robot discovery and URDF loading
    2. Trajectory CSV discovery and parsing
    3. 3-level nested loop: CSV → Robot → Knife Pose → Trajectory
    4. Kinematic feasibility analysis via IK solving
    5. Continuity verification (C¹ velocity checks)
    6. Visualization and report generation
    
    Workflow:
        For each CSV file:
            For each robot configuration:
                For each knife pose:
                    For each trajectory in the CSV:
                        - Transform trajectory to robot base frame
                        - Run IK to find joint angles for all poses
                        - Compute manipulability metrics
                        - Check C¹ continuity with velocity limits
                        - Generate 3D visualizations
                        - Save results in organized structure
    
    Results are saved in:
        results/
        ├── csv_name_1/
        │   ├── Robot_Model_pose_1/
        │   │   ├── Traj_1_[1-N]/
        │   │   │   ├── experiment_results.yaml
        │   │   │   ├── experiment.csv
        │   │   │   ├── Traj_1_visualization.png
        │   │   │   └── continuity/continuity_analysis.yaml
        │   │   └── Traj_2_[1-M]/...
        │   └── csv_summary.yaml
        └── batch_processing_summary.yaml
    """

    def __init__(self, config_path, output_dir=None, visualize=True):
        """
        Initialize the batch processor.

        Args:
            config_path (str): Path to unified configuration YAML file containing:
                - robots: List of robot configurations with paths, reach, and limits
                - toolpaths: List of CSV trajectory files or directories
                - knife_poses: Dictionary of knife pose transformations
                - visualization, analysis, continuity: Processing settings
            output_dir (str, optional): Override output directory from config. If provided,
                all results are saved to this directory instead of 'results/'
            visualize (bool, optional): Whether to generate visualization plots. Default: True
                Can be further controlled per visualization type in config file.
        
        Raises:
            SystemExit: If config file cannot be loaded or is invalid
        """
        self.config = self._load_config(config_path)
        self.visualize = visualize

        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.config.get('output', {}).get('directory'):
            self.output_dir = Path(self.config['output']['directory'])
        else:
            # Default: create results in current directory
            self.output_dir = Path('results')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.summary = {
            'timestamp': self.start_time.isoformat(),
            'config_path': str(config_path),
            'output_dir': str(self.output_dir),
            'total_csvs': 0,
            'total_trajectories': 0,
            'completed_trajectories': 0,
            'failed_trajectories': 0,
            'csv_results': []
        }

    def _load_config(self, config_path):
        """Load unified configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded config from: {config_path}")
            return config
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            sys.exit(1)

    def _discover_components(self):
        """Discover robots and trajectories from config."""
        # Get robots from config
        robot_configs = self.config.get('robots', [])
        robots = []
        
        if isinstance(robot_configs, str):
            # If it's a path, discover all robots in that directory
            robots = discover_robot_models(robot_configs)
        elif isinstance(robot_configs, list):
            # List of robot configs (can be strings or dicts)
            for robot_cfg in robot_configs:
                if isinstance(robot_cfg, str):
                    # Simple path string
                    robot_path = Path(robot_cfg)
                    urdf_path = robot_path / 'urdf'
                    if urdf_path.exists():
                        urdf_files = list(urdf_path.glob('*_ee.urdf'))
                        if not urdf_files:
                            urdf_files = list(urdf_path.glob('*.urdf'))
                        if urdf_files:
                            robots.append({
                                'robot_name': robot_path.name,
                                'robot_dir': str(robot_path),
                                'urdf_path': str(urdf_files[0]),
                                'urdf_filename': urdf_files[0].name
                            })
                elif isinstance(robot_cfg, dict):
                    # Robot config dict with metadata and limits
                    robot_path = Path(robot_cfg['path'])
                    urdf_path = robot_path / 'urdf'
                    if urdf_path.exists():
                        urdf_files = list(urdf_path.glob('*_ee.urdf'))
                        if not urdf_files:
                            urdf_files = list(urdf_path.glob('*.urdf'))
                        if urdf_files:
                            robot_entry = {
                                'robot_name': robot_path.name,
                                'robot_dir': str(robot_path),
                                'urdf_path': str(urdf_files[0]),
                                'urdf_filename': urdf_files[0].name,
                                'model': robot_cfg.get('model', robot_path.name),
                                'velocity_limits_rad_s': robot_cfg.get('velocity_limits_rad_s'),
                                'acceleration_limits_rad_s2': robot_cfg.get('acceleration_limits_rad_s2'),
                            }
                            robots.append(robot_entry)

        # Get toolpaths from config
        toolpath_configs = self.config.get('toolpaths', [])
        toolpaths = []
        for toolpath_item in toolpath_configs:
            if isinstance(toolpath_item, str):
                # Could be a file or directory
                path = Path(toolpath_item)
                if path.is_dir():
                    # Directory: find all CSVs
                    toolpaths.extend(sorted(path.glob('*.csv')))
                elif path.is_file():
                    toolpaths.append(path)
            elif isinstance(toolpath_item, dict):
                # Could have 'path' key
                toolpaths.append(Path(toolpath_item.get('path', toolpath_item)))

        # Get knife poses from config (now embedded directly)
        knife_poses = self.config.get('knife_poses', {})
        if not knife_poses:
            print("Warning: No knife poses defined in config")
            knife_poses = {'pose_1': {'description': 'default'}}

        return robots, toolpaths, knife_poses

    def process_all(self):
        """
        Execute complete batch processing pipeline.
        
        Orchestrates the 3-level nested loop:
        1. CSV Loop: Iterate through all trajectory CSV files
        2. Robot Loop: For each CSV, process with all configured robots
        3. Pose Loop: For each robot, test all knife pose configurations
        4. Trajectory Loop: For each combination, analyze all trajectories in the CSV
        
        For each trajectory, performs:
        - Coordinate transformation (plate frame → robot base frame)
        - IK solving for joint angle computation
        - Manipulability analysis (Yoshikawa metric)
        - C¹ continuity checking (velocity limits)
        - 3D visualization and delta analysis
        - Result aggregation and reporting
        
        Generates organized output:
            results/
            ├── {csv_name}/
            │   ├── {robot_name}_{pose_name}/
            │   │   ├── Traj_1_[1-N]/
            │   │   │   ├── experiment_results.yaml
            │   │   │   ├── experiment.csv
            │   │   │   ├── Traj_1_visualization.png
            │   │   │   ├── pose_viz/
            │   │   │   │   ├── Traj_1_comparison.png
            │   │   │   │   ├── Traj_1_data_comparison.png
            │   │   │   │   └── Traj_1_delta_analysis.png
            │   │   │   └── continuity/
            │   │   │       └── continuity_analysis.yaml
            │   │   └── Traj_2_[1-M]/...
            │   └── csv_summary.yaml
            └── batch_processing_summary.yaml
        
        Error Handling:
        - Continues on individual trajectory failures (if configured)
        - Logs all errors to console and summary files
        - Maintains count of completed vs. failed trajectories
        """
        print(f"\n{'='*70}")
        print("Batch Trajectory Processor")
        print(f"{'='*70}")

        # Discover components
        robots, toolpaths, knife_poses = self._discover_components()
        print(f"\n✓ Configuration:")
        print(f"  - Robots: {len(robots)}")
        print(f"  - CSV files: {len(toolpaths)}")
        print(f"  - Knife poses: {len(knife_poses)}")
        print(f"  - Total combinations: {len(robots) * len(knife_poses)} per CSV\n")

        if not robots or not toolpaths:
            print("✗ Error: No robots or toolpaths found")
            return

        # Process each CSV file
        for csv_idx, csv_path in enumerate(toolpaths, 1):
            print(f"\n{'='*70}")
            print(f"[{csv_idx}/{len(toolpaths)}] CSV: {csv_path.name}")
            print(f"{'='*70}")
            csv_result = self._process_csv_file(csv_path, robots, knife_poses)
            if csv_result:
                self.summary['csv_results'].append(csv_result)

        # Save summary
        self._save_summary()
        self._print_summary()

    def _extract_speeds_from_csv(self, csv_path: str, num_trajectories: int) -> List[Optional[float]]:
        """
        Extract speeds from CSV 8th column (first pose of each trajectory).
        
        Args:
            csv_path: Path to CSV file
            num_trajectories: Number of trajectories in file
            
        Returns:
            List of speeds (mm/s) for each trajectory, or None for each if not available
        """
        import csv
        
        speeds = [None] * num_trajectories
        current_traj_idx = 0
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    clean_row = [col.strip() for col in row if col.strip()]
                    
                    if not clean_row:
                        continue
                    
                    # Check for trajectory separator
                    if len(clean_row) == 1:
                        if current_traj_idx < num_trajectories - 1:
                            current_traj_idx += 1
                        continue
                    
                    # Skip if not enough columns
                    if len(clean_row) < 8:
                        continue
                    
                    # Skip if already have speed for this trajectory
                    if speeds[current_traj_idx] is not None:
                        continue
                    
                    # Try to extract speed from 8th column (first pose of trajectory)
                    try:
                        speed = float(clean_row[7])
                        speeds[current_traj_idx] = speed
                    except (ValueError, IndexError, TypeError):
                        pass
        except Exception as e:
            print(f"  ⚠ Could not extract speeds from CSV: {e}")
        
        return speeds
    
    def _process_csv_file(self, csv_path, robots, knife_poses):
        """Process a single CSV file with all robot/pose combinations."""
        csv_path = Path(csv_path)
        csv_stem = csv_path.stem

        # Create CSV-specific output directory
        csv_output_dir = self.output_dir / csv_stem
        csv_output_dir.mkdir(parents=True, exist_ok=True)

        # Parse trajectories from CSV
        try:
            trajectories_m = read_trajectories_from_csv(str(csv_path))
        except Exception as e:
            print(f"  ✗ Error parsing CSV: {e}")
            return None

        if not trajectories_m:
            print(f"  ✗ No trajectories found in CSV")
            return None

        print(f"  ✓ Found {len(trajectories_m)} trajectory(ies)")
        
        # Extract speeds from CSV (8th column, first pose of each trajectory)
        trajectory_speeds = self._extract_speeds_from_csv(str(csv_path), len(trajectories_m))

        csv_result = {
            'csv_file': str(csv_path),
            'num_trajectories': len(trajectories_m),
            'processed_trajectories': 0,
            'combinations': [],
            'statistics': {
                'total_trajectories': len(trajectories_m),
                'total_combinations': len(robots) * len(knife_poses),
                'by_trajectory': {},
                'by_robot_pose': {}
            }
        }

        # OUTER LOOP: Iterate through all robots
        for robot_idx, robot in enumerate(robots, 1):
            # MIDDLE LOOP: Iterate through all knife poses
            for pose_idx, (pose_name, knife_pose) in enumerate(knife_poses.items(), 1):
                combo_name = f"{robot['robot_name']}_{pose_name}"
                total_combos = len(robots) * len(knife_poses)
                current_combo = (robot_idx - 1) * len(knife_poses) + pose_idx
                
                print(f"  [{current_combo}/{total_combos}] Robot: {robot['robot_name']} | Pose: {pose_name}")
                
                # Initialize stats for this robot/pose combination
                combo_stats = {
                    'robot': robot['robot_name'],
                    'pose': pose_name,
                    'trajectories': []
                }
                
                # INNER LOOP: Iterate through all trajectories
                for traj_idx, trajectory_m in enumerate(trajectories_m, 1):
                    num_poses = len(trajectory_m)
                    
                    # Create organized output directory structure
                    traj_folder_name = f"Traj_{traj_idx}_[1-{num_poses}]"
                    combo_output_dir = csv_output_dir / combo_name
                    traj_output_dir = combo_output_dir / traj_folder_name
                    traj_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    status_str = f"    [{traj_idx}/{len(trajectories_m)}] {traj_folder_name}... "

                    try:
                        # Get speed for this trajectory (from CSV 8th column, first pose)
                        speed_mm_s = trajectory_speeds[traj_idx - 1] if traj_idx <= len(trajectory_speeds) else None
                        
                        self._process_single_trajectory(
                            csv_path, trajectory_m, robot, knife_pose,
                            traj_idx, num_poses, traj_output_dir, speed_mm_s=speed_mm_s
                        )
                        csv_result['processed_trajectories'] += 1
                        self.summary['completed_trajectories'] += 1
                        print(f"{status_str}✓")
                        
                        # Collect detailed statistics from results
                        traj_stats = self._collect_trajectory_statistics(
                            traj_output_dir, traj_idx, robot['robot_name'], pose_name
                        )
                        
                        csv_result['combinations'].append({
                            'robot': robot['robot_name'],
                            'pose': pose_name,
                            'trajectory': traj_idx,
                            'status': 'completed'
                        })
                        
                        combo_stats['trajectories'].append(traj_stats)
                        
                        # Track per-trajectory stats
                        if traj_idx not in csv_result['statistics']['by_trajectory']:
                            csv_result['statistics']['by_trajectory'][traj_idx] = {
                                'total_poses': num_poses,
                                'combinations': []
                            }
                        csv_result['statistics']['by_trajectory'][traj_idx]['combinations'].append(traj_stats)
                        
                    except Exception as e:
                        print(f"{status_str}✗ {str(e)[:50]}")
                        self.summary['failed_trajectories'] += 1
                        csv_result['combinations'].append({
                            'robot': robot['robot_name'],
                            'pose': pose_name,
                            'trajectory': traj_idx,
                            'status': 'failed',
                            'error': str(e)
                        })

                # Store robot/pose combination stats
                if combo_stats['trajectories']:
                    csv_result['statistics']['by_robot_pose'][combo_name] = combo_stats

        # Save CSV-level summary with detailed statistics
        self._save_csv_summary(csv_output_dir, csv_result)
        self.summary['total_trajectories'] += len(trajectories_m)
        self.summary['total_csvs'] += 1

        return csv_result

    def _process_single_trajectory(self, csv_path, trajectory_m, robot, knife_pose,
                                   traj_idx, num_poses, traj_output_dir, speed_mm_s=None):
        """
        Process a single trajectory with complete kinematic and continuity analysis.
        
        Main processing function that orchestrates all analysis steps for a single trajectory:
        
        Processing Steps:
        1. Coordinate Transformation:
           - Extracts knife pose from config (translation in mm, rotation as quaternion)
           - Transforms trajectory from T_P_K (knife in plate frame) to T_B_K (knife in robot base)
           - Converts translations from mm to meters for URDF compatibility
        
        2. Kinematic Feasibility Analysis:
           - Calls analyze_trajectory() with transformed poses
           - IK solver computes joint angles for all reachable poses
           - Returns reachability status and manipulability metrics
           - Skips continuity if any pose is unreachable
        
        3. Continuity Analysis (if enabled):
           - Performs C¹ (velocity) continuity checks
           - Compares against robot velocity limits per joint
           - Uses speed from CSV 8th column (first pose) as trajectory speed
           - Generates detailed continuity reports and graphs
        
        4. Visualization (if enabled):
           - Creates 3D trajectory plots (original and transformed)
           - Generates data comparison graphs (translation vs rotation)
           - Creates delta analysis (pose-to-pose changes)
           - Saves all plots as high-resolution PNG files
        
        Args:
            csv_path (str): Path to source CSV file (for reference in logs)
            trajectory_m (np.ndarray): Trajectory poses in meters, shape (n_poses, 7) where:
                                      columns are [x_m, y_m, z_m, qw, qx, qy, qz]
            robot (dict): Robot configuration containing:
                - urdf_path: Path to robot URDF file
                - robot_name: Name of robot model
                - velocity_limits_rad_s: Per-joint velocity limits
                - acceleration_limits_rad_s2: Per-joint acceleration limits
                - reach_m: Robot workspace reach (for normalized manipulability)
            knife_pose (dict): Knife transformation containing:
                - translation: {x, y, z} in millimeters
                - rotation: {w, x, y, z} quaternion components
            traj_idx (int): 1-based trajectory index (for naming)
            num_poses (int): Total number of poses in trajectory
            traj_output_dir (Path): Output directory for this trajectory's results
            speed_mm_s (float, optional): Speed from CSV 8th column in mm/s. Used for:
                - Continuity analysis (velocity checks)
                - Timing analysis
                If None, defaults to 100 mm/s
        
        Returns:
            bool: True if processing succeeded, False if failed (with details logged)
        
        Output Files Generated:
            - experiment_results.yaml: IK and kinematic analysis results
            - experiment.csv: Joint angles for all poses
            - Traj_N_visualization.png: 3D point cloud plot
            - pose_viz/Traj_N_comparison.png: Original vs transformed side-by-side
            - pose_viz/Traj_N_data_comparison.png: Translation/rotation data graphs
            - pose_viz/Traj_N_delta_analysis.png: Pose delta analysis graphs
            - continuity/continuity_analysis.yaml: C¹ continuity report
            - continuity/continuity_analysis.png: Continuity graphs
        
        Error Handling:
        - Returns False on transformation failures
        - Returns False if IK fails for all poses
        - Returns False if continuity analysis fails (but logs warning)
        - Returns False if visualization fails (but logs warning)
        - Logs detailed error messages for debugging
        """
        try:
            # Extract knife pose parameters (translations are in mm from config)
            knife_translation_mm = np.array([
                knife_pose.get('translation', {}).get('x', 0),
                knife_pose.get('translation', {}).get('y', 0),
                knife_pose.get('translation', {}).get('z', 0)
            ])
            knife_rotation = np.array([
                knife_pose.get('rotation', {}).get('w', 1),
                knife_pose.get('rotation', {}).get('x', 0),
                knife_pose.get('rotation', {}).get('y', 0),
                knife_pose.get('rotation', {}).get('z', 0)
            ])
            
            # Transform trajectory to robot base frame using handle_transforms
            from handle_transforms import transform_to_ee_poses_matrix_with_pose
            
            # NOTE: trajectory_m is already in METERS from read_trajectories_from_csv
            # Transform from T_P_K (knife poses in plate frame) to T_B_P (plate in base frame)
            trajectories_m = transform_to_ee_poses_matrix_with_pose(
                [trajectory_m],  # Already in meters
                knife_translation_mm / 1000.0,  # Convert knife translation from mm to meters
                knife_rotation
            )[0]
            
            # Run kinematic analysis on transformed trajectory
            # Pass robot reach for normalized manipulability calculation
            robot_reach_m = robot.get('reach_m', 1.0)  # Default to 1.0 m if not specified
            results = analyze_trajectory(
                urdf_path=robot['urdf_path'],
                trajectory_m=trajectories_m,
                output_dir=str(traj_output_dir),
                config=self.config,
                trajectory_id=traj_idx,
                robot_reach_m=robot_reach_m
            )
            
            if not results:
                print(f"    ✗ Analysis returned no results")
                return False
            
            # Extract joint angles for continuity analysis
            joint_angles_rad = []
            reachable_count = 0
            for r in results:
                if r['reachable']:
                    joint_angles_rad.append([r.get(f'q{j+1}_rad', 0) for j in range(6)])
                    reachable_count += 1
                else:
                    joint_angles_rad.append(None)  # Track unreachable poses
            
            # If ANY pose is unreachable, skip continuity analysis
            if reachable_count < len(results):
                print(f"    ⚠ Skipping continuity: {len(results) - reachable_count}/{len(results)} poses unreachable (IK failed)")
                return False
            
            # All poses are reachable, proceed with continuity analysis
            if joint_angles_rad:
                joint_angles_rad = np.array([jang for jang in joint_angles_rad if jang is not None])
            else:
                joint_angles_rad = None
            
            # Run continuity analysis if enabled in config
            if joint_angles_rad is not None and self.config.get('continuity', {}).get('enabled', False):
                try:
                    # Extract robot velocity/acceleration limits
                    robot_v_limits = robot.get('velocity_limits_rad_s')
                    robot_a_limits = robot.get('acceleration_limits_rad_s2')
                    
                    analyze_trajectory_continuity(
                        trajectory_m=trajectories_m,
                        joint_angles_rad=joint_angles_rad,
                        output_dir=traj_output_dir,
                        config=self.config,
                        trajectory_id=traj_idx,
                        speed_mm_s=speed_mm_s,  # From CSV 8th column (first pose)
                        robot_velocity_limits_rad_s=robot_v_limits,
                        robot_acceleration_limits_rad_s2=robot_a_limits
                    )
                except Exception as e:
                    import traceback
                    print(f"    ⚠ Continuity analysis failed: {e}")
                    if self.config.get('batch', {}).get('verbose', False):
                        traceback.print_exc()
            
            # Run visualization if enabled in config
            if self.visualize and self.config.get('visualization', {}).get('enabled', True):
                self._run_visualization(csv_path, trajectory_m, traj_idx, num_poses, traj_output_dir)
            
            return True

        except Exception as e:
            print(f"    ✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_trajectory_csv(self, trajectory_m, output_path):
        """Save trajectory to CSV file."""
        try:
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for pose in trajectory_m:
                    x_mm = pose[0] * 1000
                    y_mm = pose[1] * 1000
                    z_mm = pose[2] * 1000
                    writer.writerow([x_mm, y_mm, z_mm, pose[3], pose[4], pose[5], pose[6]])
        except Exception as e:
            print(f"    ✗ Could not save trajectory CSV: {e}")

    def _run_visualization(self, traj_csv, trajectory_m, traj_idx, num_poses, output_dir):
        """Generate visualization plots for trajectory."""
        try:
            # Create pose_viz subfolder
            pose_viz_dir = Path(output_dir) / "pose_viz"
            pose_viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Get visualization config
            viz_config = self.config.get('visualization', {})
            if not viz_config.get('enabled', True):
                return
            
            # Import visualization functions directly
            import importlib.util
            scripts_dir = Path(__file__).parent
            viz_path = scripts_dir / "trajectory_processing" / "trajectory_visualizer.py"
            
            spec = importlib.util.spec_from_file_location("trajectory_visualizer_module", viz_path)
            viz_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(viz_module)
            
            create_3d_visualization_batch = viz_module.create_3d_visualization_batch
            create_data_comparison_graphs = viz_module.create_data_comparison_graphs
            create_delta_analysis_graphs = viz_module.create_delta_analysis_graphs
            
            # trajectory_m is already in meters (transformed in robot base frame)
            trajectory_m_viz = trajectory_m.copy()
            
            # Get transformation settings
            knife_poses = self.config.get('knife_poses', {})
            first_pose_name = list(knife_poses.keys())[0] if knife_poses else 'pose_1'
            knife_pose_data = knife_poses.get(first_pose_name, {})
            knife_translation_mm = np.array([
                knife_pose_data.get('translation', {}).get('x', 0),
                knife_pose_data.get('translation', {}).get('y', 0),
                knife_pose_data.get('translation', {}).get('z', 0)
            ])
            knife_rotation = np.array([
                knife_pose_data.get('rotation', {}).get('w', 1),
                knife_pose_data.get('rotation', {}).get('x', 0),
                knife_pose_data.get('rotation', {}).get('y', 0),
                knife_pose_data.get('rotation', {}).get('z', 0)
            ])
            
            # Apply transformation (returns meters since input is meters)
            from handle_transforms import transform_to_ee_poses_matrix_with_pose
            trajectory_transformed_m = transform_to_ee_poses_matrix_with_pose(
                [trajectory_m],
                knife_translation_mm / 1000.0,
                knife_rotation
            )[0]
            
            # 3D Visualization
            if viz_config.get('plot_3d', {}).get('enabled', True):
                create_3d_visualization_batch(
                    trajectory_m_viz, trajectory_transformed_m,
                    traj_idx, num_poses, pose_viz_dir,
                    scale=viz_config.get('plot_3d', {}).get('scale', 0.001),
                    point_size=viz_config.get('plot_3d', {}).get('point_size', 20)
                )
            
            # Data Comparison Graphs
            if viz_config.get('data_comparison', {}).get('enabled', True):
                create_data_comparison_graphs(
                    [trajectory_m_viz], [trajectory_transformed_m],
                    traj_idx, num_poses, pose_viz_dir
                )
            
            # Delta Analysis Graphs (convert meters to mm for delta analysis)
            if viz_config.get('delta_analysis', {}).get('enabled', True):
                trajectory_m_mm = trajectory_m.copy()
                trajectory_m_mm[:, :3] = trajectory_m_mm[:, :3] * 1000.0
                trajectory_transformed_mm = trajectory_transformed_m.copy()
                trajectory_transformed_mm[:, :3] = trajectory_transformed_mm[:, :3] * 1000.0
                create_delta_analysis_graphs(
                    [trajectory_m_mm], [trajectory_transformed_mm],
                    traj_idx, num_poses, pose_viz_dir
                )
                
        except Exception as e:
            print(f"    ⚠ Visualization skipped: {e}")

    def _save_csv_summary(self, csv_output_dir, csv_result):
        """Save summary for CSV-level results."""
        summary_path = csv_output_dir / 'csv_summary.yaml'
        try:
            with open(summary_path, 'w') as f:
                yaml.dump(csv_result, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"  ✗ Could not save CSV summary: {e}")

    def _save_summary(self):
        """Save overall batch summary."""
        summary_path = self.output_dir / 'batch_processing_summary.yaml'
        try:
            with open(summary_path, 'w') as f:
                yaml.dump(self.summary, f, default_flow_style=False, sort_keys=False)
            print(f"\n✓ Summary saved to: {summary_path}")
        except Exception as e:
            print(f"✗ Could not save summary: {e}")

    def _collect_trajectory_statistics(self, traj_output_dir, traj_idx, robot_name, pose_name):
        """
        Collect statistics from processed trajectory results (IK and continuity data).
        
        Reads trajectory_analysis_results.csv to get reachability counts and
        continuity report to get continuity pass/fail counts.
        
        Args:
            traj_output_dir (Path): Output directory containing results
            traj_idx (int): Trajectory index for reference
            robot_name (str): Robot name for reference
            pose_name (str): Pose name for reference
            
        Returns:
            dict: Statistics with keys:
                - trajectory_id: Trajectory index
                - robot: Robot name
                - pose: Pose name
                - ik_statistics: {total_poses, reachable, unreachable, reachability_percent}
                - continuity_statistics: {passed, failed, pass_percent} (or null if disabled)
        """
        stats = {
            'trajectory_id': traj_idx,
            'robot': robot_name,
            'pose': pose_name,
            'ik_statistics': {
                'total_poses': 0,
                'reachable': 0,
                'unreachable': 0,
                'reachability_percent': 0.0
            },
            'continuity_statistics': None
        }
        
        try:
            # Read trajectory analysis results (IK data)
            csv_results_path = traj_output_dir / 'trajectory_analysis_results.csv'
            if csv_results_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_results_path)
                stats['ik_statistics']['total_poses'] = len(df)
                stats['ik_statistics']['reachable'] = int(df['reachable'].sum())
                stats['ik_statistics']['unreachable'] = stats['ik_statistics']['total_poses'] - stats['ik_statistics']['reachable']
                if stats['ik_statistics']['total_poses'] > 0:
                    stats['ik_statistics']['reachability_percent'] = round(
                        100.0 * stats['ik_statistics']['reachable'] / stats['ik_statistics']['total_poses'], 1
                    )
        except Exception as e:
            print(f"    ⚠ Could not read IK statistics: {e}")
        
        try:
            # Read continuity report
            continuity_report_path = traj_output_dir / 'continuity' / f'Traj_{traj_idx}_continuity_report.json'
            if continuity_report_path.exists():
                import json
                with open(continuity_report_path, 'r') as f:
                    continuity_data = json.load(f)
                
                # Check if continuity analysis was enabled and extract results
                if continuity_data.get('enabled', False):
                    continuity_results = continuity_data.get('continuity_results', [])
                    if continuity_results:
                        passed = sum(1 for r in continuity_results if r.get('passed', False))
                        failed = len(continuity_results) - passed
                        stats['continuity_statistics'] = {
                            'total_checks': len(continuity_results),
                            'passed': passed,
                            'failed': failed,
                            'pass_percent': round(100.0 * passed / len(continuity_results), 1) if continuity_results else 0.0
                        }
                    else:
                        # Check top-level passed field
                        if 'passed' in continuity_data:
                            stats['continuity_statistics'] = {
                                'passed': continuity_data['passed'],
                                'note': 'See continuity report for details'
                            }
        except Exception as e:
            print(f"    ⚠ Could not read continuity statistics: {e}")
        
        return stats

    def _print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*70}")
        print("Batch Processing Summary")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total CSV files: {self.summary['total_csvs']}")
        print(f"Total trajectories: {self.summary['total_trajectories']}")
        print(f"Completed: {self.summary['completed_trajectories']}")
        print(f"Failed: {self.summary['failed_trajectories']}")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Processing time: {elapsed:.1f}s")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Batch Trajectory Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_trajectory_processor.py -c config.yaml
  python batch_trajectory_processor.py -c config.yaml -o results
        """
    )

    parser.add_argument('-c', '--config', required=True,
                       help='Path to unified configuration YAML file')
    parser.add_argument('-o', '--output', help='Output directory (overrides config)')

    args = parser.parse_args()

    processor = BatchTrajectoryProcessor(
        args.config,
        output_dir=args.output,
        visualize=True  # Config file controls what gets visualized
    )
    processor.process_all()


if __name__ == '__main__':
    main()

