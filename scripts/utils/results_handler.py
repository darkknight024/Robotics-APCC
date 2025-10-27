#!/usr/bin/env python3
"""
results_handler.py

Module for handling and storing kinematic feasibility analysis results in YAML format.

This module provides functionality to:
- Create structured result containers for each experiment
- Aggregate results from multiple runs
- Save results to YAML files with metadata
- Load and parse previously saved results

Result Structure:
    Each result is organized hierarchically:
    - Experiment metadata (robot, toolpath, pose, timestamp)
    - Summary statistics (reachability rate, singularity measures)
    - Detailed waypoint analysis (position, joint angles, quality metrics)

YAML Format:
    The results YAML file contains:
    - experiment_metadata: Information about this experiment
    - summary: Overall statistics
    - detailed_results: Per-waypoint analysis data

Usage:
    from results_handler import ExperimentResult, ResultsManager

    # Create a new result container
    result = ExperimentResult(
        robot_name="IRB-1300 900",
        toolpath_name="20250820_mc_HyperFree_AF1",
        pose_name="pose_1"
    )

    # Add analysis results
    result.add_summary(summary_dict)
    result.add_waypoint_results(waypoints_list)

    # Save to YAML
    manager = ResultsManager()
    manager.save_results(result, output_dir)
"""

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

class ExperimentResult:
    """
    Container for a single kinematic feasibility analysis experiment.

    Attributes:
        robot_name (str): Name of the robot model used
        toolpath_name (str): Name of the toolpath CSV file
        pose_name (str): Name of the knife pose configuration
        timestamp (datetime): When the experiment was executed
        summary (dict): Overall statistics and summary
        waypoint_results (list): Detailed per-waypoint analysis
        metadata (dict): Additional experiment metadata
    """

    def __init__(self, robot_name: str, toolpath_name: str, pose_name: str):
        """
        Initialize an ExperimentResult container.

        Args:
            robot_name (str): Name of the robot model
            toolpath_name (str): Name of the toolpath
            pose_name (str): Name of the knife pose
        """
        self.robot_name = robot_name
        self.toolpath_name = toolpath_name
        self.pose_name = pose_name
        self.timestamp = datetime.now()
        self.summary = {}
        self.waypoint_results = []
        self.metadata = {}

    def add_summary(self, summary_dict: Dict[str, Any]) -> None:
        """
        Add summary statistics to the result.

        Args:
            summary_dict (Dict[str, Any]): Dictionary containing summary statistics
                Expected keys:
                - total_waypoints: int
                - reachable_waypoints: int
                - reachability_rate: float (0-100%)
                - avg_manipulability: float
                - avg_min_singular_value: float
                - avg_condition_number: float
                - max_position_error_m: float
        """
        self.summary = summary_dict.copy()

    def add_waypoint_results(self, waypoints: List[Dict[str, Any]]) -> None:
        """
        Add detailed waypoint analysis results.

        Args:
            waypoints (List[Dict[str, Any]]): List of waypoint result dictionaries
                Each should contain: waypoint_index, x_m, y_m, z_m, reachable,
                manipulability, min_singular_value, condition_number, etc.
        """
        self.waypoint_results = waypoints.copy()

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata to the result.

        Args:
            key (str): Metadata key
            value (Any): Metadata value
        """
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to a dictionary suitable for YAML serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return {
            'experiment_metadata': {
                'robot_name': self.robot_name,
                'toolpath_name': self.toolpath_name,
                'pose_name': self.pose_name,
                'timestamp': self.timestamp.isoformat(),
            },
            'summary': self.summary,
            'detailed_results': self.waypoint_results,
            'additional_metadata': self.metadata
        }


# =============================================================================
# RESULTS MANAGER
# =============================================================================

class ResultsManager:
    """
    Manager for saving and loading experiment results in YAML format.

    This class handles the serialization of experiment results to YAML files,
    with support for both structured data and numpy array conversion.
    """

    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the ResultsManager.

        Args:
            results_dir (str, optional): Default directory for saving results.
                                        If None, uses 'results/' in current directory.
        """
        if results_dir is None:
            self.results_dir = os.path.join(os.getcwd(), 'results')
        else:
            self.results_dir = results_dir

    def save_results(self, experiment_result: ExperimentResult,
                    output_dir: str, filename: str = 'experiment_results.yaml') -> str:
        """
        Save an experiment result to a YAML file.

        Creates the output directory if it doesn't exist and saves the result
        with a standardized YAML structure.

        Args:
            experiment_result (ExperimentResult): The result to save
            output_dir (str): Directory where the file will be saved
            filename (str): Name of the YAML file (default: 'experiment_results.yaml')

        Returns:
            str: Full path to the saved file

        Raises:
            IOError: If the file cannot be written

        Example:
            >>> result = ExperimentResult("IRB-1300 900", "toolpath_1", "pose_1")
            >>> result.add_summary({'total_waypoints': 100, 'reachable_waypoints': 95})
            >>> manager = ResultsManager()
            >>> filepath = manager.save_results(result, 'output_dir')
            >>> print(f"Results saved to {filepath}")
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        try:
            result_dict = experiment_result.to_dict()

            # Convert to YAML-serializable format
            result_dict = self._make_yaml_serializable(result_dict)

            with open(output_path, 'w') as f:
                yaml.dump(result_dict, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

            print(f"Results saved to: {output_path}")
            return output_path

        except Exception as e:
            raise IOError(f"Error saving results to {output_path}: {e}")

    def load_results(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load experiment results from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file

        Returns:
            Dict[str, Any]: Loaded result dictionary

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValueError: If YAML parsing fails

        Example:
            >>> manager = ResultsManager()
            >>> results = manager.load_results('results/experiment_results.yaml')
        """
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Results file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                results = yaml.safe_load(f)
            return results

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")
        except Exception as e:
            raise Exception(f"Error reading results file {yaml_path}: {e}")

    def append_results(self, experiment_result: ExperimentResult,
                      yaml_path: str) -> str:
        """
        Append an experiment result to an existing YAML file.

        If the file doesn't exist, it creates it. If it exists, it appends
        the new result as a new entry in the results list.

        Args:
            experiment_result (ExperimentResult): The result to append
            yaml_path (str): Path to the YAML file

        Returns:
            str: Path to the YAML file

        Raises:
            IOError: If the file cannot be written

        Example:
            >>> manager = ResultsManager()
            >>> manager.append_results(result, 'results/all_results.yaml')
        """
        # Create directory if needed
        output_dir = os.path.dirname(yaml_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Load existing results or create new list
            if os.path.isfile(yaml_path):
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                if data is None:
                    results_list = []
                elif isinstance(data, list):
                    results_list = data
                else:
                    # If single result was saved, convert to list
                    results_list = [data]
            else:
                results_list = []

            # Append new result
            result_dict = experiment_result.to_dict()
            result_dict = self._make_yaml_serializable(result_dict)
            results_list.append(result_dict)

            # Save updated list
            with open(yaml_path, 'w') as f:
                yaml.dump(results_list, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

            print(f"Results appended to: {yaml_path}")
            return yaml_path

        except Exception as e:
            raise IOError(f"Error appending results to {yaml_path}: {e}")

    @staticmethod
    def _make_yaml_serializable(obj: Any) -> Any:
        """
        Convert numpy arrays and other non-serializable types to YAML-compatible formats.

        Args:
            obj (Any): Object to convert

        Returns:
            Any: YAML-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: ResultsManager._make_yaml_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultsManager._make_yaml_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For other types, try to convert to string
            try:
                return str(obj)
            except:
                return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_experiment_result_from_analysis(robot_name: str, toolpath_name: str,
                                          pose_name: str,
                                          analysis_results: List[Dict[str, Any]]) \
                                          -> ExperimentResult:
    """
    Create an ExperimentResult from trajectory analysis results.

    This is a convenience function that takes the output of the kinematic
    analysis and packages it into an ExperimentResult object.

    Args:
        robot_name (str): Name of the robot
        toolpath_name (str): Name of the toolpath
        pose_name (str): Name of the knife pose
        analysis_results (List[Dict[str, Any]]): Output from analyze_trajectory_kinematics()

    Returns:
        ExperimentResult: Structured result object

    Example:
        >>> results = analyze_trajectory_kinematics(model, data, trajectory)
        >>> exp_result = create_experiment_result_from_analysis(
        ...     "IRB-1300 900", "toolpath_1", "pose_1", results)
    """
    result = ExperimentResult(robot_name, toolpath_name, pose_name)

    # Calculate summary statistics
    total_waypoints = len(analysis_results)
    reachable_waypoints = sum(1 for r in analysis_results if r['reachable'])
    reachability_rate = (reachable_waypoints / total_waypoints * 100) if total_waypoints > 0 else 0

    summary = {
        'total_waypoints': total_waypoints,
        'reachable_waypoints': reachable_waypoints,
        'unreachable_waypoints': total_waypoints - reachable_waypoints,
        'reachability_rate_percent': reachability_rate,
    }

    # Add quality metrics if available
    reachable_results = [r for r in analysis_results if r['reachable']]
    if reachable_results:
        manip_values = [r['manipulability'] for r in reachable_results
                       if r.get('manipulability') is not None]
        min_sv_values = [r['min_singular_value'] for r in reachable_results
                        if r.get('min_singular_value') is not None]
        cond_values = [r['condition_number'] for r in reachable_results
                      if r.get('condition_number') is not None and r['condition_number'] != np.inf]
        position_errors = [r['position_error_m'] for r in reachable_results]

        if manip_values:
            summary['avg_manipulability'] = float(np.mean(manip_values))
            summary['min_manipulability'] = float(np.min(manip_values))
            summary['max_manipulability'] = float(np.max(manip_values))

        if min_sv_values:
            summary['avg_min_singular_value'] = float(np.mean(min_sv_values))
            summary['min_min_singular_value'] = float(np.min(min_sv_values))

        if cond_values:
            summary['avg_condition_number'] = float(np.mean(cond_values))
            summary['max_condition_number'] = float(np.max(cond_values))

        if position_errors:
            summary['avg_position_error_m'] = float(np.mean(position_errors))
            summary['max_position_error_m'] = float(np.max(position_errors))

    result.add_summary(summary)
    result.add_waypoint_results(analysis_results)

    return result


# Prevent direct execution of this module
if __name__ == "__main__":
    import sys
    print("This module is a library and should not be run directly.")
    print("Import it in other scripts to use its functionality.")
    sys.exit(1)
