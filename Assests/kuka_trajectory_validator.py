# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
KUKA IRB-1300 Trajectory Feasibility Validator

A modular, extensible trajectory validation system for the KUKA IRB-1300 robot.
Users can select specific checks or run all validation tests.

Features:
- Workspace reachability validation
- Inverse kinematics feasibility checking
- Joint limit violation detection
- Manipulability analysis
- Detailed reporting and visualization

Usage:
    # Run all checks
    python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks all
    
    # Run specific checks
    python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks workspace ik joint_limits
    
    # Run with visualization
    python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks all --visualize
"""

import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from isaaclab.app import AppLauncher

# ============================================================================
# ARGUMENT PARSING (Must happen before importing Isaac Sim modules)
# ============================================================================

parser = argparse.ArgumentParser(
    description="KUKA IRB-1300 Trajectory Feasibility Validator",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Available Checks:
  workspace      - Verify waypoints are within robot workspace
  ik             - Check inverse kinematics solvability (kinematic feasibility)
  joint_limits   - Validate joint position limits
  manipulability - Analyze manipulability at each waypoint (manipulability estimation)
  singularity    - Detect kinematic singularities (singularity checks)
  all            - Run all available checks (default)

Examples:
  # Run all checks
  python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks all
  
  # Run specific checks only
  python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks workspace ik
  
  # Run with visualization
  python Assests/kuka_trajectory_validator.py --csv trajectory.csv --checks all --visualize
    """
)

parser.add_argument(
    "--csv",
    type=str,
    required=True,
    help="Path to CSV trajectory file (position + quaternion: x,y,z,qw,qx,qy,qz)"
)
parser.add_argument(
    "--checks",
    type=str,
    nargs='+',
    default=["all"],
    choices=["all", "workspace", "ik", "joint_limits", "manipulability", "singularity"],
    help="Which feasibility checks to run (default: all)"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Visualize trajectory after validation"
)
parser.add_argument(
    "--report_file",
    type=str,
    default=None,
    help="Output report filename (default: auto-generated)"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments (default: 1)"
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim initialization."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

# Import robot configuration
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from irb_1300_cfg import IRB_1300_900_CFG

# ============================================================================
# ROBOT CONFIGURATION
# ============================================================================

# Joint configuration for KUKA IRB-1300
JOINT_NAMES = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]

# Joint limits (from URDF)
JOINT_LIMITS = {
    "Joint_1": (-3.142, 3.142),      # ±180°
    "Joint_2": (-1.745, 2.269),      # -100° to 130°
    "Joint_3": (-3.665, 1.134),      # -210° to 65°
    "Joint_4": (-4.014, 4.014),      # ±230°
    "Joint_5": (-2.269, 2.269),      # ±130°
    "Joint_6": (-6.981, 6.981),      # ±400°
}

# End effector link name (renamed from ee_link to ee_pose in USD)
EE_LINK_NAME = "ee_pose"

# Home position for robot (safe starting pose)
DEFAULT_HOME_POSE = torch.tensor(
    [0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
    dtype=torch.float32
)

# Workspace limits (approximate for IRB-1300/0.9)
WORKSPACE_MAX_REACH = 0.9  # 900mm reach
WORKSPACE_MIN_REACH = 0.15  # Minimum reach (avoid base singularity)

# Visualization colors
COLOR_FEASIBLE = (0.0, 1.0, 0.0)           # Green
COLOR_INFEASIBLE = (1.0, 0.0, 0.0)         # Red
COLOR_OUT_OF_WORKSPACE = (0.0, 0.0, 0.0)   # Black

# Output directory configuration
def get_output_dir():
    """Get or create output directory for validation results."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results", "validation_test")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WaypointResult:
    """Results from validating a single waypoint."""
    idx: int
    feasible: bool = True
    
    # Workspace check results
    workspace_valid: Optional[bool] = None
    workspace_distance: Optional[float] = None
    
    # IK check results
    ik_valid: Optional[bool] = None
    ik_error: Optional[str] = None
    
    # Joint limit check results
    joint_limits_valid: Optional[bool] = None
    violated_joints: List[str] = field(default_factory=list)
    joint_positions: Optional[List[float]] = None  # Store actual joint values
    
    # Manipulability results
    manipulability: Optional[float] = None
    
    # Singularity check results
    singularity_detected: Optional[bool] = None
    singularity_type: Optional[str] = None
    condition_number: Optional[float] = None
    
    def get_failure_reasons(self) -> List[str]:
        """Get list of reasons why this waypoint is infeasible."""
        reasons = []
        if self.workspace_valid is False:
            reasons.append(f"Out-of-Workspace (dist={self.workspace_distance:.3f}m)")
        if self.ik_valid is False:
            reasons.append(f"IK Failed: {self.ik_error}")
        if self.joint_limits_valid is False:
            reasons.append(f"Joint Limits Violated: {self.violated_joints}")
        if self.singularity_detected:
            reasons.append(f"Singularity: {self.singularity_type} (cond={self.condition_number:.2f})")
        return reasons


# ============================================================================
# BASE CHECKER CLASS
# ============================================================================

class BaseChecker(ABC):
    """Abstract base class for all feasibility checkers."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """
        Check feasibility for a single waypoint.
        
        Args:
            waypoint: Waypoint pose (x, y, z, qw, qx, qy, qz)
            result: WaypointResult object to update with check results
            **kwargs: Additional context (robot state, jacobian, etc.)
        
        Returns:
            True if check passed, False otherwise
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"


# ============================================================================
# WORKSPACE CHECKER
# ============================================================================

class WorkspaceChecker(BaseChecker):
    """Validates waypoints are within robot workspace."""
    
    def __init__(self, max_reach: float = WORKSPACE_MAX_REACH, 
                 min_reach: float = WORKSPACE_MIN_REACH, enabled: bool = True):
        super().__init__("Workspace Reachability", enabled)
        self.max_reach = max_reach
        self.min_reach = min_reach
    
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """Check if waypoint is within reachable workspace."""
        pos = waypoint[:3]
        distance = torch.norm(pos).item()
        
        result.workspace_distance = distance
        result.workspace_valid = self.min_reach <= distance <= self.max_reach
        
        if not result.workspace_valid:
            result.feasible = False
        
        return result.workspace_valid


# ============================================================================
# IK CHECKER
# ============================================================================

class IKChecker(BaseChecker):
    """Validates inverse kinematics can be solved for waypoints."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("Inverse Kinematics", enabled)
    
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """Check if IK can be computed for this waypoint."""
        try:
            # IK computation happens in the main loop
            # This checker validates the results
            joint_positions = kwargs.get("joint_positions")
            
            if joint_positions is None:
                raise ValueError("IK computation failed")
            
            # Check for NaN or invalid values
            if torch.isnan(joint_positions).any() or torch.isinf(joint_positions).any():
                raise ValueError("IK produced invalid joint values")
            
            result.ik_valid = True
            return True
            
        except Exception as e:
            result.ik_valid = False
            result.ik_error = str(e)
            result.feasible = False
            return False


# ============================================================================
# JOINT LIMIT CHECKER
# ============================================================================

class JointLimitChecker(BaseChecker):
    """Validates joint positions are within limits."""
    
    def __init__(self, joint_limits: Dict[str, Tuple[float, float]] = JOINT_LIMITS, 
                 enabled: bool = True):
        super().__init__("Joint Limits", enabled)
        self.joint_limits = joint_limits
    
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """Check if joint positions violate limits."""
        joint_positions = kwargs.get("joint_positions")
        
        if joint_positions is None:
            result.joint_limits_valid = False
            result.feasible = False
            return False
        
        # Store actual joint positions for plotting
        result.joint_positions = [joint_positions[i].item() for i in range(len(JOINT_NAMES))]
        
        violated_joints = []
        for i, joint_name in enumerate(JOINT_NAMES):
            joint_val = joint_positions[i].item()
            min_limit, max_limit = self.joint_limits[joint_name]
            
            if joint_val < min_limit or joint_val > max_limit:
                violated_joints.append(f"{joint_name}({joint_val:.3f} rad)")
        
        result.violated_joints = violated_joints
        result.joint_limits_valid = len(violated_joints) == 0
        
        if not result.joint_limits_valid:
            result.feasible = False
        
        return result.joint_limits_valid


# ============================================================================
# MANIPULABILITY CHECKER
# ============================================================================

class ManipulabilityChecker(BaseChecker):
    """Computes manipulability measure (Yoshikawa's measure)."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("Manipulability", enabled)
    
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """Compute manipulability for this configuration."""
        jacobian = kwargs.get("jacobian")
        
        if jacobian is None:
            result.manipulability = 0.0
            return True  # Manipulability is informational, not a pass/fail
        
        result.manipulability = self._compute_manipulability(jacobian)
        return True
    
    @staticmethod
    def _compute_manipulability(jacobian: torch.Tensor) -> float:
        """
        Compute Yoshikawa's manipulability measure.
        
        m = sqrt(det(J * J^T))
        
        Higher values indicate better manipulability.
        """
        J = jacobian.cpu().numpy()
        
        try:
            if J.shape[0] == J.shape[1]:
                # Square Jacobian
                return float(np.sqrt(abs(np.linalg.det(J @ J.T))))
            else:
                # Non-square Jacobian, use condition number reciprocal
                return float(1.0 / np.linalg.cond(J))
        except:
            return 0.0


# ============================================================================
# SINGULARITY CHECKER
# ============================================================================

class SingularityChecker(BaseChecker):
    """Detects kinematic singularities using Jacobian analysis."""
    
    # Singularity thresholds
    CRITICAL_CONDITION_NUMBER = 100.0  # Above this is critical singularity
    WARNING_CONDITION_NUMBER = 30.0    # Above this is near-singularity
    CRITICAL_MANIPULABILITY = 0.001    # Below this is critical singularity
    WARNING_MANIPULABILITY = 0.005     # Below this is near-singularity
    
    def __init__(self, condition_threshold: float = WARNING_CONDITION_NUMBER,
                 manipulability_threshold: float = WARNING_MANIPULABILITY,
                 enabled: bool = True):
        super().__init__("Singularity Detection", enabled)
        self.condition_threshold = condition_threshold
        self.manipulability_threshold = manipulability_threshold
    
    def check(self, waypoint: torch.Tensor, result: WaypointResult, **kwargs) -> bool:
        """Detect singularities using Jacobian condition number and manipulability."""
        jacobian = kwargs.get("jacobian")
        joint_positions = kwargs.get("joint_positions")
        
        if jacobian is None:
            result.singularity_detected = None
            return True
        
        # Compute condition number
        J = jacobian.cpu().numpy()
        try:
            cond_num = np.linalg.cond(J)
            result.condition_number = float(cond_num)
        except:
            cond_num = np.inf
            result.condition_number = float('inf')
        
        # Get manipulability if available
        manip = result.manipulability if result.manipulability is not None else self._compute_manipulability(jacobian)
        
        # Detect singularity type
        singularity_type = self._identify_singularity_type(joint_positions, cond_num, manip)
        
        if singularity_type:
            result.singularity_detected = True
            result.singularity_type = singularity_type
            # Singularities are warnings, not hard failures
            # result.feasible = False  # Uncomment to make singularities fail the waypoint
            return False
        else:
            result.singularity_detected = False
            result.singularity_type = None
            return True
    
    def _identify_singularity_type(self, joint_positions: Optional[torch.Tensor], 
                                   cond_num: float, manip: float) -> Optional[str]:
        """
        Identify type of singularity based on joint configuration and Jacobian properties.
        
        Returns:
            String describing singularity type, or None if no singularity detected
        """
        # Check for critical singularity (very bad)
        if cond_num > self.CRITICAL_CONDITION_NUMBER or manip < self.CRITICAL_MANIPULABILITY:
            severity = "CRITICAL"
        # Check for near-singularity (warning)
        elif cond_num > self.condition_threshold or manip < self.manipulability_threshold:
            severity = "WARNING"
        else:
            return None  # No singularity
        
        # Identify specific singularity type based on joint configuration
        if joint_positions is not None:
            joint_vals = joint_positions.cpu().numpy()
            
            # Wrist singularity: Joint 5 near 0 or ±π
            if abs(joint_vals[4]) < 0.1 or abs(abs(joint_vals[4]) - np.pi) < 0.1:
                return f"{severity} Wrist Singularity"
            
            # Shoulder singularity: Joint 2 and 3 aligned
            if abs(joint_vals[1] + joint_vals[2]) < 0.15:  # Joints 2 and 3 cancel out
                return f"{severity} Shoulder Singularity"
            
            # Elbow singularity: Joint 3 near limits
            if abs(joint_vals[2]) < 0.15 or abs(joint_vals[2] - 1.134) < 0.15:
                return f"{severity} Elbow Singularity"
        
        # Generic singularity if we can't identify specific type
        return f"{severity} Generic Singularity"
    
    @staticmethod
    def _compute_manipulability(jacobian: torch.Tensor) -> float:
        """Compute manipulability measure (same as ManipulabilityChecker)."""
        J = jacobian.cpu().numpy()
        try:
            if J.shape[0] == J.shape[1]:
                return float(np.sqrt(abs(np.linalg.det(J @ J.T))))
            else:
                return float(1.0 / np.linalg.cond(J))
        except:
            return 0.0


# ============================================================================
# FEASIBILITY REPORT
# ============================================================================

class FeasibilityReport:
    """Generates and manages feasibility validation reports."""
    
    def __init__(self, csv_file: str, num_waypoints: int, enabled_checks: List[str]):
        self.csv_file = csv_file
        self.num_waypoints = num_waypoints
        self.enabled_checks = enabled_checks
        self.timestamp = datetime.now()
        self.results: List[WaypointResult] = []
        self.output_dir = get_output_dir()
    
    def add_result(self, result: WaypointResult):
        """Add a waypoint result to the report."""
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        """Compute summary statistics."""
        summary = {
            "total_waypoints": self.num_waypoints,
            "feasible": sum(1 for r in self.results if r.feasible),
            "infeasible": sum(1 for r in self.results if not r.feasible),
        }
        
        if "workspace" in self.enabled_checks or "all" in self.enabled_checks:
            summary["workspace_violations"] = sum(
                1 for r in self.results if r.workspace_valid is False
            )
        
        if "ik" in self.enabled_checks or "all" in self.enabled_checks:
            summary["ik_failures"] = sum(
                1 for r in self.results if r.ik_valid is False
            )
        
        if "joint_limits" in self.enabled_checks or "all" in self.enabled_checks:
            summary["joint_limit_violations"] = sum(
                1 for r in self.results if r.joint_limits_valid is False
            )
        
        if "manipulability" in self.enabled_checks or "all" in self.enabled_checks:
            manip_values = [r.manipulability for r in self.results if r.manipulability is not None]
            if manip_values:
                summary["manipulability_min"] = min(manip_values)
                summary["manipulability_max"] = max(manip_values)
                summary["manipulability_mean"] = np.mean(manip_values)
        
        if "singularity" in self.enabled_checks or "all" in self.enabled_checks:
            summary["singularity_detections"] = sum(
                1 for r in self.results if r.singularity_detected is True
            )
            cond_numbers = [r.condition_number for r in self.results 
                          if r.condition_number is not None and r.condition_number != float('inf')]
            if cond_numbers:
                summary["condition_number_max"] = max(cond_numbers)
                summary["condition_number_mean"] = np.mean(cond_numbers)
        
        return summary
    
    def print_summary(self):
        """Print report summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("KUKA IRB-1300 TRAJECTORY FEASIBILITY REPORT")
        print("=" * 80)
        print(f"\nTrajectory: {self.csv_file}")
        print(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Enabled Checks: {', '.join(self.enabled_checks)}")
        
        print(f"\n{'Total Waypoints:':<30} {summary['total_waypoints']}")
        feasibility_percent = 100 * summary['feasible'] / summary['total_waypoints'] if summary['total_waypoints'] > 0 else 0
        print(f"{'Feasible Waypoints:':<30} {summary['feasible']} ({feasibility_percent:.1f}%)")
        print(f"{'Infeasible Waypoints:':<30} {summary['infeasible']}")
        
        # Detailed breakdown
        print("\n" + "-" * 80)
        print("INFEASIBILITY BREAKDOWN")
        print("-" * 80)
        
        if "workspace_violations" in summary:
            print(f"{'Out-of-Workspace:':<30} {summary['workspace_violations']}")
        if "ik_failures" in summary:
            print(f"{'IK Failures:':<30} {summary['ik_failures']}")
        if "joint_limit_violations" in summary:
            print(f"{'Joint Limit Violations:':<30} {summary['joint_limit_violations']}")
        if "singularity_detections" in summary:
            print(f"{'Singularity Detections:':<30} {summary['singularity_detections']}")
        
        # Detailed failures (first 20)
        failures = [r for r in self.results if not r.feasible]
        if failures:
            print("\n" + "-" * 80)
            print("DETAILED FAILURES (First 20)")
            print("-" * 80)
            
            for i, result in enumerate(failures[:20]):
                reasons = result.get_failure_reasons()
                print(f"  Waypoint {result.idx:4d}: {', '.join(reasons)}")
            
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more failures")
        
        # Manipulability analysis
        if "manipulability_min" in summary:
            print("\n" + "-" * 80)
            print("MANIPULABILITY ANALYSIS")
            print("-" * 80)
            print(f"{'Min:':<30} {summary['manipulability_min']:.6f}")
            print(f"{'Max:':<30} {summary['manipulability_max']:.6f}")
            print(f"{'Mean:':<30} {summary['manipulability_mean']:.6f}")
            print("\nWhat is Manipulability?")
            print("  - Measure of how well the robot can move in all directions")
            print("  - Computed from Jacobian: m = sqrt(det(J*J^T))")
            print("  - Higher values = better maneuverability")
            print("  - Lower values = closer to singularities")
        
        # Singularity analysis
        if "singularity_detections" in summary:
            print("\n" + "-" * 80)
            print("SINGULARITY ANALYSIS")
            print("-" * 80)
            print(f"{'Singularities Detected:':<30} {summary['singularity_detections']}")
            if "condition_number_max" in summary:
                print(f"{'Max Condition Number:':<30} {summary['condition_number_max']:.2f}")
                print(f"{'Mean Condition Number:':<30} {summary['condition_number_mean']:.2f}")
            print("\nWhat are Singularities?")
            print("  - Configurations where robot loses degrees of freedom")
            print("  - Detected via Jacobian condition number and joint configuration")
            print("  - Types: Wrist, Shoulder, Elbow singularities")
            print("  - Should be avoided for smooth motion")
        
        print("=" * 80)
    
    def generate_plots(self):
        """Generate visualization plots for each enabled test."""
        print(f"\n[INFO] Generating validation plots...")
        
        waypoint_indices = list(range(len(self.results)))
        plot_configs = []
        
        # Determine which plots to create based on enabled checks
        run_all = "all" in self.enabled_checks
        
        if run_all or "workspace" in self.enabled_checks:
            plot_configs.append(("workspace", "Distance from Base (m)", "Workspace Reachability"))
        if run_all or "ik" in self.enabled_checks:
            plot_configs.append(("ik", "IK Success", "Inverse Kinematics Feasibility"))
        if run_all or "joint_limits" in self.enabled_checks:
            plot_configs.append(("joint_limits", "Max Joint Violation (rad)", "Joint Limit Violations"))
        if run_all or "manipulability" in self.enabled_checks:
            plot_configs.append(("manipulability", "Manipulability Index", "Manipulability Analysis"))
        if run_all or "singularity" in self.enabled_checks:
            plot_configs.append(("singularity", "Condition Number", "Singularity Detection"))
        
        if not plot_configs:
            print("[WARN] No plots to generate (no checks enabled)")
            return
        
        # Create individual plots for each test
        for test_name, ylabel, title in plot_configs:
            self._create_individual_plot(test_name, ylabel, title, waypoint_indices)
        
        # Create combined overview plot
        if len(plot_configs) > 1:
            self._create_combined_plot(plot_configs, waypoint_indices)
        
        print(f"[INFO] Plots saved to: {self.output_dir}")
    
    def _create_individual_plot(self, test_name: str, ylabel: str, title: str, waypoint_indices: list):
        """Create individual plot for a specific test."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if test_name == "workspace":
            distances = []
            colors = []
            for result in self.results:
                if result.workspace_distance is not None:
                    distances.append(result.workspace_distance)
                    colors.append('green' if result.workspace_valid else 'red')
                else:
                    distances.append(0)
                    colors.append('gray')
            
            ax.scatter(waypoint_indices, distances, c=colors, alpha=0.6, s=30)
            ax.axhline(y=WORKSPACE_MAX_REACH, color='orange', linestyle='--', linewidth=2, 
                      label=f'Max Reach ({WORKSPACE_MAX_REACH}m)')
            ax.axhline(y=WORKSPACE_MIN_REACH, color='blue', linestyle='--', linewidth=2,
                      label=f'Min Reach ({WORKSPACE_MIN_REACH}m)')
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(fontsize=10)
            
        elif test_name == "ik":
            ik_success = []
            colors = []
            for result in self.results:
                if result.ik_valid is not None:
                    ik_success.append(1 if result.ik_valid else 0)
                    colors.append('green' if result.ik_valid else 'red')
                else:
                    ik_success.append(0.5)
                    colors.append('gray')
            
            ax.scatter(waypoint_indices, ik_success, c=colors, alpha=0.6, s=30)
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Failed', 'Success'], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            
        elif test_name == "joint_limits":
            # Create 3x2 grid for all 6 joints (left: joints 1-3, right: joints 4-6)
            fig_joints = plt.figure(figsize=(18, 14))
            
            for joint_idx, joint_name in enumerate(JOINT_NAMES):
                ax_joint = plt.subplot(3, 2, joint_idx + 1)
                
                # Extract joint values for this specific joint
                joint_values = []
                line_colors = []  # Colors for each segment
                
                for result in self.results:
                    if result.joint_positions is not None and len(result.joint_positions) > joint_idx:
                        joint_val = result.joint_positions[joint_idx]
                        joint_values.append(joint_val)
                        
                        # Determine if this specific joint is in violation
                        min_limit, max_limit = JOINT_LIMITS[joint_name]
                        if joint_val < min_limit or joint_val > max_limit:
                            line_colors.append('red')
                        else:
                            line_colors.append('green')
                    else:
                        joint_values.append(0)
                        line_colors.append('gray')
                
                # Get limits for this joint
                min_limit, max_limit = JOINT_LIMITS[joint_name]
                
                # Plot the joint trajectory with color segments
                for i in range(len(waypoint_indices) - 1):
                    ax_joint.plot(waypoint_indices[i:i+2], joint_values[i:i+2], 
                                 color=line_colors[i], linewidth=2, alpha=0.8)
                
                # Add limit lines
                ax_joint.axhline(y=max_limit, color='darkred', linestyle='--', linewidth=2, 
                               label=f'Max Limit ({np.rad2deg(max_limit):.1f}°)', zorder=10)
                ax_joint.axhline(y=min_limit, color='darkred', linestyle='--', linewidth=2, 
                               label=f'Min Limit ({np.rad2deg(min_limit):.1f}°)', zorder=10)
                
                # Add safe zone (light green background between limits)
                ax_joint.fill_between(waypoint_indices, min_limit, max_limit, 
                                     alpha=0.1, color='green', label='Safe Zone')
                
                # Formatting
                ax_joint.set_ylabel('Joint Angle (rad)', fontsize=11, fontweight='bold')
                ax_joint.set_title(f'{joint_name} ({np.rad2deg(min_limit):.1f}° to {np.rad2deg(max_limit):.1f}°)', 
                                  fontsize=12, fontweight='bold')
                ax_joint.legend(fontsize=9, loc='best')
                ax_joint.grid(True, alpha=0.3, linestyle=':')
                
                # Add secondary y-axis with degrees
                ax2 = ax_joint.secondary_yaxis('right', functions=(np.rad2deg, np.deg2rad))
                ax2.set_ylabel('Angle (deg)', fontsize=10)
                
                if joint_idx >= 4:  # Bottom row
                    ax_joint.set_xlabel('Waypoint Index', fontsize=11, fontweight='bold')
            
            # Add legend for line colors
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='green', lw=2),
                          Line2D([0], [0], color='red', lw=2)]
            fig_joints.legend(custom_lines, ['Within Limits', 'Violation'], 
                            loc='upper center', ncol=2, fontsize=12, 
                            bbox_to_anchor=(0.5, 0.98))
            
            plt.suptitle(f'Joint Limit Analysis (All 6 Joints) - {os.path.basename(self.csv_file)}', 
                        fontsize=15, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Save this special joint limits figure
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            plot_filename = os.path.join(self.output_dir, f'joint_limits_analysis_{timestamp_str}.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig_joints)
            print(f"  - Saved: {os.path.basename(plot_filename)}")
            
            # Skip the regular single-axis plot for joint_limits - we made a custom one
            return
            
        elif test_name == "manipulability":
            manip_values = []
            for result in self.results:
                if result.manipulability is not None and result.manipulability > 0:
                    manip_values.append(result.manipulability)
                else:
                    manip_values.append(1e-10)  # Small value for log scale
            
            ax.plot(waypoint_indices, manip_values, linewidth=1.5, alpha=0.8, color='blue', label='Manipulability')
            ax.fill_between(waypoint_indices, manip_values, alpha=0.3, color='blue')
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
        elif test_name == "singularity":
            cond_numbers = []
            colors = []
            for result in self.results:
                if result.condition_number is not None and result.condition_number != float('inf'):
                    cond_numbers.append(result.condition_number)
                    if result.singularity_detected:
                        colors.append('red')
                    else:
                        colors.append('green')
                else:
                    cond_numbers.append(1000)  # High value for inf
                    colors.append('gray')
            
            ax.scatter(waypoint_indices, cond_numbers, c=colors, alpha=0.6, s=30)
            ax.axhline(y=SingularityChecker.WARNING_CONDITION_NUMBER, color='orange', 
                      linestyle='--', linewidth=2, label=f'Warning Threshold ({SingularityChecker.WARNING_CONDITION_NUMBER})')
            ax.axhline(y=SingularityChecker.CRITICAL_CONDITION_NUMBER, color='red', 
                      linestyle='--', linewidth=2, label=f'Critical Threshold ({SingularityChecker.CRITICAL_CONDITION_NUMBER})')
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_yscale('log')
            ax.legend(fontsize=10)
        
        ax.set_xlabel('Waypoint Index', fontsize=12)
        ax.set_title(f'{title} - {os.path.basename(self.csv_file)}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.output_dir, f'{test_name}_analysis_{timestamp_str}.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved: {os.path.basename(plot_filename)}")
    
    def _create_combined_plot(self, plot_configs: list, waypoint_indices: list):
        """Create combined overview plot with all tests."""
        n_plots = len(plot_configs)
        if n_plots == 0:
            return
        
        fig = plt.figure(figsize=(16, 3.5 * n_plots))
        gs = GridSpec(n_plots, 1, figure=fig, hspace=0.35)
        
        for idx, (test_name, ylabel, title) in enumerate(plot_configs):
            ax = fig.add_subplot(gs[idx, 0])
            
            if test_name == "workspace":
                distances = [r.workspace_distance if r.workspace_distance is not None else 0 
                           for r in self.results]
                colors = ['green' if r.workspace_valid else 'red' if r.workspace_valid is False else 'gray'
                         for r in self.results]
                ax.scatter(waypoint_indices, distances, c=colors, alpha=0.6, s=20)
                ax.axhline(y=WORKSPACE_MAX_REACH, color='orange', linestyle='--', linewidth=1.5, 
                          label=f'Max ({WORKSPACE_MAX_REACH}m)')
                ax.axhline(y=WORKSPACE_MIN_REACH, color='blue', linestyle='--', linewidth=1.5,
                          label=f'Min ({WORKSPACE_MIN_REACH}m)')
                ax.legend(fontsize=8)
                
            elif test_name == "ik":
                ik_success = [1 if r.ik_valid else 0 if r.ik_valid is False else 0.5 
                             for r in self.results]
                colors = ['green' if s == 1 else 'red' if s == 0 else 'gray' for s in ik_success]
                ax.scatter(waypoint_indices, ik_success, c=colors, alpha=0.6, s=20)
                ax.set_ylim(-0.1, 1.1)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Failed', 'Success'], fontsize=9)
                
            elif test_name == "joint_limits":
                # Simplified visualization for combined plot - show overall violation status
                violations = [0 if r.joint_limits_valid else 1 if r.joint_limits_valid is False else 0.5
                            for r in self.results]
                colors = ['green' if v == 0 else 'red' if v == 1 else 'gray' for v in violations]
                ax.scatter(waypoint_indices, violations, c=colors, alpha=0.6, s=20)
                ax.set_ylim(-0.1, 1.1)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Valid', 'Violated'], fontsize=9)
                ax.text(0.5, 0.95, 'See detailed 6-joint graph for per-joint analysis', 
                       transform=ax.transAxes, fontsize=8, ha='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            elif test_name == "manipulability":
                manip_values = [r.manipulability if r.manipulability is not None and r.manipulability > 0 else 1e-10 
                              for r in self.results]
                ax.plot(waypoint_indices, manip_values, linewidth=1.2, alpha=0.7, color='blue')
                ax.fill_between(waypoint_indices, manip_values, alpha=0.3, color='blue')
                ax.set_yscale('log')
                
            elif test_name == "singularity":
                cond_numbers = [r.condition_number if r.condition_number is not None and r.condition_number != float('inf') else 1000
                              for r in self.results]
                colors = ['red' if r.singularity_detected else 'green' if r.singularity_detected is False else 'gray'
                         for r in self.results]
                ax.scatter(waypoint_indices, cond_numbers, c=colors, alpha=0.6, s=15)
                ax.axhline(y=SingularityChecker.WARNING_CONDITION_NUMBER, color='orange', 
                          linestyle='--', linewidth=1, label='Warning')
                ax.axhline(y=SingularityChecker.CRITICAL_CONDITION_NUMBER, color='red',
                          linestyle='--', linewidth=1, label='Critical')
                ax.set_yscale('log')
                ax.legend(fontsize=7)
            
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if idx == n_plots - 1:
                ax.set_xlabel('Waypoint Index', fontsize=10)
        
        plt.suptitle(f'Trajectory Validation Summary - {os.path.basename(self.csv_file)}', 
                    fontsize=13, fontweight='bold', y=0.995)
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        combined_filename = os.path.join(self.output_dir, f'combined_validation_{timestamp_str}.png')
        plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - Saved: {os.path.basename(combined_filename)}")
    
    def save_to_file(self, filename: str):
        """Save detailed report to file."""
        summary = self.get_summary()
        
        # Save to output directory
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        report_filename = f"{base_name}_{timestamp_str}.txt"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KUKA IRB-1300 TRAJECTORY FEASIBILITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Trajectory: {self.csv_file}\n")
            f.write(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Enabled Checks: {', '.join(self.enabled_checks)}\n\n")
            
            feasibility_percent = 100 * summary['feasible'] / summary['total_waypoints'] if summary['total_waypoints'] > 0 else 0
            f.write(f"Total Waypoints:         {summary['total_waypoints']}\n")
            f.write(f"Feasible Waypoints:      {summary['feasible']} ({feasibility_percent:.1f}%)\n")
            f.write(f"Infeasible Waypoints:    {summary['infeasible']}\n\n")
            
            # Breakdown
            f.write("-" * 80 + "\n")
            f.write("INFEASIBILITY BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            if "workspace_violations" in summary:
                f.write(f"Out-of-Workspace:        {summary['workspace_violations']}\n")
            if "ik_failures" in summary:
                f.write(f"IK Failures:             {summary['ik_failures']}\n")
            if "joint_limit_violations" in summary:
                f.write(f"Joint Limit Violations:  {summary['joint_limit_violations']}\n")
            
            # Detailed waypoint results
            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED WAYPOINT RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for result in self.results:
                f.write(f"Waypoint {result.idx}:\n")
                f.write(f"  Feasible: {result.feasible}\n")
                
                if result.workspace_distance is not None:
                    f.write(f"  Workspace Distance: {result.workspace_distance:.3f}m ")
                    f.write(f"({'VALID' if result.workspace_valid else 'INVALID'})\n")
                
                if result.ik_valid is not None:
                    f.write(f"  IK: {'VALID' if result.ik_valid else 'INVALID'}\n")
                    if not result.ik_valid and result.ik_error:
                        f.write(f"    Error: {result.ik_error}\n")
                
                if result.joint_limits_valid is not None:
                    f.write(f"  Joint Limits: {'VALID' if result.joint_limits_valid else 'INVALID'}\n")
                    if result.violated_joints:
                        f.write(f"    Violated: {result.violated_joints}\n")
                
                if result.manipulability is not None:
                    f.write(f"  Manipulability: {result.manipulability:.6f}\n")
                
                if result.singularity_detected is not None:
                    f.write(f"  Singularity: {'DETECTED' if result.singularity_detected else 'None'}\n")
                    if result.singularity_detected and result.singularity_type:
                        f.write(f"    Type: {result.singularity_type}\n")
                    if result.condition_number is not None:
                        f.write(f"    Condition Number: {result.condition_number:.2f}\n")
                
                f.write("\n")
            
            # Manipulability summary
            if "manipulability_min" in summary:
                f.write("-" * 80 + "\n")
                f.write("MANIPULABILITY ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Min:  {summary['manipulability_min']:.6f}\n")
                f.write(f"Max:  {summary['manipulability_max']:.6f}\n")
                f.write(f"Mean: {summary['manipulability_mean']:.6f}\n\n")
            
            # Singularity summary
            if "singularity_detections" in summary:
                f.write("-" * 80 + "\n")
                f.write("SINGULARITY ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Singularities Detected: {summary['singularity_detections']}\n")
                if "condition_number_max" in summary:
                    f.write(f"Max Condition Number:   {summary['condition_number_max']:.2f}\n")
                    f.write(f"Mean Condition Number:  {summary['condition_number_mean']:.2f}\n\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"[INFO] Report saved to: {report_path}")


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class KukaSceneCfg(InteractiveSceneCfg):
    """Scene configuration for KUKA IRB-1300 validation (kinematic mode)."""
    
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0)
        ),
    )
    
    # Robot in KINEMATIC mode (no physics, just kinematics for validation)
    robot = IRB_1300_900_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=IRB_1300_900_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,  # KINEMATIC MODE: not affected by physics
            ),
        ),
    )


# ============================================================================
# TRAJECTORY VALIDATOR
# ============================================================================

class TrajectoryValidator:
    """Main orchestrator for trajectory validation."""
    
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene,
                 enabled_checks: List[str]):
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.enabled_checks = enabled_checks
        
        # Initialize checkers
        self.checkers = self._initialize_checkers()
        
        # Setup IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=scene.num_envs, device=sim.device
        )
        
        # Setup robot entity configuration
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["Joint_[1-6]"],
            body_names=[EE_LINK_NAME]
        )
        self.robot_entity_cfg.resolve(scene)
        
        # Get end effector Jacobian index
        self.ee_jacobi_idx = (
            self.robot_entity_cfg.body_ids[0] - 1
            if self.robot.is_fixed_base
            else self.robot_entity_cfg.body_ids[0]
        )
        
        print(f"[INFO] Initialized {len(self.checkers)} checkers:")
        for checker in self.checkers:
            print(f"  - {checker}")
    
    def _initialize_checkers(self) -> List[BaseChecker]:
        """Initialize enabled checkers based on user selection."""
        checkers = []
        run_all = "all" in self.enabled_checks
        
        if run_all or "workspace" in self.enabled_checks:
            checkers.append(WorkspaceChecker())
        
        if run_all or "ik" in self.enabled_checks:
            checkers.append(IKChecker())
        
        if run_all or "joint_limits" in self.enabled_checks:
            checkers.append(JointLimitChecker())
        
        if run_all or "manipulability" in self.enabled_checks:
            checkers.append(ManipulabilityChecker())
        
        if run_all or "singularity" in self.enabled_checks:
            checkers.append(SingularityChecker())
        
        return checkers
    
    def load_trajectory(self, csv_file: str) -> torch.Tensor:
        """Load trajectory from CSV file."""
        print(f"\n[INFO] Loading trajectory from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Take only the first 7 columns if more are present
            if df.shape[1] > 7:
                print(f"[INFO] CSV has {df.shape[1]} columns, using first 7 columns")
                df = df.iloc[:, :7]
            elif df.shape[1] < 7:
                raise ValueError(f"Expected at least 7 columns, got {df.shape[1]}")
            
            trajectory = torch.tensor(df.values, dtype=torch.float32, device=self.sim.device)
            print(f"[INFO] Loaded {len(trajectory)} waypoints")
            print(f"[INFO] Expected format: x, y, z, qw, qx, qy, qz (7 columns)")
            
            return trajectory
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            raise
    
    def move_to_home(self):
        """Move robot to home position (kinematic mode - no physics)."""
        print(f"[INFO] Moving to home position (kinematic mode)...")
        
        ik_commands = torch.zeros(
            self.scene.num_envs, self.ik_controller.action_dim, device=self.robot.device
        )
        home_pose = DEFAULT_HOME_POSE.to(self.robot.device)
        ik_commands[:, 0:7] = home_pose.unsqueeze(0).expand(self.scene.num_envs, -1)
        self.ik_controller.set_command(ik_commands)
        
        sim_dt = self.sim.get_physics_dt()
        
        # Reduced iterations since physics is disabled (kinematic mode)
        for _ in range(10):  # Was 100, now 10 - much faster
            self._perform_ik_step(sim_dt)
    
    def validate_trajectory(self, csv_file: str) -> FeasibilityReport:
        """
        Validate entire trajectory.
        
        Args:
            csv_file: Path to CSV trajectory file
        
        Returns:
            FeasibilityReport with validation results
        """
        trajectory = self.load_trajectory(csv_file)
        report = FeasibilityReport(csv_file, len(trajectory), self.enabled_checks)
        
        # Move to home position first
        self.move_to_home()
        
        print(f"[INFO] Validating trajectory with {len(self.checkers)} checks...")
        print(f"[INFO] Running in KINEMATIC MODE (no physics simulation)")
        print()
        
        sim_dt = self.sim.get_physics_dt()
        ik_commands = torch.zeros(
            self.scene.num_envs, self.ik_controller.action_dim, device=self.robot.device
        )
        
        for waypoint_idx in range(len(trajectory)):
            if not simulation_app.is_running():
                break
            
            waypoint = trajectory[waypoint_idx]
            result = WaypointResult(idx=waypoint_idx, feasible=True)
            
            # Set IK command for this waypoint
            ik_commands[:, 0:7] = waypoint.unsqueeze(0).expand(self.scene.num_envs, -1)
            self.ik_controller.set_command(ik_commands)
            
            # Compute IK and get robot state
            try:
                jacobian = self.robot.root_physx_view.get_jacobians()[
                    :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
                ]
                ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
                root_pose_w = self.robot.data.root_pose_w
                joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                
                joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                
                # Prepare context for checkers
                context = {
                    "joint_positions": joint_pos_des[0],
                    "jacobian": jacobian[0],
                    "ee_pose": ee_pose_w[0],
                }
                
                # Run all enabled checkers
                for checker in self.checkers:
                    checker.check(waypoint, result, **context)
                
                # Update robot state (kinematic - direct pose setting)
                self.robot.set_joint_position_target(
                    joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
                )
                self.scene.write_data_to_sim()
                self.sim.step()  # Step to update kinematics (no physics solving)
                self.scene.update(sim_dt)
                
            except Exception as e:
                # If IK computation fails entirely
                result.feasible = False
                result.ik_valid = False
                result.ik_error = str(e)
            
            report.add_result(result)
            
            # Progress update
            if (waypoint_idx + 1) % 50 == 0 or waypoint_idx == len(trajectory) - 1:
                print(f"  Validated {waypoint_idx + 1}/{len(trajectory)} waypoints...", end="\r")
        
        print(f"\n[INFO] Validation complete!")
        return report
    
    def _perform_ik_step(self, sim_dt: float):
        """Perform one IK control step."""
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)


# ============================================================================
# TRAJECTORY VISUALIZER
# ============================================================================

class TrajectoryVisualizer:
    """Handles trajectory visualization with color-coded waypoints."""
    
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        
        # Setup IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=scene.num_envs, device=sim.device
        )
        
        # Setup robot entity configuration
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["Joint_[1-6]"],
            body_names=[EE_LINK_NAME]
        )
        self.robot_entity_cfg.resolve(scene)
        
        self.ee_jacobi_idx = (
            self.robot_entity_cfg.body_ids[0] - 1
            if self.robot.is_fixed_base
            else self.robot_entity_cfg.body_ids[0]
        )
    
    def visualize(self, csv_file: str, report: FeasibilityReport):
        """Visualize trajectory with color-coded waypoints."""
        print(f"\n[INFO] Loading trajectory for visualization...")
        
        try:
            df = pd.read_csv(csv_file)
            trajectory = torch.tensor(df.values, dtype=torch.float32, device=self.sim.device)
            print(f"[INFO] Loaded {len(trajectory)} waypoints")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            return
        
        # Create markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        
        # Create color-coded trajectory markers
        traj_markers = []
        print(f"[INFO] Creating {len(trajectory)} color-coded waypoint markers...")
        
        for i, result in enumerate(report.results):
            # Determine color based on feasibility
            if result.workspace_valid is False:
                color = COLOR_OUT_OF_WORKSPACE  # Black - out of workspace
            elif result.feasible:
                color = COLOR_FEASIBLE  # Green - fully feasible
            else:
                color = COLOR_INFEASIBLE  # Red - other failures
            
            marker_cfg = VisualizationMarkersCfg(
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                }
            )
            traj_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/traj_{i}"))
            traj_markers.append(traj_marker)
        
        # Display all trajectory markers
        print(f"[INFO] Displaying waypoints:")
        print(f"       Green = Feasible | Red = Infeasible | Black = Out of Workspace")
        
        for i, waypoint in enumerate(trajectory):
            pos = waypoint[0:3].unsqueeze(0) + self.scene.env_origins[0:1]
            traj_markers[i].visualize(pos)
        
        # Move to home position
        print(f"[INFO] Moving to home position...")
        ik_commands = torch.zeros(
            self.scene.num_envs, self.ik_controller.action_dim, device=self.robot.device
        )
        home_pose = DEFAULT_HOME_POSE.to(self.robot.device)
        ik_commands[:, 0:7] = home_pose.unsqueeze(0).expand(self.scene.num_envs, -1)
        self.ik_controller.set_command(ik_commands)
        
        sim_dt = self.sim.get_physics_dt()
        
        for _ in range(100):
            self._perform_ik_step(sim_dt)
        
        # Play trajectory
        print(f"[INFO] Playing trajectory...")
        current_waypoint = 0
        steps_per_waypoint = 10
        step_count = 0
        
        while simulation_app.is_running() and current_waypoint < len(trajectory):
            if step_count >= steps_per_waypoint:
                current_waypoint += 1
                step_count = 0
                
                if current_waypoint >= len(trajectory):
                    break
                
                ik_commands[:, 0:7] = trajectory[current_waypoint].unsqueeze(0).expand(
                    self.scene.num_envs, -1
                )
                self.ik_controller.set_command(ik_commands)
                
                if current_waypoint % 10 == 0:
                    print(f"  Waypoint {current_waypoint}/{len(trajectory)}", end="\r")
            
            # Perform IK step
            self._perform_ik_step(sim_dt)
            
            # Update markers
            ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + self.scene.env_origins, ik_commands[:, 3:7])
            
            step_count += 1
        
        print(f"\n[INFO] Visualization complete!")
    
    def _perform_ik_step(self, sim_dt: float):
        """Perform one IK control step."""
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    print("\n" + "=" * 80)
    print("KUKA IRB-1300 TRAJECTORY VALIDATOR")
    print("=" * 80)
    print("Mode: KINEMATIC ONLY (Physics Disabled)")
    print("  - No physics simulation (faster validation)")
    print("  - Pure kinematic feasibility checking")
    print("  - Solver iterations: 0 (kinematic mode)")
    print("=" * 80 + "\n")
    
    # Create simulation WITHOUT physics (kinematic mode only for validation)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01, 
        device=args_cli.device,
        # Disable physics solver for faster kinematic-only validation
        physics_prim_path="/physicsScene",
        physx=sim_utils.PhysxCfg(
            solver_type=0,  # TGS (Temporal Gauss-Seidel) - but we'll disable it
            min_position_iteration_count=0,  # Disable position solver
            max_position_iteration_count=0,
            min_velocity_iteration_count=0,  # Disable velocity solver
            max_velocity_iteration_count=0,
        )
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
    
    # Create scene
    scene_cfg = KukaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Create validator and run checks
    validator = TrajectoryValidator(sim, scene, args_cli.checks)
    report = validator.validate_trajectory(args_cli.csv)
    
    # Print and save report
    report.print_summary()
    
    if args_cli.report_file:
        report_path = args_cli.report_file
    else:
        csv_basename = os.path.splitext(os.path.basename(args_cli.csv))[0]
        csv_dir = os.path.dirname(args_cli.csv) or "."
        report_path = os.path.join(csv_dir, f"{csv_basename}_feasibility_report.txt")
    
    report.save_to_file(report_path)
    
    # Generate plots
    report.generate_plots()
    
    # Visualization (if requested)
    if args_cli.visualize:
        print("\n[INFO] Starting visualization...")
        visualizer = TrajectoryVisualizer(sim, scene)
        visualizer.visualize(args_cli.csv, report)
    else:
        print("\n[INFO] Done. Use --visualize flag to see trajectory playback.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

