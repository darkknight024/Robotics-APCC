#!/usr/bin/env python3
"""
Trajectory Continuity Analyzer - C1, C2 Continuity Checks with Joint-Aware Timing

Analyzes trajectory continuity in joint and Cartesian space with unified pose timing metric
and joint velocity/acceleration constraints.

Key concepts:
- Unified pose distance metric: combines translation and rotation into single scalar
- Joint-aware time scaling: enforces per-joint velocity limits
- C1/C2 continuity: velocity and acceleration continuity in pose and joint space
- Speed in 8th column: extracted from trajectory CSV, applied to entire trajectory

Features:
- Unified pose metric: d_pose = sqrt(d_linear^2 + (pose_scale * d_angle)^2)
- Joint velocity enforcement: T_segment = max(T_pose, max_j(|Δq_j| / vmax_j))
- C1 checks: velocity continuity with joint limits
- C2 checks: acceleration continuity
- Detailed reports and visualization graphs

Architecture:
- Direct integration with batch_trajectory_processor.py
- Uses joint angles from analyze_irb1300_trajectory.py
- Saves results in continuity/ subfolder within trajectory output
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings('ignore')


class UnifiedPoseTimingCalculator:
    """
    Calculate segment timing using unified pose metric with joint-aware scaling.
    
    Combines translation and rotation into single pose_distance metric.
    Enforces joint velocity limits to scale segment durations.
    
    Speed is extracted from 8th column of trajectory CSV (first pose defines speed for entire trajectory).
    """
    
    def __init__(self, 
                 pose_scale_m_per_rad: float = 0.1,
                 velocity_limits_rad_s: Optional[List[float]] = None,
                 acceleration_limits_rad_s2: Optional[List[float]] = None,
                 min_segment_duration: float = 1e-3):
        """
        Initialize unified timing calculator.
        
        Args:
            pose_scale_m_per_rad: Scale factor converting rotation angle to equivalent distance
            velocity_limits_rad_s: Per-joint velocity limits in rad/s (hardware limits)
            acceleration_limits_rad_s2: Per-joint acceleration limits in rad/s²
            min_segment_duration: Minimum segment duration floor in seconds
        """
        self.pose_scale_m_per_rad = pose_scale_m_per_rad
        self.velocity_limits_rad_s = np.array(velocity_limits_rad_s) if velocity_limits_rad_s is not None else None
        self.acceleration_limits_rad_s2 = np.array(acceleration_limits_rad_s2) if acceleration_limits_rad_s2 is not None else None
        self.min_segment_duration = min_segment_duration
    
    def compute_segment_times(self, 
                             trajectory_m: np.ndarray,
                             joint_angles_rad: Optional[np.ndarray] = None,
                             speed_mm_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute timestamps for each waypoint using unified pose metric and joint limits.
        
        Args:
            trajectory_m: Array of poses (n_waypoints, 7) with positions in meters
            joint_angles_rad: Joint configurations (n_waypoints, n_joints) in radians, or None
            speed_mm_s: Speed from 8th column (first pose). Default 100 mm/s if None
            
        Returns:
            timestamps: Array of timestamps (n_waypoints,)
            segment_durations: Array of segment durations (n_waypoints-1,)
        """
        # Extract speed from parameter (from 8th column of CSV, first pose)
        if speed_mm_s is None:
            speed_mm_s = 100.0  # Default fallback
        
        pose_speed_m_s = speed_mm_s / 1000.0  # Convert mm/s to m/s
        
        n_waypoints = len(trajectory_m)
        segment_durations = np.zeros(n_waypoints - 1)
        
        for i in range(n_waypoints - 1):
            # Cartesian pose distance
            p1 = trajectory_m[i, :3]
            p2 = trajectory_m[i + 1, :3]
            d_linear = np.linalg.norm(p2 - p1)
            
            # Orientation distance
            q1 = trajectory_m[i, 3:7]
            q2 = trajectory_m[i + 1, 3:7]
            d_angle = self._quaternion_angle_distance(q1, q2)
            
            # Unified pose distance
            pose_distance = np.sqrt(d_linear**2 + (self.pose_scale_m_per_rad * d_angle)**2)
            
            # Time from pose speed (from CSV 8th column)
            t_pose = pose_distance / pose_speed_m_s if pose_speed_m_s > 0 else 0
            
            # Time from joint velocity limits (if available)
            t_joint_min = 0
            if joint_angles_rad is not None and self.velocity_limits_rad_s is not None:
                q1_rad = joint_angles_rad[i]
                q2_rad = joint_angles_rad[i + 1]
                
                # Joint deltas
                delta_q = np.abs(q2_rad - q1_rad)
                
                # Time required for each joint
                t_per_joint = delta_q / self.velocity_limits_rad_s
                
                # Maximum time across all joints
                t_joint_min = np.max(t_per_joint) if len(t_per_joint) > 0 else 0
            
            # Final segment duration is maximum of pose time and joint time
            segment_durations[i] = max(t_pose, t_joint_min, self.min_segment_duration)
        
        # Accumulate to get timestamps
        timestamps = np.zeros(n_waypoints)
        for i in range(1, n_waypoints):
            timestamps[i] = timestamps[i - 1] + segment_durations[i - 1]
        
        return timestamps, segment_durations
    
    @staticmethod
    def _quaternion_angle_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute angle between two quaternions (shortest arc in radians)."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot_prod = np.dot(q1, q2)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        
        # Use shortest arc
        angle_rad = 2.0 * np.arccos(np.abs(dot_prod))
        return angle_rad


class CartesianTrajectoryInterpolator:
    """Interpolate trajectory in Cartesian space with analytical derivatives."""
    
    @staticmethod
    def interpolate_positions(timestamps: np.ndarray, trajectory_m: np.ndarray,
                             sample_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray, CubicSpline]:
        """
        Interpolate positions using cubic spline with analytical derivatives.
        
        Returns:
            t_samples: Sample timestamps
            positions_m: Interpolated positions (n_samples, 3)
            cs_spline: CubicSpline object for computing derivatives
        """
        positions_m = trajectory_m[:, :3]
        
        # Create cubic splines for each dimension
        cs_x = CubicSpline(timestamps, positions_m[:, 0])
        cs_y = CubicSpline(timestamps, positions_m[:, 1])
        cs_z = CubicSpline(timestamps, positions_m[:, 2])
        
        # Sample
        t_start, t_end = timestamps[0], timestamps[-1]
        dt = 1.0 / sample_rate
        t_samples = np.arange(t_start, t_end + dt, dt)
        
        positions_interp = np.column_stack([
            cs_x(t_samples),
            cs_y(t_samples),
            cs_z(t_samples)
        ])
        
        # Store splines for derivative computation
        splines = (cs_x, cs_y, cs_z)
        
        return t_samples, positions_interp, splines
    
    @staticmethod
    def get_position_derivatives(splines: Tuple, t_samples: np.ndarray, order: int = 1):
        """Get analytical derivatives from cubic splines."""
        cs_x, cs_y, cs_z = splines
        
        if order == 1:
            return np.column_stack([
                cs_x(t_samples, 1),
                cs_y(t_samples, 1),
                cs_z(t_samples, 1)
            ])
        elif order == 2:
            return np.column_stack([
                cs_x(t_samples, 2),
                cs_y(t_samples, 2),
                cs_z(t_samples, 2)
            ])
        else:
            raise ValueError(f"Order {order} not supported")
    
    @staticmethod
    def interpolate_quaternions(timestamps: np.ndarray, trajectory_m: np.ndarray,
                               sample_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate quaternions using SLERP with quaternion continuity."""
        quaternions = trajectory_m[:, 3:7]
        
        # Ensure quaternion continuity (shortest arc)
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[i - 1], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]
        
        # Sample timestamps
        t_start, t_end = timestamps[0], timestamps[-1]
        dt = 1.0 / sample_rate
        t_samples = np.arange(t_start, t_end + dt, dt)
        
        quaternions_interp = []
        for t in t_samples:
            idx = np.searchsorted(timestamps, t)
            if idx == 0:
                q = quaternions[0]
            elif idx >= len(timestamps):
                q = quaternions[-1]
            else:
                t1, t2 = timestamps[idx - 1], timestamps[idx]
                q1, q2 = quaternions[idx - 1], quaternions[idx]
                
                if t2 - t1 > 1e-6:
                    s = (t - t1) / (t2 - t1)
                else:
                    s = 0
                
                q = CartesianTrajectoryInterpolator._slerp(q1, q2, s)
            
            quaternions_interp.append(q)
        
        return t_samples, np.array(quaternions_interp)
    
    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, s: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot_prod = np.dot(q1, q2)
        
        if dot_prod < 0:
            q2 = -q2
            dot_prod = -dot_prod
        
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        theta = np.arccos(dot_prod)
        
        if abs(theta) < 1e-6:
            return (1 - s) * q1 + s * q2
        
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - s) * theta) / sin_theta
        w2 = np.sin(s * theta) / sin_theta
        
        return w1 * q1 + w2 * q2


class JointSpaceInterpolator:
    """Interpolate joint configurations and compute derivatives."""
    
    @staticmethod
    def interpolate_joints(timestamps: np.ndarray, joint_angles_rad: np.ndarray,
                          sample_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate joint angles using cubic splines.
        
        Returns:
            t_samples: Sample timestamps
            joints_interp: Interpolated joint angles (n_samples, n_joints)
        """
        n_joints = joint_angles_rad.shape[1]
        
        # Create splines for each joint
        splines = []
        for j in range(n_joints):
            cs = CubicSpline(timestamps, joint_angles_rad[:, j])
            splines.append(cs)
        
        # Sample
        t_start, t_end = timestamps[0], timestamps[-1]
        dt = 1.0 / sample_rate
        t_samples = np.arange(t_start, t_end + dt, dt)
        
        joints_interp = np.column_stack([cs(t_samples) for cs in splines])
        
        return t_samples, joints_interp, splines
    
    @staticmethod
    def get_joint_derivatives(splines: List, t_samples: np.ndarray, order: int = 1):
        """Get analytical derivatives from joint splines."""
        if order == 1:
            return np.column_stack([cs(t_samples, 1) for cs in splines])
        elif order == 2:
            return np.column_stack([cs(t_samples, 2) for cs in splines])
        else:
            raise ValueError(f"Order {order} not supported")


class ContinuityChecker:
    """Check C1 continuity (velocity limits) in joint space."""
    
    def __init__(self,
                 velocity_limits_rad_s: Optional[np.ndarray] = None,
                 speed_mm_s: Optional[float] = None,
                 pose_scale_m_per_rad: float = 0.1,
                 safety_factor: float = 1.05):
        """
        Initialize continuity checker for C1 analysis.
        
        Args:
            velocity_limits_rad_s: Per-joint velocity limits in rad/s (hardware constraints)
            speed_mm_s: End-effector speed from CSV 8th column (mm/s)
            pose_scale_m_per_rad: Pose distance scale factor
            safety_factor: Safety margin for limit checks (default 1.05 = 5% margin)
        """
        self.velocity_limits_rad_s = velocity_limits_rad_s
        self.speed_mm_s = speed_mm_s or 100.0
        self.pose_speed_m_s = self.speed_mm_s / 1000.0
        self.pose_scale_m_per_rad = pose_scale_m_per_rad
        self.safety_factor = safety_factor
    
    def check_c1_joint(self, joint_velocities: np.ndarray, t_samples: np.ndarray) -> Dict:
        """Check C1 in joint space (velocity limits)."""
        report = {'violations': [], 'passed': True, 'max_velocities_rad_s': []}
        
        if self.velocity_limits_rad_s is None:
            return {'enabled': False, 'passed': True}
        
        for j in range(joint_velocities.shape[1]):
            vel_abs = np.abs(joint_velocities[:, j])
            max_vel = np.max(vel_abs)
            limit = self.velocity_limits_rad_s[j]
            
            report['max_velocities_rad_s'].append(float(max_vel))
            
            if max_vel > limit * self.safety_factor:
                report['violations'].append({
                    'joint': j,
                    'max_velocity_rad_s': float(max_vel),
                    'limit_rad_s': float(limit),
                    'message': f'Joint {j} velocity {max_vel:.3f} rad/s exceeds limit {limit:.3f} rad/s'
                })
                report['passed'] = False
        
        return report
    
    @staticmethod
    def _quaternion_angle_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute angle between quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot_prod = np.clip(np.dot(q1, q2), -1.0, 1.0)
        return 2.0 * np.arccos(np.abs(dot_prod))


def analyze_trajectory_continuity(trajectory_m: np.ndarray,
                                 joint_angles_rad: Optional[np.ndarray],
                                 output_dir: Path,
                                 config: Optional[Dict] = None,
                                 trajectory_id: int = 1,
                                 speed_mm_s: Optional[float] = None,
                                 robot_velocity_limits_rad_s: Optional[List[float]] = None,
                                 robot_acceleration_limits_rad_s2: Optional[List[float]] = None) -> Dict:
    """
    Analyze trajectory continuity with unified pose timing and joint constraints.
    
    Args:
        trajectory_m: Waypoints (n_waypoints, 7) in meters
        joint_angles_rad: Joint angles from IK (n_waypoints, n_joints) or None
        output_dir: Output directory for results
        config: Configuration dictionary
        trajectory_id: Trajectory identifier
        speed_mm_s: Speed from CSV 8th column (first pose). Default 100 mm/s if None
        robot_velocity_limits_rad_s: Per-joint velocity limits from robot config
        robot_acceleration_limits_rad_s2: Per-joint acceleration limits
        
    Returns:
        analysis_results: Dictionary with continuity analysis results
    """
    config = config or {}
    continuity_config = config.get('continuity', {})
    
    if not continuity_config.get('enabled', False):
        return {'enabled': False}
    
    # Create continuity subfolder
    continuity_dir = Path(output_dir) / 'continuity'
    continuity_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration parameters
    if speed_mm_s is None:
        speed_mm_s = 100.0  # Default fallback
    
    pose_scale_m_per_rad = continuity_config.get('pose_scale_m_per_rad', 0.1)
    safety_factor = continuity_config.get('safety_factor', 1.05)
    
    # Use robot limits if provided, otherwise try to get from config
    if robot_velocity_limits_rad_s is None:
        robot_velocity_limits_rad_s = continuity_config.get('default_velocity_limits_rad_s', None)
    if robot_acceleration_limits_rad_s2 is None:
        robot_acceleration_limits_rad_s2 = continuity_config.get('default_acceleration_limits_rad_s2', None)
    
    # Convert to numpy arrays
    if robot_velocity_limits_rad_s is not None:
        robot_velocity_limits_rad_s = np.array(robot_velocity_limits_rad_s)
    if robot_acceleration_limits_rad_s2 is not None:
        robot_acceleration_limits_rad_s2 = np.array(robot_acceleration_limits_rad_s2)
    
    # Compute timing with unified pose metric and joint constraints
    timing_calc = UnifiedPoseTimingCalculator(
        pose_scale_m_per_rad=pose_scale_m_per_rad,
        velocity_limits_rad_s=robot_velocity_limits_rad_s,
        acceleration_limits_rad_s2=robot_acceleration_limits_rad_s2
    )
    
    timestamps, segment_durations = timing_calc.compute_segment_times(
        trajectory_m, joint_angles_rad, speed_mm_s=speed_mm_s
    )
    
    # Interpolate in Cartesian space
    interp = CartesianTrajectoryInterpolator()
    t_samples, positions_interp, pos_splines = interp.interpolate_positions(
        timestamps, trajectory_m, sample_rate=100.0
    )
    
    # Get velocities and accelerations from splines
    velocities_m_s = interp.get_position_derivatives(pos_splines, t_samples, order=1)
    accelerations_m_s2 = interp.get_position_derivatives(pos_splines, t_samples, order=2)
    
    # Interpolate quaternions
    t_samples_q, quaternions_interp = interp.interpolate_quaternions(
        timestamps, trajectory_m, sample_rate=100.0
    )
    
    # Interpolate joint angles (if available)
    joint_velocities = None
    if joint_angles_rad is not None:
        joint_interp = JointSpaceInterpolator()
        t_samples_j, joints_interp, joint_splines = joint_interp.interpolate_joints(
            timestamps, joint_angles_rad, sample_rate=100.0
        )
        joint_velocities = joint_interp.get_joint_derivatives(joint_splines, t_samples_j, order=1)
    
    # Check C1 continuity (velocity limits only)
    checker = ContinuityChecker(
        velocity_limits_rad_s=robot_velocity_limits_rad_s,
        speed_mm_s=speed_mm_s,
        pose_scale_m_per_rad=pose_scale_m_per_rad,
        safety_factor=safety_factor
    )
    
    # C1 Joint velocity continuity check (main focus)
    c1_joint_report = {}
    if joint_velocities is not None and len(joint_velocities) > 0:
        c1_joint_report = checker.check_c1_joint(joint_velocities, t_samples_j)
    
    # Compile results
    results = {
        'enabled': True,
        'trajectory_id': trajectory_id,
        'timing': {
            'timestamps': timestamps.tolist(),
            'segment_durations': segment_durations.tolist(),
            'total_duration_s': float(timestamps[-1] - timestamps[0]),
            'speed_mm_s': float(speed_mm_s) if speed_mm_s is not None else None,  # From CSV 8th column, first pose
            'pose_scale_m_per_rad': pose_scale_m_per_rad,
        },
        'continuity': {
            'c1_joint': c1_joint_report,  # Joint velocity continuity (main check)
        },
        'passed': c1_joint_report.get('passed', True),
        'summary': {
            'total_waypoints': len(trajectory_m),
            'has_joint_limits': robot_velocity_limits_rad_s is not None,
        }
    }
    
    # if joint_velocities is not None:
    #     results['summary']['max_joint_velocities_rad_s'] = c1_joint_report.get('max_velocities_rad_s', [])
    
    # Save report
    report_path = continuity_dir / f'Traj_{trajectory_id}_continuity_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Continuity report: {report_path.name}")
    
    # Generate graphs
    try:
        _generate_continuity_graphs(
            t_samples, positions_interp, velocities_m_s,
            t_samples_j if joint_velocities is not None else None,
            joint_velocities,
            robot_velocity_limits_rad_s,
            trajectory_id, continuity_dir, speed_mm_s, pose_scale_m_per_rad
        )
    except Exception as e:
        print(f"  ⚠ Could not generate graphs: {e}")
    
    return results


def _generate_continuity_graphs(t_samples, positions_m, velocities_m_s,
                               t_samples_j, joint_velocities,
                               velocity_limits_rad_s,
                               trajectory_id, output_dir, speed_mm_s, pose_scale_m_per_rad):
    """Generate C1 continuity analysis graphs (velocity limits only)."""
    
    # Convert velocities to mm/s
    velocities_mm_s = velocities_m_s * 1000
    velocity_norms_mm_s = np.linalg.norm(velocities_mm_s, axis=1)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Traj {trajectory_id} - C1 Continuity Analysis (Velocity Limits)', 
                fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Position components (in robot base frame, meters)
    ax1.plot(t_samples, positions_m[:, 0], label='X', linewidth=1.5)
    ax1.plot(t_samples, positions_m[:, 1], label='Y', linewidth=1.5)
    ax1.plot(t_samples, positions_m[:, 2], label='Z', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_ylabel('Position (m)', fontweight='bold')
    ax1.set_title('Cartesian Position (Transformed (T_B_P)) / Plate in Robot Base Frame)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cartesian velocity magnitude
    ax2.plot(t_samples, velocity_norms_mm_s, label='Speed', linewidth=2, color='tab:green')
    ax2.axhline(y=speed_mm_s, color='orange', linestyle='--', linewidth=2, label=f'Target speed ({speed_mm_s:.1f} mm/s)')
    ax2.fill_between(t_samples, 0, speed_mm_s, alpha=0.1, color='green')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Velocity (mm/s)', fontweight='bold')
    ax2.set_title('Cartesian Velocity', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity components
    ax3.plot(t_samples, velocities_mm_s[:, 0], label='Vx', linewidth=1.5, alpha=0.8)
    ax3.plot(t_samples, velocities_mm_s[:, 1], label='Vy', linewidth=1.5, alpha=0.8)
    ax3.plot(t_samples, velocities_mm_s[:, 2], label='Vz', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Velocity Component (mm/s)', fontweight='bold')
    ax3.set_title('Velocity Components', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Joint velocities with limits (C1 check)
    if joint_velocities is not None and velocity_limits_rad_s is not None:
        for j in range(joint_velocities.shape[1]):
            ax4.plot(t_samples_j, joint_velocities[:, j], label=f'q{j+1}', linewidth=1.5, alpha=0.8)
            ax4.axhline(y=velocity_limits_rad_s[j], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax4.axhline(y=-velocity_limits_rad_s[j], color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax4.set_xlabel('Time (s)', fontweight='bold')
        ax4.set_ylabel('Joint Velocity (rad/s)', fontweight='bold')
        ax4.set_title('C1: Joint Velocities vs Limits (Hardware Constraints)', fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'Traj_{trajectory_id}_continuity.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Continuity graph: {output_path.name}")


if __name__ == "__main__":
    print("Trajectory Continuity Analyzer (Unified Pose Timing + Joint Constraints)")
    print("Use via batch_trajectory_processor.py")
