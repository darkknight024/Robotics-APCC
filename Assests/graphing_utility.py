"""
Graphing Utility for IRB-1300 Trajectory Playback

This module collects data during trajectory playback and generates visualization plots:
- End-effector speed analysis
- Joint position tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


# ===================================================================
# JOINT LIMITS FROM URDF (radians)
# ===================================================================
JOINT_LIMITS = {
    "Joint_1": (-3.142, 3.142),    # -180° to 180°
    "Joint_2": (-1.745, 2.269),    # -100° to 130°
    "Joint_3": (-3.665, 1.134),    # -210° to 65°
    "Joint_4": (-4.014, 4.014),    # -230° to 230°
    "Joint_5": (-2.269, 2.269),    # -130° to 130°
    "Joint_6": (-6.981, 6.981),    # -400° to 400°
}


# ===================================================================
# SPEED DATA COLLECTOR
# ===================================================================
class SpeedDataCollector:
    """Collects end-effector speed data during trajectory playback."""
    
    def __init__(self, dt, target_speeds_mm_s=None):
        """Initialize speed data collector.
        
        Args:
            dt: Simulation timestep in seconds
            target_speeds_mm_s: Optional list/array of target speeds from CSV (mm/s)
        """
        self.dt = dt
        self.step_speeds = []           # Speed at each step (m/s)
        self.step_waypoint_indices = [] # Active waypoint at each step
        self.step_timestamps = []       # Timestamp at each step (s)
        self.step_positions = []        # Position at each step
        self.waypoint_transitions = []  # List of (waypoint_idx, position, time)
        self.last_position = None       # Last recorded position
        self.current_time = 0.0         # Current simulation time
        self.target_speeds_mm_s = target_speeds_mm_s  # Target speeds from CSV
        
        print("[SpeedDataCollector] Initialized")
        if target_speeds_mm_s is not None and len(target_speeds_mm_s) > 0:
            print(f"[SpeedDataCollector] Loaded {len(target_speeds_mm_s)} target speeds from CSV")
    
    def record_step(self, ee_position, current_waypoint):
        """Record data for current simulation step.
        
        Args:
            ee_position: Current end-effector position (torch tensor [x, y, z])
            current_waypoint: Index of current active waypoint
        """
        # Convert to numpy
        pos = ee_position.cpu().numpy().flatten()
        
        # Calculate instantaneous speed
        if self.last_position is not None:
            delta_pos = pos - self.last_position
            distance = np.linalg.norm(delta_pos)
            speed = distance / self.dt  # m/s
        else:
            speed = 0.0
        
        # Record step data
        self.step_speeds.append(speed)
        self.step_waypoint_indices.append(current_waypoint)
        self.step_timestamps.append(self.current_time)
        self.step_positions.append(pos)
        
        # Update tracking
        self.last_position = pos
        self.current_time += self.dt
    
    def record_waypoint_transition(self, ee_position, waypoint_index, time):
        """Record when a waypoint transition occurs.
        
        Args:
            ee_position: End-effector position at transition (torch tensor)
            waypoint_index: Index of waypoint being transitioned to
            time: Current simulation time
        """
        pos = ee_position.cpu().numpy().flatten()
        self.waypoint_transitions.append((waypoint_index, pos.copy(), time))
    
    def compute_waypoint_speeds(self):
        """Compute average speeds between waypoint transitions.
        
        Returns:
            Tuple of (waypoint_indices, speeds_mm_per_s)
            waypoint_indices: Array of waypoint transition indices (0->1, 1->2, etc.)
            speeds_mm_per_s: Array of average speeds in mm/s
        """
        if len(self.waypoint_transitions) < 2:
            return np.array([]), np.array([])
        
        speeds_mm_s = []
        waypoint_indices = []
        
        for i in range(len(self.waypoint_transitions) - 1):
            idx1, pos1, time1 = self.waypoint_transitions[i]
            idx2, pos2, time2 = self.waypoint_transitions[i + 1]
            
            distance = np.linalg.norm(pos2 - pos1)  # meters
            time_diff = time2 - time1
            
            if time_diff > 0:
                avg_speed_m_s = distance / time_diff
                avg_speed_mm_s = avg_speed_m_s * 1000.0  # Convert to mm/s
            else:
                avg_speed_mm_s = 0.0
            
            speeds_mm_s.append(avg_speed_mm_s)
            waypoint_indices.append(idx1)  # Transition from idx1 to idx2
        
        return np.array(waypoint_indices), np.array(speeds_mm_s)
    
    def get_step_data(self):
        """Get per-step data.
        
        Returns:
            Tuple of (timestamps, speeds_mm_per_s, waypoint_indices)
        """
        speeds_mm_s = np.array(self.step_speeds) * 1000.0  # Convert to mm/s
        return (
            np.array(self.step_timestamps),
            speeds_mm_s,
            np.array(self.step_waypoint_indices)
        )
    
    def save_data(self, filepath):
        """Save collected data to CSV file.
        
        Args:
            filepath: Path to save CSV file (should end with .csv)
        """
        import pandas as pd
        
        waypoint_indices, waypoint_speeds = self.compute_waypoint_speeds()
        timestamps, step_speeds_mm_s, step_waypoint_indices = self.get_step_data()
        
        # Create per-step data CSV
        step_data = pd.DataFrame({
            'timestamp_s': timestamps,
            'speed_mm_s': step_speeds_mm_s,
            'active_waypoint': step_waypoint_indices
        })
        
        # Create per-waypoint data CSV
        waypoint_data = pd.DataFrame({
            'waypoint_transition': waypoint_indices,
            'calculated_speed_mm_s': waypoint_speeds
        })
        
        # Add target speeds if available
        if self.target_speeds_mm_s is not None and len(self.target_speeds_mm_s) > 0:
            # Match target speeds to waypoint indices
            target_speeds_for_waypoints = []
            for idx in waypoint_indices:
                if idx < len(self.target_speeds_mm_s):
                    target_speeds_for_waypoints.append(self.target_speeds_mm_s[idx])
                else:
                    target_speeds_for_waypoints.append(np.nan)
            waypoint_data['target_speed_mm_s'] = target_speeds_for_waypoints
        
        # Save both datasets to CSV
        step_csv = filepath.replace('.csv', '_steps.csv')
        waypoint_csv = filepath.replace('.csv', '_waypoints.csv')
        
        step_data.to_csv(step_csv, index=False)
        waypoint_data.to_csv(waypoint_csv, index=False)
        
        print(f"[SpeedDataCollector] Saved step data to: {step_csv}")
        print(f"[SpeedDataCollector] Saved waypoint data to: {waypoint_csv}")
        print(f"[SpeedDataCollector] Total steps recorded: {len(self.step_speeds)}")
        print(f"[SpeedDataCollector] Waypoint transitions: {len(waypoint_speeds)}")
        
        return step_csv, waypoint_csv


# ===================================================================
# JOINT DATA COLLECTOR
# ===================================================================
class JointDataCollector:
    """Collects joint position data during trajectory playback."""
    
    def __init__(self):
        """Initialize joint data collector."""
        self.joint_positions = {f"Joint_{i}": [] for i in range(1, 7)}
        self.step_indices = []
        self.current_step = 0
        
        print("[JointDataCollector] Initialized")
    
    def record_step(self, joint_pos_tensor):
        """Record joint positions for current simulation step.
        
        Args:
            joint_pos_tensor: Tensor of joint positions [6] in radians
        """
        # Convert to numpy
        joint_pos = joint_pos_tensor.cpu().numpy().flatten()
        
        # Record each joint
        for i in range(6):
            self.joint_positions[f"Joint_{i+1}"].append(joint_pos[i])
        
        self.step_indices.append(self.current_step)
        self.current_step += 1
    
    def save_data(self, filepath):
        """Save collected joint data to CSV file.
        
        Args:
            filepath: Path to save CSV file (should end with .csv)
        """
        import pandas as pd
        
        # Create DataFrame
        data = {'step': self.step_indices}
        for joint_name in self.joint_positions.keys():
            data[joint_name] = self.joint_positions[joint_name]
        
        joint_data = pd.DataFrame(data)
        
        # Save to CSV
        joint_csv = filepath.replace('.csv', '_joints.csv')
        joint_data.to_csv(joint_csv, index=False)
        
        print(f"[JointDataCollector] Saved joint data to: {joint_csv}")
        print(f"[JointDataCollector] Total steps recorded: {len(self.step_indices)}")
        
        return joint_csv


# ===================================================================
# POSE TRACKING DATA COLLECTOR
# ===================================================================
class PoseTrackingCollector:
    """Collects target and actual end-effector poses during trajectory playback."""
    
    def __init__(self):
        """Initialize pose tracking collector."""
        self.target_positions = []  # IK command positions (x, y, z)
        self.actual_positions = []  # Actual EE positions (x, y, z)
        self.target_orientations = []  # IK command orientations (qx, qy, qz, qw)
        self.actual_orientations = []  # Actual EE orientations (qx, qy, qz, qw)
        self.step_indices = []
        self.current_step = 0
        
        print("[PoseTrackingCollector] Initialized")
    
    def record_step(self, target_pose, actual_pose):
        """Record target and actual poses for current simulation step.
        
        Args:
            target_pose: Target pose from IK commands [7] (x, y, z, qx, qy, qz, qw)
            actual_pose: Actual end-effector pose [7] (x, y, z, qx, qy, qz, qw)
        """
        # Convert to numpy
        target = target_pose.cpu().numpy().flatten()
        actual = actual_pose.cpu().numpy().flatten()
        
        # Record positions and orientations
        self.target_positions.append(target[0:3].copy())
        self.actual_positions.append(actual[0:3].copy())
        self.target_orientations.append(target[3:7].copy())
        self.actual_orientations.append(actual[3:7].copy())
        self.step_indices.append(self.current_step)
        self.current_step += 1
    
    def save_data(self, filepath):
        """Save collected pose data to CSV file.
        
        Args:
            filepath: Path to save CSV file (should end with .csv)
        """
        import pandas as pd
        
        # Convert lists to arrays
        target_pos = np.array(self.target_positions)
        actual_pos = np.array(self.actual_positions)
        target_ori = np.array(self.target_orientations)
        actual_ori = np.array(self.actual_orientations)
        
        # Create DataFrame
        data = {
            'step': self.step_indices,
            'target_x': target_pos[:, 0],
            'target_y': target_pos[:, 1],
            'target_z': target_pos[:, 2],
            'target_qx': target_ori[:, 0],
            'target_qy': target_ori[:, 1],
            'target_qz': target_ori[:, 2],
            'target_qw': target_ori[:, 3],
            'actual_x': actual_pos[:, 0],
            'actual_y': actual_pos[:, 1],
            'actual_z': actual_pos[:, 2],
            'actual_qx': actual_ori[:, 0],
            'actual_qy': actual_ori[:, 1],
            'actual_qz': actual_ori[:, 2],
            'actual_qw': actual_ori[:, 3],
        }
        
        pose_data = pd.DataFrame(data)
        
        # Save to CSV
        pose_csv = filepath.replace('.csv', '_poses.csv')
        pose_data.to_csv(pose_csv, index=False)
        
        print(f"[PoseTrackingCollector] Saved pose data to: {pose_csv}")
        print(f"[PoseTrackingCollector] Total steps recorded: {len(self.step_indices)}")
        
        return pose_csv


# ===================================================================
# SPEED PLOTTING FUNCTIONS
# ===================================================================
def plot_speed_data(data_file, output_dir="./speed_plots"):
    """Generate speed visualization plots from saved data.
    
    Args:
        data_file: Base path to CSV files (will append _steps.csv and _waypoints.csv)
        output_dir: Directory to save plot images
    """
    import pandas as pd
    
    # Load data from CSV files
    step_csv = data_file.replace('.csv', '_steps.csv')
    waypoint_csv = data_file.replace('.csv', '_waypoints.csv')
    
    step_data = pd.read_csv(step_csv)
    waypoint_data = pd.read_csv(waypoint_csv)
    
    # Extract data
    step_timestamps = step_data['timestamp_s'].values
    step_speeds_mm_s = step_data['speed_mm_s'].values
    step_waypoint_indices = step_data['active_waypoint'].values
    waypoint_indices = waypoint_data['waypoint_transition'].values
    waypoint_speeds_mm_s = waypoint_data['calculated_speed_mm_s'].values
    
    # Check if target speeds exist
    has_target_speeds = 'target_speed_mm_s' in waypoint_data.columns
    if has_target_speeds:
        target_speeds_mm_s = waypoint_data['target_speed_mm_s'].values
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ==========================================
    # PLOT 1: Average Speed Between Waypoints (LINE PLOT)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot waypoint speeds as line plot
    if len(waypoint_indices) > 0:
        # Plot calculated speeds
        ax1.plot(waypoint_indices, waypoint_speeds_mm_s, 
                marker='o', color='steelblue', linewidth=2.5, markersize=6,
                label='Calculated Speed', alpha=0.8)
        
        # Plot target speeds if available
        if has_target_speeds:
            ax1.plot(waypoint_indices, target_speeds_mm_s, 
                    marker='s', color='orange', linewidth=2.5, markersize=6,
                    label='Target Speed (CSV)', alpha=0.8, linestyle='--')
        
        # Add value labels (smaller font, no decimals, only every 5th point to avoid overlap)
        label_interval = max(1, len(waypoint_indices) // 20)  # Show ~20 labels max
        for i, (idx, speed) in enumerate(zip(waypoint_indices, waypoint_speeds_mm_s)):
            if i % label_interval == 0:
                ax1.text(idx, speed + max(waypoint_speeds_mm_s) * 0.03, 
                        f'{int(speed)}', ha='center', va='bottom', fontsize=7, color='steelblue')
        
        # Add target speed labels if available
        if has_target_speeds:
            for i, (idx, speed) in enumerate(zip(waypoint_indices, target_speeds_mm_s)):
                if i % label_interval == 0 and not np.isnan(speed):
                    ax1.text(idx, speed - max(waypoint_speeds_mm_s) * 0.03, 
                            f'{int(speed)}', ha='center', va='top', fontsize=7, color='orange')
    
    ax1.set_xlabel('Waypoint Transition (From Waypoint i to i+1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Speed (mm/s)', fontsize=12, fontweight='bold')
    ax1.set_title('End-Effector Speed Between Waypoints', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add statistics text box
    if len(waypoint_speeds_mm_s) > 0:
        stats_text = f'Calculated:\n'
        stats_text += f'  Mean: {int(np.mean(waypoint_speeds_mm_s))} mm/s\n'
        stats_text += f'  Max: {int(np.max(waypoint_speeds_mm_s))} mm/s\n'
        stats_text += f'  Min: {int(np.min(waypoint_speeds_mm_s))} mm/s'
        
        if has_target_speeds:
            valid_targets = target_speeds_mm_s[~np.isnan(target_speeds_mm_s)]
            if len(valid_targets) > 0:
                stats_text += f'\n\nTarget:\n'
                stats_text += f'  Mean: {int(np.mean(valid_targets))} mm/s\n'
                stats_text += f'  Max: {int(np.max(valid_targets))} mm/s\n'
                stats_text += f'  Min: {int(np.min(valid_targets))} mm/s'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'waypoint_speeds.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved waypoint speed plot: {plot1_path}")
    plt.close(fig1)
    
    # ==========================================
    # PLOT 2: Instantaneous Speed at Each Step
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    # Color code by active waypoint
    unique_waypoints = np.unique(step_waypoint_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_waypoints)))
    
    # Plot each waypoint segment with different color
    for i, wp_idx in enumerate(unique_waypoints):
        mask = step_waypoint_indices == wp_idx
        ax2.plot(step_timestamps[mask], step_speeds_mm_s[mask], 
                linewidth=1.5, alpha=0.7, label=f'Waypoint {int(wp_idx)}',
                color=colors[i % len(colors)])
    
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Instantaneous Speed (mm/s)', fontsize=12, fontweight='bold')
    ax2.set_title('End-Effector Speed Over Time (Colored by Active Waypoint)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add legend (limit to max 15 entries to avoid clutter)
    if len(unique_waypoints) <= 15:
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Add statistics
    if len(step_speeds_mm_s) > 0:
        stats_text = f'Mean: {np.mean(step_speeds_mm_s):.1f} mm/s\n'
        stats_text += f'Max: {np.max(step_speeds_mm_s):.1f} mm/s\n'
        stats_text += f'Total Time: {step_timestamps[-1]:.1f} s'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'step_speeds.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved step speed plot: {plot2_path}")
    plt.close(fig2)
    
    # ==========================================
    # PLOT 3: Combined View (Bonus)
    # ==========================================
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Step speeds with waypoint transitions marked
    for i, wp_idx in enumerate(unique_waypoints):
        mask = step_waypoint_indices == wp_idx
        ax3a.plot(step_timestamps[mask], step_speeds_mm_s[mask], 
                 linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
    
    # Mark waypoint transitions with vertical lines (calculate from waypoint indices)
    # Find approximate transition times by looking at changes in active waypoint
    transition_times = []
    for i in range(1, len(step_waypoint_indices)):
        if step_waypoint_indices[i] != step_waypoint_indices[i-1]:
            transition_times.append(step_timestamps[i])
    
    for t in transition_times:
        ax3a.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax3a.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3a.set_ylabel('Speed (mm/s)', fontsize=11, fontweight='bold')
    ax3a.set_title('Instantaneous Speed with Waypoint Transitions', fontsize=12, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    
    # Bottom: Waypoint average speeds (line plot)
    if len(waypoint_indices) > 0:
        ax3b.plot(waypoint_indices, waypoint_speeds_mm_s, 
                 marker='o', color='steelblue', linewidth=2, markersize=5,
                 label='Calculated', alpha=0.8)
        if has_target_speeds:
            ax3b.plot(waypoint_indices, target_speeds_mm_s, 
                     marker='s', color='orange', linewidth=2, markersize=5,
                     label='Target (CSV)', alpha=0.8, linestyle='--')
            ax3b.legend(loc='upper right', fontsize=9)
    
    ax3b.set_xlabel('Waypoint Transition', fontsize=11, fontweight='bold')
    ax3b.set_ylabel('Average Speed (mm/s)', fontsize=11, fontweight='bold')
    ax3b.set_title('Average Speed Between Waypoints', fontsize=12, fontweight='bold')
    ax3b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'combined_speed_analysis.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved combined plot: {plot3_path}")
    plt.close(fig3)
    
    print("[GraphingUtility] All speed plots generated successfully!")
    
    return {
        'waypoint_plot': plot1_path,
        'step_plot': plot2_path,
        'combined_plot': plot3_path
    }


# ===================================================================
# JOINT PLOTTING FUNCTIONS
# ===================================================================
def plot_joint_data(data_file, output_dir="./joint_plots"):
    """Generate joint position plots from saved data.
    
    Creates 6 separate graphs, one for each joint, showing:
    - X-axis: Step number
    - Y-axis: Joint position in radians
    - Y-axis limits: Min/max from URDF joint limits
    
    Also creates a combined 6-panel plot with all joints arranged 3x2.
    
    Args:
        data_file: Base path to CSV file (will append _joints.csv)
        output_dir: Directory to save plot images
    
    Returns:
        List of paths to generated plot files (includes combined plot)
    """
    import pandas as pd
    
    # Load data from CSV
    joint_csv = data_file.replace('.csv', '_joints.csv')
    
    if not os.path.exists(joint_csv):
        print(f"[GraphingUtility] Warning: Joint data file not found: {joint_csv}")
        return []
    
    joint_data = pd.read_csv(joint_csv)
    
    # Extract data
    steps = joint_data['step'].values
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Generate one plot per joint
    plot_paths = []
    
    for joint_num in range(1, 7):
        joint_name = f"Joint_{joint_num}"
        joint_positions = joint_data[joint_name].values
        
        # Get joint limits
        limit_min, limit_max = JOINT_LIMITS[joint_name]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot joint position
        ax.plot(steps, joint_positions, linewidth=2, color='darkblue', alpha=0.8)
        
        # Add horizontal lines for limits
        ax.axhline(y=limit_min, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Min Limit ({limit_min:.3f} rad)', alpha=0.7)
        ax.axhline(y=limit_max, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Max Limit ({limit_max:.3f} rad)', alpha=0.7)
        
        # Set y-axis limits (add 10% padding)
        y_range = limit_max - limit_min
        padding = y_range * 0.1
        ax.set_ylim(limit_min - padding, limit_max + padding)
        
        # Labels and title
        ax.set_xlabel('Step Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Joint Position (radians)', fontsize=12, fontweight='bold')
        ax.set_title(f'{joint_name} Position Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add statistics text box
        stats_text = f'Mean: {np.mean(joint_positions):.3f} rad\n'
        stats_text += f'Max: {np.max(joint_positions):.3f} rad\n'
        stats_text += f'Min: {np.min(joint_positions):.3f} rad\n'
        stats_text += f'Range: {np.max(joint_positions) - np.min(joint_positions):.3f} rad'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{joint_name}_position.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[GraphingUtility] Saved {joint_name} plot: {plot_path}")
        plt.close(fig)
        
        plot_paths.append(plot_path)
    
    # Create combined 6-panel plot (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('All Joint Positions Over Time', fontsize=16, fontweight='bold')
    
    for joint_num in range(1, 7):
        joint_name = f"Joint_{joint_num}"
        joint_positions = joint_data[joint_name].values
        
        # Calculate subplot position (3 rows, 2 columns)
        row = (joint_num - 1) // 2
        col = (joint_num - 1) % 2
        ax = axes[row, col]
        
        # Get joint limits
        limit_min, limit_max = JOINT_LIMITS[joint_name]
        
        # Plot joint position
        ax.plot(steps, joint_positions, linewidth=2, color='darkblue', alpha=0.8)
        
        # Add horizontal lines for limits
        ax.axhline(y=limit_min, color='red', linestyle='--', linewidth=1, 
                   alpha=0.7, label=f'Limits')
        ax.axhline(y=limit_max, color='red', linestyle='--', linewidth=1, 
                   alpha=0.7)
        
        # Set y-axis limits (add 10% padding)
        y_range = limit_max - limit_min
        padding = y_range * 0.1
        ax.set_ylim(limit_min - padding, limit_max + padding)
        
        # Labels and title
        ax.set_xlabel('Step Number', fontsize=10, fontweight='bold')
        ax.set_ylabel('Position (rad)', fontsize=10, fontweight='bold')
        ax.set_title(f'{joint_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add compact statistics text box
        stats_text = f'Range: {np.max(joint_positions) - np.min(joint_positions):.3f} rad'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'all_joints_combined_6panel.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved combined 6-panel plot: {combined_plot_path}")
    plt.close(fig)
    
    plot_paths.append(combined_plot_path)
    
    print("[GraphingUtility] All joint plots generated successfully!")
    
    return plot_paths


def plot_all_joints_combined(data_file, output_dir="./joint_plots"):
    """Generate a combined plot showing all 6 joints together vs waypoints.
    
    Creates one large plot with all joints showing:
    - X-axis: Waypoint index
    - Y-axis: Joint position in radians
    - 6 different colored lines (one per joint)
    - Shaded regions showing joint limits
    
    Args:
        data_file: Base path to CSV file (will append _joints.csv and _steps.csv)
        output_dir: Directory to save plot images
    
    Returns:
        Path to generated plot file
    """
    import pandas as pd
    
    # Load joint data
    joint_csv = data_file.replace('.csv', '_joints.csv')
    if not os.path.exists(joint_csv):
        print(f"[GraphingUtility] Warning: Joint data file not found: {joint_csv}")
        return None
    
    joint_data = pd.read_csv(joint_csv)
    steps = joint_data['step'].values
    
    # Load speed data to get waypoint mapping
    step_csv = data_file.replace('.csv', '_steps.csv')
    if not os.path.exists(step_csv):
        print(f"[GraphingUtility] Warning: Step data file not found: {step_csv}")
        # Fallback: use step numbers as waypoints
        waypoint_indices_full = steps
        waypoint_label = "Step Number"
    else:
        speed_data = pd.read_csv(step_csv)
        # Get waypoint indices for each step (after settling)
        waypoint_indices = speed_data['active_waypoint'].values
        waypoint_label = "Waypoint Index"
        
        # Pad joint data to match speed data length if needed
        # (joint data starts from step 0, speed data starts after settling)
        if len(waypoint_indices) < len(steps):
            # Use step numbers for settling phase, then waypoints
            settling_steps = len(steps) - len(waypoint_indices)
            waypoint_indices_full = np.concatenate([
                np.full(settling_steps, -1),  # Mark settling phase as -1
                waypoint_indices
            ])
        else:
            waypoint_indices_full = waypoint_indices[:len(steps)]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with larger size for all joints
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define colors for each joint
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    joint_names = [f'Joint_{i}' for i in range(1, 7)]
    
    # Plot each joint
    for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
        joint_positions = joint_data[joint_name].values
        
        # Plot joint trajectory
        if len(waypoint_indices_full) == len(joint_positions):
            x_data = waypoint_indices_full
        else:
            x_data = steps
        
        ax.plot(x_data, joint_positions, 
                linewidth=2.5, color=color, alpha=0.9, 
                label=joint_name, marker='', linestyle='-')
        
        # Add shaded region for joint limits
        limit_min, limit_max = JOINT_LIMITS[joint_name]
        ax.axhline(y=limit_min, color=color, linestyle=':', linewidth=1, alpha=0.4)
        ax.axhline(y=limit_max, color=color, linestyle=':', linewidth=1, alpha=0.4)
        ax.fill_between(x_data, limit_min, limit_max, 
                        color=color, alpha=0.05, linewidth=0)
    
    # Labels and title
    ax.set_xlabel(waypoint_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Joint Position (radians)', fontsize=14, fontweight='bold')
    ax.set_title('All Joint Positions vs Waypoints', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Legend with limits
    legend_lines = []
    legend_labels = []
    for i, joint_name in enumerate(joint_names):
        limit_min, limit_max = JOINT_LIMITS[joint_name]
        legend_lines.append(plt.Line2D([0], [0], color=colors[i], linewidth=2.5))
        legend_labels.append(f'{joint_name} [{limit_min:.2f}, {limit_max:.2f}] rad')
    
    ax.legend(legend_lines, legend_labels, 
             loc='upper left', fontsize=10, 
             bbox_to_anchor=(1.01, 1), borderaxespad=0,
             title='Joint Limits (rad)', title_fontsize=11)
    
    # Add statistics box
    stats_text = "Joint Statistics (radians):\n"
    for i, joint_name in enumerate(joint_names):
        joint_positions = joint_data[joint_name].values
        stats_text += f"{joint_name}: "
        stats_text += f"μ={np.mean(joint_positions):.2f}, "
        stats_text += f"σ={np.std(joint_positions):.2f}\n"
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    # Tight layout to accommodate legend
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'all_joints_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved combined joints plot: {plot_path}")
    plt.close(fig)
    
    return plot_path


# ===================================================================
# ANALYSIS FUNCTIONS
# ===================================================================
def analyze_trajectory_speed(data_file):
    """Analyze and print speed statistics.
    
    Args:
        data_file: Base path to CSV files
    """
    import pandas as pd
    
    step_csv = data_file.replace('.csv', '_steps.csv')
    waypoint_csv = data_file.replace('.csv', '_waypoints.csv')
    
    step_data = pd.read_csv(step_csv)
    waypoint_data = pd.read_csv(waypoint_csv)
    
    step_speeds_mm_s = step_data['speed_mm_s'].values
    waypoint_speeds_mm_s = waypoint_data['calculated_speed_mm_s'].values
    
    print("\n" + "="*60)
    print("TRAJECTORY SPEED ANALYSIS")
    print("="*60)
    
    print("\nPer-Step Statistics:")
    print(f"  Total steps: {len(step_speeds_mm_s)}")
    print(f"  Mean speed: {np.mean(step_speeds_mm_s):.2f} mm/s")
    print(f"  Max speed: {np.max(step_speeds_mm_s):.2f} mm/s")
    print(f"  Min speed: {np.min(step_speeds_mm_s):.2f} mm/s")
    print(f"  Std dev: {np.std(step_speeds_mm_s):.2f} mm/s")
    
    print("\nPer-Waypoint Statistics (Calculated):")
    print(f"  Total transitions: {len(waypoint_speeds_mm_s)}")
    if len(waypoint_speeds_mm_s) > 0:
        print(f"  Mean speed: {np.mean(waypoint_speeds_mm_s):.2f} mm/s")
        print(f"  Max speed: {np.max(waypoint_speeds_mm_s):.2f} mm/s")
        print(f"  Min speed: {np.min(waypoint_speeds_mm_s):.2f} mm/s")
        print(f"  Std dev: {np.std(waypoint_speeds_mm_s):.2f} mm/s")
    
    # Print target speeds if available
    if 'target_speed_mm_s' in waypoint_data.columns:
        target_speeds = waypoint_data['target_speed_mm_s'].values
        valid_targets = target_speeds[~np.isnan(target_speeds)]
        if len(valid_targets) > 0:
            print("\nPer-Waypoint Statistics (Target from CSV):")
            print(f"  Total targets: {len(valid_targets)}")
            print(f"  Mean speed: {np.mean(valid_targets):.2f} mm/s")
            print(f"  Max speed: {np.max(valid_targets):.2f} mm/s")
            print(f"  Min speed: {np.min(valid_targets):.2f} mm/s")
            print(f"  Std dev: {np.std(valid_targets):.2f} mm/s")
    
    print("="*60 + "\n")


def analyze_joint_data(data_file):
    """Analyze and print joint position statistics.
    
    Args:
        data_file: Base path to CSV file
    """
    import pandas as pd
    
    joint_csv = data_file.replace('.csv', '_joints.csv')
    
    if not os.path.exists(joint_csv):
        print(f"[GraphingUtility] Warning: Joint data file not found: {joint_csv}")
        return
    
    joint_data = pd.read_csv(joint_csv)
    
    print("\n" + "="*60)
    print("JOINT POSITION ANALYSIS")
    print("="*60)
    
    for joint_num in range(1, 7):
        joint_name = f"Joint_{joint_num}"
        joint_positions = joint_data[joint_name].values
        limit_min, limit_max = JOINT_LIMITS[joint_name]
        
        print(f"\n{joint_name}:")
        print(f"  Mean: {np.mean(joint_positions):.3f} rad ({np.degrees(np.mean(joint_positions)):.1f}°)")
        print(f"  Max: {np.max(joint_positions):.3f} rad ({np.degrees(np.max(joint_positions)):.1f}°)")
        print(f"  Min: {np.min(joint_positions):.3f} rad ({np.degrees(np.min(joint_positions)):.1f}°)")
        print(f"  Range: {np.max(joint_positions) - np.min(joint_positions):.3f} rad")
        print(f"  Limit: [{limit_min:.3f}, {limit_max:.3f}] rad")
        
        # Check if within limits
        if np.max(joint_positions) > limit_max or np.min(joint_positions) < limit_min:
            print(f"  ⚠️  WARNING: Joint exceeds limits!")
    
    print("="*60 + "\n")


# ===================================================================
# POSE TRACKING PLOTTING FUNCTIONS
# ===================================================================
def plot_pose_tracking(data_file, output_dir="./"):
    """Generate pose tracking plots (3D trajectory and distance error).
    
    Args:
        data_file: Base path to CSV file (will append _poses.csv)
        output_dir: Directory to save plot images
    
    Returns:
        Dictionary with paths to generated plots
    """
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    
    # Load pose data
    pose_csv = data_file.replace('.csv', '_poses.csv')
    
    if not os.path.exists(pose_csv):
        print(f"[GraphingUtility] Warning: Pose data file not found: {pose_csv}")
        return {}
    
    pose_data = pd.read_csv(pose_csv)
    
    # Extract data
    steps = pose_data['step'].values
    target_pos = pose_data[['target_x', 'target_y', 'target_z']].values
    actual_pos = pose_data[['actual_x', 'actual_y', 'actual_z']].values
    
    # Calculate distances (in meters, then convert to mm)
    distances = np.linalg.norm(target_pos - actual_pos, axis=1) * 1000.0  # mm
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ==========================================
    # PLOT 1: 3D Trajectory Comparison
    # ==========================================
    fig1 = plt.figure(figsize=(14, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Plot target trajectory
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
            'b-', linewidth=2, alpha=0.7, label='Target (IK Command)')
    ax1.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2],
               c='blue', marker='o', s=100, label='Start')
    
    # Plot actual trajectory
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
            'r-', linewidth=2, alpha=0.7, label='Actual (End-Effector)')
    ax1.scatter(actual_pos[-1, 0], actual_pos[-1, 1], actual_pos[-1, 2],
               c='red', marker='s', s=100, label='End')
    
    # Labels and title
    ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Target vs Actual End-Effector Trajectory (Relative to Robot Base)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    stats_text = f'Position Error:\n'
    stats_text += f'  Mean: {mean_error:.2f} mm\n'
    stats_text += f'  Max: {max_error:.2f} mm\n'
    stats_text += f'  Min: {np.min(distances):.2f} mm\n'
    stats_text += f'  Std: {np.std(distances):.2f} mm'
    
    # Add text box to the figure
    fig1.text(0.02, 0.98, stats_text, transform=fig1.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'pose_3d_trajectory.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved 3D trajectory plot: {plot1_path}")
    plt.close(fig1)
    
    # ==========================================
    # PLOT 2: Distance Error Over Time
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    # Plot distance error
    ax2.plot(steps, distances, linewidth=2, color='darkred', alpha=0.8)
    ax2.fill_between(steps, 0, distances, color='red', alpha=0.2)
    
    # Add mean line
    ax2.axhline(y=mean_error, color='blue', linestyle='--', linewidth=2,
               label=f'Mean Error: {mean_error:.2f} mm', alpha=0.7)
    
    # Labels and title
    ax2.set_xlabel('Step Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position Error (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('Distance Between Target and Actual End-Effector Position', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add statistics box
    stats_text2 = f'Error Statistics (mm):\n'
    stats_text2 += f'  Mean: {mean_error:.2f}\n'
    stats_text2 += f'  Max: {max_error:.2f}\n'
    stats_text2 += f'  Min: {np.min(distances):.2f}\n'
    stats_text2 += f'  Std: {np.std(distances):.2f}'
    
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'pose_distance_error.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"[GraphingUtility] Saved distance error plot: {plot2_path}")
    plt.close(fig2)
    
    print("[GraphingUtility] All pose tracking plots generated successfully!")
    
    return {
        '3d_trajectory': plot1_path,
        'distance_error': plot2_path
    }


# ===================================================================
# MAIN PROCESSING FUNCTION
# ===================================================================
def save_and_plot_all_data(speed_collector, joint_collector, pose_collector, csv_basename, timestamp):
    """Save all collected data and generate all plots.
    
    This is the MAIN function called from irb_1300_example.py.
    Handles all directory creation, file saving, and plotting.
    
    Args:
        speed_collector: SpeedDataCollector instance
        joint_collector: JointDataCollector instance
        pose_collector: PoseTrackingCollector instance
        csv_basename: Base name of the CSV file (for naming outputs)
        timestamp: Timestamp string for unique filenames
    
    Returns:
        Dictionary with all file paths
    """
    print("\n[INFO] Processing data...")
    
    # Create results directory structure
    results_base = "./results"
    os.makedirs(results_base, exist_ok=True)
    
    # Create subdirectories
    speed_data_dir = os.path.join(results_base, "speed_data")
    joint_data_dir = os.path.join(results_base, "joint_data")
    os.makedirs(speed_data_dir, exist_ok=True)
    os.makedirs(joint_data_dir, exist_ok=True)
    
    # Define file paths
    speed_data_file = os.path.join(speed_data_dir, f"speed_{csv_basename}_{timestamp}.csv")
    joint_data_file = os.path.join(joint_data_dir, f"joint_{csv_basename}_{timestamp}.csv")
    pose_data_file = os.path.join(results_base, f"pose_{csv_basename}_{timestamp}.csv")
    
    # Save all data
    step_csv, waypoint_csv = speed_collector.save_data(speed_data_file)
    joint_csv = joint_collector.save_data(joint_data_file)
    pose_csv = pose_collector.save_data(pose_data_file)
    
    # Analyze and print statistics
    analyze_trajectory_speed(speed_data_file)
    analyze_joint_data(joint_data_file)
    
    # Generate all plots
    print("[INFO] Generating speed analysis plots...")
    plot_files = plot_speed_data(speed_data_file, output_dir=speed_data_dir)
    
    print("[INFO] Generating joint position plots...")
    joint_plot_files = plot_joint_data(joint_data_file, output_dir=joint_data_dir)
    
    print("[INFO] Generating combined joint plot...")
    combined_joint_plot = plot_all_joints_combined(joint_data_file, output_dir=joint_data_dir)
    
    print("[INFO] Generating pose tracking plots...")
    pose_plot_files = plot_pose_tracking(pose_data_file, output_dir=results_base)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n📁 All results saved to: {results_base}/")
    print("\nSpeed Data:")
    print(f"  Step data: {step_csv}")
    print(f"  Waypoint data: {waypoint_csv}")
    print(f"  Waypoint speeds plot: {plot_files['waypoint_plot']}")
    print(f"  Step speeds plot: {plot_files['step_plot']}")
    print(f"  Combined plot: {plot_files['combined_plot']}")
    print("\nJoint Data:")
    print(f"  Joint data: {joint_csv}")
    for i, plot_path in enumerate(joint_plot_files, 1):
        print(f"  Joint_{i} plot: {plot_path}")
    if combined_joint_plot:
        print(f"  All joints combined: {combined_joint_plot}")
    print("\nPose Tracking:")
    print(f"  Pose data: {pose_csv}")
    if pose_plot_files:
        if '3d_trajectory' in pose_plot_files:
            print(f"  3D trajectory plot: {pose_plot_files['3d_trajectory']}")
        if 'distance_error' in pose_plot_files:
            print(f"  Distance error plot: {pose_plot_files['distance_error']}")
    print("="*60 + "\n")
    
    # Return all paths
    return {
        'results_dir': results_base,
        'speed': {'step_csv': step_csv, 'waypoint_csv': waypoint_csv, 'plots': plot_files},
        'joint': {'csv': joint_csv, 'plots': joint_plot_files, 'combined_plot': combined_joint_plot},
        'pose': {'csv': pose_csv, 'plots': pose_plot_files}
    }


# ===================================================================
# STANDALONE USAGE
# ===================================================================
if __name__ == "__main__":
    """Standalone usage for analyzing existing data files."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graphing_utility.py <data_file.csv>")
        print("Example: python graphing_utility.py ./speed_data/speed_trajectory_20250113_120000.csv")
        print("Note: This will look for _steps.csv, _waypoints.csv, and _joints.csv files")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Check if speed files exist
    step_csv = data_file.replace('.csv', '_steps.csv')
    waypoint_csv = data_file.replace('.csv', '_waypoints.csv')
    
    if os.path.exists(step_csv) and os.path.exists(waypoint_csv):
        # Analyze speed statistics
        analyze_trajectory_speed(data_file)
        
        # Generate speed plots
        output_dir = os.path.dirname(data_file) or "./speed_plots"
        plot_speed_data(data_file, output_dir)
    else:
        print(f"[GraphingUtility] Speed data files not found, skipping speed analysis")
    
    # Check if joint file exists
    joint_csv = data_file.replace('.csv', '_joints.csv')
    
    if os.path.exists(joint_csv):
        # Analyze joint statistics
        analyze_joint_data(data_file)
        
        # Generate joint plots
        joint_output_dir = os.path.dirname(data_file) or "./joint_plots"
        plot_joint_data(data_file, joint_output_dir)
        
        # Generate combined joint plot
        plot_all_joints_combined(data_file, joint_output_dir)
    else:
        print(f"[GraphingUtility] Joint data file not found, skipping joint analysis")


