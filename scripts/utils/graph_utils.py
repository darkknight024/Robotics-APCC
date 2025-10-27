#!/usr/bin/env python3
"""
Graph utilities for ABB IRB 1300 Trajectory Analysis

This module contains all plotting and visualization functions for the trajectory analysis.
It provides clean, professional-looking plots with proper legends, colors, and formatting.

Author: Robotics Analysis Tool
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from typing import List, Dict, Tuple, Optional


class TrajectoryPlotter:
    """Main class for handling all trajectory-related plotting."""

    def __init__(self):
        """Initialize the plotter with color schemes and styling."""
        # Color schemes
        self.colors = {
            'axes': {'x': '#FF4444', 'y': '#44FF44', 'z': '#4444FF'},  # RGB colors
            'reachable': '#228B22',  # Forest green for reachable
            'unreachable': '#DC143C',  # Crimson for unreachable
            'trajectory': plt.cm.tab10,  # Use tab10 colormap for trajectories
            'warning': '#FF8C00',  # Dark orange for warnings
            'background': '#F8F8F8'  # Light gray background
        }

        # Marker styles
        self.markers = {
            'reachable': 's',  # Square for reachable
            'unreachable': 'o',  # Circle for unreachable
            'start': 'o',  # Circle for start
            'end': 's',  # Square for end
            'robot_base': 'D'  # Diamond for robot base
        }

        # Set default matplotlib styling
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.alpha'] = 0.3

    def setup_3d_axes(self, ax: Axes3D, title: str = "") -> None:
        """Set up 3D axes with RGB colored labels and proper styling."""
        ax.set_xlabel('X (meters)', color=self.colors['axes']['x'], fontweight='bold', fontsize=12)
        ax.set_ylabel('Y (meters)', color=self.colors['axes']['y'], fontweight='bold', fontsize=12)
        ax.set_zlabel('Z (meters)', color=self.colors['axes']['z'], fontweight='bold', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Color the axis ticks to match labels
        ax.tick_params(axis='x', colors=self.colors['axes']['x'])
        ax.tick_params(axis='y', colors=self.colors['axes']['y'])
        ax.tick_params(axis='z', colors=self.colors['axes']['z'])

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

    def create_coordinate_frame(self, ax: Axes3D, origin: np.ndarray = np.array([0, 0, 0]),
                               scale: float = 0.1) -> None:
        """Create a colored coordinate frame (RGB axes) at the specified origin."""
        x_axis = np.array([scale, 0, 0])
        y_axis = np.array([0, scale, 0])
        z_axis = np.array([0, 0, scale])

        # Draw RGB axes
        ax.quiver(origin[0], origin[1], origin[2],
                 x_axis[0], x_axis[1], x_axis[2],
                 color=self.colors['axes']['x'], linewidth=3, arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2],
                 y_axis[0], y_axis[1], y_axis[2],
                 color=self.colors['axes']['y'], linewidth=3, arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2],
                 z_axis[0], z_axis[1], z_axis[2],
                 color=self.colors['axes']['z'], linewidth=3, arrow_length_ratio=0.3)

    def plot_3d_trajectory_comparison(self, results: List[Dict], output_path: str,
                                    title: str = "3D Trajectory Analysis: Multiple Trajectories") -> None:
        """
        Generate a clean 3D trajectory comparison plot.

        Args:
            results: List of result dictionaries from trajectory analysis
            output_path: Path to save the plot
            title: Plot title
        """
        print("Generating clean 3D trajectory comparison plot...")

        # Group results by trajectory_id
        trajectories = {}
        for r in results:
            traj_id = r['trajectory_id']
            if traj_id not in trajectories:
                trajectories[traj_id] = []
            trajectories[traj_id].append(r)

        traj_ids = sorted(trajectories.keys())
        colors = [self.colors['trajectory'](i/len(traj_ids)) for i in range(len(traj_ids))]

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Set up axes and coordinate frame
        self.setup_3d_axes(ax, title)
        self.create_coordinate_frame(ax)

        # Plot robot base
        ax.scatter([0], [0], [0], c='black', marker=self.markers['robot_base'],
                  s=300, linewidths=2, edgecolors='white', label='Robot Base', zorder=10)

        # Statistics for legend
        total_waypoints = len(results)
        reachable_waypoints = sum(1 for r in results if r['reachable'])
        unreachable_waypoints = total_waypoints - reachable_waypoints
        reachability_rate = (reachable_waypoints / total_waypoints) * 100 if total_waypoints > 0 else 0

        # Plot each trajectory
        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            # Extract target trajectory
            target_x = [r['x_m'] for r in traj_results]
            target_y = [r['y_m'] for r in traj_results]
            target_z = [r['z_m'] for r in traj_results]

            # Extract reachable and unreachable points
            reachable_results = [r for r in traj_results if r['reachable']]
            unreachable_results = [r for r in traj_results if not r['reachable']]

            reachable_x = [r['x_m'] for r in reachable_results]
            reachable_y = [r['y_m'] for r in reachable_results]
            reachable_z = [r['z_m'] for r in reachable_results]

            unreachable_x = [r['x_m'] for r in unreachable_results]
            unreachable_y = [r['y_m'] for r in unreachable_results]
            unreachable_z = [r['z_m'] for r in unreachable_results]

            # Plot start and end points with different markers
            if target_x:
                # Start point (circle)
                ax.scatter(target_x[0], target_y[0], target_z[0],
                          color=color, marker=self.markers['start'], s=200,
                          linewidths=2, edgecolors='black', alpha=0.8,
                          label=f'Traj {traj_id} Start' if i == 0 else "", zorder=5)

                # End point (square)
                ax.scatter(target_x[-1], target_y[-1], target_z[-1],
                          color=color, marker=self.markers['end'], s=200,
                          linewidths=2, edgecolors='black', alpha=0.8,
                          label=f'Traj {traj_id} End' if i == 0 else "", zorder=5)

            # Plot trajectory path (thin line)
            if len(target_x) > 1:
                ax.plot(target_x, target_y, target_z, color=color, linestyle='-',
                       linewidth=1, alpha=0.4, label=f'Target Path {traj_id}' if i == 0 else "")

            # Plot reachable waypoints (squares)
            if reachable_x:
                ax.scatter(reachable_x, reachable_y, reachable_z,
                          color=color, marker=self.markers['reachable'], s=80,
                          linewidths=1, edgecolors='black', alpha=0.7,
                          label=f'Traj {traj_id} Reachable' if i == 0 else "", zorder=3)

            # Plot unreachable waypoints (circles)
            if unreachable_x:
                ax.scatter(unreachable_x, unreachable_y, unreachable_z,
                          color=color, marker=self.markers['unreachable'], s=80,
                          linewidths=1, edgecolors='black', alpha=0.7,
                          label=f'Traj {traj_id} Unreachable' if i == 0 else "", zorder=3)

        # Set equal aspect ratio based on all data
        all_x, all_y, all_z = [0], [0], [0]  # Include robot base
        for traj_id in traj_ids:
            traj_results = trajectories[traj_id]
            target_x = [r['x_m'] for r in traj_results]
            target_y = [r['y_m'] for r in traj_results]
            target_z = [r['z_m'] for r in traj_results]
            all_x.extend(target_x)
            all_y.extend(target_y)
            all_z.extend(target_z)

        if all_x:
            max_range = np.array([
                max(all_x) - min(all_x),
                max(all_y) - min(all_y),
                max(all_z) - min(all_z)
            ]).max() / 2.0

            mid_x = (max(all_x) + min(all_x)) * 0.5
            mid_y = (max(all_y) + min(all_y)) * 0.5
            mid_z = (max(all_z) + min(all_z)) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Create comprehensive legend
        legend_elements = []

        # Robot base
        legend_elements.append(mpatches.Circle((0.5, 0.5), 0.1, facecolor='black',
                                             edgecolor='white', linewidth=2,
                                             label='Robot Base'))

        # Trajectory markers
        for i, traj_id in enumerate(traj_ids[:3]):  # Show first 3 trajectories in legend
            color = colors[i]
            legend_elements.append(mpatches.Circle((0.5, 0.5), 0.1, facecolor=color,
                                                 edgecolor='black', linewidth=1,
                                                 label=f'Trajectory {traj_id}'))

        # Waypoint markers
        legend_elements.append(mpatches.Rectangle((0.4, 0.4), 0.2, 0.2,
                                                facecolor=self.colors['reachable'],
                                                edgecolor='black', linewidth=1,
                                                label='Reachable Waypoint'))

        legend_elements.append(mpatches.Circle((0.5, 0.5), 0.1,
                                             facecolor=self.colors['unreachable'],
                                             edgecolor='black', linewidth=1,
                                             label='Unreachable Waypoint'))

        # Statistics text box
        stats_text = f'Total Trajectories: {len(traj_ids)}\n'
        stats_text += f'Total Waypoints: {total_waypoints}\n'
        stats_text += f'Reachable: {reachable_waypoints} ({reachability_rate:.1f}%)\n'
        stats_text += f'Unreachable: {unreachable_waypoints} ({100-reachability_rate:.1f}%)'

        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.legend(handles=legend_elements, fontsize=9, loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {output_path}")
        plt.close()

    def plot_reachability_analysis(self, results: List[Dict], output_path: str,
                                 title: str = "Kinematic Reachability Analysis") -> None:
        """
        Generate a reachability plot showing percentage of reachable points per trajectory.

        Args:
            results: List of result dictionaries from trajectory analysis
            output_path: Path to save the plot
            title: Plot title
        """
        print("Generating reachability analysis plot...")

        # Group results by trajectory_id
        trajectories = {}
        for r in results:
            traj_id = r['trajectory_id']
            if traj_id not in trajectories:
                trajectories[traj_id] = []
            trajectories[traj_id].append(r)

        traj_ids = sorted(trajectories.keys())
        colors = [self.colors['trajectory'](i/len(traj_ids)) for i in range(len(traj_ids))]

        # Calculate reachability percentages for each trajectory
        reachability_data = []
        for traj_id in traj_ids:
            traj_results = trajectories[traj_id]
            total_points = len(traj_results)
            reachable_points = sum(1 for r in traj_results if r['reachable'])
            reachability_rate = (reachable_points / total_points) * 100 if total_points > 0 else 0

            reachability_data.append({
                'trajectory_id': traj_id,
                'total_points': total_points,
                'reachable_points': reachable_points,
                'reachability_rate': reachability_rate
            })

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Reachability rate per trajectory (bar chart)
        traj_names = [f'Traj {data["trajectory_id"]}' for data in reachability_data]
        reachability_rates = [data['reachability_rate'] for data in reachability_data]
        reachable_counts = [data['reachable_points'] for data in reachability_data]
        total_counts = [data['total_points'] for data in reachability_data]

        bars = ax1.bar(range(len(traj_ids)), reachability_rates, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)

        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, reachability_rates)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax1.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reachability Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Reachability Rate by Trajectory', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(traj_ids)))
        ax1.set_xticklabels(traj_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 105])

        # Right plot: Reachable vs Unreachable counts (stacked bar)
        bottom_bars = ax2.bar(range(len(traj_ids)), reachable_counts, color=self.colors['reachable'],
                             alpha=0.7, edgecolor='black', linewidth=1, label='Reachable')
        top_bars = ax2.bar(range(len(traj_ids)), [t-r for t, r in zip(total_counts, reachable_counts)],
                          bottom=reachable_counts, color=self.colors['unreachable'],
                          alpha=0.7, edgecolor='black', linewidth=1, label='Unreachable')

        # Add count labels
        for i in range(len(traj_ids)):
            # Total count at top
            ax2.text(i, total_counts[i] + max(total_counts)*0.01,
                    f'{total_counts[i]}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Reachable count in middle
            if reachable_counts[i] > 0:
                ax2.text(i, reachable_counts[i]/2, f'{reachable_counts[i]}',
                        ha='center', va='center', fontweight='bold', fontsize=9, color='white')

        ax2.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Waypoints', fontsize=12, fontweight='bold')
        ax2.set_title('Waypoint Count by Trajectory', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(traj_ids)))
        ax2.set_xticklabels(traj_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=10)

        # Overall statistics
        total_waypoints = sum(total_counts)
        total_reachable = sum(reachable_counts)
        overall_rate = (total_reachable / total_waypoints) * 100 if total_waypoints > 0 else 0

        stats_text = f'Overall Statistics:\n'
        stats_text += f'Total Waypoints: {total_waypoints}\n'
        stats_text += f'Reachable: {total_reachable} ({overall_rate:.1f}%)\n'
        stats_text += f'Unreachable: {total_waypoints-total_reachable} ({100-overall_rate:.1f}%)'

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {output_path}")
        plt.close()

    def plot_manipulability_analysis(self, results: List[Dict], output_path: str,
                                   title: str = "Manipulability Analysis") -> None:
        """Generate manipulability analysis plot."""
        print("Generating manipulability analysis plot...")

        # Group results by trajectory_id
        trajectories = {}
        for r in results:
            traj_id = r['trajectory_id']
            if traj_id not in trajectories:
                trajectories[traj_id] = []
            trajectories[traj_id].append(r)

        traj_ids = sorted(trajectories.keys())
        colors = [self.colors['trajectory'](i/len(traj_ids)) for i in range(len(traj_ids))]

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            # Filter reachable results
            reachable_results = [r for r in traj_results if r['reachable'] and r['manipulability'] is not None]

            if reachable_results:
                indices = [r['waypoint_index'] for r in reachable_results]
                manipulability = [r['manipulability'] for r in reachable_results]

                ax.plot(indices, manipulability, color=color, linewidth=2,
                       marker='o', markersize=4, alpha=0.7, label=f'Trajectory {traj_id}')

        ax.set_xlabel('Waypoint Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Manipulability Index', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add warning threshold
        ax.axhline(y=0.1, color=self.colors['warning'], linestyle='--', linewidth=2,
                  label='Warning Threshold (0.1)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {output_path}")
        plt.close()

    def plot_singularity_analysis(self, results: List[Dict], output_path: str,
                                title: str = "Singularity Analysis") -> None:
        """Generate singularity analysis plot."""
        print("Generating singularity analysis plot...")

        # Group results by trajectory_id
        trajectories = {}
        for r in results:
            traj_id = r['trajectory_id']
            if traj_id not in trajectories:
                trajectories[traj_id] = []
            trajectories[traj_id].append(r)

        traj_ids = sorted(trajectories.keys())
        colors = [self.colors['trajectory'](i/len(traj_ids)) for i in range(len(traj_ids))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot minimum singular values
        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            reachable_results = [r for r in traj_results if r['reachable'] and r['min_singular_value'] is not None]

            if reachable_results:
                indices = [r['waypoint_index'] for r in reachable_results]
                min_singular_values = [r['min_singular_value'] for r in reachable_results]

                ax1.plot(indices, min_singular_values, color=color, linewidth=2,
                        marker='o', markersize=4, alpha=0.7, label=f'Trajectory {traj_id}')

        ax1.set_xlabel('Waypoint Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Minimum Singular Value', fontsize=12, fontweight='bold')
        ax1.set_title('Minimum Singular Value', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.01, color=self.colors['warning'], linestyle='--', linewidth=2,
                   label='Warning Threshold (0.01)')

        # Plot condition numbers
        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            reachable_results = [r for r in traj_results if r['reachable'] and r['condition_number'] is not None and r['condition_number'] != np.inf]

            if reachable_results:
                indices = [r['waypoint_index'] for r in reachable_results]
                condition_numbers = [r['condition_number'] for r in reachable_results]

                ax2.plot(indices, condition_numbers, color=color, linewidth=2,
                        marker='s', markersize=4, alpha=0.7, label=f'Trajectory {traj_id}')

        ax2.set_xlabel('Waypoint Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Condition Number', fontsize=12, fontweight='bold')
        ax2.set_title('Condition Number', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color=self.colors['warning'], linestyle='--', linewidth=2,
                   label='Warning Threshold (100)')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {output_path}")
        plt.close()

    def plot_joint_angles_analysis(self, results: List[Dict], model, output_path: str,
                                 title: str = "Joint Angles Analysis") -> None:
        """Generate joint angles analysis plot."""
        print("Generating joint angles analysis plot...")

        # Group results by trajectory_id
        trajectories = {}
        for r in results:
            traj_id = r['trajectory_id']
            if traj_id not in trajectories:
                trajectories[traj_id] = []
            trajectories[traj_id].append(r)

        traj_ids = sorted(trajectories.keys())
        colors = [self.colors['trajectory'](i/len(traj_ids)) for i in range(len(traj_ids))]

        # Filter reachable waypoints
        reachable_results = [r for r in results if r['reachable']]

        if not reachable_results:
            print("  - No reachable waypoints to plot joint angles")
            return

        joint_names = ['Joint 1 (Base)', 'Joint 2 (Shoulder)', 'Joint 3 (Elbow)',
                      'Joint 4 (Wrist Roll)', 'Joint 5 (Wrist Bend)', 'Joint 6 (Wrist Twist)']

        # Joint limits
        lower_limits = model.lowerPositionLimit
        upper_limits = model.upperPositionLimit

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i in range(6):
            ax = axes[i]

            # Plot joint angles for each trajectory
            for j, traj_id in enumerate(traj_ids):
                traj_results = trajectories[traj_id]
                color = colors[j]

                traj_reachable_results = [r for r in traj_results if r['reachable']]
                if not traj_reachable_results:
                    continue

                indices = [r['waypoint_index'] for r in traj_reachable_results]
                joint_angles = [r[f'q{i+1}_rad'] for r in traj_reachable_results]

                ax.plot(indices, joint_angles, color=color, linewidth=2,
                       marker='o', markersize=3, alpha=0.7, label=f'Trajectory {traj_id}' if i == 0 else "")

            # Plot joint limits
            ax.axhline(y=lower_limits[i], color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label='Joint Limits' if i == 0 else "")
            ax.axhline(y=upper_limits[i], color='red', linestyle='--', linewidth=2,
                      alpha=0.7)

            # Fill valid range
            ax.fill_between([min([r['waypoint_index'] for r in reachable_results]),
                           max([r['waypoint_index'] for r in reachable_results])],
                           lower_limits[i], upper_limits[i], alpha=0.1, color='green')

            ax.set_xlabel('Waypoint Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Angle (rad)', fontsize=10, fontweight='bold')
            ax.set_title(joint_names[i], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Show legend only on first subplot
            if i == 0:
                ax.legend(fontsize=8)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {output_path}")
        plt.close()


def generate_all_analysis_plots(results: List[Dict], model, output_dir: str) -> None:
    """
    Generate all analysis plots using the clean plotting utilities.

    Args:
        results: List of result dictionaries from trajectory analysis
        model: Pinocchio robot model
        output_dir: Output directory for saving plots
    """
    plotter = TrajectoryPlotter()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    plotter.plot_3d_trajectory_comparison(
        results, os.path.join(output_dir, 'trajectory_3d_comparison.png')
    )

    plotter.plot_reachability_analysis(
        results, os.path.join(output_dir, 'reachability_plot.png')
    )

    plotter.plot_manipulability_analysis(
        results, os.path.join(output_dir, 'manipulability_plot.png')
    )

    plotter.plot_singularity_analysis(
        results, os.path.join(output_dir, 'singularity_measure_plot.png')
    )

    plotter.plot_joint_angles_analysis(
        results, model, os.path.join(output_dir, 'joint_angles_plot.png')
    )
