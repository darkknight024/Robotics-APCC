#!/usr/bin/env python3
"""
IK Comparison Plot Generation

Generates plots comparing IK results:
1. Joint Angles Comparison: 2x3 subplot (J1-J6) comparing reference vs computed
2. Joint Deltas: 2x3 subplot showing |reference - computed| per joint

All angles displayed in degrees.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


def plot_joint_comparison(
    ref_joints_deg: np.ndarray,
    computed_joints_deg: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Comparison",
    ref_label: str = "Reference (RobotStudio)",
    computed_label: str = "Computed (IK)",
    adaptive_scale: bool = False
) -> None:
    """
    Plot joint angle comparison between reference and computed values.
    
    Generates a 2x3 subplot grid comparing joint angles J1-J6.
    
    Args:
        ref_joints_deg: Reference joint angles (n_waypoints, 6) in degrees
        computed_joints_deg: Computed joint angles (n_waypoints, 6) in degrees
        output_path: Path to save the output image
        title: Main plot title
        ref_label: Label for reference data in legend
        computed_label: Label for computed data in legend
        adaptive_scale: If False, use uniform scale across all subplots
    """
    waypoints = np.arange(len(ref_joints_deg))
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    # Compute uniform scale if needed
    if not adaptive_scale:
        all_data = [ref_joints_deg[:, i] for i in range(6)] + \
                   [computed_joints_deg[:, i] for i in range(6)]
        y_min, y_max = _compute_uniform_scale(all_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, name in enumerate(joint_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.plot(waypoints, ref_joints_deg[:, idx], 'b-o', 
                label=ref_label, linewidth=2, markersize=3)
        ax.plot(waypoints, computed_joints_deg[:, idx], 'r-s', 
                label=computed_label, linewidth=2, markersize=3)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} (deg)', fontweight='bold')
        ax.set_title(f'{name} Comparison', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_deltas(
    ref_joints_deg: np.ndarray,
    computed_joints_deg: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Errors",
    adaptive_scale: bool = False
) -> None:
    """
    Plot joint angle errors (absolute difference) between reference and computed.
    
    Generates a 2x3 subplot grid showing |reference - computed| for J1-J6.
    
    Args:
        ref_joints_deg: Reference joint angles (n_waypoints, 6) in degrees
        computed_joints_deg: Computed joint angles (n_waypoints, 6) in degrees
        output_path: Path to save the output image
        title: Main plot title
        adaptive_scale: If False, use uniform scale across all subplots
    """
    waypoints = np.arange(len(ref_joints_deg))
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    
    # Compute absolute errors
    errors = np.abs(ref_joints_deg - computed_joints_deg)
    
    # Compute uniform scale if needed
    if not adaptive_scale:
        y_min, y_max = _compute_uniform_scale([errors[:, i] for i in range(6)])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, color) in enumerate(zip(joint_names, colors)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.plot(waypoints, errors[:, idx], '-o', 
                linewidth=2, markersize=3, color=color)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} Error (deg)', fontweight='bold')
        ax.set_title(f'{name} Error |Ref - Computed|', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _compute_uniform_scale(data_arrays: List[np.ndarray]) -> tuple:
    """Compute uniform y-axis scale for multiple data arrays."""
    valid_arrays = [arr for arr in data_arrays if len(arr) > 0]
    if not valid_arrays:
        return -1, 1
    
    all_min = min(np.nanmin(arr) for arr in valid_arrays)
    all_max = max(np.nanmax(arr) for arr in valid_arrays)
    
    if np.isnan(all_min) or np.isnan(all_max):
        return -1, 1
    
    if all_min == all_max:
        margin = 0.1 if all_min == 0 else abs(all_min) * 0.1
    else:
        margin = (all_max - all_min) * 0.1
    
    return all_min - margin, all_max + margin
