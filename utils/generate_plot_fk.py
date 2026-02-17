#!/usr/bin/env python3
"""
FK Comparison Plot Generation

Generates plots comparing FK results (all positions in millimeters):
1. Position Comparison: 1x3 subplot (X, Y, Z in mm)
2. Position Deltas: 1x3 subplot (ΔX, ΔY, ΔZ in mm)
3. Quaternion Comparison: 2x2 subplot (qw, qx, qy, qz)
4. Euclidean Error: Single plot with distance per waypoint (mm)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


def plot_position_comparison(
    ref_positions_mm: np.ndarray,
    computed_positions_mm: np.ndarray,
    output_path: str,
    title: str = "Position Comparison",
    ref_label: str = "Reference (RobotStudio)",
    computed_label: str = "Computed (FK)",
    adaptive_scale: bool = False
) -> None:
    """
    Plot position comparison between reference and computed FK results.
    
    Args:
        ref_positions_mm: Reference positions (n_waypoints, 3) in mm
        computed_positions_mm: Computed positions (n_waypoints, 3) in mm
        output_path: Path to save the output image
        title: Main plot title
        ref_label: Label for reference data
        computed_label: Label for computed data
        adaptive_scale: If False, use uniform scale across all subplots
    """
    waypoints = np.arange(len(ref_positions_mm))
    axis_names = ['X', 'Y', 'Z']
    
    if not adaptive_scale:
        all_data = [ref_positions_mm[:, i] for i in range(3)] + \
                   [computed_positions_mm[:, i] for i in range(3)]
        y_min, y_max = _compute_uniform_scale(all_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, name in enumerate(axis_names):
        ax = axes[idx]
        
        ax.plot(waypoints, ref_positions_mm[:, idx], 'b-o', 
                label=ref_label, linewidth=2, markersize=4)
        ax.plot(waypoints, computed_positions_mm[:, idx], 'r-s', 
                label=computed_label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} Position (mm)', fontweight='bold')
        ax.set_title(f'{name} Position Comparison', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_position_deltas(
    ref_positions_mm: np.ndarray,
    computed_positions_mm: np.ndarray,
    output_path: str,
    title: str = "Position Deltas",
    adaptive_scale: bool = False
) -> None:
    """
    Plot position deltas (computed - reference) in mm.
    
    Args:
        ref_positions_mm: Reference positions (n_waypoints, 3) in mm
        computed_positions_mm: Computed positions (n_waypoints, 3) in mm
        output_path: Path to save the output image
        title: Main plot title
        adaptive_scale: If False, use fixed or uniform scale
    """
    waypoints = np.arange(len(ref_positions_mm))
    axis_names = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    deltas = computed_positions_mm - ref_positions_mm
    
    if not adaptive_scale:
        max_abs_delta = np.max(np.abs(deltas))
        if max_abs_delta <= 2.0:
            y_min, y_max = -2.0, 2.0
        else:
            y_min, y_max = _compute_uniform_scale([deltas[:, i] for i in range(3)])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, color) in enumerate(zip(axis_names, colors)):
        ax = axes[idx]
        
        ax.plot(waypoints, deltas[:, idx], f'{color[0]}-o', linewidth=2, markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} Delta (mm)', fontweight='bold')
        ax.set_title(f'{name} Position Delta', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_quaternion_comparison(
    ref_quaternions: np.ndarray,
    computed_quaternions: np.ndarray,
    output_path: str,
    title: str = "Quaternion Comparison",
    ref_label: str = "Reference (RobotStudio)",
    computed_label: str = "Computed (FK)",
    adaptive_scale: bool = False
) -> None:
    """
    Plot quaternion comparison between reference and computed FK results.
    
    Args:
        ref_quaternions: Reference quaternions (n_waypoints, 4) [qw, qx, qy, qz]
        computed_quaternions: Computed quaternions (n_waypoints, 4) [qw, qx, qy, qz]
        output_path: Path to save the output image
        title: Main plot title
        ref_label: Label for reference data
        computed_label: Label for computed data
        adaptive_scale: If False, use uniform scale across all subplots
    """
    waypoints = np.arange(len(ref_quaternions))
    quat_names = ['Qw', 'Qx', 'Qy', 'Qz']
    
    if not adaptive_scale:
        all_data = [ref_quaternions[:, i] for i in range(4)] + \
                   [computed_quaternions[:, i] for i in range(4)]
        y_min, y_max = _compute_uniform_scale(all_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, name in enumerate(quat_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.plot(waypoints, ref_quaternions[:, idx], 'b-o', 
                label=ref_label, linewidth=2, markersize=4)
        ax.plot(waypoints, computed_quaternions[:, idx], 'r-s', 
                label=computed_label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(name, fontweight='bold')
        ax.set_title(f'{name} Comparison', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_euclidean_error(
    ref_positions_mm: np.ndarray,
    computed_positions_mm: np.ndarray,
    output_path: str,
    title: str = "Position Error (Euclidean Distance)",
    adaptive_scale: bool = False
) -> None:
    """
    Plot Euclidean distance error per waypoint.
    
    Args:
        ref_positions_mm: Reference positions (n_waypoints, 3) in mm
        computed_positions_mm: Computed positions (n_waypoints, 3) in mm
        output_path: Path to save the output image
        title: Main plot title
        adaptive_scale: If False, use fixed scale for small errors
    """
    waypoints = np.arange(len(ref_positions_mm))
    distances = np.linalg.norm(computed_positions_mm - ref_positions_mm, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(waypoints, distances, 'purple', marker='o', linewidth=2, markersize=5)
    ax.fill_between(waypoints, 0, distances, alpha=0.3, color='purple')
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Euclidean Distance (mm)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if not adaptive_scale and np.max(distances) <= 2.0:
        ax.set_ylim(0, 2.0)
    
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
