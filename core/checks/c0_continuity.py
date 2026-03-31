"""
C0 (position-level) continuity check.

Analyses the discrete joint sequence for large jumps that indicate
IK configuration flips (e.g. elbow-up -> elbow-down).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from utils.math import shortest_angular_distance, compute_joint_space_distance


@dataclass
class C0Result:
    """Output of a C0 continuity analysis."""
    passed: bool
    max_joint_delta: float
    per_joint_deltas: np.ndarray  # (n_segments, n_joints)
    joint_space_distances: np.ndarray  # (n_segments,) Euclidean norm
    flip_indices: List[int] = field(default_factory=list)


def compute_per_joint_deltas(joint_positions: np.ndarray) -> np.ndarray:
    """Compute per-joint absolute angular deltas between consecutive waypoints.

    Args:
        joint_positions: (n_waypoints, n_joints) in radians.

    Returns:
        (n_segments, n_joints) absolute angular deltas using shortest-path wrapping.
    """
    return np.abs(np.diff(joint_positions, axis=0))


def detect_config_flips(
    joint_positions: np.ndarray,
    flip_threshold_rad: float = 1.0,
) -> List[int]:
    """Return segment indices where the Euclidean joint delta exceeds *flip_threshold_rad*.

    A spike in ||delta_q|| typically means the IK solver flipped branch
    (e.g. elbow-up to elbow-down).
    """
    flips: List[int] = []
    for i in range(len(joint_positions) - 1):
        d = compute_joint_space_distance(joint_positions[i], joint_positions[i + 1])
        if d > flip_threshold_rad:
            flips.append(i)
    return flips


def check_c0_continuity(
    joint_positions: np.ndarray,
    joint_jump_limit_rad: Optional[float] = None,
    flip_threshold_rad: float = 1.0,
) -> C0Result:
    """Full C0 analysis: per-joint deltas, Euclidean distances, flip detection.

    Args:
        joint_positions: (n_waypoints, n_joints) in radians.
        joint_jump_limit_rad: If set, ``passed`` is False when max distance
            exceeds this value.  None means the check always passes.
        flip_threshold_rad: Threshold for config-flip detection.

    Returns:
        C0Result dataclass.
    """
    per_joint = compute_per_joint_deltas(joint_positions)
    dists = np.array([
        compute_joint_space_distance(joint_positions[i], joint_positions[i + 1])
        for i in range(len(joint_positions) - 1)
    ])
    max_delta = float(np.max(dists)) if len(dists) > 0 else 0.0
    flips = detect_config_flips(joint_positions, flip_threshold_rad)

    passed = True
    if joint_jump_limit_rad is not None and max_delta > joint_jump_limit_rad:
        passed = False

    return C0Result(
        passed=passed,
        max_joint_delta=max_delta,
        per_joint_deltas=per_joint,
        joint_space_distances=dists,
        flip_indices=flips,
    )
