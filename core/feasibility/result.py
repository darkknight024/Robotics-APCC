#!/usr/bin/env python3
"""Per-waypoint feasibility result (Phase 1 IK + metrics)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FeasibilityResult:
    """Result of feasibility analysis for a single waypoint."""

    is_reachable: bool
    manipulability: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray] = None
    ik_debug_info: Optional[Dict[str, Any]] = None
    target_position: Optional[np.ndarray] = None
    target_quaternion: Optional[np.ndarray] = None
    joint_velocity_ratio: Optional[float] = None
    distance_to_joint_limits: Optional[float] = None
    joint_space_distance: Optional[float] = None
    translational_manipulability: Optional[float] = None
    rotational_manipulability: Optional[float] = None
    normalized_manipulability: Optional[float] = None
    directional_manipulability: Optional[float] = None
