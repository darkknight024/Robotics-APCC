#!/usr/bin/env python3
"""
Base Solver Interfaces
======================

Abstract base classes that define the **unified solver contract** for all
FK and IK backends (EAIK analytical, Pinocchio numerical, and any future
additions).

This abstraction enables:

* **Transparent solver switching** — the ``create_solvers()`` factory in
  ``core/__init__.py`` instantiates the correct backend from a single
  config string (``"eaik"`` or ``"pin"``); all downstream scripts
  interact exclusively through the base interfaces.
* **Hybrid architectures** — because both solvers expose the same
  ``solve()`` / ``solve_with_retries()`` API, a sequential hybrid
  (analytical seed → numerical refinement) or concurrent hybrid
  (race both, take first success) can be composed without modifying
  calling code.

All solver backends must implement these interfaces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FKResult:
    """Result of forward kinematics computation."""
    position_m: np.ndarray    # [x, y, z] in meters
    quaternion: np.ndarray    # [qw, qx, qy, qz]
    rotation_matrix: np.ndarray  # 3x3 rotation matrix


@dataclass
class BaseIKConfig:
    """Base configuration shared by all IK solvers."""
    ee_frame_name: str = "Link_6"


class BaseFKSolver(ABC):
    """
    Abstract base class for Forward Kinematics solvers.

    All FK solver implementations must provide:
    - solve(q) -> FKResult
    - solve_batch(joint_positions) -> (positions, quaternions)
    - get_jacobian(q) -> 6xn matrix
    - ee_frame_name property
    - solver_name property (for labels/reports)
    """

    @property
    @abstractmethod
    def ee_frame_name(self) -> str:
        """Name of the end-effector frame."""
        ...

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Human-readable solver name for reports/plot labels."""
        ...

    @abstractmethod
    def solve(self, q: np.ndarray) -> FKResult:
        """Compute forward kinematics for a single joint configuration."""
        ...

    def solve_batch(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for multiple joint configurations.

        Default implementation calls solve() in a loop.
        Subclasses may override for vectorised performance.
        """
        n_waypoints = len(joint_positions)
        positions = np.zeros((n_waypoints, 3))
        quaternions = np.zeros((n_waypoints, 4))
        for i, q in enumerate(joint_positions):
            result = self.solve(q)
            positions[i] = result.position_m
            quaternions[i] = result.quaternion
        return positions, quaternions

    @abstractmethod
    def get_jacobian(self, q: np.ndarray, local_frame: bool = True) -> np.ndarray:
        """
        Compute 6xn Jacobian at the given configuration.

        Convention: [angular_vel (3); linear_vel (3)].
        """
        ...


class BaseIKSolver(ABC):
    """
    Abstract base class for Inverse Kinematics solvers.

    All IK solver implementations must provide:
    - solve(target_position, target_quaternion, q_init) -> (success, q, info)
    - solve_with_retries(...) -> (success, q, info)
    - solver_name property (for labels/reports)
    """

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Human-readable solver name for reports/plot labels."""
        ...

    @abstractmethod
    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK for a single target pose.

        Returns:
            success: Whether a valid solution was found
            q: Joint configuration (n_joints,)
            info: Dict with solver-specific diagnostics
        """
        ...

    @abstractmethod
    def solve_with_retries(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_random_retries: int = 3
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """Solve IK with retry / fallback strategy."""
        ...

    @staticmethod
    def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
        qw, qx, qy, qz = quat
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm < 1e-10:
            return np.eye(3)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        return R
