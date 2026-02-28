#!/usr/bin/env python3
"""
FK Solver Module — EAIK Forward Kinematics
===========================================

Forward kinematics using the EAIK product-of-exponentials formulation.

EAIK computes FK to the last actuated link; the fixed ``ee_transform_4x4``
(from ``RobotModel``) is applied as post-processing to reach the
configured end-effector frame.

Jacobian computation uses **central finite differences** (ε = 1e-8)
rather than an analytical derivative.  This is adequate for feasibility
analysis and manipulability scoring but introduces a small numerical
error compared to Pinocchio's analytical Jacobian.  If high-fidelity
Jacobian-dependent operations are required (e.g., optimal control with
Crocoddyl), prefer the Pinocchio FK backend.

See Also
--------
* ``core/pin_fk_solver.py`` — Pinocchio FK with analytical Jacobian.
* ``core/base_solvers.py``  — abstract ``BaseFKSolver`` interface.
"""

import numpy as np
from typing import Tuple

from core.base_solvers import BaseFKSolver, FKResult
from utils.urdf_loader import RobotModel


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Args:
        R: 3x3 rotation matrix

    Returns:
    
        Quaternion as [qw, qx, qy, qz]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])


def _log_rotation(R: np.ndarray) -> np.ndarray:
    """
    Compute the logarithmic map of a rotation matrix (rotation vector).

    Args:
        R: 3x3 rotation matrix

    Returns:
        3-vector (rotation vector / axis-angle representation)
    """
    trace = np.trace(R)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if abs(angle) < 1e-10:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    elif abs(angle - np.pi) < 1e-6:
        M = R + np.eye(3)
        norms = [np.linalg.norm(M[:, i]) for i in range(3)]
        k = np.argmax(norms)
        v = M[:, k] / norms[k]
        return v * angle
    else:
        factor = angle / (2.0 * np.sin(angle))
        return factor * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class EAIKFKSolver(BaseFKSolver):
    """
    Forward Kinematics solver using EAIK.

    Example:
        robot_model = load_robot_model(urdf_path)
        solver = EAIKFKSolver(robot_model)
        result = solver.solve(q)
        print(f"Position: {result.position_m}")
    """

    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model
        self.n_joints = robot_model.n_joints
        self._ee_frame_name = robot_model.ee_frame_name

    @property
    def ee_frame_name(self) -> str:
        return self._ee_frame_name

    @property
    def solver_name(self) -> str:
        return "EAIK"

    def solve(self, q: np.ndarray) -> FKResult:
        """
        Compute forward kinematics for given joint configuration.

        EAIK computes FK to the last actuated link. The ee_transform_4x4
        is applied as post-processing to get the true end-effector pose.
        """
        T_link = self.robot_model.eaik_robot.fwdkin(q)
        T = T_link @ self.robot_model.ee_transform_4x4

        position = T[:3, 3].copy()
        rotation = T[:3, :3].copy()
        quaternion = _rotation_matrix_to_quaternion(rotation)

        return FKResult(
            position_m=position,
            quaternion=quaternion,
            rotation_matrix=rotation
        )

    def get_jacobian(self, q: np.ndarray, local_frame: bool = True) -> np.ndarray:
        """
        Compute Jacobian at given configuration using central finite differences.

        Uses the convention [angular; linear].
        """
        eps = 1e-8
        n = self.n_joints
        J = np.zeros((6, n))
        ee_T = self.robot_model.ee_transform_4x4

        T0 = self.robot_model.eaik_robot.fwdkin(q) @ ee_T
        R0 = T0[:3, :3]

        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            T_plus = self.robot_model.eaik_robot.fwdkin(q_plus) @ ee_T
            T_minus = self.robot_model.eaik_robot.fwdkin(q_minus) @ ee_T

            dp = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * eps)
            dR = T_minus[:3, :3].T @ T_plus[:3, :3]
            omega_local = _log_rotation(dR) / (2.0 * eps)

            if local_frame:
                J[:3, i] = omega_local
                J[3:, i] = R0.T @ dp
            else:
                J[:3, i] = R0 @ omega_local
                J[3:, i] = dp

        return J
