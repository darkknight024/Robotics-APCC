#!/usr/bin/env python3
"""
FK Solver Module - Pinocchio Forward Kinematics

Provides a clean abstraction for forward kinematics computation using Pinocchio.

Restored from commit d78ff39, adapted to inherit from BaseFKSolver.
"""

import numpy as np
import pinocchio as pin
from typing import Tuple, Optional
from dataclasses import dataclass

from core.base_solvers import BaseFKSolver, FKResult


class PinocchioFKSolver(BaseFKSolver):
    """
    Forward Kinematics solver using Pinocchio.
    
    Example:
        model, data = load_robot_model(urdf_path)
        solver = PinocchioFKSolver(model, data)
        result = solver.solve(q)
        print(f"Position: {result.position_m}")
    """
    
    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        ee_frame_name: str = "ee_link"
    ):
        """
        Initialize FK solver.
        
        Args:
            model: Pinocchio robot model
            data: Pinocchio data object
            ee_frame_name: Name of end-effector frame in URDF
        """
        self.model = model
        self.data = data
        self._ee_frame_name = ee_frame_name
        
        try:
            self.ee_frame_id = model.getFrameId(ee_frame_name)
        except Exception as e:
            raise ValueError(f"Frame '{ee_frame_name}' not found in model: {e}")

    @property
    def ee_frame_name(self) -> str:
        return self._ee_frame_name

    @property
    def solver_name(self) -> str:
        return "Pinocchio"

    def solve(self, q: np.ndarray) -> FKResult:
        """
        Compute forward kinematics for given joint configuration.
        
        Args:
            q: Joint configuration (n_joints,)
            
        Returns:
            FKResult with position, quaternion, and rotation matrix
        """
        # Compute FK
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector pose
        ee_pose = self.data.oMf[self.ee_frame_id]
        position = ee_pose.translation.copy()
        rotation = ee_pose.rotation.copy()
        
        # Convert rotation to quaternion
        quat = pin.Quaternion(rotation)
        quaternion = np.array([quat.w, quat.x, quat.y, quat.z])
        
        return FKResult(
            position_m=position,
            quaternion=quaternion,
            rotation_matrix=rotation
        )
    
    def solve_batch(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for multiple joint configurations.
        
        Args:
            joint_positions: Joint configurations (n_waypoints, n_joints)
            
        Returns:
            positions: End-effector positions (n_waypoints, 3) in meters
            quaternions: End-effector quaternions (n_waypoints, 4) [qw, qx, qy, qz]
        """
        n_waypoints = len(joint_positions)
        positions = np.zeros((n_waypoints, 3))
        quaternions = np.zeros((n_waypoints, 4))
        
        for i, q in enumerate(joint_positions):
            result = self.solve(q)
            positions[i] = result.position_m
            quaternions[i] = result.quaternion
        
        return positions, quaternions
    
    def get_jacobian(self, q: np.ndarray, local_frame: bool = True) -> np.ndarray:
        """
        Compute Jacobian at given configuration.
        
        Args:
            q: Joint configuration
            local_frame: If True, use LOCAL frame; else use WORLD frame
            
        Returns:
            jacobian: 6xn Jacobian matrix
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        frame_type = pin.LOCAL if local_frame else pin.WORLD
        return pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_frame_id, frame_type
        )
