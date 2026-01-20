"""
Core module for kinematic solvers and feasibility analysis.

This module provides:
- IKSolver: Inverse kinematics solver using Pinocchio
- FKSolver: Forward kinematics solver using Pinocchio
- FeasibilityAnalyzer: Trajectory feasibility analysis
"""

from .ik_solver import IKSolver, IKConfig
from utils.urdf_loader import load_robot_model
from .fk_solver import FKSolver, FKResult
from .feasibility_checks import (
    FeasibilityAnalyzer,
    FeasibilityResult,
    compute_manipulability,
    compute_singularity_proximity,
    compute_condition_number,
    check_reachability
)

__all__ = [
    'IKSolver',
    'IKConfig',
    'FKSolver',
    'FKResult',
    'FeasibilityAnalyzer',
    'FeasibilityResult',
    'load_robot_model',
    'compute_manipulability',
    'compute_singularity_proximity',
    'compute_condition_number',
    'check_reachability'
]
