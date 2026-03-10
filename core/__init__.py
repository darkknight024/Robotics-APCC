"""
Core module for kinematic solvers and feasibility analysis.

Provides:
- Base abstract classes: BaseFKSolver, BaseIKSolver, BaseIKConfig, FKResult
- EAIK backend:  EAIKFKSolver, EAIKIKSolver, EAIKConfig
- Pinocchio backend: PinocchioFKSolver, PinocchioIKSolver, PinocchioIKConfig
- create_solvers() factory to instantiate the right backend from config
- FeasibilityAnalyzer and helpers
- RobotModel dataclass
"""

# --- Base classes ---
from .base_solvers import BaseFKSolver, BaseIKSolver, BaseIKConfig, FKResult

# --- EAIK backend ---
from .eaik_fk_solver import EAIKFKSolver
from .eaik_ik_solver import EAIKIKSolver, EAIKConfig

# --- Robot model ---
from utils.urdf_loader import load_robot_model, load_robot_model_eaik, load_robot_model_pin, RobotModel

# --- Feasibility ---
from .feasibility_checks import (
    FeasibilityAnalyzer,
    FeasibilityResult,
    compute_manipulability,
    compute_singularity_proximity,
    compute_max_singular_value,
    compute_condition_number,
    check_reachability
)

# --- Singularity analysis ---
from .singularity_analysis import (
    SingularityAnalyzer,
    SingularityReport,
    SingularityType,
)
from .unified_singularity import (
    UnifiedSingularity,
    UnifiedSingularityReport,
)

# --- Self-collision ---
from .collision_checker import SelfCollisionChecker, CollisionResult


def create_solvers(urdf_path: str, solver: str = "eaik",
                   ik_config=None, ee_frame_name: str = "ee_link"):
    """
    Factory: create an (fk_solver, ik_solver) pair for the requested backend.

    Args:
        urdf_path: Path to URDF file
        solver: "eaik" or "pin"
        ik_config: Pre-built IKConfig object (EAIKConfig or PinocchioIKConfig).
                   If None, default config for the backend is used.
        ee_frame_name: End-effector frame name

    Returns:
        (fk_solver, ik_solver, robot_model_or_tuple)
        - fk_solver: BaseFKSolver subclass instance
        - ik_solver: BaseIKSolver subclass instance
        - robot_data: RobotModel (eaik) or (pin.Model, pin.Data) tuple (pin)
    """
    solver = solver.lower().strip()

    if solver == "eaik":
        robot_model = load_robot_model_eaik(urdf_path, ee_frame_name=ee_frame_name)
        fk = EAIKFKSolver(robot_model)
        if ik_config is None:
            ik_config = EAIKConfig(ee_frame_name=ee_frame_name)
        ik = EAIKIKSolver(robot_model, config=ik_config)
        return fk, ik, robot_model

    elif solver in ("pin", "pinocchio"):
        from .pin_fk_solver import PinocchioFKSolver
        from .pin_ik_solver import PinocchioIKSolver, PinocchioIKConfig
        model, data = load_robot_model_pin(urdf_path)
        fk = PinocchioFKSolver(model, data, ee_frame_name=ee_frame_name)
        if ik_config is None:
            ik_config = PinocchioIKConfig(ee_frame_name=ee_frame_name)
        ik = PinocchioIKSolver(model, data, config=ik_config)
        return fk, ik, (model, data)

    else:
        raise ValueError(f"Unknown solver backend: '{solver}'. Use 'eaik' or 'pin'.")


# --- Backward-compatibility aliases ---
# Existing scripts that do `from core import IKSolver, IKConfig, FKSolver`
# will continue to work, getting the Pinocchio variants.
from .pin_ik_solver import PinocchioIKSolver as IKSolver, PinocchioIKConfig as IKConfig
from .pin_fk_solver import PinocchioFKSolver as FKSolver


__all__ = [
    # Base
    'BaseFKSolver', 'BaseIKSolver', 'BaseIKConfig', 'FKResult',
    # EAIK
    'EAIKFKSolver', 'EAIKIKSolver', 'EAIKConfig',
    # Pinocchio (lazy -- not imported at module level to avoid hard dependency)
    'PinocchioFKSolver', 'PinocchioIKSolver', 'PinocchioIKConfig',
    # Backward-compatibility aliases
    'IKSolver', 'IKConfig', 'FKSolver',
    # Robot model
    'RobotModel', 'load_robot_model', 'load_robot_model_eaik', 'load_robot_model_pin',
    # Factory
    'create_solvers',
    # Feasibility
    'FeasibilityAnalyzer', 'FeasibilityResult',
    'compute_manipulability', 'compute_singularity_proximity',
    'compute_max_singular_value', 'compute_condition_number',
    'check_reachability',
    # Singularity analysis
    'SingularityAnalyzer', 'SingularityReport', 'SingularityType',
    'UnifiedSingularity', 'UnifiedSingularityReport',
    # Self-collision
    'SelfCollisionChecker', 'CollisionResult',
]
