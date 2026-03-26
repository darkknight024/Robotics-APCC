"""
Core module for kinematic solvers and feasibility analysis.

Provides:
- Base abstract classes: BaseFKSolver, BaseIKSolver, BaseIKConfig, FKResult
- EAIK backend:  EAIKFKSolver, EAIKIKSolver, EAIKConfig
- Pinocchio backend: PinocchioFKSolver, PinocchioIKSolver, PinocchioIKConfig
- create_solvers() factory to instantiate the right backend from config
- FeasibilityAnalyzer and helpers
- RobotModel dataclass
- Modular checks (core.checks sub-package)
- Consolidated singularity analysis (core.checks.singularity)
"""

# --- Base classes ---
from .base_solvers import BaseFKSolver, BaseIKSolver, BaseIKConfig, FKResult

# --- EAIK backend ---
from .eaik_fk_solver import EAIKFKSolver
from .eaik_ik_solver import EAIKIKSolver, EAIKConfig, ECFXLabel, compute_ecfx

# --- Robot model ---
from utils.urdf_loader import load_robot_model, load_robot_model_eaik, load_robot_model_pin, RobotModel

# --- Feasibility (orchestrator) ---
from .feasibility_checks import (
    FeasibilityAnalyzer,
    FeasibilityResult,
    IkSolutionScoreBreakdown,
    MixedBranchResult,
    check_reachability,
    score_ik_solution_breakdown,
    select_best_cfx_branch,
    select_mixed_cfx_branches,
)

# --- Modular checks (core.checks sub-package) ---
from .checks.singularity import (
    compute_singularity_proximity,
    compute_condition_number,
    compute_max_singular_value,
    analyze_singularity_spectrum,
    j5_wrist_singularity_band_active,
    SingularityAnalyzer,
    SingularityReport,
    SingularityType,
    SingularityMode,
)
from .checks.manipulability import (
    compute_manipulability,
    compute_translational_manipulability,
    compute_rotational_manipulability,
    compute_normalized_manipulability,
    compute_directional_manipulability,
)
from .checks.c0_continuity import (
    check_c0_continuity,
    detect_config_flips,
    compute_per_joint_deltas,
)
from .checks.c1_continuity import check_c1_continuity, C1Result
from .checks.task_space_velocity import (
    compute_task_space_velocity,
    check_speed_limits,
    TaskSpaceVelocityResult,
)

# --- Time parameterization (waypoint density) ---
from utils.time_parameterization import (
    compute_arc_lengths,
    check_waypoint_density,
    interpolate_sparse_segments,
)

# --- TOPP-RA (mandatory at runtime; import guarded so core package loads) ---
try:
    from .topp_check import parameterize_trajectory, ToppraResult
except ImportError:
    parameterize_trajectory = None  # type: ignore[assignment,misc]
    ToppraResult = None  # type: ignore[assignment,misc]

# --- Self-collision (optional: requires pinocchio) ---
try:
    from .collision_checker import SelfCollisionChecker, CollisionResult
except ImportError:
    SelfCollisionChecker = None  # type: ignore[assignment,misc]
    CollisionResult = None  # type: ignore[assignment,misc]


def create_solvers(urdf_path: str, solver: str = "eaik",
                   ik_config=None, ee_frame_name: str = "ee_link"):
    """Factory: create an (fk_solver, ik_solver) pair for the requested backend."""
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


# --- Backward-compatibility aliases (optional: requires pinocchio) ---
try:
    from .pin_ik_solver import PinocchioIKSolver as IKSolver, PinocchioIKConfig as IKConfig
    from .pin_fk_solver import PinocchioFKSolver as FKSolver
except ImportError:
    IKSolver = None  # type: ignore[assignment,misc]
    IKConfig = None  # type: ignore[assignment,misc]
    FKSolver = None  # type: ignore[assignment,misc]


__all__ = [
    # Base
    'BaseFKSolver', 'BaseIKSolver', 'BaseIKConfig', 'FKResult',
    # EAIK
    'EAIKFKSolver', 'EAIKIKSolver', 'EAIKConfig', 'ECFXLabel', 'compute_ecfx',
    # Pinocchio (lazy)
    'PinocchioFKSolver', 'PinocchioIKSolver', 'PinocchioIKConfig',
    # Backward-compatibility
    'IKSolver', 'IKConfig', 'FKSolver',
    # Robot model
    'RobotModel', 'load_robot_model', 'load_robot_model_eaik', 'load_robot_model_pin',
    # Factory
    'create_solvers',
    # Feasibility orchestrator
    'FeasibilityAnalyzer', 'FeasibilityResult',
    'IkSolutionScoreBreakdown', 'MixedBranchResult',
    'check_reachability', 'score_ik_solution_breakdown',
    'select_best_cfx_branch', 'select_mixed_cfx_branches',
    # Consolidated singularity
    'SingularityAnalyzer', 'SingularityReport', 'SingularityType', 'SingularityMode',
    'j5_wrist_singularity_band_active',
    # Low-level checks
    'compute_singularity_proximity', 'compute_condition_number',
    'compute_max_singular_value', 'analyze_singularity_spectrum',
    'compute_manipulability',
    'compute_translational_manipulability', 'compute_rotational_manipulability',
    'compute_normalized_manipulability', 'compute_directional_manipulability',
    'check_c0_continuity', 'detect_config_flips', 'compute_per_joint_deltas',
    'check_c1_continuity', 'C1Result',
    'compute_task_space_velocity', 'check_speed_limits', 'TaskSpaceVelocityResult',
    # TOPP-RA
    'parameterize_trajectory', 'ToppraResult',
    # Self-collision
    'SelfCollisionChecker', 'CollisionResult',
]
