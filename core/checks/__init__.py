"""
Modular feasibility check modules.

Each check lives in its own script for clarity and testability:
- singularity: SingularityAnalyzer (unified + classified modes),
  low-level helpers (sigma_min, condition number, spectrum, j5 wrist band)
- manipulability: Yoshikawa, translational, rotational, normalised, directional
- c0_continuity: joint delta spikes / config flip detection
- c1_continuity: joint velocity smoothness from TOPP-RA output
- task_space_velocity: forward differential kinematics & CSV limit verification
"""

from .singularity import (
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
from .manipulability import (
    compute_manipulability,
    compute_translational_manipulability,
    compute_rotational_manipulability,
    compute_normalized_manipulability,
    compute_directional_manipulability,
)
from .c0_continuity import (
    check_c0_continuity,
    detect_config_flips,
    compute_per_joint_deltas,
)
from .c1_continuity import (
    check_c1_continuity,
    C1Result,
)
from .task_space_velocity import (
    compute_task_space_velocity,
    check_speed_limits,
    TaskSpaceVelocityResult,
)

__all__ = [
    # singularity
    "compute_singularity_proximity",
    "compute_condition_number",
    "compute_max_singular_value",
    "analyze_singularity_spectrum",
    "j5_wrist_singularity_band_active",
    "SingularityAnalyzer",
    "SingularityReport",
    "SingularityType",
    "SingularityMode",
    # manipulability
    "compute_manipulability",
    "compute_translational_manipulability",
    "compute_rotational_manipulability",
    "compute_normalized_manipulability",
    "compute_directional_manipulability",
    # c0
    "check_c0_continuity",
    "detect_config_flips",
    "compute_per_joint_deltas",
    # c1
    "check_c1_continuity",
    "C1Result",
    # task-space velocity
    "compute_task_space_velocity",
    "check_speed_limits",
    "TaskSpaceVelocityResult",
]
