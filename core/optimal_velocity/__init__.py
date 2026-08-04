"""Optimal TCP velocity profile (TOPP) — core mathematical pipeline.

Public entry points for Steps 0–3 and the diagnostic orchestrator
:func:`run_diagnostics`.
"""

from .differentiation import eval_splines, fit_joint_splines, step1_differentiate
from .heun_topp import step3_time_optimal
from .mvc_ceilings import secant_accel_ceiling, step2_velocity_limit
from .pipeline import run_diagnostics
from .regions import compute_regions
from .types import JointLimits, ProfileResult
from .validate import step0_validate

__all__ = [
    "JointLimits",
    "ProfileResult",
    "compute_regions",
    "eval_splines",
    "fit_joint_splines",
    "run_diagnostics",
    "secant_accel_ceiling",
    "step0_validate",
    "step1_differentiate",
    "step2_velocity_limit",
    "step3_time_optimal",
]
