"""Thin re-export shim — SE(3) arc-length lives in ``core.path_parameterization``.

Legacy callers under ``core.blend_zone`` keep working via this module.
"""

from core.path_parameterization.se3_arc_length import (  # noqa: F401
    DEFAULT_LAMBDA_MM_PER_RAD,
    LEGACY_TOPP_LAMBDA_MM_PER_RAD,
    compute_se3_arc_length,
    estimate_lambda,
    pose_arc_length_mm,
    resolve_lambda,
    run_lambda_sensitivity,
    se3_parameterisation_summary,
)

__all__ = [
    "DEFAULT_LAMBDA_MM_PER_RAD",
    "LEGACY_TOPP_LAMBDA_MM_PER_RAD",
    "compute_se3_arc_length",
    "estimate_lambda",
    "pose_arc_length_mm",
    "resolve_lambda",
    "run_lambda_sensitivity",
    "se3_parameterisation_summary",
]
