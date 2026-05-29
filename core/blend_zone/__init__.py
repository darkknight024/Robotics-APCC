"""
Feature 3 Deliverable 1 — Blend Zone Analysis Package
======================================================

Predicts the actual TCP speed profile and SE(3) path an ABB IRB-1300 will
execute for a toolpath with per-waypoint ``zonedata``.  Validated against
RobotStudio Signal-Analyser recordings in ``tests/run_experiment_23_full.py``.

Pipeline (left→right; see :func:`pipeline.run_feature3_d1`)
-----------------------------------------------------------
M1 :mod:`zone_resolver`      Parse zone spec (``z0``..``z100``, ``fine``)
M2 :mod:`blend_geometry`     Cubic-Bézier blend arcs (``shape_k = 0.78``)
M3 :mod:`orientation_zone`   Effective orientation-zone onset (TRM-RAPID p.1796)
M4 :mod:`path_sampler`       Dense SE(3) path: straights + blend arcs
M5 :mod:`speed_profile`      TCP speed: centripetal ceiling + ramp limits
 · :mod:`calibration`        Identifies ``a_tcp``, ``T_settle`` etc. from RS data
 · :mod:`reporting`          Writes ``trajectory_N_result.csv`` + JSON report
 · :mod:`verification`       Solver-vs-RS speed / pose / joint comparison
 · :mod:`blend_comparison`   Per-blend-arc Fréchet / Hausdorff / deviation
"""

from .zone_resolver import (
    ZoneParams,
    PREDEFINED_ZONES,
    ZONE_NUMBER_MAP,
    resolve_zone_spec,
    resolve_zone_from_number,
    resolve_zone_list,
    apply_overlap_reduction,
)
from .blend_geometry import (
    BlendArcGeometry,
    compute_blend_geometry,
    compute_blend_geometries,
)
from .orientation_zone import (
    EffectiveOrientationZone,
    compute_effective_orientation_zone,
    populate_orientation_zones,
)
from .path_sampler import (
    DensePath,
    sample_blended_path,
)
from .speed_profile import (
    SpeedCalibration,
    SpeedProfileResult,
    predict_speed_profile,
)
from .pipeline import (
    Feature3D1Result,
    run_feature3,
    run_feature3_d1,
)
from .calibration import (
    CalibrationResult,
    CalibrationOffset,
    run_calibration,
    compute_calibration_offsets,
    save_calibration_report,
    generate_calibration_plots,
    load_rs_csv,
    load_trajectory_csv,
)
from .verification import (
    TrajectoryVerification,
    verify_trajectory,
    verify_batch,
    generate_verification_report,
    generate_verification_plots,
    generate_trajectory_comparison_plots,
    show_3d_blend_comparison,
)
from .blend_comparison import (
    BlendArcComparisonResult,
    WaypointBlendComparison,
    compare_blend_arcs,
    generate_blend_comparison_plots,
    show_3d_blend_arc_comparison,
)

__all__ = [
    # M1
    "ZoneParams", "PREDEFINED_ZONES", "ZONE_NUMBER_MAP",
    "resolve_zone_spec", "resolve_zone_from_number",
    "resolve_zone_list", "apply_overlap_reduction",
    # M2
    "BlendArcGeometry", "compute_blend_geometry", "compute_blend_geometries",
    # M3
    "EffectiveOrientationZone", "compute_effective_orientation_zone",
    "populate_orientation_zones",
    # M4
    "DensePath", "sample_blended_path",
    # M5
    "SpeedCalibration", "SpeedProfileResult", "predict_speed_profile",
    # Pipeline
    "Feature3D1Result", "run_feature3", "run_feature3_d1",
    # Calibration
    "CalibrationResult", "CalibrationOffset",
    "run_calibration", "compute_calibration_offsets",
    "save_calibration_report", "generate_calibration_plots",
    "load_rs_csv", "load_trajectory_csv",
    # Verification
    "TrajectoryVerification",
    "verify_trajectory", "verify_batch",
    "generate_verification_report", "generate_verification_plots",
    "generate_trajectory_comparison_plots",
    "show_3d_blend_comparison",
    # Blend arc comparison
    "BlendArcComparisonResult", "WaypointBlendComparison",
    "compare_blend_arcs", "generate_blend_comparison_plots",
    "show_3d_blend_arc_comparison",
]
