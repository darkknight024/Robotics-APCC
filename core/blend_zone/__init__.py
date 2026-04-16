"""
Feature 3 Deliverable 1 — Blend Zone Analysis Package
======================================================

Implements the zone-blending pipeline for predicting actual TCP speed profiles
on ABB IRB-1300 toolpaths with per-waypoint zonedata specifications.

Modules
-------
- :mod:`zone_resolver`      M1 — Zone lookup table and overlap reduction
- :mod:`blend_geometry`     M2 — Parabolic blend arc geometry (quadratic Bézier)
- :mod:`orientation_zone`   M3 — Effective orientation zone onset (ABB p.1796)
- :mod:`path_sampler`       M4 — Dense SE(3) path sampler with blend arcs
- :mod:`speed_profile`      M5 — TCP speed profile prediction
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
    "Feature3D1Result", "run_feature3_d1",
    # Calibration
    "CalibrationResult", "CalibrationOffset",
    "run_calibration", "compute_calibration_offsets",
    "save_calibration_report", "generate_calibration_plots",
    "load_rs_csv",
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
