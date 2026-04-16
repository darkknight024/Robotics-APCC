"""
M5 — Speed Profile
====================

Predicts the actual TCP speed at every arc-length sample.  This module's
outputs are the primary Deliverable 1 answer.

Physics:
    (a) On straight segments the planner executes a trapezoidal or triangular
        velocity profile.
    (b) Through blend arcs the centripetal constraint
        ``v_blend = sqrt(a_tcp * rho_min)`` limits the speed.
    (c) Fine points: TCP decelerates to zero and settles for T_settle seconds.

Both models require the calibration constant ``a_tcp``, which must be measured
from Experiment V1.  Until calibrated, a placeholder value is used and a
warning is emitted.

Speed Profile Equation:
    ``v_actual(s) = min(v_cmd(s), v_blend_ceiling(s), v_topp_ceiling(s))``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .blend_geometry import BlendArcGeometry
from .path_sampler import DensePath

logger = logging.getLogger(__name__)

_PLACEHOLDER_A_TCP = 2500.0   # mm/s^2 — placeholder until calibrated from V1
_PLACEHOLDER_T_SETTLE = 0.2   # seconds — placeholder settling time at fine points


@dataclass(frozen=True)
class SpeedCalibration:
    """Calibration constants for the speed profile model.

    Attributes:
        a_tcp_mm_s2:     Effective TCP acceleration capability (mm/s^2).
        T_settle_s:      Fine-point settling time (seconds).
        is_calibrated:   True only after constants have been measured from site data.
    """

    a_tcp_mm_s2: float = _PLACEHOLDER_A_TCP
    T_settle_s: float = _PLACEHOLDER_T_SETTLE
    is_calibrated: bool = False


@dataclass(frozen=True)
class SpeedProfileResult:
    """Complete speed profile prediction over the dense path.

    Attributes:
        arc_lengths_mm:     (M,) cumulative arc-length from path start.
        v_actual:           (M,) predicted actual TCP speed (mm/s).
        v_cmd:              (M,) commanded speed at each sample (mm/s).
        v_blend_ceiling:    (M,) centripetal speed ceiling (mm/s; inf on straights).
        is_blend_arc:       (M,) True on blend arc samples.
        total_duration_s:   Estimated total path duration (seconds).
        fine_point_indices: Indices of fine-point stops in the dense path.
        calibration:        The :class:`SpeedCalibration` used.
    """

    arc_lengths_mm: np.ndarray
    v_actual: np.ndarray
    v_cmd: np.ndarray
    v_blend_ceiling: np.ndarray
    is_blend_arc: np.ndarray
    total_duration_s: float
    fine_point_indices: List[int] = field(default_factory=list)
    calibration: SpeedCalibration = field(default_factory=SpeedCalibration)


def _blend_speed_ceiling(
    rho_min_mm: float,
    a_tcp: float,
) -> float:
    """Centripetal speed constraint: v_blend_max = sqrt(a_tcp × ρ_min).

    Returns inf when rho_min is infinite (straight path, no curvature limit).
    """
    if not np.isfinite(rho_min_mm) or rho_min_mm <= 0:
        return np.inf
    return np.sqrt(a_tcp * rho_min_mm)


def _compute_trapezoidal_speed(
    s_local: float,
    L_eff: float,
    v_start: float,
    v_end: float,
    v_cmd: float,
    a_tcp: float,
) -> float:
    """Speed at arc-length s_local along a straight segment with trapezoidal profile.

    The segment spans [0, L_eff] mm.  The robot accelerates from v_start toward
    v_cmd, cruises if there's room, then decelerates to v_end.

    For short segments where v_cmd cannot be reached, a triangular profile peaks
    at v_peak = sqrt(a_tcp * L_eff_available).
    """
    if L_eff < 1e-9 or a_tcp < 1e-9:
        return min(v_start, v_end)

    s = np.clip(s_local, 0.0, L_eff)

    # Acceleration distance from v_start to v_cmd
    d_accel = max(0.0, (v_cmd ** 2 - v_start ** 2) / (2.0 * a_tcp)) if v_cmd > v_start else 0.0
    # Deceleration distance from v_cmd to v_end
    d_decel = max(0.0, (v_cmd ** 2 - v_end ** 2) / (2.0 * a_tcp)) if v_cmd > v_end else 0.0

    if d_accel + d_decel <= L_eff:
        # Trapezoidal: accel → cruise → decel
        L_cruise = L_eff - d_accel - d_decel
        if s < d_accel:
            return np.sqrt(max(v_start ** 2 + 2.0 * a_tcp * s, 0.0))
        elif s < d_accel + L_cruise:
            return v_cmd
        else:
            s_from_end = L_eff - s
            return np.sqrt(max(v_end ** 2 + 2.0 * a_tcp * s_from_end, 0.0))
    else:
        # Triangular: cannot reach v_cmd
        v_peak_sq = (2.0 * a_tcp * L_eff + v_start ** 2 + v_end ** 2) / 2.0
        v_peak = np.sqrt(max(v_peak_sq, 0.0))
        d_to_peak = max(0.0, (v_peak ** 2 - v_start ** 2) / (2.0 * a_tcp))
        if s < d_to_peak:
            return np.sqrt(max(v_start ** 2 + 2.0 * a_tcp * s, 0.0))
        else:
            s_from_end = L_eff - s
            return np.sqrt(max(v_end ** 2 + 2.0 * a_tcp * s_from_end, 0.0))


def predict_speed_profile(
    dense_path: DensePath,
    blend_geoms: List[Optional[BlendArcGeometry]],
    calibration: Optional[SpeedCalibration] = None,
    v_topp_ceiling: Optional[np.ndarray] = None,
) -> SpeedProfileResult:
    """Predict the actual TCP speed profile over the full dense path.

    The algorithm:
        1. For each blend arc sample, compute the centripetal speed ceiling from
           ρ_min of the arc's geometry.
        2. For each straight-segment sample, compute the trapezoidal/triangular
           speed between the adjacent blend arc speeds (or fine-point stops).
        3. Apply the TOPP-RA kinematic ceiling if available.
        4. Take the element-wise minimum.

    Args:
        dense_path:       :class:`DensePath` from M4.
        blend_geoms:      Per-waypoint blend geometry (from M2+M3).
        calibration:      :class:`SpeedCalibration` constants.
        v_topp_ceiling:   (M,) optional TOPP-RA speed ceiling in mm/s.

    Returns:
        :class:`SpeedProfileResult` with the full speed prediction.
    """
    if calibration is None:
        calibration = SpeedCalibration()

    if not calibration.is_calibrated:
        logger.warning(
            "Running with placeholder a_tcp=%.0f mm/s², T_settle=%.2f s. "
            "Outputs are structurally correct but quantitatively unvalidated. "
            "Run site experiments V1 and V2 first.",
            calibration.a_tcp_mm_s2, calibration.T_settle_s,
        )

    M = dense_path.n_samples
    a_tcp = calibration.a_tcp_mm_s2
    arc_s = dense_path.arc_lengths
    v_cmd = dense_path.v_cmd_at_s.copy()
    is_blend = dense_path.is_blend_arc
    seg_ids = dense_path.segment_ids

    # Build blend arc speed ceiling
    v_blend_ceil = np.full(M, np.inf)

    # Map each blend arc to its rho_min
    geom_by_idx = {}
    for g in blend_geoms:
        if g is not None:
            geom_by_idx[g.waypoint_idx] = g

    # For blend arc samples: find which waypoint's arc they belong to and use its rho_min
    # We identify blend arcs by the is_blend_arc flag and match to closest waypoint geom
    for g in blend_geoms:
        if g is None:
            continue
        v_ceiling = _blend_speed_ceiling(g.rho_min_mm, a_tcp)
        # Find samples that belong to this arc by checking proximity to the arc
        entry_s = None
        exit_s = None
        for k in range(M):
            if is_blend[k]:
                pos_mm = dense_path.poses[k, :3] * 1000.0
                d_to_control = np.linalg.norm(pos_mm - g.control_point_mm)
                if d_to_control < g.r_tcp_eff_mm * 2.5:
                    v_blend_ceil[k] = min(v_blend_ceil[k], v_ceiling)

    # For non-blend (straight) samples: compute trapezoidal profile
    v_profile = np.copy(v_cmd)

    # Identify contiguous straight segments between blend arcs / fine points
    # Simple approach: scan through all samples
    for k in range(M):
        if is_blend[k]:
            v_profile[k] = min(v_cmd[k], v_blend_ceil[k])

    # Apply trapezoidal acceleration/deceleration to ALL samples (including
    # blend arcs).  The blend ceiling is already baked into v_profile for
    # blend samples; this pass additionally enforces the kinematic ramp-rate
    # constraint v² ≤ v_prev² + 2·a·Δs, preventing instantaneous speed
    # jumps at straight→blend and start/end transitions.
    v_forward = np.copy(v_profile)
    for k in range(1, M):
        ds = arc_s[k] - arc_s[k - 1]
        if ds < 1e-9:
            v_forward[k] = v_forward[k - 1]
            continue
        v_max_accel = np.sqrt(max(v_forward[k - 1] ** 2 + 2.0 * a_tcp * ds, 0.0))
        v_forward[k] = min(v_forward[k], v_max_accel)

    # Backward pass: decelerate toward each blend entry / path end
    v_backward = np.copy(v_profile)
    v_backward[-1] = 0.0  # path end: fine point
    for k in range(M - 2, -1, -1):
        ds = arc_s[k + 1] - arc_s[k]
        if ds < 1e-9:
            v_backward[k] = v_backward[k + 1]
            continue
        v_max_decel = np.sqrt(max(v_backward[k + 1] ** 2 + 2.0 * a_tcp * ds, 0.0))
        v_backward[k] = min(v_backward[k], v_max_decel)

    # Combine: v_actual = min(v_cmd, v_blend_ceiling, v_forward, v_backward, v_topp)
    v_actual = np.minimum(v_forward, v_backward)
    v_actual = np.minimum(v_actual, v_blend_ceil)
    v_actual = np.minimum(v_actual, v_cmd)

    if v_topp_ceiling is not None and len(v_topp_ceiling) == M:
        v_actual = np.minimum(v_actual, v_topp_ceiling)

    # First and last samples are fine points (v=0)
    v_actual[0] = 0.0
    v_actual[-1] = 0.0

    # Identify fine-point samples (v_actual == 0 interior)
    fine_indices = [0, M - 1]

    # Estimate total duration by integrating ds / v
    total_time = 0.0
    for k in range(1, M):
        ds = arc_s[k] - arc_s[k - 1]
        v_avg = 0.5 * (v_actual[k - 1] + v_actual[k])
        if v_avg > 1e-6:
            total_time += ds / v_avg
        elif ds > 1e-6:
            total_time += ds / 1.0  # near-zero speed: use 1 mm/s as floor

    # Add settling time at fine points
    total_time += len(fine_indices) * calibration.T_settle_s

    logger.info(
        "Speed profile: v_actual range [%.1f, %.1f] mm/s, "
        "total duration %.2f s, %d fine-point stops",
        float(np.min(v_actual)),
        float(np.max(v_actual)),
        total_time,
        len(fine_indices),
    )

    return SpeedProfileResult(
        arc_lengths_mm=arc_s,
        v_actual=v_actual,
        v_cmd=v_cmd,
        v_blend_ceiling=v_blend_ceil,
        is_blend_arc=is_blend,
        total_duration_s=total_time,
        fine_point_indices=fine_indices,
        calibration=calibration,
    )
