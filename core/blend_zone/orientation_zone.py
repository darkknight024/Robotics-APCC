"""
M3 — Orientation Zone
======================

Implements the exact ABB formula (RAPID manual p. 1796) that determines where
along each segment the orientation SLERP begins.

The effective orientation onset is *not* always equal to ``pzone_ori``.  The
``zone_ori`` field (degrees) can constrain it more tightly on short segments
with large orientation change.  The formula is::

    r_ori_eff = min(pzone_ori, (zone_ori_rad / Δθ_segment) × L)

    Floor: r_ori_eff ≥ pzone_tcp   (orientation zone ≥ position zone, always)

The outputs are written back into the :class:`BlendArcGeometry` dataclasses
from M2 — this is the documented exception to the "immutable outputs" rule.

ABB Reference:
    *RAPID Instructions, Functions and Data types*, Section 3.95,
    "Calculation of reorientation and additional axis zone", pp. 1794–1796.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .zone_resolver import ZoneParams
from .blend_geometry import BlendArcGeometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectiveOrientationZone:
    """Result of the orientation zone formula for one waypoint.

    Attributes:
        waypoint_idx:        Which waypoint this applies to.
        r_ori_eff_mm:        Effective orientation zone onset (mm from waypoint).
        governed_by:         ``'pzone_ori'`` or ``'zone_ori'``.
        delta_theta_in_rad:  Orientation change on the incoming segment.
        delta_theta_out_rad: Orientation change on the outgoing segment.
        segment_len_in_mm:   Length of the incoming segment.
        segment_len_out_mm:  Length of the outgoing segment.
    """

    waypoint_idx: int
    r_ori_eff_mm: float
    governed_by: str
    delta_theta_in_rad: float
    delta_theta_out_rad: float
    segment_len_in_mm: float
    segment_len_out_mm: float


def _quaternion_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Compute the angular distance (radians) between two unit quaternions [w,x,y,z]."""
    dot = np.clip(np.abs(np.dot(q0, q1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def compute_effective_orientation_zone(
    waypoints_m: np.ndarray,
    idx: int,
    zone: ZoneParams,
) -> EffectiveOrientationZone:
    """Compute the effective orientation zone onset for one waypoint.

    Uses the ABB formula: smallest of the pzone_ori-based and zone_ori-based
    zones, floored at pzone_tcp.

    Args:
        waypoints_m:  (N, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        idx:          Waypoint index.
        zone:         :class:`ZoneParams` for this waypoint.

    Returns:
        :class:`EffectiveOrientationZone` with the computed onset.
    """
    n = len(waypoints_m)
    positions_mm = waypoints_m[:, :3] * 1000.0
    quats = waypoints_m[:, 3:7]

    seg_len_in = 0.0
    delta_theta_in = 0.0
    if idx > 0:
        seg_len_in = float(np.linalg.norm(positions_mm[idx] - positions_mm[idx - 1]))
        delta_theta_in = _quaternion_angle(quats[idx - 1], quats[idx])

    seg_len_out = 0.0
    delta_theta_out = 0.0
    if idx < n - 1:
        seg_len_out = float(np.linalg.norm(positions_mm[idx + 1] - positions_mm[idx]))
        delta_theta_out = _quaternion_angle(quats[idx], quats[idx + 1])

    if zone.finep or idx == 0 or idx == n - 1:
        return EffectiveOrientationZone(
            waypoint_idx=idx,
            r_ori_eff_mm=0.0,
            governed_by="fine_or_endpoint",
            delta_theta_in_rad=delta_theta_in,
            delta_theta_out_rad=delta_theta_out,
            segment_len_in_mm=seg_len_in,
            segment_len_out_mm=seg_len_out,
        )

    pzone_ori = zone.eff_pzone_ori_mm
    pzone_tcp = zone.eff_pzone_tcp_mm
    zone_ori_rad = np.radians(zone.zone_ori_deg)

    # Orientation zone from pzone_ori: fraction of segment length
    r_from_pzone_ori = pzone_ori

    # Orientation zone from zone_ori: (zone_ori_rad / Δθ) × L
    # Use the max of incoming/outgoing delta_theta for the binding constraint
    delta_theta_max = max(delta_theta_in, delta_theta_out)
    seg_len_max = max(seg_len_in, seg_len_out)

    if delta_theta_max > 1e-9 and seg_len_max > 1e-9:
        r_from_zone_ori = (zone_ori_rad / delta_theta_max) * seg_len_max
    else:
        r_from_zone_ori = np.inf

    if r_from_pzone_ori <= r_from_zone_ori:
        r_ori_eff = r_from_pzone_ori
        governed_by = "pzone_ori"
    else:
        r_ori_eff = r_from_zone_ori
        governed_by = "zone_ori"

    # Floor: orientation zone must be >= position zone
    r_ori_eff = max(r_ori_eff, pzone_tcp)

    return EffectiveOrientationZone(
        waypoint_idx=idx,
        r_ori_eff_mm=r_ori_eff,
        governed_by=governed_by,
        delta_theta_in_rad=delta_theta_in,
        delta_theta_out_rad=delta_theta_out,
        segment_len_in_mm=seg_len_in,
        segment_len_out_mm=seg_len_out,
    )


def populate_orientation_zones(
    blend_geoms: List[Optional[BlendArcGeometry]],
    zones: List[ZoneParams],
    waypoints_m: np.ndarray,
) -> List[EffectiveOrientationZone]:
    """Compute effective orientation zones and write them into blend geometries.

    This mutates the :class:`BlendArcGeometry` objects in *blend_geoms* by
    setting ``r_ori_eff_mm``, ``ori_onset_in_mm``, and ``ori_onset_out_mm``.

    Args:
        blend_geoms:  Per-waypoint blend geometry (from M2).  ``None`` entries
                      are skipped.
        zones:        Per-waypoint :class:`ZoneParams` (overlap-reduced).
        waypoints_m:  (N, 7) waypoint array.

    Returns:
        List of :class:`EffectiveOrientationZone` for every waypoint.
    """
    n = len(zones)
    ori_zones: List[EffectiveOrientationZone] = []

    for i in range(n):
        eff = compute_effective_orientation_zone(waypoints_m, i, zones[i])
        ori_zones.append(eff)

        geom = blend_geoms[i] if i < len(blend_geoms) else None
        if geom is not None:
            geom.r_ori_eff_mm = eff.r_ori_eff_mm
            geom.ori_onset_in_mm = eff.r_ori_eff_mm
            geom.ori_onset_out_mm = eff.r_ori_eff_mm

    n_governed_zone_ori = sum(1 for e in ori_zones if e.governed_by == "zone_ori")
    if n_governed_zone_ori > 0:
        logger.info(
            "Orientation zone: %d/%d waypoints governed by zone_ori (angular constraint)",
            n_governed_zone_ori, n,
        )

    return ori_zones
