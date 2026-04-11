"""
M1 — Zone Resolver
===================

Single source of truth for all zone-related values in the Feature 3 pipeline.

Responsibilities:
    1. Parse any zone specification (predefined name or custom triplet) into
       a structured :class:`ZoneParams` dataclass.
    2. Apply ABB's overlap-reduction rule across a waypoint list so that
       neighbouring zones never exceed half the inter-waypoint distance.

Nothing downstream should ever inspect a raw string zone name — it receives
:class:`ZoneParams`.

ABB Reference:
    *Technical reference manual — RAPID Instructions, Functions and Data types*,
    Section 3.95 ``zonedata``, pp. 1794–1799.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZoneParams:
    """Resolved zone parameters for a single waypoint.

    Attributes:
        finep:              True if this is a stop point (fine).
        pzone_tcp_mm:       TCP position zone radius in mm (programmed).
        pzone_ori_mm:       Orientation zone radius in mm of TCP movement (programmed).
        zone_ori_deg:       Orientation zone in degrees of tool reorientation (programmed).
        eff_pzone_tcp_mm:   Effective TCP zone after overlap reduction.
        eff_pzone_ori_mm:   Effective orientation zone after overlap reduction.
        source:             Origin label for diagnostics ('fine', 'z10', 'custom', …).
    """

    finep: bool
    pzone_tcp_mm: float
    pzone_ori_mm: float
    zone_ori_deg: float
    eff_pzone_tcp_mm: float = 0.0
    eff_pzone_ori_mm: float = 0.0
    source: str = ""

    def with_effective(
        self,
        eff_tcp: float,
        eff_ori: float,
    ) -> ZoneParams:
        """Return a copy with updated effective radii after overlap reduction."""
        return ZoneParams(
            finep=self.finep,
            pzone_tcp_mm=self.pzone_tcp_mm,
            pzone_ori_mm=self.pzone_ori_mm,
            zone_ori_deg=self.zone_ori_deg,
            eff_pzone_tcp_mm=eff_tcp,
            eff_pzone_ori_mm=eff_ori,
            source=self.source,
        )


# ── Predefined ABB zone table (RAPID manual p. 1797) ────────────────────────
# Keys: zone name -> (pzone_tcp mm, pzone_ori mm, zone_ori deg)
PREDEFINED_ZONES: Dict[str, Tuple[float, float, float]] = {
    "fine": (0.0, 0.0, 0.0),
    "z0":   (0.3, 0.3, 0.15),
    "z1":   (1.0, 1.0, 0.5),
    "z5":   (5.0, 8.0, 4.0),
    "z10":  (10.0, 15.0, 7.5),
    "z15":  (15.0, 23.0, 11.0),
    "z20":  (20.0, 30.0, 15.0),
    "z30":  (30.0, 45.0, 22.0),
    "z40":  (40.0, 60.0, 30.0),
    "z50":  (50.0, 75.0, 35.0),
    "z60":  (60.0, 90.0, 40.0),
    "z80":  (80.0, 120.0, 50.0),
    "z100": (100.0, 150.0, 60.0),
    "z150": (150.0, 225.0, 80.0),
    "z200": (200.0, 300.0, 90.0),
}


def resolve_zone_spec(spec: Union[str, Tuple[float, float, float]]) -> ZoneParams:
    """Resolve a single zone specification into :class:`ZoneParams`.

    Args:
        spec: Either a predefined zone name (``'fine'``, ``'z10'``, …) or a
              3-tuple ``(pzone_tcp_mm, pzone_ori_mm, zone_ori_deg)``.

    Returns:
        :class:`ZoneParams` with programmed values.  Effective values are set
        equal to programmed values (overlap reduction has not been applied yet).

    Raises:
        ValueError: If the zone name is unknown or the tuple has wrong length.
    """
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in PREDEFINED_ZONES:
            raise ValueError(
                f"Unknown predefined zone '{spec}'. "
                f"Valid: {sorted(PREDEFINED_ZONES.keys())}"
            )
        tcp, ori, ori_deg = PREDEFINED_ZONES[key]
        is_fine = key == "fine"
        return ZoneParams(
            finep=is_fine,
            pzone_tcp_mm=tcp,
            pzone_ori_mm=ori,
            zone_ori_deg=ori_deg,
            eff_pzone_tcp_mm=tcp,
            eff_pzone_ori_mm=ori,
            source=key,
        )

    if isinstance(spec, (tuple, list)):
        if len(spec) != 3:
            raise ValueError(
                f"Custom zone spec must have 3 values (pzone_tcp, pzone_ori, zone_ori), "
                f"got {len(spec)}"
            )
        tcp, ori, ori_deg = float(spec[0]), float(spec[1]), float(spec[2])
        is_fine = tcp <= 0.0
        # ABB: pzone_ori must be >= pzone_tcp; clamp up if needed.
        ori = max(ori, tcp)
        return ZoneParams(
            finep=is_fine,
            pzone_tcp_mm=tcp,
            pzone_ori_mm=ori,
            zone_ori_deg=ori_deg,
            eff_pzone_tcp_mm=tcp,
            eff_pzone_ori_mm=ori,
            source=f"custom({tcp},{ori},{ori_deg})",
        )

    raise TypeError(f"Zone spec must be str or 3-tuple, got {type(spec)}")


def resolve_zone_list(
    zone_specs: List[Union[str, Tuple[float, float, float]]],
) -> List[ZoneParams]:
    """Resolve a list of per-waypoint zone specifications.

    Args:
        zone_specs: One entry per waypoint — a preset name or custom triplet.

    Returns:
        List of :class:`ZoneParams` with programmed values (no overlap reduction yet).
    """
    return [resolve_zone_spec(s) for s in zone_specs]


def apply_overlap_reduction(
    zones: List[ZoneParams],
    waypoints_m: np.ndarray,
) -> List[ZoneParams]:
    """Apply ABB overlap-reduction rule to effective zone radii.

    When consecutive waypoints are closer than ``2 × pzone_tcp`` the controller
    automatically reduces the effective zone radius::

        r_eff[i] = min(r_programmed[i], 0.5 × min(d[i-1], d[i]))

    This applies independently to ``pzone_tcp`` and ``pzone_ori``.  ``zone_ori``
    is angular and not subject to overlap reduction.

    The first and last waypoints are treated as fine points regardless of their
    zone specification (the path must start and end with a stop).

    Args:
        zones:        Per-waypoint :class:`ZoneParams` (from :func:`resolve_zone_list`).
        waypoints_m:  ``(N, 7)`` waypoint array ``[x_m, y_m, z_m, qw, qx, qy, qz]``.

    Returns:
        New list of :class:`ZoneParams` with effective radii populated.
    """
    n = len(zones)
    if n != len(waypoints_m):
        raise ValueError(
            f"zones length ({n}) does not match waypoints length ({len(waypoints_m)})"
        )
    if n < 2:
        return [z.with_effective(0.0, 0.0) for z in zones]

    positions_mm = waypoints_m[:, :3] * 1000.0
    seg_lengths = np.linalg.norm(np.diff(positions_mm, axis=0), axis=1)

    result: List[ZoneParams] = []
    for i, zp in enumerate(zones):
        if zp.finep or i == 0 or i == n - 1:
            result.append(zp.with_effective(0.0, 0.0))
            continue

        d_prev = seg_lengths[i - 1] if i > 0 else np.inf
        d_next = seg_lengths[i] if i < n - 1 else np.inf
        half_min_d = 0.5 * min(d_prev, d_next)

        eff_tcp = min(zp.pzone_tcp_mm, half_min_d)
        eff_ori = min(zp.pzone_ori_mm, half_min_d)
        eff_ori = max(eff_ori, eff_tcp)

        if eff_tcp < zp.pzone_tcp_mm:
            logger.debug(
                "Waypoint %d: pzone_tcp reduced %.1f → %.1f mm (overlap, d_min=%.1f)",
                i, zp.pzone_tcp_mm, eff_tcp, 2.0 * half_min_d,
            )

        result.append(zp.with_effective(eff_tcp, eff_ori))

    return result
