#!/usr/bin/env python3
"""Shared datatypes for collision checking (Feature 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CollisionResult:
    """Result of a collision query (self, object, or combined scene)."""

    has_collision: bool
    colliding_pairs: List[Tuple[str, str]]
    min_distance_m: float
    closest_pair: Tuple[str, str]
    all_distances: List[Tuple[str, str, float]]
    # Optional diagnostics (e.g. which subsystem flagged)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaypointCollisionResult:
    """Per-waypoint summary for trajectory checks."""

    waypoint_index: int
    has_collision: bool
    colliding_pairs: List[Tuple[str, str]]
    min_distance_m: float


@dataclass
class TrajectoryCollisionReport:
    """Aggregated trajectory collision report."""

    has_any_collision: bool
    first_collision_waypoint: Optional[int]
    first_colliding_pairs: List[Tuple[str, str]]
    min_clearance_m: float
    per_waypoint: List[WaypointCollisionResult]
    timings_s: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
