#!/usr/bin/env python3
"""Joint-space trajectory collision sweep."""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .types import TrajectoryCollisionReport, WaypointCollisionResult


class TrajectoryCollisionChecker:
    """Evaluate ``has_collision`` / :meth:`check` on a dense joint trajectory."""

    def __init__(self, collision_fn: Callable[[np.ndarray], bool]):
        self._collision_fn = collision_fn

    def check_trajectory(
        self,
        q_path: Sequence[np.ndarray],
        *,
        stop_at_first: bool = True,
        detailed_check: Optional[Callable[[np.ndarray], object]] = None,
        time_it: bool = False,
    ) -> TrajectoryCollisionReport:
        per_wp: List[WaypointCollisionResult] = []
        first_hit: Optional[int] = None
        first_pairs: List[tuple] = []
        min_clear = float("inf")
        timings: Optional[List[float]] = None
        if time_it:
            timings = []

        for i, q in enumerate(q_path):
            t0 = time.perf_counter() if time_it else 0.0
            bad = self._collision_fn(np.asarray(q, dtype=float))
            if time_it and timings is not None:
                timings.append(time.perf_counter() - t0)

            dmin = -1.0
            pairs: List[tuple] = []
            if detailed_check is not None:
                res = detailed_check(np.asarray(q, dtype=float))
                if hasattr(res, "min_distance_m"):
                    dmin = float(res.min_distance_m)
                    min_clear = min(min_clear, dmin) if dmin >= 0 else min_clear
                if hasattr(res, "colliding_pairs"):
                    pairs = list(res.colliding_pairs)

            per_wp.append(
                WaypointCollisionResult(
                    waypoint_index=i,
                    has_collision=bad,
                    colliding_pairs=pairs,
                    min_distance_m=dmin,
                )
            )
            if bad and first_hit is None:
                first_hit = i
                first_pairs = pairs
                if stop_at_first:
                    break

        any_bad = first_hit is not None or any(p.has_collision for p in per_wp)
        return TrajectoryCollisionReport(
            has_any_collision=any_bad,
            first_collision_waypoint=first_hit,
            first_colliding_pairs=first_pairs,
            min_clearance_m=min_clear if min_clear != float("inf") else -1.0,
            per_waypoint=per_wp,
            timings_s=timings,
        )
