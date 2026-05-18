#!/usr/bin/env python3
"""Robot vs static environment collision (subset of :class:`SceneCollisionChecker`)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .scene_checker import SceneCollisionChecker
from .types import CollisionResult


class ObjectCollisionChecker:
    """Flags collisions between the moving robot assembly and static scene meshes."""

    def __init__(self, scene: SceneCollisionChecker):
        self._scene = scene

    def check(self, q: np.ndarray) -> CollisionResult:
        full = self._scene.check(q)
        env_pairs = []
        for a, b in full.colliding_pairs:
            try:
                ia = self._scene._geom_index(a)
                ib = self._scene._geom_index(b)
            except KeyError:
                continue
            rs = self._scene._robot_set
            es = self._scene._env_set
            if (ia in rs and ib in es) or (ib in rs and ia in es):
                env_pairs.append((a, b))
        all_d = [
            t
            for t in full.all_distances
            if self._pair_is_robot_env(t[0], t[1])
        ]
        min_d = min((t[2] for t in all_d), default=-1.0)
        closest: Tuple[str, str] = ("", "")
        if all_d:
            closest = min(all_d, key=lambda t: t[2])[:2]
        return CollisionResult(
            has_collision=len(env_pairs) > 0,
            colliding_pairs=env_pairs,
            min_distance_m=float(min_d),
            closest_pair=closest,
            all_distances=sorted(all_d, key=lambda t: t[2]),
            meta={"subset": "robot_environment"},
        )

    def _pair_is_robot_env(self, n1: str, n2: str) -> bool:
        ia = self._scene._geom_index(n1)
        ib = self._scene._geom_index(n2)
        rs = self._scene._robot_set
        es = self._scene._env_set
        return (ia in rs and ib in es) or (ib in rs and ia in es)

    def has_collision(self, q: np.ndarray) -> bool:
        return self._scene.has_environment_collision(q)
