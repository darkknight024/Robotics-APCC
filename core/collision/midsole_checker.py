#!/usr/bin/env python3
"""Midsole-specific collision rules (optional; driven by scene YAML)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .scene_checker import SceneCollisionChecker
from .types import CollisionResult


class MidsoleCollisionChecker:
    """When a midsole URDF geometry is configured, flag **non-whitelisted** midsole contacts.

    Allowed contacts (not reported as violations):
      - Pairs removed at the scene level (midsole ↔ knife blade whitelist)
      - Any geometry whose name contains ``knife`` and group ``tool`` (heuristic fallback)

    This checker is meant for diagnostics: the scene checker already removes the
    whitelisted collision **pair**. Use :meth:`violating_pairs` to detect midsole
    touching unexpected bodies if those pairs were not pre-removed.
    """

    def __init__(
        self,
        scene: SceneCollisionChecker,
        midsole_geom_name: Optional[str],
        allowed_env_name_substrings: Optional[Tuple[str, ...]] = None,
    ):
        self._scene = scene
        self._midsole = midsole_geom_name
        self._allowed_sub = allowed_env_name_substrings or ("knife", "blade", "tool")

    @property
    def enabled(self) -> bool:
        return self._midsole is not None and len(self._midsole) > 0

    def _allowed_env(self, env_name: str) -> bool:
        low = env_name.lower()
        return any(s in low for s in self._allowed_sub)

    def violating_pairs(self, q: np.ndarray) -> List[Tuple[str, str]]:
        if not self.enabled:
            return []
        res = self._scene.check(q)
        mid = self._midsole
        out: List[Tuple[str, str]] = []
        for a, b in res.colliding_pairs:
            if a == mid or b == mid:
                other = b if a == mid else a
                try:
                    io = self._scene._geom_index(other)
                except KeyError:
                    continue
                if io in self._scene._env_set and self._allowed_env(other):
                    continue
                out.append((a, b))
        return out

    def has_violation(self, q: np.ndarray) -> bool:
        return len(self.violating_pairs(q)) > 0

    def check(self, q: np.ndarray) -> CollisionResult:
        vp = self.violating_pairs(q)
        return CollisionResult(
            has_collision=len(vp) > 0,
            colliding_pairs=vp,
            min_distance_m=-1.0,
            closest_pair=vp[0] if vp else ("", ""),
            all_distances=[],
            meta={"subset": "midsole_policy"},
        )

    def has_collision(self, q: np.ndarray) -> bool:
        """True if midsole touches something outside the allowed tool naming heuristic."""
        return self.has_violation(q)
