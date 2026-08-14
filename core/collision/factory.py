#!/usr/bin/env python3
"""Build collision gates for feasibility / EAIK (scene, C-space, or both)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

from .cspace_checker import CSpaceForbiddenChecker
from .scene_checker import SceneCollisionChecker


class _CollisionGate(Protocol):
    def has_collision(self, q: np.ndarray) -> bool: ...


class CompositeCollisionChecker:
    """OR-combine multiple checkers (e.g. scene self+env AND optional C-space test zones)."""

    def __init__(self, checkers: Sequence[_CollisionGate]):
        self._checkers = list(checkers)

    def has_collision(self, q: np.ndarray) -> bool:
        return any(c.has_collision(q) for c in self._checkers)

    def colliding_sources(self, q: np.ndarray) -> List[str]:
        names: List[str] = []
        for c in self._checkers:
            if c.has_collision(q):
                names.append(type(c).__name__)
        return names


def build_collision_checker_for_feasibility(
    *,
    urdf_path: str,
    project_root: Optional[Path] = None,
    scene_yaml: Optional[str] = None,
    scene_calibrate: bool = True,
    scene_calibrate_n_samples: int = 32,
    scene_calibrate_seed: int = 42,
    cspace_forbidden_yaml: Optional[str] = None,
    verbose: bool = False,
    fixture_name: Optional[str] = "ee_link",
) -> Optional[CompositeCollisionChecker]:
    """Create the collision gate used by :class:`~core.feasibility.analyzer.FeasibilityAnalyzer`.

    Returns ``None`` when no collision sources are configured.

    * ``scene_yaml`` — full :class:`SceneCollisionChecker` (URDF self + static obstacles).
    * ``cspace_forbidden_yaml`` — optional pure joint-space zones (tests / artificial failures).
    * ``fixture_name`` — ``fixture_config.yaml`` entry (default ``ee_link``) whose
      ``stl`` is attached to the flange for collision.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    parts: List[_CollisionGate] = []

    if scene_yaml:
        scene_path = Path(scene_yaml)
        if not scene_path.is_absolute():
            scene_path = (project_root / scene_path).resolve()
        parts.append(
            SceneCollisionChecker.from_urdf_and_scene_yaml(
                urdf_path,
                str(scene_path),
                project_root=project_root,
                calibrate=scene_calibrate,
                calibrate_n_samples=scene_calibrate_n_samples,
                calibrate_seed=scene_calibrate_seed,
                verbose=verbose,
                fixture_name=fixture_name,
            )
        )

    if cspace_forbidden_yaml:
        parts.append(
            CSpaceForbiddenChecker.from_yaml(
                cspace_forbidden_yaml, project_root=project_root,
            )
        )

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]  # type: ignore[return-value]
    return CompositeCollisionChecker(parts)
