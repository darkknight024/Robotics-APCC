#!/usr/bin/env python3
"""
Pure C-space forbidden-zone gate (no FK, no meshes, no coal).

Use for deterministic unit tests and feasibility/EAIK branch filtering::

    checker = CSpaceForbiddenChecker.from_yaml("config/cspace_forbidden_zones_irb1300_714.yaml")
    checker.has_collision(q_rad)  # True if q lies in any forbidden hyper-rectangle
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .cspace_config import CSpaceForbiddenZonesFile, ForbiddenZoneDeg, load_cspace_forbidden_zones


class CSpaceForbiddenChecker:
    """Test double: collision = membership in any configured joint-space box."""

    def __init__(self, spec: CSpaceForbiddenZonesFile):
        self.spec = spec
        self.robot_name = spec.robot_name
        self.zones: List[ForbiddenZoneDeg] = list(spec.zones)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        *,
        project_root: Optional[Path] = None,
    ) -> "CSpaceForbiddenChecker":
        return cls(load_cspace_forbidden_zones(yaml_path, project_root=project_root))

    def has_collision(self, q: np.ndarray) -> bool:
        """True if ``q`` (radians, length n_joints) lies inside any forbidden zone."""
        q_deg = np.rad2deg(np.asarray(q, dtype=float).flatten())
        for zone in self.zones:
            if zone.contains_q_deg(q_deg):
                return True
        return False

    def colliding_zone_names(self, q: np.ndarray) -> List[str]:
        q_deg = np.rad2deg(np.asarray(q, dtype=float).flatten())
        return [z.name for z in self.zones if z.contains_q_deg(q_deg)]

    def __len__(self) -> int:
        return len(self.zones)
