#!/usr/bin/env python3
"""Unit tests for pure C-space forbidden-zone gating (no Pinocchio / meshes)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from core.collision import CSpaceForbiddenChecker, load_cspace_forbidden_zones

_REPO = Path(__file__).resolve().parent.parent
_YAML = _REPO / "config" / "cspace_forbidden_zones_irb1300_714.yaml"


class TestCSpaceForbidden(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = CSpaceForbiddenChecker.from_yaml(_YAML, project_root=_REPO)

    def test_loads_five_zones(self) -> None:
        self.assertEqual(len(self.checker), 5)
        self.assertEqual(self.checker.robot_name, "IRB 1300-7/1.4")

    def test_j1_band_hit_and_miss(self) -> None:
        q_in = np.deg2rad([5.0, 0, 0, 0, 0, 0])
        q_out = np.deg2rad([30.0, 0, 0, 0, 0, 0])
        self.assertTrue(self.checker.has_collision(q_in))
        self.assertEqual(
            self.checker.colliding_zone_names(q_in), ["zone_j1_near_zero"]
        )
        self.assertFalse(self.checker.has_collision(q_out))

    def test_j3_band_isolated(self) -> None:
        q = np.deg2rad([30.0, 0, -190, 0, 0, 0])
        self.assertTrue(self.checker.has_collision(q))
        self.assertIn("zone_j3_low_elbow", self.checker.colliding_zone_names(q))

    def test_single_point_band_rejected(self) -> None:
        bad_yaml = {
            "robot_name": "IRB 1300-7/1.4",
            "forbidden_zones": [
                {"name": "point", "joints_deg": {"j1": {"min_deg": 10.0, "max_deg": 10.0}}},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_yaml, f)
            path = f.name
        with self.assertRaisesRegex(ValueError, "single point"):
            load_cspace_forbidden_zones(path)

    def test_band_outside_robot_limits_rejected(self) -> None:
        bad_yaml = {
            "robot_name": "IRB 1300-7/1.4",
            "forbidden_zones": [
                {
                    "name": "too_high_j2",
                    "joints_deg": {"j2": {"min_deg": 200.0, "max_deg": 210.0}},
                },
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_yaml, f)
            path = f.name
        with self.assertRaisesRegex(ValueError, "exceeds robot limits"):
            load_cspace_forbidden_zones(path)


if __name__ == "__main__":
    unittest.main()
