#!/usr/bin/env python3
"""Load pseudo C-space forbidden zones from YAML (test harness / deterministic gating)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.config_loader import get_robot_by_name, load_yaml


@dataclass(frozen=True)
class JointBandDeg:
    """Inclusive joint-angle band in degrees for one axis."""

    joint_index: int  # 1-based (j1 .. j6)
    min_deg: float
    max_deg: float

    def contains_deg(self, angle_deg: float) -> bool:
        return self.min_deg <= angle_deg <= self.max_deg


@dataclass(frozen=True)
class ForbiddenZoneDeg:
    """Hyper-rectangle in joint space: all listed joints must lie inside their bands."""

    name: str
    bands: Tuple[JointBandDeg, ...]
    description: str = ""

    def contains_q_deg(self, q_deg: np.ndarray) -> bool:
        for band in self.bands:
            idx = band.joint_index - 1
            if idx < 0 or idx >= len(q_deg):
                return False
            if not band.contains_deg(float(q_deg[idx])):
                return False
        return True


@dataclass(frozen=True)
class CSpaceForbiddenZonesFile:
    robot_name: str
    zones: Tuple[ForbiddenZoneDeg, ...]
    joint_limits_deg: Tuple[Tuple[float, float], ...]  # (min, max) per joint from robots_config


def _parse_joint_band(raw: Dict[str, Any], joint_key: str) -> JointBandDeg:
    j_idx = int(str(joint_key).lower().replace("j", ""))
    min_deg = float(raw["min_deg"])
    max_deg = float(raw["max_deg"])
    if max_deg < min_deg:
        raise ValueError(f"joint {joint_key}: max_deg < min_deg ({max_deg} < {min_deg})")
    if max_deg == min_deg:
        raise ValueError(
            f"joint {joint_key}: forbidden band must span a range (got single point {min_deg})"
        )
    return JointBandDeg(joint_index=j_idx, min_deg=min_deg, max_deg=max_deg)


def _robot_joint_limits_deg(robot_name: str) -> Tuple[Tuple[float, float], ...]:
    robots_yaml = load_yaml(
        str(Path(__file__).resolve().parents[2] / "config" / "robots_config.yaml")
    )
    for entry in robots_yaml.get("robots", []):
        if entry.get("name") != robot_name:
            continue
        limits = entry.get("joint_limits_deg")
        if not limits:
            break
        out: List[Tuple[float, float]] = []
        for i, lim in enumerate(limits):
            lo = float(lim["min"])
            hi = float(lim["max"])
            if hi <= lo:
                raise ValueError(f"robot {robot_name} joint {i + 1}: invalid limits [{lo}, {hi}]")
            out.append((lo, hi))
        return tuple(out)
    # Fallback: URDF via RobotConfig name (no limits in yaml)
    robot = get_robot_by_name(robot_name)
    if robot.joint_limits_deg:
        return tuple((float(a), float(b)) for a, b in robot.joint_limits_deg)
    raise ValueError(
        f"No joint_limits_deg for robot {robot_name!r} in robots_config.yaml"
    )


def validate_zones_within_robot_limits(
    zones: List[ForbiddenZoneDeg],
    joint_limits_deg: Tuple[Tuple[float, float], ...],
) -> None:
    """Ensure every forbidden band lies inside the robot's joint limits."""
    n_j = len(joint_limits_deg)
    for zone in zones:
        for band in zone.bands:
            if band.joint_index < 1 or band.joint_index > n_j:
                raise ValueError(
                    f"zone {zone.name!r}: joint j{band.joint_index} out of range (n={n_j})"
                )
            lo, hi = joint_limits_deg[band.joint_index - 1]
            if band.min_deg < lo or band.max_deg > hi:
                raise ValueError(
                    f"zone {zone.name!r} j{band.joint_index}: band "
                    f"[{band.min_deg}, {band.max_deg}] exceeds robot limits [{lo}, {hi}]"
                )


def load_cspace_forbidden_zones(
    yaml_path: str | Path,
    *,
    project_root: Optional[Path] = None,
) -> CSpaceForbiddenZonesFile:
    """Load ``cspace_forbidden_zones*.yaml`` and validate bands against ``robots_config``."""
    path = Path(yaml_path)
    if project_root is not None and not path.is_absolute():
        path = (Path(project_root) / path).resolve()
    raw = load_yaml(str(path))
    robot_name = str(raw.get("robot_name", ""))
    if not robot_name:
        raise ValueError(f"Missing robot_name in {path}")

    joint_limits_deg = _robot_joint_limits_deg(robot_name)
    zones: List[ForbiddenZoneDeg] = []
    for z in raw.get("forbidden_zones", []):
        name = str(z.get("name", "unnamed"))
        desc = str(z.get("description", ""))
        joints_raw = z.get("joints_deg") or {}
        if not joints_raw:
            raise ValueError(f"zone {name!r}: empty joints_deg")
        bands = tuple(
            _parse_joint_band(band_raw, joint_key)
            for joint_key, band_raw in sorted(
                joints_raw.items(),
                key=lambda kv: int(str(kv[0]).lower().replace("j", "")),
            )
        )
        zones.append(ForbiddenZoneDeg(name=name, bands=bands, description=desc))

    validate_zones_within_robot_limits(zones, joint_limits_deg)
    return CSpaceForbiddenZonesFile(
        robot_name=robot_name,
        zones=tuple(zones),
        joint_limits_deg=joint_limits_deg,
    )
