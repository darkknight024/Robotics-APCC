#!/usr/bin/env python3
"""YAML schema and loader for static collision objects (Feature 4).

Supported object schemas (robot base frame):

1. **Flat Exp25 / Viser schema** (preferred for cell meshes)::

       name: ...
       mesh_path: ...
       position_mm: [x, y, z]          # millimetres
       quat_wxyz: [qw, qx, qy, qz]     # required for collision
       scale: 1.0                      # uniform mesh vertex scale
       collision: true|false           # skip object when false

   ``orientation_deg`` may appear for viewer display but is **ignored** for
   collision; use ``quat_wxyz`` only.

2. **Legacy nested pose**::

       mesh_path: ...
       pose: {xyz: [mm,mm,mm], quaternion: [qw,qx,qy,qz]}
       mesh_scale: [sx, sy, sz]
       enabled: true|false

Pose → SE(3) conversion:
:func:`core.collision.geometry.se3_from_collision_object_pose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _normalize_pose(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Build canonical ``{xyz, quaternion}`` from flat or nested object fields.

    Collision always uses quaternion ``[qw, qx, qy, qz]``. Euler / RPY fields
    are not accepted for collision placement.
    """
    pose_raw = raw.get("pose")
    if isinstance(pose_raw, dict) and pose_raw:
        xyz = pose_raw.get("xyz", pose_raw.get("position_mm"))
        quat = (
            pose_raw.get("quaternion")
            or pose_raw.get("quat_wxyz")
            or pose_raw.get("quat")
        )
        for banned in ("rpy_deg", "rpy_rad", "orientation_deg", "rpy"):
            if banned in pose_raw:
                raise ValueError(
                    f"{name}: collision pose must use quaternion [qw,qx,qy,qz]; "
                    f"remove {banned!r}"
                )
    else:
        xyz = raw.get("position_mm", raw.get("xyz"))
        quat = raw.get("quat_wxyz", raw.get("quaternion", raw.get("quat")))
        for banned in ("rpy_deg", "rpy_rad"):
            if banned in raw and quat is None:
                raise ValueError(
                    f"{name}: collision objects require quat_wxyz [qw,qx,qy,qz]; "
                    f"do not use {banned!r} alone"
                )

    if xyz is None:
        raise ValueError(
            f"{name}: missing position_mm / pose.xyz [mm, mm, mm]"
        )
    if quat is None:
        raise ValueError(
            f"{name}: missing quat_wxyz / pose.quaternion [qw, qx, qy, qz] "
            "(orientation_deg is not used for collision)"
        )

    xyz_list = [float(v) for v in list(xyz)]
    quat_list = [float(v) for v in list(quat)]
    if len(xyz_list) != 3:
        raise ValueError(f"{name}: position must have 3 values, got {len(xyz_list)}")
    if len(quat_list) != 4:
        raise ValueError(f"{name}: quaternion must have 4 values, got {len(quat_list)}")
    return {"xyz": xyz_list, "quaternion": quat_list}


def _resolve_enabled(raw: Dict[str, Any]) -> bool:
    """``collision: false`` disables the object; else ``enabled`` (default True)."""
    if "collision" in raw:
        return bool(raw["collision"])
    return bool(raw.get("enabled", True))


def _resolve_mesh_scale(raw: Dict[str, Any]) -> Optional[List[float]]:
    """Uniform ``scale`` or explicit ``mesh_scale`` → length-3 list for Pinocchio."""
    ms = raw.get("mesh_scale")
    if ms is not None:
        return [float(x) for x in list(ms)]
    if "scale" in raw and raw["scale"] is not None:
        s = float(raw["scale"])
        return [s, s, s]
    return None


@dataclass
class CollisionObjectSpec:
    """One static collidable object in the cell.

    ``pose`` is always normalized to ``xyz`` (mm) + ``quaternion`` ``[qw,qx,qy,qz]``.
    """

    name: str
    group: str  # environment | tool | other
    mesh_path: str
    pose: Dict[str, Any]
    enabled: bool = True
    collision_tolerance_m: float = 0.0
    simplified_mesh_path: Optional[str] = None
    decimation_ratio: Optional[float] = None
    decimation_cache_path: Optional[str] = None
    convex_mesh_paths: List[str] = field(default_factory=list)
    whitelist_pairs: List[Tuple[str, str]] = field(default_factory=list)
    # Optional uniform/per-axis scale on mesh vertices (Pinocchio GeometryObject).
    # Exp25 STLs are already in metres → scale 1.0. Internal-collision box STLs
    # that store millimetre vertices use mesh_scale [0.001, 0.001, 0.001].
    mesh_scale: Optional[List[float]] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "CollisionObjectSpec":
        name = str(raw["name"])
        if "mesh_path" not in raw:
            raise ValueError(f"{name}: mesh_path is required for collision objects")

        pose = _normalize_pose(raw, name)
        dec = raw.get("decimation") or {}
        conv = raw.get("convex_decomposition") or {}
        paths = conv.get("mesh_paths") or raw.get("convex_mesh_paths") or []
        wl = raw.get("whitelist_pairs") or []
        parsed_wl: List[Tuple[str, str]] = []
        for entry in wl:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                parsed_wl.append((str(entry[0]), str(entry[1])))
            elif isinstance(entry, dict) and "a" in entry and "b" in entry:
                parsed_wl.append((str(entry["a"]), str(entry["b"])))

        return CollisionObjectSpec(
            name=name,
            group=str(raw.get("group", "environment")),
            mesh_path=str(raw["mesh_path"]),
            pose=pose,
            enabled=_resolve_enabled(raw),
            collision_tolerance_m=float(raw.get("collision_tolerance_m", 0.0)),
            simplified_mesh_path=raw.get("simplified_mesh_path"),
            decimation_ratio=(
                float(dec["ratio"]) if dec.get("ratio") is not None else None
            ),
            decimation_cache_path=dec.get("cache_path"),
            convex_mesh_paths=[str(p) for p in paths],
            whitelist_pairs=parsed_wl,
            mesh_scale=_resolve_mesh_scale(raw),
        )


@dataclass
class CollisionObjectsFile:
    """Root document for ``collision_objects.yaml``."""

    version: int
    objects: List[CollisionObjectSpec]
    # Default midsole / knife geometry names in the **URDF** for whitelist
    midsole_geom_name: Optional[str] = None
    knife_blade_geom_name: Optional[str] = None

    @staticmethod
    def load(path: str | Path) -> "CollisionObjectsFile":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        objs = [
            CollisionObjectSpec.from_dict(o)
            for o in (doc.get("objects") or [])
        ]
        return CollisionObjectsFile(
            version=int(doc.get("version", 1)),
            objects=objs,
            midsole_geom_name=doc.get("midsole_geom_name"),
            knife_blade_geom_name=doc.get("knife_blade_geom_name"),
        )
