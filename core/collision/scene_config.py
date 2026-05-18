#!/usr/bin/env python3
"""YAML schema and loader for static collision objects (Feature 4).

Object ``pose`` uses millimetre ``xyz`` and ``quaternion`` [qw,qx,qy,qz]; see
:func:`core.collision.geometry.se3_from_collision_object_pose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class CollisionObjectSpec:
    """One static collidable object in the cell.

    ``pose`` dict must follow :func:`core.collision.geometry.se3_from_collision_object_pose`:
    ``xyz`` in millimetres, ``quaternion`` as ``[qw, qx, qy, qz]``.
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
    # Optional uniform scale on mesh vertices (e.g. [0.001,0.001,0.001] if STL is in mm).
    mesh_scale: Optional[List[float]] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "CollisionObjectSpec":
        pose = raw.get("pose") or {}
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
        ms = raw.get("mesh_scale")
        mesh_scale: Optional[List[float]] = None
        if ms is not None:
            mesh_scale = [float(x) for x in list(ms)]

        return CollisionObjectSpec(
            name=str(raw["name"]),
            group=str(raw.get("group", "environment")),
            mesh_path=str(raw["mesh_path"]),
            pose=pose if isinstance(pose, dict) else {},
            enabled=bool(raw.get("enabled", True)),
            collision_tolerance_m=float(raw.get("collision_tolerance_m", 0.0)),
            simplified_mesh_path=raw.get("simplified_mesh_path"),
            decimation_ratio=(
                float(dec["ratio"]) if dec.get("ratio") is not None else None
            ),
            decimation_cache_path=dec.get("cache_path"),
            convex_mesh_paths=[str(p) for p in paths],
            whitelist_pairs=parsed_wl,
            mesh_scale=mesh_scale,
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
