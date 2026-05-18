#!/usr/bin/env python3
"""URDF collision geometry, static mesh attachment, SE(3) helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pinocchio as pin

# Pinocchio fixed-base IRB models: universe joint 0, base frame id 1 (matches Base_link_0).
UNIVERSE_JOINT_ID = 0
BASE_FRAME_ID = 1


def resolve_urdf_path(urdf_path: str) -> str:
    """Resolve URDF path (absolute, CWD, then fuzzy URDF search)."""
    p = Path(urdf_path)
    if p.is_absolute() and p.exists():
        return str(p)
    cwd_path = (Path.cwd() / p).resolve()
    if cwd_path.exists():
        return str(cwd_path)
    try:
        from utils.urdf_loader import resolve_urdf_path as fuzzy

        return str(fuzzy(urdf_path))
    except Exception:
        pass
    raise FileNotFoundError(f"URDF not found: {urdf_path}")


def se3_from_pose_dict(pose: Dict[str, Any]) -> pin.SE3:
    """Build ``pin.SE3`` from a generic YAML pose (translation in **metres**).

    Supported keys:
      - ``xyz``: length-3 list (metres)
      - ``quaternion``: ``[qw, qx, qy, qz]`` (optional if rpy present)
      - ``rpy_rad`` or ``rpy_deg``: roll-pitch-yaw (optional if quaternion present)
    """
    xyz = np.asarray(pose.get("xyz", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    if "quaternion" in pose:
        q = np.asarray(pose["quaternion"], dtype=float).reshape(4)
        quat = pin.Quaternion(q[0], q[1], q[2], q[3])
        R = quat.matrix()
    elif "rpy_rad" in pose:
        rpy = np.asarray(pose["rpy_rad"], dtype=float).reshape(3)
        R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    elif "rpy_deg" in pose:
        rpy = np.deg2rad(np.asarray(pose["rpy_deg"], dtype=float).reshape(3))
        R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    else:
        R = np.eye(3)
    return pin.SE3(R, xyz)


def se3_from_collision_object_pose(pose: Dict[str, Any]) -> pin.SE3:
    """Build ``pin.SE3`` for ``collision_objects.yaml`` static objects.

    **Contract** (robot base frame):

    - ``xyz``: three translations in **millimetres**.
    - ``quaternion``: ``[qw, qx, qy, qz]`` only (no ``rpy_*``).

    Translation is converted to metres for Pinocchio.
    """
    if not isinstance(pose, dict):
        raise TypeError("pose must be a dict")
    for k in ("rpy_deg", "rpy_rad"):
        if k in pose:
            raise ValueError(
                f"collision_objects.yaml pose must use quaternion [qw,qx,qy,qz] only; remove {k!r}"
            )
    if "xyz" not in pose:
        raise ValueError("collision object pose requires 'xyz' [mm, mm, mm]")
    if "quaternion" not in pose:
        raise ValueError(
            "collision object pose requires 'quaternion' [qw, qx, qy, qz]"
        )
    xyz_mm = np.asarray(pose["xyz"], dtype=float).reshape(3)
    t_m = xyz_mm * 0.001
    q = np.asarray(pose["quaternion"], dtype=float).reshape(4)
    quat = pin.Quaternion(q[0], q[1], q[2], q[3])
    R = quat.matrix()
    return pin.SE3(R, t_m)


def load_mesh_collision_geometry(mesh_abs_path: str):
    """Load a triangle mesh as a **coal** collision geometry via Pinocchio."""
    loader = pin.MeshLoader()
    geom = loader.load(mesh_abs_path)
    if geom is None:
        raise RuntimeError(f"MeshLoader returned None for: {mesh_abs_path}")
    return geom


def build_robot_collision_geometry(
    urdf_path: str,
) -> Tuple[pin.Model, pin.GeometryModel, str, str, str, int]:
    """Load Pinocchio model + collision ``GeometryModel`` from URDF.

    Returns:
        (model, geom_model, urdf_abs, urdf_dir, mesh_root, n_robot_geom)
    """
    urdf_abs = resolve_urdf_path(urdf_path)
    urdf_dir = os.path.dirname(urdf_abs)
    mesh_root = os.path.dirname(urdf_dir)
    model = pin.buildModelFromUrdf(urdf_abs)
    geom_model = pin.buildGeomFromUrdf(
        model,
        urdf_abs,
        pin.GeometryType.COLLISION,
        package_dirs=[urdf_dir, mesh_root],
    )
    n_robot = len(geom_model.geometryObjects)
    if n_robot == 0:
        raise ValueError(
            f"No collision geometry in URDF: {urdf_abs}\n"
            "Ensure <collision> tags reference valid STL mesh files."
        )
    return model, geom_model, urdf_abs, urdf_dir, mesh_root, n_robot


def append_fixed_scene_geometry(
    geom_model: pin.GeometryModel,
    name: str,
    mesh_abs_path: str,
    placement_world: pin.SE3,
    mesh_scale: Optional[np.ndarray] = None,
) -> int:
    """Append a static mesh fixed in the robot base / universe frame.

    Args:
        mesh_scale: Optional length-3 scale applied to mesh vertices (Pinocchio
            ``GeometryObject`` mesh scale); use e.g. ``0.001`` if STL units are mm.

    Returns:
        Index of the new ``GeometryObject`` in ``geom_model.geometryObjects``.
    """
    mesh = load_mesh_collision_geometry(mesh_abs_path)
    if mesh_scale is not None:
        ms = np.asarray(mesh_scale, dtype=float).reshape(3)
        go = pin.GeometryObject(
            name,
            UNIVERSE_JOINT_ID,
            BASE_FRAME_ID,
            placement_world,
            mesh,
            mesh_abs_path,
            ms,
        )
    else:
        go = pin.GeometryObject(
            name,
            UNIVERSE_JOINT_ID,
            BASE_FRAME_ID,
            placement_world,
            mesh,
            mesh_abs_path,
        )
    return geom_model.addGeometryObject(go)


def pad_q(model: pin.Model, q: np.ndarray) -> np.ndarray:
    """Pad *q* to ``model.nq`` if shorter (e.g. 6-DOF arm)."""
    q = np.asarray(q, dtype=float).flatten()
    if len(q) < model.nq:
        q_full = np.zeros(model.nq)
        q_full[: len(q)] = q
        return q_full
    return q[: model.nq]


def ensure_collision_requests(
    model: pin.Model,
    data: pin.Data,
    geom_model: pin.GeometryModel,
    geom_data: pin.GeometryData,
    q: np.ndarray,
) -> None:
    """Run one collision pass to size ``geom_data.collisionRequests`` (Pinocchio 3)."""
    q_full = pad_q(model, q)
    pin.computeCollisions(model, data, geom_model, geom_data, q_full, False)
