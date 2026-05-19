#!/usr/bin/env python3
"""Feature 4 — collision checking (Pinocchio + coal / hpp-fcl)."""

from .cspace_checker import CSpaceForbiddenChecker
from .factory import CompositeCollisionChecker, build_collision_checker_for_feasibility
from .cspace_config import CSpaceForbiddenZonesFile, ForbiddenZoneDeg, load_cspace_forbidden_zones
from .geometry import se3_from_collision_object_pose, se3_from_pose_dict
from .midsole_checker import MidsoleCollisionChecker
from .object_checker import ObjectCollisionChecker
from .scene_checker import SceneCollisionChecker
from .self_checker import SelfCollisionChecker
from .trajectory_checker import TrajectoryCollisionChecker
from .types import CollisionResult, TrajectoryCollisionReport, WaypointCollisionResult
from .scene_config import CollisionObjectSpec, CollisionObjectsFile

__all__ = [
    "CompositeCollisionChecker",
    "build_collision_checker_for_feasibility",
    "CSpaceForbiddenChecker",
    "CSpaceForbiddenZonesFile",
    "ForbiddenZoneDeg",
    "load_cspace_forbidden_zones",
    "CollisionResult",
    "CollisionObjectSpec",
    "CollisionObjectsFile",
    "se3_from_collision_object_pose",
    "se3_from_pose_dict",
    "MidsoleCollisionChecker",
    "ObjectCollisionChecker",
    "SceneCollisionChecker",
    "SelfCollisionChecker",
    "TrajectoryCollisionChecker",
    "TrajectoryCollisionReport",
    "WaypointCollisionResult",
]
