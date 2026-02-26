#!/usr/bin/env python3
"""
Self-collision checker using Pinocchio + hpp-fcl.

Loads the URDF collision geometry and checks whether a given joint
configuration causes any non-adjacent links to penetrate each other.

NOTE:  The quality of collision detection depends entirely on the collision
meshes in the URDF.  If the meshes overlap at joint boundaries (common with
exported STL files), those pairs should be excluded.  The ``calibrate``
method identifies always-colliding pairs at the neutral configuration and
excludes them automatically.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pinocchio as pin


@dataclass
class CollisionResult:
    """Result of a self-collision query."""
    has_collision: bool
    colliding_pairs: List[Tuple[str, str]]
    min_distance_m: float
    closest_pair: Tuple[str, str]
    all_distances: List[Tuple[str, str, float]]


class SelfCollisionChecker:
    """Check self-collisions for a robot described by a URDF.

    Usage::

        checker = SelfCollisionChecker(urdf_path)
        checker.calibrate()   # removes structurally-overlapping mesh pairs
        result = checker.check(q)
        if result.has_collision:
            print("Self-collision detected:", result.colliding_pairs)
    """

    def __init__(self, urdf_path: str, min_joint_gap: int = 1):
        """
        Args:
            urdf_path:     Absolute or relative path to the URDF file.
            min_joint_gap: Minimum parent-joint index difference for a pair
                           to be considered.  1 keeps all non-self pairs;
                           increase to skip near-neighbour overlaps.
        """
        urdf_path = os.path.abspath(urdf_path)
        urdf_dir = os.path.dirname(urdf_path)

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.geom_model = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION,
            package_dirs=[urdf_dir],
        )

        self.geom_model.addAllCollisionPairs()

        # Remove pairs whose parent joints are too close in the kinematic chain
        to_remove = []
        for cp in self.geom_model.collisionPairs:
            j1 = self.geom_model.geometryObjects[cp.first].parentJoint
            j2 = self.geom_model.geometryObjects[cp.second].parentJoint
            if abs(j1 - j2) <= min_joint_gap:
                to_remove.append(cp)
        for cp in to_remove:
            self.geom_model.removeCollisionPair(cp)

        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)

        self._calibrated_pairs_removed: List[Tuple[str, str]] = []

    # -----------------------------------------------------------------
    def calibrate(self, q_ref: Optional[np.ndarray] = None) -> List[Tuple[str, str]]:
        """Remove collision pairs that *always* collide (mesh structural overlap).

        Tests at ``q_ref`` (default: neutral) and removes pairs that report
        collision there — these are mesh artefacts, not real self-collisions.

        Returns the list of removed pair names (for logging).
        """
        if q_ref is None:
            q_ref = pin.neutral(self.model)
        q = self._pad(q_ref)

        pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, q, False)

        to_remove = []
        for i, cr in enumerate(self.geom_data.collisionResults):
            if cr.isCollision():
                to_remove.append(self.geom_model.collisionPairs[i])

        removed_names = []
        for cp in to_remove:
            n1 = self.geom_model.geometryObjects[cp.first].name
            n2 = self.geom_model.geometryObjects[cp.second].name
            removed_names.append((n1, n2))
            self.geom_model.removeCollisionPair(cp)

        # Rebuild geom_data after modifying pairs
        self.geom_data = pin.GeometryData(self.geom_model)
        self._calibrated_pairs_removed = removed_names
        return removed_names

    # -----------------------------------------------------------------
    def check(self, q: np.ndarray) -> CollisionResult:
        """Check for self-collision at joint configuration *q* (6-DOF)."""
        q_full = self._pad(q)

        pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, q_full, False)
        pin.computeDistances(self.model, self.data, self.geom_model, self.geom_data, q_full)

        colliding: List[Tuple[str, str]] = []
        all_dist: List[Tuple[str, str, float]] = []
        min_dist = float('inf')
        closest = ('', '')

        for i in range(len(self.geom_model.collisionPairs)):
            cp = self.geom_model.collisionPairs[i]
            n1 = self.geom_model.geometryObjects[cp.first].name
            n2 = self.geom_model.geometryObjects[cp.second].name
            d = self.geom_data.distanceResults[i].min_distance
            all_dist.append((n1, n2, d))

            if self.geom_data.collisionResults[i].isCollision():
                colliding.append((n1, n2))

            if d < min_dist:
                min_dist = d
                closest = (n1, n2)

        return CollisionResult(
            has_collision=len(colliding) > 0,
            colliding_pairs=colliding,
            min_distance_m=min_dist if min_dist != float('inf') else -1.0,
            closest_pair=closest,
            all_distances=sorted(all_dist, key=lambda t: t[2]),
        )

    # -----------------------------------------------------------------
    def _pad(self, q: np.ndarray) -> np.ndarray:
        """Pad *q* to model.nq if it has fewer elements (e.g. 6 -> 7)."""
        q = np.asarray(q).flatten()
        if len(q) < self.model.nq:
            q_full = np.zeros(self.model.nq)
            q_full[: len(q)] = q
            return q_full
        return q[:self.model.nq]
