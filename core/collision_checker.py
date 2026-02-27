#!/usr/bin/env python3
"""
Solver-agnostic self-collision checker using Pinocchio + hpp-fcl.

Loads the robot URDF with its collision STL meshes and checks whether a
given joint configuration causes non-adjacent links to penetrate each other.

The checker is intentionally decoupled from any particular IK solver —
it accepts a raw joint-angle vector (radians) and returns whether
self-collision exists.  This allows it to work identically with EAIK,
Pinocchio IK, or any future solver backend.

Typical usage::

    checker = SelfCollisionChecker.from_robot_name("IRB 1300-7/1.4")
    checker.calibrate()          # exclude structural mesh overlaps

    if checker.has_self_collision(q):
        print("Self-collision at this configuration!")
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import pinocchio as pin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CollisionResult:
    """Detailed result of a self-collision query."""
    has_collision: bool
    colliding_pairs: List[Tuple[str, str]]
    min_distance_m: float
    closest_pair: Tuple[str, str]
    all_distances: List[Tuple[str, str, float]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SelfCollisionChecker:
    """Solver-agnostic self-collision checker for 6-DOF robot arms.

    Loads the URDF collision geometry (STL meshes) via Pinocchio + hpp-fcl
    and checks whether a given joint configuration *q* (radians) causes any
    non-adjacent links to collide.

    The URDF ``<collision>`` tags reference the same STL files used for
    visualization.  These detailed meshes often overlap at joint boundaries
    (a structural artifact of SolidWorks export).  :meth:`calibrate` tests
    several configurations and automatically excludes pairs that always
    collide — treating them as mesh artifacts rather than real collisions.
    """

    # --------------------------------------------------------------------- #
    #  Construction helpers
    # --------------------------------------------------------------------- #

    @classmethod
    def from_robot_name(
        cls,
        robot_name: str,
        robots_config_path: Optional[str] = None,
        **kwargs,
    ) -> "SelfCollisionChecker":
        """Instantiate by looking up the robot in ``robots_config.yaml``.

        Args:
            robot_name: Name as listed in *robots_config.yaml*
                        (e.g. ``"IRB 1300-7/1.4"``).
            robots_config_path: Optional override for the config file path.
            **kwargs: Forwarded to :meth:`__init__`
                      (*min_joint_gap*, *verbose*).
        """
        from utils.config_loader import get_robot_by_name

        robot_cfg = get_robot_by_name(robot_name, robots_config_path)
        return cls(urdf_path=robot_cfg.urdf_path, **kwargs)

    # --------------------------------------------------------------------- #
    #  Init
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        urdf_path: str,
        min_joint_gap: int = 1,
        verbose: bool = False,
    ):
        """
        Args:
            urdf_path: Absolute path, or path relative to the project root.
                       The URDF ``<collision>`` tags must reference STL files
                       via relative paths (e.g. ``../meshes/Link_1.STL``).
            min_joint_gap: Minimum parent-joint index separation for a link
                           pair to be included.  1 keeps all non-self pairs;
                           increase to skip near-neighbour overlaps outright.
            verbose: Log diagnostic info during init / calibrate.
        """
        self._verbose = verbose

        urdf_abs = self._resolve_path(urdf_path)
        urdf_dir = os.path.dirname(urdf_abs)
        mesh_root = os.path.dirname(urdf_dir)  # one level up for ../meshes/

        # Kinematic model (needed for forward kinematics during collision)
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_abs)

        # Collision geometry model — loads STL meshes from <collision> tags.
        # package_dirs provides search roots so Pinocchio can resolve the
        # relative ``../meshes/*.STL`` paths found in the URDF.
        self.geom_model: pin.GeometryModel = pin.buildGeomFromUrdf(
            self.model,
            urdf_abs,
            pin.GeometryType.COLLISION,
            package_dirs=[urdf_dir, mesh_root],
        )

        n_geom = len(self.geom_model.geometryObjects)
        if n_geom == 0:
            raise ValueError(
                f"No collision geometry found in URDF: {urdf_abs}\n"
                "Ensure <collision> tags reference valid STL mesh files."
            )

        if verbose:
            logger.info(
                "Loaded %d collision geometries from %s", n_geom, urdf_abs
            )
            for i, go in enumerate(self.geom_model.geometryObjects):
                logger.info(
                    "  [%d] %s  (parent joint %d)", i, go.name, go.parentJoint
                )

        # Register all possible link-pair collisions, then prune adjacent ones
        self.geom_model.addAllCollisionPairs()
        self._remove_adjacent_pairs(min_joint_gap)

        self.data: pin.Data = self.model.createData()
        self.geom_data: pin.GeometryData = pin.GeometryData(self.geom_model)

        self._excluded_pairs: List[Tuple[str, str]] = []
        self._is_calibrated = False

        if verbose:
            logger.info(
                "Active collision pairs after adjacency filter: %d",
                len(self.geom_model.collisionPairs),
            )

    # --------------------------------------------------------------------- #
    #  Calibration — exclude structural mesh overlaps
    # --------------------------------------------------------------------- #

    def calibrate(
        self,
        n_samples: int = 10,
        seed: int = 42,
    ) -> List[Tuple[str, str]]:
        """Exclude link pairs that collide at *every* test configuration.

        These are structural STL overlaps at joint boundaries — not real
        collisions.  Testing at multiple random configs (plus neutral)
        avoids mistakenly excluding pairs that only collide in certain poses.

        Args:
            n_samples: Number of random configs to probe (in addition to the
                       neutral / zero configuration).
            seed: RNG seed for reproducibility.

        Returns:
            Names of excluded pairs (for logging / debugging).
        """
        rng = np.random.RandomState(seed)

        lower = self.model.lowerPositionLimit[: self.model.nq]
        upper = self.model.upperPositionLimit[: self.model.nq]

        configs = [pin.neutral(self.model)]
        for _ in range(n_samples):
            configs.append(lower + rng.rand(self.model.nq) * (upper - lower))

        n_pairs = len(self.geom_model.collisionPairs)
        hit_counts = np.zeros(n_pairs, dtype=int)

        for q in configs:
            q_full = self._pad(q)
            pin.computeCollisions(
                self.model, self.data,
                self.geom_model, self.geom_data,
                q_full, False,
            )
            for i, cr in enumerate(self.geom_data.collisionResults):
                if cr.isCollision():
                    hit_counts[i] += 1

        # Pairs that collided in ALL configs are artifacts
        n_total = len(configs)
        to_remove = []
        for i in range(n_pairs):
            if hit_counts[i] == n_total:
                cp = self.geom_model.collisionPairs[i]
                to_remove.append(
                    (pin.CollisionPair(cp.first, cp.second),
                     self.geom_model.geometryObjects[cp.first].name,
                     self.geom_model.geometryObjects[cp.second].name)
                )

        excluded_names: List[Tuple[str, str]] = []
        for cp, n1, n2 in to_remove:
            excluded_names.append((n1, n2))
            self.geom_model.removeCollisionPair(cp)

        # Rebuild data after removing pairs
        self.geom_data = pin.GeometryData(self.geom_model)
        self._excluded_pairs = excluded_names
        self._is_calibrated = True

        if self._verbose:
            logger.info(
                "Calibration excluded %d always-colliding pairs:",
                len(excluded_names),
            )
            for n1, n2 in excluded_names:
                logger.info("  %s <-> %s", n1, n2)
            logger.info(
                "Remaining active pairs: %d",
                len(self.geom_model.collisionPairs),
            )

        return excluded_names

    # --------------------------------------------------------------------- #
    #  Collision queries
    # --------------------------------------------------------------------- #

    def check(self, q: np.ndarray) -> CollisionResult:
        """Full collision query with distance information.

        Args:
            q: Joint angles in radians, shape ``(n_joints,)``
               (typically 6 for a 6-DOF arm).

        Returns:
            :class:`CollisionResult` with boolean status, colliding pair
            names, minimum distance, and per-pair distances.
        """
        if not self._is_calibrated:
            logger.warning(
                "check() called before calibrate(). "
                "Results may include structural mesh-overlap false positives."
            )

        q_full = self._pad(q)

        pin.computeCollisions(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q_full, False,
        )
        pin.computeDistances(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q_full,
        )

        colliding: List[Tuple[str, str]] = []
        all_dist: List[Tuple[str, str, float]] = []
        min_dist = float("inf")
        closest: Tuple[str, str] = ("", "")

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
            min_distance_m=min_dist if min_dist != float("inf") else -1.0,
            closest_pair=closest,
            all_distances=sorted(all_dist, key=lambda t: t[2]),
        )

    def has_self_collision(self, q: np.ndarray) -> bool:
        """Quick boolean: does configuration *q* cause self-collision?

        This is the primary entry point for the feasibility pipeline.

        Args:
            q: Joint angles in radians, shape ``(n_joints,)``.

        Returns:
            ``True`` if any (non-excluded) link pair collides.
        """
        q_full = self._pad(q)
        return pin.computeCollisions(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q_full, True,  # stop_at_first_collision=True for speed
        )

    # --------------------------------------------------------------------- #
    #  Properties
    # --------------------------------------------------------------------- #

    @property
    def excluded_pairs(self) -> List[Tuple[str, str]]:
        """Pairs excluded during calibration (structural overlaps)."""
        return list(self._excluded_pairs)

    @property
    def active_pair_count(self) -> int:
        """Number of collision pairs actively being checked."""
        return len(self.geom_model.collisionPairs)

    @property
    def n_joints(self) -> int:
        """Number of joints in the kinematic model (``model.nq``)."""
        return self.model.nq

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    def _remove_adjacent_pairs(self, min_joint_gap: int) -> None:
        """Drop collision pairs whose parent joints are kinematically adjacent."""
        to_remove = []
        for i in range(len(self.geom_model.collisionPairs)):
            cp = self.geom_model.collisionPairs[i]
            j1 = self.geom_model.geometryObjects[cp.first].parentJoint
            j2 = self.geom_model.geometryObjects[cp.second].parentJoint
            if abs(j1 - j2) <= min_joint_gap:
                to_remove.append(pin.CollisionPair(cp.first, cp.second))
        for cp in to_remove:
            self.geom_model.removeCollisionPair(cp)

    def _pad(self, q: np.ndarray) -> np.ndarray:
        """Pad *q* to ``model.nq`` if it has fewer elements."""
        q = np.asarray(q, dtype=float).flatten()
        if len(q) < self.model.nq:
            q_full = np.zeros(self.model.nq)
            q_full[: len(q)] = q
            return q_full
        return q[: self.model.nq]

    @staticmethod
    def _resolve_path(urdf_path: str) -> str:
        """Resolve a URDF path to an absolute path on disk.

        Tries, in order:
          1. Path as-is (if already absolute and exists).
          2. Relative to the current working directory.
          3. Via the project's :func:`utils.urdf_loader.resolve_urdf_path`
             (which supports fuzzy matching).
        """
        p = Path(urdf_path)
        if p.is_absolute() and p.exists():
            return str(p)

        cwd_path = (Path.cwd() / p).resolve()
        if cwd_path.exists():
            return str(cwd_path)

        try:
            from utils.urdf_loader import resolve_urdf_path
            return str(resolve_urdf_path(urdf_path))
        except Exception:
            pass

        raise FileNotFoundError(
            f"URDF not found: {urdf_path}\n"
            f"Tried absolute, relative to CWD ({Path.cwd()}), "
            "and fuzzy URDF search."
        )
