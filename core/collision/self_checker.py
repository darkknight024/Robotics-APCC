#!/usr/bin/env python3
"""URDF-only self-collision checker (robot + optional fixture STL from fixture_config)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from .geometry import build_robot_collision_geometry, pad_q
from .pair_rules import add_robot_self_pairs, remove_adjacent_pairs
from .types import CollisionResult

logger = logging.getLogger(__name__)


class SelfCollisionChecker:
    """Solver-agnostic self-collision checker for serial arms (URDF collision STLs)."""

    @classmethod
    def from_robot_name(
        cls,
        robot_name: str,
        robots_config_path: Optional[str] = None,
        **kwargs,
    ) -> "SelfCollisionChecker":
        from utils.config_loader import get_robot_by_name

        robot_cfg = get_robot_by_name(robot_name, robots_config_path)
        kwargs.setdefault("fixture_name", robot_cfg.fixture_name)
        return cls(urdf_path=robot_cfg.urdf_path, **kwargs)

    def __init__(
        self,
        urdf_path: str,
        min_joint_gap: int = 1,
        verbose: bool = False,
        fixture_name: Optional[str] = "ee_link",
    ):
        self._verbose = verbose
        self.model, self.geom_model, self._urdf_abs, self._urdf_dir, self._mesh_root, self._n_robot = (
            build_robot_collision_geometry(
                urdf_path,
                fixture_name=fixture_name,
            )
        )
        self._robot_indices = list(range(self._n_robot))

        if verbose:
            logger.info(
                "Loaded %d collision geometries from %s",
                self._n_robot,
                self._urdf_abs,
            )
            for i, go in enumerate(self.geom_model.geometryObjects):
                logger.info("  [%d] %s  (parent joint %d)", i, go.name, go.parentJoint)

        self.geom_model.removeAllCollisionPairs()
        add_robot_self_pairs(self.geom_model, self._robot_indices)
        remove_adjacent_pairs(self.geom_model, min_joint_gap, self._robot_indices)

        self.data: pin.Data = self.model.createData()
        self.geom_data: pin.GeometryData = pin.GeometryData(self.geom_model)

        self._excluded_pairs: List[Tuple[str, str]] = []
        self._is_calibrated = False

        if verbose:
            logger.info(
                "Active collision pairs after adjacency filter: %d",
                len(self.geom_model.collisionPairs),
            )

    def calibrate(
        self,
        n_samples: int = 10,
        seed: int = 42,
    ) -> List[Tuple[str, str]]:
        """Exclude link pairs that collide at *every* test configuration."""
        rng = np.random.RandomState(seed)
        lower = self.model.lowerPositionLimit[: self.model.nq]
        upper = self.model.upperPositionLimit[: self.model.nq]
        configs = [pin.neutral(self.model)]
        for _ in range(n_samples):
            configs.append(lower + rng.rand(self.model.nq) * (upper - lower))

        n_pairs = len(self.geom_model.collisionPairs)
        hit_counts = np.zeros(n_pairs, dtype=int)

        for q in configs:
            q_full = pad_q(self.model, q)
            pin.computeCollisions(
                self.model,
                self.data,
                self.geom_model,
                self.geom_data,
                q_full,
                False,
            )
            for i, cr in enumerate(self.geom_data.collisionResults):
                if cr.isCollision():
                    hit_counts[i] += 1

        n_total = len(configs)
        to_remove = []
        for i in range(n_pairs):
            if hit_counts[i] == n_total:
                cp = self.geom_model.collisionPairs[i]
                to_remove.append(
                    (
                        pin.CollisionPair(cp.first, cp.second),
                        self.geom_model.geometryObjects[cp.first].name,
                        self.geom_model.geometryObjects[cp.second].name,
                    )
                )

        excluded_names: List[Tuple[str, str]] = []
        for cp, n1, n2 in to_remove:
            excluded_names.append((n1, n2))
            self.geom_model.removeCollisionPair(cp)

        self.geom_data = pin.GeometryData(self.geom_model)
        self._excluded_pairs = excluded_names
        self._is_calibrated = True

        if self._verbose:
            logger.info("Calibration excluded %d always-colliding pairs:", len(excluded_names))
            for n1, n2 in excluded_names:
                logger.info("  %s <-> %s", n1, n2)
            logger.info("Remaining active pairs: %d", len(self.geom_model.collisionPairs))

        return excluded_names

    def _apply_pair_margins(self, q: np.ndarray) -> None:
        """Self checker uses zero margin by default (tight geometry)."""
        from .geometry import ensure_collision_requests

        ensure_collision_requests(
            self.model, self.data, self.geom_model, self.geom_data, q
        )
        for req in self.geom_data.collisionRequests:
            req.security_margin = 0.0

    def check(self, q: np.ndarray) -> CollisionResult:
        if not self._is_calibrated:
            logger.warning(
                "check() called before calibrate(). "
                "Results may include structural mesh-overlap false positives."
            )
        q_full = pad_q(self.model, q)
        self._apply_pair_margins(q)

        pin.computeCollisions(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
            q_full,
            False,
        )
        pin.computeDistances(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
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
        q_full = pad_q(self.model, q)
        self._apply_pair_margins(q)
        return pin.computeCollisions(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
            q_full,
            True,
        )

    def has_collision(self, q: np.ndarray) -> bool:
        """Duck-typed alias for :meth:`has_self_collision` (Feature 4 feasibility gate)."""
        return self.has_self_collision(q)

    @property
    def excluded_pairs(self) -> List[Tuple[str, str]]:
        return list(self._excluded_pairs)

    @property
    def active_pair_count(self) -> int:
        return len(self.geom_model.collisionPairs)

    @property
    def n_joints(self) -> int:
        return self.model.nq

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
