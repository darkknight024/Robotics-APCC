#!/usr/bin/env python3
"""
Solver-agnostic self-collision checker using Pinocchio + Coal (hpp-fcl).

Loads the robot URDF with its collision STL meshes and checks whether a
given joint configuration causes non-adjacent links to penetrate each other.

The checker is intentionally decoupled from any particular IK solver --
it accepts a raw joint-angle vector (radians) and returns whether
self-collision exists.  This allows it to work identically with EAIK,
Pinocchio IK, or any future solver backend.

Key features
~~~~~~~~~~~~
*  **Security margin** (``security_margin_m``):  A negative value shrinks
   the effective collision boundary inward, requiring deeper mesh
   penetration before reporting a collision.  This compensates for the
   fact that the URDF visual STL meshes overlap at joint boundaries
   (SolidWorks export artifact).  A value of ``-0.005`` (5 mm) is a
   reasonable starting point; ``0.0`` gives strict binary collision.

*  **Calibration** (:class:`CollisionCalibrator`):  Probes many random
   configurations and automatically excludes link pairs whose meshes
   *always* overlap -- structural artifacts, not real collisions.

Typical usage::

    checker = SelfCollisionChecker.from_robot_name("IRB 1300-7/1.4")
    checker.calibrate()

    if checker.has_self_collision(q):
        print("Self-collision at this configuration!")
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path

import pinocchio as pin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CollisionResult:
    """Detailed result of a self-collision query."""
    has_collision: bool
    colliding_pairs: List[Tuple[str, str]]
    min_distance_m: float
    closest_pair: Tuple[str, str]
    all_distances: List[Tuple[str, str, float]]


@dataclass
class CalibrationReport:
    """Detailed output from a calibration run."""
    n_samples: int
    n_pairs_before: int
    n_pairs_after: int
    excluded_pairs: List[Tuple[str, str]]
    hit_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calibrator (decoupled from checker)
# ---------------------------------------------------------------------------

class CollisionCalibrator:
    """Probes a collision model with many configurations to identify
    structurally-always-colliding pairs.

    Separated from :class:`SelfCollisionChecker` so it can be reused with
    different sample counts, seeds, thresholds, or custom config sets.
    """

    def __init__(
        self,
        model: pin.Model,
        geom_model: pin.GeometryModel,
    ):
        self._model = model
        self._geom_model = geom_model

    def run(
        self,
        n_samples: int = 5000,
        seed: int = 42,
        threshold: float = 1.0,
        extra_configs: Optional[List[np.ndarray]] = None,
    ) -> CalibrationReport:
        """Probe random configurations and return a :class:`CalibrationReport`.

        Args:
            n_samples: Number of uniform-random configs to test (plus neutral).
            seed: RNG seed.
            threshold: Fraction in ``(0, 1]``.  A pair is marked for exclusion
                       if it collides in >= ``threshold * total_configs``.
                       Default ``1.0`` means *every* config must collide.
                       Lower values (e.g. 0.95) catch near-always pairs.
            extra_configs: Additional configurations to include in the probe
                           set (e.g. known RS-valid configs).

        Returns:
            :class:`CalibrationReport` with pair-level hit rates and the
            list of pairs to exclude.
        """
        rng = np.random.RandomState(seed)
        lower = self._model.lowerPositionLimit[: self._model.nq]
        upper = self._model.upperPositionLimit[: self._model.nq]

        configs = [pin.neutral(self._model)]
        for _ in range(n_samples):
            configs.append(lower + rng.rand(self._model.nq) * (upper - lower))
        if extra_configs:
            configs.extend(extra_configs)

        n_pairs = len(self._geom_model.collisionPairs)
        data = self._model.createData()
        geom_data = pin.GeometryData(self._geom_model)

        hit_counts = np.zeros(n_pairs, dtype=int)
        n_total = len(configs)

        for q in configs:
            q = np.asarray(q, dtype=float).flatten()
            if len(q) < self._model.nq:
                q_full = np.zeros(self._model.nq)
                q_full[: len(q)] = q
                q = q_full
            pin.computeCollisions(
                self._model, data,
                self._geom_model, geom_data,
                q, False,
            )
            for i, cr in enumerate(geom_data.collisionResults):
                if cr.isCollision():
                    hit_counts[i] += 1

        pair_names = []
        for i in range(n_pairs):
            cp = self._geom_model.collisionPairs[i]
            n1 = self._geom_model.geometryObjects[cp.first].name
            n2 = self._geom_model.geometryObjects[cp.second].name
            pair_names.append((n1, n2))

        hit_rates = {}
        excluded = []
        cutoff = int(np.ceil(threshold * n_total))
        for i in range(n_pairs):
            rate = hit_counts[i] / n_total
            hit_rates[pair_names[i]] = rate
            if hit_counts[i] >= cutoff:
                excluded.append(pair_names[i])

        return CalibrationReport(
            n_samples=n_total,
            n_pairs_before=n_pairs,
            n_pairs_after=n_pairs - len(excluded),
            excluded_pairs=excluded,
            hit_rates=hit_rates,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SelfCollisionChecker:
    """Solver-agnostic self-collision checker for 6-DOF robot arms.

    Loads the URDF collision geometry (STL meshes) via Pinocchio + Coal
    and checks whether a given joint configuration *q* (radians) causes any
    non-adjacent links to collide.

    The URDF ``<collision>`` tags reference the same STL files used for
    visualization.  These detailed meshes often overlap at joint boundaries
    (a structural artifact of SolidWorks export).  :meth:`calibrate` tests
    several configurations and automatically excludes pairs that always
    collide -- treating them as mesh artifacts rather than real collisions.

    A **negative security margin** can be applied to all collision requests
    to further reduce false positives from mesh overlap.
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
                      (*min_joint_gap*, *verbose*, *security_margin_m*).
        """
        from utils.config_loader import get_robot_by_name

        robot_cfg = get_robot_by_name(robot_name, robots_config_path)
        return cls(urdf_path=robot_cfg.urdf_path, **kwargs)

    # --------------------------------------------------------------------- #
    #  Init
    # --------------------------------------------------------------------- #

    # Default pairs to exclude -- identified via diagnostic analysis as
    # structural false positives caused by the oversized Base_link fixture
    # mesh extending into the wrist workspace.  These pairs never represent
    # real collisions within the robot's joint limits.
    DEFAULT_EXCLUDE_PAIRS: List[Tuple[str, str]] = [
        ("Base_link_0", "Link_4_0"),
        ("Base_link_0", "Link_5_0"),
        ("Base_link_0", "Link_6_0"),
    ]

    def __init__(
        self,
        urdf_path: str,
        min_joint_gap: int = 1,
        verbose: bool = False,
        security_margin_m: float = 0.0,
        exclude_pairs: Optional[List[Tuple[str, str]]] = None,
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
            security_margin_m: Safety margin applied to every collision
                               request.  **Negative** values shrink the
                               effective collision boundary (e.g. ``-0.005``
                               requires 5 mm penetration before flagging).
                               Default ``0.0`` gives strict binary collision.
            exclude_pairs: Explicit list of ``(geom_name_1, geom_name_2)``
                           pairs to remove from collision checking.  Pass
                           ``"default"`` or ``None`` to use
                           :attr:`DEFAULT_EXCLUDE_PAIRS`.  Pass an empty
                           list ``[]`` to disable pair exclusion entirely.
        """
        self._verbose = verbose
        self._security_margin_m = security_margin_m

        urdf_abs = self._resolve_path(urdf_path)
        self._urdf_abs = urdf_abs
        urdf_dir = os.path.dirname(urdf_abs)
        mesh_root = os.path.dirname(urdf_dir)

        self.model: pin.Model = pin.buildModelFromUrdf(urdf_abs)

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

        self.geom_model.addAllCollisionPairs()
        self._remove_adjacent_pairs(min_joint_gap)

        if exclude_pairs is None:
            exclude_pairs = self.DEFAULT_EXCLUDE_PAIRS
        self._remove_named_pairs(exclude_pairs)

        self.data: pin.Data = self.model.createData()
        self.geom_data: pin.GeometryData = pin.GeometryData(self.geom_model)
        self._apply_security_margin()

        self._excluded_pairs: List[Tuple[str, str]] = []
        self._is_calibrated = False

        if verbose:
            logger.info(
                "Active collision pairs after adjacency + exclusion filter: %d",
                len(self.geom_model.collisionPairs),
            )

    # --------------------------------------------------------------------- #
    #  Security margin
    # --------------------------------------------------------------------- #

    def _apply_security_margin(self) -> None:
        """Set the security margin on every active collision request."""
        if self._security_margin_m == 0.0:
            return
        for i in range(len(self.geom_model.collisionPairs)):
            self.geom_data.collisionRequests[i].security_margin = (
                self._security_margin_m
            )

    @property
    def security_margin_m(self) -> float:
        return self._security_margin_m

    @security_margin_m.setter
    def security_margin_m(self, value: float) -> None:
        """Update the security margin on the fly."""
        self._security_margin_m = value
        for i in range(len(self.geom_model.collisionPairs)):
            self.geom_data.collisionRequests[i].security_margin = value

    # --------------------------------------------------------------------- #
    #  Calibration — exclude structural mesh overlaps
    # --------------------------------------------------------------------- #

    def calibrate(
        self,
        n_samples: int = 5000,
        seed: int = 42,
        threshold: float = 1.0,
        extra_configs: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[str, str]]:
        """Exclude link pairs that collide at (nearly) every test config.

        Delegates to :class:`CollisionCalibrator` and applies the result
        by removing offending pairs from ``geom_model``.

        Args:
            n_samples: Random configs to probe (plus neutral).
            seed: RNG seed.
            threshold: Fraction in ``(0, 1]``.  Pairs that collide in
                       ``>= threshold * total`` configs are excluded.
            extra_configs: Additional configs to include in the probe set.

        Returns:
            Names of excluded pairs.
        """
        cal = CollisionCalibrator(self.model, self.geom_model)
        report = cal.run(
            n_samples=n_samples,
            seed=seed,
            threshold=threshold,
            extra_configs=extra_configs,
        )

        for n1, n2 in report.excluded_pairs:
            g1 = g2 = None
            for idx, go in enumerate(self.geom_model.geometryObjects):
                if go.name == n1:
                    g1 = idx
                elif go.name == n2:
                    g2 = idx
            if g1 is not None and g2 is not None:
                self.geom_model.removeCollisionPair(
                    pin.CollisionPair(g1, g2)
                )

        self.geom_data = pin.GeometryData(self.geom_model)
        self._apply_security_margin()
        self._excluded_pairs = report.excluded_pairs
        self._is_calibrated = True
        self._last_calibration_report = report

        if self._verbose:
            logger.info(
                "Calibration excluded %d pairs (threshold=%.2f, n=%d):",
                len(report.excluded_pairs), threshold, report.n_samples,
            )
            for n1, n2 in report.excluded_pairs:
                logger.info("  %s <-> %s", n1, n2)
            logger.info(
                "Remaining active pairs: %d",
                len(self.geom_model.collisionPairs),
            )

        return report.excluded_pairs

    @property
    def last_calibration_report(self) -> Optional[CalibrationReport]:
        return getattr(self, "_last_calibration_report", None)

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

        Respects the current ``security_margin_m`` setting.

        Args:
            q: Joint angles in radians, shape ``(n_joints,)``.

        Returns:
            ``True`` if any (non-excluded) link pair collides.
        """
        q_full = self._pad(q)
        return pin.computeCollisions(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q_full, True,
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

    @property
    def urdf_path(self) -> str:
        return self._urdf_abs

    # --------------------------------------------------------------------- #
    #  Debug report generation
    # --------------------------------------------------------------------- #

    def generate_debug_report(
        self,
        fp_csv_path: str,
        full_rs_csv_path: str,
        out_dir: str,
        mesh_dir: Optional[str] = None,
        max_mesh_exports: int = 5,
    ) -> dict:
        """Generate a comprehensive collision-checker debug report.

        Produces pair-level diagnosis, security-margin sweep, mesh
        bounding-box plots, before/after comparison, and positioned
        mesh exports.  All visualization code lives in
        :mod:`utils.collision_debug`.

        Args:
            fp_csv_path: CSV with known false-positive joint configs
                         (columns: waypoint_index, is_reachable, j_1..j_6).
            full_rs_csv_path: Full RobotStudio results CSV (same columns,
                              used to validate no false flags on reachable).
            out_dir: Output directory for all artifacts.
            mesh_dir: Directory containing original STL meshes.
            max_mesh_exports: Max configs to export as positioned STL.

        Returns:
            Dict mapping artifact name to file path.
        """
        from utils.collision_debug import generate_collision_debug_report

        return generate_collision_debug_report(
            checker=self,
            fp_csv_path=fp_csv_path,
            full_rs_csv_path=full_rs_csv_path,
            out_dir=out_dir,
            mesh_dir=mesh_dir,
            max_mesh_exports=max_mesh_exports,
        )

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

    def _remove_named_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        """Remove specific named geometry pairs from collision checking."""
        name_to_idx = {
            go.name: idx
            for idx, go in enumerate(self.geom_model.geometryObjects)
        }
        for n1, n2 in pairs:
            g1 = name_to_idx.get(n1)
            g2 = name_to_idx.get(n2)
            if g1 is not None and g2 is not None:
                try:
                    self.geom_model.removeCollisionPair(
                        pin.CollisionPair(g1, g2)
                    )
                except Exception:
                    pass

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
        """Resolve a URDF path to an absolute path on disk."""
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
