#!/usr/bin/env python3
"""Full-scene collision: URDF robot + static STLs, whitelisted midsole–tool contact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pinocchio as pin

from .geometry import (
    append_fixed_scene_geometry,
    build_robot_collision_geometry,
    ensure_collision_requests,
    pad_q,
    se3_from_collision_object_pose,
)
from .mesh_processing import effective_mesh_path
from .pair_rules import (
    add_robot_environment_pairs,
    add_robot_self_pairs,
    calibrate_exclude_frequent_self_pairs,
    index_pair_for_names,
    remove_adjacent_pairs,
)
from .scene_config import CollisionObjectsFile
from .types import CollisionResult

logger = logging.getLogger(__name__)


class SceneCollisionChecker:
    """Robot self-collision + static environment meshes (Pinocchio + coal)."""

    def __init__(
        self,
        model: pin.Model,
        geom_model: pin.GeometryModel,
        data: pin.Data,
        geom_data: pin.GeometryData,
        n_robot_geom: int,
        env_geom_indices: List[int],
        geom_tolerance_m: List[float],
        excluded_calibration_pairs: List[Tuple[str, str]],
        scene_yaml: Optional[str] = None,
        midsole_geom_name: Optional[str] = None,
        knife_blade_geom_name: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.geom_model = geom_model
        self.data = data
        self.geom_data = geom_data
        self.n_robot_geom = n_robot_geom
        self.env_geom_indices = list(env_geom_indices)
        self._geom_tolerance_m = geom_tolerance_m
        self._excluded_calibration_pairs = excluded_calibration_pairs
        self._scene_yaml = scene_yaml
        self._verbose = verbose
        self.midsole_geom_name = midsole_geom_name
        self.knife_blade_geom_name = knife_blade_geom_name
        self._robot_set = set(range(n_robot_geom))
        self._env_set = set(env_geom_indices)
        self._is_calibrated = True

    @classmethod
    def from_urdf_and_scene_yaml(
        cls,
        urdf_path: str,
        scene_yaml_path: str,
        *,
        project_root: Optional[Path] = None,
        min_joint_gap: int = 1,
        calibrate: bool = True,
        calibrate_n_samples: int = 32,
        calibrate_seed: int = 42,
        calibrate_min_hit_fraction: float = 0.95,
        verbose: bool = False,
        fixture_name: Optional[str] = "ee_link",
    ) -> "SceneCollisionChecker":
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        model, geom_model, _urdf_abs, _urdf_dir, _mesh_root, n_robot = (
            build_robot_collision_geometry(
                urdf_path,
                fixture_name=fixture_name,
                project_root=project_root,
            )
        )
        robot_indices = list(range(n_robot))

        geom_model.removeAllCollisionPairs()
        add_robot_self_pairs(geom_model, robot_indices)
        remove_adjacent_pairs(
            geom_model, min_joint_gap, robot_indices, model=model,
        )

        data = model.createData()
        geom_data = pin.GeometryData(geom_model)
        excluded: List[Tuple[str, str]] = []
        if calibrate:
            excluded, geom_data = calibrate_exclude_frequent_self_pairs(
                model,
                geom_model,
                data,
                geom_data,
                robot_indices,
                n_samples=calibrate_n_samples,
                seed=calibrate_seed,
                min_hit_fraction=calibrate_min_hit_fraction,
            )
            if verbose:
                logger.info(
                    "Scene calibration excluded %d frequent self-collision pairs",
                    len(excluded),
                )

        scene_doc = CollisionObjectsFile.load(scene_yaml_path)
        geom_tol = [0.0] * len(geom_model.geometryObjects)

        env_indices: List[int] = []
        for spec in scene_doc.objects:
            if not spec.enabled:
                continue
            placement = se3_from_collision_object_pose(spec.pose)
            mesh_paths: List[Tuple[str, str]] = []
            if spec.convex_mesh_paths:
                for k, mp in enumerate(spec.convex_mesh_paths):
                    abs_p, tag = effective_mesh_path(
                        mp,
                        None,
                        None,
                        None,
                        project_root=project_root,
                    )
                    mesh_paths.append((abs_p, f"{spec.name}__c{k}"))
            else:
                abs_p, _tag = effective_mesh_path(
                    spec.mesh_path,
                    spec.simplified_mesh_path,
                    spec.decimation_ratio,
                    spec.decimation_cache_path,
                    project_root=project_root,
                )
                mesh_paths.append((abs_p, spec.name))

            for abs_mesh, gname in mesh_paths:
                ms_arr = None
                if spec.mesh_scale is not None:
                    ms_arr = np.asarray(spec.mesh_scale, dtype=float).reshape(3)
                idx = append_fixed_scene_geometry(
                    geom_model, gname, abs_mesh, placement, mesh_scale=ms_arr,
                )
                env_indices.append(idx)
                while len(geom_tol) <= idx:
                    geom_tol.append(0.0)
                geom_tol[idx] = float(spec.collision_tolerance_m)

        forbidden: Set[Tuple[int, int]] = set()
        for spec in scene_doc.objects:
            for a, b in spec.whitelist_pairs:
                try:
                    forbidden.add(index_pair_for_names(geom_model, a, b))
                except KeyError:
                    if verbose:
                        logger.warning("Whitelist pair not resolved yet (skip): %s <-> %s", a, b)

        if scene_doc.midsole_geom_name and scene_doc.knife_blade_geom_name:
            try:
                forbidden.add(
                    index_pair_for_names(
                        geom_model,
                        scene_doc.midsole_geom_name,
                        scene_doc.knife_blade_geom_name,
                    )
                )
            except KeyError as exc:
                logger.warning("Default midsole–knife whitelist not applied: %s", exc)

        add_robot_environment_pairs(geom_model, robot_indices, env_indices, forbidden)

        geom_data = pin.GeometryData(geom_model)

        return cls(
            model=model,
            geom_model=geom_model,
            data=data,
            geom_data=geom_data,
            n_robot_geom=n_robot,
            env_geom_indices=env_indices,
            geom_tolerance_m=geom_tol,
            excluded_calibration_pairs=excluded,
            scene_yaml=str(scene_yaml_path),
            midsole_geom_name=scene_doc.midsole_geom_name,
            knife_blade_geom_name=scene_doc.knife_blade_geom_name,
            verbose=verbose,
        )

    def _set_pair_margins(self, q: np.ndarray) -> None:
        ensure_collision_requests(self.model, self.data, self.geom_model, self.geom_data, q)
        for k, cp in enumerate(self.geom_model.collisionPairs):
            ti = self._geom_tolerance_m[cp.first]
            tj = self._geom_tolerance_m[cp.second]
            m = max(float(ti), float(tj))
            self.geom_data.collisionRequests[k].security_margin = float(m)

    def check(self, q: np.ndarray) -> CollisionResult:
        q_full = pad_q(self.model, q)
        self._set_pair_margins(q)
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

        meta = {
            "scene_yaml": self._scene_yaml,
            "n_robot_geom": self.n_robot_geom,
            "n_env_geom": len(self.env_geom_indices),
        }
        return CollisionResult(
            has_collision=len(colliding) > 0,
            colliding_pairs=colliding,
            min_distance_m=min_dist if min_dist != float("inf") else -1.0,
            closest_pair=closest,
            all_distances=sorted(all_dist, key=lambda t: t[2]),
            meta=meta,
        )

    def has_collision(self, q: np.ndarray) -> bool:
        q_full = pad_q(self.model, q)
        self._set_pair_margins(q)
        return pin.computeCollisions(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
            q_full,
            True,
        )

    def has_self_collision(self, q: np.ndarray) -> bool:
        """True only if a **robot self** pair is in collision (ignores env pairs)."""
        res = self.check(q)
        for (a, b) in res.colliding_pairs:
            ia = self._geom_index(a)
            ib = self._geom_index(b)
            if ia in self._robot_set and ib in self._robot_set:
                return True
        return False

    def has_environment_collision(self, q: np.ndarray) -> bool:
        """True if any **robot–environment** pair collides."""
        res = self.check(q)
        for (a, b) in res.colliding_pairs:
            ia = self._geom_index(a)
            ib = self._geom_index(b)
            if (ia in self._robot_set and ib in self._env_set) or (
                ib in self._robot_set and ia in self._env_set
            ):
                return True
        return False

    def _geom_index(self, name: str) -> int:
        for k, go in enumerate(self.geom_model.geometryObjects):
            if go.name == name:
                return k
        raise KeyError(name)
