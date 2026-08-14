#!/usr/bin/env python3
"""Collision pair construction and kinematic adjacency filtering."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pinocchio as pin

from .geometry import pad_q


def kinematic_joint_distance(model: pin.Model, j1: int, j2: int) -> int:
    """Hops along the kinematic tree between two joint indices (0 if same)."""
    j1 = int(j1)
    j2 = int(j2)
    if j1 == j2:
        return 0

    def path_to_root(j: int) -> List[int]:
        path = [j]
        while j != 0:
            j = int(model.parents[j])
            path.append(j)
        return path

    a = path_to_root(j1)
    b = path_to_root(j2)
    ia = {j: k for k, j in enumerate(a)}
    for dist_b, jb in enumerate(b):
        if jb in ia:
            return ia[jb] + dist_b
    return abs(j1 - j2)


def remove_adjacent_pairs(
    geom_model: pin.GeometryModel,
    min_joint_gap: int,
    robot_geom_indices: Iterable[int],
    model: Optional[pin.Model] = None,
) -> int:
    """Drop robot self-pairs that are kinematically successive.

    * Same parent joint (e.g. flange mesh + fixture on ``Link_6``) → distance 0.
    * Parent/child joints in the URDF tree → distance 1.
    * ``min_joint_gap`` is the maximum tree distance to exclude (default 1 =
      successive links). Index ``|j1-j2|`` is used only if ``model`` is omitted.

    Environment pairs are never removed here. Returns how many pairs were dropped.
    """
    robot_set = set(robot_geom_indices)
    to_remove: List[pin.CollisionPair] = []
    for i in range(len(geom_model.collisionPairs)):
        cp = geom_model.collisionPairs[i]
        if cp.first not in robot_set or cp.second not in robot_set:
            continue
        j1 = int(geom_model.geometryObjects[cp.first].parentJoint)
        j2 = int(geom_model.geometryObjects[cp.second].parentJoint)
        if model is not None:
            dist = kinematic_joint_distance(model, j1, j2)
        else:
            dist = abs(j1 - j2)
        if dist <= min_joint_gap:
            to_remove.append(pin.CollisionPair(cp.first, cp.second))
    for cp in to_remove:
        geom_model.removeCollisionPair(cp)
    return len(to_remove)


def calibrate_exclude_frequent_self_pairs(
    model: pin.Model,
    geom_model: pin.GeometryModel,
    data: pin.Data,
    geom_data: pin.GeometryData,
    robot_geom_indices: Iterable[int],
    *,
    n_samples: int = 32,
    seed: int = 42,
    min_hit_fraction: float = 0.95,
) -> Tuple[List[Tuple[str, str]], pin.GeometryData]:
    """Drop robot–robot pairs that collide in most sampled configurations.

    Samples include neutral, joint-range midpoint, and uniform random poses in
    URDF limits. A pair is excluded when
    ``hits / n_configs >= min_hit_fraction``. That catches overlapping visual
    STLs on neighbouring links without requiring a hit on *every* random fold.

    Environment pairs are never excluded (they are typically added after this
    step). Returns ``(excluded_name_pairs, rebuilt GeometryData)``.
    """
    robot_set = set(robot_geom_indices)
    rng = np.random.RandomState(seed)
    nq = model.nq
    lower = np.asarray(model.lowerPositionLimit[:nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[:nq], dtype=float)
    finite = np.isfinite(lower) & np.isfinite(upper)
    lower = np.where(finite, lower, -np.pi)
    upper = np.where(finite, upper, np.pi)
    mid = 0.5 * (lower + upper)

    configs = [pin.neutral(model), mid.copy()]
    n_rand = max(int(n_samples), 0)
    for _ in range(n_rand):
        configs.append(lower + rng.rand(nq) * (upper - lower))

    n_pairs = len(geom_model.collisionPairs)
    if n_pairs == 0:
        return [], geom_data

    hit_counts = np.zeros(n_pairs, dtype=int)
    for q in configs:
        q_full = pad_q(model, q)
        pin.computeCollisions(model, data, geom_model, geom_data, q_full, False)
        for i, cr in enumerate(geom_data.collisionResults):
            if cr.isCollision():
                hit_counts[i] += 1

    n_total = len(configs)
    threshold = min_hit_fraction * n_total
    to_remove: List[Tuple[pin.CollisionPair, str, str]] = []
    for i in range(n_pairs):
        cp = geom_model.collisionPairs[i]
        if cp.first not in robot_set or cp.second not in robot_set:
            continue
        if hit_counts[i] + 1e-9 >= threshold:
            to_remove.append(
                (
                    pin.CollisionPair(cp.first, cp.second),
                    geom_model.geometryObjects[cp.first].name,
                    geom_model.geometryObjects[cp.second].name,
                )
            )

    excluded_names: List[Tuple[str, str]] = []
    for cp, n1, n2 in to_remove:
        excluded_names.append((n1, n2))
        geom_model.removeCollisionPair(cp)

    return excluded_names, pin.GeometryData(geom_model)


def add_robot_environment_pairs(
    geom_model: pin.GeometryModel,
    robot_indices: Iterable[int],
    env_indices: Iterable[int],
    forbidden_pairs: Set[Tuple[int, int]],
) -> None:
    """Register collision pairs between each robot geometry and each environment body."""
    robots = list(robot_indices)
    envs = list(env_indices)
    for ri in robots:
        for ej in envs:
            a, b = pair_key(ri, ej)
            if (a, b) in forbidden_pairs:
                continue
            cp = pin.CollisionPair(ri, ej)
            if not geom_model.existCollisionPair(cp):
                geom_model.addCollisionPair(cp)


def add_robot_self_pairs(
    geom_model: pin.GeometryModel,
    robot_indices: Iterable[int],
) -> None:
    """Add all unique unordered self pairs among ``robot_indices``."""
    rlist = sorted(set(robot_indices))
    for i in range(len(rlist)):
        for j in range(i + 1, len(rlist)):
            a, b = rlist[i], rlist[j]
            cp = pin.CollisionPair(a, b)
            if not geom_model.existCollisionPair(cp):
                geom_model.addCollisionPair(cp)


def pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def index_pair_for_names(
    geom_model: pin.GeometryModel,
    name_a: str,
    name_b: str,
) -> Tuple[int, int]:
    """Resolve geometry object names to a sorted index pair."""
    idx = {go.name: k for k, go in enumerate(geom_model.geometryObjects)}
    if name_a not in idx:
        raise KeyError(f"Unknown collision geometry name: {name_a}")
    if name_b not in idx:
        raise KeyError(f"Unknown collision geometry name: {name_b}")
    ia, ib = idx[name_a], idx[name_b]
    return pair_key(ia, ib)
