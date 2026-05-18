#!/usr/bin/env python3
"""Collision pair construction and kinematic adjacency filtering."""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import pinocchio as pin


def remove_adjacent_pairs(
    geom_model: pin.GeometryModel,
    min_joint_gap: int,
    robot_geom_indices: Iterable[int],
) -> None:
    """Remove robot self-collision pairs whose parent joints differ by ≤ ``min_joint_gap``.

    Only pairs where **both** geometry indices lie in ``robot_geom_indices`` are
    considered (static scene bodies are ignored here).
    """
    robot_set = set(robot_geom_indices)
    to_remove: List[pin.CollisionPair] = []
    for i in range(len(geom_model.collisionPairs)):
        cp = geom_model.collisionPairs[i]
        if cp.first not in robot_set or cp.second not in robot_set:
            continue
        j1 = geom_model.geometryObjects[cp.first].parentJoint
        j2 = geom_model.geometryObjects[cp.second].parentJoint
        if abs(j1 - j2) <= min_joint_gap:
            to_remove.append(pin.CollisionPair(cp.first, cp.second))
    for cp in to_remove:
        geom_model.removeCollisionPair(cp)


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
