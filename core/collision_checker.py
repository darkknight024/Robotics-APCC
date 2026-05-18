#!/usr/bin/env python3
"""Backward-compatible shim — implementation lives in :mod:`core.collision`."""

from .collision import CollisionResult, SelfCollisionChecker

__all__ = ["SelfCollisionChecker", "CollisionResult"]
