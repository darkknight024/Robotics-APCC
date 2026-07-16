"""Experiment 24 per-joint dynamics calibration.

All public values are stored in SI units:

* joint velocity limits: rad/s
* joint acceleration/deceleration limits: rad/s²
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class JointDynamicsCalibration:
    """Measured per-joint motion limits for a specific robot configuration."""

    q_dot_max: np.ndarray
    q_ddot_accel: np.ndarray
    q_ddot_decel: np.ndarray
    configuration: str
    source: str

    def __post_init__(self) -> None:
        for name in ("q_dot_max", "q_ddot_accel", "q_ddot_decel"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (6,):
                raise ValueError(f"{name} must have shape (6,), got {value.shape}")
            object.__setattr__(self, name, value)


def get_exp24_neutral() -> JointDynamicsCalibration:
    """Return the hardcoded Experiment 24 neutral-position limits."""

    # Experiment 24 summary values in degrees (accel/decel from Exp24 v8 RS):
    # q_dot_max_deg_s:       [280.0, 180.0, 250.0, 500.0, 415.8, 720.0]
    # q_ddot_accel_deg_s2:   [11102, 21533, 33677, 144, 10037, 11259]
    # q_ddot_decel_deg_s2:   [7275,  22498, 30712, 246, 11370, 7083]
    return JointDynamicsCalibration(
        q_dot_max=np.deg2rad([280.0, 180.0, 250.0, 500.0, 415.8, 720.0]),
        q_ddot_accel=np.deg2rad([11102.0, 21533.0, 33677.0, 144.0, 10037.0, 11259.0]),
        q_ddot_decel=np.deg2rad([7275.0, 22498.0, 30712.0, 246.0, 11370.0, 7083.0]),
        configuration="neutral",
        source="Experiment_24_v8",
    )


def _first_joint_dynamics_block(raw: Mapping[str, Any], robot_name: Optional[str]) -> Mapping[str, Any]:
    """Find ``calibration.joint_dynamics`` in robots_config.yaml."""

    top_cal = raw.get("calibration", {}) or {}
    if isinstance(top_cal, Mapping) and "joint_dynamics" in top_cal:
        return top_cal["joint_dynamics"] or {}

    robots = raw.get("robots", []) or []
    if robot_name:
        for robot in robots:
            if isinstance(robot, Mapping) and robot.get("name") == robot_name:
                cal = robot.get("calibration", {}) or {}
                return cal.get("joint_dynamics", {}) or {}

    for robot in robots:
        if not isinstance(robot, Mapping):
            continue
        cal = robot.get("calibration", {}) or {}
        jd = cal.get("joint_dynamics")
        if jd:
            return jd

    return {}


def load_joint_dynamics(
    config_path: str | Path,
    robot_name: Optional[str] = "IRB 1300-7/1.4",
) -> JointDynamicsCalibration:
    """Load joint dynamics from ``robots_config.yaml``.

    If no ``calibration.joint_dynamics`` block exists, the Exp 24 neutral
    calibration is returned so existing configs remain runnable.
    """

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    block = _first_joint_dynamics_block(raw, robot_name)
    if not block:
        return get_exp24_neutral()

    q_dot_deg = block.get("q_dot_max_deg_s")
    q_acc_deg = block.get("q_ddot_accel_deg_s2")
    q_dec_deg = block.get("q_ddot_decel_deg_s2")
    if q_dot_deg is None or q_acc_deg is None or q_dec_deg is None:
        raise ValueError(
            "joint_dynamics requires q_dot_max_deg_s, "
            "q_ddot_accel_deg_s2, and q_ddot_decel_deg_s2"
        )

    return JointDynamicsCalibration(
        q_dot_max=np.deg2rad(np.asarray(q_dot_deg, dtype=float)),
        q_ddot_accel=np.deg2rad(np.asarray(q_acc_deg, dtype=float)),
        q_ddot_decel=np.deg2rad(np.asarray(q_dec_deg, dtype=float)),
        configuration=str(block.get("configuration", "unknown")),
        source=str(block.get("source", "robots_config.yaml")),
    )
