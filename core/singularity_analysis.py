#!/usr/bin/env python3
"""
Singularity Analysis Module — Type Classification
===================================================

Robust singularity analysis for 6-DOF robots with spherical wrist
(e.g., ABB IRB 1300 family).

For a 6R spherical-wrist manipulator, three distinct singularity types
exist:

* **Wrist singularity** — joints 4 and 6 axes align when joint 5
  approaches 0 or π.  Detected via the determinant of the 3×3
  orientation sub-Jacobian for the wrist joints (J[0:3, 3:6]) and
  the angular distance of q₅ from 0 / π.

* **Shoulder singularity** — the wrist center lies on (or near) the
  joint-1 axis.  Detected via the minimum singular value of the
  linear-velocity sub-Jacobian for the arm joints (J[3:6, 0:3]).

* **Elbow singularity** — the arm is fully extended or fully folded.
  Detected via rank deficiency of the positional contributions of
  joints 2 and 3 (collinearity of J[3:6, 1] and J[3:6, 2]).

The Jacobian convention used in this codebase is
``[angular_vel (3); linear_vel (3)]`` × ``n_joints``.

Provides
--------
- SingularityType enum
- SingularityReport dataclass
- SingularityAnalyzer class (main entry point)
"""

import csv
import numpy as np
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# ============================================================================
# Enums
# ============================================================================

class SingularityType(Enum):
    """Classification of singularity type for 6R spherical-wrist robots."""
    NONE = "none"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    SHOULDER_ELBOW = "shoulder+elbow"
    SHOULDER_WRIST = "shoulder+wrist"
    ELBOW_WRIST = "elbow+wrist"
    SHOULDER_ELBOW_WRIST = "shoulder+elbow+wrist"


# ============================================================================
# Data class
# ============================================================================

_EMPTY_FLOAT_ARRAY = np.array([], dtype=np.float64)


@dataclass
class SingularityReport:
    """Per-waypoint singularity analysis result."""
    singularity_type: SingularityType
    is_singular: bool
    is_reachable: bool = True
    active_types: List[str] = field(default_factory=list)

    overall_sigma_min: float = 0.0
    overall_condition_number: float = np.inf
    overall_manipulability: float = 0.0
    singular_values: np.ndarray = field(default_factory=lambda: _EMPTY_FLOAT_ARRAY.copy())

    wrist_metrics: Dict[str, float] = field(default_factory=dict)
    shoulder_metrics: Dict[str, float] = field(default_factory=dict)
    elbow_metrics: Dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten the report into a single-level dict suitable for CSV rows."""
        d: Dict[str, Any] = {
            "singularity_type": self.singularity_type.value if self.is_reachable else "unreachable",
            "is_singular": "unreachable" if not self.is_reachable else self.is_singular,
            "overall_sigma_min": self.overall_sigma_min,
            "overall_condition_number": self.overall_condition_number,
            "overall_manipulability": self.overall_manipulability,
        }
        for i, sv in enumerate(self.singular_values):
            d[f"sv_{i}"] = sv
        for k, v in self.wrist_metrics.items():
            d[f"wrist_{k}"] = v
        for k, v in self.shoulder_metrics.items():
            d[f"shoulder_{k}"] = v
        for k, v in self.elbow_metrics.items():
            d[f"elbow_{k}"] = v
        return d


_DEFAULT_TYPE_THRESHOLDS = {
    "wrist": 0.1,
    "shoulder": 0.1,
    "elbow": 0.1,
}


# ============================================================================
# Compound type helper
# ============================================================================

_COMPOUND_MAP = {
    frozenset():                                        SingularityType.NONE,
    frozenset(["shoulder"]):                            SingularityType.SHOULDER,
    frozenset(["elbow"]):                               SingularityType.ELBOW,
    frozenset(["wrist"]):                               SingularityType.WRIST,
    frozenset(["shoulder", "elbow"]):                   SingularityType.SHOULDER_ELBOW,
    frozenset(["shoulder", "wrist"]):                   SingularityType.SHOULDER_WRIST,
    frozenset(["elbow", "wrist"]):                      SingularityType.ELBOW_WRIST,
    frozenset(["shoulder", "elbow", "wrist"]):          SingularityType.SHOULDER_ELBOW_WRIST,
}


# ============================================================================
# Analyzer
# ============================================================================

class SingularityAnalyzer:
    """
    Classifies singularity type for 6-DOF spherical-wrist robots
    from the Jacobian matrix and joint positions.

    Compatible with both EAIK and Pinocchio FK backends — operates
    solely on the 6×6 Jacobian (convention ``[angular; linear]``) and
    the joint-angle vector.

    Example::

        sa = SingularityAnalyzer()
        report = sa.analyze(jacobian, q)
        print(report.singularity_type, report.overall_sigma_min)
    """

    def __init__(
        self,
        n_joints: int = 6,
        type_thresholds: Optional[Dict[str, float]] = None,
        check_j5_only: bool = True,
        j5_threshold_deg: float = 0.76,
    ):
        self.n_joints = n_joints
        self.type_thresholds = dict(_DEFAULT_TYPE_THRESHOLDS)
        if type_thresholds:
            self.type_thresholds.update(type_thresholds)
        self.check_j5_only = check_j5_only
        self.j5_threshold_deg = j5_threshold_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        jacobian: np.ndarray,
        joint_positions: np.ndarray,
        fk_solver=None,
    ) -> SingularityReport:
        """
        Analyze a single configuration for singularity type.

        Args:
            jacobian: 6×n Jacobian ([angular(3); linear(3)] convention).
            joint_positions: Joint-angle vector (n,).
            fk_solver: Optional FK solver for wrist-center computation
                       (used to improve shoulder-singularity detection).

        Returns:
            SingularityReport with classification and metrics.
        """
        J = np.asarray(jacobian, dtype=np.float64)
        q = np.asarray(joint_positions, dtype=np.float64)

        # Full-Jacobian singular values (descending order)
        try:
            sv = np.linalg.svd(J, compute_uv=False)
        except np.linalg.LinAlgError:
            sv = np.zeros(min(J.shape))
        sv_sorted = np.sort(sv)[::-1]
        sigma_min = float(sv_sorted[-1]) if len(sv_sorted) > 0 else 0.0
        sigma_max = float(sv_sorted[0]) if len(sv_sorted) > 0 else 0.0
        cond = sigma_max / sigma_min if sigma_min > 1e-15 else np.inf
        manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

        # Per-type classification
        wrist_active, wrist_metrics = self._classify_wrist(
            J, q, check_j5_only=self.check_j5_only,
        )
        shoulder_active, shoulder_metrics = self._classify_shoulder(J, q, fk_solver)
        elbow_active, elbow_metrics = self._classify_elbow(J, q)

        # Combine active types
        active: List[str] = []
        if shoulder_active:
            active.append("shoulder")
        if elbow_active:
            active.append("elbow")
        if wrist_active:
            active.append("wrist")

        stype = _COMPOUND_MAP.get(frozenset(active), SingularityType.NONE)
        is_singular = len(active) > 0

        return SingularityReport(
            singularity_type=stype,
            is_singular=is_singular,
            active_types=active,
            overall_sigma_min=sigma_min,
            overall_condition_number=cond,
            overall_manipulability=manip,
            singular_values=sv_sorted,
            wrist_metrics=wrist_metrics,
            shoulder_metrics=shoulder_metrics,
            elbow_metrics=elbow_metrics,
        )

    def analyze_trajectory(
        self,
        jacobians: List[np.ndarray],
        joint_positions_list: List[np.ndarray],
        fk_solver=None,
    ) -> List[SingularityReport]:
        """Batch convenience: analyze every waypoint in a trajectory."""
        return [
            self.analyze(J, q, fk_solver)
            for J, q in zip(jacobians, joint_positions_list)
        ]

    def summarize_trajectory(
        self,
        reports: List[SingularityReport],
    ) -> Dict[str, Any]:
        """Aggregate singularity statistics over a trajectory."""
        n = len(reports)
        if n == 0:
            return {"num_waypoints": 0}

        type_counts: Dict[str, int] = {}
        singular_count = 0

        for r in reports:
            type_counts[r.singularity_type.value] = type_counts.get(r.singularity_type.value, 0) + 1
            if r.is_singular:
                singular_count += 1

        sigma_mins = [r.overall_sigma_min for r in reports]
        cond_numbers = [r.overall_condition_number for r in reports
                        if np.isfinite(r.overall_condition_number)]

        wrist_count = sum(1 for r in reports if "wrist" in r.active_types)
        shoulder_count = sum(1 for r in reports if "shoulder" in r.active_types)
        elbow_count = sum(1 for r in reports if "elbow" in r.active_types)

        return {
            "num_waypoints": n,
            "singular_count": singular_count,
            "singular_percent": 100.0 * singular_count / n,
            "type_counts": type_counts,
            "wrist_singular_count": wrist_count,
            "shoulder_singular_count": shoulder_count,
            "elbow_singular_count": elbow_count,
            "mean_sigma_min": float(np.mean(sigma_mins)),
            "min_sigma_min": float(np.min(sigma_mins)),
            "max_sigma_min": float(np.max(sigma_mins)),
            "mean_condition_number": float(np.mean(cond_numbers)) if cond_numbers else np.inf,
            "max_condition_number": float(np.max(cond_numbers)) if cond_numbers else np.inf,
        }

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    @staticmethod
    def export_csv(
        reports: List[SingularityReport],
        output_path: str,
    ) -> None:
        """Write per-waypoint singularity data to a CSV file."""
        if not reports:
            return
        rows = []
        for idx, r in enumerate(reports):
            row = {"waypoint_index": idx}
            row.update(r.to_flat_dict())
            rows.append(row)

        # Collect union of all keys so schema works when first waypoint is unreachable
        all_keys: set = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = ["waypoint_index"] + sorted(k for k in all_keys if k != "waypoint_index")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Private: per-type classifiers
    # ------------------------------------------------------------------

    def _classify_wrist(
        self,
        J: np.ndarray,
        q: np.ndarray,
        check_j5_only: bool = False,
    ) -> tuple:
        """
        Detect wrist singularity.

        Two modes:

        * ``check_j5_only=True`` — fast geometric check.
          Joint 5 (q[4]) is compared against a configurable dead-band
          (default ±0.76°, matching ABB RobotWare's empirical boundary).
          If ``|q[4]|`` < ``j5_threshold_deg`` the configuration is
          flagged as wrist-singular.

        * ``check_j5_only=False`` (default) — SVD of the 3×3 wrist
          orientation sub-Jacobian ``J[0:3, 3:6]``.  Flagged when
          σ_min of that sub-Jacobian drops below
          ``type_thresholds['wrist']``.

        In both modes the full set of diagnostic metrics is always
        computed and returned.
        """
        _J5_SINGULARITY_THRESHOLD_RAD = np.radians(self.j5_threshold_deg)

        metrics: Dict[str, float] = {}

        # Sub-Jacobian for wrist orientation (angular rows, wrist columns)
        J_wrist_orient = J[0:3, 3:6]
        try:
            det_w = float(np.linalg.det(J_wrist_orient))
        except np.linalg.LinAlgError:
            det_w = 0.0
        sv_w = np.linalg.svd(J_wrist_orient, compute_uv=False)
        sigma_min_w = float(np.min(sv_w))

        # q5 proximity to 0 or π  (works correctly for negative angles)
        q5 = float(q[4]) if len(q) > 4 else 0.0
        dist_to_singularity = abs(np.sin(q5))

        metrics["det_wrist_jacobian"] = det_w
        metrics["sigma_min"] = sigma_min_w
        metrics["j5_angle_rad"] = q5
        metrics["j5_angle_deg"] = np.degrees(q5)
        metrics["j5_distance_to_singularity_rad"] = dist_to_singularity

        if check_j5_only:
            is_active = dist_to_singularity < np.sin(_J5_SINGULARITY_THRESHOLD_RAD)
        else:
            threshold = self.type_thresholds.get("wrist", 0.01)
            is_active = sigma_min_w < threshold

        return is_active, metrics

    def _classify_shoulder(
        self,
        J: np.ndarray,
        q: np.ndarray,
        fk_solver=None,
    ) -> tuple:
        """
        Detect shoulder singularity.

        The wrist center approaching the joint-1 axis causes rank
        deficiency in the position sub-Jacobian of the first three joints
        ``J[3:6, 0:3]``.

        If *fk_solver* is available, the XY distance of the wrist center
        from the base Z-axis is also computed as an independent metric.
        """
        metrics: Dict[str, float] = {}

        # Position sub-Jacobian for arm joints
        J_arm_pos = J[3:6, 0:3]
        try:
            det_arm = float(np.linalg.det(J_arm_pos))
        except np.linalg.LinAlgError:
            det_arm = 0.0
        sv_arm = np.linalg.svd(J_arm_pos, compute_uv=False)
        sigma_min_arm = float(np.min(sv_arm))

        metrics["det_arm_jacobian"] = det_arm
        metrics["sigma_min"] = sigma_min_arm

        # Wrist-center XY distance from J1 axis (base Z) when FK is available
        if fk_solver is not None:
            try:
                fk_result = fk_solver.solve(q)
                wc_pos = fk_result.position_m
                xy_dist = float(np.sqrt(wc_pos[0]**2 + wc_pos[1]**2))
                metrics["wrist_center_xy_distance_m"] = xy_dist
            except Exception:
                pass

        threshold = self.type_thresholds.get("shoulder", 0.01)
        # Shoulder singularity: arm position sub-Jacobian loses rank
        # but wrist sub-Jacobian is fine (otherwise it's a different type).
        is_active = sigma_min_arm < threshold

        return is_active, metrics

    def _classify_elbow(
        self,
        J: np.ndarray,
        q: np.ndarray,
    ) -> tuple:
        """
        Detect elbow singularity.

        When the arm is fully extended or folded, the positional columns
        of joints 2 and 3 in the linear-velocity sub-Jacobian become
        collinear, collapsing a degree of freedom.
        """
        metrics: Dict[str, float] = {}

        # Columns for J2 and J3 in the linear velocity block
        j2_col = J[3:6, 1]
        j3_col = J[3:6, 2]

        # SVD of the 3×2 sub-matrix [j2 | j3]
        J_elbow = np.column_stack([j2_col, j3_col])
        sv_elbow = np.linalg.svd(J_elbow, compute_uv=False)
        sigma_min_elbow = float(np.min(sv_elbow))

        # Collinearity metric: |sin(angle)| between the two columns
        n2 = np.linalg.norm(j2_col)
        n3 = np.linalg.norm(j3_col)
        if n2 > 1e-12 and n3 > 1e-12:
            cos_angle = np.clip(np.dot(j2_col, j3_col) / (n2 * n3), -1.0, 1.0)
            collinearity = 1.0 - abs(cos_angle)
        else:
            collinearity = 0.0

        metrics["sigma_min"] = sigma_min_elbow
        metrics["j2_j3_collinearity"] = collinearity

        threshold = self.type_thresholds.get("elbow", 0.01)
        is_active = sigma_min_elbow < threshold

        return is_active, metrics
