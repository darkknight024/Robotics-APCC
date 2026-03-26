#!/usr/bin/env python3
"""
Singularity Analysis — Consolidated Module
============================================

Single source of truth for all singularity detection in the project.
Supports two modes via :class:`SingularityAnalyzer`:

* **unified** — full-Jacobian σ_min threshold (fast, no type breakdown).
* **classified** — decomposes into shoulder / elbow / wrist sub-types
  for 6-DOF spherical-wrist robots (e.g., ABB IRB 1300 family).

For a 6R spherical-wrist manipulator the three sub-types are:

* **Wrist** — J4/J6 axes align when J5 ≈ 0 or π.
* **Shoulder** — wrist center on/near the J1 axis.
* **Elbow** — arm fully extended or folded (J2/J3 columns collinear).

The Jacobian convention is ``[angular_vel (3); linear_vel (3)] × n_joints``.

Provides
--------
- :class:`SingularityMode` enum
- :class:`SingularityType` enum  (classified-mode sub-types)
- :class:`SingularityReport` dataclass  (universal, both modes)
- :class:`SingularityAnalyzer` class  (main entry point)
- Low-level helpers: :func:`compute_singularity_proximity`,
  :func:`compute_max_singular_value`, :func:`compute_condition_number`,
  :func:`analyze_singularity_spectrum`, :func:`j5_wrist_singularity_band_active`
"""

import csv
import numpy as np
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# ============================================================================
# Enums
# ============================================================================

class SingularityMode(Enum):
    """Analysis mode for :class:`SingularityAnalyzer`."""
    UNIFIED = "unified"
    CLASSIFIED = "classified"


class SingularityType(Enum):
    """Sub-type classification for 6R spherical-wrist robots (classified mode)."""
    NONE = "none"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    SHOULDER_ELBOW = "shoulder+elbow"
    SHOULDER_WRIST = "shoulder+wrist"
    ELBOW_WRIST = "elbow+wrist"
    SHOULDER_ELBOW_WRIST = "shoulder+elbow+wrist"


_COMPOUND_MAP = {
    frozenset(): SingularityType.NONE,
    frozenset(["shoulder"]): SingularityType.SHOULDER,
    frozenset(["elbow"]): SingularityType.ELBOW,
    frozenset(["wrist"]): SingularityType.WRIST,
    frozenset(["shoulder", "elbow"]): SingularityType.SHOULDER_ELBOW,
    frozenset(["shoulder", "wrist"]): SingularityType.SHOULDER_WRIST,
    frozenset(["elbow", "wrist"]): SingularityType.ELBOW_WRIST,
    frozenset(["shoulder", "elbow", "wrist"]): SingularityType.SHOULDER_ELBOW_WRIST,
}

_DEFAULT_TYPE_THRESHOLDS = {
    "wrist": 0.1,
    "shoulder": 0.1,
    "elbow": 0.1,
}

_EMPTY_FLOAT_ARRAY = np.array([], dtype=np.float64)


# ============================================================================
# Report dataclass (universal — works for both modes)
# ============================================================================

@dataclass
class SingularityReport:
    """Per-waypoint singularity analysis result.

    Works for both unified and classified modes.  Fields that are only
    meaningful in classified mode (``singularity_type``, ``active_types``,
    ``wrist_metrics``, etc.) are ``None`` / empty when mode is unified.
    """

    is_singular: bool
    is_reachable: bool = True
    mode: SingularityMode = SingularityMode.UNIFIED

    sigma_min: float = 0.0
    sigma_max: float = 0.0
    condition_number: float = np.inf
    manipulability: float = 0.0
    singular_values: np.ndarray = field(
        default_factory=lambda: _EMPTY_FLOAT_ARRAY.copy()
    )

    singularity_type: Optional[SingularityType] = None
    active_types: List[str] = field(default_factory=list)
    wrist_metrics: Optional[Dict[str, float]] = None
    shoulder_metrics: Optional[Dict[str, float]] = None
    elbow_metrics: Optional[Dict[str, float]] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten into a single-level dict suitable for CSV rows."""
        d: Dict[str, Any] = {
            "is_singular": "unreachable" if not self.is_reachable else self.is_singular,
            "mode": self.mode.value,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "condition_number": self.condition_number,
            "manipulability": self.manipulability,
        }
        if self.singularity_type is not None:
            d["singularity_type"] = (
                self.singularity_type.value if self.is_reachable else "unreachable"
            )
        for i, sv in enumerate(self.singular_values):
            d[f"sv_{i}"] = sv
        if self.wrist_metrics:
            for k, v in self.wrist_metrics.items():
                d[f"wrist_{k}"] = v
        if self.shoulder_metrics:
            for k, v in self.shoulder_metrics.items():
                d[f"shoulder_{k}"] = v
        if self.elbow_metrics:
            for k, v in self.elbow_metrics.items():
                d[f"elbow_{k}"] = v
        return d


# ============================================================================
# Low-level helpers (used by FeasibilityAnalyzer and plotting)
# ============================================================================

def compute_singularity_proximity(jacobian: np.ndarray) -> float:
    """Minimum singular value of the Jacobian (σ_min → 0 at singularity)."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.min(singular_values))


def compute_max_singular_value(jacobian: np.ndarray) -> float:
    """Maximum singular value of the Jacobian."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.max(singular_values))


def compute_condition_number(jacobian: np.ndarray) -> float:
    """Condition number κ = σ_max / σ_min (∞ near singularity)."""
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if np.any(np.isnan(singular_values)):
            return np.inf
        min_sv = np.min(singular_values)
        max_sv = np.max(singular_values)
        if min_sv < 1e-10 or np.isnan(min_sv) or np.isnan(max_sv):
            return np.inf
        cond = max_sv / min_sv
        return np.inf if np.isnan(cond) else float(cond)
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def analyze_singularity_spectrum(jacobian: np.ndarray) -> Dict[str, Any]:
    """Full singular-value spectrum for dashboarding.

    Returns:
        Dict with keys ``singular_values``, ``sigma_min``, ``sigma_max``,
        ``condition_number``.
    """
    try:
        svs = np.linalg.svd(jacobian, compute_uv=False)
    except np.linalg.LinAlgError:
        n = jacobian.shape[1]
        svs = np.zeros(n)

    return {
        "singular_values": svs,
        "sigma_min": float(np.min(svs)),
        "sigma_max": float(np.max(svs)),
        "condition_number": compute_condition_number(jacobian),
    }


def j5_wrist_singularity_band_active(q: np.ndarray, threshold_deg: float) -> bool:
    """True when joint 5 lies in the wrist singularity band for *threshold_deg*.

    Same geometry as :meth:`SingularityAnalyzer._classify_wrist` with
    ``check_j5_only=True``: flag when ``|sin(q5)| < sin(threshold_rad)``.
    Used by EAIK branch scoring, feasibility plots, and RobotStudio overlays.

    Args:
        q: Joint vector in radians; uses ``q[4]`` as J5 when ``len(q) > 4``.
        threshold_deg: Angular half-width of the singular band (degrees).

    Returns:
        ``True`` if the configuration is classified as wrist-singular for this band.
    """
    thr_rad = np.radians(float(threshold_deg))
    q5 = float(q[4]) if len(q) > 4 else 0.0
    dist_to_singularity = abs(np.sin(q5))
    return bool(dist_to_singularity < np.sin(thr_rad))


# ============================================================================
# SingularityAnalyzer — single class, two modes
# ============================================================================

class SingularityAnalyzer:
    """Singularity detector supporting *unified* and *classified* modes.

    * ``mode='unified'`` — flags a configuration as near-singular when the
      full-Jacobian σ_min drops below *singularity_threshold*.  Fast; no
      per-type breakdown.

    * ``mode='classified'`` — additionally decomposes into shoulder /
      elbow / wrist sub-types for 6-DOF spherical-wrist robots.  Requires
      joint positions.

    Example::

        analyzer = SingularityAnalyzer(mode="unified", singularity_threshold=0.01)
        report = analyzer.analyze(jacobian)
        print(report.is_singular, report.sigma_min)

        analyzer_c = SingularityAnalyzer(mode="classified")
        report_c = analyzer_c.analyze(jacobian, joint_positions=q, fk_solver=fk)
        print(report_c.singularity_type, report_c.active_types)
    """

    def __init__(
        self,
        mode: str = "unified",
        singularity_threshold: float = 0.01,
        characteristic_length_m: float = 1.0,
        n_joints: int = 6,
        type_thresholds: Optional[Dict[str, float]] = None,
        check_j5_only: bool = True,
        j5_threshold_deg: float = 0.76,
    ):
        self.mode = SingularityMode(mode.lower().strip())
        self.singularity_threshold = singularity_threshold
        self.characteristic_length_m = characteristic_length_m
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
        joint_positions: Optional[np.ndarray] = None,
        fk_solver=None,
    ) -> SingularityReport:
        """Analyze a single configuration for singularity.

        Args:
            jacobian: 6×n Jacobian ([angular(3); linear(3)] convention).
            joint_positions: Joint-angle vector (n,).  Required for
                classified mode; ignored for unified mode.
            fk_solver: Optional FK solver for wrist-center computation
                (classified mode only).

        Returns:
            :class:`SingularityReport` with metrics and optional type info.
        """
        J = np.asarray(jacobian, dtype=np.float64)

        try:
            sv = np.linalg.svd(J, compute_uv=False)
        except np.linalg.LinAlgError:
            sv = np.zeros(min(J.shape))

        if np.any(np.isnan(sv)):
            return SingularityReport(
                is_singular=True,
                mode=self.mode,
                sigma_min=0.0,
                sigma_max=0.0,
                condition_number=np.inf,
                manipulability=0.0,
                singular_values=np.zeros(min(J.shape)),
            )

        sv_sorted = np.sort(sv)[::-1]
        sigma_min = float(sv_sorted[-1]) if len(sv_sorted) > 0 else 0.0
        sigma_max = float(sv_sorted[0]) if len(sv_sorted) > 0 else 0.0
        cond = sigma_max / sigma_min if sigma_min > 1e-15 else np.inf
        manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

        is_singular = sigma_min < self.singularity_threshold

        if self.mode == SingularityMode.CLASSIFIED:
            return self._analyze_classified(
                J, sv_sorted, sigma_min, sigma_max, cond, manip,
                joint_positions, fk_solver,
            )

        return SingularityReport(
            is_singular=is_singular,
            mode=SingularityMode.UNIFIED,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            condition_number=cond,
            manipulability=manip,
            singular_values=sv_sorted,
        )

    def analyze_trajectory(
        self,
        jacobians: List[np.ndarray],
        joint_positions_list: Optional[List[np.ndarray]] = None,
        fk_solver=None,
    ) -> List[SingularityReport]:
        """Batch convenience: analyze every waypoint in a trajectory."""
        if joint_positions_list is None:
            return [self.analyze(J) for J in jacobians]
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

        singular_count = sum(1 for r in reports if r.is_singular)
        sigma_mins = [r.sigma_min for r in reports]
        cond_numbers = [
            r.condition_number for r in reports if np.isfinite(r.condition_number)
        ]

        result: Dict[str, Any] = {
            "num_waypoints": n,
            "singular_count": singular_count,
            "singular_percent": 100.0 * singular_count / n,
            "mean_sigma_min": float(np.mean(sigma_mins)),
            "min_sigma_min": float(np.min(sigma_mins)),
            "max_sigma_min": float(np.max(sigma_mins)),
            "mean_condition_number": (
                float(np.mean(cond_numbers)) if cond_numbers else np.inf
            ),
            "max_condition_number": (
                float(np.max(cond_numbers)) if cond_numbers else np.inf
            ),
        }

        if self.mode == SingularityMode.CLASSIFIED:
            type_counts: Dict[str, int] = {}
            for r in reports:
                if r.singularity_type is not None:
                    key = r.singularity_type.value
                    type_counts[key] = type_counts.get(key, 0) + 1
            result["type_counts"] = type_counts
            result["wrist_singular_count"] = sum(
                1 for r in reports if "wrist" in r.active_types
            )
            result["shoulder_singular_count"] = sum(
                1 for r in reports if "shoulder" in r.active_types
            )
            result["elbow_singular_count"] = sum(
                1 for r in reports if "elbow" in r.active_types
            )

        return result

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
            row: Dict[str, Any] = {"waypoint_index": idx}
            row.update(r.to_flat_dict())
            rows.append(row)

        all_keys: set = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = ["waypoint_index"] + sorted(
            k for k in all_keys if k != "waypoint_index"
        )
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Private: classified-mode analysis
    # ------------------------------------------------------------------

    def _analyze_classified(
        self,
        J: np.ndarray,
        sv_sorted: np.ndarray,
        sigma_min: float,
        sigma_max: float,
        cond: float,
        manip: float,
        joint_positions: Optional[np.ndarray],
        fk_solver,
    ) -> SingularityReport:
        """Run the shoulder/elbow/wrist sub-type classification."""
        q = (
            np.asarray(joint_positions, dtype=np.float64)
            if joint_positions is not None
            else np.zeros(self.n_joints)
        )

        wrist_active, wrist_metrics = self._classify_wrist(J, q)
        shoulder_active, shoulder_metrics = self._classify_shoulder(J, q, fk_solver)
        elbow_active, elbow_metrics = self._classify_elbow(J, q)

        active: List[str] = []
        if shoulder_active:
            active.append("shoulder")
        if elbow_active:
            active.append("elbow")
        if wrist_active:
            active.append("wrist")

        stype = _COMPOUND_MAP.get(frozenset(active), SingularityType.NONE)

        return SingularityReport(
            is_singular=len(active) > 0,
            mode=SingularityMode.CLASSIFIED,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            condition_number=cond,
            manipulability=manip,
            singular_values=sv_sorted,
            singularity_type=stype,
            active_types=active,
            wrist_metrics=wrist_metrics,
            shoulder_metrics=shoulder_metrics,
            elbow_metrics=elbow_metrics,
        )

    # ------------------------------------------------------------------
    # Private: per-type classifiers (classified mode only)
    # ------------------------------------------------------------------

    def _classify_wrist(self, J: np.ndarray, q: np.ndarray) -> tuple:
        """Detect wrist singularity.

        ``check_j5_only=True`` uses a fast geometric check on joint 5.
        ``check_j5_only=False`` uses SVD of the 3x3 wrist sub-Jacobian.
        """
        metrics: Dict[str, float] = {}

        J_wrist_orient = J[0:3, 3:6]
        try:
            det_w = float(np.linalg.det(J_wrist_orient))
        except np.linalg.LinAlgError:
            det_w = 0.0
        sv_w = np.linalg.svd(J_wrist_orient, compute_uv=False)
        sigma_min_w = float(np.min(sv_w))

        q5 = float(q[4]) if len(q) > 4 else 0.0
        dist_to_singularity = abs(np.sin(q5))

        metrics["det_wrist_jacobian"] = det_w
        metrics["sigma_min"] = sigma_min_w
        metrics["j5_angle_rad"] = q5
        metrics["j5_angle_deg"] = np.degrees(q5)
        metrics["j5_distance_to_singularity_rad"] = dist_to_singularity

        if self.check_j5_only:
            is_active = j5_wrist_singularity_band_active(q, self.j5_threshold_deg)
        else:
            threshold = self.type_thresholds.get("wrist", 0.01)
            is_active = sigma_min_w < threshold

        return is_active, metrics

    def _classify_shoulder(
        self, J: np.ndarray, q: np.ndarray, fk_solver=None
    ) -> tuple:
        """Detect shoulder singularity via arm position sub-Jacobian."""
        metrics: Dict[str, float] = {}

        J_arm_pos = J[3:6, 0:3]
        try:
            det_arm = float(np.linalg.det(J_arm_pos))
        except np.linalg.LinAlgError:
            det_arm = 0.0
        sv_arm = np.linalg.svd(J_arm_pos, compute_uv=False)
        sigma_min_arm = float(np.min(sv_arm))

        metrics["det_arm_jacobian"] = det_arm
        metrics["sigma_min"] = sigma_min_arm

        if fk_solver is not None:
            try:
                fk_result = fk_solver.solve(q)
                wc_pos = fk_result.position_m
                xy_dist = float(np.sqrt(wc_pos[0] ** 2 + wc_pos[1] ** 2))
                metrics["wrist_center_xy_distance_m"] = xy_dist
            except Exception:
                pass

        threshold = self.type_thresholds.get("shoulder", 0.01)
        is_active = sigma_min_arm < threshold
        return is_active, metrics

    def _classify_elbow(self, J: np.ndarray, q: np.ndarray) -> tuple:
        """Detect elbow singularity via J2/J3 column collinearity."""
        metrics: Dict[str, float] = {}

        j2_col = J[3:6, 1]
        j3_col = J[3:6, 2]

        J_elbow = np.column_stack([j2_col, j3_col])
        sv_elbow = np.linalg.svd(J_elbow, compute_uv=False)
        sigma_min_elbow = float(np.min(sv_elbow))

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
