#!/usr/bin/env python3
"""
Unified Singularity Analysis Module
====================================

Provides a single-metric singularity detector that uses the full 6×6
Jacobian without decomposing into shoulder / elbow / wrist sub-types.

This was the original approach used in ``core/feasibility_checks.py``
before the type-classified ``SingularityAnalyzer`` was introduced.
It is retained as a simpler, faster alternative when per-type
classification is not needed.

Detection signals
-----------------
* **Minimum singular value** (σ_min):  σ_min → 0 at any singularity.
* **Condition number** (κ = σ_max / σ_min):  κ → ∞ at any singularity.
* **Manipulability** (Yoshikawa w = √det(JJᵀ)):  w → 0 at singularity.

A waypoint is flagged ``near_singularity`` when σ_min drops below a
configurable threshold.

Provides
--------
- UnifiedSingularityReport  dataclass
- UnifiedSingularity        class (main entry point)
"""

import csv
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


_EMPTY_FLOAT_ARRAY = np.array([], dtype=np.float64)


@dataclass
class UnifiedSingularityReport:
    """Per-waypoint singularity result using the unified (non-typed) approach."""

    is_singular: bool
    is_reachable: bool = True

    sigma_min: float = 0.0
    sigma_max: float = 0.0
    condition_number: float = np.inf
    manipulability: float = 0.0
    singular_values: np.ndarray = field(
        default_factory=lambda: _EMPTY_FLOAT_ARRAY.copy()
    )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten the report into a single-level dict suitable for CSV rows."""
        d: Dict[str, Any] = {
            "is_singular": "unreachable" if not self.is_reachable else self.is_singular,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "condition_number": self.condition_number,
            "manipulability": self.manipulability,
        }
        for i, sv in enumerate(self.singular_values):
            d[f"sv_{i}"] = sv
        return d


class UnifiedSingularity:
    """
    Full-Jacobian singularity detector (no type classification).

    Uses σ_min of the complete 6×6 Jacobian to decide whether a
    configuration is near-singular.  Faster than the type-classified
    ``SingularityAnalyzer`` when per-type information is not required.

    Example::

        us = UnifiedSingularity(singularity_threshold=0.01)
        report = us.analyze(jacobian)
        print(report.is_singular, report.sigma_min)
    """

    def __init__(
        self,
        singularity_threshold: float = 0.01,
        characteristic_length_m: float = 1.0,
    ):
        self.singularity_threshold = singularity_threshold
        self.characteristic_length_m = characteristic_length_m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, jacobian: np.ndarray) -> UnifiedSingularityReport:
        """
        Analyze a single configuration for singularity.

        Args:
            jacobian: 6×n Jacobian matrix.

        Returns:
            UnifiedSingularityReport with metrics.
        """
        J = np.asarray(jacobian, dtype=np.float64)

        try:
            sv = np.linalg.svd(J, compute_uv=False)
        except np.linalg.LinAlgError:
            sv = np.zeros(min(J.shape))

        if np.any(np.isnan(sv)):
            return UnifiedSingularityReport(
                is_singular=True,
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

        J_norm = J.copy()
        J_norm[3:6, :] = J_norm[3:6, :] / self.characteristic_length_m
        manip = float(np.sqrt(max(np.linalg.det(J_norm @ J_norm.T), 0.0)))

        is_singular = sigma_min < self.singularity_threshold

        return UnifiedSingularityReport(
            is_singular=is_singular,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            condition_number=cond,
            manipulability=manip,
            singular_values=sv_sorted,
        )

    def analyze_trajectory(
        self,
        jacobians: List[np.ndarray],
    ) -> List[UnifiedSingularityReport]:
        """Batch convenience: analyze every waypoint in a trajectory."""
        return [self.analyze(J) for J in jacobians]

    def summarize_trajectory(
        self,
        reports: List[UnifiedSingularityReport],
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

        return {
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

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    @staticmethod
    def export_csv(
        reports: List[UnifiedSingularityReport],
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

        # Collect union of all keys across rows (unreachable waypoints have sparse dicts)
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = ["waypoint_index"] + sorted(k for k in all_keys if k != "waypoint_index")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
