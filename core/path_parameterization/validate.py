"""Path-parameter construction (SE(3) vs position arc, de-duplication)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.path_parameterization.position_arc import compute_position_arc_length
from core.path_parameterization.se3_arc_length import compute_se3_arc_length


def build_path_parameter(
    pos_mm: np.ndarray,
    quat_wxyz: np.ndarray,
    se3_lambda_mm_per_rad: Optional[float] = None,
    ds_min_mm: float = 1e-6,
) -> Dict[str, Any]:
    """Build the active path parameter and de-dup near-duplicates.

    Returns a dict with keys
    ``s_mm, s_pos_mm, dp_ds, dtheta_ds, se3_enabled, report_fields, keep_mask``.

    ``s_mm`` / ``dp_ds`` / ``dtheta_ds`` / ``s_pos_mm`` are on the *full*
    (pre-dedup) grid; ``keep_mask`` selects samples that survive
    ``ds >= ds_min_mm`` in the active parameter.
    """
    pos_mm = np.asarray(pos_mm, dtype=float)
    quat = np.asarray(quat_wxyz, dtype=float)
    M = pos_mm.shape[0]

    s_pos_full, total_pos = compute_position_arc_length(pos_mm)
    lam = float(se3_lambda_mm_per_rad) if se3_lambda_mm_per_rad is not None else 0.0

    report_fields: Dict[str, Any] = {
        "s_pos_total_mm": total_pos,
    }

    if lam > 0.0:
        s_full, dp_ds_full, dtheta_ds_full = compute_se3_arc_length(
            pos_mm, quat, lam,
        )
        total_len = float(s_full[-1])
        se3_enabled = True
        report_fields.update({
            "se3_enabled": True,
            "se3_lambda_mm_per_rad": lam,
            "s_se3_total_mm": total_len,
            "total_arc_length_mm": total_len,
            "checks_0_3": (
                True,
                f"SE(3) arc-length = {total_len:.3f} mm "
                f"(s_pos={total_pos:.3f} mm, λ={lam:.1f} mm/rad, "
                f"+{100.0 * (total_len / total_pos - 1.0) if total_pos > 1e-9 else 0.0:.1f}%)",
            ),
        })
    else:
        s_full = s_pos_full
        dp_ds_full = np.ones(M, dtype=float)
        dtheta_ds_full = np.zeros(M, dtype=float)
        total_len = total_pos
        se3_enabled = False
        report_fields.update({
            "se3_enabled": False,
            "se3_lambda_mm_per_rad": 0.0,
            "s_se3_total_mm": total_len,
            "total_arc_length_mm": total_len,
            "checks_0_3": (
                True, f"total arc-length = {total_len:.3f} mm"
            ),
        })

    # 0.4 MONOTONE / DE-DUP — drop near-duplicates in the *active* parameter.
    ds_param = np.diff(s_full)
    keep_mask = np.concatenate([[True], ds_param >= ds_min_mm])
    n_removed = int((~keep_mask).sum())
    report_fields["n_removed"] = n_removed
    report_fields["n_kept"] = int(int(keep_mask.sum()))
    report_fields["checks_0_4"] = (
        True,
        f"removed {n_removed} near-duplicate samples (ds < {ds_min_mm} mm)",
    )

    return {
        "s_mm": s_full,
        "s_pos_mm": s_pos_full,
        "dp_ds": dp_ds_full,
        "dtheta_ds": dtheta_ds_full,
        "se3_enabled": se3_enabled,
        "report_fields": report_fields,
        "keep_mask": keep_mask,
    }
