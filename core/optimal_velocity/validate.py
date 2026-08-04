"""STEP 0 — validate and condition the input joint path q(s)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

def step0_validate(
    q_raw: np.ndarray,
    poses: np.ndarray,
    ds_min_mm: float = 1e-6,
    jump_tol_rad: float = 0.3,
    jump_spacing_mm: float = 5.0,
    q_lower: Optional[np.ndarray] = None,
    q_upper: Optional[np.ndarray] = None,
    joint_types: Optional[List[str]] = None,
    se3_lambda_mm_per_rad: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Validate + condition the input joint path. Fails loudly.

    Returns ``(s_mm, q_kept, pos_kept, quat_kept, report)`` where ``s_mm``
    is the strictly increasing arc-length of the retained samples,
    ``q_kept`` the retained joint samples, ``pos_kept`` the retained TCP
    xyz [mm], and ``quat_kept`` the retained TCP quaternions [wxyz].

    When ``se3_lambda_mm_per_rad`` is set and > 0, ``s_mm`` is the weighted
    SE(3) arc ``√(|Δp|² + (λ·Δθ)²)`` instead of position-only Σ|Δp|.  The
    report then also carries ``s_pos_mm``, ``dp_ds``, and ``dtheta_ds`` for
    converting path speed ↔ TCP linear/angular speed.

    Joint continuity (check 0.5) respects URDF joint *type*:
      * **revolute** — remap each sample by ``±2πk`` into ``[q_lower, q_upper]``
        choosing the equivalent nearest the previous sample (interval metric).
        Unbounded ``np.unwrap`` is incorrect: revolute joints have hard stops.
      * **continuous** — ``np.unwrap`` (multi-turn on the circle is allowed).
    """
    from utils.math import make_joint_path_continuous

    report: Dict = {"checks": {}}
    q = np.asarray(q_raw, dtype=float)
    poses = np.asarray(poses, dtype=float)

    # 0.1 SHAPE ------------------------------------------------------------
    if q.ndim != 2:
        raise ValueError(f"[0.1] q_raw must be 2-D, got shape {q.shape}")
    if q.shape[1] != 6 and q.shape[0] == 6:
        print("[0.1] WARN: q_raw looks like (6, M); transposing to (M, 6).")
        q = q.T
    if q.shape[1] != 6:
        raise ValueError(
            f"[0.1] q_raw must have 6 joints; got {q.shape[1]}. Aborting."
        )
    M = q.shape[0]
    if M < 50:
        raise ValueError(f"[0.1] need M >= 50 samples, got {M}. Aborting.")
    report["checks"]["0.1_shape"] = (True, f"q_raw is ({M}, 6)")

    # 0.2 6-DOF POSE ORIGIN ------------------------------------------------
    if poses.ndim != 2 or poses.shape[0] != M:
        raise ValueError(
            f"[0.2] poses must be ({M}, 7) to match q_raw; got {poses.shape}"
        )
    if poses.shape[1] == 3:
        raise ValueError(
            "[0.2] input lacks orientation; cannot be the 6-DOF path we require."
        )
    if poses.shape[1] != 7:
        raise ValueError(
            f"[0.2] poses must be (M, 7) = [x,y,z,qw,qx,qy,qz]; got {poses.shape}"
        )
    quat = poses[:, 3:7]
    qnorm = np.linalg.norm(quat, axis=1)
    if not np.all(np.abs(qnorm - 1.0) < 1e-6):
        worst = float(np.max(np.abs(qnorm - 1.0)))
        raise ValueError(
            f"[0.2] quaternions not unit-norm (max |‖q‖-1| = {worst:.2e} > 1e-6)."
        )
    ori_span = float(np.max(np.ptp(quat, axis=0)))
    report["checks"]["0.2_pose_origin"] = (
        True, f"(M,7) poses, unit quats, ori span={ori_span:.4f}"
    )

    # 0.3–0.4 path parameter (s / λ) + de-dup — see core.path_parameterization
    from core.path_parameterization.validate import build_path_parameter

    pos_mm = poses[:, :3]
    pp = build_path_parameter(
        pos_mm, quat,
        se3_lambda_mm_per_rad=se3_lambda_mm_per_rad,
        ds_min_mm=ds_min_mm,
    )
    rf = pp["report_fields"]
    report["checks"]["0.3_arc_length"] = rf["checks_0_3"]
    report["checks"]["0.4_monotone_dedup"] = rf["checks_0_4"]
    report["se3_enabled"] = bool(pp["se3_enabled"])
    report["se3_lambda_mm_per_rad"] = float(rf["se3_lambda_mm_per_rad"])
    report["s_pos_total_mm"] = float(rf["s_pos_total_mm"])
    report["s_se3_total_mm"] = float(rf["s_se3_total_mm"])
    report["total_arc_length_mm"] = float(rf["total_arc_length_mm"])
    report["n_removed"] = int(rf["n_removed"])

    keep = pp["keep_mask"]
    s_mm = np.asarray(pp["s_mm"], dtype=float)[keep]
    q_kept = q[keep]
    pos_kept = pos_mm[keep]
    quat_kept = quat[keep]
    s_pos_kept = np.asarray(pp["s_pos_mm"], dtype=float)[keep]
    dp_ds_kept = np.asarray(pp["dp_ds"], dtype=float)[keep]
    dtheta_ds_kept = np.asarray(pp["dtheta_ds"], dtype=float)[keep]
    # Rebuild strictly-increasing arc-length from retained points.
    if not np.all(np.diff(s_mm) > 0):
        s_mm = np.maximum.accumulate(s_mm + np.arange(len(s_mm)) * 1e-9)
    if not np.all(np.diff(s_pos_kept) >= 0):
        s_pos_kept = np.maximum.accumulate(
            s_pos_kept + np.arange(len(s_pos_kept)) * 1e-9
        )
    report["n_kept"] = int(len(s_mm))
    report["s_pos_mm"] = s_pos_kept
    report["dp_ds"] = dp_ds_kept
    report["dtheta_ds"] = dtheta_ds_kept

    # 0.5 CONTINUITY / BRANCH CHECK ---------------------------------------
    # Remap IK principal-value wraps using URDF joint semantics (revolute
    # stroke vs continuous unwrap).  See make_joint_path_continuous.
    types = joint_types if joint_types is not None else ["revolute"] * 6
    if q_lower is None or q_upper is None:
        # Safe IRB 1300-7/1.4 fallback if caller forgot URDF limits.
        q_lower = np.array([-3.1416, -1.6581, -3.6652, -4.0143, -2.2689, -6.9813])
        q_upper = np.array([ 3.1416,  2.7053,  1.2043,  4.0143,  2.2689,  6.9813])
        print(
            "  [WARN] step0: no URDF position limits passed; "
            "using IRB 1300-7/1.4 revolute stroke defaults."
        )
    q_kept = make_joint_path_continuous(
        q_kept, lower=q_lower, upper=q_upper, joint_types=types,
    )
    # After remapping, consecutive samples must be close on the joint stroke.
    # A remaining large jump is a true IK branch flip (not a ±π principal wrap).
    dq = np.max(np.abs(np.diff(q_kept, axis=0)), axis=1)
    ds_kept = np.diff(s_mm)
    dense = ds_kept <= jump_spacing_mm
    viol = np.where(dense & (dq > jump_tol_rad))[0]
    if viol.size:
        k = int(viol[0])
        raise ValueError(
            f"[0.5] IK branch flip at sample {k} "
            f"(|Δq|={dq[k]:.3f} rad over {ds_kept[k]:.3f} mm > {jump_tol_rad} rad). "
            "Differentiation across a branch flip is meaningless. Aborting."
        )
    n_rev = sum(1 for t in types if str(t).lower() == "revolute")
    n_cont = sum(1 for t in types if str(t).lower() == "continuous")
    report["checks"]["0.5_continuity"] = (
        True,
        f"max |Δq| = {float(dq.max()):.4f} rad (< {jump_tol_rad}) "
        f"after URDF remap (revolute={n_rev}, continuous={n_cont})",
    )
    report["q_lower"] = np.asarray(q_lower, dtype=float)
    report["q_upper"] = np.asarray(q_upper, dtype=float)
    report["joint_types"] = list(types)

    # 0.6 PASS/FAIL TABLE --------------------------------------------------
    print("\n" + "=" * 64)
    print("STEP 0 — input validation (q(s))")
    print("=" * 64)
    for name, (ok, msg) in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {msg}")
    print("=" * 64)

    return s_mm, q_kept, pos_kept, quat_kept, report
