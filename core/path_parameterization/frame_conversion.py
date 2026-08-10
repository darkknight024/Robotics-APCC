"""Base ↔ tool (plate) frame conversion for path speeds.

The toolpath CSV, the commanded speed (column 8) and the RobotStudio
``speed_mm_per_s`` log all live in the **plate / tool frame** (T_P_K —
knife tip traced in the cutting-plate frame).  The solver's parameterised
path q(s), however, uses the **robot-base** arc (T_B_P TCP positions after
the Zund knife transform).  Because the knife transform is pose-dependent
(the plate rotates under a fixed knife), equal steps in one frame are NOT
equal steps in the other: near reorientation-heavy corners the base-frame
step can compress ~10× relative to the plate-frame step.

This module provides the *frame gain*

    g(s) = ds_tool / ds_base

so that speed limits authored in the tool frame can be enforced on the
base-frame path parameter (``ṡ ≤ v_tool / g``) and solver profiles can be
reported back in the tool frame (``v_tool = g · ṡ``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Base-frame steps shorter than this contribute unreliable ratios; the
# gain is clipped instead of allowed to blow up (dedup upstream removes
# true zero-length steps in the active path parameter).
_GAIN_MAX = 1.0e4


def plate_tcp_from_base_poses(
    poses_base: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Recover plate-frame knife-tip positions from base-frame TCP poses.

    ``T_B_P = T_B_K · (T_P_K)^{-1}``  ⇒  ``T_P_K = (T_B_P)^{-1} · T_B_K``,
    whose translation is ``R_BP^T (t_BK − t_BP)``.

    Parameters
    ----------
    poses_base : (M, 7) ``[x_mm, y_mm, z_mm, qw, qx, qy, qz]`` — plate/EE
        pose in robot base (same layout as the dense Feature-3 poses).
    knife_translation_m : (3,) knife position in base [m].
    knife_quaternion_wxyz : (4,) knife orientation in base [wxyz].

    Returns
    -------
    (M, 3) knife-tip positions in the plate frame [mm].
    """
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses_base, dtype=float)
    t_bp_mm = poses[:, :3]
    quat_wxyz = poses[:, 3:7]
    # scipy uses xyzw ordering.
    rot_bp = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    t_bk_mm = np.asarray(knife_translation_m, dtype=float) * 1000.0
    return rot_bp.inv().apply(t_bk_mm[None, :] - t_bp_mm)


def plate_arc_and_gain(
    s_param_mm: np.ndarray,
    plate_xyz_mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative plate-frame arc and samplewise gain vs the path parameter.

    Both arrays are aligned with the input samples.  The gain uses the
    *outgoing* transition of each sample (last sample repeats the previous
    value), matching the ``dp_ds`` convention of the SE(3) parameterisation.

    Returns ``(s_plate_mm, gain)`` with ``gain[i] ≈ ds_plate/ds_param`` at
    sample ``i``.  Samples where the parameter step vanishes inherit the
    neighbouring gain (clipped to ``1e4``); pure base-frame motion with no
    plate motion yields ``gain = 0`` (tool-frame caps then do not bind).
    """
    s_param = np.asarray(s_param_mm, dtype=float)
    xyz = np.asarray(plate_xyz_mm, dtype=float)
    if len(s_param) != len(xyz):
        raise ValueError(
            f"s_param ({len(s_param)}) and plate_xyz ({len(xyz)}) length mismatch"
        )
    ds_plate = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s_plate = np.concatenate([[0.0], np.cumsum(ds_plate)])

    ds_param = np.diff(s_param)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_seg = np.where(ds_param > 1e-12, ds_plate / ds_param, np.nan)
    # Fill degenerate segments from the previous valid ratio.
    if np.any(np.isnan(g_seg)):
        valid = ~np.isnan(g_seg)
        if not np.any(valid):
            g_seg = np.ones_like(g_seg)
        else:
            idx = np.arange(len(g_seg))
            filled = np.interp(idx, idx[valid], g_seg[valid])
            g_seg = filled
    g_seg = np.clip(g_seg, 0.0, _GAIN_MAX)

    gain = np.empty(len(s_param), dtype=float)
    gain[:-1] = g_seg
    gain[-1] = g_seg[-1] if len(g_seg) else 1.0
    return s_plate, gain
