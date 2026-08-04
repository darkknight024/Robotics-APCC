"""RobotStudio recording load and path-derivative estimates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

_DEFAULT_RS_DIR = (
    _REPO / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Results - RobotStudio" / "v9_snake_toolpaths_orientation_test"
)


@dataclass
class RSRecording:
    """RobotStudio trajectory recording aligned to robot-base arc-length."""

    s_mm: np.ndarray                  # (K,) arc-length in robot base [mm]
    t_s: np.ndarray                   # (K,) time from CSV [s] (t0 = 0)
    q_deg: np.ndarray                 # (K, 6) joint position [deg]
    qdot_deg_s: np.ndarray            # (K, 6) joint velocity [deg/s]
    qddot_deg_s2: np.ndarray          # (K, 6) joint acceleration [deg/s²]
    tcp_speed_mm_s: np.ndarray        # (K,) logged TCP linear speed [mm/s]
    tcp_accel_mm_s2: np.ndarray       # (K,) logged TCP linear accel [mm/s²]
    xyz_mm: np.ndarray                # (K, 3) TCP xyz in robot base [mm]
    path: Path = field(default_factory=Path)


@dataclass
class RSPathDerivatives:
    """Geometric path derivatives estimated from an RS recording.

    Reliability
    -----------
    * ``q(s)``, ``s_dot`` — direct logs; reliable.
    * ``s_ddot`` — Savitzky–Golay ``d(speed)/dt`` (tangential). More consistent
      with ``s_dot`` than the CSV ``linear_acceleration`` column (which only
      correlates ~0.6 with ``dv/dt`` and may mix path-normal content).
    * ``dq/ds = q̇ / ṡ``, ``d²q/ds² = (q̈ − dq/ds·s̈) / ṡ²`` — reliable only
      where ``|ṡ| ≥ v_min`` (elsewhere NaN). RS sampling is coarse (~2–3 mm,
      ~24 ms), so ``d²q/ds²`` is noisier than the dense IK/spline estimates.
    """

    s_mm: np.ndarray
    q_deg: np.ndarray
    dqds_deg_mm: np.ndarray
    d2qds2_deg_mm2: np.ndarray
    s_dot_mm_s: np.ndarray
    s_ddot_mm_s2: np.ndarray
    valid_geom: np.ndarray
    v_min_mm_s: float


def estimate_rs_path_derivatives(
    rs: RSRecording,
    v_min_mm_s: float = 5.0,
) -> RSPathDerivatives:
    """Estimate q(s), dq/ds, d²q/ds², ṡ, s̈ from RobotStudio logs."""
    s_dot = np.asarray(rs.tcp_speed_mm_s, dtype=float).copy()
    # Tangential path accel from the logged speed schedule (not CSV accel).
    s_ddot = _savgol_time_derivative(s_dot[:, None], rs.t_s).ravel()

    qdot = np.deg2rad(np.asarray(rs.qdot_deg_s, dtype=float))
    qdd = np.deg2rad(np.asarray(rs.qddot_deg_s2, dtype=float))
    valid = np.isfinite(s_dot) & (np.abs(s_dot) >= float(v_min_mm_s))

    dqds = np.full_like(qdot, np.nan)
    d2qds2 = np.full_like(qdot, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqds[valid] = qdot[valid] / s_dot[valid, None]
        d2qds2[valid] = (
            (qdd[valid] - dqds[valid] * s_ddot[valid, None])
            / np.maximum(s_dot[valid, None] ** 2, 1e-12)
        )

    return RSPathDerivatives(
        s_mm=np.asarray(rs.s_mm, dtype=float),
        q_deg=np.asarray(rs.q_deg, dtype=float),
        dqds_deg_mm=np.rad2deg(dqds),
        d2qds2_deg_mm2=np.rad2deg(d2qds2),
        s_dot_mm_s=s_dot,
        s_ddot_mm_s2=s_ddot,
        valid_geom=valid,
        v_min_mm_s=float(v_min_mm_s),
    )


def find_matching_rs_csv(
    toolpath_csv: str | Path,
    rs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Locate RobotStudio CSV with the same basename as the input toolpath."""
    name = Path(toolpath_csv).name
    root = Path(rs_dir) if rs_dir is not None else _DEFAULT_RS_DIR
    candidate = root / name
    return candidate if candidate.is_file() else None


def load_rs_recording(
    rs_csv: Path,
    repo: Optional[Path] = None,
) -> RSRecording:
    """Load a full RobotStudio recording for solver benchmarking.

    Positions in the RS CSV are in the tool/plate frame; they are transformed
    to robot base with the Zund knife pose (same as experiment24_validation)
    before arc-length is computed, so the x-axis is comparable to our solver ``s``.

    TCP speed / accel and joint vel / accel are taken from the CSV columns
    logged by RobotStudio (``speed_mm_per_s``, ``linear_acceleration_mm_s_2``,
    ``rs_j*_speed_deg_s``, ``rs_j*_accel_deg_s2``).
    """
    repo = repo or _REPO
    from utils.config_loader import load_knife_config
    from utils.transform_handler import transform_trajectory_to_base_frame

    data = np.genfromtxt(rs_csv, delimiter=",", names=True, dtype=float)
    q_deg = np.column_stack([data[f"rs_j{i}_deg"] for i in range(1, 7)])
    qdot = np.column_stack([data[f"rs_j{i}_speed_deg_s"] for i in range(1, 7)])
    qddot = np.column_stack([data[f"rs_j{i}_accel_deg_s2"] for i in range(1, 7)])
    tcp_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
    tcp_accel = np.asarray(data["linear_acceleration_mm_s_2"], dtype=float)
    t_s = np.asarray(data["time_ms"], dtype=float) / 1000.0
    t_s = t_s - t_s[0]

    poses_tpk = np.column_stack([
        data["rs_x_mm"] / 1000.0,
        data["rs_y_mm"] / 1000.0,
        data["rs_z_mm"] / 1000.0,
        data["rs_qw"], data["rs_qx"], data["rs_qy"], data["rs_qz"],
    ])
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]
    poses_base = transform_trajectory_to_base_frame(
        poses_tpk, knife.translation_m, knife.quaternion,
    )
    xyz_mm = poses_base[:, :3] * 1000.0
    ds = np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1)
    s_mm = np.concatenate([[0.0], np.cumsum(ds)])
    return RSRecording(
        s_mm=s_mm, t_s=t_s, q_deg=q_deg, qdot_deg_s=qdot, qddot_deg_s2=qddot,
        tcp_speed_mm_s=tcp_speed, tcp_accel_mm_s2=tcp_accel, xyz_mm=xyz_mm,
        path=Path(rs_csv),
    )


def load_rs_joint_vs_arc(
    rs_csv: Path,
    repo: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper: return ``(s_mm, q_deg)`` only."""
    rec = load_rs_recording(rs_csv, repo=repo)
    return rec.s_mm, rec.q_deg


def _savgol_time_derivative(
    y: np.ndarray,
    t: np.ndarray,
    *,
    window_s: float = 0.08,
    polyorder: int = 3,
) -> np.ndarray:
    """Differentiate ``y(t)`` with a Savitzky–Golay filter (low-noise).

    Preferred over raw ``np.gradient`` for RS accel→jerk and for TOPP
    bang-bang ``s̈``/``q̈`` which are piecewise and otherwise ring under CD.

    * Nearly-uniform ``t``: S-G with ``delta = median(dt)``.
    * Non-uniform ``t``: interpolate to a uniform grid, differentiate, map back.
    * Falls back to ``np.gradient`` when the series is too short for S-G.
    """
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float).ravel().copy()
    if len(t) != len(y):
        raise ValueError(f"y/t length mismatch: {len(y)} vs {len(t)}")
    n = len(t)
    if n < 3:
        return np.zeros_like(y)

    # Duplicate / non-monotone timestamps (RS CSV) → nudge before gradient/S-G.
    for i in range(1, n):
        if not np.isfinite(t[i]) or t[i] <= t[i - 1]:
            prev = t[i - 1] if np.isfinite(t[i - 1]) else 0.0
            t[i] = prev + 1e-9

    dt_med = float(np.median(np.diff(t)))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return np.gradient(y, t, axis=0)

    # Odd window covering ~window_s seconds, within [polyorder+2, n].
    n_win = int(round(float(window_s) / dt_med))
    if n_win % 2 == 0:
        n_win += 1
    min_win = polyorder + 2 + (1 - (polyorder + 2) % 2)  # next odd ≥ poly+2
    n_win = max(min_win, n_win)
    max_win = n if (n % 2 == 1) else (n - 1)
    if max_win < min_win:
        return np.gradient(y, t, axis=0)
    n_win = min(n_win, max_win)

    dt_arr = np.diff(t)
    nearly_uniform = float(np.std(dt_arr)) <= 0.05 * abs(dt_med)

    def _sg_1d(yy: np.ndarray, tt: np.ndarray) -> np.ndarray:
        if nearly_uniform:
            return savgol_filter(
                yy, window_length=n_win, polyorder=polyorder,
                deriv=1, delta=dt_med, mode="interp",
            )
        tt_u = np.linspace(tt[0], tt[-1], n)
        du = float(tt_u[1] - tt_u[0])
        if du <= 0:
            return np.gradient(yy, tt)
        yy_u = np.interp(tt_u, tt, yy)
        dyy_u = savgol_filter(
            yy_u, window_length=n_win, polyorder=polyorder,
            deriv=1, delta=du, mode="interp",
        )
        return np.interp(tt, tt_u, dyy_u)

    if y.ndim == 1:
        return _sg_1d(y, t)
    out = np.empty_like(y)
    for j in range(y.shape[1]):
        out[:, j] = _sg_1d(y[:, j], t)
    return out



def _interp_rs_to_solver(
    rs_s: np.ndarray, rs_y: np.ndarray, s_eval: np.ndarray,
    unwrap_deg: bool = False,
) -> np.ndarray:
    """Resample an RS series onto the solver arc-length axis."""
    rs_s = np.asarray(rs_s, dtype=float)
    rs_y = np.asarray(rs_y, dtype=float)
    if rs_y.ndim == 1:
        return np.interp(s_eval, rs_s, rs_y)
    out = np.empty((len(s_eval), rs_y.shape[1]), dtype=float)
    for j in range(rs_y.shape[1]):
        col = rs_y[:, j]
        if unwrap_deg:
            col = np.rad2deg(np.unwrap(np.deg2rad(col)))
        out[:, j] = np.interp(s_eval, rs_s, col)
    return out
