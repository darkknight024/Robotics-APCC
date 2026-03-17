"""
Manipulability checks.

Provides Yoshikawa (unified), translational, rotational, normalised-combined,
and directional manipulability indices.

Jacobian convention: rows are [angular_vel(3); linear_vel(3)] x n_joints.
"""

import numpy as np


def compute_manipulability(
    jacobian: np.ndarray,
    characteristic_length_m: float = 1.0,
) -> float:
    """Normalised Yoshikawa manipulability sqrt(det(J_norm @ J_norm^T)).

    Linear rows (3:6) are divided by *characteristic_length_m* so the
    result is dimensionless.
    """
    J = jacobian.copy()
    J[3:6, :] /= characteristic_length_m
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def compute_translational_manipulability(jacobian: np.ndarray) -> float:
    """w_v = sqrt(det(Jv @ Jv^T)) where Jv = J[3:6, :]."""
    Jv = jacobian[3:6, :]
    return float(np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0.0)))


def compute_rotational_manipulability(jacobian: np.ndarray) -> float:
    """w_omega = sqrt(det(Jw @ Jw^T)) where Jw = J[0:3, :]."""
    Jw = jacobian[0:3, :]
    return float(np.sqrt(max(np.linalg.det(Jw @ Jw.T), 0.0)))


def compute_normalized_manipulability(
    jacobian: np.ndarray,
    Lc: float,
) -> float:
    """Combined manipulability with characteristic-length scaling on angular rows.

    J_norm = diag(Lc*I3, I3) @ J  ->  w_norm = sqrt(det(J_norm @ J_norm^T)).
    """
    if Lc < 1e-9:
        return 0.0
    J = jacobian.copy()
    J[0:3, :] *= Lc
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def compute_directional_manipulability(
    jacobian: np.ndarray,
    t_hat: np.ndarray,
) -> float:
    """w_d = ||Jv^T @ t_hat||_2 along the end-effector travel direction."""
    Jv = jacobian[3:6, :]
    return float(np.linalg.norm(Jv.T @ t_hat))
