"""
Math utility functions for robotics kinematics and transformations.

This module provides common mathematical operations for:
- Quaternion operations (multiplication, rotation matrix conversion,
  normalization)
- Transformation matrix operations (pose to matrix, matrix to pose)
- Rotation matrix conversions (quaternion to rotation matrix, RPY to
  rotation matrix)
- Coordinate frame transformations
"""

import numpy as np


# ---------------------------
# Quaternion utilities (w,x,y,z order)
# ---------------------------
def quat_mul(q1, q2):
    """
    Multiply quaternions q = q1 ⊗ q2.

    Both quaternions in (w,x,y,z) order.

    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)

    Returns:
        Product quaternion (w, x, y, z)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def quat_to_rot_matrix(q):
    """
    Convert quaternion (w,x,y,z) to 3x3 rotation matrix.

    Normalizes input quaternion first.

    Args:
        q: Quaternion (w, x, y, z)

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def quat_conjugate(q):
    """
    Return the conjugate (inverse) of a quaternion q = [w,x,y,z].

    Args:
        q: Quaternion (w, x, y, z)

    Returns:
        Conjugate quaternion (w, -x, -y, -z)
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def invert_quaternion(q):
    """
    Invert a quaternion [w, x, y, z].

    Args:
        q: Quaternion (w, x, y, z)

    Returns:
        Inverted quaternion
    """
    q = np.array(q, dtype=float)
    w, x, y, z = q
    conj = np.array([w, -x, -y, -z])
    norm_sq = np.dot(q, q)
    if norm_sq == 0:
        raise ValueError("Cannot invert a zero-norm quaternion.")
    return conj / norm_sq


def normalize_quat(q):
    """
    Normalize a quaternion to unit length.

    Args:
        q: Quaternion (w, x, y, z)

    Returns:
        Normalized quaternion
    """
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


# ---------------------------
# Rotation matrix conversions
# ---------------------------
def rpy_to_rot_matrix(roll, pitch, yaw):
    """
    Convert URDF RPY (roll=X, pitch=Y, yaw=Z) to rotation matrix.

    URDF applies R = Rz(yaw) * Ry(pitch) * Rx(roll).

    Args:
        roll: Roll angle in radians (rotation around X)
        pitch: Pitch angle in radians (rotation around Y)
        yaw: Yaw angle in radians (rotation around Z)

    Returns:
        3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------
# Transformation matrix utilities
# ---------------------------
def make_transform(xyz, rpy):
    """
    Build 4x4 transform from xyz (3,) in mm and rpy (3,) in radians.

    Args:
        xyz: Translation vector (x, y, z) in millimeters
        rpy: Roll, pitch, yaw angles in radians

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.array(xyz, dtype=float)  # xyz should already be in mm
    return T


def pose_to_matrix(translation, quaternion):
    """
    Convert translation vector and quaternion to 4x4 transformation matrix.

    Args:
        translation: (3,) array [x, y, z]
        quaternion: (4,) array [w, x, y, z] (unit quaternion)

    Returns:
        (4, 4) transformation matrix
    """
    # Ensure quaternion is normalized
    q_norm = quaternion / np.linalg.norm(quaternion)

    # Convert quaternion to rotation matrix
    R = quat_to_rot_matrix(q_norm)

    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T


def matrix_to_pose(T):
    """
    Extract translation vector and quaternion from 4x4 transformation matrix.

    Args:
        T: (4, 4) transformation matrix

    Returns:
        translation: (3,) array [x, y, z]
        quaternion: (4,) array [w, x, y, z]
    """
    # Extract translation
    translation = T[:3, 3]

    # Extract rotation matrix
    R = T[:3, :3]

    # Convert rotation matrix to quaternion using trace method
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # pick largest diagonal
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([w, x, y, z])

    return translation, quaternion


# ---------------------------
# Specialized transformation functions
# ---------------------------
def transform_to_ee_poses_matrix(trajectories_T_P_K, T_B_K_t_mm, T_B_K_quat):
    """
    Transform trajectories using 4x4 transformation matrices.

    trajectories_T_P_K: List of trajectories where each point is T_P_K
                        (plate w.r.t. knife) in format [x, y, z, qw, qx, qy, qz]

    T_B_K_t_mm: (3,) translation of knife origin expressed in base frame (mm)
    T_B_K_quat: (4,) quaternion (w,x,y,z) of knife expressed in base frame

    Returns: List of trajectories representing T_B_P (plate w.r.t. base,
             i.e., end effector poses)
    """
    # Normalize the knife transform quaternion
    T_B_K_quat_norm = T_B_K_quat / np.linalg.norm(T_B_K_quat)

    # Construct 4x4 transformation matrix for T_B_K (knife pose in base frame)
    T_B_K = pose_to_matrix(T_B_K_t_mm, T_B_K_quat_norm)

    out_trajs = []
    for traj in trajectories_T_P_K:
        pts_P_K = traj[:, 0:3]  # Positions of plate w.r.t. knife
        q_P_K = traj[:, 3:7]    # Orientations of plate w.r.t. knife
        # Normalize all quaternions at once
        q_P_K_norm = q_P_K / np.linalg.norm(q_P_K, axis=1, keepdims=True)
        N = len(traj)
        out_pts = np.zeros((N, 3))
        out_quats = np.zeros((N, 4))

        for i in range(N):
            # Construct 4x4 transformation matrix for T_P_K
            t_P_K = pts_P_K[i]
            q_P_K_i = q_P_K_norm[i]
            T_P_K = pose_to_matrix(t_P_K, q_P_K_i)

            # Invert T_P_K to get T_K_P
            T_K_P = np.linalg.inv(T_P_K)

            # Multiply T_B_K with T_K_P to get T_B_P
            T_B_P = T_B_K @ T_K_P

            # Extract position and quaternion from T_B_P
            out_pts[i], out_quats[i] = matrix_to_pose(T_B_P)

        # Combine positions and orientations
        new_traj = np.hstack([out_pts, out_quats])
        out_trajs.append(new_traj)

    return out_trajs
