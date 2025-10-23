#!/usr/bin/env python3
"""
Unit tests for math_utils.py functions.

Tests individual mathematical utility functions used throughout the trajectory
visualization system.
"""

import numpy as np
import sys
import os

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from math_utils import (
    quat_mul, quat_to_rot_matrix, quat_conjugate,
    invert_quaternion, normalize_quat, rpy_to_rot_matrix,
    pose_to_matrix, matrix_to_pose, make_transform
)


def test_quaternion_multiplication():
    """Test quaternion multiplication."""
    print("Testing quaternion multiplication...")

    # Test identity multiplication
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.707, 0.0, 0.707, 0.0])
    result = quat_mul(q1, q2)
    expected = q2  # q1 is identity
    assert np.allclose(result, expected, atol=1e-6), "Identity multiplication failed"
    print("  Identity multiplication: +")

    # Test commutativity check (quaternions are not generally commutative)
    q3 = np.array([0.5, 0.5, 0.5, 0.5])
    result1 = quat_mul(q1, q3)
    result2 = quat_mul(q3, q1)
    assert np.allclose(result1, result2, atol=1e-6), "Identity should commute"
    print("  Identity commutativity: +")

    print("+ Quaternion multiplication tests passed!")


def test_rotation_matrix_conversions():
    """Test quaternion to rotation matrix conversions."""
    print("\nTesting rotation matrix conversions...")

    # Test identity quaternion
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    R_identity = quat_to_rot_matrix(q_identity)
    R_expected = np.eye(3)
    assert np.allclose(R_identity, R_expected, atol=1e-10), "Identity quaternion should give identity matrix"
    print("  Identity quaternion: +")

    # Test 90-degree rotation around Z-axis
    q_90z = np.array([0.707, 0.0, 0.0, 0.707])  # cos(45), sin(45) around Z
    R_90z = quat_to_rot_matrix(q_90z)
    # 90 degrees around Z should swap X and Y axes
    expected_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(R_90z, expected_90z, atol=1e-6), "90-degree Z rotation failed"
    print("  90-degree Z rotation: +")

    # Test quaternion normalization in conversion
    q_unnormalized = np.array([2.0, 0.0, 0.0, 0.0])
    R_normalized = quat_to_rot_matrix(q_unnormalized)
    R_identity_again = quat_to_rot_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R_normalized, R_identity_again, atol=1e-10), "Unnormalized quaternion should be normalized"
    print("  Quaternion normalization: +")

    print("+ Rotation matrix conversion tests passed!")


def test_rpy_conversions():
    """Test RPY (Roll-Pitch-Yaw) to rotation matrix conversions."""
    print("\nTesting RPY conversions...")

    # Test zero RPY (should give identity)
    rpy_zero = np.array([0.0, 0.0, 0.0])
    R_zero = rpy_to_rot_matrix(rpy_zero[0], rpy_zero[1], rpy_zero[2])
    assert np.allclose(R_zero, np.eye(3), atol=1e-10), "Zero RPY should give identity"
    print("  Zero RPY: +")

    # Test 90-degree yaw (around Z)
    rpy_yaw = np.array([0.0, 0.0, np.pi/2])
    R_yaw = rpy_to_rot_matrix(rpy_yaw[0], rpy_yaw[1], rpy_yaw[2])
    expected_yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(R_yaw, expected_yaw, atol=1e-6), "90-degree yaw failed"
    print("  90-degree yaw: +")

    print("+ RPY conversion tests passed!")


def test_transformation_matrices():
    """Test 4x4 transformation matrix operations."""
    print("\nTesting transformation matrices...")

    # Test pose_to_matrix and matrix_to_pose round-trip
    translation = np.array([10.0, 20.0, 30.0])
    quaternion = np.array([0.707, 0.0, 0.707, 0.0])

    T = pose_to_matrix(translation, quaternion)
    t_back, q_back = matrix_to_pose(T)

    assert np.allclose(translation, t_back, atol=1e-10), "Translation round-trip failed"

    # Check that the returned quaternion produces the same rotation matrix
    # (accounting for sign ambiguity and numerical precision)
    R_original = quat_to_rot_matrix(quaternion)
    R_returned = quat_to_rot_matrix(q_back)
    assert np.allclose(R_original, R_returned, atol=1e-6), "Rotation matrices should be equivalent"
    print("  Pose round-trip: +")

    # Test make_transform (xyz + rpy)
    xyz = np.array([5.0, 10.0, 15.0])
    rpy = np.array([0.1, 0.2, 0.3])
    T_transform = make_transform(xyz, rpy)

    # Extract components back
    t_extracted = T_transform[:3, 3]
    R_extracted = T_transform[:3, :3]

    assert np.allclose(xyz, t_extracted, atol=1e-10), "Translation extraction failed"
    print("  Transform creation: +")

    print("+ Transformation matrix tests passed!")


def test_quaternion_utilities():
    """Test quaternion utility functions."""
    print("\nTesting quaternion utilities...")

    # Test conjugate
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q_conj = quat_conjugate(q)
    expected_conj = np.array([0.5, -0.5, -0.5, -0.5])
    assert np.allclose(q_conj, expected_conj, atol=1e-10), "Quaternion conjugate failed"
    print("  Quaternion conjugate: +")

    # Test inversion
    q_inv = invert_quaternion(q)
    q_identity = quat_mul(q, q_inv)
    expected_identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(q_identity, expected_identity, atol=1e-6), "Quaternion inversion failed"
    print("  Quaternion inversion: +")

    # Test normalization
    q_unnorm = np.array([1.0, 2.0, 3.0, 4.0])
    q_norm = normalize_quat(q_unnorm)
    norm = np.linalg.norm(q_norm)
    assert abs(norm - 1.0) < 1e-10, "Quaternion normalization failed"
    print("  Quaternion normalization: +")

    print("+ Quaternion utility tests passed!")


def main():
    """Run all math utility tests."""
    print("Running math utility unit tests...")
    print("=" * 50)

    try:
        test_quaternion_multiplication()
        test_rotation_matrix_conversions()
        test_rpy_conversions()
        test_transformation_matrices()
        test_quaternion_utilities()

        print("\n" + "=" * 50)
        print("*** ALL MATH UTILITY TESTS PASSED! ***")
        print("\nVerified:")
        print("- Quaternion multiplication: +")
        print("- Rotation matrix conversions: +")
        print("- RPY conversions: +")
        print("- Transformation matrices: +")
        print("- Quaternion utilities: +")

    except Exception as e:
        print(f"\n*** MATH UTILITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
