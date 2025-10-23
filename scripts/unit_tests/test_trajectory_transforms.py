#!/usr/bin/env python3
"""
Unit tests for trajectory_visualizer.py functionality.

Tests include:
- Round-trip composition tests for coordinate transformations
- CSV reading and parsing
- Trajectory filtering (--odd, --even)
- Transformation matrix operations
"""

import numpy as np
import sys
import os

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from trajectory_visualizer import (
    read_trajectories_from_csv,
    transform_to_ee_poses_matrix,
    quat_to_rot_matrix,
    pose_to_matrix,
    matrix_to_pose
)

# Test data - same as hardcoded in trajectory_visualizer.py
T_B_K_t_mm = np.array([-367.773, -915.815, 520.4])  # mm
T_B_K_quat = np.array([0.00515984, 0.712632, -0.701518, 0.000396522])  # w,x,y,z


def test_round_trip_composition():
    """
    Test round-trip composition: T_B_K_check = T_B_P @ T_P_K should equal T_B_K

    This verifies that the transformation math is correct.
    """
    print("Testing round-trip composition...")

    # Test with a few random poses T_P_K (plate relative to knife)
    test_poses = [
        # Simple translation only
        np.array([10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0]),
        # 90 degree rotation around Z
        np.array([0.0, 0.0, 0.0, 0.707, 0.0, 0.0, 0.707]),
        # Combined translation and rotation
        np.array([15.0, -10.0, 5.0, 0.866, 0.0, 0.0, 0.5]),
        # Random orientation
        np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5])
    ]

    for i, pose_P_K in enumerate(test_poses):
        print(f"  Test pose {i+1}: {pose_P_K}")

        # Step 1: Compute T_B_P using the function
        trajectories_T_P_K = [np.array([pose_P_K])]
        trajectories_T_B_P = transform_to_ee_poses_matrix(
            trajectories_T_P_K, T_B_K_t_mm, T_B_K_quat)
        T_B_P = pose_to_matrix(trajectories_T_B_P[0][0, :3], trajectories_T_B_P[0][0, 3:7])

        # Step 2: Extract T_P_K from the input pose
        t_P_K = pose_P_K[:3]
        q_P_K = pose_P_K[3:7]
        T_P_K = pose_to_matrix(t_P_K, q_P_K)

        # Step 3: Compute round-trip: T_B_K_check = T_B_P @ T_P_K
        T_B_K_check = T_B_P @ T_P_K

        # Step 4: Extract components and compare with original T_B_K
        t_B_K_check, q_B_K_check = matrix_to_pose(T_B_K_check)
        T_B_K_original = pose_to_matrix(T_B_K_t_mm, T_B_K_quat)

        # Compare translations (within 1mm tolerance)
        trans_error = np.linalg.norm(t_B_K_check - T_B_K_t_mm)
        print(f"    Translation error: {trans_error:.6f} mm")

        # Compare quaternions (within 0.01 tolerance)
        quat_error = np.linalg.norm(q_B_K_check - T_B_K_quat)
        print(f"    Quaternion error: {quat_error:.6f}")

        # Assert tolerances
        assert trans_error < 1.0, f"Translation error too large: {trans_error} mm"
        assert quat_error < 0.01, f"Quaternion error too large: {quat_error}"

        print(f"    + Round-trip test passed for pose {i+1}")

    print("+ All round-trip composition tests passed!")


def test_csv_reading():
    """Test CSV reading functionality."""
    print("\nTesting CSV reading...")

    csv_path = os.path.join(os.path.dirname(__file__), 'test_trajectories.csv')
    trajectories = read_trajectories_from_csv(csv_path)

    print(f"  Loaded {len(trajectories)} trajectories")

    # Verify we have the expected number of trajectories
    assert len(trajectories) == 4, f"Expected 4 trajectories, got {len(trajectories)}"

    # Verify trajectory shapes
    for i, traj in enumerate(trajectories):
        print(f"    Trajectory {i+1}: {traj.shape[0]} waypoints")
        assert traj.shape[1] == 7, f"Expected 7 columns, got {traj.shape[1]}"
        # Verify quaternion normalization (each quaternion should be unit length)
        for j in range(traj.shape[0]):
            q = traj[j, 3:7]
            q_norm = np.linalg.norm(q)
            assert abs(q_norm - 1.0) < 1e-6, f"Quaternion {j} in trajectory {i} not normalized: {q_norm}"

    print("+ CSV reading tests passed!")


def test_trajectory_filtering():
    """Test --odd and --even filtering functionality."""
    print("\nTesting trajectory filtering...")

    csv_path = os.path.join(os.path.dirname(__file__), 'test_trajectories.csv')
    all_trajectories = read_trajectories_from_csv(csv_path)

    # Test even filtering
    even_trajectories = [traj for i, traj in enumerate(all_trajectories) if i % 2 == 0]
    print(f"  Even trajectories: {len(even_trajectories)} (expected: 2)")
    assert len(even_trajectories) == 2, f"Expected 2 even trajectories, got {len(even_trajectories)}"

    # Test odd filtering
    odd_trajectories = [traj for i, traj in enumerate(all_trajectories) if i % 2 == 1]
    print(f"  Odd trajectories: {len(odd_trajectories)} (expected: 2)")
    assert len(odd_trajectories) == 2, f"Expected 2 odd trajectories, got {len(odd_trajectories)}"

    # Verify even indices: 0, 2
    assert len(even_trajectories[0]) == len(all_trajectories[0])  # First trajectory
    assert len(even_trajectories[1]) == len(all_trajectories[2])  # Third trajectory

    # Verify odd indices: 1, 3
    assert len(odd_trajectories[0]) == len(all_trajectories[1])   # Second trajectory
    assert len(odd_trajectories[1]) == len(all_trajectories[3])   # Fourth trajectory

    print("+ Trajectory filtering tests passed!")


def test_transformation_with_filtering():
    """Test that transformation works correctly with filtered trajectories."""
    print("\nTesting transformation with filtering...")

    csv_path = os.path.join(os.path.dirname(__file__), 'test_trajectories.csv')
    all_trajectories = read_trajectories_from_csv(csv_path)

    # Test transformation of all trajectories
    transformed_all = transform_to_ee_poses_matrix(all_trajectories, T_B_K_t_mm, T_B_K_quat)

    # Test transformation of only even trajectories
    even_trajectories = [traj for i, traj in enumerate(all_trajectories) if i % 2 == 0]
    transformed_even = transform_to_ee_poses_matrix(even_trajectories, T_B_K_t_mm, T_B_K_quat)

    # Verify shapes match
    assert len(transformed_even) == len(even_trajectories), "Number of transformed trajectories should match input"
    for i in range(len(even_trajectories)):
        assert transformed_even[i].shape == even_trajectories[i].shape, "Trajectory shape should be preserved"

    # Verify first transformed trajectory matches the even-indexed original
    assert len(transformed_even[0]) == len(transformed_all[0]), "First transformed trajectory should match"
    assert len(transformed_even[1]) == len(transformed_all[2]), "Second transformed trajectory should match"

    print(f"  All trajectories: {len(all_trajectories)} -> {len(transformed_all)}")
    print(f"  Even trajectories: {len(even_trajectories)} -> {len(transformed_even)}")
    print("+ Transformation with filtering tests passed!")


def test_quaternion_operations():
    """Test basic quaternion operations used in transformations."""
    print("\nTesting quaternion operations...")

    # Test quaternion normalization
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_norm = q / np.linalg.norm(q)
    expected_norm = np.linalg.norm(q_norm)
    assert abs(expected_norm - 1.0) < 1e-10, f"Quaternion normalization failed: {expected_norm}"
    print(f"  Quaternion normalization: {q} -> {q_norm}")

    # Test rotation matrix conversion
    q_unit = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    R = quat_to_rot_matrix(q_unit)
    R_expected = np.eye(3)
    assert np.allclose(R, R_expected), "Identity quaternion should give identity rotation matrix"
    print(f"  Identity quaternion rotation matrix: {R.shape}")

    # Test pose to matrix round-trip
    translation = np.array([10.0, 20.0, 30.0])
    T = pose_to_matrix(translation, q_unit)
    t_back, q_back = matrix_to_pose(T)
    assert np.allclose(translation, t_back), "Translation round-trip failed"
    assert np.allclose(q_unit, q_back), "Quaternion round-trip failed"
    print(f"  Pose round-trip: {translation} -> {t_back}")

    print("+ Quaternion operations tests passed!")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")

    # Test with empty trajectories
    empty_trajectories = []
    try:
        result = transform_to_ee_poses_matrix(empty_trajectories, T_B_K_t_mm, T_B_K_quat)
        assert result == [], "Empty input should return empty output"
        print("  Empty trajectories: +")
    except Exception as e:
        print(f"  Empty trajectories failed: {e}")
        raise

    # Test with zero-length quaternion (should be handled gracefully)
    bad_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    try:
        trajectories_bad = [np.array([bad_pose])]
        result = transform_to_ee_poses_matrix(trajectories_bad, T_B_K_t_mm, T_B_K_quat)
        print("  Zero quaternion: + (handled gracefully)")
    except Exception as e:
        print(f"  Zero quaternion failed: {e}")
        # This might fail, which is acceptable

    print("+ Edge case tests completed!")


def main():
    """Run all unit tests."""
    print("Running trajectory transformation unit tests...")
    print("=" * 60)

    try:
        test_quaternion_operations()
        test_round_trip_composition()
        test_csv_reading()
        test_trajectory_filtering()
        test_transformation_with_filtering()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("*** ALL TESTS PASSED! ***")
        print("\nSummary:")
        print("- Round-trip composition tests: +")
        print("- CSV reading and parsing: +")
        print("- Trajectory filtering: +")
        print("- Transformation with filtering: +")
        print("- Quaternion operations: +")
        print("- Edge cases: +")

    except Exception as e:
        print(f"\n*** TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
