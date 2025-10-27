# detect_ordering_fixed.py
import numpy as np
import pinocchio as pin

def _get_vector_from_motion(m):
    """Safely get the 6-vector representation from a Motion object."""
    if hasattr(m, 'vector'):
        return m.vector
    if hasattr(m, 'toVector'):
        return m.toVector()
    # if neither exist, try converting to numpy directly
    return np.asarray(m)

def detect_motion_ordering():
    linear = np.array([1000.0, 2000.0, 3000.0])
    angular = np.array([1.0, 2.0, 3.0])

    results = []

    # 1) (linear, angular)
    try:
        m1 = pin.Motion(linear, angular)
        vec1 = _get_vector_from_motion(m1)
        results.append(("constructor_lin_ang", vec1, m1))
    except Exception as e:
        results.append(("constructor_lin_ang_failed", None, e))

    # 2) (angular, linear)
    try:
        m2 = pin.Motion(angular, linear)
        vec2 = _get_vector_from_motion(m2)
        results.append(("constructor_ang_lin", vec2, m2))
    except Exception as e:
        results.append(("constructor_ang_lin_failed", None, e))

    # 3) vector [lin, ang]
    try:
        v_lin_ang = np.hstack([linear, angular])
        m3 = pin.Motion(v_lin_ang)
        vec3 = _get_vector_from_motion(m3)
        results.append(("vector_lin_ang", vec3, m3))
    except Exception as e:
        results.append(("vector_lin_ang_failed", None, e))

    # 4) vector [ang, lin]
    try:
        v_ang_lin = np.hstack([angular, linear])
        m4 = pin.Motion(v_ang_lin)
        vec4 = _get_vector_from_motion(m4)
        results.append(("vector_ang_lin", vec4, m4))
    except Exception as e:
        results.append(("vector_ang_lin_failed", None, e))

    # Analyze
    ordering_answers = []
    for name, vec, obj in results:
        if vec is None:
            ordering_answers.append((name, "failed_or_unavailable", repr(obj)))
            continue
        vec = np.asarray(vec).flatten()
        if vec.shape != (6,):
            ordering_answers.append((name, "unexpected_shape", vec.shape))
            continue
        if np.allclose(vec[:3], angular) and np.allclose(vec[3:], linear):
            ordering_answers.append((name, "angular_first (ω, v)"))
        elif np.allclose(vec[:3], linear) and np.allclose(vec[3:], angular):
            ordering_answers.append((name, "linear_first (v, ω)"))
        else:
            ordering_answers.append((name, "unexpected: " + str(vec.tolist())))
    return ordering_answers

def test_pin_log_ordering():
    """Test what order pin.log() returns the motion vector in."""
    # Create two poses
    pose1 = pin.SE3.Identity()
    pose2 = pin.SE3(np.eye(3), np.array([0.1, 0.2, 0.3]))  # small translation

    # Get log
    motion = pin.log(pose1.inverse() * pose2)
    vec = motion.vector

    print(f"pin.log() vector: {vec}")
    print(f"Vector shape: {vec.shape}")
    print(f"First 3 elements (likely angular): {vec[:3]}")
    print(f"Last 3 elements (likely linear): {vec[3:]}")

    # Also check with rotation
    pose3 = pin.SE3(pin.utils.rotate('x', 0.1), np.zeros(3))
    motion_rot = pin.log(pose1.inverse() * pose3)
    vec_rot = motion_rot.vector

    print(f"\nRotation-only log: {vec_rot}")
    print(f"Rotation vector first 3: {vec_rot[:3]}")
    print(f"Rotation vector last 3: {vec_rot[3:]}")

if __name__ == "__main__":
    print("=== Motion Constructor Tests ===")
    for r in detect_motion_ordering():
        print(r)

    print("\n=== pin.log() Order Test ===")
    test_pin_log_ordering()
