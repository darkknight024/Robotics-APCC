#!/usr/bin/env python3
"""
URDF Loader Utility

Handles URDF file path resolution, fuzzy matching, and model loading.
Supports two backends:
  - EAIK  (urchin + EAIK analytical solver)
  - Pinocchio (pin.buildModelFromUrdf)
Separates file system operations from core IK solving logic.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class RobotModel:
    """
    Robot model loaded from URDF for use with EAIK analytical solver.
    
    Replaces the model/data tuple used by the previous solver.
    Contains the EAIK robot object and joint metadata extracted from the URDF.
    
    EAIK computes FK/IK to the last actuated joint's child link.
    The ee_transform_4x4 stores the fixed transformation from that link
    to the configured end-effector frame, which is applied as post-processing
    for FK and pre-processing for IK.
    """
    eaik_robot: Any                  # EAIK.Robot for FK and IK (up to last actuated link)
    n_joints: int                    # Number of actuated joints
    joint_names: List[str]           # Names of actuated joints
    lower_position_limit: np.ndarray # (n_joints,) lower joint limits in radians
    upper_position_limit: np.ndarray # (n_joints,) upper joint limits in radians
    ee_frame_name: str               # End-effector frame name used
    ee_transform_4x4: np.ndarray    # 4x4 homogeneous transform from last actuated link to ee
    # URDF joint types for actuated joints ("revolute", "continuous", "prismatic", ...)
    joint_types: List[str] = field(default_factory=list)


def resolve_urdf_path(urdf_path: str) -> Path:
    """
    Resolve URDF file path with fuzzy matching for similar files.
    
    This function handles:
    - Path normalization (Windows backslashes)
    - Relative path resolution
    - Fuzzy matching when exact path doesn't exist
    - Automatic selection of similar URDF files
    
    Args:
        urdf_path: Path to URDF file (can be relative or absolute)
        
    Returns:
        Resolved Path object pointing to the URDF file
        
    Raises:
        FileNotFoundError: If file not found and no good match exists
    """
    # Normalize path separators (handle Windows backslashes)
    normalized_path = str(urdf_path).replace('\\', '/')
    
    # Resolve to absolute path
    urdf_file = Path(normalized_path)
    if not urdf_file.is_absolute():
        # Resolve relative to current working directory
        urdf_file = Path.cwd() / urdf_file
        urdf_file = urdf_file.resolve(strict=False)
    
    # Check if file exists
    if not urdf_file.exists():
        # Try alternative: check if it exists as-is (without resolve)
        alt_path = Path(urdf_path)
        if not alt_path.is_absolute():
            alt_path = Path.cwd() / alt_path
        
        if alt_path.exists():
            urdf_file = alt_path
        else:
            # Fuzzy matching: try to find similar URDF files
            urdf_file = _find_similar_urdf(urdf_file, urdf_path)
    
    return urdf_file


def _find_similar_urdf(requested_path: Path, original_path: str) -> Path:
    """
    Find similar URDF file using fuzzy matching.
    
    Args:
        requested_path: The resolved path that doesn't exist
        original_path: Original path string provided by user
        
    Returns:
        Path to similar URDF file
        
    Raises:
        FileNotFoundError: If no good match found
    """
    requested_filename = requested_path.name.lower()
    # Extract key parts from filename (e.g., "irb", "1300", "1400", "urdf")
    key_parts = []
    for part in requested_filename.replace('_', '-').replace('.urdf', '').split('-'):
        if part and len(part) > 2:
            key_parts.append(part.lower())
    
    # Check if user requested "_ee" or "_with_fixture" variant
    prefers_ee = '_ee' in requested_filename or '_with_fixture' in requested_filename or 'fixture' in requested_filename
    
    cwd = Path.cwd()
    search_dirs = _get_search_directories(requested_path)
    best_match = None
    best_score = 0
    
    # Search for best matching file
    for search_dir in search_dirs:
        try:
            for f in search_dir.glob("*.urdf"):
                filename_lower = f.name.lower()
                score = _score_urdf_match(filename_lower, key_parts, prefers_ee)
                
                if score > best_score:
                    best_score = score
                    best_match = f
        except:
            pass
    
    # If we found a good match, use it automatically
    if best_match and best_score > 20:  # Threshold for auto-matching
        print(f"Warning: URDF file not found at specified path.")
        print(f"  Requested: {original_path}")
        rel_path = best_match.relative_to(cwd) if best_match.is_relative_to(cwd) else best_match
        print(f"  Using similar file: {rel_path}")
        return best_match
    else:
        # No good match found, show error with suggestions
        suggestions = _get_urdf_suggestions(search_dirs, cwd)
        error_msg = _build_error_message(requested_path, original_path, suggestions)
        raise FileNotFoundError(error_msg)


def _get_search_directories(requested_path: Path) -> List[Path]:
    """
    Get list of directories to search for URDF files.
    
    Args:
        requested_path: The requested URDF path
        
    Returns:
        List of directories to search
    """
    search_dirs = []
    cwd = Path.cwd()
    
    # Search from project root - look for "Assests" or "Assets" folder
    try:
        for root_name in ["Assests", "Assets", "assets"]:
            root_dir = cwd / root_name / "Robot APCC"
            if root_dir.exists():
                # Search all subdirectories for urdf folders
                for subdir in root_dir.iterdir():
                    if subdir.is_dir():
                        urdf_subdir = subdir / "urdf"
                        if urdf_subdir.exists():
                            search_dirs.append(urdf_subdir)
    except:
        pass
    
    # Also try the parent directories
    if requested_path.parent.parent.exists():
        try:
            for subdir in requested_path.parent.parent.iterdir():
                if subdir.is_dir():
                    urdf_subdir = subdir / "urdf"
                    if urdf_subdir.exists() and urdf_subdir not in search_dirs:
                        search_dirs.append(urdf_subdir)
        except:
            pass
    
    return search_dirs


def _score_urdf_match(filename_lower: str, key_parts: List[str], prefers_ee: bool) -> int:
    """
    Score how well a URDF filename matches the requested file.
    
    Args:
        filename_lower: Lowercase filename to score
        key_parts: Key parts extracted from requested filename
        prefers_ee: Whether user requested _ee variant
        
    Returns:
        Score (higher is better match)
    """
    score = 0
    
    # Score based on matching key parts
    for part in key_parts:
        if part in filename_lower:
            score += len(part)
    
    # Bonus for matching directory structure
    if "irb" in filename_lower and "1300" in filename_lower and "1400" in filename_lower:
        score += 50
    
    # Prefer _ee variant if user requested it
    if prefers_ee and '_ee' in filename_lower:
        score += 30
    elif prefers_ee and '_ee' not in filename_lower:
        score -= 20
    
    return score


def _get_urdf_suggestions(search_dirs: List[Path], cwd: Path) -> List[str]:
    """
    Get list of URDF file suggestions for error message.
    
    Args:
        search_dirs: Directories to search
        cwd: Current working directory
        
    Returns:
        List of suggested URDF file paths
    """
    suggestions = []
    seen_paths = set()
    
    for search_dir in search_dirs:
        try:
            similar_files = list(search_dir.glob("*.urdf"))
            for f in similar_files:
                try:
                    rel_path = f.relative_to(cwd)
                    path_str = str(rel_path).replace('\\', '/')
                except:
                    path_str = str(f).replace('\\', '/')
                
                if path_str not in seen_paths:
                    suggestions.append(path_str)
                    seen_paths.add(path_str)
                    if len(suggestions) >= 10:
                        break
        except:
            pass
    
    return suggestions


def _build_error_message(requested_path: Path, original_path: str, suggestions: List[str]) -> str:
    """
    Build error message for FileNotFoundError.
    
    Args:
        requested_path: The resolved path that doesn't exist
        original_path: Original path string provided by user
        suggestions: List of suggested URDF files
        
    Returns:
        Error message string
    """
    error_msg = (
        f"URDF file not found: {requested_path}\n"
        f"Original path: {original_path}\n"
        f"Current working directory: {Path.cwd()}\n"
    )
    if suggestions:
        error_msg += f"\nFound URDF files nearby (you might want to use one of these):\n"
        for sug in suggestions[:10]:  # Limit to 10 suggestions
            error_msg += f"  - {sug}\n"
    else:
        error_msg += f"\nTip: Make sure the URDF file path is correct and the file exists."
    
    return error_msg


def _urdf_to_sp_conv(axis_trafo, axis, parent_p):
    """
    Convert urchin axis to axis-translation convention for subproblems.
    
    Replicates the logic from eaik.IK_Robot.IKRobot.urdf_to_sp_conv.
    
    Args:
        axis_trafo: 4x4 homogeneous transformation of a joint w.r.t. a world frame
        axis: Joint axis within axis_trafo (e.g., z-axis)
        parent_p: Linear global offset of last joint
        
    Returns:
        (axis_vector, translation)
    """
    R = axis_trafo[:-1, :-1]  # Rotation in global basis frame
    T = axis_trafo[:-1, -1] - parent_p  # Translation offset from parent
    axis_n = R.dot(axis)
    return axis_n, T


def _clean_axis(axis: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    Clean up a joint axis vector by snapping near-zero components to exactly zero.
    
    URDF conversions can introduce tiny numerical artifacts (e.g., -3.67e-6)
    that prevent EAIK from recognizing kinematic structures like spherical wrists.
    
    Args:
        axis: Joint axis vector (3,)
        tol: Tolerance below which values are snapped to zero
        
    Returns:
        Cleaned axis vector (normalized)
    """
    cleaned = axis.copy()
    cleaned[np.abs(cleaned) < tol] = 0.0
    # Re-normalize to unit length
    norm = np.linalg.norm(cleaned)
    if norm > 1e-10:
        cleaned = cleaned / norm
    return cleaned


def _find_ee_link(robot, joints, ee_frame_name: str):
    """
    Find the end-effector link in the URDF, traversing fixed joints from the
    last actuated joint's child link.
    
    Args:
        robot: urchin.URDF robot object
        joints: Sorted list of actuated joints
        ee_frame_name: Name of the desired end-effector link
        
    Returns:
        ee_link object or None if not found / same as last actuated link
    """
    if not ee_frame_name:
        return None
    
    # Check if the ee_frame_name exists as a link
    try:
        ee_link = robot.link_map.get(ee_frame_name)
        if ee_link is not None:
            return ee_link
    except:
        pass
    
    return None


def load_robot_model_eaik(urdf_path: str, ee_frame_name: str = "ee_link") -> RobotModel:
    """
    Load robot model from URDF file using urchin and EAIK.
    
    This function:
    1. Resolves the URDF path (with fuzzy matching)
    2. Parses the URDF with urchin to extract joint axes, offsets, and limits
    3. Computes the end-effector offset from the last actuated joint to ee_frame_name
    4. Creates an EAIK.Robot with the proper end-effector transformation
    
    Args:
        urdf_path: Path to URDF file (can be relative or absolute)
        ee_frame_name: Name of end-effector frame in URDF (default: "ee_link").
                       If the frame doesn't exist, falls back to last actuated link.
        
    Returns:
        RobotModel containing EAIK robot and joint metadata
        
    Raises:
        FileNotFoundError: If URDF file not found
        ValueError: If URDF file is invalid or robot has no known decomposition
    """
    from urchin import URDF
    import eaik.pybindings.EAIK as EAIK

    # Resolve path (handles fuzzy matching)
    urdf_file = resolve_urdf_path(urdf_path)
    urdf_path_str = str(urdf_file).replace('\\', '/')
    
    try:
        # Parse URDF with urchin
        # lazy_load_meshes=True is used to speed up the loading process since setting it true means meshes are not loaded.
        robot = URDF.load(urdf_path_str, lazy_load_meshes=True)
        joints = robot._sort_joints(robot.actuated_joints)
        n_joints = len(joints)
        
        if n_joints == 0:
            raise ValueError(f"No actuated joints found in URDF: {urdf_path_str}")
        
        # Compute FK in zero configuration
        fk_zero_pose = robot.link_fk()
        
        # Extract joint axes (H) and offsets (P) - replicating EAIK UrdfRobot logic
        parent_p = np.zeros(3)
        H = np.array([], dtype=np.float64).reshape(0, 3)
        P = np.array([], dtype=np.float64).reshape(0, 3)
        
        for i in range(n_joints):
            joint_child_link = robot.link_map[joints[i].child]
            h, p = _urdf_to_sp_conv(fk_zero_pose[joint_child_link], joints[i].axis, parent_p)
            # Clean up axis to remove tiny numerical artifacts from URDF
            # This ensures EAIK can properly detect kinematic structures (e.g., spherical wrist)
            h = _clean_axis(h)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            # Update parent_p to absolute position of current joint
            parent_p = fk_zero_pose[joint_child_link][:-1, -1]
        
        # Append zero end-effector offset to P (EAIK handles up to last actuated link)
        # The actual ee offset is stored separately and applied as post-processing
        P = np.vstack([P, np.zeros(3)])
        
        # Create EAIK Robot with identity R_ee (preserves spherical wrist detection)
        # H.T shape: (3, n_joints), P.T shape: (3, n_joints+1)
        eaik_robot = EAIK.Robot(H.T, P.T, np.eye(3), [], True)
        
        # Compute ee_transform_4x4: the fixed transform to append after EAIK's FK
        # to reach the configured end-effector frame.
        #
        # IMPORTANT: EAIK uses product-of-exponentials, so its zero-config FK returns
        # identity rotation (not the URDF link frame rotation). We must compute the
        # ee transform relative to EAIK's output frame, NOT urchin's link frame.
        #   ee_transform_4x4 = inv(T_eaik_zero) @ T_urchin_ee_zero
        #
        last_actuated_link = robot.link_map[joints[-1].child]
        ee_link = _find_ee_link(robot, joints, ee_frame_name)
        
        # Compute inv(T_eaik_zero) — needed for both branches below
        T_eaik_zero = eaik_robot.fwdkin(np.zeros(n_joints))
        T_eaik_zero_inv = np.eye(4)
        T_eaik_zero_inv[:3, :3] = T_eaik_zero[:3, :3].T
        T_eaik_zero_inv[:3, 3] = -T_eaik_zero[:3, :3].T @ T_eaik_zero[:3, 3]
        
        if ee_link is not None and ee_link in fk_zero_pose and ee_link != last_actuated_link:
            # ee_link exists and differs from last actuated link (e.g., fixture)
            ee_fk = fk_zero_pose[ee_link]
            ee_transform_4x4 = T_eaik_zero_inv @ ee_fk
        else:
            # No ee_link or it matches the last actuated link — still need to
            # correct EAIK's rotation convention (PoE uses identity at zero-config,
            # while urchin preserves the URDF frame orientation)
            if ee_frame_name and ee_frame_name != joints[-1].child:
                print(f"Warning: End-effector frame '{ee_frame_name}' not found in URDF. "
                      f"Using last actuated link '{joints[-1].child}' as end-effector.")
                ee_frame_name = joints[-1].child
            last_link_fk = fk_zero_pose[last_actuated_link]
            ee_transform_4x4 = T_eaik_zero_inv @ last_link_fk
        
        # Check if robot has a known decomposition
        if not eaik_robot.has_known_decomposition():
            print(f"Warning: EAIK could not find a known kinematic decomposition for this robot. "
                  f"IK may return least-squares solutions only.")
        
        # Extract joint limits + types from URDF
        lower_limits = np.zeros(n_joints)
        upper_limits = np.zeros(n_joints)
        joint_names = []
        joint_types: List[str] = []
        
        for i, joint in enumerate(joints):
            joint_names.append(joint.name)
            joint_types.append(str(getattr(joint, "joint_type", "revolute")).lower())
            if joint.limit is not None:
                lower_limits[i] = joint.limit.lower if joint.limit.lower is not None else -2 * np.pi
                upper_limits[i] = joint.limit.upper if joint.limit.upper is not None else 2 * np.pi
            else:
                # Continuous / unspecified: no hard stroke — wide sentinel range.
                lower_limits[i] = -2 * np.pi
                upper_limits[i] = 2 * np.pi
        
        return RobotModel(
            eaik_robot=eaik_robot,
            n_joints=n_joints,
            joint_names=joint_names,
            lower_position_limit=lower_limits,
            upper_position_limit=upper_limits,
            ee_frame_name=ee_frame_name,
            ee_transform_4x4=ee_transform_4x4,
            joint_types=joint_types,
        )
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to load URDF model from: {urdf_path_str}\n"
            f"Error: {str(e)}\n"
            f"Please verify the URDF file is valid."
        ) from e


def load_robot_model_pin(urdf_path: str):
    """
    Load robot model from URDF file using Pinocchio.

    Restored from commit d78ff39.

    Args:
        urdf_path: Path to URDF file (can be relative or absolute)

    Returns:
        (model, data): Pinocchio model and data objects

    Raises:
        FileNotFoundError: If URDF file not found
        ValueError: If URDF file is invalid
    """
    import pinocchio as pin

    # Resolve path (handles fuzzy matching)
    urdf_file = resolve_urdf_path(urdf_path)

    # Convert to string for Pinocchio (use forward slashes for compatibility)
    urdf_path_str = str(urdf_file).replace('\\', '/')

    try:
        model = pin.buildModelFromUrdf(urdf_path_str)
        data = model.createData()
        return model, data
    except Exception as e:
        raise ValueError(
            f"Failed to load URDF model from: {urdf_path_str}\n"
            f"Error: {str(e)}\n"
            f"Please verify the URDF file is valid."
        ) from e


def load_robot_model(urdf_path: str, solver: str = "eaik", ee_frame_name: str = "ee_link"):
    """
    Dispatcher: load robot model using the requested backend.

    Args:
        urdf_path: Path to URDF file
        solver: "eaik" or "pin"
        ee_frame_name: End-effector frame name (used by EAIK loader;
                       for Pinocchio, pass to the FK/IK solver instead)

    Returns:
        - If solver == "eaik": RobotModel dataclass
        - If solver == "pin":  (pin.Model, pin.Data) tuple
    """
    solver = solver.lower().strip()
    if solver == "eaik":
        return load_robot_model_eaik(urdf_path, ee_frame_name=ee_frame_name)
    elif solver in ("pin", "pinocchio"):
        return load_robot_model_pin(urdf_path)
    else:
        raise ValueError(f"Unknown solver backend: '{solver}'. Use 'eaik' or 'pin'.")


@dataclass
class ActuatedJointMeta:
    """Lightweight URDF actuated-joint metadata (no EAIK / Pinocchio)."""

    joint_names: List[str]
    joint_types: List[str]
    lower_position_limit: np.ndarray
    upper_position_limit: np.ndarray


def load_actuated_joint_meta(urdf_path: str) -> ActuatedJointMeta:
    """Parse actuated joint types + position limits from a URDF (urchin only).

    Use this when you need revolute-vs-continuous semantics / stroke limits
    without constructing an IK backend.
    """
    from urchin import URDF

    urdf_file = resolve_urdf_path(urdf_path)
    robot = URDF.load(str(urdf_file).replace("\\", "/"), lazy_load_meshes=True)
    joints = list(robot.actuated_joints)
    if not joints:
        raise ValueError(f"No actuated joints in URDF: {urdf_file}")

    names: List[str] = []
    types: List[str] = []
    lower = np.zeros(len(joints))
    upper = np.zeros(len(joints))
    for i, joint in enumerate(joints):
        names.append(joint.name)
        types.append(str(getattr(joint, "joint_type", "revolute")).lower())
        if joint.limit is not None:
            lower[i] = (
                joint.limit.lower if joint.limit.lower is not None else -2 * np.pi
            )
            upper[i] = (
                joint.limit.upper if joint.limit.upper is not None else 2 * np.pi
            )
        else:
            # Continuous joints omit <limit> lower/upper in URDF.
            lower[i] = -np.inf
            upper[i] = np.inf
    return ActuatedJointMeta(
        joint_names=names,
        joint_types=types,
        lower_position_limit=lower,
        upper_position_limit=upper,
    )
