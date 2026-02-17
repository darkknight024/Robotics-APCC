#!/usr/bin/env python3
"""
URDF Loader Utility

Handles URDF file path resolution, fuzzy matching, and model loading.
Separates file system operations from core IK solving logic.
"""

import numpy as np
import pinocchio as pin
from pathlib import Path
from typing import Tuple, List


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
    
    # Search from project root - look for "Assets" or "Assets" folder
    try:
        for root_name in ["Assets", "Assets", "assets"]:
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


def load_robot_model(urdf_path: str) -> Tuple[pin.Model, pin.Data]:
    """
    Load robot model from URDF file.
    
    This function handles path resolution and fuzzy matching, then loads
    the Pinocchio model. For path resolution only, use resolve_urdf_path().
    
    Args:
        urdf_path: Path to URDF file (can be relative or absolute)
        
    Returns:
        model: Pinocchio model
        data: Pinocchio data
        
    Raises:
        FileNotFoundError: If URDF file not found
        ValueError: If URDF file is invalid
    """
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
