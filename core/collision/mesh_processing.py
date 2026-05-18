#!/usr/bin/env python3
"""Hybrid mesh simplification: precomputed paths preferred, optional runtime decimation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_mesh_path(mesh_path: str, project_root: Optional[Path] = None) -> str:
    """Resolve mesh path to absolute (CWD, then project root)."""
    p = Path(mesh_path)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    cwd = (Path.cwd() / p).resolve()
    if cwd.exists():
        return str(cwd)
    if project_root is not None:
        pr = (project_root / p).resolve()
        if pr.exists():
            return str(pr)
    raise FileNotFoundError(f"Mesh not found: {mesh_path}")


def effective_mesh_path(
    original_path: str,
    simplified_mesh_path: Optional[str],
    decimation_ratio: Optional[float],
    cache_path: Optional[str],
    *,
    project_root: Optional[Path] = None,
    force_runtime_decimation: bool = False,
) -> Tuple[str, str]:
    """Pick STL path to load and describe the source ('precomputed'|'original'|'decimated').

    Returns:
        (absolute_path, source_tag)
    """
    if simplified_mesh_path:
        abs_s = resolve_mesh_path(simplified_mesh_path, project_root)
        return abs_s, "precomputed"

    if decimation_ratio is not None and decimation_ratio < 1.0 - 1e-9:
        if cache_path and not force_runtime_decimation:
            try:
                abs_c = resolve_mesh_path(cache_path, project_root)
                return abs_c, "decimated_cache"
            except FileNotFoundError:
                pass
        dec_path, ok = decimate_stl_optional(
            resolve_mesh_path(original_path, project_root),
            decimation_ratio,
            out_path=cache_path,
            project_root=project_root,
        )
        if ok:
            return dec_path, "decimated_runtime"
        logger.warning(
            "Runtime mesh decimation unavailable or failed; using full mesh for %s",
            original_path,
        )

    return resolve_mesh_path(original_path, project_root), "original"


def decimate_stl_optional(
    src_stl: str,
    ratio: float,
    out_path: Optional[str] = None,
    *,
    project_root: Optional[Path] = None,
) -> Tuple[str, bool]:
    """Optionally decimate ``src_stl`` with **trimesh** if installed.

    Args:
        src_stl: Absolute path to STL.
        ratio: Target fraction of faces (0<ratio<=1).
        out_path: If set, write decimated mesh here (directories created).

    Returns:
        (path_to_use, success) — on failure returns (src_stl, False).
    """
    if ratio >= 1.0 - 1e-9:
        return src_stl, True
    try:
        import trimesh  # type: ignore
    except ImportError:
        return src_stl, False

    try:
        m = trimesh.load(src_stl, force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            if hasattr(m, "geometry") and m.geometry:
                m = trimesh.util.concatenate(list(m.geometry.values()))
            else:
                return src_stl, False
        n_face = max(1, int(float(ratio) * len(m.faces)))
        simp = m.simplify_quadric_decimation(face_count=n_face)
        if out_path:
            outp = Path(out_path)
            if not outp.is_absolute() and project_root is not None:
                outp = project_root / outp
            outp.parent.mkdir(parents=True, exist_ok=True)
            simp.export(str(outp))
            return str(outp.resolve()), True
        # No cache path: temp file
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        simp.export(tmp)
        return tmp, True
    except Exception as exc:  # pragma: no cover - robustness
        logger.warning("decimate_stl_optional failed: %s", exc)
        return src_stl, False
