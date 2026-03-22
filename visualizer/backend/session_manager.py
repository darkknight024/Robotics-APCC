"""
Session storage for visualizer uploads under output/visualizer_sessions/{session_id}/.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

SESSION_TTL_SEC = 24 * 3600
ORIGINAL_CSV = "original.csv"
NORMALIZED_CSV = "normalized.csv"
METADATA_JSON = "metadata.json"


def sessions_root(project_root: Path) -> Path:
    return project_root / "output" / "visualizer_sessions"


def cleanup_old_sessions(project_root: Path, max_age_sec: float = SESSION_TTL_SEC) -> int:
    """Remove session directories older than max_age_sec. Returns count removed."""
    root = sessions_root(project_root)
    if not root.exists():
        return 0
    now = time.time()
    removed = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        meta = p / METADATA_JSON
        try:
            if meta.exists():
                with open(meta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                created = float(data.get("created_at", 0))
            else:
                created = p.stat().st_mtime
            if now - created > max_age_sec:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            continue
    return removed


def create_session(project_root: Path) -> Path:
    """Create a new session directory and return its path."""
    cleanup_old_sessions(project_root)
    root = sessions_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    session_dir = root / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.time(),
        "session_id": sid,
        "column_map": {},
        "active_csv": ORIGINAL_CSV,
    }
    write_metadata(session_dir, meta)
    return session_dir


def session_path(project_root: Path, session_id: str) -> Optional[Path]:
    p = sessions_root(project_root) / session_id
    return p if p.is_dir() else None


def read_metadata(session_dir: Path) -> Dict[str, Any]:
    meta_path = session_dir / METADATA_JSON
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(session_dir: Path, data: Dict[str, Any]) -> None:
    path = session_dir / METADATA_JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_metadata(session_dir: Path, **kwargs: Any) -> Dict[str, Any]:
    meta = read_metadata(session_dir)
    meta.update(kwargs)
    write_metadata(session_dir, meta)
    return meta
