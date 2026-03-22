"""Parse `export_final_trajectory_csv` output for dense playback and Viser."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_final_trajectory_csv(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load dense TOPP CSV: time_ms, x_m..qz, j*_rad, ..."""
    p = Path(path)
    if not p.is_file():
        return None
    time_ms: List[float] = []
    q_rows: List[List[float]] = []
    pos: List[List[float]] = []
    quat: List[List[float]] = []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        try:
            ti = header.index("time_ms")
        except ValueError:
            return None
        ix, iy, iz = header.index("x_m"), header.index("y_m"), header.index("z_m")
        iqw, iqx, iqy, iqz = header.index("qw"), header.index("qx"), header.index("qy"), header.index("qz")
        jcols = [c for c in header if c.startswith("j") and c.endswith("_rad") and "_dot" not in c and "_ddot" not in c]
        jcols.sort(key=lambda s: int(s.split("_")[0][1:]))
        ij = [header.index(c) for c in jcols]
        for row in r:
            if len(row) < max(ij + [ti, ix, iqz]):
                continue
            time_ms.append(float(row[ti]))
            pos.append([float(row[ix]), float(row[iy]), float(row[iz])])
            quat.append(
                [float(row[iqw]), float(row[iqx]), float(row[iqy]), float(row[iqz])]
            )
            q_rows.append([float(row[j]) for j in ij])
    if not time_ms:
        return None
    return {
        "time_ms": np.asarray(time_ms, dtype=np.float64),
        "q_rad": np.asarray(q_rows, dtype=np.float64),
        "position_m": np.asarray(pos, dtype=np.float64),
        "quaternion_wxyz": np.asarray(quat, dtype=np.float64),
        "n_samples": len(time_ms),
    }


def row_at_index(data: Dict[str, Any], idx: int) -> Tuple[List[float], List[float]]:
    """Joint rad (6) and TCP [x,y,z,qw,qx,qy,qz] for one row."""
    i = int(np.clip(idx, 0, data["n_samples"] - 1))
    q = data["q_rad"][i].tolist()
    p = data["position_m"][i]
    qn = data["quaternion_wxyz"][i]
    tcp = np.concatenate([p, qn]).tolist()
    return q, tcp
