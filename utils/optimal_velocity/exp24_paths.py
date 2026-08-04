"""Experiment-24 dataset folder resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.optimal_velocity.rs_recording import _DEFAULT_RS_DIR, find_matching_rs_csv

_REPO = Path(__file__).resolve().parents[2]

# Short --dataset keys → Experiment-24 Toolpaths/ and Results - RobotStudio/
# locations.  A plain string means the same relative folder under both roots.
# A dict allows asymmetric layouts (e.g. v7 sidewall segments).
_DATASET_FOLDERS: Dict[str, str | Dict[str, str]] = {
    "v6": "v6_constant_tool_orientation_recordings",
    "v6_2": "v6_2",
    "v7_cropped": {
        "toolpaths": "v7_sidewall_wrapped_toolpath/cropped_toolpath_by_segment",
        "rs": (
            "v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/"
            "cropped_toolpath"
        ),
    },
    "v7_full": {
        "toolpaths": "v7_sidewall_wrapped_toolpath/full_toolpath_by_segment",
        "rs": (
            "v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/"
            "full_toolpath"
        ),
    },
    "v8": "v8_snake_toolpath_with_variable_wp_spacing",
    "v9": "v9_snake_toolpaths_orientation_test",
    "v11": "v11_snake_toolpaths_with_x_axis_ori_changes",
}


def _exp24_root() -> Path:
    return _REPO / "Robot_APCC" / "Experiments" / "Experiement_24"


def _dataset_dirs(dataset: str) -> Tuple[Path, Path]:
    """Return ``(toolpath_dir, rs_dir)`` for a ``--dataset`` key."""
    spec = _DATASET_FOLDERS[dataset]
    root = _exp24_root()
    if isinstance(spec, str):
        return (
            root / "Toolpaths" / spec,
            root / "Results - RobotStudio" / spec,
        )
    return (
        root / "Toolpaths" / spec["toolpaths"],
        root / "Results - RobotStudio" / spec["rs"],
    )


def _resolve_cases(
    dataset: Optional[str],
    toolpath: Optional[str],
    rs_dir: Optional[str],
    rs_csv: Optional[str],
) -> List[Tuple[Path, Optional[Path]]]:
    """Return ``[(toolpath_csv, rs_csv_or_None), ...]``.

    * ``--dataset`` → every CSV under the dataset toolpath folder, each
      matched by basename under the dataset RobotStudio folder.
    * ``--toolpath`` → one toolpath; RS from ``--rs-csv``, else basename
      match under ``--rs-dir`` (default = v9 RS folder).
    """
    if dataset and toolpath:
        raise SystemExit("Pass either --dataset or --toolpath, not both.")
    if not dataset and not toolpath:
        raise SystemExit(
            "Provide --dataset <"
            + "|".join(sorted(_DATASET_FOLDERS))
            + "> or --toolpath <csv>."
        )

    if dataset:
        if dataset not in _DATASET_FOLDERS:
            raise SystemExit(
                f"Unknown --dataset {dataset!r}; "
                f"choices: {sorted(_DATASET_FOLDERS)}"
            )
        tp_dir, rs_root = _dataset_dirs(dataset)
        if not tp_dir.is_dir():
            raise SystemExit(f"Toolpath folder not found: {tp_dir}")
        cases = []
        for tp in sorted(tp_dir.glob("*.csv")):
            rs = rs_root / tp.name
            cases.append((tp, rs if rs.is_file() else None))
        if not cases:
            raise SystemExit(f"No CSV toolpaths in {tp_dir}")
        n_rs = sum(1 for _, rs in cases if rs is not None)
        print(f"  dataset {dataset}: {len(cases)} toolpaths, "
              f"{n_rs} with matching RobotStudio CSV")
        print(f"    toolpaths: {tp_dir}")
        print(f"    RS:        {rs_root}")
        return cases

    tp = Path(toolpath)
    if not tp.is_file():
        raise SystemExit(f"Toolpath not found: {tp}")
    if rs_csv:
        rs = Path(rs_csv)
        if not rs.is_file():
            raise SystemExit(f"RobotStudio CSV not found: {rs}")
        return [(tp, rs)]
    matched = find_matching_rs_csv(
        tp, rs_dir=Path(rs_dir) if rs_dir else _DEFAULT_RS_DIR,
    )
    return [(tp, matched)]
