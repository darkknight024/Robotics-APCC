"""Merge API JSON overrides into FeasibilityConfig (defaults from batch YAML)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from utils.config_loader import FeasibilityConfig, load_batch_config


def _merge_dataclass(obj: Any, patch: Dict[str, Any]) -> None:
    for k, v in patch.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if isinstance(v, dict) and cur is not None and not isinstance(cur, (str, int, float, bool)):
            if hasattr(cur, "__dataclass_fields__"):
                _merge_dataclass(cur, v)
            else:
                setattr(obj, k, v)
        else:
            setattr(obj, k, v)


def build_feasibility_config(
    project_root: str,
    overrides: Dict[str, Any] | None = None,
) -> FeasibilityConfig:
    """Load `config/batch_feasibility_config.yaml` and apply optional nested overrides."""
    path = str(Path(project_root) / "config" / "batch_feasibility_config.yaml")
    cfg = deepcopy(load_batch_config(path))

    if overrides:
        if "use_base_frame" in overrides:
            cfg.use_base_frame = bool(overrides["use_base_frame"])
        if "solver" in overrides:
            cfg.solver = str(overrides["solver"])
        if "max_ik_failures_per_trajectory" in overrides:
            cfg.max_ik_failures_per_trajectory = int(overrides["max_ik_failures_per_trajectory"])

        if "output" in overrides and isinstance(overrides["output"], dict):
            _merge_dataclass(cfg.output, overrides["output"])
        if "singularity" in overrides and isinstance(overrides["singularity"], dict):
            _merge_dataclass(cfg.singularity, overrides["singularity"])
        if "manipulability" in overrides and isinstance(overrides["manipulability"], dict):
            _merge_dataclass(cfg.manipulability, overrides["manipulability"])
        if "continuity" in overrides and isinstance(overrides["continuity"], dict):
            _merge_dataclass(cfg.continuity, overrides["continuity"])
        if "waypoint_density" in overrides and isinstance(overrides["waypoint_density"], dict):
            _merge_dataclass(cfg.waypoint_density, overrides["waypoint_density"])
        if "topp_ra" in overrides and isinstance(overrides["topp_ra"], dict):
            _merge_dataclass(cfg.topp_ra, overrides["topp_ra"])
        if "reachability" in overrides and isinstance(overrides["reachability"], dict):
            _merge_dataclass(cfg.reachability, overrides["reachability"])
        if "eaik_multi_solution" in overrides and isinstance(overrides["eaik_multi_solution"], dict):
            _merge_dataclass(cfg.eaik_multi_solution, overrides["eaik_multi_solution"])
        if "ranking" in overrides and isinstance(overrides["ranking"], dict):
            _merge_dataclass(cfg.ranking, overrides["ranking"])
        if "save_analysis" in overrides:
            cfg.output.save_analysis = bool(overrides["save_analysis"])

    # Phase 4 visualizer: skip slow PNG generation; data comes from return dict + JSON
    cfg.reachability.generate_graphs = False
    cfg.singularity.generate_graphs = False
    cfg.manipulability.generate_graphs = False
    cfg.continuity.generate_graphs = False
    cfg.topp_ra.generate_graphs = False
    cfg.waypoint_density.generate_graphs = False
    cfg.eaik_multi_solution.generate_graphs = False

    return cfg
