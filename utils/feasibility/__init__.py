"""Feasibility pipeline orchestration (typed inputs, runner, plots, reports).

Import submodules explicitly, e.g. ``from utils.feasibility.pipeline_runner import run_feasibility_pipeline``,
to avoid loading matplotlib until needed.
"""

from utils.feasibility.pipeline_types import FeasibilityPipelineInputs, PipelineRuntimeContext

__all__ = [
    "FeasibilityPipelineInputs",
    "PipelineRuntimeContext",
]
