"""Convert feasibility / numpy-heavy dicts to JSON-serializable structures."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)
