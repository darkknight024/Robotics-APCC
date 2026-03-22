"""
Scene State — shared communication between FastAPI and Viser servers.

Uses multiprocessing.Queue for cross-process communication.
"""

from multiprocessing import Queue
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# Global scene update queue — created in start.py, shared between processes
scene_update_queue: Optional[Queue] = None


def set_queue(q: Queue):
    """Set the global scene update queue (called during startup)."""
    global scene_update_queue
    scene_update_queue = q


def get_queue() -> Optional[Queue]:
    """Get the global scene update queue."""
    return scene_update_queue


# ---- Scene update command types ----

def cmd_load_robot(urdf_path: str, robot_name: str) -> Dict[str, Any]:
    return {"cmd": "load_robot", "urdf_path": urdf_path, "robot_name": robot_name}


def cmd_set_waypoint(index: int, q: List[float]) -> Dict[str, Any]:
    return {"cmd": "set_waypoint", "index": index, "q": q}


def cmd_draw_trajectory(waypoints: List[List[float]], colors: List[str]) -> Dict[str, Any]:
    return {"cmd": "draw_trajectory", "waypoints": waypoints, "colors": colors}


def cmd_draw_frame(name: str, pos: List[float], wxyz: List[float]) -> Dict[str, Any]:
    return {"cmd": "draw_frame", "name": name, "pos": pos, "wxyz": wxyz}


def cmd_clear_scene() -> Dict[str, Any]:
    return {"cmd": "clear_scene"}


def cmd_show_ecfx_ghosts(solutions: List[Dict]) -> Dict[str, Any]:
    return {"cmd": "show_ecfx_ghosts", "solutions": solutions}
