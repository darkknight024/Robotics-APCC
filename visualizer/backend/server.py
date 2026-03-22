#!/usr/bin/env python3
"""
Robotics-APCC Live Visualizer — FastAPI Backend Server

Runs on port 8080. Provides REST API for the React frontend.
Communicates with the Viser 3D server via scene_state queue.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---- Add project root to sys.path so we can import existing modules ----
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_robots_config, load_knife_config


# ---- App Setup ----

app = FastAPI(title="Robotics-APCC Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Response Helpers ----

def ok_response(data: Any = None) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def error_response(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


# ---- Config Paths ----

ROBOTS_CONFIG = os.path.join(PROJECT_ROOT, "config", "robots_config.yaml")
KNIFE_CONFIG = os.path.join(PROJECT_ROOT, "config", "knife_config.yaml")


# ---- Scene Queue (set by start.py) ----

_scene_queue = None

def set_scene_queue(q):
    global _scene_queue
    _scene_queue = q


# ---- Endpoints ----

@app.get("/api/health")
async def health():
    return ok_response({"status": "running", "version": "0.1.0"})


@app.get("/api/robots")
async def get_robots():
    """Return all robots from robots_config.yaml."""
    try:
        robots_db = load_robots_config(ROBOTS_CONFIG)
        robots_list = []
        for name, rc in robots_db.items():
            robots_list.append({
                "name": rc.name,
                "description": getattr(rc, 'description', rc.name),
                "urdf_path": rc.urdf_path,
                "reach_m": rc.reach_m,
                "payload_kg": getattr(rc, 'payload_kg', 0),
                "velocity_limits_rad_s": rc.velocity_limits_rad_s or [],
                "acceleration_limits_rad_s2": rc.acceleration_limits_rad_s2 or [],
            })
        return ok_response(robots_list)
    except Exception as e:
        return error_response(str(e))


@app.get("/api/knives")
async def get_knives():
    """Return all knife poses from knife_config.yaml."""
    try:
        knives_db = load_knife_config(KNIFE_CONFIG)
        knives_list = []
        for name, kp in knives_db.items():
            knives_list.append({
                "name": kp.name,
                "description": kp.description,
                "translation_mm": (kp.translation_m * 1000).tolist(),
                "quaternion": kp.quaternion.tolist(),
            })
        return ok_response(knives_list)
    except Exception as e:
        return error_response(str(e))


class LoadRobotRequest(BaseModel):
    robot_name: str


@app.post("/api/load-robot")
async def load_robot(req: LoadRobotRequest):
    """Load a robot URDF into the Viser 3D scene."""
    try:
        robots_db = load_robots_config(ROBOTS_CONFIG)
        if req.robot_name not in robots_db:
            return error_response(f"Robot '{req.robot_name}' not found. Available: {list(robots_db.keys())}")

        robot = robots_db[req.robot_name]
        urdf_path = robot.urdf_path

        # Make path absolute if relative
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(PROJECT_ROOT, urdf_path)

        if not os.path.exists(urdf_path):
            return error_response(f"URDF file not found: {urdf_path}")

        # Send load command to Viser server
        if _scene_queue is not None:
            from visualizer.backend.scene_state import cmd_load_robot
            _scene_queue.put(cmd_load_robot(urdf_path, req.robot_name))

        return ok_response({
            "robot_name": req.robot_name,
            "urdf_path": urdf_path,
            "loaded": True,
        })
    except Exception as e:
        return error_response(str(e))


# ---- Standalone run (used by start.py) ----

def run_server(scene_queue=None, port=8080):
    """Run the FastAPI server (called from start.py)."""
    if scene_queue is not None:
        set_scene_queue(scene_queue)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_server()
