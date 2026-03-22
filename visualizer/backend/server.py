#!/usr/bin/env python3
"""
Robotics-APCC Live Visualizer — FastAPI Backend Server

Runs on port 8080. Provides REST API for the React frontend.
Communicates with the Viser 3D server via scene_state queue.
"""

import asyncio
import queue
import sys
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---- Add project root to sys.path so we can import existing modules ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_robots_config, load_knife_config
from utils.csv_loader_toolpath import load_toolpath_trajectories_ext
from utils.transform_handler import transform_trajectories_to_base_frame
from utils.urdf_loader import load_robot_model_eaik, resolve_urdf_path

from visualizer.backend import session_manager as sm
from visualizer.backend.data_detection import (
    merge_column_map,
    sniff_csv,
    waypoint_counts_from_toolpath,
    write_normalized_toolpath_csv,
)
from visualizer.backend.pipeline_runner import (
    load_run_result,
    run_fk_pipeline,
    run_ik_pipeline,
    save_run_result,
    update_last_job_id,
)
from visualizer.backend.scene_state import (
    cmd_clear_trajectory_preview,
    cmd_draw_trajectory,
    cmd_load_robot,
    cmd_set_waypoint,
)

# ---- App Setup ----

app = FastAPI(title="Robotics-APCC Visualizer API", version="0.3.0")
_executor = ThreadPoolExecutor(max_workers=4)
_job_async_queues: Dict[str, asyncio.Queue] = {}

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

ROBOTS_CONFIG = str(PROJECT_ROOT / "config" / "robots_config.yaml")
KNIFE_CONFIG = str(PROJECT_ROOT / "config" / "knife_config.yaml")


# ---- Scene Queue (set by start.py) ----

_scene_queue = None


def set_scene_queue(q):
    global _scene_queue
    _scene_queue = q


# ---- Endpoints ----

@app.get("/api/health")
async def health():
    return ok_response({"status": "running", "version": "0.3.0"})


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


def _resolve_urdf(urdf_path: str) -> str:
    if not os.path.isabs(urdf_path):
        urdf_path = str(PROJECT_ROOT / urdf_path)
    resolved = resolve_urdf_path(urdf_path)
    return str(resolved)


def _robot_joint_metadata(urdf_path: str) -> Dict[str, Any]:
    rm = load_robot_model_eaik(urdf_path, ee_frame_name="ee_link")
    names = list(rm.joint_names)
    lows = np.rad2deg(rm.lower_position_limit).tolist()
    highs = np.rad2deg(rm.upper_position_limit).tolist()
    limits = [[lows[i], highs[i]] for i in range(len(names))]
    return {
        "joint_names": names,
        "joint_limits_deg": limits,
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Save uploaded CSV and create a session."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return error_response("Please upload a .csv file.")

    sm.cleanup_old_sessions(PROJECT_ROOT)
    session_dir = sm.create_session(PROJECT_ROOT)
    session_id = session_dir.name

    raw = await file.read()
    max_bytes = 50 * 1024 * 1024
    if len(raw) > max_bytes:
        return error_response("File too large (max 50 MB).")

    dest = session_dir / sm.ORIGINAL_CSV
    dest.write_bytes(raw)

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    preview_lines = lines[:21]
    raw_columns: List[str] = []
    preview_rows: List[List[str]] = []
    import csv as _csv
    for i, row in enumerate(_csv.reader(preview_lines)):
        if i == 0:
            raw_columns = [c.strip() for c in row]
        preview_rows.append([c.strip() for c in row])
        if i >= 20:
            break

    meta = sm.read_metadata(session_dir)
    meta["active_csv"] = sm.ORIGINAL_CSV
    sm.write_metadata(session_dir, meta)

    return ok_response({
        "session_id": session_id,
        "num_rows": len(lines),
        "raw_columns": raw_columns,
        "preview_rows": preview_rows,
    })


class DetectRequest(BaseModel):
    column_map: Optional[Dict[str, str]] = None


@app.post("/api/detect/{session_id}")
async def detect_columns(session_id: str, body: DetectRequest = DetectRequest()):
    """Run column detection; optional user column_map writes normalized.csv."""
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")

    original = session_dir / sm.ORIGINAL_CSV
    if not original.exists():
        return error_response("No uploaded CSV in session.")

    column_map = body.column_map or {}

    sniff = sniff_csv(original)
    merged = merge_column_map(sniff.detected_columns, column_map)

    unknown_remaining = [u for u in sniff.unknown_columns if u not in column_map]

    normalized_path = session_dir / sm.NORMALIZED_CSV
    warnings: List[str] = list(sniff.warnings)

    if column_map:
        ok, w = write_normalized_toolpath_csv(original, normalized_path, merged)
        warnings.extend(w)
        if ok:
            sm.update_metadata(session_dir, active_csv=sm.NORMALIZED_CSV, column_map=column_map)
            active = normalized_path
        else:
            active = original
            warnings.append("Normalization failed; using original file.")
    else:
        active = original
        if normalized_path.exists():
            normalized_path.unlink()
        sm.update_metadata(session_dir, active_csv=sm.ORIGINAL_CSV)

    wp_counts, load_err = waypoint_counts_from_toolpath(active)
    if load_err:
        warnings.append(load_err)

    num_traj = len(wp_counts) if wp_counts else 0
    if num_traj == 0 and sniff.has_task_space:
        num_traj = 1

    detection_payload = {
        "has_task_space": sniff.has_task_space,
        "has_joint_space": sniff.has_joint_space,
        "num_trajectories": num_traj,
        "num_waypoints_per_trajectory": wp_counts,
        "detected_columns": merged,
        "unknown_columns": unknown_remaining,
        "warnings": warnings,
    }

    meta = sm.read_metadata(session_dir)
    meta["last_detection"] = detection_payload
    sm.write_metadata(session_dir, meta)

    return ok_response(detection_payload)


class ConfigureRequest(BaseModel):
    use_base_frame: bool = True
    knife_name: Optional[str] = None
    robot_name: str
    trajectory_index: int = 0


@app.post("/api/configure/{session_id}")
async def configure_session(session_id: str, req: ConfigureRequest):
    """Transform toolpath if needed, load robot, preview trajectory in Viser."""
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")

    meta = sm.read_metadata(session_dir)
    active_name = meta.get("active_csv", sm.ORIGINAL_CSV)
    csv_path = session_dir / active_name
    if not csv_path.exists():
        csv_path = session_dir / sm.ORIGINAL_CSV
    if not csv_path.exists():
        return error_response("No CSV file in session.")

    last = meta.get("last_detection")
    if not last:
        return error_response("Run POST /api/detect for this session first.")
    has_task = bool(last.get("has_task_space"))

    robots_db = load_robots_config(ROBOTS_CONFIG)
    if req.robot_name not in robots_db:
        return error_response(f"Unknown robot: {req.robot_name}")

    robot = robots_db[req.robot_name]
    urdf_path = _resolve_urdf(robot.urdf_path)

    if _scene_queue is not None:
        _scene_queue.put(cmd_load_robot(urdf_path, req.robot_name))

    preview_xyz: List[List[float]] = []
    traj_idx = max(0, req.trajectory_index)

    if has_task:
        try:
            result = load_toolpath_trajectories_ext(str(csv_path))
            trajs = result.trajectories
            if not trajs:
                if _scene_queue is not None:
                    _scene_queue.put(cmd_clear_trajectory_preview())
            elif trajs:
                t = trajs[min(traj_idx, len(trajs) - 1)]
                trajectories = [t]
                if not req.use_base_frame:
                    if not req.knife_name:
                        return error_response("knife_name required when not using base frame.")
                    knives = load_knife_config(KNIFE_CONFIG)
                    if req.knife_name not in knives:
                        return error_response(f"Unknown knife: {req.knife_name}")
                    kp = knives[req.knife_name]
                    trajectories = transform_trajectories_to_base_frame(
                        [t],
                        kp.translation_m,
                        kp.quaternion,
                    )
                t_vis = trajectories[0]
                preview_xyz = t_vis[:, :3].tolist()
                n = len(preview_xyz)
                colors = ["#22c55e"] * n
                if _scene_queue is not None:
                    _scene_queue.put(cmd_draw_trajectory(preview_xyz, colors))
        except Exception as e:
            if _scene_queue is not None:
                _scene_queue.put(cmd_clear_trajectory_preview())
            return error_response(f"Could not load toolpath for preview: {e}")
    else:
        if _scene_queue is not None:
            _scene_queue.put(cmd_clear_trajectory_preview())

    try:
        jmeta = _robot_joint_metadata(urdf_path)
    except Exception as e:
        jmeta = {"joint_names": [], "joint_limits_deg": [], "error": str(e)}

    robot_payload = {
        "name": robot.name,
        "urdf_path": urdf_path,
        "reach_mm": robot.reach_m * 1000.0,
        "velocity_limits_rad_s": robot.velocity_limits_rad_s or [],
        **jmeta,
    }

    sm.update_metadata(
        session_dir,
        use_base_frame=req.use_base_frame,
        knife_name=req.knife_name,
        robot_name=req.robot_name,
        urdf_path=urdf_path,
        trajectory_index=req.trajectory_index,
    )

    return ok_response({
        "transformed_waypoints_preview": preview_xyz,
        "robot_metadata": robot_payload,
    })


def _push_result_to_viser(result: Dict[str, Any]) -> None:
    if _scene_queue is None:
        return
    xyz = result.get("tcp_xyz") or []
    colors = result.get("waypoint_colors_hex") or []
    if len(xyz) == len(colors) and xyz:
        _scene_queue.put(cmd_draw_trajectory(xyz, colors))


async def _execute_ik_job(
    session_id: str,
    job_id: str,
    body: "RunIKRequest",
    out_q: asyncio.Queue,
) -> None:
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        await out_q.put({"type": "error", "message": "Session not found"})
        return
    meta = sm.read_metadata(session_dir)
    urdf = meta.get("urdf_path")
    if not urdf or not os.path.isfile(urdf):
        robots_db = load_robots_config(ROBOTS_CONFIG)
        rn = meta.get("robot_name")
        if rn and rn in robots_db:
            urdf = _resolve_urdf(robots_db[rn].urdf_path)
    if not urdf:
        await out_q.put({"type": "error", "message": "Missing urdf_path; run POST /api/configure first."})
        return

    loop = asyncio.get_event_loop()
    pq: queue.Queue = queue.Queue()

    def worker():
        try:
            return run_ik_pipeline(
                session_dir,
                meta,
                urdf,
                body.solver,
                body.ee_frame_name,
                body.trajectory_index,
                progress=lambda m: pq.put(m),
            )
        except Exception as e:
            pq.put({"type": "error", "message": str(e)})
            return None

    fut = loop.run_in_executor(_executor, worker)  # concurrent.futures.Future

    while not fut.done():
        await asyncio.sleep(0.03)
        while True:
            try:
                m = pq.get_nowait()
                if m.get("type") == "error":
                    await out_q.put(m)
                    await asyncio.wrap_future(fut)
                    return
                await out_q.put(m)
            except queue.Empty:
                break

    while True:
        try:
            m = pq.get_nowait()
            await out_q.put(m)
        except queue.Empty:
            break

    try:
        result = await asyncio.wrap_future(fut)
    except Exception as e:
        await out_q.put({"type": "error", "message": str(e)})
        return

    if result is None:
        await out_q.put({"type": "error", "message": "IK run failed"})
        return

    save_run_result(session_dir, job_id, result)
    update_last_job_id(session_dir, job_id)
    _push_result_to_viser(result)
    await out_q.put({"type": "done", "result": result})


async def _execute_fk_job(
    session_id: str,
    job_id: str,
    body: "RunFKRequest",
    out_q: asyncio.Queue,
) -> None:
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        await out_q.put({"type": "error", "message": "Session not found"})
        return
    meta = sm.read_metadata(session_dir)
    urdf = meta.get("urdf_path")
    if not urdf or not os.path.isfile(urdf):
        robots_db = load_robots_config(ROBOTS_CONFIG)
        rn = meta.get("robot_name")
        if rn and rn in robots_db:
            urdf = _resolve_urdf(robots_db[rn].urdf_path)
    if not urdf:
        await out_q.put({"type": "error", "message": "Missing urdf_path; run POST /api/configure first."})
        return

    loop = asyncio.get_event_loop()
    pq: queue.Queue = queue.Queue()

    def worker():
        try:
            return run_fk_pipeline(
                session_dir,
                meta,
                urdf,
                body.solver,
                body.ee_frame_name,
                body.trajectory_index,
                progress=lambda m: pq.put(m),
            )
        except Exception as e:
            pq.put({"type": "error", "message": str(e)})
            return None

    fut = loop.run_in_executor(_executor, worker)

    while not fut.done():
        await asyncio.sleep(0.03)
        while True:
            try:
                m = pq.get_nowait()
                if m.get("type") == "error":
                    await out_q.put(m)
                    await asyncio.wrap_future(fut)
                    return
                await out_q.put(m)
            except queue.Empty:
                break

    while True:
        try:
            m = pq.get_nowait()
            await out_q.put(m)
        except queue.Empty:
            break

    try:
        result = await asyncio.wrap_future(fut)
    except Exception as e:
        await out_q.put({"type": "error", "message": str(e)})
        return

    if result is None:
        await out_q.put({"type": "error", "message": "FK run failed"})
        return

    save_run_result(session_dir, job_id, result)
    update_last_job_id(session_dir, job_id)
    _push_result_to_viser(result)
    await out_q.put({"type": "done", "result": result})


class RunIKRequest(BaseModel):
    solver: str = "eaik"
    ee_frame_name: str = "ee_link"
    trajectory_index: int = 0


class RunFKRequest(BaseModel):
    solver: str = "eaik"
    ee_frame_name: str = "ee_link"
    trajectory_index: int = 0


class TimelineBody(BaseModel):
    index: int = 0


@app.post("/api/run-ik/{session_id}")
async def post_run_ik(session_id: str, body: RunIKRequest):
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")
    meta = sm.read_metadata(session_dir)
    last = meta.get("last_detection") or {}
    if not last.get("has_task_space"):
        return error_response("Session has no task-space data for IK.")

    job_id = str(uuid.uuid4())
    out_q: asyncio.Queue = asyncio.Queue()
    _job_async_queues[job_id] = out_q
    asyncio.create_task(_execute_ik_job(session_id, job_id, body, out_q))
    return ok_response({"job_id": job_id})


@app.post("/api/run-fk/{session_id}")
async def post_run_fk(session_id: str, body: RunFKRequest):
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")
    meta = sm.read_metadata(session_dir)
    last = meta.get("last_detection") or {}
    if not last.get("has_joint_space"):
        return error_response("Session has no joint-space data for FK.")

    job_id = str(uuid.uuid4())
    out_q: asyncio.Queue = asyncio.Queue()
    _job_async_queues[job_id] = out_q
    asyncio.create_task(_execute_fk_job(session_id, job_id, body, out_q))
    return ok_response({"job_id": job_id})


@app.get("/api/results/{session_id}/{job_id}")
async def get_results(session_id: str, job_id: str):
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")
    data = load_run_result(session_dir, job_id)
    if not data:
        return error_response("Job not found.")
    return ok_response(data)


@app.websocket("/ws/stream/{session_id}/{job_id}")
async def ws_stream(websocket: WebSocket, session_id: str, job_id: str):
    await websocket.accept()
    q = _job_async_queues.get(job_id)
    if q is None:
        await websocket.close(code=1008)
        return
    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        _job_async_queues.pop(job_id, None)


@app.post("/api/session/{session_id}/timeline")
async def post_timeline(session_id: str, body: TimelineBody):
    session_dir = sm.session_path(PROJECT_ROOT, session_id)
    if session_dir is None:
        return error_response("Session not found.")
    meta = sm.read_metadata(session_dir)
    jid = meta.get("last_job_id")
    if not jid:
        return error_response("No completed run in this session.")
    data = load_run_result(session_dir, jid)
    if not data or not data.get("result"):
        return error_response("No result data.")
    result = data["result"]
    joints = result.get("joints_rad") or []
    n = len(joints)
    if n == 0:
        return error_response("Empty trajectory.")
    idx = max(0, min(int(body.index), n - 1))
    q = joints[idx]
    if _scene_queue is not None:
        _scene_queue.put(cmd_set_waypoint(idx, list(map(float, q))))
    return ok_response({"index": idx, "n_waypoints": n})


@app.post("/api/load-robot")
async def load_robot(req: LoadRobotRequest):
    """Load a robot URDF into the Viser 3D scene."""
    try:
        robots_db = load_robots_config(ROBOTS_CONFIG)
        if req.robot_name not in robots_db:
            return error_response(f"Robot '{req.robot_name}' not found. Available: {list(robots_db.keys())}")

        robot = robots_db[req.robot_name]
        urdf_path = _resolve_urdf(robot.urdf_path)

        if not os.path.exists(urdf_path):
            return error_response(f"URDF file not found: {urdf_path}")

        if _scene_queue is not None:
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
    sm.cleanup_old_sessions(PROJECT_ROOT)
    if scene_queue is not None:
        set_scene_queue(scene_queue)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_server()
