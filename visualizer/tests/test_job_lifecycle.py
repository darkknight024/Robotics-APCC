"""
TestClient checks for Phase 3 job endpoints (mocked IK to avoid heavy kinematics).

Uses httpx ASGITransport so asyncio.create_task background jobs complete (sync
TestClient does not advance the loop enough for executor-backed jobs).

Run: python -m unittest visualizer.tests.test_job_lifecycle -v
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from httpx import ASGITransport

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualizer.backend import session_manager as sm
from visualizer.backend.pipeline_runner import save_run_result
from visualizer.backend.server import app

_FAKE_IK_RESULT = {
    "kind": "ik",
    "solver": "eaik",
    "ee_frame_name": "ee_link",
    "trajectory_index": 0,
    "n_waypoints": 1,
    "joints_rad": [[0.0] * 6],
    "joints_deg": [[0.0] * 6],
    "tcp_xyz": [[0.0, 0.0, 0.1]],
    "tcp_quat": [[1.0, 0.0, 0.0, 0.0]],
    "ik_success": [True],
    "waypoint_colors_hex": ["#22c55e"],
}

_FAKE_FEASIBILITY_RESULT = {
    "kind": "feasibility",
    "num_trajectories": 2,
    "trajectory_results": [
        {
            "trajectory_index": 0,
            "num_waypoints": 2,
            "reachable_flags": [True, True],
            "tcp_xyz_m": [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]],
            "joint_angles_deg": [[0.0] * 6, [1.0] * 6],
            "manipulability": [1.0, 0.9],
            "min_singular_value": [0.5, 0.4],
            "condition_number": [2.0, 2.5],
            "near_singularity": [False, True],
            "joint_space_distances": [0.1],
            "per_joint_jumps": [[0.01] * 6],
            "c0_segment_violation": [False],
            "topp_series": {
                "t_samples_s": [0.0, 0.1],
                "q_rad": [[0.0] * 6, [0.1] * 6],
                "qdot_rad_s": [[0.0] * 6, [0.0] * 6],
                "qddot_rad_s2": [[0.0] * 6, [0.0] * 6],
            },
        },
        {
            "trajectory_index": 1,
            "num_waypoints": 2,
            "reachable_flags": [True, False],
            "tcp_xyz_m": [[0.2, 0.0, 0.1], [0.3, 0.0, 0.1]],
            "joint_angles_deg": [[0.0] * 6, [2.0] * 6],
            "manipulability": [0.8, 0.7],
            "min_singular_value": [0.3, 0.2],
            "condition_number": [3.0, 3.5],
            "near_singularity": [True, False],
            "joint_space_distances": [0.05],
            "per_joint_jumps": [[0.02] * 6],
            "c0_segment_violation": [True],
            "topp_series": {
                "t_samples_s": [0.0, 0.05],
                "q_rad": [[0.0] * 6, [0.2] * 6],
                "qdot_rad_s": [[0.0] * 6, [0.0] * 6],
                "qddot_rad_s2": [[0.0] * 6, [0.0] * 6],
            },
        },
    ],
}


class TestVisualizerJobLifecycle(unittest.TestCase):
    def setUp(self):
        self.session_dir = sm.create_session(PROJECT_ROOT)
        self.session_id = sm.read_metadata(self.session_dir)["session_id"]

    def tearDown(self):
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def test_run_ik_unknown_session(self):
        async def _run():
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.post(
                    "/api/run-ik/00000000-0000-0000-0000-000000000000",
                    json={"solver": "eaik", "ee_frame_name": "ee_link", "trajectory_index": 0},
                )
                self.assertFalse(r.json()["ok"])

        asyncio.run(_run())

    def test_run_ik_returns_job_id_and_result_file(self):
        async def _run():
            sm.update_metadata(
                self.session_dir,
                last_detection={"has_task_space": True, "has_joint_space": False},
                robot_name="IRB 1300-7/1.4",
            )
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with patch("visualizer.backend.server.run_ik_pipeline", return_value=_FAKE_IK_RESULT):
                    r = await client.post(
                        f"/api/run-ik/{self.session_id}",
                        json={"solver": "eaik", "ee_frame_name": "ee_link", "trajectory_index": 0},
                    )
                    self.assertTrue(r.json()["ok"], r.text)
                    job_id = r.json()["data"]["job_id"]
                    self.assertTrue(job_id)

                    for _ in range(80):
                        await asyncio.sleep(0.05)
                        gr = await client.get(f"/api/results/{self.session_id}/{job_id}")
                        body = gr.json()
                        if body.get("ok") and body.get("data", {}).get("result"):
                            data = body["data"]
                            self.assertEqual(data["result"]["n_waypoints"], 1)
                            self.assertEqual(data["status"], "done")
                            return
                    self.fail("Timed out waiting for mocked IK result")

        asyncio.run(_run())

    def test_run_feasibility_returns_job_id_and_result_file(self):
        async def _run():
            sm.update_metadata(
                self.session_dir,
                last_detection={"has_task_space": True, "has_joint_space": False},
                robot_name="IRB 1300-7/1.4",
            )
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with patch(
                    "visualizer.backend.server.run_feasibility_pipeline",
                    return_value=_FAKE_FEASIBILITY_RESULT,
                ):
                    r = await client.post(
                        f"/api/run-feasibility/{self.session_id}",
                        json={"speed_mm_s": 100.0, "config": {}},
                    )
                    self.assertTrue(r.json()["ok"], r.text)
                    job_id = r.json()["data"]["job_id"]
                    self.assertTrue(job_id)

                    for _ in range(80):
                        await asyncio.sleep(0.05)
                        gr = await client.get(f"/api/results/{self.session_id}/{job_id}")
                        body = gr.json()
                        if body.get("ok") and body.get("data", {}).get("result"):
                            data = body["data"]
                            self.assertEqual(data["result"]["kind"], "feasibility")
                            self.assertEqual(len(data["result"]["trajectory_results"]), 2)
                            self.assertEqual(data["status"], "done")
                            return
                    self.fail("Timed out waiting for mocked feasibility result")

        asyncio.run(_run())

    def test_feasibility_scene_switches_trajectory(self):
        async def _run():
            job_id = "feas-test-job"
            save_run_result(self.session_dir, job_id, _FAKE_FEASIBILITY_RESULT)
            sm.update_metadata(self.session_dir, last_job_id=job_id)
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.post(
                    f"/api/session/{self.session_id}/feasibility-scene",
                    json={"trajectory_index": 1},
                )
                self.assertTrue(r.json()["ok"], r.text)
                self.assertEqual(r.json()["data"]["trajectory_index"], 1)
                self.assertEqual(r.json()["data"]["n_points"], 2)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
