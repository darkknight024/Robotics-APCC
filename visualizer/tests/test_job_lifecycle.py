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


if __name__ == "__main__":
    unittest.main()
