#!/usr/bin/env python3
"""
Robotics-APCC Live Visualizer — Viser 3D Scene Server

Runs on port 8081. Manages the 3D robot scene using Viser.
Receives scene update commands from the FastAPI server via multiprocessing.Queue.
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import Queue

import numpy as np

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import viser
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation


class ViserSceneServer:
    """Manages the Viser 3D scene for robot visualization."""

    def __init__(self, port: int = 8081, scene_queue: Optional[Queue] = None):
        self.port = port
        self.scene_queue = scene_queue
        self.server: Optional[viser.ViserServer] = None
        self.current_urdf: Optional[ViserUrdf] = None
        self.current_urdf_model: Optional[yourdfpy.URDF] = None
        self.current_robot_name: str = ""

    def start(self):
        """Start the Viser server and begin processing commands."""
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.server.gui.configure_theme(dark_mode=True)
        print(f"Viser 3D server started on port {self.port}")

        # Initial scene setup
        self._setup_scene()

        # Load default robot
        self._load_default_robot()

        # Start command processing thread
        if self.scene_queue is not None:
            cmd_thread = threading.Thread(target=self._process_commands, daemon=True)
            cmd_thread.start()

        # Keep alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Viser server shutting down...")

    def _setup_scene(self):
        """Set up the default scene with world frame (no floor grid)."""
        if self.server is None:
            return

        self.server.scene.add_frame("world_frame", show_axes=True,
                                     axes_length=0.15, axes_radius=0.004)

    def _load_default_robot(self):
        """Load the first robot from config as default."""
        try:
            from utils.config_loader import load_robots_config
            robots_config_path = os.path.join(PROJECT_ROOT, "config", "robots_config.yaml")
            robots = load_robots_config(robots_config_path)

            if robots:
                first_robot = next(iter(robots.values()))
                urdf_path = first_robot.urdf_path
                if not os.path.isabs(urdf_path):
                    urdf_path = os.path.join(PROJECT_ROOT, urdf_path)
                self._load_robot(urdf_path, first_robot.name)
        except Exception as e:
            print(f"Warning: Could not load default robot: {e}")

    def _load_robot(self, urdf_path: str, robot_name: str):
        """Load a robot URDF into the scene."""
        if self.server is None:
            return

        print(f"Loading robot: {robot_name} from {urdf_path}")

        # Clear previous robot
        if self.current_urdf is not None:
            try:
                # Remove old robot meshes from scene
                self.server.scene.remove_by_filter(f"/robot")
            except Exception:
                pass

        try:
            # Reset scene and re-add base elements
            self.server.scene.reset()
            self._setup_scene()

            # Load URDF
            urdf_model = yourdfpy.URDF.load(
                urdf_path,
                build_scene_graph=True,
                load_meshes=True,
                load_collision_meshes=False,
            )

            # Create ViserUrdf handle
            viser_urdf = ViserUrdf(
                self.server,
                urdf_or_path=urdf_model,
                root_node_name="/robot",
            )

            # Set to home position (all zeros)
            num_joints = len(list(viser_urdf.get_actuated_joint_limits().keys()))
            home_cfg = np.zeros(num_joints)
            viser_urdf.update_cfg(home_cfg)

            self.current_urdf = viser_urdf
            self.current_urdf_model = urdf_model
            self.current_robot_name = robot_name

            print(f"Robot '{robot_name}' loaded successfully ({num_joints} joints)")

        except Exception as e:
            print(f"Error loading robot '{robot_name}': {e}")
            import traceback
            traceback.print_exc()

    def _set_joint_config(self, q: list):
        """Update robot joint configuration."""
        if self.current_urdf is None:
            return
        try:
            q_arr = np.array(q, dtype=float)
            self.current_urdf.update_cfg(q_arr)
            if self.current_urdf_model is not None:
                self.current_urdf_model.update_cfg(q_arr)
        except Exception as e:
            print(f"Error setting joint config: {e}")

    def _clear_trajectory_preview(self):
        """Remove prior trajectory geometry so re-configure does not stack duplicates."""
        if self.server is None:
            return
        try:
            self.server.scene.remove_by_filter("/trajectory")
        except Exception:
            pass

    def _draw_trajectory(self, waypoints: list, colors: list):
        """Draw trajectory waypoints as a point cloud + spline."""
        if self.server is None or not waypoints:
            return
        try:
            self._clear_trajectory_preview()
            positions = np.array(waypoints, dtype=float)
            # Draw as spline
            self.server.scene.add_spline_catmull_rom(
                "/trajectory/path",
                positions,
                tension=0.5,
                line_width=2.0,
                color=(59, 130, 246),  # accent blue
            )
            # Draw as point cloud
            color_map = {
                '#22c55e': (34, 197, 94),
                '#eab308': (234, 179, 8),
                '#f97316': (249, 115, 22),
                '#ef4444': (239, 68, 68),
                '#3b82f6': (59, 130, 246),
                '#a855f7': (168, 85, 247),
                '#6b7280': (107, 114, 128),
            }
            point_colors = np.array([
                color_map.get(c, (34, 197, 94)) for c in colors
            ], dtype=np.uint8)
            self.server.scene.add_point_cloud(
                "/trajectory/points",
                points=positions,
                colors=point_colors,
                point_size=0.008,
            )
        except Exception as e:
            print(f"Error drawing trajectory: {e}")

    def _process_commands(self):
        """Process scene update commands from the queue."""
        while True:
            try:
                cmd = self.scene_queue.get(timeout=0.5)
                if cmd is None:
                    continue

                action = cmd.get("cmd")

                if action == "load_robot":
                    self._load_robot(cmd["urdf_path"], cmd["robot_name"])
                elif action == "set_waypoint":
                    self._set_joint_config(cmd["q"])
                elif action == "draw_trajectory":
                    self._draw_trajectory(cmd["waypoints"], cmd.get("colors", []))
                elif action == "clear_scene":
                    if self.server:
                        self.server.scene.reset()
                        self._setup_scene()
                elif action == "draw_frame":
                    if self.server:
                        self.server.scene.add_frame(
                            f"/markers/{cmd['name']}",
                            show_axes=True,
                            axes_length=0.08,
                            axes_radius=0.003,
                            position=tuple(cmd["pos"]),
                            wxyz=tuple(cmd["wxyz"]),
                        )
                elif action == "set_joint_config":
                    self._set_joint_config(cmd["q"])
                elif action == "clear_trajectory_preview":
                    self._clear_trajectory_preview()
                else:
                    print(f"Unknown scene command: {action}")

            except Exception:
                # Queue.get timeout — normal, just continue
                pass


def run_viser_server(scene_queue: Optional[Queue] = None, port: int = 8081):
    """Entry point for the Viser server (called from start.py)."""
    server = ViserSceneServer(port=port, scene_queue=scene_queue)
    server.start()


if __name__ == "__main__":
    run_viser_server()
