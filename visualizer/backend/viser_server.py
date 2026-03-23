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
from typing import Optional, Dict, Any, List
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
        self._tcp_marker = None
        self._stored_input_wp: Optional[np.ndarray] = None
        self._stored_input_colors: Optional[List[str]] = None
        self._dense_positions_cache: Optional[np.ndarray] = None
        self._vis_show_input: bool = False
        self._vis_show_dense: bool = True
        self._vis_ecfx_ghosts_enabled: bool = False

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

        self._setup_trajectory_gui()
        self._setup_ecfx_ghost_gui()

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

    def _setup_trajectory_gui(self):
        """Viser checkboxes: toggle input vs TOPP dense path visibility."""
        if self.server is None:
            return
        try:
            with self.server.gui.add_folder("Trajectory"):
                cb_in = self.server.gui.add_checkbox("Show input waypoints", initial_value=False)
                cb_dn = self.server.gui.add_checkbox("Show TOPP (dense) path", initial_value=True)

                @cb_in.on_update
                def _(_):
                    self._vis_show_input = bool(cb_in.value)
                    self._redraw_stored_trajectories()

                @cb_dn.on_update
                def _(_):
                    self._vis_show_dense = bool(cb_dn.value)
                    self._redraw_stored_trajectories()
        except Exception as e:
            print(f"Trajectory GUI: {e}")

    def _setup_ecfx_ghost_gui(self):
        """Local toggle: skip ECFX ghost mesh updates when disabled (performance)."""
        if self.server is None:
            return
        try:
            with self.server.gui.add_folder("ECFX"):
                cb = self.server.gui.add_checkbox("Show ECFX ghost solutions", initial_value=False)

                @cb.on_update
                def _(_):
                    self._vis_ecfx_ghosts_enabled = bool(cb.value)
                    if not self._vis_ecfx_ghosts_enabled:
                        self._clear_ecfx_ghosts_visual()
        except Exception as e:
            print(f"ECFX GUI: {e}")

    def _clear_ecfx_ghosts_visual(self):
        if self.server is None:
            return
        try:
            self.server.scene.remove_by_filter("/ecfx_ghosts")
        except Exception:
            pass

    def _redraw_stored_trajectories(self):
        if self.server is None:
            return
        try:
            self.server.scene.remove_by_filter("/trajectory")
        except Exception:
            pass
        if self._dense_positions_cache is not None and self._vis_show_dense and len(self._dense_positions_cache) >= 2:
            pos = self._dense_positions_cache
            self.server.scene.add_spline_catmull_rom(
                "/trajectory/dense_path",
                pos,
                tension=0.5,
                line_width=3.0,
                color=(0, 255, 0),
            )
        if (
            self._stored_input_wp is not None
            and self._vis_show_input
            and len(self._stored_input_wp) >= 2
        ):
            positions = self._stored_input_wp
            colors = self._stored_input_colors or ["#22c55e"] * len(positions)
            self.server.scene.add_spline_catmull_rom(
                "/trajectory/input_path",
                positions,
                tension=0.5,
                line_width=2.0,
                color=(59, 130, 246),
            )
            color_map = {
                "#22c55e": (34, 197, 94),
                "#eab308": (234, 179, 8),
                "#f97316": (249, 115, 22),
                "#ef4444": (239, 68, 68),
                "#3b82f6": (59, 130, 246),
            }
            pc = np.array([color_map.get(c, (34, 197, 94)) for c in colors], dtype=np.uint8)
            self.server.scene.add_point_cloud(
                "/trajectory/input_points",
                points=positions,
                colors=pc,
                point_size=0.006,
            )

    def _load_feasibility_trajectories_cmd(self, cmd: Dict[str, Any]):
        from visualizer.backend.final_trajectory_csv import load_final_trajectory_csv

        if self.server is None:
            return
        path = cmd.get("dense_csv_path") or ""
        data = load_final_trajectory_csv(path) if path else None
        inp = cmd.get("input_waypoints") or []
        cols = cmd.get("input_colors") or []
        self._vis_show_input = bool(cmd.get("show_input", False))
        self._vis_show_dense = bool(cmd.get("show_dense", True))
        self._stored_input_wp = np.array(inp, dtype=float) if inp else None
        self._stored_input_colors = cols if cols else None
        if data is not None and data.get("position_m") is not None:
            pos_full = np.asarray(data["position_m"], dtype=float)
            if len(pos_full) > 1000:
                step = max(1, len(pos_full) // 1000)
                self._dense_positions_cache = pos_full[::step]
            else:
                self._dense_positions_cache = pos_full
        else:
            self._dense_positions_cache = None
        self._redraw_stored_trajectories()

    def _set_tcp_marker_cmd(self, cmd: Dict[str, Any]):
        if self.server is None:
            return
        pos = cmd.get("pos") or [0, 0, 0]
        wxyz = cmd.get("wxyz") or [1, 0, 0, 0]
        try:
            if self._tcp_marker is None:
                self._tcp_marker = self.server.scene.add_frame(
                    "/tcp_ee_marker",
                    show_axes=True,
                    axes_length=0.12,
                    axes_radius=0.004,
                    position=tuple(float(x) for x in pos),
                    wxyz=tuple(float(x) for x in wxyz),
                )
            else:
                self._tcp_marker.position = tuple(float(x) for x in pos)
                self._tcp_marker.wxyz = tuple(float(x) for x in wxyz)
        except Exception as e:
            print(f"TCP marker: {e}")

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
            self._tcp_marker = None
            self._clear_ecfx_ghosts_visual()

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

    def _clear_ecfx_ghosts_cmd(self, _cmd: Optional[Dict[str, Any]] = None):
        self._clear_ecfx_ghosts_visual()

    def _show_ecfx_ghosts_cmd(self, cmd: Dict[str, Any]):
        """Semi-transparent duplicate URDF poses for alternate IK branches (current waypoint)."""
        if self.server is None or self.current_urdf_model is None:
            return
        if not self._vis_ecfx_ghosts_enabled:
            self._clear_ecfx_ghosts_visual()
            return
        solutions = cmd.get("solutions") or []
        self._clear_ecfx_ghosts_visual()
        if not solutions:
            return
        try:
            urdf_model = self.current_urdf_model
            max_g = 8
            for i, sol in enumerate(solutions[:max_g]):
                q = np.asarray(sol.get("q_rad") or [], dtype=float)
                if q.size == 0:
                    continue
                sel = bool(sol.get("selected"))
                rgba = (0.45, 0.75, 1.0, 0.52) if sel else (0.55, 0.55, 0.62, 0.28)
                ghost = ViserUrdf(
                    self.server,
                    urdf_or_path=urdf_model,
                    root_node_name=f"/ecfx_ghosts/g{i}",
                    mesh_color_override=rgba,
                    load_meshes=True,
                    load_collision_meshes=False,
                )
                ghost.update_cfg(q)
        except Exception as e:
            print(f"ECFX ghosts: {e}")

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
                    self._set_joint_config(cmd.get("q", []))
                elif action == "load_feasibility_trajectories":
                    self._load_feasibility_trajectories_cmd(cmd)
                elif action == "set_tcp_marker":
                    self._set_tcp_marker_cmd(cmd)
                elif action == "set_trajectory_visibility":
                    self._vis_show_input = bool(cmd.get("show_input", False))
                    self._vis_show_dense = bool(cmd.get("show_dense", True))
                    self._redraw_stored_trajectories()
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
                elif action == "show_ecfx_ghosts":
                    self._show_ecfx_ghosts_cmd(cmd)
                elif action == "clear_ecfx_ghosts":
                    self._clear_ecfx_ghosts_cmd(cmd)
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
