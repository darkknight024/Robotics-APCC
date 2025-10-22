# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ABB IRB-1300 IK Controller with Record and Playback functionality.

This script allows you to:
- Record end-effector trajectories using teleop control
- Play back trajectories with automatic coordinate conversion
- Analyze playback speed and generate reports

Usage:
    # Record mode (use teleop to control, press ESC to save and exit)
    python irb_1300_example.py --mode r

    # Playback mode - DEFAULT: Auto-convert from tool frame to robot base frame
    python irb_1300_example.py --mode p --csv tool_trajectory.csv
    
    # Playback mode - With --relative: Use CSV as-is (already in robot base frame)
    python irb_1300_example.py --mode p --csv recorded_trajectory.csv --relative

Coordinate Frames:
    - DEFAULT (no --relative): CSV is in TOOL/LOCAL frame → auto-converted to robot base
    - WITH --relative: CSV is in ROBOT BASE frame → used directly (like recorded trajectories)

Output:
    - Recorded trajectories: ./recorded_trajectories/trajectory_YYYYMMDD_HHMMSS.csv
    - Speed analysis: ./speed_data/speed_*.npz and plots
    - Camera videos: camera1_video_*.mp4, camera2_video_*.mp4, topdowncamera_video_*.mp4
    - Temporary conversions: temp_converted/ (auto-deleted after playback)
"""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="ABB IRB-1300 IK Record and Playback")
parser.add_argument("--mode", type=str, required=True, choices=["r", "p"], 
                    help="Mode: 'r' for record, 'p' for playback")
parser.add_argument("--csv", type=str, default=None,
                    help="Path to CSV file (required for playback mode)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
parser.add_argument("--relative", action="store_true",
                    help="CSV is already in robot base frame (skip conversion). Default: CSV is in tool frame and will be converted")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Validate arguments
if args_cli.mode == "p" and args_cli.csv is None:
    parser.error("--csv is required when --mode is 'p' (playback)")

# Validate --relative flag (only warn, doesn't cause error)
if args_cli.relative and args_cli.mode == "r":
    print("[WARNING] --relative flag has no effect in record mode (ignored)")

# Create enable_camera boolean based on --enable_cameras flag
enable_camera = hasattr(args_cli, 'enable_cameras') and args_cli.enable_cameras

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import pandas as pd
import numpy as np
import sys
import subprocess
import tempfile
import shutil

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCameraCfg

# Import InputManager for keyboard handling
sys.path.append(os.path.dirname(__file__))
from InputManager import input_manager

# Import graphing utility for playback mode
from graphing_utility import (SpeedDataCollector, JointDataCollector, PoseTrackingCollector,
                              save_and_plot_all_data)

# Pre-defined configs - import IRB-1300 configuration
from irb_1300_cfg import IRB_1300_900_CFG
from CameraRecorder import create_camera_recorder
# Default home position for the IRB-1300 robot (x, y, z, qx, qy, qz, qw)
# Position that's EASILY reachable from [0,0,0,0,1.57,0] initial joint state
DEFAULT_HOME_POSITION = torch.tensor([
    0.75,      # x - forward (750mm) - straight ahead reach
    0.0,       # y - lateral (centered)
    0.25,      # z - height (250mm) - pointing down from base
    0.0,       # qx
    1.0,       # qy - pointing straight down (180 degrees)
    0.0,       # qz
    0.0        # qw
], dtype=torch.float32)

CUBE_POSITION = (0.6, 0.8, 0.3)  
CUBE_SIZE = (0.3, 0.3, 0.3) 
CONE_SIZE = (0.1, 0.1, 0.3) 
CONE_POSITION = (-0.36777, -0.91582 , 0.3204)  

@configclass
class IRB1300SceneCfg(InteractiveSceneCfg):
    """Configuration for ABB IRB-1300 scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # Single Table mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", 
    #         scale=(2.0, 2.0, 2.0)
    #     ),
    # )


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cone.usd",
            scale=(0.1, 0.1, 0.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CONE_POSITION),
    )

    if enable_camera:
        #(w,x,y,z)  -> (-x, w, z , -y)
        #camers 1 rotation  0.43826, 0.43826, 0.55491, 0.55491 -> (-0.43826, 0.43826, -0.55491, 0.55491)
        # some recording Cameras
        camera1 = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera1",
            offset=TiledCameraCfg.OffsetCfg(pos=(2.068,0.0, 0.695), rot=(-0.43826, 0.43826, 0.55491, -0.55491)),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=1280,
            height=720,
            data_types=["rgb"],
        )

        # camera 2 rotation 0.60573, 0.60573, -0.36481, -0.36481 -> (-0.60573, 0.60573, 0.36481, 0.36481)
        camera2 = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera2",
            offset=TiledCameraCfg.OffsetCfg(pos=(-2.396,-1.66592, 0.6956), rot=(-0.60573, 0.60573, -0.36481, 0.36481)),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=1280,
            height=720,
            data_types=["rgb"],
        )

        # top down camera rotation 1.0, 0.0, 0.0, 1.0 -> (0.0, 1.0, 0.0, 1.0)

        topdowncamera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/TopDownCamera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.53735, -0.44226, 3.7503), rot=(0.0, 1.0, 1.0, 0.0)),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=1280,
            height=720,
            data_types=["rgb"],
        )
    
    # # Collision cube (visual only, no physics, translucent)
    # cube = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=CUBE_SIZE,
    #         collision_props=None,  # No collision
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(1.0, 0.0, 0.0), 
    #             # roughness=1.0,  # Maximum roughness to reduce reflections
    #             # metallic=0.2,   # No metallic properties to avoid reflections
    #             # opacity=0.1     # Translucent so waypoints inside are visible
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=CUBE_POSITION),
    # )
    
    # ABB IRB-1300 robot
    robot = IRB_1300_900_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class TrajectoryRecorder:
    """Records end-effector trajectories to CSV."""
    
    def __init__(self, save_dir="./recorded_trajectories"):
        """Initialize trajectory recorder.
        
        Args:
            save_dir: Directory to save recorded trajectories
        """
        self.save_dir = save_dir
        self.recorded_poses = []
        self.last_pose = None
        self.pose_threshold = 0.02  # Minimum change to record (meters/radians)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[TrajectoryRecorder] Initialized. Trajectories will be saved to: {save_dir}")
        print(f"[TrajectoryRecorder] Pose change threshold: {self.pose_threshold}")
    
    def record_pose(self, ee_pose: torch.Tensor):
        """Record an end-effector pose if it has changed significantly.
        
        Args:
            ee_pose: End-effector pose tensor [x, y, z, qx, qy, qz, qw]
        """
        # Convert to numpy for easier comparison
        current_pose = ee_pose.cpu().numpy().flatten()
        
        # Check if pose has changed significantly
        if self.last_pose is not None:
            diff = np.linalg.norm(current_pose - self.last_pose)
            if diff < self.pose_threshold:
                return  # Pose hasn't changed enough, skip recording
        
        # Record the pose
        self.recorded_poses.append(current_pose.tolist())
        self.last_pose = current_pose
    
    def save(self, filename=None):
        """Save recorded trajectory to CSV.
        
        Args:
            filename: Optional filename, otherwise auto-generate with timestamp
        """
        if len(self.recorded_poses) == 0:
            print("[TrajectoryRecorder] No poses recorded, not saving.")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        # Full path
        filepath = os.path.join(self.save_dir, filename)
        
        # Reorder poses from [x,y,z,qx,qy,qz,qw] to [x,y,z,qw,qx,qy,qz]
        reordered_poses = []
        for pose in self.recorded_poses:
            # pose is [x, y, z, qx, qy, qz, qw]
            # reorder to [x, y, z, qw, qx, qy, qz]
            reordered = [pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]]
            reordered_poses.append(reordered)
        
        # Create dataframe and save
        df = pd.DataFrame(
            reordered_poses,
            columns=["x", "y", "z", "qw", "qx", "qy", "qz"]
        )
        df.to_csv(filepath, index=False)
        
        print(f"[TrajectoryRecorder] Saved {len(self.recorded_poses)} poses to: {filepath}")
        print(f"[TrajectoryRecorder] Format: x, y, z, qw, qx, qy, qz")
        return filepath
    
    def get_count(self):
        """Get number of recorded poses."""
        return len(self.recorded_poses)


class TeleopIKController:
    """Teleop controller for IK commands."""
    
    def __init__(self, step_size=0.01, rot_step_size=0.05):
        """Initialize teleop controller.
        
        Args:
            step_size: Position step size in meters
            rot_step_size: Rotation step size in radians
        """
        self.step_size = step_size
        self.rot_step_size = rot_step_size
        
        print("[TeleopIK] Initialized")
        print("[TeleopIK] Controls:")
        print("[TeleopIK]   Position: A/D (X), Q/E (Y), W/S (Z)")
        print("[TeleopIK]   Rotation: N/M (Roll), U/J (Pitch), H/K (Yaw)")
        print("[TeleopIK]   Press ESC to save and exit")
    
    def get_command_delta(self):
        """Get movement deltas based on keyboard input.
        
        Returns:
            Tuple of (position_delta, rotation_delta) as numpy arrays
        """
        pos_delta = np.zeros(3)
        rot_delta = np.zeros(3)
        
        # Position control
        if input_manager.is_pressed("a"):
            pos_delta[0] += self.step_size
        if input_manager.is_pressed("d"):
            pos_delta[0] -= self.step_size
        if input_manager.is_pressed("q"):
            pos_delta[1] += self.step_size
        if input_manager.is_pressed("e"):
            pos_delta[1] -= self.step_size
        if input_manager.is_pressed("w"):
            pos_delta[2] += self.step_size
        if input_manager.is_pressed("s"):
            pos_delta[2] -= self.step_size
        
        # Rotation control (Euler angles)
        if input_manager.is_pressed("n"):
            rot_delta[0] += self.rot_step_size  # Roll
        if input_manager.is_pressed("m"):
            rot_delta[0] -= self.rot_step_size
        if input_manager.is_pressed("u"):
            rot_delta[1] += self.rot_step_size  # Pitch
        if input_manager.is_pressed("j"):
            rot_delta[1] -= self.rot_step_size
        if input_manager.is_pressed("h"):
            rot_delta[2] += self.rot_step_size  # Yaw
        if input_manager.is_pressed("k"):
            rot_delta[2] -= self.rot_step_size
        
        return pos_delta, rot_delta


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
    
    Returns:
        Quaternion as [qx, qy, qz, qw]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qx, qy, qz, qw])


def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to Euler angles.
    
    Args:
        qx, qy, qz, qw: Quaternion components
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def apply_rotation_delta(quat, rot_delta):
    """Apply rotation delta to quaternion.
    
    Args:
        quat: Current quaternion [qx, qy, qz, qw]
        rot_delta: Rotation delta as Euler angles [roll, pitch, yaw]
    
    Returns:
        New quaternion after applying delta
    """
    # Convert current quat to Euler
    roll, pitch, yaw = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])
    
    # Apply delta
    roll += rot_delta[0]
    pitch += rot_delta[1]
    yaw += rot_delta[2]
    
    # Convert back to quaternion
    return euler_to_quaternion(roll, pitch, yaw)


def run_record_mode(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run simulation in record mode with teleop control."""
    robot = scene["robot"]
    
    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, 
        num_envs=scene.num_envs, 
        device=sim.device
    )
    
    # Create markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
    # Setup robot entity configuration for IRB-1300
    # IRB-1300's end effector is ee_pose (last link in the chain)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_pose"])
    robot_entity_cfg.resolve(scene)
    
    # Get end-effector jacobian index
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    
    # === Quick robot verification ===
    print(f"\n[INFO] Robot: {robot.num_joints} joints, End effector: {robot.body_names[robot_entity_cfg.body_ids[0]]}")
    
    # Initialize teleop controller and recorder
    teleop = TeleopIKController(step_size=0.01, rot_step_size=0.05)
    recorder = TrajectoryRecorder()
    
    # Set initial command to DEFAULT HOME POSITION
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    home_pose = DEFAULT_HOME_POSITION.to(robot.device)
    ik_commands[:, 0:7] = home_pose.unsqueeze(0).expand(scene.num_envs, -1)
    diff_ik_controller.set_command(ik_commands)
    
    print(f"\n[INFO] Home position: [{home_pose[0]:.2f}, {home_pose[1]:.2f}, {home_pose[2]:.2f}]m")
    print(f"[INFO] Actuators: Main axes - Stiffness={robot.actuators['main_axes'].stiffness}, Damping={robot.actuators['main_axes'].damping}")
    print(f"[INFO]            Wrist axes - Stiffness={robot.actuators['wrist_axes'].stiffness}, Damping={robot.actuators['wrist_axes'].damping}")
    print("\n[INFO] Record mode started. Use keyboard to control robot.")
    print("[INFO] Press ESC to save trajectory and exit.")
    
    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    settling_steps = 300  # 300 steps (3 seconds) - give IK time to converge
    is_settled = False
    
    while simulation_app.is_running():
        # Settling phase - move to home position first
        if count < settling_steps:
            if count == 0:
                print("[INFO] Moving to home position... (will take 5 seconds)")
            # Show progress every second
            if count % 100 == 0 and count > 0:
                ee_pose_w_temp = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                temp_error = torch.norm(ee_pose_w_temp[0,0:3] - ik_commands[0,0:3]).item()
                print(f"  Settling... {count}/{settling_steps} steps, error: {temp_error:.3f}m", end="\r")
            if count == settling_steps - 1:
                is_settled = True
                # Verify home position was reached
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
                pos_error = torch.norm(ee_pose_w[0,0:3] - ik_commands[0,0:3]).item()
                
                # Check for joint limits
                joint_limits = [3.142, 2.269, 1.134, 4.014, 2.269, 6.981]
                at_limit = [i+1 for i, (pos, lim) in enumerate(zip(joint_pos, joint_limits)) if pos >= lim - 0.02]
                
                if at_limit:
                    print(f"\n🚨 CRITICAL: Joints {at_limit} at limit! Home position UNREACHABLE!")
                    print(f"   Position error: {pos_error:.3f}m")
                    print(f"   Joint positions (rad): {joint_pos}")
                    print(f"   ❌ IK has FAILED. Robot will not respond to commands!\n")
                elif pos_error > 0.15:
                    print(f"\n⚠️  Warning: Large position error ({pos_error:.3f}m) after settling")
                    print(f"   Joint positions (rad): {joint_pos}")
                    print(f"   IK may not have fully converged. Actuators may be too stiff/weak.\n")
                else:
                    print(f"\n✅ Home position reached! Error: {pos_error:.4f}m")
                    print(f"   Ready to record trajectories.\n")
        
        # Only allow control and recording after settling
        if is_settled:
            # Check for ESC key to save and exit
            if input_manager.is_pressed_once("escape"):
                print("\n[INFO] ESC pressed. Saving trajectory...")
                filepath = recorder.save()
                if filepath:
                    print(f"[INFO] Trajectory saved successfully!")
                break
            
            # Get teleop command deltas
            pos_delta, rot_delta = teleop.get_command_delta()
            
            # Update IK command based on deltas
            if np.any(pos_delta != 0) or np.any(rot_delta != 0):
                # Apply position delta
                ik_commands[:, 0:3] += torch.from_numpy(pos_delta).float().to(robot.device)
                
                # Apply rotation delta
                if np.any(rot_delta != 0):
                    current_quat = ik_commands[0, 3:7].cpu().numpy()
                    new_quat = apply_rotation_delta(current_quat, rot_delta)
                    ik_commands[:, 3:7] = torch.from_numpy(new_quat).float().to(robot.device)
                
                # Set new command
                diff_ik_controller.set_command(ik_commands)
        
        # Compute IK
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Apply joint commands
        # PHYSICS-BASED CONTROL (current - uses PD controller with stiffness/damping)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        
        # KINEMATIC CONTROL (uncomment below and comment above for pure kinematic control)
        #robot.write_joint_state_to_sim(position=joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # Step simulation
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Record current end-effector pose (only after settling)
        if is_settled:
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            recorder.record_pose(ee_pose_w[0])  # Only record first environment
            
            # === SIMPLIFIED STATUS OUTPUT ===
            if count % 100 == 0:  # Print every 100 steps (1 second)
                ee_pos = ee_pose_w[0, 0:3].cpu().numpy()
                goal_pos = ik_commands[0, 0:3].cpu().numpy()
                joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
                
                # Position error
                pos_error = np.linalg.norm(ee_pos - goal_pos)
                
                # Check joint limits
                joint_limits = [3.142, 2.269, 1.134, 4.014, 2.269, 6.981]
                at_limit = [i+1 for i, (pos, limit) in enumerate(zip(joint_pos, joint_limits)) if pos >= limit - 0.02]
                
                # Status indicator
                status = "✅ OK" if pos_error < 0.05 and not at_limit else "⚠️ ISSUE"
                limit_warn = f" | Joints at limit: {at_limit}" if at_limit else ""
                
                print(f"[{status}] Step {count} | Error: {pos_error:.3f}m | Poses: {recorder.get_count()}{limit_warn}", end="\r")
        
        # Update markers
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
    
    print(f"\n[INFO] Recording complete. Total poses: {recorder.get_count()}")


def convert_trajectory_to_base_frame(input_csv: str, output_csv: str) -> bool:
    """Convert trajectory from tool frame to robot base frame using trajectory_transform.py.
    
    Args:
        input_csv: Path to input CSV in tool frame
        output_csv: Path to output CSV in base frame
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Get the path to trajectory_transform.py (in scripts/utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transform_script = os.path.join(current_dir, "../scripts/utils/trajectory_transform.py")
    
    # Check if the script exists
    if not os.path.exists(transform_script):
        print(f"[ERROR] trajectory_transform.py not found at: {transform_script}")
        return False
    
    print(f"[INFO] Converting trajectory from tool frame to robot base frame...")
    print(f"[INFO]   Input: {input_csv}")
    print(f"[INFO]   Output: {output_csv}")
    
    try:
        # Call trajectory_transform.py with --meters flag for Isaac Lab compatibility
        result = subprocess.run(
            ["python", transform_script, input_csv, output_csv, "--meters"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print the output from the script
        if result.stdout:
            print("[trajectory_transform.py]", result.stdout.strip())
        
        print(f"[INFO] ✅ Conversion successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed!")
        print(f"[ERROR] {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during conversion: {e}")
        return False


def run_playback_mode(sim: sim_utils.SimulationContext, scene: InteractiveScene, csv_file: str):
    """Run simulation in playback mode."""
    robot = scene["robot"]
    
    # Load trajectory from CSV
    print(f"[INFO] Loading trajectory from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        # CSV format: x, y, z, qw, qx, qy, qz [, speed_mm_s (optional)]
        # Need to convert to: x, y, z, qx, qy, qz, qw (internal format)
        data = df.values[:, :7]  # Take only first 7 columns (pose)
        
        # Reorder from [x,y,z,qw,qx,qy,qz] to [x,y,z,qx,qy,qz,qw]
        reordered_data = np.zeros_like(data)
        reordered_data[:, 0:3] = data[:, 0:3]  # x, y, z
        reordered_data[:, 3] = data[:, 4]      # qx (from index 4)
        reordered_data[:, 4] = data[:, 5]      # qy (from index 5)
        reordered_data[:, 5] = data[:, 6]      # qz (from index 6)
        reordered_data[:, 6] = data[:, 3]      # qw (from index 3)
        
        trajectory = torch.tensor(reordered_data, dtype=torch.float32, device=sim.device)
        print(f"[INFO] Loaded {len(trajectory)} waypoints (format: x,y,z,qw,qx,qy,qz)")
        
        # Load target speeds if available (8th column)
        target_speeds_mm_s = None
        if df.shape[1] >= 8 and 'speed_mm_s' in df.columns:
            target_speeds_mm_s = df['speed_mm_s'].values
            print(f"[INFO] Loaded {len(target_speeds_mm_s)} target speeds from CSV (8th column)")
        
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return
    
    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,
        device=sim.device
    )
    
    # Create markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
    # Create trajectory path markers as small green spheres (showing 1/5 of waypoints)
    trajectory_marker_cfg = VisualizationMarkersCfg(
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.002,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green color
            ),
        }
    )
    
    # Take every waypoint (all of the trajectory)
    num_waypoints = len(trajectory)
    step_size = 1  # Every waypoint
    trajectory_indices = list(range(0, num_waypoints, step_size))
    num_traj_markers = len(trajectory_indices)
    
    print(f"[INFO] Creating {num_traj_markers} trajectory sphere markers (showing every {step_size} waypoints)")
    
    # Create sphere markers for trajectory visualization
    traj_markers = []
    for i in range(num_traj_markers):
        traj_marker = VisualizationMarkers(
            trajectory_marker_cfg.replace(prim_path=f"/Visuals/trajectory_sphere_{i}")
        )
        traj_markers.append(traj_marker)
    
    # Visualize all trajectory markers at their fixed positions
    # For spheres, we only need position (no orientation needed)
    for i, traj_idx in enumerate(trajectory_indices):
        waypoint = trajectory[traj_idx]
        # Position: first 3 elements
        pos = waypoint[0:3].unsqueeze(0) + scene.env_origins[0:1]
        # Spheres don't need orientation, but VisualizationMarkers.visualize expects both
        # We'll pass a default identity quaternion
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos.device)
        traj_markers[i].visualize(pos, quat)
    
    # Setup robot entity configuration for IRB-1300
    # IRB-1300's end effector is ee_pose (last link in the chain)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_pose"])
    robot_entity_cfg.resolve(scene)
    
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    
    # Prepare trajectory playback
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    
    # Start from DEFAULT HOME POSITION first
    home_pose = DEFAULT_HOME_POSITION.to(robot.device)
    ik_commands[:, 0:7] = home_pose.unsqueeze(0).expand(scene.num_envs, -1)
    diff_ik_controller.set_command(ik_commands)
    
    print(f"[INFO] Starting from default home position:")
    print(f"[INFO]   Position: [{home_pose[0]:.3f}, {home_pose[1]:.3f}, {home_pose[2]:.3f}]")
    print(f"[INFO]   Orientation (quat): [{home_pose[3]:.3f}, {home_pose[4]:.3f}, {home_pose[5]:.3f}, {home_pose[6]:.3f}]")
    
    current_waypoint = 0
    
    print("[INFO] Playback mode started.")
    print(f"[INFO] Following {len(trajectory)} waypoints smoothly...")
    
    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    settling_steps = 300  # Number of steps to settle at home position
    is_settled = False
    steps_per_waypoint = 50  # Number of steps to reach each waypoint
    step_count = 0

    # Initialize data collectors
    speed_collector = SpeedDataCollector(dt=sim_dt, target_speeds_mm_s=target_speeds_mm_s)
    joint_collector = JointDataCollector()
    pose_tracker = PoseTrackingCollector()
    print("[INFO] Speed, joint, and pose tracking collectors initialized")

    if enable_camera:
        # Create camera recorders
        camera1_recorder = create_camera_recorder(camera_name="Camera1", data_type="rgb", fps=30, video_format="mp4")
        camera2_recorder = create_camera_recorder(camera_name="Camera2", data_type="rgb", fps=30, video_format="mp4")
        topdowncamera_recorder = create_camera_recorder(camera_name="TopDownCamera", data_type="rgb", fps=30, video_format="mp4")
    
    is_first_waypoint = True  # New flag for special handling of first waypoint
    last_waypoint_index = -1  # Track last waypoint for transition detection
    
    while simulation_app.is_running():
        # Settling phase - move to home position first
        if count < settling_steps:
            if count == 0:
                print("[INFO] Moving to home position...")
            if count == settling_steps - 1:
                print("[INFO] Home position reached. Starting playback!")
                is_settled = True
                # Now set the first waypoint of the trajectory
                if len(trajectory) > 0:
                    ik_commands[:, 0:7] = trajectory[current_waypoint].unsqueeze(0).expand(scene.num_envs, -1)
                    diff_ik_controller.set_command(ik_commands)
        # Only progress through trajectory after settling
        if is_settled:
            # Check if we've reached the end of trajectory
            if current_waypoint >= len(trajectory):
                print("\n[INFO] Playback complete!")
                break
            
            # Use settling_steps for first waypoint, regular for others
            current_steps_needed = settling_steps if is_first_waypoint else steps_per_waypoint
            
            # Update waypoint if enough steps have passed
            if step_count >= current_steps_needed:
                if is_first_waypoint:
                    is_first_waypoint = False  # Switch to regular steps after first waypoint
                
                current_waypoint += 1
                step_count = 0
                
                if current_waypoint < len(trajectory):
                    # Record waypoint transition for speed analysis
                    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                    speed_collector.record_waypoint_transition(
                        ee_pose_w[0, 0:3], 
                        current_waypoint, 
                        speed_collector.current_time
                    )
                    
                    # Set next waypoint
                    ik_commands[:, 0:7] = trajectory[current_waypoint].unsqueeze(0).expand(scene.num_envs, -1)
                    diff_ik_controller.set_command(ik_commands)
                    
                    if current_waypoint % 10 == 0:
                        print(f"[INFO] Waypoint {current_waypoint}/{len(trajectory)}", end="\r")
            
            step_count += 1
        
        # Compute IK
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Apply joint commands
        # PHYSICS-BASED CONTROL (current - uses PD controller with stiffness/damping)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        
        # KINEMATIC CONTROL (uncomment below and comment above for pure kinematic control)
        # robot.write_joint_state_to_sim(position=joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # Step simulation
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        if count % 10 == 0 and enable_camera:
            camera1_recorder.capture_frame_scene(scene)
            camera2_recorder.capture_frame_scene(scene)
            topdowncamera_recorder.capture_frame_scene(scene)

        # Update markers
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        
        # Record joint data at EVERY step
        joint_collector.record_step(robot.data.joint_pos[0])
        
        # Record pose tracking data at EVERY step
        pose_tracker.record_step(ik_commands[0, 0:7], ee_pose_w[0, 0:7])
        
        # Record speed data (only after settling, since it's tied to waypoints)
        if is_settled:
            speed_collector.record_step(ee_pose_w[0, 0:3], current_waypoint)
    
    if enable_camera:
        camera1_recorder.create_video(filename=f"camera1_video_{csv_file.split('/')[-1]}")
        camera2_recorder.create_video(filename=f"camera2_video_{csv_file.split('/')[-1]}")
        topdowncamera_recorder.create_video(filename=f"topdowncamera_video_{csv_file.split('/')[-1]}")
    
    # Save and plot all data using graphing_utility
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = save_and_plot_all_data(speed_collector, joint_collector, pose_tracker, csv_basename, timestamp)


def main():
    """Main function."""
    
    # Handle trajectory conversion (DEFAULT behavior in playback mode, unless --relative is set)
    csv_file_to_use = args_cli.csv
    temp_dir = None
    
    if args_cli.mode == "p" and not args_cli.relative:
        # DEFAULT: Convert from tool frame to robot base frame
        print("\n" + "="*80)
        print("TRAJECTORY CONVERSION MODE (Tool → Robot Base Frame)")
        print("="*80)
        print("[INFO] CSV is assumed to be in TOOL/LOCAL coordinates")
        print("[INFO] Converting to robot base frame for playback...")
        print("[INFO] (Use --relative flag if CSV is already in robot base frame)")
        print("="*80)
        
        # Create temp directory for converted trajectory
        input_csv = args_cli.csv
        input_basename = os.path.basename(input_csv)
        input_dir = os.path.dirname(os.path.abspath(input_csv))
        
        # Create temp folder in the same directory as input CSV
        temp_dir = os.path.join(input_dir, "temp_converted")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate output filename
        name_without_ext = os.path.splitext(input_basename)[0]
        converted_csv = os.path.join(temp_dir, f"{name_without_ext}_base_frame.csv")
        
        # Convert the trajectory
        conversion_success = convert_trajectory_to_base_frame(input_csv, converted_csv)
        
        if not conversion_success:
            print("[ERROR] Trajectory conversion failed. Exiting.")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            simulation_app.close()
            return
        
        # Use the converted CSV for playback
        csv_file_to_use = converted_csv
        print("="*80 + "\n")
    elif args_cli.mode == "p" and args_cli.relative:
        # --relative flag: Skip conversion
        print("\n" + "="*80)
        print("DIRECT PLAYBACK MODE (--relative flag set)")
        print("="*80)
        print("[INFO] CSV is assumed to be in ROBOT BASE FRAME")
        print("[INFO] No conversion will be performed")
        print("="*80 + "\n")
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Design scene with IRB-1300
    scene_cfg = IRB1300SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    print("[INFO] Setup complete...")
    
    # Run appropriate mode
    if args_cli.mode == "r":
        print("[INFO] Starting RECORD mode...")
        run_record_mode(sim, scene)
    else:  # mode == "p"
        print("[INFO] Starting PLAYBACK mode...")
        if not args_cli.relative:
            print(f"[INFO] Using converted trajectory: {csv_file_to_use}")
        else:
            print(f"[INFO] Using trajectory as-is (robot base frame): {csv_file_to_use}")
        run_playback_mode(sim, scene, csv_file_to_use)
    
    # Cleanup temp directory if it was created
    if temp_dir and os.path.exists(temp_dir):
        print(f"\n[INFO] Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation app
    simulation_app.close()

