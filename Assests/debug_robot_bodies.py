#!/usr/bin/env python3
"""
Quick script to check what bodies and joints exist in the IRB-1300 robot.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug IRB-1300 bodies")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg

# Import robot config
from irb_1300_cfg import IRB_1300_900_CFG

@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    """Minimal scene for debugging."""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = IRB_1300_900_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = DebugSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset
    sim.reset()
    
    # Get robot
    robot = scene["robot"]
    
    # Print all bodies
    print("\n" + "="*80)
    print("ROBOT BODIES FOUND:")
    print("="*80)
    for i, body_name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {body_name}")
    
    print("\n" + "="*80)
    print("ROBOT JOINTS FOUND:")
    print("="*80)
    for i, joint_name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {joint_name}")
    
    print("\n" + "="*80)
    print(f"Total bodies: {robot.num_bodies}")
    print(f"Total joints: {robot.num_joints}")
    print("="*80 + "\n")
    
    # Check if ee_pose exists
    if "ee_pose" in robot.body_names:
        ee_idx = robot.body_names.index("ee_pose")
        print(f"✅ ee_pose FOUND at index {ee_idx}")
    else:
        print("❌ ee_pose NOT FOUND")
        print("\nAvailable bodies that might be the end effector:")
        for name in robot.body_names:
            if any(keyword in name.lower() for keyword in ['link_6', 'ee', 'end', 'effector', 'tool']):
                print(f"  - {name}")
    
    simulation_app.close()

if __name__ == "__main__":
    main()


