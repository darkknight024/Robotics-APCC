# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ABB IRB-1300 Robot.

The following configuration parameters are available:

* :obj:`IRB_1300_900_CFG`: The IRB-1300/0.9 arm without a gripper.

Reference: https://new.abb.com/products/robotics/industrial-robots/irb-1300
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# Get the absolute path to the USD file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#_USD_PATH = os.path.join(_CURRENT_DIR, "Robot APCC/IRB-1300 900 URDF/urdf/kuka_normal.usd")
_USD_PATH = os.path.join(_CURRENT_DIR, "Robot APCC/IRB-1300 900 URDF/urdf/kuka_updated.usd")

IRB_1300_900_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Joint_1": 0.0,      # Range: -3.142 to 3.142 rad (-180° to 180°)
            "Joint_2": 0.0,      # Range: -1.745 to 2.269 rad - START AT 0 FOR EASIER REACH
            "Joint_3": 0.0,      # Range: -3.665 to 1.134 rad - START AT 0 FOR EASIER REACH
            "Joint_4": 0.0,      # Range: -4.014 to 4.014 rad (-230° to 230°)
            "Joint_5": 1.57,     # Range: -2.269 to 2.269 rad - POINT DOWN (90 degrees)
            "Joint_6": 0.0,      # Range: -6.981 to 6.981 rad (-400° to 400°)
        },
    ),
    actuators={
        # Main axes (Joints 1-3) - Larger joints with higher torque
        "main_axes": ImplicitActuatorCfg(
            joint_names_expr=["Joint_[1-3]"],
            # Realistic torque limits for main axes (estimated 150 Nm)
            effort_limit={
                "Joint_1": 150.0,  # Base rotation - high torque needed
                "Joint_2": 150.0,  # Shoulder - high torque needed
                "Joint_3": 150.0,  # Elbow - high torque needed
            },
            # Real-world velocity limits from manual (rad/s)
            velocity_limit={
                "Joint_1": 4.89,   # ~280°/s
                "Joint_2": 3.98,   # ~228°/s
                "Joint_3": 5.76,   # ~330°/s
            },
            # Reduced stiffness for realistic behavior
            stiffness=100.0,
            # Critical damping: 2 * sqrt(stiffness) = 2 * sqrt(100) = 20
            damping=20.0,
            # Note: Joint friction should be defined in the URDF/USD file's joint properties
            # friction=0.1,  # Not a supported parameter in ImplicitActuatorCfg
        ),
        # Wrist axes (Joints 4-6) - Smaller joints with lower torque
        "wrist_axes": ImplicitActuatorCfg(
            joint_names_expr=["Joint_[4-6]"],
            # Realistic torque limits from manual
            effort_limit={
                "Joint_4": 20.45,  # Max wrist torque axis 4 (Nm)
                "Joint_5": 20.45,  # Max wrist torque axis 5 (Nm)
                "Joint_6": 10.8,   # Max wrist torque axis 6 (Nm)
            },
            # Real-world velocity limits from manual (rad/s)
            velocity_limit={
                "Joint_4": 8.73,   # ~500°/s
                "Joint_5": 7.24,   # ~415°/s
                "Joint_6": 12.57,  # ~720°/s
            },
            # Reduced stiffness for realistic behavior
            stiffness=100.0,
            # Critical damping: 2 * sqrt(stiffness) = 2 * sqrt(100) = 20
            damping=20.0,
            # Note: Joint friction should be defined in the URDF/USD file's joint properties
            # friction=0.1,  # Not a supported parameter in ImplicitActuatorCfg
        ),
    },
    soft_joint_pos_limit_factor=0.95,  # Use 95% of joint limits for safety
)
"""Configuration of ABB IRB-1300/0.9 arm using implicit actuator models.

This configuration uses the IRB-1300 with 900mm reach. The robot has 6 DOF
and is designed for high-speed material handling and assembly tasks.

Joint limits (from URDF):
- Joint_1: -180° to 180° (base rotation)
- Joint_2: -100° to 130° (shoulder)
- Joint_3: -210° to 65° (elbow)
- Joint_4: -230° to 230° (wrist roll)
- Joint_5: -130° to 130° (wrist pitch)
- Joint_6: -400° to 400° (wrist yaw, multi-turn)
"""

"""
old configs of arm
  "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint_[1-6]"],
            # ABB IRB-1300/0.9 actuator parameters - VERY STRONG for IK with no gravity
            effort_limit=1000.0,     # Very high torque for guaranteed movement
            velocity_limit=100.0,    # High velocity for responsive tracking
            stiffness=2000.0,        # High stiffness for precise positioning
            damping=100.0,           # High damping for stability
        ),
"""