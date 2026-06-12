# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 robot with a 6-DOF robotic arm."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the USD asset
_current_dir = os.path.dirname(os.path.abspath(__file__))
GO2ARM_USD_PATH = os.path.join(_current_dir, "usd", "go2_arm.usd")

GO2ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2ARM_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # official Go2: disabled
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),    # standing height
        joint_pos={
            # Leg joints
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # Arm joints
            "waist": 0.0,
            "shoulder": 0.0,
            "elbow": 0.1,
            "forearm_roll": -0.0,
            "wrist_angle": -0.54,
            "wrist_rotate": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=30.0,      # LeggedManip_Lab: arm+Go2 needs extra torque
            damping=0.6,         # LeggedManip_Lab value
            armature=0.01,       # LeggedManip: motor inertia
            friction=0.01,       # LeggedManip: joint friction
        ),
        "widow_arm": DCMotorCfg(
            joint_names_expr=[
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            ],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=3.14,
            stiffness=10.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

# Define joint name lists for convenience
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ARM_JOINT_NAMES = [
    "waist", "shoulder", "elbow",
    "forearm_roll", "wrist_angle", "wrist_rotate",
]

ALL_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES

# End-effector body name
EE_BODY_NAME = "gripper_link"

# Number of DOFs
NUM_LEG_DOFS = 12
NUM_ARM_DOFS = 6
NUM_DOFS = 18
