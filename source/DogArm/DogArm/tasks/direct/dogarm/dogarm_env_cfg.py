# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the DogArm DirectRL task.

Go2 quadruped + 6-DOF arm: simultaneous locomotion velocity tracking
and end-effector pose tracking.
"""

from __future__ import annotations

import math

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from ....robots.go2arm import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    GO2ARM_CFG,
    LEG_JOINT_NAMES,
)


@configclass
class DogarmEnvCfg(DirectRLEnvCfg):
    """Configuration for the DogArm direct RL environment."""

    # -- Environment --
    # -- Task mode ("velocity" | "navigation" | "align") --
    task_mode: str = "align"

    decimation: int = 4  # 200Hz sim → 50Hz control
    episode_length_s: float = 20.0

    # -- Spaces --
    # Action: 18-dim joint position offsets (12 leg + 6 arm)
    action_space: int = 12  # legs only (official Go2: 12 joints); arm fixed via default pos
    # Observation: 55-dim (12 leg actions, official Go2 style)
    #   base_ang_vel(3) + base_lin_vel(3) + joint_pos_rel(18) + joint_vel_rel(18)
    #   + prev_actions(12) + velocity_commands(3) + projected_gravity(3) + base_height(1)
    observation_space: int = 3 + 3 + 18 + 18 + 12 + 3 + 3 + 1  # 61
    # State: observation + priv info for critic
    #   + joint_torques(18) + feet_contact(4)
    state_space: int = 61 + 18 + 4  # 83

    # -- Observation groups for asymmetric actor-critic --
    obs_groups: dict[str, list[str]] = {
        "actor": ["obs"],
        "critic": ["critic"],
    }

    # -- Observation history --
    num_obs_history_steps: int = 10 # 10-step history stack

    # -- Simulation --
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,                   # 200 Hz physics
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
        ),
    )

    # -- Robot --
    robot_cfg = GO2ARM_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # -- Scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -- Terrain --
    terrain_type: str = "plane"  # "plane" | "rough" | "cs2map"
    cs2_map_name: str = "dust2"  # which CS2 map to load

    rough_terrain_cfg: TerrainGeneratorCfg = TerrainGeneratorCfg(
        size=(200.0, 200.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,               # single continuous block
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(-0.05, 0.05),  # gravel texture
                noise_step=0.01,              # fine grid
                border_width=0.0,
            ),
        },
    )

    # ========================================================================
    # Joint names
    # ========================================================================
    leg_joint_names: list[str] = LEG_JOINT_NAMES
    arm_joint_names: list[str] = ARM_JOINT_NAMES
    all_joint_names: list[str] = ALL_JOINT_NAMES
    ee_body_name: str = EE_BODY_NAME

    # ========================================================================
    # Action scales
    # ========================================================================
    leg_action_scale: float = 0.25
    arm_action_scale: float = 0.1  # conservative for align; velocity/nav set to 0

    # ========================================================================
    # Command generation
    # ========================================================================
    # Velocity heading command — world-frame heading + body-frame speed
    # Robot must turn to match the world-frame heading then walk at the given speed.
    vel_cmd_speed_range_init: tuple[float, float] = (0.05, 0.3)
    vel_cmd_speed_range_final: tuple[float, float] = (0.1, 1.5)
    vel_cmd_heading_range: tuple[float, float] = (-3.14, 3.14)

    # -- Navigation mode params --
    nav_vel_speed_range: tuple[float, float] = (0.03, 0.15)  # slower, stable walk
    target_distance_range: tuple[float, float] = (1.5, 4.0)
    target_reach_threshold: float = 0.3
    rew_target_progress: float = 8.0
    rew_target_reach: float = 20.0
    rew_target_alignment: float = 1.0
    rew_forward_speed: float = 2.0  # locomotion scaffold: reward moving forward
    vel_cmd_resample_time_range: tuple[float, float] = (10.0, 10.0)  # Go2Arm_Lab: 10s

    # -- Align mode params --
    align_vel_speed_range: tuple[float, float] = (0.03, 0.15)  # slower, stable walk
    arm_base_body_name: str = "shoulder_link"  # link0 for body-frame EE commands
    align_target_distance_range: tuple[float, float] = (1.5, 3.0)  # target point
    align_target_reach_threshold: float = 0.5  # close enough for arm
    ee_cmd_arm_length: float = 0.55  # max arm reach [m]
    ee_cmd_min_radius: float = 0.15  # min radius to avoid self-collision
    ee_cmd_theta_range: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    ee_cmd_phi_range: tuple[float, float] = (0.0, math.pi / 2)
    ee_cmd_rpy_range: dict = {
        "roll": (-math.pi / 4, math.pi / 4),
        "pitch": (-math.pi / 4, math.pi / 4),
        "yaw": (-math.pi / 4, math.pi / 4),
    }
    ee_cmd_resample_time_range: tuple[float, float] = (6.0, 8.0)
    rew_ee_pos_tracking: float = 5.0  # stronger position pull
    rew_ee_ori_tracking: float = -2.0
    rew_ee_action_rate: float = -0.005
    rew_ee_action_smoothness: float = -0.02
    ee_pos_tracking_std: float = 0.3  # wider tolerance for early training
    rew_align_forward_speed: float = 2.0
    align_curriculum_steps: tuple[int, int, int] = (120000, 240000, 360000)

    # Curriculum
    curriculum_coeff: int = 1000

    # ========================================================================
    # Reward weights
    # ========================================================================
    # Velocity tracking
    rew_lin_vel_tracking: float = 1.5
    rew_ang_vel_tracking: float = 0.75

    # Leg locomotion / stability (official Go2 flat weights)
    rew_lin_vel_z: float = -2.0
    rew_ang_vel_xy: float = -0.05
    rew_dof_torques: float = -2.0e-4  # official Go2: 20x our old weight
    rew_dof_acc: float = -2.5e-7
    rew_action_rate: float = -0.01
    rew_feet_air_time: float = 0.25  # official Go2 flat
    rew_flat_orientation: float = -2.5  # official Go2 flat
    # Gait posture (LegoManip_Lab: prevents limping/asymmetry)
    rew_joint_mirror: float = -0.15  # penalize asymmetric leg pairs
    rew_air_time_variance: float = -1.0  # penalize irregular stepping rhythm
    rew_feet_slide: float = -0.1  # penalize foot dragging
    rew_feet_long_air: float = -0.5  # penalize keeping a foot lifted too long

    # Reward std parameters
    lin_vel_tracking_std: float = math.sqrt(0.25)
    ang_vel_tracking_std: float = math.sqrt(0.25)

    # Observation noise (official Go2 domain rand on observations)
    obs_noise_base_lin_vel: float = 0.1
    obs_noise_base_ang_vel: float = 0.2
    obs_noise_projected_gravity: float = 0.05
    obs_noise_joint_pos: float = 0.01
    obs_noise_joint_vel: float = 1.5

    # ========================================================================
    # Termination thresholds
    # ========================================================================
    base_contact_force_threshold: float = 0.5
    arm_contact_force_threshold: float = 0.5
    thigh_contact_force_threshold: float = 0.5
    calf_contact_force_threshold: float = 0.5
    
    # ========================================================================
    # Domain randomization
    # ========================================================================
    # Friction
    dr_static_friction_range: tuple[float, float] = (0.5, 4.0)
    dr_dynamic_friction_range: tuple[float, float] = (0.5, 2.0)

    # Base mass additive
    dr_base_mass_range: tuple[float, float] = (-3.0, 3.0)

    # EE mass additive
    dr_ee_mass_range: tuple[float, float] = (-0.1, 0.5)

    # COM offset
    dr_com_range: dict = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}

    # Actuator gains (scale)
    dr_stiffness_range: tuple[float, float] = (0.8, 1.2)
    dr_damping_range: tuple[float, float] = (0.8, 1.2)

    # Base pose
    dr_pose_range: dict = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
    dr_velocity_range: dict = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
    }

    # Joint positions
    dr_joint_pos_range: tuple[float, float] = (0.5, 1.5)

    # Push robot
    dr_push_range: dict = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    dr_push_interval_range: tuple[float, float] = (10.0, 15.0)
