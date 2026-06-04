# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL for Go2 + 6-DOF arm — locomotion baseline (Go2Arm_Lab style).

Phase 1: Pure velocity tracking. Arm fixed at default. No target/EE.
Training: python scripts/rsl_rl/train.py --task=Template-Dogarm-Direct-v0 --headless

cmd
python ./scripts/rsl_rl/train.py --task=Template-Dogarm-Direct-v0 \
                                --num_envs 5000 --max_iterations 10000 --headless

python ./scripts/rsl_rl/resume.py \
    --task Template-Dogarm-Direct-v0 \
    --num_envs 5000 --max_iterations 10000 \
    --resume_path logs/rsl_rl/go2arm_direct/2026-05-28_17-02-34/model_2100.pt

python ./scripts/rsl_rl/play.py --task=Template-Dogarm-Direct-v0 --num_envs=1 

pkill -9 -f train.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

pkill -9 -f play.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

pkill -9 -f resume.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

rm -rf logs/rsl_rl/go2arm_direct/* outputs/*
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .dogarm_env_cfg import DogarmEnvCfg
from .utils.rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    dof_acc_l2,
    dof_torques_l2,
    flat_orientation_l2,
    gait_trot_penalty,
    lin_vel_z_l2,
)
from .utils.velocity.rewards import (
    ang_vel_tracking_exp,
    lin_vel_tracking_exp,
)
from .utils.domain_rand import (
    add_observation_noise,
    apply_push_velocity,
    init_push_timers,
    randomize_joint_positions,
    randomize_root_state,
)
from .utils.velocity import commands as vel_cmd
from .utils.velocity import observations as vel_obs
from .utils.navigation import commands as nav_cmd
from .utils.navigation import rewards as nav_rewards
from .utils.navigation import observations as nav_obs


class DogarmEnv(DirectRLEnv):
    """Locomotion baseline: velocity tracking, arm fixed."""

    cfg: DogarmEnvCfg
    robot: Articulation

    # Joint/body indices
    dof_idx: torch.Tensor
    hip_dof_idx: torch.Tensor
    thigh_calf_idx: torch.Tensor
    ee_body_idx: int  # needed for viz
    base_body_idx: int
    # TODO(target+EE): restore link0_body_idx when adding arm tasks
    foot_body_idx: torch.Tensor

    # State buffers
    vel_commands: torch.Tensor  # body-frame [vx, vy, wz]
    cmd_heading_w: torch.Tensor  # world-frame commanded heading
    cmd_speed: torch.Tensor  # commanded forward speed
    vel_cmd_timers: torch.Tensor
    push_timers: torch.Tensor
    # TODO(target+EE): restore when adding navigation + manipulation
    # target_pos: torch.Tensor
    # ee_pose_commands: torch.Tensor
    # ee_cmd_timers: torch.Tensor
    # _prev_distance: torch.Tensor
    obs_history: torch.Tensor
    obs_history_idx: int
    prev_actions: torch.Tensor
    prev_leg_dof_vel: torch.Tensor
    _curriculum_step: int

    # Markers
    robot_markers: object
    ee_curr_frame_markers: object
    ee_tgt_frame_markers: object

    def __init__(self, cfg: DogarmEnvCfg, render_mode: Optional[str] = None, **kwargs: object) -> None:
        # Set observation dim based on task mode (must happen before super().__init__)
        if cfg.task_mode == "navigation":
            cfg.observation_space = 67  # velocity(61) + target(6)
        super().__init__(cfg, render_mode, **kwargs)

        # Joints
        self.dof_idx, _ = self.robot.find_joints(cfg.all_joint_names, preserve_order=True)
        self.hip_dof_idx, _ = self.robot.find_joints(
            ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        )
        self.thigh_calf_idx, _ = self.robot.find_joints([
            "FR_thigh_joint", "FR_calf_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_thigh_joint", "RR_calf_joint", "RL_thigh_joint", "RL_calf_joint",
        ])
        # Bodies
        ee_ids, _ = self.robot.find_bodies([cfg.ee_body_name])
        self.ee_body_idx = ee_ids[0]
        base_ids, _ = self.robot.find_bodies(["base"])
        self.base_body_idx = base_ids[0]
        # TODO(target+EE): self.link0_body_idx = ...
        foot_ids, _ = self.robot.find_bodies(["FR_foot", "FL_foot", "RR_foot", "RL_foot"])
        self.foot_body_idx = torch.tensor(foot_ids, device=self.device, dtype=torch.long)

        # State
        self.vel_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.cmd_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.cmd_speed = torch.zeros(self.num_envs, device=self.device)
        self.vel_cmd_timers = torch.zeros(self.num_envs, device=self.device)
        self.push_timers = torch.zeros(self.num_envs, device=self.device)
        self._init_push_timers()

        # History
        obs_dim = cfg.observation_space  # 64
        self.obs_history = torch.zeros(self.num_envs, cfg.num_obs_history_steps, obs_dim, device=self.device)
        self.obs_history_idx = 0
        self.prev_actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.prev_leg_dof_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self._curriculum_step = 0

        # Navigation mode extras
        if cfg.task_mode == "navigation":
            self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self._prev_distance = torch.zeros(self.num_envs, device=self.device)

    # === Scene ================================================================

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Markers (lazy import)
        from .utils import visualize_tools as _vt
        self.robot_markers = _vt.define_robot_markers()
        self.ee_curr_frame_markers = _vt.define_ee_frame_markers("/Visuals/DogArm/EECurrFrame", scale=(0.08, 0.08, 0.08))
        self.ee_tgt_frame_markers = _vt.define_ee_frame_markers("/Visuals/DogArm/EETgtFrame", scale=(0.1, 0.1, 0.1))

    def _heading_to_body_vel(self) -> None:
        """Delegate to velocity command module."""
        vel_cmd.heading_to_body_vel(
            self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B,
            self.cmd_heading_w, self.cmd_speed, self.vel_commands,
        )

    def _resample_heading_command(self, env_ids: torch.Tensor) -> None:
        """Delegate to velocity command module."""
        self.cmd_speed[env_ids], self.cmd_heading_w[env_ids] = vel_cmd.resample_heading_command(
            len(env_ids), self._curriculum_step, self.cfg.curriculum_coeff,
            self.cfg.vel_cmd_speed_range_init, self.cfg.vel_cmd_speed_range_final,
            self.cfg.vel_cmd_heading_range, self.device,
        )

    # === Step =================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._curriculum_step += 1
        self._update_push()

        self.vel_cmd_timers -= self.step_dt
        vel_needs = self.vel_cmd_timers <= 0.0

        if self.cfg.task_mode == "velocity":
            if vel_needs.any():
                vel_ids = torch.where(vel_needs)[0]
                self._resample_heading_command(vel_ids)
                dr = self.cfg.vel_cmd_resample_time_range
                self.vel_cmd_timers[vel_ids] = torch.rand(len(vel_ids), device=self.device) * (dr[1] - dr[0]) + dr[0]
            self._heading_to_body_vel()
        else:  # navigation
            if vel_needs.any():
                vel_ids = torch.where(vel_needs)[0]
                new_cmds = nav_cmd.velocity_toward_target(
                    vel_ids, self.robot.data.root_pos_w[:, :2], self.target_pos,
                    self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B,
                    self._curriculum_step, self.cfg.curriculum_coeff,
                    self.cfg.vel_cmd_speed_range_init, self.cfg.vel_cmd_speed_range_final,
                    self.device,
                )
                self.vel_commands[vel_ids] = new_cmds
                dr = self.cfg.vel_cmd_resample_time_range
                self.vel_cmd_timers[vel_ids] = torch.rand(len(vel_ids), device=self.device) * (dr[1] - dr[0]) + dr[0]
            else:
                # Update toward-target vel every step (heading changes as robot moves)
                new_cmds = nav_cmd.velocity_toward_target(
                    torch.arange(self.num_envs, device=self.device),
                    self.robot.data.root_pos_w[:, :2], self.target_pos,
                    self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B,
                    self._curriculum_step, self.cfg.curriculum_coeff,
                    self.cfg.vel_cmd_speed_range_init, self.cfg.vel_cmd_speed_range_final,
                    self.device,
                )
                self.vel_commands = new_cmds

        # Markers after commands are fresh
        if self.render_mode != "headless":
            self._update_markers()

    def _apply_action(self) -> None:
        actions = torch.clamp(self.actions, -1.0, 1.0)
        leg_actions = actions * self.cfg.leg_action_scale

        default_pos = self.robot.data.default_joint_pos[:, self.dof_idx]
        target_pos = default_pos.clone()
        target_pos[:, :12] += leg_actions
        self.robot.set_joint_position_target(target_pos, joint_ids=self.dof_idx)

    # === Observations =========================================================

    def _get_observations(self) -> dict[str, torch.Tensor]:
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
        default_pos = self.robot.data.default_joint_pos[:, self.dof_idx]
        joint_pos_rel = joint_pos - default_pos
        joint_vel_rel = self.robot.data.joint_vel[:, self.dof_idx]
        prev_actions = self.prev_actions
        vel_cmds = self.vel_commands
        projected_gravity = self.robot.data.projected_gravity_b
        base_height = torch.clamp(self.robot.data.root_pos_w[:, 2:3], max=0.4)
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Observation noise (official Go2 domain randomization)
        if self.cfg.obs_noise_base_lin_vel:
            add_observation_noise(base_lin_vel, self.cfg.obs_noise_base_lin_vel)
            add_observation_noise(base_ang_vel, self.cfg.obs_noise_base_ang_vel)
            add_observation_noise(projected_gravity, self.cfg.obs_noise_projected_gravity)
            add_observation_noise(joint_pos_rel, self.cfg.obs_noise_joint_pos)
            add_observation_noise(joint_vel_rel, self.cfg.obs_noise_joint_vel)

        # Build policy observation (61-dim velocity / 67-dim navigation)
        if self.cfg.task_mode == "velocity":
            policy_obs = vel_obs.build_policy_obs(
                base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel,
                prev_actions, vel_cmds, projected_gravity, base_height,
            )  # (B, 61)
        else:
            policy_obs = nav_obs.build_policy_obs(
                base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel,
                prev_actions, vel_cmds, projected_gravity, base_height,
                self.target_pos, root_pos, root_quat, self.robot.data.FORWARD_VEC_B,
            )  # (B, 67)

        # History stacking
        hlen = self.cfg.num_obs_history_steps
        self.obs_history[:, self.obs_history_idx, :] = policy_obs
        self.obs_history_idx = (self.obs_history_idx + 1) % hlen
        obs_stacked = torch.cat(
            [self.obs_history[:, (self.obs_history_idx + i) % hlen, :] for i in range(hlen)], dim=-1
        )  # (B, 640)

        # Privileged critic obs (lin_vel already in policy, add torques + contacts)
        joint_torques = self.robot.data.applied_torque[:, self.dof_idx]
        foot_h = self.robot.data.body_state_w[:, self.foot_body_idx, 2]
        foot_contacts = (foot_h < 0.03).float()

        critic_obs = torch.cat([policy_obs, joint_torques, foot_contacts], dim=-1)  # (B, 89)
        return {"obs": obs_stacked, "critic": critic_obs}

    # === Rewards ==============================================================

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        applied_torque = self.robot.data.applied_torque[:, self.dof_idx]

        actions = self.actions
        prev_actions = self.prev_actions
        leg_actions = actions[:, :12]
        prev_leg_actions = prev_actions[:, :12]

        vel_cmd = self.vel_commands

        leg_joint_vel = joint_vel[:, :12]
        leg_joint_acc = (leg_joint_vel - self.prev_leg_dof_vel) / self.step_dt
        leg_torques = applied_torque[:, :12]

        default_joint_pos = self.robot.data.default_joint_pos[:, self.dof_idx]

        # === Reward terms (Go2Arm_Lab locomotion baseline) ===
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Velocity tracking (velocity mode only — this IS the task)
        if self.cfg.task_mode == "velocity":
            rewards += self.cfg.rew_lin_vel_tracking * lin_vel_tracking_exp(
                vel_cmd[:, :2], root_lin_vel_b[:, :2], self.cfg.lin_vel_tracking_std)
            rewards += self.cfg.rew_ang_vel_tracking * ang_vel_tracking_exp(
                vel_cmd[:, 2], root_ang_vel_b[:, 2], self.cfg.ang_vel_tracking_std)

        # Stability
        rewards += self.cfg.rew_lin_vel_z * lin_vel_z_l2(root_lin_vel_b[:, 2])
        rewards += self.cfg.rew_ang_vel_xy * ang_vel_xy_l2(root_ang_vel_b[:, :2])

        # Effort (legs only)
        rewards += self.cfg.rew_dof_torques * dof_torques_l2(leg_torques)
        rewards += self.cfg.rew_dof_acc * dof_acc_l2(leg_joint_acc)

        # Action smoothness
        rewards += self.cfg.rew_action_rate * action_rate_l2(leg_actions, prev_leg_actions)

        # Posture
        rewards += self.cfg.rew_flat_orientation * flat_orientation_l2(projected_gravity)

        # Foot clearance (official Go2: simple instant + EMA)
        foot_h = self.robot.data.body_state_w[:, self.foot_body_idx, 2]
        foot_in_air = (foot_h > 0.04).float()
        if not hasattr(self, "_foot_air_accum"):
            self._foot_air_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_air_accum = 0.8 * self._foot_air_accum + 0.2 * foot_in_air
        rewards += self.cfg.rew_feet_air_time * torch.sum(self._foot_air_accum, dim=-1)

        # Gait posture (LegoManip: symmetry + rhythm)
        rewards += self.cfg.rew_joint_mirror * gait_trot_penalty(joint_pos)
        # Air time variance: penalize irregular stepping (one foot lifted much longer than others)
        air_var = torch.var(self._foot_air_accum, dim=-1)
        rewards += self.cfg.rew_air_time_variance * air_var
        # Foot slide: penalize foot velocity when near ground
        foot_vel = self.robot.data.body_lin_vel_w[:, self.foot_body_idx]  # (B, 4, 3)
        foot_slide = torch.norm(foot_vel[:, :, :2], dim=-1) * (foot_h < 0.03).float()
        rewards += self.cfg.rew_feet_slide * torch.sum(foot_slide, dim=-1)
        # Long air: penalize any foot lifted >70% of the time
        rewards += self.cfg.rew_feet_long_air * torch.sum((self._foot_air_accum > 0.7).float(), dim=-1)

        # Navigation rewards
        if self.cfg.task_mode == "navigation":
            target_dist = torch.norm(self.target_pos[:, :2] - root_pos_w[:, :2], dim=-1)
            rewards += self.cfg.rew_target_progress * nav_rewards.target_progress(
                self._prev_distance, target_dist)
            rewards += self.cfg.rew_target_reach * nav_rewards.target_reach(
                target_dist, self.cfg.target_reach_threshold)
            fwd_w = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)[:, :2]
            tgt_dir = (self.target_pos[:, :2] - root_pos_w[:, :2]) / (target_dist.unsqueeze(-1) + 1e-6)
            rewards += self.cfg.rew_target_alignment * nav_rewards.target_alignment(fwd_w, tgt_dir)
            self._prev_distance = target_dist

        # TODO(arm): restore ee_pos_tracking, ee_ori_tracking, ee_action_rate, ee_action_smoothness

        self.prev_actions = actions.clone()
        self.prev_leg_dof_vel = leg_joint_vel.clone()
        return torch.clamp(rewards, -20.0, 500.0)

    # === Termination ==========================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        projected_gravity = self.robot.data.projected_gravity_b
        tilt_angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
        bad_orientation = tilt_angle > 0.75

        base_height = self.robot.data.body_state_w[:, self.base_body_idx, 2]
        base_too_low = base_height < 0.05
        base_collapsed = base_height < 0.22
        body_inverted = projected_gravity[:, 2] > 0.2

        terminated = bad_orientation | base_too_low | base_collapsed | body_inverted
        return terminated, time_out

    # === Reset ================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(dtype=torch.long, device=self.device)
        else:
            ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        ids = cast(torch.Tensor, ids)
        if ids.numel() == 0:
            return

        super()._reset_idx(ids)

        # Root state randomization
        default_root = randomize_root_state(
            self.robot.data.default_root_state, self.scene.env_origins, ids,
            self.cfg.dr_pose_range, self.cfg.dr_velocity_range, self.device,
        )

        # Joint position randomization
        joint_pos = randomize_joint_positions(
            self.robot.data.default_joint_pos, ids,
            self.cfg.dr_joint_pos_range, self.device,
        )

        self.robot.write_root_state_to_sim(default_root, ids)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), None, ids)

        # Reset internal
        self.actions[ids] = 0.0
        self.prev_actions[ids] = 0.0
        self.prev_leg_dof_vel[ids] = 0.0
        self.obs_history[ids] = 0.0

        # Navigation: generate target point
        if self.cfg.task_mode == "navigation":
            robot_xy = self.robot.data.root_pos_w[:, :2]
            self.target_pos[ids] = nav_cmd.sample_target_points(
                ids, robot_xy, self.cfg.target_distance_range, self.device)
            self._prev_distance[ids] = torch.norm(
                self.target_pos[ids, :2] - robot_xy[ids], dim=-1)

        # Initial heading command (world-frame heading + speed)
        self._resample_heading_command(ids)
        self._heading_to_body_vel()
        dr = self.cfg.vel_cmd_resample_time_range
        self.vel_cmd_timers[ids] = torch.rand(ids.numel(), device=self.device) * (dr[1] - dr[0]) + dr[0]

        self._init_push_timers(ids)
        self._update_markers()

    # === Push =================================================================

    def _init_push_timers(self, env_ids: torch.Tensor | None = None) -> None:
        _env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.push_timers[_env_ids] = init_push_timers(
            self.num_envs, env_ids, self.cfg.dr_push_interval_range, self.device)

    def _update_push(self) -> None:
        self.push_timers -= self.step_dt
        needs_push = self.push_timers <= 0.0
        if needs_push.any():
            push_ids = torch.where(needs_push)[0]
            push_vel = apply_push_velocity(
                self.num_envs, push_ids, self.cfg.dr_push_range, self.device)
            self.robot.write_root_velocity_to_sim(push_vel[push_ids], push_ids)
            self.push_timers[push_ids] = init_push_timers(
                self.num_envs, push_ids, self.cfg.dr_push_interval_range, self.device)

    # === Viz ==================================================================

    def _update_markers(self) -> None:
        from .utils import visualize_tools as _vt
        _vt.update_dogarm_markers(self)
