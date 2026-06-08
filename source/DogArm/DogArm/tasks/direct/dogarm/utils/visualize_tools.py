# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization for DogArm: forward arrow + target markers + EE 3-axis frames.

Pattern references:
  - LearnIsaac: forward (cyan) + target_dir (yellow) + target_point (red sphere)
  - LeggedManip_Lab: 3-axis frame (FRAME_MARKER_CFG) for EE current + target pose

Locomotion-baseline mode (no target_pos / ee_pose_commands):
  - Only shows cyan forward arrow + EE current frame
Navigation+arm mode (target_pos / ee_pose_commands available):
  - Full markers: forward + target dir + target sphere + EE current + EE target
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def define_robot_markers(prim_path: str = "/Visuals/DogArm/RobotMarkers") -> VisualizationMarkers:
    """Create markers: forward (cyan), target_dir (yellow), target_point (red sphere)."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "target_dir": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            ),
            "target_point": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=cfg)


def define_ee_frame_markers(
    prim_path: str = "/Visuals/DogArm/EEFrame",
    scale: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> VisualizationMarkers:
    """Create 3-axis frame marker for EE pose (RGB: X=red, Y=green, Z=blue)."""
    cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    cfg.markers["frame"].scale = scale
    return VisualizationMarkers(cfg=cfg)


def update_dogarm_markers(
    env,
    robot_world_pos: torch.Tensor | None = None,
) -> None:
    """Update all markers.

    Locomotion baseline: only forward arrow + EE current frame.
    Full mode (target_pos + ee_pose_commands): all markers active.
    """
    n = env.num_envs
    device = env.device

    robot_pos = robot_world_pos[:, :3] if robot_world_pos is not None else env.robot.data.root_pos_w
    robot_quat = env.robot.data.root_quat_w
    arrow_base = robot_pos.clone()
    arrow_base[:, 2] += 0.5

    # === 1. Forward arrow (cyan) — always shown ===
    fwd_loc = arrow_base
    fwd_quat = robot_quat

    # === Check what features are active ===
    tgt = getattr(env, "target_pos", None)
    has_target = tgt is not None
    ee_tgt = getattr(env, "ee_pose_commands", None)
    has_ee_tgt = ee_tgt is not None and ee_tgt.abs().sum() > 0  # non-zero check

    # === 2. Direction arrow (yellow): target_dir or velocity command ===
    if has_target:
        # Target-point mode: yellow arrow points toward target
        to_tgt = tgt - robot_pos
        to_tgt_xy = to_tgt[:, :2]
        tgt_dist = torch.norm(to_tgt_xy, dim=-1)
        tgt_dir = to_tgt_xy / (tgt_dist.unsqueeze(-1) + 1e-6)
        tgt_yaw = torch.atan2(tgt_dir[:, 1], tgt_dir[:, 0])
        tgt_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, -1)
        tgt_quat = math_utils.quat_from_angle_axis(tgt_yaw, tgt_axis)
        tgt_loc = arrow_base
        tp_loc = tgt.clone()
        tp_loc[:, 2] = 0.2
        tp_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    else:
        # Locomotion baseline: yellow arrow = world-frame heading command
        heading_w = getattr(env, "cmd_heading_w", None)
        if heading_w is not None:
            cmd_yaw = heading_w  # (B,)
            vel_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, -1)
            tgt_quat = math_utils.quat_from_angle_axis(cmd_yaw, vel_axis)
        else:
            tgt_quat = robot_quat
        tgt_loc = arrow_base
        tp_loc = arrow_base
        tp_quat = tgt_quat

    # === 3. EE current pose frame ===
    ee_state = env.robot.data.body_link_pose_w[:, env.ee_body_idx]
    ee_curr_pos = ee_state[:, :3]
    ee_curr_quat = ee_state[:, 3:7]

    # === 4. EE target pose frame ===
    if has_ee_tgt and ee_tgt is not None:
        # Align mode: EE is already world-frame (anchored at target)
        # Other modes: EE might be body-frame, convert via link0
        link0_idx = getattr(env, "link0_body_idx", None)
        task_mode = getattr(env.cfg, "task_mode", "")
        if link0_idx is not None and task_mode != "align":
            link0_pose = env.robot.data.body_link_pose_w[:, link0_idx]
            ee_tgt_pos, ee_tgt_quat = math_utils.combine_frame_transforms(
                link0_pose[:, :3], link0_pose[:, 3:7],
                ee_tgt[:, :3], ee_tgt[:, 3:7],
            )
        else:
            ee_tgt_pos = ee_tgt[:, :3]
            ee_tgt_quat = ee_tgt[:, 3:7]
    else:
        ee_tgt_pos = ee_curr_pos
        ee_tgt_quat = ee_curr_quat

    # === Render robot markers ===
    if has_target:
        r_loc = torch.cat([fwd_loc, tgt_loc, tp_loc], dim=0)
        r_rot = torch.cat([fwd_quat, tgt_quat, tp_quat], dim=0)
        r_idx = torch.cat([
            torch.zeros(n, dtype=torch.int32, device=device),
            torch.ones(n, dtype=torch.int32, device=device),
            torch.full((n,), 2, dtype=torch.int32, device=device),
        ], dim=0)
    else:
        r_loc = torch.cat([fwd_loc, tgt_loc], dim=0)
        r_rot = torch.cat([fwd_quat, tgt_quat], dim=0)
        r_idx = torch.cat([
            torch.zeros(n, dtype=torch.int32, device=device),
            torch.ones(n, dtype=torch.int32, device=device),
        ], dim=0)
    env.robot_markers.visualize(r_loc, r_rot, marker_indices=r_idx)

    # === Render EE frame markers ===
    ee_curr_idx = torch.zeros(n, dtype=torch.int32, device=device)
    env.ee_curr_frame_markers.visualize(ee_curr_pos, ee_curr_quat, marker_indices=ee_curr_idx)
    if has_ee_tgt and ee_tgt is not None:
        ee_tgt_idx = torch.zeros(n, dtype=torch.int32, device=device)
        env.ee_tgt_frame_markers.visualize(ee_tgt_pos, ee_tgt_quat, marker_indices=ee_tgt_idx)
