"""Align mode observation: velocity base + target-relative + EE pose command."""

from __future__ import annotations

import torch


def target_to_body_frame(
    target_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute target-relative observations in body frame.

    Returns: (to_target_b, target_dist, cos_yaw, sin_yaw).
    """
    import isaaclab.utils.math as _mu

    to_target_w = target_pos_w - root_pos_w
    inv_quat = _mu.quat_conjugate(robot_quat_w)
    to_target_b = _mu.quat_apply(inv_quat, to_target_w)

    target_dist = torch.norm(to_target_w[:, :2], dim=-1, keepdim=True)
    fwd_w = _mu.quat_apply(robot_quat_w, forward_vec_b)[:, :2]
    tgt_dir_w = to_target_w[:, :2] / (target_dist + 1e-6)
    cos_yaw = torch.sum(fwd_w * tgt_dir_w, dim=-1, keepdim=True)
    sin_yaw = (fwd_w[:, 0] * tgt_dir_w[:, 1] - fwd_w[:, 1] * tgt_dir_w[:, 0]).unsqueeze(-1)

    return to_target_b, target_dist, cos_yaw, sin_yaw


def build_policy_obs(
    base_ang_vel: torch.Tensor,
    base_lin_vel: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel_rel: torch.Tensor,
    prev_actions: torch.Tensor,
    vel_commands: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    target_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
    ee_pose_cmd_b: torch.Tensor,   # (B, 7) body-frame (already in link0 frame)
) -> torch.Tensor:
    """Assemble policy observation: 67 base + 6 target + 7 body-frame EE = 80 dims."""
    tgt_b, tgt_dist, cos_y, sin_y = target_to_body_frame(
        target_pos_w, root_pos_w, robot_quat_w, forward_vec_b,
    )
    return torch.cat([
        base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel,
        prev_actions, vel_commands, projected_gravity, base_height,
        tgt_b, tgt_dist, cos_y, sin_y,
        ee_pose_cmd_b,
    ], dim=-1)  # (B, 80)
