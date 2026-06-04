"""Navigation observation helpers: body-frame target conversion."""

from __future__ import annotations

import torch


def target_to_body_frame(
    target_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute target-relative observations in body frame.

    Returns:
        to_target_b: (B, 3) body-frame vector to target
        target_dist: (B, 1) distance to target
        cos_yaw: (B, 1) cos of angle to target
        sin_yaw: (B, 1) sin of angle to target
    """
    import isaaclab.utils.math as _mu

    to_target_w = target_pos_w - root_pos_w  # (B, 3)
    # Rotate to body frame
    inv_quat = _mu.quat_conjugate(robot_quat_w)
    to_target_b = _mu.quat_apply(inv_quat, to_target_w)

    target_dist = torch.norm(to_target_w[:, :2], dim=-1, keepdim=True)

    # Yaw error from forward to target direction
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
    root_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
) -> torch.Tensor:
    """Assemble 67-dim policy observation for navigation (velocity obs + 6 target dims)."""
    tgt_b, tgt_dist, cos_y, sin_y = target_to_body_frame(
        target_pos_w, root_pos_w, root_quat_w, forward_vec_b,
    )
    return torch.cat([
        base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel,
        prev_actions, vel_commands, projected_gravity, base_height,
        tgt_b, tgt_dist, cos_y, sin_y,
    ], dim=-1)  # (B, 67)
