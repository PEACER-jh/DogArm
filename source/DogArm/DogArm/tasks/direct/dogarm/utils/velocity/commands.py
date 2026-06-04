"""Velocity tracking command generation (world-frame heading → body-frame vel)."""

from __future__ import annotations

import math

import torch


def heading_to_body_vel(
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,  # (3,) body-frame forward vector
    cmd_heading_w: torch.Tensor,  # (B,) world-frame heading
    cmd_speed: torch.Tensor,  # (B,) forward speed
    vel_commands: torch.Tensor,  # (B, 3) write target
) -> None:
    """Convert world-frame heading + speed → body-frame velocity command (in-place)."""
    import isaaclab.utils.math as _mu

    fwd = _mu.quat_apply(robot_quat_w, forward_vec_b)
    robot_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
    heading_err = cmd_heading_w - robot_yaw
    heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    vel_commands[:, 0] = cmd_speed * heading_err.cos().clamp(min=0.2)
    vel_commands[:, 1] = 0.0
    vel_commands[:, 2] = torch.clamp(heading_err, -1.0, 1.0)


def resample_heading_command(
    n: int,
    curriculum_step: int,
    curriculum_coeff: int,
    speed_range_init: tuple[float, float],
    speed_range_final: tuple[float, float],
    heading_range: tuple[float, float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample new world-frame heading + speed. Returns (speed, heading_w)."""
    t_val = min(1.0, curriculum_step / (1 * curriculum_coeff))
    sp_lo = speed_range_init[0] * (1 - t_val) + speed_range_final[0] * t_val
    sp_hi = speed_range_init[1] * (1 - t_val) + speed_range_final[1] * t_val
    cmd_speed = torch.rand(n, device=device) * (sp_hi - sp_lo) + sp_lo
    cmd_heading_w = torch.rand(n, device=device) * (heading_range[1] - heading_range[0]) + heading_range[0]
    return cmd_speed, cmd_heading_w
