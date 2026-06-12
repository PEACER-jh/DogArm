"""Velocity tracking command generation (world-frame heading → body-frame vel)."""

from __future__ import annotations

import math

import torch


def heading_to_body_vel(
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
    cmd_heading_w: torch.Tensor,
    cmd_speed: torch.Tensor,
    cmd_lateral: torch.Tensor,
    vel_commands: torch.Tensor,
) -> None:
    """Convert world-frame heading + speed → body-frame velocity command (in-place).

    Args:
        robot_quat_w: (B, 4) world-frame quaternion [w, x, y, z].
        forward_vec_b: (3,) body-frame forward vector.
        cmd_heading_w: (B,) world-frame heading [rad].
        cmd_speed: (B,) forward speed [m/s].
        cmd_lateral: (B,) lateral speed [m/s] (+ right, - left).
        vel_commands: (B, 3) write target [vx, vy, wz].
    """
    import isaaclab.utils.math as _mu

    fwd = _mu.quat_apply(robot_quat_w, forward_vec_b)
    robot_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
    heading_err = cmd_heading_w - robot_yaw
    heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    vel_commands[:, 0] = cmd_speed * heading_err.cos().clamp(min=0.2)
    vel_commands[:, 1] = cmd_lateral
    vel_commands[:, 2] = torch.clamp(heading_err, -1.0, 1.0)


def resample_heading_command(
    n: int,
    curriculum_step: int,
    curriculum_coeff: int,
    speed_range_init: tuple[float, float],
    speed_range_final: tuple[float, float],
    lateral_range: tuple[float, float],
    heading_range: tuple[float, float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample new world-frame heading + speed + lateral.

    Returns (speed, heading_w, lateral_speed).
    Small commands (|speed|<0.2 and |lateral|<0.2) are zeroed out
    following HIMLoco practice — the robot must either move clearly
    or stand still, not shuffle at 0.02 m/s.
    """
    t_val = min(1.0, curriculum_step / (1 * curriculum_coeff))
    sp_lo = speed_range_init[0] * (1 - t_val) + speed_range_final[0] * t_val
    sp_hi = speed_range_init[1] * (1 - t_val) + speed_range_final[1] * t_val
    cmd_speed = torch.rand(n, device=device) * (sp_hi - sp_lo) + sp_lo
    cmd_lateral = (torch.rand(n, device=device) * 2 - 1) * (
        lateral_range[1] * (1 - t_val) + lateral_range[0] * t_val
        if t_val < 0.5
        else lateral_range[1]
    )
    cmd_heading_w = (
        torch.rand(n, device=device) * (heading_range[1] - heading_range[0])
        + heading_range[0]
    )

    # Zero out tiny commands (HIMLoco-style): ‖(vx, vy)‖ < 0.2 → 0
    speed_norm = torch.sqrt(cmd_speed**2 + cmd_lateral**2)
    zero_mask = speed_norm < 0.2
    cmd_speed[zero_mask] = 0.0
    cmd_lateral[zero_mask] = 0.0

    return cmd_speed, cmd_heading_w, cmd_lateral
