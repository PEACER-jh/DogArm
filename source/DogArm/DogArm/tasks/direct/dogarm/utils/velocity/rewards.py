"""Velocity tracking reward functions."""

from __future__ import annotations

import torch


@torch.jit.script
def lin_vel_tracking_exp(
    cmd_vel_xy: torch.Tensor, body_vel_xy: torch.Tensor, std: float
) -> torch.Tensor:
    error = torch.sum(torch.square(cmd_vel_xy - body_vel_xy), dim=1)
    return torch.exp(-error / (std**2))


@torch.jit.script
def ang_vel_tracking_exp(
    cmd_ang_vel_z: torch.Tensor, body_ang_vel_z: torch.Tensor, std: float
) -> torch.Tensor:
    error = torch.square(cmd_ang_vel_z - body_ang_vel_z)
    return torch.exp(-error / (std**2))
