"""Velocity mode observation assembly."""

from __future__ import annotations

import torch


def build_policy_obs(
    base_ang_vel: torch.Tensor,   # (B, 3)
    base_lin_vel: torch.Tensor,   # (B, 3)
    joint_pos_rel: torch.Tensor,  # (B, 18)
    joint_vel_rel: torch.Tensor,  # (B, 18)
    prev_actions: torch.Tensor,   # (B, 12)
    vel_commands: torch.Tensor,   # (B, 3)
    projected_gravity: torch.Tensor,  # (B, 3)
    base_height: torch.Tensor,    # (B, 1)
) -> torch.Tensor:
    """Assemble 61-dim policy observation for velocity tracking."""
    return torch.cat([
        base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel,
        prev_actions, vel_commands, projected_gravity, base_height,
    ], dim=-1)  # (B, 61)
