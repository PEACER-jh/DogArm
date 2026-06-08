"""Align mode rewards: EE position & orientation tracking."""

from __future__ import annotations

import torch


@torch.jit.script
def ee_pos_tracking_exp(
    curr_pos_w: torch.Tensor, des_pos_w: torch.Tensor, std: float
) -> torch.Tensor:
    """exp(-‖curr − des‖₁ / std) — higher when closer."""
    return torch.exp(-torch.sum(torch.abs(curr_pos_w - des_pos_w), dim=1) / std)


@torch.jit.script
def ee_ori_tracking(
    curr_quat_w: torch.Tensor, des_quat_w: torch.Tensor
) -> torch.Tensor:
    """Shortest-path quaternion error magnitude (0 = perfect)."""
    q_err_w = (
        curr_quat_w[:, 0] * des_quat_w[:, 0]
        + curr_quat_w[:, 1] * des_quat_w[:, 1]
        + curr_quat_w[:, 2] * des_quat_w[:, 2]
        + curr_quat_w[:, 3] * des_quat_w[:, 3]
    )
    q_err_w = torch.clamp(torch.abs(q_err_w), -1.0, 1.0)
    return 2.0 * torch.acos(q_err_w)
