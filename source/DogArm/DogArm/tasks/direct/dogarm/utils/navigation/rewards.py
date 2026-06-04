"""Navigation task rewards: target progress, reach, alignment."""

from __future__ import annotations

import torch


@torch.jit.script
def target_progress(prev_dist: torch.Tensor, curr_dist: torch.Tensor) -> torch.Tensor:
    """Reward for reducing distance to target."""
    return prev_dist - curr_dist


@torch.jit.script
def target_reach(curr_dist: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binary bonus for reaching within threshold."""
    return (curr_dist < threshold).float()


@torch.jit.script
def target_alignment(
    forward_w: torch.Tensor, tgt_dir_w: torch.Tensor
) -> torch.Tensor:
    """cos(angle) between robot forward and target direction."""
    return torch.sum(forward_w * tgt_dir_w, dim=-1)
