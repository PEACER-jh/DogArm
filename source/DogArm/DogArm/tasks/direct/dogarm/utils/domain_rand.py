"""Domain randomization functions for DogArm.

Called from env._reset_idx, env._update_push, env._get_observations.
"""

from __future__ import annotations

import torch


def randomize_root_state(
    default_root: torch.Tensor,
    env_origins: torch.Tensor,
    env_ids: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Randomize root state (position, yaw, velocity) for the given envs.

    Args:
        default_root: Default root state from robot data, shape (N, 13).
        env_origins: Environment origins, shape (num_envs, 3).
        env_ids: Indices of envs to randomize.
        pose_range: Dict with 'x', 'y', 'yaw' ranges.
        velocity_range: Dict with 'x', 'y', 'z' ranges (optional).
        device: Torch device.

    Returns:
        Randomized root state, shape (len(env_ids), 13).
    """
    root_state = default_root[env_ids].clone()
    root_state[:, :3] += env_origins[env_ids]
    n = len(env_ids)

    # XY position
    for i, key in enumerate(("x", "y")):
        lo, hi = pose_range[key]
        root_state[:, i] += torch.rand(n, device=device) * (hi - lo) + lo

    # Yaw → quaternion
    yaw_lo, yaw_hi = pose_range["yaw"]
    rand_yaw = torch.rand(n, device=device) * (yaw_hi - yaw_lo) + yaw_lo
    root_state[:, 3] = torch.cos(rand_yaw / 2.0)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(rand_yaw / 2.0)

    # Velocities
    for i, key in enumerate(("x", "y", "z")):
        if key in velocity_range:
            lo, hi = velocity_range[key]
            root_state[:, 7 + i] += torch.rand(n, device=device) * (hi - lo) + lo

    return root_state


def randomize_joint_positions(
    default_joint_pos: torch.Tensor,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Scale default joint positions uniformly for the given envs.

    Args:
        default_joint_pos: Default joint positions, shape (N, dof).
        env_ids: Indices of envs to randomize.
        position_range: (min_scale, max_scale) multiplier.
        device: Torch device.

    Returns:
        Randomized joint positions, shape (len(env_ids), dof).
    """
    joint_pos = default_joint_pos[env_ids].clone()
    lo, hi = position_range
    scale = torch.rand(len(env_ids), 1, device=device) * (hi - lo) + lo
    return joint_pos * scale


def init_push_timers(
    num_envs: int,
    env_ids: torch.Tensor | None,
    interval_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Initialize or reset push timers with random intervals.

    Returns timer values for the specified envs (updated in-place elsewhere).
    """
    n = num_envs if env_ids is None else len(env_ids)
    lo, hi = interval_range
    return torch.rand(n, device=device) * (hi - lo) + lo


def apply_push_velocity(
    num_envs: int,
    env_ids: torch.Tensor,
    velocity_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate random push velocity for the base.

    Returns push velocity tensor, shape (num_envs, 6). Only non-pushed envs remain 0.
    """
    n = len(env_ids)
    push_vel = torch.zeros(num_envs, 6, device=device)
    for i, key in enumerate(("x", "y")):
        lo, hi = velocity_range[key]
        push_vel[env_ids, i] = torch.rand(n, device=device) * (hi - lo) + lo
    return push_vel


def add_observation_noise(
    tensor: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Apply uniform additive noise to observation tensor (in-place).

    Args:
        tensor: Observation tensor.
        noise_scale: Max absolute noise value.

    Returns:
        The same tensor with noise added (modified in-place).
    """
    tensor += (torch.rand_like(tensor) * 2 - 1) * noise_scale
    return tensor
