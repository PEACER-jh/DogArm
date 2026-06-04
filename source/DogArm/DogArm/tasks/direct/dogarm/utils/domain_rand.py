# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization functions for the DogArm DirectRL environment.

Since we're using DirectRL (not ManagerBasedRL with EventManager),
all randomization is applied manually in _reset_idx and _pre_physics_step.
Adapted from Go2Arm_Lab and LeggedManip_Lab domain randomization events.
"""

from __future__ import annotations

import torch


@torch.jit.script
def randomize_physics_material(
    static_friction: torch.Tensor,
    dynamic_friction: torch.Tensor,
    static_range: tuple[float, float],
    dynamic_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomize physics material properties.

    Args:
        static_friction: Static friction tensor, shape (B,) or scalar.
        dynamic_friction: Dynamic friction tensor, shape (B,) or scalar.
        static_range: (min, max) for static friction.
        dynamic_range: (min, max) for dynamic friction.

    Returns:
        Tuple of randomized (static_friction, dynamic_friction).
    """
    # Note: In DirectRL, physics material randomization is applied
    # through the articulation/view API, not through tensors.
    # This function is a placeholder for API-level randomization.
    return static_friction, dynamic_friction


def randomize_root_state(
    default_root_state: torch.Tensor,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Randomize root state (position + orientation + velocity) on reset.

    Args:
        default_root_state: Default root state, shape (N, 13).
        env_ids: Environment indices to randomize.
        env_origins: Environment origins, shape (num_envs, 3).
        pose_range: Dict with keys 'x', 'y', 'yaw' ranges.
        velocity_range: Dict with velocity range keys.
        device: Torch device.

    Returns:
        Randomized root state tensor, shape (len(env_ids), 13).
    """
    n = len(env_ids)
    root_state = default_root_state[env_ids].clone()

    # Add environment origins
    root_state[:, :3] = root_state[:, :3] + env_origins[env_ids]

    # Randomize XY position
    root_state[:, 0] += (
        torch.rand(n, device=device) * (pose_range["x"][1] - pose_range["x"][0])
        + pose_range["x"][0]
    )
    root_state[:, 1] += (
        torch.rand(n, device=device) * (pose_range["y"][1] - pose_range["y"][0])
        + pose_range["y"][0]
    )

    # Randomize yaw
    rand_yaw = (
        torch.rand(n, device=device) * (pose_range["yaw"][1] - pose_range["yaw"][0])
        + pose_range["yaw"][0]
    )
    # Convert yaw to quaternion
    root_state[:, 3] = torch.cos(rand_yaw / 2.0)  # qw
    root_state[:, 4] = 0.0  # qx
    root_state[:, 5] = 0.0  # qy
    root_state[:, 6] = torch.sin(rand_yaw / 2.0)  # qz

    # Randomize velocities
    for i, key in enumerate(["x", "y", "z"]):
        if key in velocity_range:
            lo, hi = velocity_range[key]
            root_state[:, 7 + i] = (
                torch.rand(n, device=device) * (hi - lo) + lo
            )

    return root_state


def randomize_joint_positions(
    default_joint_pos: torch.Tensor,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    num_dofs: int,
    device: torch.device,
) -> torch.Tensor:
    """Randomize joint positions by scaling default positions.

    Args:
        default_joint_pos: Default joint positions, shape (N, num_dofs).
        env_ids: Environment indices to randomize.
        position_range: (min_scale, max_scale) for default positions.
        num_dofs: Number of degrees of freedom.
        device: Torch device.

    Returns:
        Randomized joint positions, shape (len(env_ids), num_dofs).
    """
    n = len(env_ids)
    joint_pos = default_joint_pos[env_ids].clone()

    # Uniformly scale default positions
    rand_scale = (
        torch.rand(n, 1, device=device)
        * (position_range[1] - position_range[0])
        + position_range[0]
    )
    joint_pos = joint_pos * rand_scale

    return joint_pos


def randomize_actuator_gains(
    stiffness: float,
    damping: float,
    stiffness_range: tuple[float, float],
    damping_range: tuple[float, float],
    device: torch.device,
) -> tuple[float, float]:
    """Randomize actuator stiffness and damping gains.

    In DirectRL, this is applied through the actuator API before reset.

    Args:
        stiffness: Base stiffness value.
        damping: Base damping value.
        stiffness_range: (min_scale, max_scale) for stiffness.
        damping_range: (min_scale, max_scale) for damping.
        device: Torch device.

    Returns:
        Tuple of randomized (stiffness, damping).
    """
    # This is a placeholder — actual randomization happens via the
    # articulation/actuator API in the environment.
    return stiffness, damping


def generate_push_velocity(
    num_envs: int,
    velocity_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate random push velocities for the robot base.

    Args:
        num_envs: Number of environments.
        velocity_range: Dict with 'x' and 'y' velocity ranges.
        device: Torch device.

    Returns:
        Push velocity tensor, shape (num_envs, 6) — linear + angular.
    """
    push_vel = torch.zeros(num_envs, 6, device=device)
    for i, key in enumerate(["x", "y"]):
        if key in velocity_range:
            lo, hi = velocity_range[key]
            push_vel[:, i] = torch.rand(num_envs, device=device) * (hi - lo) + lo
    return push_vel
