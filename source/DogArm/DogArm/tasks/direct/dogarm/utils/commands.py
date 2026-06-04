# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generation: target point + world-frame EE pose anchored at target."""

from __future__ import annotations

import math

import torch


def sample_target_points(
    env_ids: torch.Tensor,
    robot_xy: torch.Tensor,
    distance_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Generate random target points around the robot in world frame.

    Args:
        env_ids: Indices of environments to sample for.
        robot_xy: Robot XY positions in world frame, shape (num_envs, 2).
        distance_range: (min_dist, max_dist) in meters.
        device: Torch device.

    Returns:
        Target positions in world frame, shape (len(env_ids), 3).
    """
    n = len(env_ids)
    r = torch.empty(n, 2, device=device)
    r[:, 0].uniform_(*distance_range)
    r[:, 1].uniform_(-torch.pi, torch.pi)

    target_pos = torch.zeros(n, 3, device=device)
    target_pos[:, 0] = robot_xy[env_ids, 0] + r[:, 0] * torch.cos(r[:, 1])
    target_pos[:, 1] = robot_xy[env_ids, 1] + r[:, 0] * torch.sin(r[:, 1])
    target_pos[:, 2] = 0.0
    return target_pos


def sample_velocity_toward_target(
    env_ids: torch.Tensor,
    robot_xy: torch.Tensor,
    target_pos: torch.Tensor,
    robot_forward: torch.Tensor,
    curriculum_step: int,
    num_env_steps: int,
    curriculum_coeff: int,
    ranges_init: dict,
    ranges_final: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate velocity commands that point toward the target point.

    Returns:
        Velocity commands [vx, vy, wz] in base frame, shape (len(env_ids), 3).
    """
    n = len(env_ids)

    t = torch.tensor(
        curriculum_step / (num_env_steps * curriculum_coeff), device=device
    ).clamp(0.0, 1.0)

    vx_lo = ranges_init["lin_vel_x"][0] * (1 - t) + ranges_final["lin_vel_x"][0] * t
    vx_hi = ranges_init["lin_vel_x"][1] * (1 - t) + ranges_final["lin_vel_x"][1] * t

    to_target_w = target_pos[env_ids, :2] - robot_xy[env_ids, :2]
    dist = torch.norm(to_target_w, dim=-1, keepdim=True)
    target_dir_w = to_target_w / (dist + 1e-6)

    fwd = robot_forward[env_ids]
    left = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)

    body_vx = torch.sum(target_dir_w * fwd, dim=-1)
    body_vy = torch.sum(target_dir_w * left, dim=-1)

    forward_speed = torch.empty(n, device=device).uniform_(vx_lo, vx_hi)

    cmd = torch.zeros(n, 3, device=device)
    # Forward: always some forward + more when facing target
    cmd[:, 0] = forward_speed * body_vx.clamp(min=0.1)
    # Lateral: steer toward target direction
    cmd[:, 1] = forward_speed * body_vy * 0.3
    # Angular: turn toward target
    cmd[:, 2] = -body_vy * 0.6

    return torch.clamp(cmd, -1.0, 1.0)


def sample_ee_pose_at_target(
    env_ids: torch.Tensor,
    target_pos: torch.Tensor,
    sphere_center_z: float,
    arm_length: float,
    rpy_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Sample EE target pose in WORLD frame on a hollow half-sphere surface.

    The sphere center is at (target_x, target_y, sphere_center_z) where
    sphere_center_z is the arm base default height. The EE target is sampled
    uniformly on the UPPER half-sphere surface (z ≥ center_z), representing
    the maximum reach of the arm when the robot is at the target point.

    Args:
        env_ids: Indices to sample for.
        target_pos: Target point positions (world), shape (num_envs, 3).
        sphere_center_z: Z-height of the sphere center [m].
        arm_length: Sphere radius (= arm reach) [m].
        rpy_range: Dict with 'roll', 'pitch', 'yaw' ranges.
        device: Torch device.

    Returns:
        EE pose [px_w, py_w, pz_w, qw, qx, qy, qz] in WORLD frame, shape (N, 7).
    """
    n = len(env_ids)
    rng = torch.empty(n, 5, device=device)

    # --- Position: hollow upper half-sphere surface ---
    # Uniform on sphere surface: azimuth uniform [0, 2π], cos(elevation) uniform
    azimuth = rng[:, 0].uniform_(0.0, 2.0 * math.pi)
    # φ ∈ [0, π/2] for upper half: cos(φ) ∈ [0, 1]
    cos_phi = rng[:, 1].uniform_(0.0, 1.0)
    phi = torch.acos(cos_phi)  # φ ∈ [0, π/2]

    # Fixed radius = arm_length (hollow surface, not filled)
    r = arm_length

    # Sphere center at target XY, arm-base height Z
    center_x = target_pos[env_ids, 0]
    center_y = target_pos[env_ids, 1]
    center_z = sphere_center_z

    pos_w = torch.zeros(n, 3, device=device)
    pos_w[:, 0] = center_x + r * torch.cos(phi) * torch.cos(azimuth)
    pos_w[:, 1] = center_y + r * torch.cos(phi) * torch.sin(azimuth)
    pos_w[:, 2] = center_z + r * torch.sin(phi)  # always ≥ center_z (upper half)

    # --- Orientation: uniform RPY ---
    roll = rng[:, 3].uniform_(*rpy_range["roll"])
    pitch = rng[:, 4].uniform_(*rpy_range["pitch"])
    yaw = torch.empty(n, device=device).uniform_(*rpy_range["yaw"])

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat_w = torch.stack([qw, qx, qy, qz], dim=-1)
    return torch.cat([pos_w, quat_w], dim=-1)  # (N, 7)
