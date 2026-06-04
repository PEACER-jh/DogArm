# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the DogArm DirectRL environment.

All functions are decorated with @torch.jit.script for performance.
IMPORTANT: Functions must be defined in dependency order for TorchScript.
Adapted from Go2Arm_Lab's mdp/rewards.py.
"""

from __future__ import annotations

import torch


# ==============================================================================
# Low-level math primitives (no dependencies on other custom functions)
# ==============================================================================


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        q1: First quaternion (w, x, y, z), shape (B, 4).
        q2: Second quaternion (w, x, y, z), shape (B, 4).

    Returns:
        Product quaternion, shape (B, 4).
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate.

    Args:
        q: Quaternion (w, x, y, z), shape (B, 4).

    Returns:
        Conjugate quaternion, shape (B, 4).
    """
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the magnitude of the quaternion error (shortest path).

    Args:
        q1: First quaternion (w, x, y, z), shape (B, 4).
        q2: Second quaternion (w, x, y, z), shape (B, 4).

    Returns:
        Error magnitude, shape (B,).
    """
    # Compute quaternion product q1 * conj(q2)
    q_err_w = q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1] + q1[:, 2] * q2[:, 2] + q1[:, 3] * q2[:, 3]
    q_err_x = -q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] - q1[:, 2] * q2[:, 3] + q1[:, 3] * q2[:, 2]
    q_err_y = -q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] - q1[:, 3] * q2[:, 1]
    q_err_z = -q1[:, 0] * q2[:, 3] - q1[:, 1] * q2[:, 2] + q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Clamp the real part to [-1, 1] for numerical stability
    q_err_w = torch.clamp(q_err_w, -1.0, 1.0)

    # Rotation angle from the error quaternion
    # |cos(theta/2)| = |q_err_w|, sin(theta/2) = sqrt(1 - q_err_w^2)
    # Error magnitude = 2 * |theta/2| = 2 * |arccos(|q_err_w|)|
    error = 2.0 * torch.acos(torch.abs(q_err_w))

    return error


@torch.jit.script
def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to a vector.

    Args:
        q: Quaternion (w, x, y, z), shape (B, 4).
        v: Vector to rotate, shape (B, 3).

    Returns:
        Rotated vector, shape (B, 3).
    """
    # Construct pure quaternion from v
    qv = torch.cat([torch.zeros_like(v[:, :1]), v], dim=-1)

    # q * qv * conj(q)
    q_inv = quat_conj(q)
    qv_rot = quat_mul(quat_mul(q, qv), q_inv)

    return qv_rot[:, 1:]


@torch.jit.script
def combine_frame_transforms(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine frame transforms: A_T_C = A_T_B * B_T_C.

    Args:
        pos_a: Position of frame A's origin in parent, shape (B, 3).
        quat_a: Orientation of frame A in parent, shape (B, 4).
        pos_b: Position of frame B in frame A, shape (B, 3).
        quat_b: Orientation of frame B in frame A, shape (B, 4).

    Returns:
        Combined position and quaternion, each shape (B, 3) and (B, 4).
    """
    pos_result = pos_a + quat_apply(quat_a, pos_b)
    quat_result = quat_mul(quat_a, quat_b)
    return pos_result, quat_result


# ==============================================================================
# Reward functions (depend on primitives above)
# ==============================================================================


@torch.jit.script
def ee_pos_tracking_exp(
    curr_pos_w: torch.Tensor, des_pos_w: torch.Tensor, std: float
) -> torch.Tensor:
    """Reward EE position tracking using exponential kernel.

    Args:
        curr_pos_w: Current EE position in world frame, shape (B, 3).
        des_pos_w: Desired EE position in world frame, shape (B, 3).
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (B,).
    """
    return torch.exp(-torch.sum(torch.abs(curr_pos_w - des_pos_w), dim=1) / std)


@torch.jit.script
def ee_ori_tracking(
    curr_quat_w: torch.Tensor, des_quat_w: torch.Tensor
) -> torch.Tensor:
    """Penalize EE orientation tracking error using shortest-path quaternion distance.

    Args:
        curr_quat_w: Current EE quaternion in world frame, shape (B, 4).
        des_quat_w: Desired EE quaternion in world frame, shape (B, 4).

    Returns:
        Penalty tensor of shape (B,). Returns 0 when perfectly aligned.
    """
    return quat_error_magnitude(curr_quat_w, des_quat_w)


@torch.jit.script
def lin_vel_tracking_exp(
    cmd_vel_xy: torch.Tensor, body_vel_xy: torch.Tensor, std: float
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel.

    Args:
        cmd_vel_xy: Commanded linear velocity in body xy, shape (B, 2).
        body_vel_xy: Actual body linear velocity in body xy, shape (B, 2).
        std: Standard deviation for exponential.

    Returns:
        Reward tensor of shape (B,).
    """
    error = torch.sum(torch.square(cmd_vel_xy - body_vel_xy), dim=1)
    return torch.exp(-error / (std**2))


@torch.jit.script
def ang_vel_tracking_exp(
    cmd_ang_vel_z: torch.Tensor, body_ang_vel_z: torch.Tensor, std: float
) -> torch.Tensor:
    """Reward tracking of angular velocity command (yaw) using exponential kernel.

    Args:
        cmd_ang_vel_z: Commanded angular velocity z, shape (B,).
        body_ang_vel_z: Actual body angular velocity z, shape (B,).
        std: Standard deviation for exponential.

    Returns:
        Reward tensor of shape (B,).
    """
    error = torch.square(cmd_ang_vel_z - body_ang_vel_z)
    return torch.exp(-error / (std**2))


@torch.jit.script
def lin_vel_z_l2(body_vel_z: torch.Tensor) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel.

    Args:
        body_vel_z: Body linear velocity z, shape (B,).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.square(body_vel_z)


@torch.jit.script
def ang_vel_xy_l2(body_ang_vel_xy: torch.Tensor) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel.

    Args:
        body_ang_vel_xy: Body angular velocity xy, shape (B, 2).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.square(body_ang_vel_xy), dim=1)


@torch.jit.script
def dof_torques_l2(torques: torch.Tensor) -> torch.Tensor:
    """Penalize joint torques using L2 squared kernel.

    Args:
        torques: Joint torques, shape (B, N).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.square(torques), dim=1)


@torch.jit.script
def dof_acc_l2(accelerations: torch.Tensor) -> torch.Tensor:
    """Penalize joint accelerations using L2 squared kernel.

    Args:
        accelerations: Joint accelerations, shape (B, N).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.square(accelerations), dim=1)


@torch.jit.script
def action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Penalize the rate of change of actions using L2 squared kernel.

    Args:
        action: Current actions, shape (B, N).
        prev_action: Previous actions, shape (B, N).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.square(action - prev_action), dim=1)


@torch.jit.script
def action_smoothness(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Penalize large instantaneous changes in the action output (L1 norm).

    Args:
        action: Current actions, shape (B, N).
        prev_action: Previous actions, shape (B, N).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.linalg.norm(action - prev_action, dim=1)


@torch.jit.script
def joint_deviation_l1(
    joint_pos: torch.Tensor, default_pos: torch.Tensor
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default (L1 norm).

    Args:
        joint_pos: Current joint positions, shape (B, N).
        default_pos: Default joint positions, shape (B, N).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.abs(joint_pos - default_pos), dim=1)


@torch.jit.script
def flat_orientation_l2(projected_gravity: torch.Tensor) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    Args:
        projected_gravity: Projected gravity vector (gx, gy, gz), shape (B, 3).

    Returns:
        Penalty tensor, shape (B,).
    """
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


@torch.jit.script
def base_height_l2(
    root_pos_z: torch.Tensor, target_height: float
) -> torch.Tensor:
    """Penalize base height deviation from target using L2 squared kernel.

    Args:
        root_pos_z: Root position z in world, shape (B,).
        target_height: Target height in meters.

    Returns:
        Penalty tensor, shape (B,).
    """
    curr_height = torch.clamp(root_pos_z, max=0.4)
    return torch.square(curr_height - target_height)


@torch.jit.script
def gait_trot_penalty(joint_pos: torch.Tensor) -> torch.Tensor:
    """Penalize pace gait (same-side legs together) to encourage trot.

    Joint indices: FR(0-2), FL(3-5), RR(6-8), RL(9-11).
    Pace: same-side legs swing together → |FR_hip − RR_hip| small
    Trot: diagonal legs swing together → |FR_hip − RL_hip| small

    Penalty = |FR_hip − RR_hip|² + |FL_hip − RL_hip|² − α·(|FR_hip − RL_hip|² + |FL_hip − RR_hip|²)
    Positive when pacing, negative when trotting.

    Args:
        joint_pos: Joint positions, shape (B, N). Must contain at least 12 leg joints.
    """
    # Hip joints (relative positions from default)
    fr_hip = joint_pos[:, 0]
    fl_hip = joint_pos[:, 3]
    rr_hip = joint_pos[:, 6]
    rl_hip = joint_pos[:, 9]

    pace_score = (fr_hip - rr_hip).pow(2) + (fl_hip - rl_hip).pow(2)
    trot_score = (fr_hip - rl_hip).pow(2) + (fl_hip - rr_hip).pow(2)

    # Positive = pacing, encourage reducing this
    return pace_score - 0.5 * trot_score
