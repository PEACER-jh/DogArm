"""Align mode commands: target point + EE pose at target + toward-target velocity."""

from __future__ import annotations

import math

import torch


def sample_target_points(
    env_ids: torch.Tensor,
    robot_xy: torch.Tensor,
    distance_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Generate random target points around the robot in world frame."""
    n = len(env_ids)
    r = torch.empty(n, 2, device=device)
    r[:, 0].uniform_(*distance_range)
    r[:, 1].uniform_(-math.pi, math.pi)
    target_pos = torch.zeros(n, 3, device=device)
    target_pos[:, 0] = robot_xy[env_ids, 0] + r[:, 0] * torch.cos(r[:, 1])
    target_pos[:, 1] = robot_xy[env_ids, 1] + r[:, 0] * torch.sin(r[:, 1])
    target_pos[:, 2] = 0.0
    return target_pos


def sample_ee_pose_at_target(
    env_ids: torch.Tensor,
    target_pos: torch.Tensor,
    sphere_center_z: float,
    arm_length: float,
    rpy_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Sample EE target pose in WORLD frame on hollow half-sphere above target.

    Returns [px_w, py_w, pz_w, qw, qx, qy, qz], shape (N, 7).
    """
    n = len(env_ids)
    rng = torch.empty(n, 5, device=device)

    azimuth = rng[:, 0].uniform_(0.0, 2.0 * math.pi)
    cos_phi = rng[:, 1].uniform_(0.0, 1.0)
    phi = torch.acos(cos_phi)

    pos_w = torch.zeros(n, 3, device=device)
    pos_w[:, 0] = target_pos[env_ids, 0] + arm_length * torch.cos(phi) * torch.cos(azimuth)
    pos_w[:, 1] = target_pos[env_ids, 1] + arm_length * torch.cos(phi) * torch.sin(azimuth)
    pos_w[:, 2] = sphere_center_z + arm_length * torch.sin(phi)

    roll = rng[:, 3].uniform_(*rpy_range["roll"])
    pitch = rng[:, 4].uniform_(*rpy_range["pitch"])
    yaw = torch.empty(n, device=device).uniform_(*rpy_range["yaw"])

    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.cat([pos_w, torch.stack([qw, qx, qy, qz], dim=-1)], dim=-1)


def sample_ee_pose_body_frame(
    env_ids: torch.Tensor,
    arm_length: float,
    min_radius: float,
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    rpy_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Sample EE target pose in body-frame (link0) half-sphere within arm reach."""
    n = len(env_ids)
    rng = torch.empty(n, 4, device=device)
    r_cube = rng[:, 0].uniform_(min_radius**3, arm_length**3)
    r = r_cube ** (1.0 / 3.0)
    cos_hi = math.cos(phi_range[1])
    cos_lo = math.cos(phi_range[0])
    cos_phi = rng[:, 1].uniform_(cos_hi, cos_lo)
    phi = torch.acos(cos_phi)
    theta = rng[:, 2].uniform_(*theta_range)
    px = r * torch.cos(phi) * torch.cos(theta)
    py = r * torch.cos(phi) * torch.sin(theta)
    pz = r * torch.sin(phi)
    roll = rng[:, 3].uniform_(*rpy_range["roll"])
    pitch = torch.empty(n, device=device).uniform_(*rpy_range["pitch"])
    yaw = torch.empty(n, device=device).uniform_(*rpy_range["yaw"])
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.cat([torch.stack([px, py, pz], dim=-1),
                      torch.stack([qw, qx, qy, qz], dim=-1)], dim=-1)


def velocity_toward_target(
    robot_xy: torch.Tensor,
    target_pos: torch.Tensor,
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
    speed_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Generate body-frame velocity steering toward target.

    Returns [vx, vy, wz] for all envs, shape (num_envs, 3).
    """
    import isaaclab.utils.math as _mu
    n = target_pos.shape[0]

    to_target_w = target_pos[:, :2] - robot_xy
    tgt_dist = torch.norm(to_target_w, dim=-1, keepdim=True)
    tgt_dir_w = to_target_w / (tgt_dist + 1e-6)

    fwd = _mu.quat_apply(robot_quat_w, forward_vec_b)
    robot_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
    tgt_yaw = torch.atan2(tgt_dir_w[:, 1], tgt_dir_w[:, 0])
    heading_err = tgt_yaw - robot_yaw
    heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    spd = torch.rand(n, device=device) * (speed_range[1] - speed_range[0]) + speed_range[0]
    cmd = torch.zeros(n, 3, device=device)
    cmd[:, 0] = spd * heading_err.cos().clamp(min=0.2)
    cmd[:, 1] = 0.0
    cmd[:, 2] = torch.clamp(heading_err, -1.0, 1.0)
    return cmd
