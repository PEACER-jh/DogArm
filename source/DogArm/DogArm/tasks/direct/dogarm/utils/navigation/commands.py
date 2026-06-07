"""Navigation command generation: target point + toward-target velocity."""

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
        env_ids: Indices to sample for.
        robot_xy: Robot XY positions in world frame, shape (num_envs, 2).
        distance_range: (min_dist, max_dist) in meters.
        device: Torch device.

    Returns:
        Target positions in world frame, shape (len(env_ids), 3).
    """
    n = len(env_ids)
    r = torch.empty(n, 2, device=device)
    r[:, 0].uniform_(*distance_range)
    r[:, 1].uniform_(-math.pi, math.pi)

    target_pos = torch.zeros(n, 3, device=device)
    target_pos[:, 0] = robot_xy[env_ids, 0] + r[:, 0] * torch.cos(r[:, 1])
    target_pos[:, 1] = robot_xy[env_ids, 1] + r[:, 0] * torch.sin(r[:, 1])
    target_pos[:, 2] = 0.0
    return target_pos


def velocity_toward_target(
    env_ids: torch.Tensor,
    robot_xy: torch.Tensor,
    target_pos: torch.Tensor,
    robot_quat_w: torch.Tensor,
    forward_vec_b: torch.Tensor,
    curriculum_step: int,
    curriculum_coeff: int,
    speed_range_init: tuple[float, float],
    speed_range_final: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Generate velocity commands that steer toward the target point.

    Returns:
        Body-frame velocity commands [vx, vy, wz], shape (len(env_ids), 3).
    """
    import isaaclab.utils.math as _mu

    n = len(env_ids)
    t_val = min(1.0, curriculum_step / (1 * curriculum_coeff))

    sp_lo = speed_range_init[0] * (1 - t_val) + speed_range_final[0] * t_val
    sp_hi = speed_range_init[1] * (1 - t_val) + speed_range_final[1] * t_val

    # Target direction in world frame
    to_target_w = target_pos[env_ids, :2] - robot_xy[env_ids, :2]
    tgt_dist = torch.norm(to_target_w, dim=-1, keepdim=True)
    tgt_dir_w = to_target_w / (tgt_dist + 1e-6)

    # Robot forward in world frame (handle per-env or single forward vec)
    fwd_vec = forward_vec_b[env_ids] if forward_vec_b.ndim == 2 else forward_vec_b
    fwd = _mu.quat_apply(robot_quat_w[env_ids], fwd_vec)
    robot_yaw = torch.atan2(fwd[:, 1], fwd[:, 0])

    # Target yaw
    tgt_yaw = torch.atan2(tgt_dir_w[:, 1], tgt_dir_w[:, 0])
    heading_err = tgt_yaw - robot_yaw
    heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    # Body-frame velocity: always forward, turn toward target
    forward_speed = torch.rand(n, device=device) * (sp_hi - sp_lo) + sp_lo
    cmd = torch.zeros(n, 3, device=device)
    cmd[:, 0] = forward_speed * heading_err.cos().clamp(min=0.2)
    cmd[:, 1] = 0.0
    cmd[:, 2] = torch.clamp(heading_err, -1.0, 1.0)

    return cmd


def sample_ee_pose_at_target(
    env_ids: torch.Tensor,
    target_pos: torch.Tensor,
    sphere_center_z: float,
    arm_length: float,
    rpy_range: dict,
    device: torch.device,
) -> torch.Tensor:
    """Sample EE target pose in WORLD frame, on a hollow half-sphere above target point.

    Returns [px_w, py_w, pz_w, qw, qx, qy, qz], shape (N, 7).
    """
    n = len(env_ids)
    rng = torch.empty(n, 5, device=device)

    azimuth = rng[:, 0].uniform_(0.0, 2.0 * math.pi)
    cos_phi = rng[:, 1].uniform_(0.0, 1.0)
    phi = torch.acos(cos_phi)
    r_val = arm_length

    pos_w = torch.zeros(n, 3, device=device)
    pos_w[:, 0] = target_pos[env_ids, 0] + r_val * torch.cos(phi) * torch.cos(azimuth)
    pos_w[:, 1] = target_pos[env_ids, 1] + r_val * torch.cos(phi) * torch.sin(azimuth)
    pos_w[:, 2] = sphere_center_z + r_val * torch.sin(phi)

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

    quat_w = torch.stack([qw, qx, qy, qz], dim=-1)
    return torch.cat([pos_w, quat_w], dim=-1)
