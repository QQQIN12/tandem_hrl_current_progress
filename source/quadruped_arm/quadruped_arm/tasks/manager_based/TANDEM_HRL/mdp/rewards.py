"""Reward terms for the privileged navigation Skill gate."""

from __future__ import annotations

import torch

from .geometry import navigation_error
from .observations import FOOT_SENSORS, contact_loads


def _tilt(env) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    return torch.asin(
        torch.sqrt(gravity[:, 0].square() + gravity[:, 1].square()).clamp(
            0.0, 1.0
        )
    )


def support_count(env) -> torch.Tensor:
    supported = []
    for name in FOOT_SENSORS:
        sensor = env.scene[name]
        forces = sensor.data.force_matrix_w
        if forces is None:
            forces = sensor.data.net_forces_w
        magnitude = torch.linalg.vector_norm(
            forces.reshape(forces.shape[0], -1, 3), dim=-1
        ).amax(dim=1)
        supported.append(magnitude > 1.0)
    return torch.stack(supported, dim=1).sum(dim=1).float()


def navigation_progress(env) -> torch.Tensor:
    """Potential decrease; positive only when the base approaches the stance."""

    _, distance, _ = navigation_error(env)
    if not hasattr(env, "tandem_previous_navigation_distance"):
        env.tandem_previous_navigation_distance = distance.detach().clone()
    reset = env.episode_length_buf <= 1
    previous = torch.where(
        reset, distance, env.tandem_previous_navigation_distance
    )
    progress = previous - distance
    env.tandem_previous_navigation_distance[:] = distance.detach()
    return progress.clamp(-0.10, 0.10)


def navigation_target_pose(
    env, distance_sigma: float = 0.35, yaw_sigma: float = 0.45
) -> torch.Tensor:
    _, distance, yaw_error = navigation_error(env)
    position_score = torch.exp(-distance.square() / distance_sigma**2)
    yaw_score = torch.exp(-yaw_error.square() / yaw_sigma**2)
    return position_score * (0.25 + 0.75 * yaw_score)


def navigation_velocity_profile(
    env,
    maximum_speed: float = 0.45,
    distance_scale: float = 0.35,
    velocity_sigma: float = 0.16,
    lateral_sigma: float = 0.20,
) -> torch.Tensor:
    """Reward progress with a distance-dependent braking profile."""

    delta_b, distance, _ = navigation_error(env)
    direction = delta_b / distance.unsqueeze(1).clamp_min(1.0e-4)
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    toward = torch.sum(velocity * direction, dim=1)
    lateral = velocity[:, 0] * direction[:, 1] - velocity[:, 1] * direction[:, 0]
    desired = maximum_speed * torch.tanh(distance / distance_scale)
    tracking = torch.exp(-(toward - desired).square() / velocity_sigma**2)
    lateral_score = torch.exp(-lateral.square() / lateral_sigma**2)
    direction_score = torch.tanh(toward / 0.12)
    return 0.75 * direction_score + 0.25 * tracking * lateral_score


def navigation_heading_alignment(
    env, heading_sigma: float = 0.60, final_radius: float = 0.25
) -> torch.Tensor:
    delta_b, distance, yaw_error = navigation_error(env)
    bearing = torch.atan2(delta_b[:, 1], delta_b[:, 0])
    reverse = bearing.abs() > (0.5 * torch.pi)
    reverse_error = torch.atan2(
        torch.sin(bearing - torch.sign(bearing) * torch.pi),
        torch.cos(bearing - torch.sign(bearing) * torch.pi),
    )
    steering_error = torch.where(reverse, reverse_error, bearing)
    pose_error = torch.where(distance <= final_radius, yaw_error, steering_error)
    return torch.exp(-pose_error.square() / heading_sigma**2)


def navigation_braking(env, radius: float = 0.22) -> torch.Tensor:
    _, distance, _ = navigation_error(env)
    robot = env.scene["robot"]
    speed_squared = torch.sum(robot.data.root_lin_vel_b[:, :2].square(), dim=1)
    near_weight = (1.0 - distance / radius).clamp(0.0, 1.0)
    return near_weight * speed_squared


def navigation_arrival(
    env,
    distance_threshold: float = 0.10,
    yaw_threshold: float = 0.18,
    tilt_threshold: float = 0.35,
) -> torch.Tensor:
    _, distance, yaw_error = navigation_error(env)
    robot = env.scene["robot"]
    speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    return (
        (distance <= distance_threshold)
        & (yaw_error.abs() <= yaw_threshold)
        & (_tilt(env) <= tilt_threshold)
        & (speed <= 0.20)
        & (support_count(env) >= 3.0)
    ).float()


def base_stability(env) -> torch.Tensor:
    robot = env.scene["robot"]
    tilt_cost = _tilt(env).square()
    height_cost = (robot.data.root_pos_w[:, 2] - 0.46).square()
    support_cost = (4.0 - support_count(env)).clamp_min(0.0)
    return tilt_cost + 0.5 * height_cost + 0.08 * support_cost


def action_rate_l2(env) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(delta.square(), dim=1)


def locomotion_velocity_tracking(
    env,
    tracking_scale: float = 0.20,
    active_threshold: float = 0.05,
) -> torch.Tensor:
    """Track commanded velocity without rewarding an active command at rest."""

    command = env.command_manager.get_command("locomotion")
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error_squared = torch.sum((velocity - command[:, :2]).square(), dim=1)
    score = 1.0 / (1.0 + error_squared / tracking_scale**2)
    command_squared = torch.sum(command[:, :2].square(), dim=1)
    standing_score = 1.0 / (1.0 + command_squared / tracking_scale**2)
    active = torch.sqrt(command_squared) > active_threshold
    return torch.where(active, score - standing_score, score)


def locomotion_yaw_tracking(
    env,
    tracking_scale: float = 0.30,
    active_threshold: float = 0.08,
) -> torch.Tensor:
    """Track yaw rate using improvement over the zero-yaw baseline."""

    command = env.command_manager.get_command("locomotion")
    yaw_velocity = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error_squared = (yaw_velocity - command[:, 2]).square()
    score = 1.0 / (1.0 + error_squared / tracking_scale**2)
    standing_score = 1.0 / (
        1.0 + command[:, 2].square() / tracking_scale**2
    )
    active = command[:, 2].abs() > active_threshold
    return torch.where(active, score - standing_score, score)


def locomotion_command_alignment(env) -> torch.Tensor:
    command = env.command_manager.get_command("locomotion")
    robot = env.scene["robot"]
    linear_product = torch.sum(
        command[:, :2] * robot.data.root_lin_vel_b[:, :2], dim=1
    )
    yaw_product = command[:, 2] * robot.data.root_ang_vel_b[:, 2]
    linear_score = torch.tanh(linear_product / 0.04)
    yaw_score = torch.tanh(yaw_product / 0.08)
    linear_active = torch.linalg.vector_norm(command[:, :2], dim=1) > 0.05
    yaw_active = command[:, 2].abs() > 0.08
    active_count = linear_active.float() + yaw_active.float()
    score = (
        linear_score * linear_active.float()
        + yaw_score * yaw_active.float()
    )
    return score / active_count.clamp_min(1.0)


def yaw_load_redistribution(
    env,
    target_effective_support: float = 3.0,
    tracking_scale: float = 0.45,
    command_threshold: float = 0.12,
    upright_scale: float = 0.35,
) -> torch.Tensor:
    """Reward learned load transfer that releases skid-steer yaw constraints."""

    loads = contact_loads(env).clamp_min(0.0)
    total = loads.sum(dim=1)
    effective_support = total.square() / loads.square().sum(dim=1).clamp_min(1.0)
    support_score = torch.exp(
        -(effective_support - target_effective_support).square()
        / tracking_scale**2
    )
    upright_score = torch.exp(-_tilt(env).square() / upright_scale**2)
    command = env.command_manager.get_command("locomotion")
    active = command[:, 2].abs() > command_threshold
    return support_score * upright_score * active.float()


def support_fraction(env) -> torch.Tensor:
    return 0.25 * support_count(env)


def leg_residual_l2(env) -> torch.Tensor:
    return torch.sum(env.action_manager.action[:, :12].square(), dim=1)


def wheel_coordinate_l2(env) -> torch.Tensor:
    return torch.sum(env.action_manager.action[:, 12:16].square(), dim=1)


def support_allocation_l2(env) -> torch.Tensor:
    """Regularise the separate learned support-allocation head."""

    return torch.sum(env.action_manager.action[:, 16:20].square(), dim=1)
