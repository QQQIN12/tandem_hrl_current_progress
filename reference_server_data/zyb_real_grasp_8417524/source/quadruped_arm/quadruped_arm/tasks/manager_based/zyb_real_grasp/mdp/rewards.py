"""Reward terms for the non-hierarchical ZYB-v0 grasp baseline."""

from __future__ import annotations

import torch

from ..scene_cfg import OBJECT_START_POSITION, TARGET_PLATFORM_SIZE
from .observations import _asset_position_w, support_metrics


def _wrist_object_distance(env) -> torch.Tensor:
    robot = env.scene["robot"]
    link_ids, _ = robot.find_bodies(["link6"], preserve_order=True)
    wrist = robot.data.body_pos_w[:, link_ids[0]]
    return torch.linalg.vector_norm(env.scene["grasp_object"].data.root_pos_w - wrist, dim=1)


def wrist_object_proximity(env) -> torch.Tensor:
    return torch.exp(-_wrist_object_distance(env).square() / 0.20**2)


def bilateral_grasp(env) -> torch.Tensor:
    metrics = support_metrics(env)
    loads = metrics["finger_loads"]
    balance = 1.0 - (loads[:, 0] - loads[:, 1]).abs() / loads.sum(dim=1).clamp_min(0.5)
    return metrics["bilateral_contact"].float() * balance.clamp(0.0, 1.0)


def object_lift(env) -> torch.Tensor:
    lift = env.scene["grasp_object"].data.root_pos_w[:, 2] - float(OBJECT_START_POSITION[2])
    return (lift / 0.12).clamp(0.0, 1.0)


def payload_retention(env) -> torch.Tensor:
    metrics = support_metrics(env)
    lifted = object_lift(env) > 0.15
    return lifted.float() * metrics["bilateral_contact"].float()


def target_transport_progress(env) -> torch.Tensor:
    obj = env.scene["grasp_object"].data.root_pos_w
    target = _asset_position_w(env, "target_platform")
    xy_error = torch.linalg.vector_norm(obj[:, :2] - target[:, :2], dim=1)
    return object_lift(env) * torch.exp(-xy_error.square() / 0.70**2)


def placement_quality(env) -> torch.Tensor:
    obj = env.scene["grasp_object"].data.root_pos_w
    target = _asset_position_w(env, "target_platform")
    half_size = obj.new_tensor(TARGET_PLATFORM_SIZE[:2]) * 0.5
    inside = ((obj[:, :2] - target[:, :2]).abs() < (half_size - 0.015)).all(dim=1)
    top_z = target[:, 2] + 0.5 * TARGET_PLATFORM_SIZE[2]
    supported = (obj[:, 2] - top_z).abs() < 0.09
    slow = torch.linalg.vector_norm(env.scene["grasp_object"].data.root_lin_vel_w, dim=1) < 0.12
    return (inside & supported & slow).float()


def support_quality(env) -> torch.Tensor:
    metrics = support_metrics(env)
    return (metrics["support_count"] / 4.0) * torch.exp(-metrics["tilt"].square() / 0.20**2)


def rear_support_deficit(env) -> torch.Tensor:
    return ((2.0 - support_metrics(env)["rear_support_count"]) / 2.0).clamp(0.0, 1.0)


def vertical_motion_cost(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_b[:, 2].square() + robot.data.root_ang_vel_b[:, :2].square().sum(dim=1)


def success_bonus(env) -> torch.Tensor:
    return placement_quality(env)
