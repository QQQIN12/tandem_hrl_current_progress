"""Shared relation geometry for the privileged-state mainline."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


GRASP_STANCE_FROM_OBJECT = (0.356, -0.034)
GRASP_STANCE_YAW = 0.0


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    forward = torch.zeros(
        quaternion.shape[0], 3, device=quaternion.device
    )
    forward[:, 0] = 1.0
    forward_w = math_utils.quat_apply(quaternion, forward)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0])


def navigation_error(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    grasp_object = env.scene["grasp_object"]
    target_xy = grasp_object.data.root_pos_w[:, :2]
    target_xy = target_xy + target_xy.new_tensor(
        GRASP_STANCE_FROM_OBJECT
    )
    delta_w = torch.cat(
        (
            target_xy - robot.data.root_pos_w[:, :2],
            torch.zeros(env.num_envs, 1, device=env.device),
        ),
        dim=1,
    )
    delta_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, delta_w
    )[:, :2]
    distance = torch.linalg.vector_norm(delta_b, dim=1)
    yaw_error = wrap_angle(
        delta_b.new_full((env.num_envs,), GRASP_STANCE_YAW)
        - yaw_from_quaternion(robot.data.root_quat_w)
    )
    return delta_b, distance, yaw_error
