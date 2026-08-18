"""Termination terms used by the navigation Skill gate."""

from __future__ import annotations

import torch

from .geometry import navigation_error
from .rewards import navigation_arrival
from .rewards import _tilt, support_count


def _record_terminal_state(env) -> None:
    """Capture the physical state before ManagerBasedRLEnv auto-resets."""

    delta_b, distance, yaw_error = navigation_error(env)
    robot = env.scene["robot"]
    snapshot = {
        "distance_m": distance.detach().clone(),
        "target_bearing_rad": torch.atan2(
            delta_b[:, 1], delta_b[:, 0]
        ).detach().clone(),
        "yaw_error_rad": yaw_error.detach().clone(),
        "tilt_rad": _tilt(env).detach().clone(),
        "base_height_m": robot.data.root_pos_w[:, 2].detach().clone(),
        "root_position_w": robot.data.root_pos_w.detach().clone(),
        "body_vx_mps": robot.data.root_lin_vel_b[:, 0].detach().clone(),
        "body_vy_mps": robot.data.root_lin_vel_b[:, 1].detach().clone(),
        "body_wz_radps": robot.data.root_ang_vel_b[:, 2].detach().clone(),
        "support_count": support_count(env).detach().clone(),
        "raw_action": env.action_manager.action.detach().clone(),
    }
    if hasattr(env, "tandem_wheel_coordinates"):
        snapshot["wheel_coordinates"] = (
            env.tandem_wheel_coordinates.detach().clone()
        )
    if hasattr(env, "tandem_wheel_target"):
        snapshot["wheel_target"] = env.tandem_wheel_target.detach().clone()
    diagnostics = getattr(env, "tandem_wbc_diagnostics", None)
    if diagnostics is not None:
        snapshot["wheel_command"] = diagnostics.wheel_command.detach().clone()
        snapshot["leg_correction_norm"] = (
            diagnostics.leg_correction_norm.detach().clone()
        )
        snapshot["policy_leg_residual_norm"] = (
            diagnostics.policy_leg_residual_norm.detach().clone()
        )
        snapshot["support_position_error_norm"] = (
            diagnostics.support_position_error_norm.detach().clone()
        )
    env.tandem_navigation_snapshot = snapshot


def navigation_reached(env) -> torch.Tensor:
    _record_terminal_state(env)
    return navigation_arrival(env).bool()
