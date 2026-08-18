"""Reward terms for LR_HRL benchmark tasks.

The terms are thin wrappers around measurable task signals: route progress,
base and end-effector tracking, stability, obstacle clearance, and gripper
closure relative to the configured grasp target.  The inherited baseline reward
terms are still applied through the base environment configuration.
"""

import torch

from .LR_HRL_command import PHASE_GRASP, PHASE_RECOVER


def _route_term(env, command_name: str = "locomotion"):
    return env.command_manager.get_term(command_name)


def LR_HRL_route_progress(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    return term.tau_up_packet[:, 0].clamp(0.0, 1.0)


def LR_HRL_goal_tracking(env, command_name: str = "locomotion", sigma: float = 0.65) -> torch.Tensor:
    term = _route_term(env, command_name)
    dist = term.tau_up_packet[:, 1].clamp(min=0.0)
    return torch.exp(-dist / sigma)


def LR_HRL_yaw_alignment(env, command_name: str = "locomotion", sigma: float = 0.45) -> torch.Tensor:
    term = _route_term(env, command_name)
    yaw_err = term.tau_up_packet[:, 2].clamp(min=0.0)
    return torch.exp(-yaw_err / sigma)


def LR_HRL_ee_tracking(env, command_name: str = "locomotion", sigma: float = 0.18) -> torch.Tensor:
    term = _route_term(env, command_name)
    err = term.tau_up_packet[:, 3].clamp(min=0.0)
    return torch.exp(-err / sigma)


def LR_HRL_stability_margin(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    tilt = term.tau_up_packet[:, 4]
    support = term.tau_up_packet[:, 5]
    tilt_score = (1.0 - tilt / 0.6).clamp(0.0, 1.0)
    support_score = (support / 4.0).clamp(0.0, 1.0)
    return 0.5 * tilt_score + 0.5 * support_score


def LR_HRL_obstacle_clearance(env, command_name: str = "locomotion", desired_margin: float = 0.65) -> torch.Tensor:
    term = _route_term(env, command_name)
    margin = term.tau_up_packet[:, 6]
    return (margin / desired_margin).clamp(0.0, 1.0)


def LR_HRL_feasibility(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    return term.tau_up_packet[:, 7].clamp(0.0, 1.0)


def LR_HRL_mismatch_penalty(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    return term.tau_up_packet[:, 8].clamp(0.0, 1.0)


def LR_HRL_recovery_penalty(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    return (term.phase_ids == PHASE_RECOVER).float()


def LR_HRL_grasp_proxy(env, command_name: str = "locomotion") -> torch.Tensor:
    """Reward final-phase gripper closure when the TCP is close and stable."""

    term = _route_term(env, command_name)
    near_ee = torch.exp(-term.tau_up_packet[:, 3].clamp(min=0.0) / 0.12)
    stable = LR_HRL_stability_margin(env, command_name=command_name)
    close_cmd = term.tau_up_packet[:, 10].clamp(0.0, 1.0)
    grasp_phase = (term.phase_ids == PHASE_GRASP).float()
    return grasp_phase * close_cmd * near_ee * stable


def LR_HRL_forward_efficiency(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _route_term(env, command_name)
    improvement = term.tau_up_packet[:, 11]
    return improvement.clamp(min=0.0)
