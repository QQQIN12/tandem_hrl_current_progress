"""Observation terms for LR_HRL packets and diagnostics."""

import torch

from .LR_HRL_command import TAU_DOWN_DIM, TAU_UP_DIM


def _get_term(env, command_name: str):
    try:
        return env.command_manager.get_term(command_name)
    except Exception:
        return None


def LR_HRL_tau_down_obs(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _get_term(env, command_name)
    if term is None or not hasattr(term, "tau_down_packet"):
        return torch.zeros(env.num_envs, TAU_DOWN_DIM, device=env.device)
    return term.tau_down_packet


def LR_HRL_tau_up_obs(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _get_term(env, command_name)
    if term is None or not hasattr(term, "tau_up_packet"):
        return torch.zeros(env.num_envs, TAU_UP_DIM, device=env.device)
    return term.tau_up_packet


def LR_HRL_phase_skill_obs(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _get_term(env, command_name)
    if term is None or not hasattr(term, "phase_ids"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.stack([term.phase_ids.float(), term.skill_ids.float()], dim=-1)
