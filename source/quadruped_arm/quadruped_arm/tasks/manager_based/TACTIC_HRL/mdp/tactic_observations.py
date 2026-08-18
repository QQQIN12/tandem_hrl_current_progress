"""Observation groups used by TANDEM-HRL."""

from __future__ import annotations

import torch

from ..tactic_layout import ACTION_LAYOUT, HIERARCHY_CONTEXT_DIM


def _term(env, command_name: str):
    return env.command_manager.get_term(command_name)


def tactic_tau_down(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).tau_down_packet


def tactic_tau_up(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).tau_up_packet


def tactic_task_skill(env, command_name: str = "locomotion") -> torch.Tensor:
    """Expose the policy's current task and skill commitments."""

    hierarchy = getattr(env, "tactic_hierarchy", None)
    if hierarchy is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    task = hierarchy.task_id.float() / max(1, ACTION_LAYOUT.task_dim - 1)
    skill = hierarchy.skill_id.float() / max(1, ACTION_LAYOUT.skill_dim - 1)
    return torch.stack((task, skill), dim=-1)


def tactic_hierarchy_context(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    term = _term(env, command_name)
    context = term.hierarchy_context
    if context.shape[1] != HIERARCHY_CONTEXT_DIM:
        raise RuntimeError(
            f"TACTIC context has {context.shape[1]} values; "
            f"expected {HIERARCHY_CONTEXT_DIM}"
        )
    return context
