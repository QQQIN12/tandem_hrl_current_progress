"""Task, interaction, and control-aware rewards for TANDEM-HRL."""

from __future__ import annotations

import torch


def _term(env, command_name: str):
    return env.command_manager.get_term(command_name)


def mission_completion(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).mission_completion_delta


def mission_success(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).mission_success_event


def mission_succeeded(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    return _term(env, command_name).mission_success > 0.5


def selected_task_progress(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    return _term(env, command_name).selected_task_progress_delta


def selected_task_tracking(
    env, command_name: str = "locomotion", sigma: float = 0.75
) -> torch.Tensor:
    error = _term(env, command_name).task_error
    return torch.exp(-error / sigma)


def ee_tracking(
    env, command_name: str = "locomotion", sigma: float = 0.16
) -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.ee_error)
    manipulation = (
        (hierarchy.task_id == 3)
        | ((hierarchy.task_id >= 5) & (hierarchy.task_id <= 10))
    ).float()
    return manipulation * torch.exp(-term.ee_error / sigma)


def valid_task_choice(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    valid = env.tactic_task_valid_mask
    return valid.gather(1, hierarchy.task_id.unsqueeze(1)).squeeze(1)


def invalid_task_choice(env, command_name: str = "locomotion") -> torch.Tensor:
    return 1.0 - valid_task_choice(env, command_name)


def option_switch_penalty(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    return 0.5 * (hierarchy.task_switch + hierarchy.skill_switch)


def control_aware_progress(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    term = _term(env, command_name)
    robust_margin = (
        0.38 * term.safety_margin
        + 0.24 * term.preview_margin
        + 0.20 * term.clf_decrease_score
        + 0.18 * term.disturbance_quality
    ).clamp(0.0, 1.0)
    progress = (
        0.62 * term.selected_task_progress_delta
        + 0.38 * term.mission_completion_delta
    ).clamp(-0.25, 1.0)
    return progress * (0.35 + 0.65 * robust_margin)


def cbf_margin(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).safety_margin


def predicted_margin(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).preview_margin


def clf_decrease(env, command_name: str = "locomotion") -> torch.Tensor:
    return _term(env, command_name).clf_decrease_score


def disturbance_rejection(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    return _term(env, command_name).disturbance_quality


def safety_violation(
    env,
    command_name: str = "locomotion",
    safety_threshold: float = 0.12,
    preview_threshold: float = 0.10,
) -> torch.Tensor:
    term = _term(env, command_name)
    safety_shortfall = (
        (float(safety_threshold) - term.safety_margin)
        / max(float(safety_threshold), 1.0e-4)
    ).clamp(0.0, 1.0)
    preview_shortfall = (
        (float(preview_threshold) - term.preview_margin)
        / max(float(preview_threshold), 1.0e-4)
    ).clamp(0.0, 1.0)
    return torch.maximum(
        safety_shortfall, 0.65 * preview_shortfall
    )


def unsafe_progress(
    env,
    command_name: str = "locomotion",
    safety_threshold: float = 0.12,
    preview_threshold: float = 0.10,
) -> torch.Tensor:
    term = _term(env, command_name)
    violation = safety_violation(
        env,
        command_name,
        safety_threshold=safety_threshold,
        preview_threshold=preview_threshold,
    )
    return term.selected_task_progress_delta.clamp_min(0.0) * violation


def object_contact(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_contact_memory[rows, object_id]


def _selected_object(term, hierarchy) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the object owned by the currently executed task option."""

    rows = torch.arange(term.num_envs, device=term.device)
    selected = hierarchy.object_id
    delivery_slot = (hierarchy.task_id - 5).clamp(0, 5)
    object_id = torch.where(
        (hierarchy.task_id >= 5) & (hierarchy.task_id <= 10),
        delivery_slot,
        selected,
    )
    return rows, object_id


def object_lift(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_lift_memory[rows, object_id]


def object_transport(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_transport_memory[rows, object_id]


def object_place(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_place[rows, object_id]


def object_completion(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_completion[rows, object_id]


def robust_interaction_frontier(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    frontier_delta = term.object_interaction_frontier_delta[
        rows, object_id
    ]
    robust_margin = (
        0.38 * term.safety_margin
        + 0.24 * term.preview_margin
        + 0.20 * term.clf_decrease_score
        + 0.18 * term.disturbance_quality
    ).clamp(0.0, 1.0)
    return frontier_delta * (0.45 + 0.55 * robust_margin)


def grasp_hold_quality(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    contact = term.object_contact_memory[rows, object_id]
    return contact * term.gripper_closure


def payload_target_progress(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    """Reward target convergence only while the object is physically carried."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    carrying = term.object_carrying[rows, object_id].float()
    progress_delta = term.object_target_progress_delta[rows, object_id]
    robust_margin = torch.minimum(
        term.safety_margin, term.preview_margin
    ).clamp(0.0, 1.0)
    return (
        carrying
        * (progress_delta / 0.012).clamp(-1.0, 1.0)
        * (0.35 + 0.65 * robust_margin)
    )


def payload_release_readiness(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    """Shape the carried object toward the learned release set."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    readiness = term.object_release_readiness[rows, object_id]
    readiness_delta = term.object_release_readiness_delta[
        rows, object_id
    ]
    robust_margin = torch.minimum(
        term.safety_margin, term.preview_margin
    ).clamp(0.0, 1.0)
    return (
        (
            (readiness_delta / 0.015).clamp(-1.0, 1.0)
            + 0.12 * readiness
        )
        * (0.40 + 0.60 * robust_margin)
    )


def intended_payload_release(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    """Credit opening the gripper from a physically admissible hover set."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_release_event_quality[rows, object_id]


def payload_retention(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    """Measure persistent grasp integrity during mobile transport."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    carrying = term.object_carrying[rows, object_id].float()
    closure = term.gripper_closure.clamp(0.0, 1.0)
    return carrying * (
        0.45
        + 0.25 * closure
        + 0.15 * term.safety_margin
        + 0.15 * term.preview_margin
    )


def payload_drop(env, command_name: str = "locomotion") -> torch.Tensor:
    """Flag an unintended release before physical placement is verified."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return term.object_drop_event[rows, object_id]


def release_quality(env, command_name: str = "locomotion") -> torch.Tensor:
    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    return (
        term.object_place[rows, object_id]
        * (1.0 - term.gripper_closure)
    )


def wrong_object_interaction(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    """Penalize contact chains that do not belong to the active task."""

    term = _term(env, command_name)
    hierarchy = term._hierarchy()
    if hierarchy is None:
        return torch.zeros_like(term.mission_completion)
    rows, object_id = _selected_object(term, hierarchy)
    selected_mask = torch.nn.functional.one_hot(
        object_id, num_classes=term.object_contact_memory.shape[1]
    ).bool()
    wrong_contact = term.object_contact_memory.masked_fill(
        selected_mask, 0.0
    ).amax(dim=1)
    wrong_lift = term.object_lift_memory.masked_fill(
        selected_mask, 0.0
    ).amax(dim=1)
    wrong_carry = term.object_carrying.float().masked_fill(
        selected_mask, 0.0
    ).amax(dim=1)
    interaction_task = (
        (hierarchy.task_id == 3)
        | ((hierarchy.task_id >= 5) & (hierarchy.task_id <= 10))
    ).float()
    return interaction_task * (
        0.20 * wrong_contact
        + 0.35 * wrong_lift
        + 0.45 * wrong_carry
    )


def all_objects_delivered(
    env, command_name: str = "locomotion"
) -> torch.Tensor:
    return _term(env, command_name).object_completion.min(dim=1).values
