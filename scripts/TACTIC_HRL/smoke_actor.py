"""Numerical smoke test for TACTIC and the ZYB-v0 migration."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


relative_package = (
    Path("source")
    / "quadruped_arm"
    / "quadruped_arm"
    / "tasks"
    / "manager_based"
    / "TACTIC_HRL"
)
package_root = None
for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    candidate = parent / relative_package
    if candidate.is_dir():
        package_root = candidate
        break
if package_root is None:
    raise RuntimeError("Could not locate the TACTIC_HRL source package")
for package_name, package_path in (
    ("quadruped_arm", package_root.parents[3]),
    ("quadruped_arm.tasks", package_root.parents[2]),
    ("quadruped_arm.tasks.manager_based", package_root.parents[1]),
    ("quadruped_arm.tasks.manager_based.TACTIC_HRL", package_root),
    ("quadruped_arm.tasks.manager_based.TACTIC_HRL.agents", package_root / "agents"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

layout = _load_module(
    "quadruped_arm.tasks.manager_based.TACTIC_HRL.tactic_layout",
    package_root / "tactic_layout.py",
)
actor_module = _load_module(
    "quadruped_arm.tasks.manager_based.TACTIC_HRL.agents.tactic_actor_critic",
    package_root / "agents" / "tactic_actor_critic.py",
)
checkpoint_module = _load_module(
    "quadruped_arm.tasks.manager_based.TACTIC_HRL.agents.tactic_checkpoint",
    package_root / "agents" / "tactic_checkpoint.py",
)
ppo_module = _load_module(
    "quadruped_arm.tasks.manager_based.TACTIC_HRL.agents.tactic_ppo",
    package_root / "agents" / "tactic_ppo.py",
)

TACTICActorCritic = actor_module.TACTICActorCritic
TACTICPPO = ppo_module.TACTICPPO
load_zyb_baseline_physical = checkpoint_module.load_zyb_baseline_physical
ACTION_LAYOUT = layout.ACTION_LAYOUT
HIERARCHY_CONTEXT_DIM = layout.HIERARCHY_CONTEXT_DIM
GLOBAL_CONTEXT_DIM = layout.GLOBAL_CONTEXT_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_checkpoint", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260727)
    batch_size = 8
    obs = {
        "policy": torch.randn(batch_size, 1003, device=device),
        "hierarchy_context": torch.rand(
            batch_size, HIERARCHY_CONTEXT_DIM, device=device
        ),
    }
    context_slots = obs["hierarchy_context"][:, GLOBAL_CONTEXT_DIM:].view(
        batch_size, ACTION_LAYOUT.task_dim, -1
    )
    context_slots[:, :, 11] = 1.0
    context_slots[:, :, 12] = 0.0
    context_slots[:, :, 13] = 1.0
    obs["hierarchy_context"][:, layout.COMMAND_VX_INDEX] = 0.0
    obs["hierarchy_context"][:, layout.COMMAND_VY_INDEX] = 0.0
    obs["hierarchy_context"][:, layout.COMMAND_WZ_INDEX] = 0.0
    groups = {
        "policy": ["policy"],
        "critic": ["policy", "hierarchy_context"],
    }
    policy = TACTICActorCritic(
        obs=obs,
        obs_groups=groups,
        num_actions=ACTION_LAYOUT.total_dim,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)

    executor_parameters = dict(
        policy.actor.physical_executor_named_parameters()
    )
    if not executor_parameters:
        raise RuntimeError("Physical executor boundary is empty")
    frozen_count = policy.actor.set_physical_executor_trainable(False)
    if frozen_count != sum(
        parameter.numel()
        for parameter in executor_parameters.values()
    ):
        raise RuntimeError("Physical executor parameter count is inconsistent")
    if any(
        parameter.requires_grad
        for parameter in executor_parameters.values()
    ):
        raise RuntimeError("Physical executor boundary did not freeze")
    hierarchy_parameters = dict(policy.actor.named_parameters())
    for parameter_name in (
        "task_query.weight",
        "motion_skill_logits_head.weight",
        "interaction_skill_logits_head.weight",
        "task_outcome_head.weight",
        "skill_effect_head.weight",
        "embodiment_response_matrix",
    ):
        if not hierarchy_parameters[parameter_name].requires_grad:
            raise RuntimeError(
                f"Hierarchy parameter was frozen: {parameter_name}"
            )
    policy.actor.set_physical_executor_trainable(True)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    adapter_count = policy.actor.set_recovery_adapter_trainable(True)
    if adapter_count <= 0:
        raise RuntimeError("Recovery adapter boundary is empty")
    adapter_prefixes = tuple(
        f"{name}."
        for name in policy.actor.RECOVERY_ADAPTER_MODULE_NAMES
    )
    unexpected_trainable = [
        name
        for name, parameter in policy.actor.named_parameters()
        if parameter.requires_grad
        and not name.startswith(adapter_prefixes)
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Recovery boundary exposed shared parameters: "
            + ", ".join(unexpected_trainable)
        )
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    decomposition_count = policy.actor.set_decomposition_trainable(True)
    if decomposition_count <= 0:
        raise RuntimeError("Task-skill decomposition boundary is empty")
    decomposition_prefixes = tuple(
        f"{name}."
        for name in policy.actor.DECOMPOSITION_MODULE_NAMES
    )
    unexpected_decomposition_trainable = [
        name
        for name, parameter in policy.actor.named_parameters()
        if parameter.requires_grad
        and not name.startswith(decomposition_prefixes)
    ]
    if unexpected_decomposition_trainable:
        raise RuntimeError(
            "Decomposition boundary exposed shared parameters: "
            + ", ".join(unexpected_decomposition_trainable)
        )
    for module_name in policy.actor.DECOMPOSITION_MODULE_NAMES:
        module = getattr(policy.actor, module_name)
        if not all(
            parameter.requires_grad for parameter in module.parameters()
        ):
            raise RuntimeError(
                f"Decomposition module was frozen: {module_name}"
            )
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    motion_selector_count = (
        policy.actor.set_motion_selector_trainable(True)
    )
    if motion_selector_count <= 0:
        raise RuntimeError("Motion-selector boundary is empty")
    motion_selector_prefixes = tuple(
        f"{name}."
        for name in policy.actor.MOTION_SELECTOR_MODULE_NAMES
    )
    unexpected_motion_selector_trainable = [
        name
        for name, parameter in policy.actor.named_parameters()
        if parameter.requires_grad
        and not name.startswith(motion_selector_prefixes)
    ]
    if unexpected_motion_selector_trainable:
        raise RuntimeError(
            "Motion-selector boundary exposed shared parameters: "
            + ", ".join(unexpected_motion_selector_trainable)
        )
    for module_name in policy.actor.MOTION_SELECTOR_MODULE_NAMES:
        module = getattr(policy.actor, module_name)
        if not all(
            parameter.requires_grad for parameter in module.parameters()
        ):
            raise RuntimeError(
                f"Motion-selector module was frozen: {module_name}"
            )
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    interaction_selector_count = (
        policy.actor.set_interaction_selector_trainable(True)
    )
    if interaction_selector_count <= 0:
        raise RuntimeError("Interaction-selector boundary is empty")
    interaction_selector_prefixes = tuple(
        f"{name}."
        for name in policy.actor.INTERACTION_SELECTOR_MODULE_NAMES
    )
    unexpected_interaction_selector_trainable = [
        name
        for name, parameter in policy.actor.named_parameters()
        if parameter.requires_grad
        and not name.startswith(interaction_selector_prefixes)
    ]
    if unexpected_interaction_selector_trainable:
        raise RuntimeError(
            "Interaction-selector boundary exposed shared parameters: "
            + ", ".join(unexpected_interaction_selector_trainable)
        )
    for module_name in policy.actor.INTERACTION_SELECTOR_MODULE_NAMES:
        module = getattr(policy.actor, module_name)
        if not all(
            parameter.requires_grad for parameter in module.parameters()
        ):
            raise RuntimeError(
                f"Interaction-selector module was frozen: {module_name}"
            )
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    survival_count = policy.actor.set_payload_survival_trainable(True)
    if survival_count <= 0:
        raise RuntimeError("Payload-survival boundary is empty")
    survival_prefixes = (
        "payload_survival_encoder.",
        "payload_survival_head.",
    )
    unexpected_survival_trainable = [
        name
        for name, parameter in policy.actor.named_parameters()
        if parameter.requires_grad
        and not name.startswith(survival_prefixes)
    ]
    if unexpected_survival_trainable:
        raise RuntimeError(
            "Payload-survival boundary exposed shared parameters: "
            + ", ".join(unexpected_survival_trainable)
        )
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    adapter_task, adapter_motion, adapter_interaction = (
        policy.actor.recovery_adapter_outputs(
            obs["hierarchy_context"][:, :GLOBAL_CONTEXT_DIM],
            context_slots,
        )
    )
    if (
        float(adapter_task.abs().max()) > 1.0e-8
        or float(adapter_motion.abs().max()) > 1.0e-8
        or float(adapter_interaction.abs().max()) > 1.0e-8
    ):
        raise RuntimeError("Fresh recovery adapter is not neutral")
    legacy_state = {
        name: value
        for name, value in policy.state_dict().items()
        if not name.startswith(
            policy.OPTIONAL_RECOVERY_ADAPTER_PREFIXES
        )
    }
    policy.load_state_dict(legacy_state, strict=True)

    copied = load_zyb_baseline_physical(policy, args.checkpoint)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    source = payload.get("model_state_dict", payload)
    normalized = policy.actor_obs_normalizer(obs["policy"])
    baseline_hidden = F.elu(
        F.linear(
            normalized[:, :876],
            source["actor.0.weight"].to(device),
            source["actor.0.bias"].to(device),
        )
    )
    baseline_hidden = F.elu(
        F.linear(
            baseline_hidden,
            source["actor.2.weight"].to(device),
            source["actor.2.bias"].to(device),
        )
    )
    baseline_mean = F.linear(
        baseline_hidden,
        source["actor.4.weight"].to(device),
        source["actor.4.bias"].to(device),
    )
    tactic_mean = policy.act_inference(obs)[:, :16]
    skill_survival = policy.actor.last_skill_survival
    if (
        skill_survival is None
        or float((skill_survival - 0.5).abs().max()) > 1.0e-8
        or int(policy.actor.payload_survival_updates.item()) != 0
    ):
        raise RuntimeError("Fresh payload-survival model is not neutral")
    leg_error = (baseline_mean[:, :12] - tactic_mean[:, :12]).abs().max()
    idle_wheel_effort = tactic_mean[:, 12:16].abs().max()
    if float(leg_error) > 1.0e-6:
        raise RuntimeError(
            f"Baseline leg-policy equivalence failed: {float(leg_error)}"
        )
    if float(idle_wheel_effort) > 1.0e-6:
        raise RuntimeError(
            f"Idle wheel effort is not zero: {float(idle_wheel_effort)}"
        )
    affordance_slots = torch.zeros(
        1,
        ACTION_LAYOUT.task_dim,
        layout.TASK_SLOT_FEATURE_DIM,
        device=device,
    )
    affordance_slots[:, :3, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    affordance_slots[:, :3, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    affordance_slots[:, :3, layout.TASK_SLOT_REMAINING_PROGRESS_INDEX] = 1.0
    affordance_slots[:, 0, layout.TASK_SLOT_DISTANCE_INDEX] = 0.15
    affordance_slots[:, 1, layout.TASK_SLOT_DISTANCE_INDEX] = 0.75
    affordance_slots[:, 2, layout.TASK_SLOT_DISTANCE_INDEX] = 0.30
    affordance_slots[:, 2, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    affordance_slots[:, 2, layout.TASK_SLOT_OBJECT_DELTA_SLICE] = 0.20
    affordance_slots[:, 2, layout.TASK_SLOT_REACHABILITY_INDEX] = 0.80
    grounded_affordance = policy.actor.grounded_task_utility(
        affordance_slots,
        obs["hierarchy_context"][:1, :GLOBAL_CONTEXT_DIM],
    )
    if not grounded_affordance[0, 0] > grounded_affordance[0, 1]:
        raise RuntimeError("Grounded task credit did not prefer the nearer goal")
    engaged_slots = affordance_slots.clone()
    engaged_slots[:, 2, layout.TASK_SLOT_INTERACTION_STATE_SLICE] = (
        torch.tensor((0.9, 0.7, 0.4, 0.0), device=device)
    )
    engaged_slots[:, 2, layout.TASK_SLOT_TARGET_DELTA_SLICE] = 0.08
    engaged_affordance = policy.actor.grounded_task_utility(
        engaged_slots,
        obs["hierarchy_context"][:1, :GLOBAL_CONTEXT_DIM],
    )
    if not engaged_affordance[0, 2] > grounded_affordance[0, 2]:
        raise RuntimeError(
            "Grounded task credit did not preserve an active delivery chain"
        )
    recovery_slots = torch.zeros_like(affordance_slots)
    recovery_slots[:, [0, 4], layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    recovery_slots[:, [0, 4], layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    recovery_slots[:, 0, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    recovery_slots[
        :, 4, layout.TASK_SLOT_RECOVERY_TYPE_INDEX
    ] = 1.0
    safe_context = torch.zeros(
        1, GLOBAL_CONTEXT_DIM, device=device
    )
    safe_context[:, layout.SUPPORT_COUNT_INDEX] = 1.0
    safe_context[:, layout.SAFETY_MARGIN_INDEX] = 1.0
    safe_context[:, layout.PREVIEW_MARGIN_INDEX] = 1.0
    unsafe_context = safe_context.clone()
    unsafe_context[:, layout.BASE_TILT_INDEX] = 0.80
    unsafe_context[:, layout.SUPPORT_COUNT_INDEX] = 0.50
    unsafe_context[:, layout.SAFETY_MARGIN_INDEX] = 0.0
    unsafe_context[:, layout.PREVIEW_MARGIN_INDEX] = 0.0
    safe_recovery_utility = policy.actor.grounded_task_utility(
        recovery_slots, safe_context
    )
    unsafe_recovery_utility = policy.actor.grounded_task_utility(
        recovery_slots, unsafe_context
    )
    if float(
        (
            unsafe_recovery_utility[:, 4]
            - safe_recovery_utility[:, 4]
        ).min()
    ) <= 0.20:
        raise RuntimeError(
            "Control pressure did not raise recovery task utility"
        )
    latched_context = obs["hierarchy_context"][:1].clone()
    latched_global = latched_context[:, :GLOBAL_CONTEXT_DIM]
    latched_slots = latched_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(1, ACTION_LAYOUT.task_dim, -1)
    latched_slots.zero_()
    latched_slots[:, [4, 9], layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    latched_slots[:, [4, 9], layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    latched_slots[
        :, 4, layout.TASK_SLOT_RECOVERY_TYPE_INDEX
    ] = 1.0
    latched_slots[:, 9, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    latched_global[:, 1] = 0.50
    latched_global[:, layout.EXECUTED_TASK_INDEX] = 9.0 / 11.0
    latched_global[:, layout.TERMINATION_STATE_SLICE] = 0.0
    latched_global[:, layout.BASE_TILT_INDEX] = 0.0
    latched_global[:, layout.SUPPORT_COUNT_INDEX] = 1.0
    latched_global[:, layout.SAFETY_MARGIN_INDEX] = 1.0
    latched_global[:, layout.PREVIEW_MARGIN_INDEX] = 1.0
    safe_latched_logits, _ = policy.actor.task_choice(
        latched_context, apply_commitment=True
    )
    if float(safe_latched_logits[0, 4]) > -19.0:
        raise RuntimeError(
            "Safe task commitment was released without margin pressure"
        )
    latched_global[:, layout.SAFETY_MARGIN_INDEX] = 0.0
    latched_global[:, layout.PREVIEW_MARGIN_INDEX] = 0.0
    unsafe_latched_logits, _ = policy.actor.task_choice(
        latched_context, apply_commitment=True
    )
    if float(unsafe_latched_logits[0, 4]) <= -19.0:
        raise RuntimeError(
            "Margin pressure did not reopen learned task selection"
        )
    policy.actor.task_choice(
        obs["hierarchy_context"], apply_commitment=False
    )
    if policy.actor.last_task_blended_utility is None:
        raise RuntimeError("Grounded task utility was not exposed")
    cold_start_residual = (
        policy.actor.last_task_blended_utility
        - policy.actor.last_task_grounded_utility
    ).abs().max()
    if float(cold_start_residual) > 1.0e-7:
        raise RuntimeError(
            "Uncalibrated outcomes affected cold-start task selection"
        )
    scoped_context = obs["hierarchy_context"].clone()
    scoped_slots = scoped_context[:, GLOBAL_CONTEXT_DIM:].view(
        batch_size, ACTION_LAYOUT.task_dim, -1
    )
    scoped_slots.zero_()
    scoped_slots[:, :2, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    scoped_slots[:, :2, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    scoped_slots[:, 0, 5] = 1.0
    scoped_slots[:, 1, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    _, scoped_mission = policy.actor.task_choice(
        scoped_context, apply_commitment=False
    )
    route_code = F.one_hot(
        torch.zeros(batch_size, dtype=torch.long, device=device),
        num_classes=ACTION_LAYOUT.task_dim,
    ).to(dtype=obs["policy"].dtype)
    _, route_subgoal, _ = policy.actor.task_detail_parameters(
        route_code, scoped_mission
    )
    route_arm_leakage = route_subgoal[:, 3:6].abs().max()
    if float(route_arm_leakage) > 1.0e-7:
        raise RuntimeError(
            "A non-interaction task activated the manipulation subspace"
        )
    delivery_code = F.one_hot(
        torch.ones(batch_size, dtype=torch.long, device=device),
        num_classes=ACTION_LAYOUT.task_dim,
    ).to(dtype=obs["policy"].dtype)
    _, far_delivery_subgoal, _ = policy.actor.task_detail_parameters(
        delivery_code, scoped_mission
    )
    far_delivery_arm_leakage = far_delivery_subgoal[:, 3:6].abs().max()
    if float(far_delivery_arm_leakage) > 1.0e-7:
        raise RuntimeError(
            "An unreachable delivery activated the manipulation subspace"
        )
    prior_context = scoped_context.clone()
    prior_context[:, layout.SAFETY_MARGIN_INDEX] = 1.0
    prior_context[:, layout.PREVIEW_MARGIN_INDEX] = 1.0
    prior_context[:, layout.BASE_VX_INDEX] = 0.0
    prior_context[:, layout.COMMAND_VX_INDEX] = 0.0
    prior_context[:, layout.BASE_TILT_INDEX] = 0.0
    prior_context[:, layout.SUPPORT_COUNT_INDEX] = 1.0
    prior_context[:, layout.CURRICULUM_LEVEL_INDEX] = 0.0
    prior_slots = prior_context[:, GLOBAL_CONTEXT_DIM:].view(
        batch_size, ACTION_LAYOUT.task_dim, -1
    )
    prior_slots[:, 0, 0] = 0.5
    prior_slots[:, 0, 1] = 0.0
    prior_slots[:, 0, layout.TASK_SLOT_DISTANCE_INDEX] = 0.375
    relational_prior, relational_authority = (
        policy.actor.relational_task_subgoal_prior(
            prior_slots[:, 0], prior_context[:, :GLOBAL_CONTEXT_DIM]
        )
    )
    if float(relational_prior[:, 0].mean()) <= 0.25:
        raise RuntimeError(
            "Relational task prior did not produce forward progress"
        )
    if not 0.49 <= float(relational_authority.mean()) <= 0.51:
        raise RuntimeError(
            "Cold-start task authority is outside the bounded prior"
        )
    measured_context = prior_context[:, :GLOBAL_CONTEXT_DIM].clone()
    measured_context[:, layout.BASE_TILT_INDEX] = 0.276 / 0.60
    measured_context[:, layout.SUPPORT_COUNT_INDEX] = 1.0
    measured_context[:, layout.SAFETY_MARGIN_INDEX] = 0.42
    measured_context[:, layout.PREVIEW_MARGIN_INDEX] = 0.42
    _, measured_authority = policy.actor.relational_task_subgoal_prior(
        prior_slots[:, 0], measured_context
    )
    if float(measured_authority.mean()) < 0.49:
        raise RuntimeError(
            "A measured stable posture was misread as a binding barrier"
        )
    prior_slots[:, 0, 0] = -0.35
    prior_slots[:, 0, 1] = 0.35
    prior_slots[:, 0, layout.TASK_SLOT_HEADING_INDEX] = 0.75
    turning_prior, _ = policy.actor.relational_task_subgoal_prior(
        prior_slots[:, 0], prior_context[:, :GLOBAL_CONTEXT_DIM]
    )
    turning_motion = policy.actor.task_subgoal_motion_components(
        turning_prior
    )[0]
    if float(turning_motion[:, 0].min()) <= 0.0:
        raise RuntimeError(
            "A rear target requested reverse motion instead of turning first"
        )
    if float(turning_motion[:, 1].min()) <= 0.05:
        raise RuntimeError(
            "A rear-left target did not produce a positive turn request"
        )
    delivery_slots = prior_slots.clone()
    delivery_slots[:, 0, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    delivery_slots[:, 0, layout.TASK_SLOT_REACHABILITY_INDEX] = 0.0
    delivery_slots[:, 0, layout.TASK_SLOT_HEADING_INDEX] = 0.0
    delivery_prior, _ = policy.actor.relational_task_subgoal_prior(
        delivery_slots[:, 0], prior_context[:, :GLOBAL_CONTEXT_DIM]
    )
    delivery_motion = policy.actor.task_subgoal_motion_components(
        delivery_prior
    )[0]
    if float(delivery_motion[:, 0].max()) >= -1.0e-4:
        raise RuntimeError(
            "A rear-mounted arm task did not preserve reverse approach"
        )
    delivery_slots[
        :, 0, layout.TASK_SLOT_CARRYING_INDEX
    ] = 1.0
    carrying_prior, _ = policy.actor.relational_task_subgoal_prior(
        delivery_slots[:, 0], prior_context[:, :GLOBAL_CONTEXT_DIM]
    )
    carrying_motion = policy.actor.task_subgoal_motion_components(
        carrying_prior
    )[0]
    if float(carrying_motion[:, 0].max()) >= -1.0e-4:
        raise RuntimeError(
            "A secured payload did not preserve relation-consistent reverse "
            f"transport: {float(carrying_motion[:, 0].max()):.6f}"
        )
    payload_context = prior_context.clone()
    payload_slots = payload_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(batch_size, ACTION_LAYOUT.task_dim, -1)
    payload_slots.zero_()
    payload_slots[:, 4, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    payload_slots[:, 4, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    payload_slots[
        :, 4, layout.TASK_SLOT_RECOVERY_TYPE_INDEX
    ] = 1.0
    payload_slots[:, 5, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    payload_slots[:, 5, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    payload_slots[:, 5, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    payload_slots[:, 5, layout.TASK_SLOT_CARRYING_INDEX] = 1.0
    payload_slots[:, 5, layout.TASK_SLOT_REACHABILITY_INDEX] = 1.0
    _, payload_mission = policy.actor.task_choice(
        payload_context, apply_commitment=False
    )
    safe_recovery_utility = policy.actor.grounded_task_utility(
        payload_slots,
        payload_context[:, :GLOBAL_CONTEXT_DIM],
    )[:, 4]
    unsafe_recovery_context = payload_context.clone()
    unsafe_recovery_context[:, layout.SAFETY_MARGIN_INDEX] = 0.0
    unsafe_recovery_context[:, layout.PREVIEW_MARGIN_INDEX] = 0.0
    unsafe_recovery_utility = policy.actor.grounded_task_utility(
        payload_slots,
        unsafe_recovery_context[:, :GLOBAL_CONTEXT_DIM],
    )[:, 4]
    policy.actor.task_choice(
        unsafe_recovery_context, apply_commitment=False
    )
    recovery_preference = policy.actor.last_recovery_task_preference
    if recovery_preference is None:
        raise RuntimeError("Recovery preference was not evaluated")
    if float(
        (unsafe_recovery_utility - safe_recovery_utility).min()
    ) <= 0.20:
        raise RuntimeError(
            "A binding margin did not increase recovery task utility"
        )
    if float(recovery_preference[:, 4].min()) < 0.99:
        raise RuntimeError(
            "A binding margin did not activate the learned recovery candidate"
        )
    recovery_prior, _ = policy.actor.relational_task_subgoal_prior(
        payload_slots[:, 4],
        unsafe_recovery_context[:, :GLOBAL_CONTEXT_DIM],
    )
    recovery_motion = policy.actor.task_subgoal_motion_components(
        recovery_prior
    )[0]
    if float(recovery_motion.abs().max()) > 1.0e-6:
        raise RuntimeError(
            "Recovery task did not define a zero-motion regulation subgoal"
        )
    task_block_dim = (
        ACTION_LAYOUT.task_dim
        + ACTION_LAYOUT.object_dim
        + ACTION_LAYOUT.task_subgoal_dim
        + 1
    )
    recovery_task_block = torch.zeros(
        batch_size, task_block_dim, device=device
    )
    recovery_task_block[:, 4] = 1.0
    policy.actor.skill_choice(
        recovery_task_block,
        payload_mission,
        apply_commitment=False,
    )
    if float(policy.actor.last_interaction_active.min()) < 0.99:
        raise RuntimeError(
            "Recovery task detached the skill layer from a carried payload"
        )
    unsafe_context = prior_context[:, :GLOBAL_CONTEXT_DIM].clone()
    unsafe_context[:, layout.SAFETY_MARGIN_INDEX] = 0.0
    unsafe_context[:, layout.PREVIEW_MARGIN_INDEX] = 0.0
    unsafe_carrying_prior, _ = (
        policy.actor.relational_task_subgoal_prior(
            delivery_slots[:, 0], unsafe_context
        )
    )
    unsafe_payload_progress = float(
        unsafe_carrying_prior[:, 0].abs().max()
    )
    safe_payload_progress = float(carrying_prior[:, 0].abs().max())
    if unsafe_payload_progress <= 1.0e-4:
        raise RuntimeError(
            "Low-margin payload transport lost its creep authority"
        )
    if unsafe_payload_progress >= 0.5 * safe_payload_progress:
        raise RuntimeError(
            "Low-margin payload transport was not sufficiently attenuated: "
            f"safe={safe_payload_progress:.6f} "
            f"unsafe={unsafe_payload_progress:.6f}"
        )
    overspeed_context = prior_context[:, :GLOBAL_CONTEXT_DIM].clone()
    overspeed_context[:, layout.BASE_VX_INDEX] = 0.55
    overspeed_context[:, layout.COMMAND_VX_INDEX] = 0.20
    overspeed_carrying_prior, _ = (
        policy.actor.relational_task_subgoal_prior(
            delivery_slots[:, 0], overspeed_context
        )
    )
    if float(overspeed_carrying_prior[:, 0].max()) >= 0.0:
        raise RuntimeError(
            "Payload overspeed did not request a task-level braking transient"
        )
    drive_obs = {
        key: value.clone() for key, value in obs.items()
    }
    drive_obs["hierarchy_context"][:, layout.COMMAND_VX_INDEX] = 0.10
    drive_obs["hierarchy_context"][:, layout.BASE_HEIGHT_INDEX] = 0.36
    drive_obs["hierarchy_context"][:, layout.BASE_TILT_INDEX] = 0.0
    drive_obs["hierarchy_context"][:, layout.SAFETY_MARGIN_INDEX] = 1.0
    drive_obs["hierarchy_context"][:, layout.PREVIEW_MARGIN_INDEX] = 1.0
    drive_obs["hierarchy_context"][:, layout.CLF_DECREASE_INDEX] = 1.0
    drive_obs["hierarchy_context"][:, layout.DISTURBANCE_QUALITY_INDEX] = 1.0
    drive_obs["hierarchy_context"][:, layout.SUPPORT_COUNT_INDEX] = 1.0
    drive_obs["hierarchy_context"][:, layout.CURRICULUM_LEVEL_INDEX] = 0.0
    policy.act_inference(drive_obs)
    wheel_prior = policy.actor.last_wheel_prior
    wheel_gate = policy.actor.last_wheel_skill_gate
    wheel_control_authority = policy.actor.last_wheel_control_authority
    motion_candidates = policy.actor.last_motion_action_candidates
    support_reference = policy.actor.last_support_reference
    support_gate = policy.actor.last_support_gate
    if wheel_prior is None or motion_candidates is None:
        raise RuntimeError("Motion diagnostics were not evaluated")
    if wheel_gate is None or wheel_control_authority is None:
        raise RuntimeError("Wheel authority envelope was not evaluated")
    if float(wheel_prior.abs().mean()) <= 0.10:
        raise RuntimeError(
            "The learned motion skill decoder did not produce wheel effort"
        )
    if float(wheel_gate.min()) <= 0.0:
        raise RuntimeError("The selected motion skill has no execution gate")
    if float(wheel_control_authority.min()) < 0.99:
        raise RuntimeError(
            "An interior control margin attenuated the motion skill"
        )
    if support_reference is None or support_gate is None:
        raise RuntimeError("Motion support skill was not evaluated")
    progress_action = motion_candidates[:, 1]
    print(
        "progress_action_abs_mean={:.6f} progress_action_mean={}".format(
            float(progress_action.abs().mean()),
            progress_action.mean(dim=0).detach().cpu().tolist(),
        )
    )
    if not 0.15 < float(progress_action.abs().mean()) <= 24.0:
        raise RuntimeError(
            "Progress-skill effort is outside the bounded action range"
        )
    chart_context = drive_obs["hierarchy_context"][
        :, :GLOBAL_CONTEXT_DIM
    ].clone()
    chart_context[:, layout.COMMAND_VY_INDEX] = 0.10
    chart_context[:, layout.COMMAND_WZ_INDEX] = 0.0
    feasible_request = policy.actor._spatial_to_feasible_motion(
        chart_context[:, layout.COMMAND_VX_INDEX],
        chart_context[:, layout.COMMAND_VY_INDEX],
        chart_context[:, layout.COMMAND_WZ_INDEX],
    )
    reverse_request = policy.actor._spatial_to_feasible_motion(
        -torch.full((batch_size,), 0.10, device=device),
        torch.full((batch_size,), 0.10, device=device),
        torch.full((batch_size,), 0.20, device=device),
    )
    if not torch.all(reverse_request[:, 1] < 0.0):
        raise RuntimeError(
            "Reverse travel did not project lateral/yaw intent into the "
            "feasible B2W direction"
        )
    chart_candidates = policy.actor._motion_action_candidates(
        chart_context, motion_request=feasible_request
    )
    chart_effect, chart_confidence = policy.predict_motion_execution(
        chart_context, chart_candidates
    )
    signed_sweep_action = 18.0 * torch.tensor(
        (
            (1.0, 1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        device=device,
    )
    signed_response = policy.actor.embodiment_response_prior(
        signed_sweep_action
    )
    if not (
        float(signed_response[0, 0]) > 0.0
        and float(signed_response[0, 1]) < 0.0
        and float(signed_response[1, 0]) > 0.0
        and float(signed_response[1, 1]) > 0.0
        and float(signed_response[2, 1]) > 0.0
        and float(signed_response[3, 1]) < 0.0
    ):
        raise RuntimeError("Embodiment response prior lost its signed chart")
    signed_response_options = signed_response.unsqueeze(0)
    positive_turn_request = signed_response.new_tensor(((0.0, 0.12),))
    negative_turn_request = signed_response.new_tensor(((0.0, -0.12),))
    response_margin = signed_response.new_ones(1)
    _, positive_turn_score, positive_turn_authority = (
        policy.actor.identified_motion_response_score(
            signed_response_options,
            positive_turn_request,
            response_margin,
        )
    )
    _, negative_turn_score, negative_turn_authority = (
        policy.actor.identified_motion_response_score(
            signed_response_options,
            negative_turn_request,
            response_margin,
        )
    )
    if int(positive_turn_score.argmax(dim=1).item()) != 2:
        raise RuntimeError(
            "Positive yaw request did not select the identified positive chart"
        )
    if int(negative_turn_score.argmax(dim=1).item()) != 3:
        raise RuntimeError(
            "Negative yaw request did not select the identified negative chart"
        )
    if not (
        float(positive_turn_authority.item()) > 0.0
        and float(negative_turn_authority.item()) > 0.0
    ):
        raise RuntimeError("Identified response selector has no authority")
    lateral_task_block = torch.zeros(
        batch_size, task_block_dim, device=device
    )
    subgoal_start = ACTION_LAYOUT.task_dim + ACTION_LAYOUT.object_dim
    lateral_task_block[:, subgoal_start + 1] = 0.8
    current_task_request = policy.actor.task_motion_request(
        lateral_task_block
    )
    candidate_span = (
        chart_candidates.amax(dim=1)
        - chart_candidates.amin(dim=1)
    ).mean()
    transit_effort = chart_candidates[:, 1].abs().mean()
    maneuver_difference = (
        chart_candidates[:, 2] - chart_candidates[:, 1]
    ).abs().mean()
    if float(candidate_span) <= 0.15:
        raise RuntimeError("Motion action chart does not separate options")
    if not torch.isfinite(chart_effect).all() or not torch.isfinite(
        chart_confidence
    ).all():
        raise RuntimeError("Cold-start motion successor is non-finite")
    if float(feasible_request[:, 1].mean()) <= 0.10:
        raise RuntimeError("Lateral task intent did not reach motion skills")
    if float(current_task_request[:, 1].mean()) <= 0.10:
        raise RuntimeError(
            "Current task subgoal did not reach the skill layer"
        )
    if float(transit_effort) <= 0.05:
        raise RuntimeError("Transit option lacks physical effort")
    if float(maneuver_difference) <= 0.05:
        raise RuntimeError("Maneuver option lacks distinct authority")
    if not 0.0 < float(support_gate.mean()) < 0.10:
        raise RuntimeError(
            "The motion skill support gate is outside its conservative start"
        )
    if not torch.isfinite(support_reference).all():
        raise RuntimeError("Baseline support action contains non-finite values")
    actions = policy.act(obs)
    termination_slice = ACTION_LAYOUT.slices()["termination"]
    termination_sampling_error = (
        actions[:, termination_slice]
        - policy.action_mean[:, termination_slice]
    ).abs().max()
    if float(termination_sampling_error) > 1.0e-7:
        raise RuntimeError(
            "Option termination must be a deterministic learned hazard"
        )
    log_prob = policy.get_actions_log_prob(actions)
    value = policy.evaluate(obs)
    next_context = obs["hierarchy_context"].clone()
    next_context[:, 0] = (next_context[:, 0] + 0.02).clamp_max(1.0)
    task_transition, skill_transition = policy.transition_logits(
        obs["hierarchy_context"], next_context
    )
    task_outcome, skill_outcome = policy.predict_option_outcomes(
        obs["hierarchy_context"], actions
    )
    (
        candidate_task_logits,
        candidate_task_subgoal,
        candidate_raw_slots,
        candidate_task_utility,
        candidate_task_confidence,
    ) = policy.actor.candidate_task_details(
        obs["hierarchy_context"]
    )
    task_outcome_confidence = (
        policy.actor.last_task_outcome_confidence
    )
    skill_effect, skill_effect_confidence = policy.predict_skill_effects(
        obs["hierarchy_context"], actions
    )
    task_constraint_multiplier = (
        policy.actor.last_task_constraint_multiplier
    )
    skill_constraint_multipliers = (
        policy.actor.last_skill_constraint_multipliers
    )
    skill_constraint_violation = (
        policy.actor.last_skill_constraint_violation
    )
    if (
        task_outcome_confidence is None
        or task_constraint_multiplier is None
        or skill_constraint_multipliers is None
        or skill_constraint_violation is None
    ):
        raise RuntimeError("Hierarchy world-model diagnostics are missing")
    motion_execution, motion_execution_confidence = (
        policy.predict_motion_execution(
            obs["hierarchy_context"], actions[:, 12:16]
        )
    )
    loss = -(log_prob.mean()) + 0.01 * value.square().mean()
    loss = loss + 0.01 * (
        task_transition.square().mean()
        + skill_transition.square().mean()
        + task_outcome.square().mean()
        + task_outcome_confidence.square().mean()
        + candidate_task_logits.square().mean()
        + candidate_task_subgoal.square().mean()
        + candidate_task_utility.square().mean()
        + candidate_task_confidence.square().mean()
        + skill_outcome.square().mean()
        + skill_effect.square().mean()
        + skill_effect_confidence.square().mean()
        + task_constraint_multiplier.square().mean()
        + skill_constraint_multipliers.square().mean()
        + skill_constraint_violation.square().mean()
        + motion_execution.square().mean()
        + motion_execution_confidence.square().mean()
    )
    loss.backward()
    actor_parameters = dict(policy.actor.named_parameters())
    for parameter_name in (
        "task_query.weight",
        "motion_skill_logits_head.weight",
        "interaction_skill_logits_head.weight",
    ):
        gradient = actor_parameters[parameter_name].grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all()
            or float(gradient.abs().sum()) <= 0.0
        ):
            raise RuntimeError(
                f"Learned selector has no usable gradient: {parameter_name}"
            )
    response_gradient = policy.actor.embodiment_response_matrix.grad
    if response_gradient is None or not torch.isfinite(
        response_gradient
    ).all():
        raise RuntimeError("Embodiment response chart is not trainable")

    if actions.shape != (batch_size, ACTION_LAYOUT.total_dim):
        raise RuntimeError(f"Unexpected action shape: {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("Non-finite policy action")
    if not torch.isfinite(log_prob).all() or not torch.isfinite(value).all():
        raise RuntimeError("Non-finite PPO output")
    if policy.actor.last_control_prediction is None:
        raise RuntimeError("Control-aware representation head was not evaluated")
    if task_transition.shape != (batch_size, ACTION_LAYOUT.task_dim):
        raise RuntimeError("Unexpected task-transition logits")
    if skill_transition.shape != (batch_size, ACTION_LAYOUT.skill_dim):
        raise RuntimeError("Unexpected skill-transition logits")
    if task_outcome.shape != (batch_size, layout.TASK_OUTCOME_DIM):
        raise RuntimeError("Unexpected task-outcome prediction")
    if candidate_task_subgoal.shape != (
        batch_size,
        ACTION_LAYOUT.task_dim,
        ACTION_LAYOUT.task_subgoal_dim,
    ):
        raise RuntimeError("Unexpected candidate task-subgoal tensor")
    if candidate_raw_slots.shape != (
        batch_size,
        ACTION_LAYOUT.task_dim,
        layout.TASK_SLOT_FEATURE_DIM,
    ):
        raise RuntimeError("Unexpected candidate task-slot tensor")
    if skill_outcome.shape != (batch_size, layout.SKILL_OUTCOME_DIM):
        raise RuntimeError("Unexpected skill-outcome prediction")
    if skill_effect.shape != (batch_size, layout.SKILL_EFFECT_DIM):
        raise RuntimeError("Unexpected skill-effect prediction")
    if skill_effect_confidence.shape != (batch_size,):
        raise RuntimeError("Unexpected skill-effect confidence")
    if motion_execution.shape != (
        batch_size,
        layout.MOTION_EXECUTION_EFFECT_DIM,
    ):
        raise RuntimeError("Unexpected motion-execution prediction")
    if motion_execution_confidence.shape != (batch_size,):
        raise RuntimeError("Unexpected motion-execution confidence")

    phase_slots = torch.zeros(
        4,
        layout.TASK_SLOT_FEATURE_DIM,
        device=device,
    )
    phase_slots[0, layout.TASK_SLOT_OBJECT_DELTA_SLICE] = torch.tensor(
        (1.0, 0.0, 0.0), device=device
    )
    phase_slots[
        0, layout.TASK_SLOT_LEFT_FINGER_DELTA_SLICE
    ] = torch.tensor((1.0, 0.0, 0.0), device=device)
    phase_slots[
        0, layout.TASK_SLOT_RIGHT_FINGER_DELTA_SLICE
    ] = torch.tensor((1.0, 0.0, 0.0), device=device)
    phase_slots[1, layout.TASK_SLOT_OBJECT_DELTA_SLICE] = torch.tensor(
        (0.03, 0.0, 0.0), device=device
    )
    phase_slots[
        1, layout.TASK_SLOT_LEFT_FINGER_DELTA_SLICE
    ] = torch.tensor((0.0, 0.02 / 0.75, 0.0), device=device)
    phase_slots[
        1, layout.TASK_SLOT_RIGHT_FINGER_DELTA_SLICE
    ] = torch.tensor((0.0, -0.02 / 0.75, 0.0), device=device)
    phase_slots[2, 22:25] = 1.0
    phase_slots[2, layout.TASK_SLOT_CARRYING_INDEX] = 1.0
    phase_slots[2, layout.TASK_SLOT_TARGET_DELTA_SLICE] = torch.tensor(
        (0.18 / 1.5, 0.0, 0.0), device=device
    )
    phase_slots[3, 22:25] = 1.0
    phase_slots[3, layout.TASK_SLOT_CARRYING_INDEX] = 1.0
    phase_slots[3, layout.TASK_SLOT_TARGET_DELTA_SLICE] = torch.tensor(
        (
            0.05 / 1.5,
            0.0,
            -layout.RELEASE_HOVER_HEIGHT / 1.5,
        ),
        device=device,
    )
    phase_probability = policy.actor.interaction_phase_distribution(
        phase_slots
    )
    expected_phase = torch.tensor((0, 1, 1, 2), device=device)
    if not torch.equal(
        torch.argmax(phase_probability, dim=-1), expected_phase
    ):
        raise RuntimeError(
            "Interaction stages are not ordered as approach, secure, release"
        )
    capture_feasibility = (
        policy.actor.last_interaction_capture_feasibility
    )
    release_feasibility = (
        policy.actor.last_interaction_release_feasibility
    )
    release_frontier = policy.actor.last_interaction_release_frontier
    if (
        capture_feasibility is None
        or float(capture_feasibility[0]) > 0.05
        or float(capture_feasibility[1]) < 0.80
    ):
        raise RuntimeError(
            "Relation-space capture feasibility is not geometry selective"
        )
    if (
        release_feasibility is None
        or release_frontier is None
        or float(release_feasibility[2]) > 0.05
        or float(release_feasibility[3]) < 0.80
        or float(release_frontier[2]) < 0.35
    ):
        raise RuntimeError(
            "Release feasibility did not enforce the terminal relation set"
        )

    event_context = torch.zeros(
        2, layout.HIERARCHY_CONTEXT_DIM, device=device
    )
    event_next_context = event_context.clone()
    event_slots = event_context[:, GLOBAL_CONTEXT_DIM:].view(
        2, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM
    )
    event_next_slots = event_next_context[:, GLOBAL_CONTEXT_DIM:].view(
        2, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM
    )
    event_slots[:, 5, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    event_slots[:, 5, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    event_slots[0, 5, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    event_next_slots.copy_(event_slots)
    event_next_slots[
        :, 5, layout.TASK_SLOT_INTERACTION_STATE_SLICE
    ] = torch.tensor((1.0, 0.5, 0.3, 0.0), device=device)
    event_next_slots[
        :, 5, layout.TASK_SLOT_GRIPPER_CLOSURE_INDEX
    ] = 0.6
    event_next_slots[
        :, 5, layout.TASK_SLOT_CONTACT_SYMMETRY_INDEX
    ] = 0.8
    event_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    event_next_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    event_actions = torch.zeros(
        2, ACTION_LAYOUT.total_dim, device=device
    )
    slices = ACTION_LAYOUT.slices()
    event_actions[:, slices["task"].start + 6] = 1.0
    event_actions[:, slices["object"].start + 1] = 1.0
    event_actions[:, slices["skill"].start + 6] = 1.0
    (
        relabeled_event_actions,
        event_priority,
        event_task_valid,
        event_task_target,
        event_recovery_valid,
    ) = TACTICPPO._hindsight_event_actions(
        event_context,
        event_next_context,
        event_actions,
        torch.zeros(2, device=device),
    )
    if event_task_target.tolist() != [5, 5]:
        raise RuntimeError("Physical event was not relabeled to its object slot")
    if torch.any(event_recovery_valid):
        raise RuntimeError("A physical interaction was mislabeled as recovery")
    if event_task_valid.tolist() != [1.0, 0.0]:
        raise RuntimeError("Event task credit ignored the required-task mask")
    if (
        relabeled_event_actions[:, slices["object"]]
        .argmax(dim=1)
        .tolist()
        != [0, 0]
    ):
        raise RuntimeError("Physical event was not relabeled to object zero")
    if (
        relabeled_event_actions[:, slices["skill"]]
        .argmax(dim=1)
        .tolist()
        != [7, 7]
    ):
        raise RuntimeError(
            "Lift relabeling did not preserve the motion-skill factor"
        )
    if not torch.all(event_priority > 0.0):
        raise RuntimeError("Grounded event priority is not positive")

    approach_context = torch.zeros(
        1, layout.HIERARCHY_CONTEXT_DIM, device=device
    )
    approach_next_context = approach_context.clone()
    approach_slots = approach_context[:, GLOBAL_CONTEXT_DIM:].view(
        1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM
    )
    approach_next_slots = approach_next_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM)
    approach_slots[:, 5, layout.TASK_SLOT_DELIVERY_TYPE_INDEX] = 1.0
    approach_slots[:, 5, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    approach_slots[:, 5, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    approach_slots[
        :, 5, layout.TASK_SLOT_OBJECT_DELTA_SLICE
    ] = torch.tensor((0.50, 0.0, 0.0), device=device)
    approach_next_slots.copy_(approach_slots)
    approach_next_slots[
        :, 5, layout.TASK_SLOT_OBJECT_DELTA_SLICE
    ] = torch.tensor((0.48, 0.0, 0.0), device=device)
    approach_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    approach_next_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    approach_actions = torch.zeros(
        1, ACTION_LAYOUT.total_dim, device=device
    )
    approach_actions[:, slices["task"].start + 5] = 1.0
    approach_actions[:, slices["object"].start] = 1.0
    approach_actions[:, slices["skill"].start + 6] = 1.0
    (
        relabeled_approach_actions,
        approach_priority,
        _,
        _,
        _,
    ) = TACTICPPO._hindsight_event_actions(
        approach_context,
        approach_next_context,
        approach_actions,
        torch.zeros(1, device=device),
    )
    if float(approach_priority[0]) <= 0.0:
        raise RuntimeError("Approach progress did not enter event replay")
    if (
        int(
            relabeled_approach_actions[
                0, slices["skill"]
            ].argmax().item()
        )
        != 6
    ):
        raise RuntimeError("Approach replay changed the interaction phase")

    unsafe_approach_context = approach_context.clone()
    unsafe_approach_next_context = approach_next_context.clone()
    unsafe_approach_context[:, layout.CONTROL_TARGET_SLICE] = 0.02
    unsafe_approach_next_context[
        :, layout.CONTROL_TARGET_SLICE
    ] = 0.02
    _, unsafe_approach_priority, _, _, _ = (
        TACTICPPO._hindsight_event_actions(
            unsafe_approach_context,
            unsafe_approach_next_context,
            approach_actions,
            torch.zeros(1, device=device),
        )
    )
    if float(unsafe_approach_priority[0]) >= (
        0.01 * float(approach_priority[0])
    ):
        raise RuntimeError("Unsafe progress was not removed from event credit")

    recovery_event_context = torch.zeros(
        1, layout.HIERARCHY_CONTEXT_DIM, device=device
    )
    recovery_event_next_context = recovery_event_context.clone()
    recovery_event_slots = recovery_event_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM)
    recovery_event_next_slots = recovery_event_next_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM)
    recovery_event_slots[
        :, 4, layout.TASK_SLOT_RECOVERY_TYPE_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 4, layout.TASK_SLOT_REQUIRED_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 4, layout.TASK_SLOT_AVAILABLE_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 5, layout.TASK_SLOT_DELIVERY_TYPE_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 5, layout.TASK_SLOT_REQUIRED_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 5, layout.TASK_SLOT_AVAILABLE_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 5, layout.TASK_SLOT_CARRYING_INDEX
    ] = 1.0
    recovery_event_slots[
        :, 5, layout.TASK_SLOT_INTERACTION_STATE_SLICE
    ] = 1.0
    recovery_event_next_slots.copy_(recovery_event_slots)
    recovery_event_context[
        :, layout.CONTROL_TARGET_SLICE
    ] = torch.tensor((0.02, 0.02, 0.30, 0.30), device=device)
    recovery_event_next_context[
        :, layout.CONTROL_TARGET_SLICE
    ] = torch.tensor((0.10, 0.10, 0.50, 0.50), device=device)
    recovery_event_context[:, layout.BASE_TILT_INDEX] = 0.80
    recovery_event_next_context[:, layout.BASE_TILT_INDEX] = 0.50
    recovery_event_context[:, layout.SUPPORT_COUNT_INDEX] = 0.50
    recovery_event_next_context[:, layout.SUPPORT_COUNT_INDEX] = 0.75
    recovery_event_actions = torch.zeros(
        1, ACTION_LAYOUT.total_dim, device=device
    )
    recovery_event_actions[:, slices["task"].start + 5] = 1.0
    recovery_event_actions[:, slices["object"].start] = 1.0
    recovery_event_actions[:, slices["skill"].start + 6] = 1.0
    (
        relabeled_recovery_actions,
        recovery_event_priority,
        recovery_event_valid,
        recovery_event_target,
        recovery_event_mask,
    ) = TACTICPPO._hindsight_event_actions(
        recovery_event_context,
        recovery_event_next_context,
        recovery_event_actions,
        torch.zeros(1, device=device),
    )
    if (
        int(recovery_event_target[0]) != 4
        or float(recovery_event_priority[0]) <= 0.0
        or float(recovery_event_valid[0]) < 0.99
        or not bool(recovery_event_mask[0])
    ):
        raise RuntimeError(
            "Payload recovery did not select the recovery subtask"
        )
    if int(
        relabeled_recovery_actions[
            0, slices["skill"]
        ].argmax().item()
    ) != 7:
        raise RuntimeError(
            "Recovery hindsight did not retain the measured motion skill"
        )

    payload_context = torch.zeros(
        3, layout.HIERARCHY_CONTEXT_DIM, device=device
    )
    payload_next_context = payload_context.clone()
    payload_slots = payload_context[:, GLOBAL_CONTEXT_DIM:].view(
        3, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM
    )
    payload_next_slots = payload_next_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(3, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM)
    payload_slots[:, 5, layout.TASK_SLOT_CARRYING_INDEX] = 1.0
    payload_next_slots[0, 5, layout.TASK_SLOT_CARRYING_INDEX] = 1.0
    payload_next_slots[
        1,
        5,
        layout.TASK_SLOT_INTERACTION_STATE_SLICE.stop - 1,
    ] = 1.0
    (
        payload_support,
        payload_target,
        payload_drop,
    ) = TACTICPPO._payload_survival_targets(
        payload_context, payload_next_context
    )
    if (
        not torch.allclose(payload_support, torch.ones_like(payload_support))
        or not torch.equal(
            payload_target,
            payload_target.new_tensor((1.0, 1.0, 0.0)),
        )
        or not torch.equal(
            payload_drop,
            payload_drop.new_tensor((0.0, 0.0, 1.0)),
        )
    ):
        raise RuntimeError("Payload survival labels are inconsistent")
    payload_probe = object.__new__(TACTICPPO)
    payload_probe.payload_survival_replay_drop_fraction = 0.50
    payload_probe._payload_replay_drop = torch.cat(
        (
            torch.ones(5, device=device),
            torch.zeros(95, device=device),
        )
    )
    payload_indices = payload_probe._sample_payload_survival_indices(40)
    payload_sample_drop = payload_probe._payload_replay_drop[
        payload_indices
    ].mean()
    if abs(float(payload_sample_drop) - 0.50) > 1.0e-6:
        raise RuntimeError(
            "Payload survival replay did not balance rare drops"
        )

    route_context = torch.zeros(
        1, layout.HIERARCHY_CONTEXT_DIM, device=device
    )
    route_next_context = route_context.clone()
    route_slots = route_context[:, GLOBAL_CONTEXT_DIM:].view(
        1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM
    )
    route_next_slots = route_next_context[
        :, GLOBAL_CONTEXT_DIM:
    ].view(1, ACTION_LAYOUT.task_dim, layout.TASK_SLOT_FEATURE_DIM)
    route_slots[:, 0, layout.TASK_SLOT_REQUIRED_INDEX] = 1.0
    route_slots[:, 0, layout.TASK_SLOT_AVAILABLE_INDEX] = 1.0
    route_slots[
        :, 0, layout.TASK_SLOT_REMAINING_PROGRESS_INDEX
    ] = 0.80
    route_next_slots.copy_(route_slots)
    route_next_slots[
        :, 0, layout.TASK_SLOT_REMAINING_PROGRESS_INDEX
    ] = 0.78
    route_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    route_next_context[:, layout.CONTROL_TARGET_SLICE] = 1.0
    route_actions = torch.zeros(
        1, ACTION_LAYOUT.total_dim, device=device
    )
    route_actions[:, slices["task"].start] = 1.0
    route_actions[:, slices["object"].start] = 1.0
    route_actions[:, slices["skill"].start + 6] = 1.0
    (
        relabeled_route_actions,
        route_priority,
        _,
        route_task_target,
        route_recovery_valid,
    ) = TACTICPPO._hindsight_event_actions(
        route_context,
        route_next_context,
        route_actions,
        torch.zeros(1, device=device),
    )
    if int(route_task_target[0]) != 0 or float(route_priority[0]) <= 0.0:
        raise RuntimeError("Safe route progress did not receive task credit")
    if torch.any(route_recovery_valid):
        raise RuntimeError("Safe route progress was mislabeled as recovery")
    if (
        int(
            relabeled_route_actions[
                0, slices["skill"]
            ].argmax().item()
        )
        != 6
    ):
        raise RuntimeError("Route replay changed the motion-skill factor")

    replay_probe = object.__new__(TACTICPPO)
    replay_probe.event_replay_delivery_fraction = 0.55
    replay_probe.event_replay_recovery_fraction = 0.15
    replay_probe.event_replay_secure_fraction = 0.30
    replay_probe.event_replay_release_fraction = 0.15
    replay_probe.event_replay_role_oversample_cap = 4.0
    replay_probe.event_replay_phase_oversample_cap = 4.0
    replay_probe._event_replay_actions = torch.zeros(
        100, ACTION_LAYOUT.total_dim, device=device
    )
    replay_probe._event_replay_priority = torch.ones(100, device=device)
    replay_probe._event_replay_recovery_valid = torch.zeros(
        100, dtype=torch.bool, device=device
    )
    replay_probe._event_replay_actions[:, slices["task"].start] = 1.0
    replay_probe._event_replay_actions[:, slices["skill"].start] = 1.0
    replay_probe._event_replay_actions[:20, slices["task"]].zero_()
    replay_probe._event_replay_actions[
        :20, slices["task"].start + 5
    ] = 1.0
    replay_probe._event_replay_actions[20:30, slices["task"]].zero_()
    replay_probe._event_replay_actions[
        20:30, slices["task"].start + 4
    ] = 1.0
    replay_probe._event_replay_recovery_valid[20:30] = True
    replay_probe._event_replay_actions[
        14:18, slices["skill"]
    ].zero_()
    replay_probe._event_replay_actions[
        14:18, slices["skill"].start + 1
    ] = 1.0
    replay_probe._event_replay_actions[
        18:20, slices["skill"]
    ].zero_()
    replay_probe._event_replay_actions[
        18:20, slices["skill"].start + 2
    ] = 1.0
    replay_indices = replay_probe._sample_event_replay_indices(40)
    replay_task = torch.argmax(
        replay_probe._event_replay_actions[
            replay_indices, slices["task"]
        ],
        dim=1,
    )
    replay_skill = torch.argmax(
        replay_probe._event_replay_actions[
            replay_indices, slices["skill"]
        ],
        dim=1,
    )
    replay_recovery = replay_probe._event_replay_recovery_valid[
        replay_indices
    ]
    replay_delivery = (
        replay_probe._delivery_task_mask(replay_task)
        & ~replay_recovery
    )
    replay_delivery_fraction = replay_delivery.float().mean()
    replay_recovery_fraction = replay_recovery.float().mean()
    replay_delivery_phase = replay_skill[replay_delivery].remainder(
        layout.INTERACTION_SKILL_COUNT
    )
    replay_phase_fraction = torch.stack(
        [
            (replay_delivery_phase == phase_id).float().mean()
            for phase_id in range(layout.INTERACTION_SKILL_COUNT)
        ]
    )
    if not 0.50 <= float(replay_delivery_fraction) <= 0.60:
        raise RuntimeError("Event replay did not preserve the delivery quota")
    if not 0.10 <= float(replay_recovery_fraction) <= 0.20:
        raise RuntimeError("Event replay did not preserve the recovery quota")
    expected_phase_fraction = replay_phase_fraction.new_tensor(
        (0.55, 0.30, 0.15)
    )
    if torch.max(
        torch.abs(replay_phase_fraction - expected_phase_fraction)
    ) > 0.08:
        raise RuntimeError("Delivery replay did not preserve stage quotas")

    print(f"action_shape={tuple(actions.shape)}")
    print(f"value_shape={tuple(value.shape)}")
    print(f"copied_tensor_count={len(copied)}")
    print(f"leg_equivalence_max_error={float(leg_error):.8g}")
    print(f"idle_wheel_effort_abs_max={float(idle_wheel_effort):.8g}")
    print(
        "grounded_affordance_near_far={:.6f}/{:.6f} "
        "delivery_before_after={:.6f}/{:.6f}".format(
            float(grounded_affordance[0, 0]),
            float(grounded_affordance[0, 1]),
            float(grounded_affordance[0, 2]),
            float(engaged_affordance[0, 2]),
        )
    )
    print(
        "noninteraction_arm_subgoal_abs_max="
        f"{float(route_arm_leakage):.8g} "
        "unreachable_delivery_arm_subgoal_abs_max="
        f"{float(far_delivery_arm_leakage):.8g}"
    )
    print(
        "relational_task_prior_vx="
        f"{float(relational_prior[:, 0].mean()):.6f} "
        "relational_task_authority="
        f"{float(relational_authority.mean()):.6f}"
    )
    print(
        "rear_target_forward_request="
        f"{float(turning_motion[:, 0].mean()):.6f} "
        "rear_target_turn_request="
        f"{float(turning_motion[:, 1].mean()):.6f}"
    )
    print(
        "delivery_motion_acquire_carry_unsafe="
        f"{float(delivery_motion[:, 0].mean()):.6f}/"
        f"{float(carrying_motion[:, 0].mean()):.6f}/"
        f"{float(unsafe_carrying_prior[:, 0].mean()):.6f} "
        "overspeed_brake="
        f"{float(overspeed_carrying_prior[:, 0].mean()):.6f}"
    )
    print(
        f"progress_action_abs_mean={float(progress_action.abs().mean()):.6f}"
    )
    print(
        "action_chart_span={:.6f} transit_effort={:.6f} "
        "maneuver_difference={:.6f}".format(
            float(candidate_span),
            float(transit_effort),
            float(maneuver_difference),
        )
    )
    print(
        "signed_response_chart={}".format(
            signed_response.detach().cpu().tolist()
        )
    )
    print(
        "support_gate_mean={:.6f} support_reference_mean={}".format(
            float(support_gate.mean()),
            support_reference.mean(dim=0).detach().cpu().tolist(),
        )
    )
    print(f"finite_log_prob_mean={float(log_prob.mean()):.6f}")
    print(
        "termination_sampling_max_error="
        f"{float(termination_sampling_error):.8g}"
    )
    print(
        "interaction_phase_argmax={}".format(
            torch.argmax(phase_probability, dim=-1).cpu().tolist()
        )
    )
    print(
        "event_hindsight_task={} task_valid={} skill={} priority={}".format(
            event_task_target.cpu().tolist(),
            event_task_valid.cpu().tolist(),
            relabeled_event_actions[:, slices["skill"]]
            .argmax(dim=1)
            .cpu()
            .tolist(),
            event_priority.detach().cpu().tolist(),
        )
    )
    print(
        "event_replay_delivery={:.3f} recovery={:.3f} phases={}".format(
            float(replay_delivery_fraction),
            float(replay_recovery_fraction),
            [round(float(value), 3) for value in replay_phase_fraction],
        )
    )
    if args.save_checkpoint:
        output = Path(args.save_checkpoint).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "iter": 0,
                "infos": {
                    "initialization": (
                        "ZYB-v0 leg core with learned bounded-effort wheels"
                    ),
                    "algorithm": "TACTIC-HRL",
                },
            },
            output,
        )
        print(f"saved_checkpoint={output}")
    print("TACTIC actor smoke test passed")


if __name__ == "__main__":
    main()
