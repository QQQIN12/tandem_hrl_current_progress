"""Report parameter changes between two TANDEM-HRL checkpoints."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch


PHYSICAL_PREFIXES = (
    "actor.physical_backbone.",
    "actor.physical_head.",
    "actor.physical_conditioner.",
    "actor.film_head.",
    "actor.physical_residual_head.",
    "actor.support_skill_encoder.",
    "actor.support_reference_head.",
    "actor.support_gate_head.",
    "actor.support_residual_head.",
    "actor.wheel_residual_encoder.",
    "actor.wheel_residual_head.",
    "actor.wheel_skill_gate_head.",
    "actor.gripper_head.",
)
PHYSICAL_PARAMETERS = {
    "actor.motion_support_basis",
    "actor.support_gate_logit",
    "actor.wheel_skill_gate_logit",
    "actor.embodiment_motion_basis",
    "actor.motion_action_capacity",
    "actor.motion_kinematic_gain",
    "actor.wheel_breakaway_action",
    "actor.interaction_gripper_basis",
}
RECOVERY_ADAPTER_PREFIXES = (
    "actor.recovery_adapter_encoder.",
    "actor.recovery_task_adapter_head.",
    "actor.recovery_motion_adapter_head.",
    "actor.recovery_interaction_adapter_head.",
)
RECOVERY_OUTPUT_PREFIXES = (
    "actor.recovery_task_adapter_head.",
    "actor.recovery_motion_adapter_head.",
    "actor.recovery_interaction_adapter_head.",
)
OPTIONAL_NEUTRAL_SCHEMA_PREFIXES = (
    "actor.skill_survival_head.",
    "actor.payload_survival_updates",
)
PREDICTION_MARKERS = (
    "outcome",
    "effect",
    "successor",
    "control_prediction",
    "motion_execution",
    "constraint_multiplier",
    "embodiment_response",
)
TASK_MARKERS = (
    "task_",
    "task.",
    "mission_",
    "slot_",
    "graph_",
    "morphology_",
    "global_encoder",
)
SKILL_MARKERS = (
    "skill_",
    "skill.",
    "motion_skill",
    "interaction_skill",
)


def _load(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    return {
        name: value.detach().float()
        for name, value in state.items()
        if torch.is_tensor(value) and value.is_floating_point()
    }


def _group(name: str) -> str:
    if name.startswith(RECOVERY_ADAPTER_PREFIXES):
        return "recovery_adapter"
    if name.startswith(PHYSICAL_PREFIXES) or name in PHYSICAL_PARAMETERS:
        return "physical_executor"
    if any(marker in name for marker in PREDICTION_MARKERS):
        return "prediction_constraint"
    if any(marker in name for marker in TASK_MARKERS):
        return "task_decomposition"
    if any(marker in name for marker in SKILL_MARKERS):
        return "skill_decomposition"
    if name.startswith("actor."):
        return "shared_actor"
    if name.startswith("critic."):
        return "critic"
    return "shared_state"


def _accumulate(
    totals: dict,
    group_name: str,
    before: torch.Tensor,
    after: torch.Tensor,
) -> None:
    delta = after - before
    group = totals[group_name]
    group["parameters"] += delta.numel()
    group["changed_parameters"] += int(torch.count_nonzero(delta).item())
    group["delta_sq"] += float(torch.sum(delta.square()).item())
    group["reference_sq"] += float(torch.sum(before.square()).item())
    group["max_abs"] = max(
        group["max_abs"],
        float(delta.abs().max().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument(
        "--require_frozen_physical",
        action="store_true",
        help="Exit with an error if an actuator-decoder tensor changed.",
    )
    parser.add_argument(
        "--require_recovery_only_actor",
        action="store_true",
        help="Exit if an actor tensor outside the recovery adapter changed.",
    )
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    common = sorted(set(before) & set(after))
    if not common:
        raise RuntimeError("The checkpoints have no common tensors")
    before_only = sorted(set(before) - set(after))
    after_only = sorted(set(after) - set(before))

    totals = defaultdict(
        lambda: {
            "parameters": 0,
            "changed_parameters": 0,
            "delta_sq": 0.0,
            "reference_sq": 0.0,
            "max_abs": 0.0,
        }
    )
    for name in common:
        if before[name].shape != after[name].shape:
            continue
        if name in ("std", "log_std") and before[name].numel() >= 17:
            _accumulate(
                totals,
                "physical_exploration",
                before[name][:17],
                after[name][:17],
            )
            _accumulate(
                totals,
                "hierarchy_exploration",
                before[name][17:],
                after[name][17:],
            )
            continue
        _accumulate(totals, _group(name), before[name], after[name])
    for name in after_only:
        if name.startswith(RECOVERY_ADAPTER_PREFIXES):
            _accumulate(
                totals,
                "recovery_adapter",
                torch.zeros_like(after[name]),
                after[name],
            )

    ordered_groups = (
        "recovery_adapter",
        "task_decomposition",
        "skill_decomposition",
        "prediction_constraint",
        "physical_executor",
        "physical_exploration",
        "hierarchy_exploration",
        "shared_actor",
        "critic",
        "shared_state",
    )
    for group_name in ordered_groups:
        values = totals[group_name]
        delta_norm = math.sqrt(values["delta_sq"])
        reference_norm = math.sqrt(values["reference_sq"])
        relative = delta_norm / max(reference_norm, 1.0e-12)
        changed_fraction = values["changed_parameters"] / max(
            values["parameters"], 1
        )
        print(
            "{} parameters={} changed_fraction={:.6f} "
            "delta_l2={:.8f} relative_l2={:.8f} max_abs={:.8f}".format(
                group_name,
                values["parameters"],
                changed_fraction,
                delta_norm,
                relative,
                values["max_abs"],
            )
        )
    print(
        f"checkpoint_schema added={len(after_only)} removed={len(before_only)}"
    )

    if args.require_frozen_physical:
        changed = (
            totals["physical_executor"]["changed_parameters"]
            + totals["physical_exploration"]["changed_parameters"]
        )
        if changed:
            raise RuntimeError(
                f"Physical executor changed in {changed} scalar entries"
            )
        print("physical_freeze_check=passed")
    if args.require_recovery_only_actor:
        unexpected_schema_changes = [
            name
            for name in (*before_only, *after_only)
            if name.startswith("actor.")
            and not name.startswith(RECOVERY_ADAPTER_PREFIXES)
            and not name.startswith(OPTIONAL_NEUTRAL_SCHEMA_PREFIXES)
        ]
        if unexpected_schema_changes:
            raise RuntimeError(
                "Unexpected non-adapter actor schema changes: "
                + ", ".join(unexpected_schema_changes)
            )
        nonneutral_optional = [
            name
            for name in after_only
            if name.startswith(OPTIONAL_NEUTRAL_SCHEMA_PREFIXES)
            and torch.count_nonzero(after[name]).item() != 0
        ]
        if nonneutral_optional:
            raise RuntimeError(
                "Optional upper-stage tensors changed during recovery: "
                + ", ".join(nonneutral_optional)
            )
        forbidden_groups = (
            "task_decomposition",
            "skill_decomposition",
            "prediction_constraint",
            "physical_executor",
            "physical_exploration",
            "hierarchy_exploration",
            "shared_actor",
        )
        changed = sum(
            totals[group_name]["changed_parameters"]
            for group_name in forbidden_groups
        )
        if changed:
            raise RuntimeError(
                f"Non-adapter actor changed in {changed} scalar entries"
            )
        if not totals["recovery_adapter"]["changed_parameters"]:
            raise RuntimeError("Recovery adapter did not update")
        learned_output_entries = 0
        for name in set(before) | set(after):
            if not name.startswith(RECOVERY_OUTPUT_PREFIXES):
                continue
            before_value = before.get(name)
            after_value = after.get(name)
            if after_value is None:
                continue
            if before_value is None:
                before_value = torch.zeros_like(after_value)
            if before_value.shape == after_value.shape:
                learned_output_entries += int(
                    torch.count_nonzero(after_value - before_value).item()
                )
        if learned_output_entries == 0:
            raise RuntimeError("Recovery adapter output heads stayed neutral")
        print("recovery_only_actor_check=passed")


if __name__ == "__main__":
    main()
