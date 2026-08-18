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

PREDICTION_MARKERS = (
    "outcome",
    "effect",
    "successor",
    "control_prediction",
    "motion_execution",
    "constraint_multiplier",
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
    if name.startswith(PHYSICAL_PREFIXES):
        return "physical_executor"
    if any(marker in name for marker in PREDICTION_MARKERS):
        return "prediction_constraint"
    if any(marker in name for marker in TASK_MARKERS):
        return "task_decomposition"
    if any(marker in name for marker in SKILL_MARKERS):
        return "skill_decomposition"
    return "shared_or_critic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    common = sorted(set(before) & set(after))
    if not common:
        raise RuntimeError("The checkpoints have no common tensors")

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
        delta = after[name] - before[name]
        group = totals[_group(name)]
        group["parameters"] += delta.numel()
        group["changed_parameters"] += int(
            torch.count_nonzero(delta).item()
        )
        group["delta_sq"] += float(torch.sum(delta.square()).item())
        group["reference_sq"] += float(
            torch.sum(before[name].square()).item()
        )
        group["max_abs"] = max(
            group["max_abs"],
            float(delta.abs().max().item()),
        )

    for name in (
        "task_decomposition",
        "skill_decomposition",
        "prediction_constraint",
        "physical_executor",
        "shared_or_critic",
    ):
        values = totals[name]
        delta_norm = math.sqrt(values["delta_sq"])
        reference_norm = math.sqrt(values["reference_sq"])
        changed_fraction = values["changed_parameters"] / max(
            values["parameters"],
            1,
        )
        print(
            "{} parameters={} changed_fraction={:.6f} "
            "delta_l2={:.8f} relative_l2={:.8f} max_abs={:.8f}".format(
                name,
                values["parameters"],
                changed_fraction,
                delta_norm,
                delta_norm / max(reference_norm, 1.0e-12),
                values["max_abs"],
            )
        )


if __name__ == "__main__":
    main()
