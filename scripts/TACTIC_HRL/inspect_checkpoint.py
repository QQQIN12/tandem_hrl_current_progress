"""Inspect a TANDEM-HRL checkpoint and its learned option factors."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = payload.get("model_state_dict", payload)
    std = state.get("std")
    if std is None:
        std = state["log_std"].exp()
    print(f"shape={tuple(std.shape)}")
    print(f"physical_0_15={std[:16].tolist()}")
    print(f"gripper={float(std[16])}")
    print(f"hierarchy_mean={float(std[17:].mean())}")
    print(f"all_mean={float(std.mean())}")
    for key in (
        "actor.wheel_skill_gate_logit",
        "actor.motion_wheel_basis",
        "actor.motion_kinematic_gain",
        "actor.wheel_breakaway_action",
        "actor.support_gate_logit",
    ):
        value = state.get(key)
        if value is not None:
            print(f"{key}={value.detach().cpu().tolist()}")
    embedding = state.get("actor.skill_embedding.weight")
    effect_weight = state.get("actor.skill_effect_head.weight")
    if embedding is not None and effect_weight is not None:
        effect = 0.35 * torch.tanh(
            F.linear(embedding, effect_weight)
        )
        print("skill_effects=")
        for index, row in enumerate(effect):
            values = ", ".join(f"{float(value):+.4f}" for value in row)
            print(f"  {index:02d}: [{values}]")
        pair_distance = torch.cdist(effect, effect)
        off_diagonal = pair_distance[
            ~torch.eye(
                pair_distance.shape[0], dtype=torch.bool
            )
        ]
        print(
            "skill_effect_pair_distance="
            f"min:{float(off_diagonal.min()):.6f}, "
            f"mean:{float(off_diagonal.mean()):.6f}, "
            f"max:{float(off_diagonal.max()):.6f}"
        )
        return

    motion_embedding = state.get("actor.motion_skill_embedding.weight")
    interaction_embedding = state.get(
        "actor.interaction_skill_embedding.weight"
    )
    motion_effect_weight = state.get(
        "actor.motion_skill_effect_head.weight"
    )
    interaction_effect_weight = state.get(
        "actor.interaction_skill_effect_head.weight"
    )
    if any(
        value is None
        for value in (
            motion_embedding,
            interaction_embedding,
            motion_effect_weight,
            interaction_effect_weight,
        )
    ):
        return
    motion_effect = F.linear(motion_embedding, motion_effect_weight)
    interaction_effect = F.linear(
        interaction_embedding, interaction_effect_weight
    )
    composite = 0.35 * torch.tanh(
        0.5
        * (
            motion_effect[:, None, :]
            + interaction_effect[None, :, :]
        )
    ).reshape(-1, motion_effect.shape[1])
    print("factorized_skill_effects_at_equal_gate=")
    for index, row in enumerate(composite):
        motion_id = index // interaction_embedding.shape[0]
        interaction_id = index % interaction_embedding.shape[0]
        values = ", ".join(f"{float(value):+.4f}" for value in row)
        print(
            f"  m{motion_id}/i{interaction_id}: [{values}]"
        )
    pair_distance = torch.cdist(composite, composite)
    off_diagonal = pair_distance[
        ~torch.eye(pair_distance.shape[0], dtype=torch.bool)
    ]
    print(
        "factorized_effect_pair_distance="
        f"min:{float(off_diagonal.min()):.6f}, "
        f"mean:{float(off_diagonal.mean()):.6f}, "
        f"max:{float(off_diagonal.max()):.6f}"
    )


if __name__ == "__main__":
    main()
