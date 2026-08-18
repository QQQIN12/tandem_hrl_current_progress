"""Strict migration of the original ZYB-v0 physical policy."""

from __future__ import annotations

import torch


def load_zyb_baseline_physical(policy, checkpoint_path: str) -> list[str]:
    """Load the 876-D locomotion core without importing any HRL heads."""

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = payload.get("model_state_dict", payload)
    target = policy.state_dict()
    copied: list[str] = []

    backbone_mapping = {
        "actor.physical_backbone.0.weight": "actor.0.weight",
        "actor.physical_backbone.0.bias": "actor.0.bias",
        "actor.physical_backbone.2.weight": "actor.2.weight",
        "actor.physical_backbone.2.bias": "actor.2.bias",
    }
    for target_key, source_key in backbone_mapping.items():
        value = source.get(source_key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"ZYB-v0 checkpoint is missing {source_key}")
        if target_key not in target or target[target_key].shape != value.shape:
            target_shape = (
                tuple(target[target_key].shape)
                if target_key in target
                else "missing"
            )
            raise ValueError(
                f"Physical migration mismatch for {source_key}: "
                f"{tuple(value.shape)} -> {target_shape}"
            )
        target[target_key] = value.to(dtype=target[target_key].dtype)
        copied.append(f"{source_key}->{target_key}")

    source_weight = source.get("actor.4.weight")
    source_bias = source.get("actor.4.bias")
    target_weight = target["actor.physical_head.weight"]
    target_bias = target["actor.physical_head.bias"]
    if (
        not isinstance(source_weight, torch.Tensor)
        or not isinstance(source_bias, torch.Tensor)
        or source_weight.shape != (16, target_weight.shape[1])
        or source_bias.shape != (16,)
        or target_weight.shape[0] != 17
        or target_bias.shape != (17,)
    ):
        raise ValueError("ZYB-v0 physical head is incompatible with TACTIC")
    target_weight[:16] = source_weight.to(dtype=target_weight.dtype)
    target_bias[:16] = source_bias.to(dtype=target_bias.dtype)
    target["actor.physical_head.weight"] = target_weight
    target["actor.physical_head.bias"] = target_bias
    copied.extend(
        (
            "actor.4.weight->actor.physical_head.weight[:16]",
            "actor.4.bias->actor.physical_head.bias[:16]",
        )
    )

    for suffix, neutral in (
        ("_mean", 0.0),
        ("_var", 1.0),
        ("_std", 1.0),
    ):
        key = f"actor_obs_normalizer.{suffix}"
        source_value = source.get(key)
        target_value = target.get(key)
        if (
            not isinstance(source_value, torch.Tensor)
            or not isinstance(target_value, torch.Tensor)
            or source_value.ndim != 2
            or target_value.ndim != 2
            or source_value.shape[0] != target_value.shape[0]
            or source_value.shape[1] != 876
            or target_value.shape[1] < 876
        ):
            raise ValueError(f"ZYB-v0 normalizer is incompatible at {key}")
        merged = torch.full_like(target_value, neutral)
        merged[:, :876] = source_value.to(dtype=merged.dtype)
        target[key] = merged
        copied.append(f"{key}[:876]")

    count_key = "actor_obs_normalizer.count"
    source_count = source.get(count_key)
    if isinstance(source_count, torch.Tensor):
        target[count_key] = source_count.to(dtype=target[count_key].dtype)
        copied.append(count_key)

    policy.load_state_dict(target, strict=True)
    loaded = policy.state_dict()
    for target_key, source_key in backbone_mapping.items():
        expected = source[source_key].to(
            device=loaded[target_key].device, dtype=loaded[target_key].dtype
        )
        if not torch.equal(loaded[target_key], expected):
            raise RuntimeError(
                f"Migration verification failed for {source_key}"
            )
    if not torch.equal(
        loaded["actor.physical_head.weight"][:16],
        source_weight.to(
            device=loaded["actor.physical_head.weight"].device,
            dtype=loaded["actor.physical_head.weight"].dtype,
        ),
    ):
        raise RuntimeError("Migration verification failed for physical head")

    forbidden_fragments = (
        "task",
        "phase",
        "skill",
        "object",
        "option",
        "mission",
        "hierarchy",
    )
    if any(
        fragment in item
        for item in copied
        for fragment in forbidden_fragments
    ):
        raise RuntimeError("An HRL-specific tensor entered baseline migration")
    return copied
