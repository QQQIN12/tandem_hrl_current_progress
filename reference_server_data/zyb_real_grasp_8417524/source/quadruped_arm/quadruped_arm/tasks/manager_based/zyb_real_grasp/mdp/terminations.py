"""Physical failure and task-success conditions."""

from __future__ import annotations

import torch

from .rewards import placement_quality


def excessive_tilt(env, limit: float = 0.65) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    return torch.linalg.vector_norm(gravity[:, :2], dim=1) > torch.sin(
        gravity.new_tensor(limit)
    )


def low_base(env, minimum_height: float = 0.25) -> torch.Tensor:
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height < minimum_height


def placed_object(env) -> torch.Tensor:
    return placement_quality(env) > 0.5
