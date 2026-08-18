"""Command distributions used to train reusable locomotion Skills."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


class SkillVelocityCommand(UniformVelocityCommand):
    """Sample straight, yaw-only, arc, and stationary command modes."""

    cfg: SkillVelocityCommandCfg

    @staticmethod
    def _minimum_magnitude(values: torch.Tensor, minimum: float) -> torch.Tensor:
        signs = torch.where(values >= 0.0, 1.0, -1.0)
        return signs * values.abs().clamp_min(minimum)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        count = len(env_ids)
        selector = torch.rand(count, device=self.device)
        standing = selector < self.cfg.stationary_fraction
        yaw_only = (
            selector >= self.cfg.stationary_fraction
        ) & (
            selector
            < self.cfg.stationary_fraction + self.cfg.yaw_only_fraction
        )
        straight = (
            selector
            >= self.cfg.stationary_fraction + self.cfg.yaw_only_fraction
        ) & (
            selector
            < self.cfg.stationary_fraction
            + self.cfg.yaw_only_fraction
            + self.cfg.straight_fraction
        )
        arc = ~(standing | yaw_only | straight)

        selected = self.vel_command_b[env_ids]
        selected[standing] = 0.0
        selected[yaw_only, 0:2] = 0.0
        selected[yaw_only, 2] = self._minimum_magnitude(
            selected[yaw_only, 2], self.cfg.minimum_yaw_speed
        )
        selected[straight, 0] = self._minimum_magnitude(
            selected[straight, 0], self.cfg.minimum_linear_speed
        )
        selected[straight, 1:3] = 0.0
        selected[arc, 0] = self._minimum_magnitude(
            selected[arc, 0], self.cfg.minimum_linear_speed
        )
        selected[arc, 2] = self._minimum_magnitude(
            selected[arc, 2], self.cfg.minimum_yaw_speed
        )
        self.vel_command_b[env_ids] = selected
        self.is_standing_env[env_ids] = standing


@configclass
class SkillVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SkillVelocityCommand
    stationary_fraction: float = 0.10
    yaw_only_fraction: float = 0.30
    straight_fraction: float = 0.30
    minimum_linear_speed: float = 0.16
    minimum_yaw_speed: float = 0.22
