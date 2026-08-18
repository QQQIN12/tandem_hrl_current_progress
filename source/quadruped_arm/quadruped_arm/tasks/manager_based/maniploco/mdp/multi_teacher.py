"""Multi-teacher support for the lower-body student policy.

The environment keeps the physical action interface unchanged.  The teacher
bank only supplies a privileged imitation target and a safety gate:

* ZYB-v0: the archived learned support/stability teacher;
* conservative ZYB: a lower-amplitude transient teacher;
* neutral stance: a recovery teacher that asks the student to return to the
  default leg posture when tilt or height margin deteriorates.

The wheel action term remains the command-conditioned differential-drive
teacher.  Its four policy outputs are residuals, so the raw wheel target for
imitation is zero; this prevents PPO exploration from fighting the bounded
wheel feed-forward controller.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _FrozenZybTeacher(nn.Module):
    """Small loader for the 876 -> 16 archived ZYB actor."""

    def __init__(self, checkpoint: str, device: torch.device):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        required = (
            "actor.0.weight", "actor.0.bias",
            "actor.2.weight", "actor.2.bias",
            "actor.4.weight", "actor.4.bias",
            "actor_obs_normalizer._mean", "actor_obs_normalizer._std",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"ZYB teacher checkpoint is missing: {missing}")
        self.net = nn.Sequential(
            nn.Linear(876, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 16),
        ).to(device)
        self.net.load_state_dict(
            {
                "0.weight": state["actor.0.weight"],
                "0.bias": state["actor.0.bias"],
                "2.weight": state["actor.2.weight"],
                "2.bias": state["actor.2.bias"],
                "4.weight": state["actor.4.weight"],
                "4.bias": state["actor.4.bias"],
            }
        )
        self.register_buffer(
            "obs_mean",
            state["actor_obs_normalizer._mean"].reshape(-1).to(device),
        )
        self.register_buffer(
            "obs_std",
            state["actor_obs_normalizer._std"].reshape(-1).to(device).clamp_min(1.0e-4),
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.obs_mean) / self.obs_std
        # The archived ZYB actor is an unconstrained Gaussian mean.  Preserve
        # its action scale here so the imitation target is compatible with the
        # original 16-action executor; the wheel action term still clamps its
        # physical residual independently.
        return self.net(normalized)


class MultiTeacherVecEnv:
    """Vec-env proxy that publishes a gated teacher ensemble before stepping."""

    def __init__(
        self,
        env,
        teacher_checkpoint: str | None = None,
        teacher_sigma: float = 0.20,
        teacher_blend_start: float = 0.85,
        teacher_blend_end: float = 0.20,
        teacher_blend_steps: int = 100_000,
    ):
        self.env = env
        self.teacher_sigma = float(teacher_sigma)
        self.teacher_blend_start = float(teacher_blend_start)
        self.teacher_blend_end = float(teacher_blend_end)
        self.teacher_blend_steps = max(int(teacher_blend_steps), 1)
        self._total_steps = 0
        self._last_obs: torch.Tensor | None = None
        self._last_command: torch.Tensor | None = None
        self.teacher = None
        if teacher_checkpoint:
            checkpoint = Path(teacher_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")
            self.teacher = _FrozenZybTeacher(str(checkpoint), self.device)

        base = getattr(env, "unwrapped", None)
        if base is None:
            base = getattr(env, "env", env)
        self.base_env = base
        self.base_env.teacher_ensemble_candidates = torch.zeros(
            self.num_envs, 3, self.num_actions, device=self.device
        )
        self.base_env.teacher_ensemble_weights = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.base_env.teacher_ensemble_action_target = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        self.base_env.teacher_ensemble_stability = torch.zeros(
            self.num_envs, device=self.device
        )
        self.base_env.teacher_action_blend = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def device(self):
        return self.env.device

    @property
    def episode_length_buf(self):
        """Forward RSL-RL's random episode-age buffer to the real vec env."""

        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        """Do not shadow the underlying wrapper when RSL-RL assigns ages."""

        self.env.episode_length_buf = value

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _publish_teacher(self) -> None:
        if self._last_obs is None:
            return
        obs = self._last_obs
        if self.teacher is None:
            zyb = torch.zeros(obs.shape[0], self.num_actions, device=obs.device, dtype=obs.dtype)
        else:
            with torch.inference_mode():
                zyb = self.teacher(obs).to(dtype=obs.dtype)
        if zyb.shape[-1] != self.num_actions:
            raise ValueError(f"Teacher action width {zyb.shape[-1]} != env width {self.num_actions}")

        robot = self.base_env.scene["robot"]
        tilt = torch.asin(
            torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1).clamp(0.0, 1.0)
        )
        height = robot.data.root_pos_w[:, 2]
        stability = (
            torch.sigmoid((0.16 - tilt) / 0.04)
            * torch.sigmoid((height - 0.40) / 0.03)
        ).clamp(0.0, 1.0)

        command = self.base_env.command_manager.get_command("locomotion")
        if self._last_command is None or self._last_command.shape != command.shape:
            transient = torch.zeros(command.shape[0], device=command.device, dtype=command.dtype)
        else:
            transient = (command - self._last_command).abs().sum(dim=1).clamp(0.0, 1.0)
        self._last_command = command.detach().clone()
        transient = torch.sigmoid((transient - 0.03) / 0.02)

        # The raw wheel outputs are residuals around the safe wheel teacher.
        # Keep all candidates at zero in those four dimensions.
        zyb = zyb.clone()
        zyb[:, 12:16] = 0.0
        conservative = zyb.clone()
        conservative[:, :12] *= 0.50
        neutral = torch.zeros_like(zyb)
        candidates = torch.stack((zyb, conservative, neutral), dim=1)

        # Stable cruising prefers the learned support teacher.  During a
        # command transition or a deteriorating support margin, the mixture
        # gives more probability to the conservative and neutral teachers.
        weight_zyb = 0.65 * stability * (1.0 - 0.25 * transient)
        weight_conservative = 0.20 + 0.20 * transient + 0.10 * (1.0 - stability)
        weight_neutral = (1.0 - weight_zyb - weight_conservative).clamp_min(0.05)
        weights = torch.stack((weight_zyb, weight_conservative, weight_neutral), dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        self.base_env.teacher_ensemble_candidates = candidates.detach()
        self.base_env.teacher_ensemble_weights = weights.detach()
        self.base_env.teacher_ensemble_action_target = (
            (weights.unsqueeze(-1) * candidates).sum(dim=1).detach()
        )
        self.base_env.teacher_ensemble_stability = stability.detach()

    def get_observations(self):
        observations = self.env.get_observations()
        if isinstance(observations, dict):
            policy_observations = observations["policy"]
        elif hasattr(observations, "keys") and "policy" in observations.keys():
            policy_observations = observations["policy"]
        else:
            policy_observations = observations
        self._last_obs = policy_observations.detach()
        return observations

    def step(self, actions):
        self._publish_teacher()
        teacher_action = self.base_env.teacher_ensemble_action_target
        stability = self.base_env.teacher_ensemble_stability
        progress = min(float(self._total_steps) / float(self.teacher_blend_steps), 1.0)
        scheduled = self.teacher_blend_start + progress * (
            self.teacher_blend_end - self.teacher_blend_start
        )
        # If the support margin is deteriorating, temporarily increase the
        # teacher shield.  This is the safety part of teacher-student training;
        # it prevents one bad PPO update from destroying the stance policy.
        alpha = torch.full_like(stability, scheduled)
        alpha = torch.maximum(alpha, (1.0 - stability).clamp(0.0, 1.0) * 0.85)
        alpha = alpha.clamp(0.0, 1.0)
        self.base_env.teacher_action_blend = alpha.detach()
        blended = actions + alpha.unsqueeze(-1) * (teacher_action - actions)
        self._total_steps += self.num_envs
        return self.env.step(blended)

    def reset(self):
        self._last_obs = None
        self._last_command = None
        return self.env.reset()

    def close(self):
        return self.env.close()
