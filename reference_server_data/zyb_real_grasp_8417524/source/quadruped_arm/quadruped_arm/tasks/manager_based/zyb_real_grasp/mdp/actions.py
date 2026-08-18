"""Task-side gripper action for physical object transport."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


def _contact_load(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.reshape(forces.shape[0], -1, 3), dim=-1).amax(dim=1)


class ContactHoldGripperAction(ActionTerm):
    """Symmetric aperture command with contact-based payload retention.

    Action 0 sets the half aperture in ``[-1, 1]``.  Action 1 is an explicit
    release gate.  Bilateral contact latches a slightly compressed target until
    release is requested.  The latch uses signals available from finger force
    sensing and has no access to object pose or task stage.
    """

    cfg: "ContactHoldGripperActionCfg"

    def __init__(self, cfg: "ContactHoldGripperActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != 2:
            raise RuntimeError("contact-hold gripper requires joint7 and joint8")
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target = torch.zeros(self.num_envs, 2, device=self.device)
        self._hold_target = torch.zeros_like(self._target)
        self._contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 2:
            raise ValueError(f"expected two gripper actions, got {actions.shape[-1]}")
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))

        half_gap = self.cfg.minimum_half_aperture + 0.5 * (self._raw_actions[:, 0] + 1.0) * (
            self.cfg.maximum_half_aperture - self.cfg.minimum_half_aperture
        )
        requested = torch.stack((half_gap, -half_gap), dim=1)
        release = self._raw_actions[:, 1] > self.cfg.release_threshold

        loads = torch.stack(
            (
                _contact_load(self._env, self.cfg.left_contact_sensor),
                _contact_load(self._env, self.cfg.right_contact_sensor),
            ),
            dim=1,
        )
        bilateral = (loads > self.cfg.contact_force_threshold).all(dim=1)
        self._contact_streak = torch.where(
            bilateral, self._contact_streak + 1, torch.zeros_like(self._contact_streak)
        )
        entering = (
            (self._contact_streak >= self.cfg.contact_latch_steps)
            & ~self._locked
            & ~release
        )
        if entering.any():
            current = self._asset.data.joint_pos[entering][:, self._joint_ids]
            compression = current.new_tensor((-self.cfg.secure_compression, self.cfg.secure_compression))
            self._hold_target[entering] = current + compression

        self._locked[release] = False
        self._locked |= entering
        holding = self._locked & ~release

        # If one pad unloads while carrying, tighten slowly instead of opening.
        unload = holding & ~bilateral
        if unload.any():
            tighten = self._hold_target.new_tensor(
                (-self.cfg.retention_tighten_step, self.cfg.retention_tighten_step)
            )
            self._hold_target[unload] += tighten

        self._target.copy_(requested)
        self._target[holding] = self._hold_target[holding]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._target = torch.maximum(torch.minimum(self._target, limits[..., 1]), limits[..., 0])
        self._processed_actions.copy_(self._target)
        self._env.zyb_real_grasp_gripper_diagnostics = {
            "finger_loads": loads.detach(),
            "bilateral_contact": bilateral.detach(),
            "contact_streak": self._contact_streak.detach().clone(),
            "hold_locked": self._locked.detach().clone(),
            "joint_target": self._target.detach().clone(),
            "release_requested": release.detach(),
        }

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._target[ids, 0] = self.cfg.maximum_half_aperture
        self._target[ids, 1] = -self.cfg.maximum_half_aperture
        self._hold_target[ids] = 0.0
        self._contact_streak[ids] = 0
        self._locked[ids] = False


@configclass
class ContactHoldGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ContactHoldGripperAction
    joint_names: tuple[str, str] = ("joint7", "joint8")
    left_contact_sensor: str = "left_finger_object_contact"
    right_contact_sensor: str = "right_finger_object_contact"
    minimum_half_aperture: float = 0.002
    maximum_half_aperture: float = 0.034
    contact_force_threshold: float = 0.20
    contact_latch_steps: int = 3
    secure_compression: float = 0.001
    retention_tighten_step: float = 0.0001
    release_threshold: float = 0.50
