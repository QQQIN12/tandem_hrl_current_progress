"""Training action term for learned B2W locomotion under support WBC."""

from __future__ import annotations

from collections.abc import Sequence
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..controllers import PayloadAwareSupportWBC


class SupportWBCAction(ActionTerm):
    """Apply learned leg residuals and wheel commands through support WBC."""

    cfg: SupportWBCActionCfg

    def __init__(self, cfg: SupportWBCActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.leg_joint_names, preserve_order=True
        )
        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            cfg.wheel_joint_names, preserve_order=True
        )
        if len(self._joint_ids) != 12 or len(self._wheel_ids) != 4:
            raise RuntimeError("B2W action contract requires 12 legs and 4 wheels")
        if len(cfg.leg_action_scales) != 12:
            raise ValueError("leg_action_scales must have 12 entries")

        self._scale = torch.tensor(
            cfg.leg_action_scales, device=self.device
        ).view(1, 12).repeat(self.num_envs, 1)
        self._offset = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()
        # Keep the original ZYB-v0 12-leg + 4-wheel action contract in the
        # first 16 channels.  The final four channels are a separate learned
        # support-allocation head for the WBC; they must not replace wheel
        # commands or change their ordering.
        self._raw_actions = torch.zeros(
            self.num_envs, 20, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wbc: PayloadAwareSupportWBC | None = None
        self._reference_pending = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    @property
    def action_dim(self) -> int:
        return 20

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _ensure_wbc(self) -> PayloadAwareSupportWBC:
        if self._wbc is None:
            self._wbc = PayloadAwareSupportWBC(
                self._env,
                support_gain=self.cfg.support_gain,
                wheel_slew_per_step=self.cfg.wheel_slew_per_step,
                max_policy_joint_residual=(
                    self.cfg.max_policy_joint_residual
                ),
                max_turn_xy_relaxation=self.cfg.max_turn_xy_relaxation,
                support_xy_tracking_scale=(
                    self.cfg.support_xy_tracking_scale
                ),
                max_learned_unload_shift_m=(
                    self.cfg.max_learned_unload_shift_m
                ),
                max_learned_support_relaxation=(
                    self.cfg.max_learned_support_relaxation
                ),
                max_learned_unload_joint_correction=(
                    self.cfg.max_learned_unload_joint_correction
                ),
            )
        return self._wbc

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 20:
            raise ValueError(
                f"Expected 20 Skill actions, got {actions.shape[-1]}"
            )
        # Match ZYB-v0's action interface: leg residuals are normalized, but
        # wheel velocity actions retain the wrapper's [-100, 100] range and
        # are converted to rad/s only by wheel_velocity_scale below.
        self._raw_actions[:, :12] = actions[:, :12].clamp(-1.0, 1.0)
        self._raw_actions[:, 12:16] = actions[:, 12:16].clamp(-100.0, 100.0)
        self._raw_actions[:, 16:20] = actions[:, 16:20].clamp(-1.0, 1.0)
        wbc = self._ensure_wbc()
        pending_ids = self._reference_pending.nonzero(
            as_tuple=False
        ).flatten()
        if pending_ids.numel() > 0:
            wbc.reset_reference(pending_ids)
            self._reference_pending[pending_ids] = False
        wheel_coordinates = self._raw_actions[:, 12:16]
        support_coordinates = self._raw_actions[:, 16:20]
        if self.cfg.wheel_coordinate_mode == "independent":
            wheel_target = wheel_coordinates.clone()
        elif self.cfg.wheel_coordinate_mode == "structured":
            drive = wheel_coordinates[:, 0]
            turn = wheel_coordinates[:, 1]
            front_rear = 0.25 * wheel_coordinates[:, 2]
            diagonal = 0.25 * wheel_coordinates[:, 3]
            wheel_target = torch.stack(
                (
                    drive - turn + front_rear + diagonal,
                    drive + turn + front_rear - diagonal,
                    drive - turn - front_rear - diagonal,
                    drive + turn - front_rear + diagonal,
                ),
                dim=1,
            ).clamp(-1.0, 1.0)
        elif self.cfg.wheel_coordinate_mode == "structured_support":
            # Compatibility/debug mode: the first two wheel channels encode
            # drive and turn, while support remains a separate four-channel
            # head at the end of the action vector.
            drive = wheel_coordinates[:, 0]
            turn = wheel_coordinates[:, 1]
            wheel_target = torch.stack(
                (
                    drive - turn,
                    drive + turn,
                    drive - turn,
                    drive + turn,
                ),
                dim=1,
            ).clamp(-1.0, 1.0)
        elif self.cfg.wheel_coordinate_mode == "independent_support":
            # This is the mainline mode.  It preserves the four independent
            # wheel references used by ZYB-v0 and appends learned support
            # allocation without reparameterising the locomotion command.
            wheel_target = wheel_coordinates.clone()
        else:
            raise ValueError(
                f"Unknown wheel coordinate mode: {self.cfg.wheel_coordinate_mode}"
            )
        if self.cfg.wheel_coordinate_mode in (
            "structured",
            "structured_support",
        ):
            wheel_target = wheel_target * self.cfg.wheel_policy_limit
        self._env.tandem_wheel_coordinates = wheel_coordinates.detach().clone()
        self._env.tandem_wheel_target = wheel_target.detach().clone()
        # ZYB-v0's leg action is position control with per-joint scales
        # (0.4/0.45/0.45).  Convert the normalized policy output once here;
        # the WBC then adds only its bounded support correction in radians.
        leg_residual = self._raw_actions[:, :12] * self._scale
        processed, diagnostics = wbc.compute(
            wheel_target,
            policy_leg_residual=leg_residual,
            learned_support_coordinates=support_coordinates,
        )
        processed[:, 16:20] = support_coordinates
        self._processed_actions[:] = processed
        self._env.tandem_wbc_diagnostics = diagnostics

    def apply_actions(self) -> None:
        leg_target = (
            self._offset
            + self._processed_actions[:, :12] * self._scale
        )
        wheel_target = (
            self._processed_actions[:, 12:16]
            * self.cfg.wheel_velocity_scale
        )
        self._asset.set_joint_position_target(
            leg_target, joint_ids=self._joint_ids
        )
        self._asset.set_joint_velocity_target(
            wheel_target, joint_ids=self._wheel_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        if env_ids is None:
            ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reference_pending[ids] = True

@configclass
class SupportWBCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SupportWBCAction
    leg_joint_names: list[str] = None  # type: ignore[assignment]
    wheel_joint_names: list[str] = None  # type: ignore[assignment]
    leg_action_scales: tuple[float, ...] = ()
    wheel_velocity_scale: float = 0.1
    wheel_policy_limit: float = 100.0
    wheel_slew_per_step: float = 100.0
    support_gain: float = 0.55
    # Zero means preserve the per-joint ZYB-v0 action scale and let the
    # articulation's soft joint limits perform the final projection.
    max_policy_joint_residual: float = 0.0
    max_turn_xy_relaxation: float = 0.0
    support_xy_tracking_scale: float = 1.0
    wheel_coordinate_mode: str = "independent_support"
    max_learned_unload_shift_m: float = 0.0
    max_learned_support_relaxation: float = 0.0
    max_learned_unload_joint_correction: float = 0.0
