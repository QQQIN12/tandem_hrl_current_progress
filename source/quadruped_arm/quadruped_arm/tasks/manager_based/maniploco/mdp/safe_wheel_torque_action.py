"""Bounded differential wheel torque with wheel-speed and yaw-rate feedback."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointEffortAction
from isaaclab.utils import configclass


@configclass
class SafeDifferentialWheelTorqueActionCfg(JointEffortActionCfg):
    """Command-conditioned wheel torque with a bounded policy residual."""

    class_type: type = None  # type: ignore[assignment]
    command_name: str = "locomotion"
    wheel_radius: float = 0.11
    track_width: float = 0.4693
    wheel_dir_signs: tuple[float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0
    )
    max_ref_vx: float = 0.25
    max_ref_wz: float = 0.10
    vx_feedback_gain: float = 0.15
    wz_feedback_gain: float = 1.0
    wheel_velocity_kp: float = 3.0
    yaw_torque_gain: float = 12.0
    torque_limit: float = 8.0
    residual_scale: float = 0.10
    max_torque_rate: float = 100.0
    min_height: float = 0.24
    height_gate_width: float = 0.045
    tilt_gate_gain: float = 1.5

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = SafeDifferentialWheelTorqueAction


class SafeDifferentialWheelTorqueAction(JointEffortAction):
    """Low-level wheel torque teacher with a safety-limited residual.

    The nominal command is physically scaled differential-drive kinematics.
    A bounded yaw-rate error term supplies a torque difference only when the
    measured body yaw rate is below the command.  This avoids encoding a
    permanent, unexplained large wheel-speed multiplier in the teacher.
    """

    cfg: SafeDifferentialWheelTorqueActionCfg

    def __init__(self, cfg: SafeDifferentialWheelTorqueActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._wheel_signs = torch.tensor(
            cfg.wheel_dir_signs, device=self.device
        ).view(1, -1)
        self._last_target = torch.zeros_like(self._processed_actions)
        env.safe_wheel_torque_feedforward = torch.zeros_like(
            self._processed_actions
        )
        env.safe_wheel_torque_residual = torch.zeros_like(
            self._processed_actions
        )
        env.safe_wheel_torque_target = torch.zeros_like(
            self._processed_actions
        )
        env.safe_wheel_torque_speed_reference = torch.zeros_like(
            self._processed_actions
        )
        env.safe_wheel_torque_wz_reference = torch.zeros(
            self._env.num_envs, device=env.device
        )

    def _safety_gate(self) -> torch.Tensor:
        robot = self._asset
        height = robot.data.root_pos_w[:, 2]
        tilt = torch.linalg.vector_norm(
            robot.data.projected_gravity_b[:, :2], dim=1
        )
        height_gate = torch.sigmoid(
            (height - self.cfg.min_height) / self.cfg.height_gate_width
        )
        tilt_gate = torch.exp(-self.cfg.tilt_gate_gain * tilt.square())
        return (height_gate * tilt_gate).clamp(0.0, 1.0)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        command = self._env.command_manager.get_command(
            self.cfg.command_name
        )
        vx = command[:, 0].clamp(
            -self.cfg.max_ref_vx, self.cfg.max_ref_vx
        )
        wz = command[:, 2].clamp(
            -self.cfg.max_ref_wz, self.cfg.max_ref_wz
        )

        body_velocity = self._asset.data.root_lin_vel_b[:, 0]
        body_yaw_rate = self._asset.data.root_ang_vel_b[:, 2]
        vx_ref = (
            vx + float(self.cfg.vx_feedback_gain) * (vx - body_velocity)
        ).clamp(-self.cfg.max_ref_vx, self.cfg.max_ref_vx)
        wz_ref = (
            wz + float(self.cfg.wz_feedback_gain) * (wz - body_yaw_rate)
        ).clamp(-self.cfg.max_ref_wz, self.cfg.max_ref_wz)
        self._env.safe_wheel_torque_wz_reference[:] = wz_ref.detach()

        left_speed = (
            vx_ref - 0.5 * self.cfg.track_width * wz_ref
        ) / self.cfg.wheel_radius
        right_speed = (
            vx_ref + 0.5 * self.cfg.track_width * wz_ref
        ) / self.cfg.wheel_radius
        speed_ref = torch.stack(
            (left_speed, right_speed, left_speed, right_speed), dim=-1
        )
        speed_ref = speed_ref * self._wheel_signs
        self._env.safe_wheel_torque_speed_reference[:] = speed_ref.detach()
        wheel_velocity = self._asset.data.joint_vel[:, self._joint_ids]
        effective_velocity = wheel_velocity * self._wheel_signs
        speed_error = speed_ref - effective_velocity
        torque_eff = float(self.cfg.wheel_velocity_kp) * speed_error

        yaw_error = wz_ref - body_yaw_rate
        yaw_torque = float(self.cfg.yaw_torque_gain) * yaw_error
        yaw_pattern = torch.stack(
            (-yaw_torque, yaw_torque, -yaw_torque, yaw_torque), dim=-1
        )
        torque_eff = torque_eff + yaw_pattern
        torque_eff = torque_eff * self._safety_gate().unsqueeze(-1)

        residual = actions * float(self.cfg.residual_scale)
        torque_eff = (torque_eff + residual * self._wheel_signs).clamp(
            -float(self.cfg.torque_limit), float(self.cfg.torque_limit)
        )
        target = torque_eff * self._wheel_signs
        max_delta = float(self.cfg.max_torque_rate) * float(
            self._env.step_dt
        )
        target = torch.maximum(
            torch.minimum(target, self._last_target + max_delta),
            self._last_target - max_delta,
        )
        self._last_target[:] = target
        self._processed_actions[:] = target
        self._env.safe_wheel_torque_feedforward[:] = (
            (torque_eff - residual * self._wheel_signs)
            * self._wheel_signs
        ).detach()
        self._env.safe_wheel_torque_residual[:] = residual.detach()
        self._env.safe_wheel_torque_target[:] = target.detach()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._last_target.zero_()
        else:
            self._last_target[env_ids] = 0.0
