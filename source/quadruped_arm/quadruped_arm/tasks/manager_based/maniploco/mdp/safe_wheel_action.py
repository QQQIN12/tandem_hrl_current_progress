"""Safe differential-drive wheel command with a bounded learned residual."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.utils import configclass


@configclass
class SafeDifferentialWheelVelocityActionCfg(JointVelocityActionCfg):
    """Command-conditioned wheel velocity plus a small policy residual."""

    class_type: type = None  # type: ignore[assignment]
    command_name: str = "locomotion"
    wheel_radius: float = 0.11
    track_width: float = 0.50
    wheel_dir_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    wz_sign: float = 1.0
    max_ref_vx: float = 0.45
    max_ref_wz: float = 0.65
    # The articulated wheel/leg contact has a much lower yaw response than
    # ideal differential-drive kinematics predict.  Keep this calibration
    # explicit instead of hiding it in track width or wheel radius.
    turn_speed_gain: float = 1.0
    max_wheel_speed: float = 5.0
    max_wheel_accel: float = 10.0
    turn_breakaway_wz: float = 0.0
    turn_breakaway_threshold: float = 0.03
    residual_scale: float = 0.10
    vx_feedback_gain: float = 0.35
    wz_feedback_gain: float = 0.20
    min_height: float = 0.24
    height_gate_width: float = 0.045
    tilt_gate_gain: float = 1.5
    # Optional measured-speed brake.  Zero keeps the historical behavior;
    # positive values add a feedback brake after target slew limiting.
    actual_speed_limit: float = 0.0
    actual_speed_brake_margin: float = 0.5

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = SafeDifferentialWheelVelocityAction


class SafeDifferentialWheelVelocityAction(JointVelocityAction):
    """Stable wheel teacher with a bounded residual for PPO fine-tuning.

    The policy still owns the four wheel action dimensions, but those values
    are interpreted as a small residual around a differential-drive command.
    The feed-forward target is rate-limited before it reaches the implicit
    PhysX wheel drive, preventing reset-time wheel impulses.
    """

    cfg: SafeDifferentialWheelVelocityActionCfg

    def __init__(self, cfg: SafeDifferentialWheelVelocityActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._last_target = torch.zeros_like(self._processed_actions)
        self._wheel_signs = torch.tensor(cfg.wheel_dir_signs, device=self.device).view(1, -1)
        env.safe_wheel_feedforward = torch.zeros_like(self._processed_actions)
        env.safe_wheel_residual = torch.zeros_like(self._processed_actions)
        env.safe_wheel_target = torch.zeros_like(self._processed_actions)
        env.safe_wheel_actual_velocity = torch.zeros_like(self._processed_actions)
        env.safe_wheel_brake_target = torch.zeros_like(self._processed_actions)
        env.safe_wheel_speed_brake = torch.zeros_like(self._processed_actions)

    def _safety_gate(self) -> torch.Tensor:
        robot = self._asset
        height = robot.data.root_pos_w[:, 2]
        tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
        height_gate = torch.sigmoid((height - self.cfg.min_height) / self.cfg.height_gate_width)
        tilt_gate = torch.exp(-self.cfg.tilt_gate_gain * tilt.square())
        return (height_gate * tilt_gate).clamp(0.0, 1.0)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        command = self._env.command_manager.get_command(self.cfg.command_name)
        vx = command[:, 0].clamp(-self.cfg.max_ref_vx, self.cfg.max_ref_vx)
        wz = (command[:, 2] * float(self.cfg.wz_sign)).clamp(
            -self.cfg.max_ref_wz, self.cfg.max_ref_wz
        )

        # A feed-forward wheel speed is not enough for this articulated
        # platform: leg compliance, wheel slip, and changing support height
        # can make the measured body velocity oppose the command.  Use a
        # bounded body-velocity error correction before converting to wheel
        # speed.  The command itself remains the reference, so this does not
        # turn the wheel term into an unconstrained learned controller.
        body_velocity = self._asset.data.root_lin_vel_b[:, 0]
        body_yaw_rate = self._asset.data.root_ang_vel_b[:, 2]
        vx = (
            vx + float(self.cfg.vx_feedback_gain) * (vx - body_velocity)
        ).clamp(-self.cfg.max_ref_vx, self.cfg.max_ref_vx)
        wz = (
            wz + float(self.cfg.wz_feedback_gain) * (wz - body_yaw_rate)
        ).clamp(-self.cfg.max_ref_wz, self.cfg.max_ref_wz)

        # The wheel/ground contact has a measurable breakaway threshold.  A
        # nonzero value is intentionally opt-in: the nominal configuration
        # remains a pure command-to-wheel-speed map until this is validated.
        breakaway = abs(float(self.cfg.turn_breakaway_wz))
        if breakaway > 0.0:
            threshold = float(self.cfg.turn_breakaway_threshold)
            active_turn = wz.abs() > threshold
            wz = torch.where(
                active_turn,
                wz.sign() * torch.maximum(wz.abs(), torch.as_tensor(breakaway, device=wz.device, dtype=wz.dtype)),
                wz,
            )

        wz_drive = wz * float(self.cfg.turn_speed_gain)
        left = (vx - 0.5 * self.cfg.track_width * wz_drive) / self.cfg.wheel_radius
        right = (vx + 0.5 * self.cfg.track_width * wz_drive) / self.cfg.wheel_radius
        ff = torch.stack((left, right, left, right), dim=-1)
        ff = ff * self._wheel_signs
        ff = ff * self._safety_gate().unsqueeze(-1)
        ff = ff.clamp(-self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)

        residual = actions * self.cfg.residual_scale
        target = (ff + residual).clamp(-self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
        max_delta = self.cfg.max_wheel_accel * float(self._env.step_dt)
        target = torch.maximum(torch.minimum(target, self._last_target + max_delta), self._last_target - max_delta)

        # A reference-speed clamp is not an actual-speed clamp: the implicit
        # PhysX drive can overshoot while the wheel is loaded.  When enabled,
        # use measured wheel velocity in the effective rolling convention and
        # command a lower same-sign target until the overspeed is removed.
        actual_velocity = self._asset.data.joint_vel[:, self._joint_ids]
        effective_velocity = actual_velocity * self._wheel_signs
        actual_limit = float(self.cfg.actual_speed_limit)
        brake_margin = max(float(self.cfg.actual_speed_brake_margin), 0.0)
        brake_target_effective = effective_velocity.sign() * max(actual_limit - brake_margin, 0.0)
        target_effective = target * self._wheel_signs
        if actual_limit > 0.0:
            too_fast_positive = effective_velocity > actual_limit
            too_fast_negative = effective_velocity < -actual_limit
            target_effective = torch.where(
                too_fast_positive,
                torch.minimum(target_effective, brake_target_effective),
                target_effective,
            )
            target_effective = torch.where(
                too_fast_negative,
                torch.maximum(target_effective, brake_target_effective),
                target_effective,
            )
            target = target_effective * self._wheel_signs
            speed_brake = (too_fast_positive | too_fast_negative).to(target.dtype)
        else:
            speed_brake = torch.zeros_like(target)

        self._last_target[:] = target
        self._processed_actions[:] = target
        self._env.safe_wheel_feedforward[:] = ff.detach()
        self._env.safe_wheel_residual[:] = residual.detach()
        self._env.safe_wheel_target[:] = target.detach()
        self._env.safe_wheel_actual_velocity[:] = actual_velocity.detach()
        self._env.safe_wheel_brake_target[:] = (brake_target_effective * self._wheel_signs).detach()
        self._env.safe_wheel_speed_brake[:] = speed_brake.detach()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._last_target.zero_()
            self._env.safe_wheel_actual_velocity.zero_()
            self._env.safe_wheel_brake_target.zero_()
            self._env.safe_wheel_speed_brake.zero_()
        else:
            self._last_target[env_ids] = 0.0
            self._env.safe_wheel_actual_velocity[env_ids] = 0.0
            self._env.safe_wheel_brake_target[env_ids] = 0.0
            self._env.safe_wheel_speed_brake[env_ids] = 0.0
