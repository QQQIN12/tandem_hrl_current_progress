"""Bounded leg residual with contact-conditioned base posture feedback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


@configclass
class StabilizedLegPositionActionCfg(JointPositionActionCfg):
    class_type: Optional[type] = None
    contact_sensor_name: str = "contact_forces"
    foot_body_names: tuple[str, ...] = (
        "FL_foot", "FR_foot", "RL_foot", "RR_foot",
    )
    foot_leg_joint_names: tuple[tuple[str, ...], ...] = (
        ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
        ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
        ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
        ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
    )
    max_policy_residual: float = 0.22
    posture_feedback_enabled: bool = True
    posture_feedback_authority: float = 0.75
    contact_force_threshold: float = 1.5
    base_height_target: float = 0.46
    base_height_gain: float = 1.40
    base_vertical_damping: float = 0.30
    orientation_gain: float = 1.80
    angular_velocity_damping: float = 0.35
    max_vertical_recovery_velocity: float = 0.12
    max_angular_recovery_velocity: float = 0.40
    posture_feedback_horizon: float = 0.10
    posture_feedback_damping: float = 0.05
    posture_feedback_joint_limit: float = 0.10
    joint_limit_margin: float = 0.02
    # Deployable safety envelope.  When enabled, the learned leg target is
    # smoothly blended back to the default stance as tilt or height margin
    # deteriorates.  It is disabled in the legacy/base configuration so old
    # probes remain comparable.
    safety_gate_enabled: bool = False
    safety_tilt_soft_limit: float = 0.16
    safety_tilt_gate_width: float = 0.04
    safety_min_height: float = 0.30
    safety_height_gate_width: float = 0.04
    # Deterministic wheel/leg turning coordinator.  The wheel channel remains
    # the primary yaw actuator; this optional term only changes the two side
    # hip targets in proportion to lateral acceleration (vx * wz).
    turn_coordination_enabled: bool = False
    turn_command_name: str = "locomotion"
    turn_max_vx: float = 0.25
    turn_max_wz: float = 0.10
    turn_coord_joint_names: tuple[str, ...] = ()
    turn_coord_joint_weights: tuple[float, ...] = ()
    turn_hip_offset_gain: float = 0.0
    turn_hip_offset_limit: float = 0.08
    turn_hip_offset_sign: float = 1.0
    turn_signal_smoothing: float = 0.20
    # Optional bounded compliance modulation.  The factor is restored toward
    # one when the safety gate closes, so a large tilt does not make the legs
    # softer while recovering.
    turn_stiffness_enabled: bool = False
    turn_stiffness_min_factor: float = 0.75
    turn_stiffness_smoothing: float = 0.20
    # Optional contact-force load-transfer feedback.  Positive signal means
    # the left side carries more measured load; the coordinator then raises
    # the left calf targets and lowers the right calf targets, based on the
    # sign identified by a static stance sweep.  This is intentionally
    # independent of the command-conditioned turn offset.
    turn_load_balance_enabled: bool = False
    turn_load_balance_gain: float = 0.12
    turn_load_balance_limit: float = 0.05
    turn_load_balance_smoothing: float = 0.12
    turn_load_balance_min_total_force: float = 40.0

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = StabilizedLegPositionAction


class StabilizedLegPositionAction(JointPositionAction):
    """Execute learned leg targets plus a bounded contact-foot correction."""

    cfg: StabilizedLegPositionActionCfg

    def __init__(self, cfg: StabilizedLegPositionActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._sensor = env.scene[cfg.contact_sensor_name]

        sensor_names = list(self._sensor.body_names)
        missing = [name for name in cfg.foot_body_names if name not in sensor_names]
        if missing:
            raise RuntimeError(
                "Contact sensor is missing stabilized-leg feet: "
                + ", ".join(missing)
            )
        self._foot_sensor_ids = torch.tensor(
            [sensor_names.index(name) for name in cfg.foot_body_names],
            device=env.device,
            dtype=torch.long,
        )

        foot_body_ids, resolved = self._robot.find_bodies(
            list(cfg.foot_body_names), preserve_order=True
        )
        if len(foot_body_ids) != 4:
            raise RuntimeError(
                "Expected four stabilized-leg feet, resolved " + str(list(resolved))
            )

        jacobians = self._robot.root_physx_view.get_jacobians()
        body_offset = (
            1
            if jacobians.shape[1] == len(self._robot.data.body_names) - 1
            else 0
        )
        joint_offset = jacobians.shape[-1] - len(self._robot.data.joint_names)
        if joint_offset not in (0, 6):
            raise RuntimeError(
                "Unsupported stabilized-leg Jacobian layout: "
                + str(tuple(jacobians.shape))
            )
        self._foot_jacobian_body_ids = torch.tensor(
            [int(body_id) - body_offset for body_id in foot_body_ids],
            device=env.device,
            dtype=torch.long,
        )
        self._jacobian_joint_offset = int(joint_offset)

        # JointPositionAction may resolve names in articulation order unless
        # preserve_order is explicitly enabled.  Build the feedback map from
        # the resolved action ids so the contact correction cannot be applied
        # to a different leg when a config omits that flag.
        robot_name_to_id = {
            name: index for index, name in enumerate(self._robot.data.joint_names)
        }
        action_name_to_local = {
            self._robot.data.joint_names[int(robot_joint_id)]: local_index
            for local_index, robot_joint_id in enumerate(self._joint_ids)
        }
        self._foot_leg_local_ids: list[torch.Tensor] = []
        self._foot_leg_jacobian_ids: list[torch.Tensor] = []
        for names in cfg.foot_leg_joint_names:
            if len(names) != 3:
                raise ValueError("Each stabilized foot must map to three leg joints")
            self._foot_leg_local_ids.append(
                torch.tensor(
                    [action_name_to_local[name] for name in names],
                    device=env.device,
                    dtype=torch.long,
                )
            )
            self._foot_leg_jacobian_ids.append(
                torch.tensor(
                    [robot_name_to_id[name] + joint_offset for name in names],
                    device=env.device,
                    dtype=torch.long,
                )
            )

        self._dls_identity = torch.eye(3, device=env.device).unsqueeze(0)
        self._feedback = torch.zeros_like(self._processed_actions)
        self._safety_gate = torch.ones(self._env.num_envs, device=env.device)
        self._turn_signal = torch.zeros(self._env.num_envs, device=env.device)
        self._turn_hip_offset = torch.zeros_like(self._processed_actions)
        self._turn_stiffness_factor = torch.ones(
            self._env.num_envs, device=env.device
        )
        self._turn_load_signal = torch.zeros(
            self._env.num_envs, device=env.device
        )
        self._turn_load_offset = torch.zeros_like(self._processed_actions)
        self._default_leg_stiffness = self._robot.data.default_joint_stiffness[
            :, self._joint_ids
        ].clone()
        self._default_leg_damping = self._robot.data.default_joint_damping[
            :, self._joint_ids
        ].clone()
        action_name_to_local = {
            name: index for index, name in enumerate(self._joint_names)
        }
        coord_names = tuple(self.cfg.turn_coord_joint_names)
        if not coord_names:
            coord_names = tuple(
                name for name in self._joint_names
                if name.endswith("_hip_joint")
            )
        missing_coord = [
            name for name in coord_names if name not in action_name_to_local
        ]
        if missing_coord:
            raise ValueError(
                "Turn coordinator joints are missing from the leg action: "
                + ", ".join(missing_coord)
            )
        coord_weights = tuple(self.cfg.turn_coord_joint_weights)
        if not coord_weights:
            coord_weights = (1.0,) * len(coord_names)
        if len(coord_weights) != len(coord_names):
            raise ValueError(
                "turn_coord_joint_weights must match turn_coord_joint_names"
            )
        self._turn_coord_local_ids = torch.tensor(
            [action_name_to_local[name] for name in coord_names],
            device=env.device,
            dtype=torch.long,
        )
        side_signs = [
            1.0 if name.startswith(("FL", "RL")) else -1.0
            for name in coord_names
        ]
        self._turn_coord_side_sign = torch.tensor(
            side_signs, device=env.device, dtype=torch.float32
        ).view(1, -1)
        self._turn_coord_weights = torch.tensor(
            coord_weights, device=env.device, dtype=torch.float32
        ).view(1, -1)
        env.safe_leg_posture_feedback = self._feedback
        env.safe_leg_safety_gate = self._safety_gate
        env.turn_coordination_signal = self._turn_signal
        env.turn_hip_offset = self._turn_hip_offset
        env.turn_stiffness_factor = self._turn_stiffness_factor
        env.turn_load_balance_signal = self._turn_load_signal
        env.turn_load_balance_offset = self._turn_load_offset

    def _compute_safety_gate(self) -> torch.Tensor:
        """Return the standalone leg-action authority in [0, 1]."""

        if not self.cfg.safety_gate_enabled:
            self._safety_gate.fill_(1.0)
            return self._safety_gate
        tilt = torch.asin(
            torch.linalg.vector_norm(
                self._robot.data.projected_gravity_b[:, :2], dim=1
            ).clamp(0.0, 1.0)
        )
        height = self._robot.data.root_pos_w[:, 2]
        tilt_gate = torch.sigmoid(
            (float(self.cfg.safety_tilt_soft_limit) - tilt)
            / max(float(self.cfg.safety_tilt_gate_width), 1.0e-4)
        )
        height_gate = torch.sigmoid(
            (height - float(self.cfg.safety_min_height))
            / max(float(self.cfg.safety_height_gate_width), 1.0e-4)
        )
        self._safety_gate[:] = (tilt_gate * height_gate).clamp(0.0, 1.0)
        return self._safety_gate

    def _posture_feedback_target(self) -> torch.Tensor:
        if not self.cfg.posture_feedback_enabled:
            self._feedback.zero_()
            return self._feedback

        root_quat = self._robot.data.root_quat_w
        body_up = quat_apply(
            root_quat,
            root_quat.new_tensor((0.0, 0.0, 1.0)).view(1, 3).expand(
                self._env.num_envs, -1
            ),
        )
        world_up = body_up.new_tensor((0.0, 0.0, 1.0)).view(1, 3)
        orientation_error = torch.linalg.cross(
            body_up, world_up.expand_as(body_up)
        )
        desired_angular = (
            float(self.cfg.orientation_gain) * orientation_error
            - float(self.cfg.angular_velocity_damping)
            * self._robot.data.root_ang_vel_w
        )
        desired_angular[:, 2] = 0.0
        angular_norm = torch.linalg.vector_norm(
            desired_angular[:, :2], dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        desired_angular[:, :2] *= (
            float(self.cfg.max_angular_recovery_velocity) / angular_norm
        ).clamp_max(1.0)

        desired_vertical = (
            float(self.cfg.base_height_gain)
            * (float(self.cfg.base_height_target) - self._robot.data.root_pos_w[:, 2])
            - float(self.cfg.base_vertical_damping)
            * self._robot.data.root_lin_vel_w[:, 2]
        ).clamp(
            -float(self.cfg.max_vertical_recovery_velocity),
            float(self.cfg.max_vertical_recovery_velocity),
        )
        desired_base_twist = torch.zeros(
            self._env.num_envs, 6, device=self._env.device
        )
        desired_base_twist[:, 2] = desired_vertical
        desired_base_twist[:, 3:6] = desired_angular

        jacobians = self._robot.root_physx_view.get_jacobians()
        damping_squared = float(self.cfg.posture_feedback_damping) ** 2
        horizon = float(self.cfg.posture_feedback_horizon)
        joint_limit = float(self.cfg.posture_feedback_joint_limit)
        foot_forces = self._sensor.data.net_forces_w[:, self._foot_sensor_ids]
        contact = (
            torch.linalg.vector_norm(foot_forces, dim=-1)
            > float(self.cfg.contact_force_threshold)
        )
        self._feedback.zero_()
        for foot_index in range(4):
            foot_jacobian = jacobians[
                :, self._foot_jacobian_body_ids[foot_index], 0:3, :
            ]
            base_jacobian = foot_jacobian[:, :, :6]
            leg_jacobian = foot_jacobian[
                :, :, self._foot_leg_jacobian_ids[foot_index]
            ]
            rhs = -torch.bmm(
                base_jacobian, desired_base_twist.unsqueeze(-1)
            )
            transpose = leg_jacobian.transpose(1, 2)
            system = (
                torch.bmm(leg_jacobian, transpose)
                + damping_squared * self._dls_identity
            )
            joint_velocity = torch.bmm(
                transpose, torch.linalg.solve(system, rhs)
            ).squeeze(-1)
            delta = (horizon * joint_velocity).clamp(
                -joint_limit, joint_limit
            )
            delta *= contact[:, foot_index].unsqueeze(1)
            self._feedback[:, self._foot_leg_local_ids[foot_index]] = delta
        return self._feedback

    def _update_turn_coordination(self, safety_gate: torch.Tensor) -> None:
        """Update the bounded, command-conditioned leg turn assistance."""

        if not (
            self.cfg.turn_coordination_enabled
            or self.cfg.turn_load_balance_enabled
            or self.cfg.turn_stiffness_enabled
        ):
            self._turn_signal.zero_()
            self._turn_hip_offset.zero_()
            self._turn_stiffness_factor.fill_(1.0)
            self._turn_load_signal.zero_()
            self._turn_load_offset.zero_()
            return

        if self.cfg.turn_coordination_enabled:
            command = self._env.command_manager.get_command(
                self.cfg.turn_command_name
            )
            max_vx = max(abs(float(self.cfg.turn_max_vx)), 1.0e-6)
            max_wz = max(abs(float(self.cfg.turn_max_wz)), 1.0e-6)
            raw_signal = (
                command[:, 0].clamp(-max_vx, max_vx) / max_vx
            ) * (command[:, 2].clamp(-max_wz, max_wz) / max_wz)
            raw_signal = raw_signal.clamp(-1.0, 1.0)
            signal_alpha = float(self.cfg.turn_signal_smoothing)
            signal_alpha = max(0.0, min(1.0, signal_alpha))
            self._turn_signal[:] = self._turn_signal + signal_alpha * (
                raw_signal - self._turn_signal
            )
        else:
            self._turn_signal.zero_()

        coord_offset = (
            self._turn_signal.unsqueeze(-1)
            * self._turn_coord_side_sign
            * self._turn_coord_weights
            * float(self.cfg.turn_hip_offset_gain)
            * float(self.cfg.turn_hip_offset_sign)
        )

        if self.cfg.turn_load_balance_enabled:
            foot_forces = torch.linalg.vector_norm(
                self._sensor.data.net_forces_w[:, self._foot_sensor_ids],
                dim=-1,
            )
            left_force = foot_forces[:, 0] + foot_forces[:, 2]
            right_force = foot_forces[:, 1] + foot_forces[:, 3]
            total_force = left_force + right_force
            raw_load_signal = (
                (left_force - right_force)
                / total_force.clamp_min(
                    float(self.cfg.turn_load_balance_min_total_force)
                )
            ).clamp(-1.0, 1.0)
            valid = (
                total_force
                >= float(self.cfg.turn_load_balance_min_total_force)
            )
            raw_load_signal = torch.where(
                valid, raw_load_signal, torch.zeros_like(raw_load_signal)
            )
            load_alpha = float(self.cfg.turn_load_balance_smoothing)
            load_alpha = max(0.0, min(1.0, load_alpha))
            self._turn_load_signal[:] = self._turn_load_signal + load_alpha * (
                raw_load_signal - self._turn_load_signal
            )
            load_offset = (
                self._turn_load_signal.unsqueeze(-1)
                * self._turn_coord_side_sign
                * float(self.cfg.turn_load_balance_gain)
            )
            load_offset = load_offset.clamp(
                -float(self.cfg.turn_load_balance_limit),
                float(self.cfg.turn_load_balance_limit),
            )
        else:
            self._turn_load_signal.zero_()
            load_offset = torch.zeros_like(coord_offset)

        self._turn_load_offset.zero_()
        self._turn_load_offset[:, self._turn_coord_local_ids] = load_offset
        self._turn_hip_offset.zero_()
        self._turn_hip_offset[:, self._turn_coord_local_ids] = (
            coord_offset + load_offset
        ).clamp(
            -float(self.cfg.turn_hip_offset_limit),
            float(self.cfg.turn_hip_offset_limit),
        )

        if not self.cfg.turn_stiffness_enabled:
            self._turn_stiffness_factor.fill_(1.0)
            return

        minimum = max(
            0.05, min(1.0, float(self.cfg.turn_stiffness_min_factor))
        )
        desired_factor = 1.0 - (1.0 - minimum) * self._turn_signal.abs()
        stiffness_alpha = float(self.cfg.turn_stiffness_smoothing)
        stiffness_alpha = max(0.0, min(1.0, stiffness_alpha))
        self._turn_stiffness_factor[:] = (
            self._turn_stiffness_factor
            + stiffness_alpha * (desired_factor - self._turn_stiffness_factor)
        )
        # Close the compliance envelope during recovery.  The controller can
        # lean while healthy, but never softens the legs as tilt/height fails.
        self._turn_stiffness_factor[:] = 1.0 - safety_gate * (
            1.0 - self._turn_stiffness_factor
        )
        stiffness = self._default_leg_stiffness * self._turn_stiffness_factor.unsqueeze(-1)
        # Preserve approximately the same damping ratio as stiffness changes.
        damping = self._default_leg_damping * torch.sqrt(
            self._turn_stiffness_factor.clamp_min(0.05).unsqueeze(-1)
        )
        self._robot.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self._joint_ids
        )
        self._robot.write_joint_damping_to_sim(
            damping, joint_ids=self._joint_ids
        )

    def apply_actions(self):
        # Preserve the standard scale/offset semantics, then constrain the
        # learned residual and add only the contact-conditioned recovery term.
        target = self._processed_actions
        default = self._robot.data.default_joint_pos[:, self._joint_ids]
        if self.cfg.max_policy_residual > 0.0:
            target = default + (
                target - default
            ).clamp(
                -float(self.cfg.max_policy_residual),
                float(self.cfg.max_policy_residual),
            )
        else:
            # A zero residual limit is an explicit freeze request.  The
            # previous implementation skipped the clamp in this case and
            # accidentally passed the unrestricted policy target through,
            # so configurations documented as "frozen" still moved the legs.
            target = default
        safety_gate = self._compute_safety_gate()
        self._update_turn_coordination(safety_gate)
        target = default + safety_gate.unsqueeze(-1) * (
            target - default + self._turn_hip_offset
        )
        feedback = self._posture_feedback_target()
        target = target + float(self.cfg.posture_feedback_authority) * feedback

        limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]
        margin = float(self.cfg.joint_limit_margin)
        target = torch.maximum(
            torch.minimum(target, limits[..., 1] - margin),
            limits[..., 0] + margin,
        )
        self._processed_actions[:] = target
        self._robot.set_joint_position_target(
            target, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._feedback.zero_()
            self._safety_gate.fill_(1.0)
            self._turn_signal.zero_()
            self._turn_hip_offset.zero_()
            self._turn_stiffness_factor.fill_(1.0)
            self._turn_load_signal.zero_()
            self._turn_load_offset.zero_()
            reset_ids = None
        else:
            self._feedback[env_ids] = 0.0
            self._safety_gate[env_ids] = 1.0
            self._turn_signal[env_ids] = 0.0
            self._turn_hip_offset[env_ids] = 0.0
            self._turn_stiffness_factor[env_ids] = 1.0
            self._turn_load_signal[env_ids] = 0.0
            self._turn_load_offset[env_ids] = 0.0
            reset_ids = env_ids
        if self.cfg.turn_stiffness_enabled:
            if reset_ids is None:
                stiffness = self._default_leg_stiffness
                damping = self._default_leg_damping
            else:
                stiffness = self._default_leg_stiffness[reset_ids]
                damping = self._default_leg_damping[reset_ids]
            self._robot.write_joint_stiffness_to_sim(
                stiffness, joint_ids=self._joint_ids, env_ids=reset_ids
            )
            self._robot.write_joint_damping_to_sim(
                damping, joint_ids=self._joint_ids, env_ids=reset_ids
            )
