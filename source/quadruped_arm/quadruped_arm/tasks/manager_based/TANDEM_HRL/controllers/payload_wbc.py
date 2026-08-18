"""Support-preserving wheel and leg executor for the B2W platform."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.utils.math as math_utils
import torch


FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEG_JOINT_NAMES = (
    ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
)


@dataclass
class SupportWBCDiagnostics:
    wheel_command: torch.Tensor
    leg_correction_norm: torch.Tensor
    policy_leg_residual_norm: torch.Tensor
    support_position_error_norm: torch.Tensor
    payload_support_shift_xy: torch.Tensor
    turn_support_shift_m: torch.Tensor
    turn_xy_relaxation: torch.Tensor
    learned_unload_shift_m: torch.Tensor


def _to_base_frame(
    root_quaternion: torch.Tensor, vectors_w: torch.Tensor
) -> torch.Tensor:
    shape = vectors_w.shape
    rotations = root_quaternion.unsqueeze(1).expand(-1, shape[1], -1)
    return math_utils.quat_apply_inverse(
        rotations.reshape(-1, 4), vectors_w.reshape(-1, 3)
    ).reshape(shape)


class PayloadAwareSupportWBC:
    """Map wheel references to stable B2W leg and wheel actions.

    The leg correction is a residual around the ZYB-v0 nominal joint target.
    This preserves the position-servo preload required to support gravity.
    Payload position contributes a bounded quasi-static support shift; it does
    not replace either learned Task or learned Skill decisions.
    """

    def __init__(
        self,
        env,
        support_gain: float = 0.55,
        wheel_slew_per_step: float = 90.0,
        max_joint_correction: float = 0.08,
        nominal_supported_mass_kg: float = 55.0,
        max_payload_support_shift_m: float = 0.035,
        max_turn_support_shift_m: float = 0.0,
        max_turn_xy_relaxation: float = 0.0,
        support_xy_tracking_scale: float = 1.0,
        turn_action_reference: float = 90.0,
        max_policy_joint_residual: float = 0.14,
        max_learned_unload_shift_m: float = 0.0,
        max_learned_support_relaxation: float = 0.0,
        max_learned_unload_joint_correction: float = 0.0,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.support_gain = float(support_gain)
        self.wheel_slew_per_step = float(wheel_slew_per_step)
        self.max_joint_correction = float(max_joint_correction)
        self.nominal_supported_mass_kg = float(nominal_supported_mass_kg)
        self.max_payload_support_shift_m = float(
            max_payload_support_shift_m
        )
        self.max_turn_support_shift_m = float(max_turn_support_shift_m)
        self.max_turn_xy_relaxation = float(max_turn_xy_relaxation)
        self.support_xy_tracking_scale = float(support_xy_tracking_scale)
        self.turn_action_reference = float(turn_action_reference)
        self.max_policy_joint_residual = float(max_policy_joint_residual)
        self.max_learned_unload_shift_m = float(
            max_learned_unload_shift_m
        )
        self.max_learned_support_relaxation = float(
            max_learned_support_relaxation
        )
        self.max_learned_unload_joint_correction = float(
            max_learned_unload_joint_correction
        )

        leg_term = env.action_manager.get_term("leg_pos")
        self.action_leg_ids = torch.as_tensor(
            leg_term._joint_ids, device=env.device, dtype=torch.long
        ).flatten()
        self.action_scales = leg_term._scale.clone()
        self.action_offsets = leg_term._offset.clone()

        self.foot_ids: list[int] = []
        self.jacobian_body_ids: list[int] = []
        self.leg_joint_ids: list[torch.Tensor] = []
        for foot_name, joint_names in zip(FOOT_NAMES, LEG_JOINT_NAMES):
            body_ids, _ = self.robot.find_bodies(
                [foot_name], preserve_order=True
            )
            joint_ids, _ = self.robot.find_joints(
                list(joint_names), preserve_order=True
            )
            foot_id = int(body_ids[0])
            self.foot_ids.append(foot_id)
            self.jacobian_body_ids.append(foot_id)
            self.leg_joint_ids.append(
                torch.as_tensor(
                    joint_ids, device=env.device, dtype=torch.long
                ).flatten()
            )

        jacobians = self.robot.root_physx_view.get_jacobians()
        if self.robot.is_fixed_base:
            self.jacobian_joint_offset = 0
            self.jacobian_body_ids = [
                body_id - 1 for body_id in self.jacobian_body_ids
            ]
        else:
            self.jacobian_joint_offset = 6
        expected_width = (
            self.robot.data.joint_pos.shape[-1]
            + self.jacobian_joint_offset
        )
        if jacobians.shape[-1] != expected_width:
            raise RuntimeError(
                f"Unexpected B2W Jacobian shape {tuple(jacobians.shape)}"
            )
        if max(self.jacobian_body_ids) >= jacobians.shape[1]:
            raise RuntimeError("Resolved foot body is absent from Jacobian")

        self.foot_ids_tensor = torch.tensor(
            self.foot_ids, device=env.device, dtype=torch.long
        )
        self.dls_identity = torch.eye(
            3, device=env.device
        ).unsqueeze(0)
        self.wheel_command = torch.zeros(
            env.num_envs, 4, device=env.device
        )
        self.desired_foot_position_b = torch.zeros(
            env.num_envs, 4, 3, device=env.device
        )
        self.desired_height = torch.zeros(
            env.num_envs, device=env.device
        )
        self.reference_valid = False

    def reset_reference(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )
        root_quaternion = self.robot.data.root_quat_w
        foot_relative_w = (
            self.robot.data.body_pos_w[:, self.foot_ids_tensor]
            - self.robot.data.root_pos_w.unsqueeze(1)
        )
        foot_position_b = _to_base_frame(
            root_quaternion, foot_relative_w
        )
        self.desired_foot_position_b[env_ids] = foot_position_b[env_ids]
        self.desired_height[env_ids] = self.robot.data.root_pos_w[env_ids, 2]
        self.wheel_command[env_ids] = 0.0
        self.reference_valid = True

    def compute(
        self,
        wheel_target: torch.Tensor,
        policy_leg_residual: torch.Tensor | None = None,
        payload_mass_kg: torch.Tensor | None = None,
        payload_position_b: torch.Tensor | None = None,
        grasp_confidence: torch.Tensor | None = None,
        learned_support_coordinates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SupportWBCDiagnostics]:
        if not self.reference_valid:
            raise RuntimeError("reset_reference() must be called after settling")
        if wheel_target.shape != self.wheel_command.shape:
            raise ValueError(
                f"Expected wheel target {tuple(self.wheel_command.shape)}, "
                f"got {tuple(wheel_target.shape)}"
            )

        wheel_delta = (wheel_target - self.wheel_command).clamp(
            -self.wheel_slew_per_step, self.wheel_slew_per_step
        )
        self.wheel_command += wheel_delta

        root_quaternion = self.robot.data.root_quat_w
        current_relative_w = (
            self.robot.data.body_pos_w[:, self.foot_ids_tensor]
            - self.robot.data.root_pos_w.unsqueeze(1)
        )
        current_foot_position_b = _to_base_frame(
            root_quaternion, current_relative_w
        )
        position_error = (
            self.desired_foot_position_b - current_foot_position_b
        )
        height_error = self.desired_height - self.robot.data.root_pos_w[:, 2]
        position_error[:, :, 2] -= height_error.unsqueeze(1)

        left_command = 0.5 * (
            self.wheel_command[:, 0] + self.wheel_command[:, 2]
        )
        right_command = 0.5 * (
            self.wheel_command[:, 1] + self.wheel_command[:, 3]
        )
        turn_demand = 0.5 * (right_command - left_command)
        normalized_turn = (
            turn_demand.abs() / max(self.turn_action_reference, 1.0e-6)
        ).clamp(0.0, 1.0)
        turn_xy_relaxation = normalized_turn * self.max_turn_xy_relaxation
        position_error[:, :, :2] *= (
            self.support_xy_tracking_scale
            * (1.0 - turn_xy_relaxation)
        ).view(-1, 1, 1)
        turn_support_shift = (
            turn_demand / max(self.turn_action_reference, 1.0e-6)
        ).clamp(-1.0, 1.0) * self.max_turn_support_shift_m
        side_pattern = position_error.new_tensor(
            (1.0, -1.0, 1.0, -1.0)
        ).view(1, 4)
        position_error[:, :, 2] += (
            turn_support_shift.unsqueeze(1) * side_pattern
        )

        learned_unload_shift = torch.zeros(
            self.env.num_envs, 4, device=self.env.device
        )
        if learned_support_coordinates is not None:
            if learned_support_coordinates.shape == (self.env.num_envs, 4):
                # One learned value per wheel/foot.  Positive values relax
                # the corresponding support target; negative values request
                # no unloading.  The WBC still projects the result through
                # joint limits and the contact dynamics.
                unload_fraction = learned_support_coordinates.clamp(0.0, 1.0)
            elif learned_support_coordinates.shape == (self.env.num_envs, 2):
                # Retain the earlier two-coordinate diagnostic mode so old
                # probe scripts remain interpretable.
                front_rear_pattern = position_error.new_tensor(
                    (1.0, 1.0, -1.0, -1.0)
                ).view(1, 4)
                allocation = 0.5 * (
                    learned_support_coordinates[:, 0:1]
                    * side_pattern
                    + learned_support_coordinates[:, 1:2]
                    * front_rear_pattern
                )
                unload_fraction = allocation.clamp(0.0, 1.0)
            else:
                raise ValueError(
                    "Learned support coordinates must have shape (N, 4)"
                )
            position_error[:, :, 2] *= (
                1.0
                - self.max_learned_support_relaxation
                * unload_fraction
            )
            learned_unload_shift = (
                unload_fraction * self.max_learned_unload_shift_m
            )
            position_error[:, :, 2] += learned_unload_shift

        payload_shift = torch.zeros(
            self.env.num_envs, 2, device=self.env.device
        )
        if payload_mass_kg is not None and payload_position_b is not None:
            if grasp_confidence is None:
                grasp_confidence = torch.ones_like(payload_mass_kg)
            total_mass = self.nominal_supported_mass_kg + payload_mass_kg
            payload_shift = (
                payload_mass_kg.unsqueeze(1)
                * payload_position_b[:, :2]
                / total_mass.unsqueeze(1).clamp_min(1.0)
            )
            payload_shift *= grasp_confidence.unsqueeze(1).clamp(0.0, 1.0)
            payload_shift.clamp_(
                -self.max_payload_support_shift_m,
                self.max_payload_support_shift_m,
            )
            position_error[:, :, :2] -= payload_shift.unsqueeze(1)

        joint_correction = torch.zeros_like(self.robot.data.joint_pos)
        correction_norm = torch.zeros(
            self.env.num_envs, device=self.env.device
        )
        jacobians = self.robot.root_physx_view.get_jacobians()
        for leg_index in range(4):
            joint_ids = self.leg_joint_ids[leg_index]
            jacobian_joint_ids = (
                joint_ids + self.jacobian_joint_offset
            )
            jacobian_w = jacobians[
                :,
                self.jacobian_body_ids[leg_index],
                :3,
                jacobian_joint_ids,
            ]
            jacobian_columns_b = _to_base_frame(
                root_quaternion, jacobian_w.transpose(1, 2)
            )
            jacobian_b = jacobian_columns_b.transpose(1, 2)
            transpose = jacobian_b.transpose(1, 2)
            system = torch.bmm(jacobian_b, transpose) + (
                0.045**2
            ) * self.dls_identity
            delta = torch.bmm(
                transpose,
                torch.linalg.solve(
                    system,
                    self.support_gain
                    * position_error[:, leg_index].unsqueeze(-1),
                ),
            ).squeeze(-1)
            correction_limit = torch.full_like(
                delta,
                self.max_joint_correction,
            )
            if learned_support_coordinates is not None:
                correction_limit = correction_limit * 0.0 + (
                    self.max_joint_correction
                    + unload_fraction[:, leg_index].unsqueeze(1)
                    * self.max_learned_unload_joint_correction
                )
            delta = torch.maximum(
                torch.minimum(delta, correction_limit),
                -correction_limit,
            )
            joint_correction[:, joint_ids] = delta
            correction_norm = torch.maximum(
                correction_norm,
                torch.linalg.vector_norm(delta, dim=1),
            )

        limits = self.robot.data.soft_joint_pos_limits[:, self.action_leg_ids]
        learned_residual = torch.zeros_like(self.action_offsets)
        if policy_leg_residual is not None:
            if policy_leg_residual.shape != self.action_offsets.shape:
                raise ValueError(
                    "Policy leg residual must match the 12 leg actions"
                )
            # The caller supplies this residual in joint radians.  Keeping
            # the original per-joint ZYB-v0 scales here is important: hip,
            # thigh and calf actions do not have the same useful range.
            learned_residual = policy_leg_residual
            if self.max_policy_joint_residual > 0.0:
                learned_residual = learned_residual.clamp(
                    -self.max_policy_joint_residual,
                    self.max_policy_joint_residual,
                )
        target = (
            self.action_offsets
            + joint_correction[:, self.action_leg_ids]
            + learned_residual
        )
        target = torch.maximum(
            torch.minimum(target, limits[:, :, 1]), limits[:, :, 0]
        )
        actions = torch.zeros(
            self.env.num_envs,
            self.env.action_manager.total_action_dim,
            device=self.env.device,
        )
        actions[:, :12] = (target - self.action_offsets) / self.action_scales
        actions[:, 12:16] = self.wheel_command
        return actions, SupportWBCDiagnostics(
            wheel_command=self.wheel_command.clone(),
            leg_correction_norm=correction_norm,
            policy_leg_residual_norm=torch.linalg.vector_norm(
                learned_residual, dim=1
            ),
            support_position_error_norm=torch.linalg.vector_norm(
                position_error, dim=2
            ).amax(dim=1),
            payload_support_shift_xy=payload_shift,
            turn_support_shift_m=turn_support_shift,
            turn_xy_relaxation=turn_xy_relaxation,
            learned_unload_shift_m=learned_unload_shift,
        )
