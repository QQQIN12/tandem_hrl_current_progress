"""Replay a validated arm path to isolate grasp and contact mechanics.

The robot starts inside a known grasp basin while the object remains a dynamic
rigid body on the source platform.  This isolates contact mechanics from policy
quality and records the data needed to diagnose a failed acquisition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--hold_steps", type=int, default=180)
parser.add_argument("--finger_static_friction", type=float, default=-1.0)
parser.add_argument("--finger_dynamic_friction", type=float, default=-1.0)
parser.add_argument("--disable_finger_material", action="store_true")
parser.add_argument("--closed_loop_alignment", action="store_true")
parser.add_argument("--retention_latch", action="store_true")
parser.add_argument("--event_gated_progress", action="store_true")
parser.add_argument("--iterative_ingress", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from quadruped_arm.tasks.manager_based.zyb_real_grasp.env_cfg import (
    ZYBRealGraspSceneEnvCfg,
)
from quadruped_arm.tasks.manager_based.zyb_real_grasp.physics_contract import (
    FINGER_SIMULATION_MATERIAL,
)
from quadruped_arm.tasks.manager_based.zyb_real_grasp.scene_cfg import (
    OBJECT_START_POSITION,
    SOURCE_PLATFORM_POSITION,
)


# Keyframes reconstructed from the previously validated contact/lift trace.
# Values are actuator-side joint targets, not privileged object motion.
GRASP_KEYFRAMES = (
    (0, (0.358049, 1.173483, -0.613057, 0.428305, -0.658628, 0.785306), (0.032451, -0.032904)),
    (15, (0.195950, 1.052133, -0.446526, 0.285876, -0.521030, 0.636299), (0.032809, -0.032809)),
    (30, (-0.085208, 0.773258, -0.427586, 0.001521, -0.230384, 0.567965), (0.032809, -0.032809)),
    (45, (0.188893, 0.482610, -0.563295, -0.283208, 0.059578, 0.230685), (0.032809, -0.032809)),
    (60, (0.264110, 0.120850, -0.571126, -0.413335, 0.351163, 0.417860), (0.032809, -0.032809)),
    (75, (0.422696, 0.045708, -0.461975, -0.705514, 0.497940, 0.628272), (0.030000, -0.030000)),
    (90, (0.472057, -0.064977, -0.304683, -0.844953, 0.512806, 0.746718), (0.030000, -0.030000)),
    (97, (0.484445, -0.149806, -0.205038, -0.861699, 0.551943, 0.767549), (0.030000, -0.030000)),
    (105, (0.631325, 0.002058, -0.215192, -0.931578, 0.599941, 0.862618), (0.021352, -0.021490)),
    (110, (0.653598, -0.063942, -0.181062, -0.933823, 0.698944, 0.859237), (0.015840, -0.014708)),
    (114, (0.684148, -0.059492, -0.187704, -0.897509, 0.774987, 0.862973), (0.014407, -0.015591)),
    (118, (0.702812, -0.128006, -0.191466, -0.903738, 0.835156, 0.833571), (0.006545, -0.023588)),
    (122, (0.678814, -0.197720, -0.216181, -0.809995, 0.790065, 0.746326), (0.005006, -0.024950)),
    (125, (0.665944, -0.152574, -0.258763, -0.821988, 0.787530, 0.800560), (0.005209, -0.024741)),
    (130, (0.635533, -0.133642, -0.281626, -0.811893, 0.788408, 0.853977), (0.005518, -0.024432)),
    (135, (0.588267, -0.090650, -0.308669, -0.748677, 0.720347, 0.889471), (0.006324, -0.023631)),
    (140, (0.545395, -0.061183, -0.332968, -0.689371, 0.666104, 0.922353), (0.007419, -0.022535)),
    (145, (0.507221, -0.101340, -0.326396, -0.658472, 0.621370, 0.981730), (0.008954, -0.021000)),
    (150, (0.466247, -0.087589, -0.308804, -0.605808, 0.595605, 1.017436), (0.010667, -0.019286)),
    (160, (0.367449, -0.084065, -0.268394, -0.591322, 0.590778, 0.998517), (0.014239, -0.015791)),
    (170, (0.260849, -0.021508, -0.246961, -0.657556, 0.551068, 1.076688), (0.018087, -0.011878)),
    (180, (0.217211, 0.040637, -0.227466, -0.492309, 0.482978, 1.091360), (0.022620, -0.007344)),
    (260, (-0.053631, 0.277952, -0.316573, 0.281573, 0.288019, 0.735165), (0.031000, -0.002000)),
)

# Object position relative to the mean of link7/link8 in the validated trace.
# The servo follows this geometry instead of assuming the replayed joints will
# produce the same Cartesian path after a small base or contact perturbation.
RELATION_KEYFRAMES = (
    (60, (0.011475, 0.030548, -0.163833)),
    (75, (-0.005671, 0.009999, -0.118296)),
    (90, (-0.000887, -0.000193, -0.083486)),
    (97, (-0.005044, -0.000121, -0.068926)),
    (105, (-0.007000, -0.001794, -0.041859)),
    (110, (-0.013436, 0.000363, -0.037400)),
    (114, (-0.014319, -0.000991, -0.023585)),
    (118, (-0.014914, 0.001526, 0.001095)),
    (122, (-0.012368, -0.001224, 0.003161)),
    (125, (-0.013179, -0.000678, 0.005932)),
    (130, (-0.013606, 0.000441, 0.004061)),
    (140, (-0.013558, -0.000598, 0.004803)),
    (160, (-0.012183, -0.002412, 0.002446)),
    (180, (-0.011027, 0.005768, 0.001435)),
    (260, (-0.011533, 0.003296, 0.001081)),
)

FINGER_PAD_OFFSET_LOCAL = (0.0, -0.03825, 0.01325)
RETENTION_CONTACT_STREAK = 8
RETENTION_FORCE_BALANCE = 0.25
RETENTION_TRANSFER_STEPS = 100
RETENTION_FORCE_TARGET_N = 0.24
RETENTION_GRIP_STEP_M = 0.00025


def _interpolate(step: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if step <= GRASP_KEYFRAMES[0][0]:
        left = right = GRASP_KEYFRAMES[0]
        alpha = 0.0
    elif step >= GRASP_KEYFRAMES[-1][0]:
        left = right = GRASP_KEYFRAMES[-1]
        alpha = 0.0
    else:
        for index in range(len(GRASP_KEYFRAMES) - 1):
            left = GRASP_KEYFRAMES[index]
            right = GRASP_KEYFRAMES[index + 1]
            if left[0] <= step <= right[0]:
                alpha = (step - left[0]) / float(right[0] - left[0])
                break
    arm_left = torch.tensor(left[1], device=device)
    arm_right = torch.tensor(right[1], device=device)
    grip_left = torch.tensor(left[2], device=device)
    grip_right = torch.tensor(right[2], device=device)
    return torch.lerp(arm_left, arm_right, alpha), torch.lerp(
        grip_left, grip_right, alpha
    )


def _interpolate_relation(step: int, device: str) -> torch.Tensor:
    if step <= RELATION_KEYFRAMES[0][0]:
        left = right = RELATION_KEYFRAMES[0]
        alpha = 0.0
    elif step >= RELATION_KEYFRAMES[-1][0]:
        left = right = RELATION_KEYFRAMES[-1]
        alpha = 0.0
    else:
        for index in range(len(RELATION_KEYFRAMES) - 1):
            left = RELATION_KEYFRAMES[index]
            right = RELATION_KEYFRAMES[index + 1]
            if left[0] <= step <= right[0]:
                alpha = (step - left[0]) / float(right[0] - left[0])
                break
    return torch.lerp(
        torch.tensor(left[1], device=device),
        torch.tensor(right[1], device=device),
        alpha,
    )


def _force(sensor) -> torch.Tensor:
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(
        forces.reshape(forces.shape[0], -1, 3), dim=-1
    ).amax(dim=1)


def _friction_force(sensor, num_envs: int, device: str) -> torch.Tensor:
    friction = getattr(sensor.data, "friction_forces_w", None)
    if friction is None:
        return torch.zeros(num_envs, device=device)
    return torch.linalg.vector_norm(
        friction.reshape(friction.shape[0], -1, 3), dim=-1
    ).amax(dim=1)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(
        vector.shape[0], 3, 3, device=vector.device, dtype=vector.dtype
    )
    matrix[:, 0, 1] = -vector[:, 2]
    matrix[:, 0, 2] = vector[:, 1]
    matrix[:, 1, 0] = vector[:, 2]
    matrix[:, 1, 2] = -vector[:, 0]
    matrix[:, 2, 0] = -vector[:, 1]
    matrix[:, 2, 1] = vector[:, 0]
    return matrix


def _finger_pad_geometry(
    robot, finger_body_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    finger_position = robot.data.body_pos_w[:, finger_body_ids]
    finger_rotation = robot.data.body_quat_w[:, finger_body_ids]
    local_offset = finger_position.new_tensor(
        FINGER_PAD_OFFSET_LOCAL
    ).view(1, 1, 3).expand(finger_position.shape[0], 2, 3)
    world_offset = math_utils.quat_apply(
        finger_rotation.reshape(-1, 4), local_offset.reshape(-1, 3)
    ).reshape_as(local_offset)
    return finger_position + world_offset, world_offset


def _configure_finger_material(
    cfg: ZYBRealGraspSceneEnvCfg,
) -> tuple[float, float] | None:
    if args.disable_finger_material:
        cfg.events.finger_contact_material = None
        return None
    if args.finger_static_friction < 0.0 and args.finger_dynamic_friction < 0.0:
        return (
            FINGER_SIMULATION_MATERIAL.static_friction,
            FINGER_SIMULATION_MATERIAL.dynamic_friction,
        )
    if args.finger_static_friction < 0.0 or args.finger_dynamic_friction < 0.0:
        raise ValueError("Both finger friction coefficients must be provided")
    if args.finger_static_friction < args.finger_dynamic_friction:
        raise ValueError("Static friction must be at least dynamic friction")
    cfg.events.finger_contact_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["link7", "link8"]
            ),
            "static_friction_range": (
                args.finger_static_friction,
                args.finger_static_friction,
            ),
            "dynamic_friction_range": (
                args.finger_dynamic_friction,
                args.finger_dynamic_friction,
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    return args.finger_static_friction, args.finger_dynamic_friction


def main() -> None:
    cfg = ZYBRealGraspSceneEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 4.0
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.episode_length_s = 30.0
    cfg.terminations.time_out = None
    cfg.terminations.excessive_tilt = None
    cfg.terminations.low_base = None
    cfg.terminations.success = None
    finger_material = _configure_finger_material(cfg)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    robot = env.scene["robot"]
    grasp_object = env.scene["grasp_object"]
    arm_ids, _ = robot.find_joints(
        [f"joint{index}" for index in range(1, 7)], preserve_order=True
    )
    finger_ids, _ = robot.find_joints(
        ["joint7", "joint8"], preserve_order=True
    )
    finger_body_ids, _ = robot.find_bodies(
        ["link7", "link8"], preserve_order=True
    )
    arm_ids = torch.as_tensor(
        arm_ids, device=env.device, dtype=torch.long
    ).flatten()
    finger_ids = torch.as_tensor(
        finger_ids, device=env.device, dtype=torch.long
    ).flatten()
    finger_body_ids = torch.as_tensor(
        finger_body_ids, device=env.device, dtype=torch.long
    ).flatten()
    jacobian_body_ids = finger_body_ids.clone()
    jacobians = robot.root_physx_view.get_jacobians()
    if robot.is_fixed_base:
        jacobian_joint_offset = 0
        jacobian_body_ids -= 1
    else:
        jacobian_joint_offset = 6
    expected_width = robot.data.joint_pos.shape[-1] + jacobian_joint_offset
    if jacobians.shape[-1] != expected_width:
        raise RuntimeError(
            "Unexpected Jacobian width: "
            f"{tuple(jacobians.shape)} for "
            f"{robot.data.joint_pos.shape[-1]} joints"
        )
    if int(jacobian_body_ids.max().item()) >= jacobians.shape[1]:
        raise RuntimeError(
            "Resolved finger body index exceeds the Jacobian body dimension: "
            f"{jacobian_body_ids.tolist()} vs {tuple(jacobians.shape)}"
        )
    dls_identity = torch.eye(3, device=env.device).unsqueeze(0)

    env_ids = torch.arange(env.num_envs, device=env.device)
    origins = env.scene.env_origins
    lateral_jitter = torch.linspace(
        -0.006, 0.006, env.num_envs, device=env.device
    )
    root_position = torch.zeros(env.num_envs, 3, device=env.device)
    root_position[:, 0] = origins[:, 0] - 0.044
    root_position[:, 1] = origins[:, 1] - 0.034 + lateral_jitter
    root_position[:, 2] = 0.54
    yaw = torch.full((env.num_envs,), -0.055, device=env.device)
    root_rotation = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    robot.write_root_pose_to_sim(
        torch.cat((root_position, root_rotation), dim=1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
    )

    object_position = torch.tensor(
        OBJECT_START_POSITION, device=env.device
    ).view(1, 3).repeat(env.num_envs, 1)
    object_position += origins
    object_rotation = torch.zeros(env.num_envs, 4, device=env.device)
    object_rotation[:, 0] = 1.0
    grasp_object.write_root_pose_to_sim(
        torch.cat((object_position, object_rotation), dim=1), env_ids=env_ids
    )
    grasp_object.write_root_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
    )

    joint_position = robot.data.default_joint_pos.clone()
    joint_velocity = torch.zeros_like(joint_position)
    initial_arm, initial_gripper = _interpolate(0, env.device)
    joint_position[:, arm_ids] = initial_arm
    joint_position[:, finger_ids] = initial_gripper
    robot.write_joint_state_to_sim(
        joint_position, joint_velocity, env_ids=env_ids
    )
    robot.set_joint_position_target(joint_position)

    actions = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    first_bilateral = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    first_lift = torch.full_like(first_bilateral, -1)
    bilateral_streak = torch.zeros_like(first_bilateral)
    max_bilateral_streak = torch.zeros_like(first_bilateral)
    max_lift = torch.zeros(env.num_envs, device=env.device)
    max_slip = torch.zeros_like(max_lift)
    grasp_relation = torch.zeros(env.num_envs, 3, device=env.device)
    relation_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    retention_latched = torch.zeros_like(relation_valid)
    retention_relation = torch.zeros(
        env.num_envs, 3, device=env.device
    )
    retention_gripper_target = initial_gripper.view(1, 2).repeat(
        env.num_envs, 1
    )
    retention_arm_anchor = initial_arm.view(1, 6).repeat(
        env.num_envs, 1
    )
    retention_reference_anchor = retention_arm_anchor.clone()
    alignment_error = torch.zeros(env.num_envs, device=env.device)
    reference_step = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    retention_reference_step = torch.zeros_like(reference_step)
    max_tilt = torch.zeros_like(max_lift)
    min_support = torch.full_like(max_lift, 4.0)
    rows: list[dict[str, float | int]] = []
    total_steps = (
        GRASP_KEYFRAMES[-1][0]
        + args.hold_steps
        + (120 if args.event_gated_progress else 0)
    )

    for step in range(total_steps):
        if args.event_gated_progress:
            trajectory_steps = reference_step.detach().cpu().tolist()
        else:
            reference_step.fill_(
                min(step, GRASP_KEYFRAMES[-1][0])
            )
            trajectory_steps = [step] * env.num_envs
        references = [
            _interpolate(int(index), env.device)
            for index in trajectory_steps
        ]
        arm_target = torch.stack([item[0] for item in references])
        reference_arm_target = arm_target.clone()
        gripper_target = torch.stack([item[1] for item in references])
        if args.retention_latch:
            # Continue the post-contact trajectory from the actual acquired
            # pose, then smoothly remove the acquisition bias.  Returning
            # directly to the nominal replay tears a valid grasp apart, while
            # retaining the bias forever prevents the full lift trajectory.
            transfer_elapsed = (
                reference_step - retention_reference_step
            ).clamp_min(0).float()
            transfer_blend = (
                1.0 - transfer_elapsed / float(RETENTION_TRANSFER_STEPS)
            ).clamp(0.0, 1.0)
            acquisition_bias = (
                retention_arm_anchor - retention_reference_anchor
            )
            continuous_target = reference_arm_target + (
                transfer_blend.unsqueeze(1) * acquisition_bias
            )
            arm_target = torch.where(
                retention_latched.unsqueeze(1),
                continuous_target,
                arm_target,
            )
        alignment_active = reference_step >= RELATION_KEYFRAMES[0][0]
        if not args.event_gated_progress:
            alignment_active.fill_(
                step >= RELATION_KEYFRAMES[0][0]
            )
        if args.closed_loop_alignment and torch.any(alignment_active):
            object_pos = grasp_object.data.root_pos_w
            pad_position, pad_offset = _finger_pad_geometry(
                robot, finger_body_ids
            )
            gripper_center = pad_position.mean(dim=1)
            relation_reference = torch.stack(
                [
                    _interpolate_relation(int(index), env.device)
                    for index in trajectory_steps
                ]
            )
            desired_relation = torch.where(
                retention_latched.unsqueeze(1),
                retention_relation,
                relation_reference,
            )
            position_error = object_pos - desired_relation - gripper_center
            position_error = torch.where(
                alignment_active.unsqueeze(1),
                position_error,
                torch.zeros_like(position_error),
            )
            alignment_error[:] = torch.where(
                alignment_active,
                torch.linalg.vector_norm(position_error, dim=1),
                torch.zeros_like(alignment_error),
            )
            jacobians = robot.root_physx_view.get_jacobians()
            pad_jacobians = []
            for side in range(2):
                body_jacobian = jacobians[
                    :,
                    int(jacobian_body_ids[side].item()),
                    :6,
                    arm_ids + jacobian_joint_offset,
                ]
                linear = body_jacobian[:, :3]
                angular = body_jacobian[:, 3:6]
                pad_jacobians.append(
                    linear
                    - torch.bmm(_skew(pad_offset[:, side]), angular)
                )
            center_jacobian = 0.5 * (
                pad_jacobians[0] + pad_jacobians[1]
            )
            transpose = center_jacobian.transpose(1, 2)
            system = torch.bmm(center_jacobian, transpose) + (
                0.035**2
            ) * dls_identity
            joint_delta = torch.bmm(
                transpose,
                torch.linalg.solve(
                    system, 0.70 * position_error.unsqueeze(-1)
                ),
            ).squeeze(-1)
            joint_delta.clamp_(-0.10, 0.10)
            # A stalled acquire phase needs an integrating Cartesian servo;
            # repeatedly adding the correction to a fixed reference leaves a
            # persistent pad-center error.  Once contact is latched, return to
            # the full lift reference with only a bounded correction.
            iterative_ingress = (
                args.iterative_ingress
                & args.event_gated_progress
                & (reference_step >= 124)
                & ~retention_latched
            )
            corrected_reference = arm_target + 0.55 * joint_delta
            current_arm = robot.data.joint_pos[:, arm_ids]
            ingress_target = current_arm + 0.35 * joint_delta
            arm_target = torch.where(
                iterative_ingress.unsqueeze(1),
                ingress_target,
                corrected_reference,
            )
            limits = robot.data.soft_joint_pos_limits[:, arm_ids]
            arm_target = torch.maximum(
                torch.minimum(arm_target, limits[:, :, 1]),
                limits[:, :, 0],
            )
        else:
            alignment_error.zero_()
        if args.retention_latch:
            gripper_target = torch.where(
                retention_latched.unsqueeze(1),
                retention_gripper_target,
                gripper_target,
            )
        robot.set_joint_position_target(
            arm_target,
            joint_ids=arm_ids,
        )
        robot.set_joint_position_target(
            gripper_target,
            joint_ids=finger_ids,
        )
        _, _, terminated, truncated, _ = env.step(actions)

        left_force = _force(env.scene["left_finger_object_contact"])
        right_force = _force(env.scene["right_finger_object_contact"])
        bilateral = (left_force >= 0.20) & (right_force >= 0.20)
        force_balance = torch.minimum(left_force, right_force) / torch.maximum(
            torch.maximum(left_force, right_force),
            torch.full_like(left_force, 1.0e-6),
        )
        bilateral_streak = torch.where(
            bilateral, bilateral_streak + 1, torch.zeros_like(bilateral_streak)
        )
        max_bilateral_streak = torch.maximum(
            max_bilateral_streak, bilateral_streak
        )
        new_contact = (first_bilateral < 0) & (bilateral_streak >= 3)
        first_bilateral[new_contact] = step - 2

        object_pos = grasp_object.data.root_pos_w
        lift = object_pos[:, 2] - float(OBJECT_START_POSITION[2])
        max_lift = torch.maximum(max_lift, lift)
        new_lift = (first_lift < 0) & (lift >= 0.02) & bilateral
        first_lift[new_lift] = step
        pad_position, _ = _finger_pad_geometry(robot, finger_body_ids)
        gripper_center = pad_position.mean(dim=1)
        relation = object_pos - gripper_center
        if args.event_gated_progress:
            retention_phase_ready = reference_step >= 118
        else:
            retention_phase_ready = torch.full_like(
                retention_latched, step >= 118
            )
        new_retention = (
            args.retention_latch
            & (bilateral_streak >= RETENTION_CONTACT_STREAK)
            & (force_balance >= RETENTION_FORCE_BALANCE)
            & (alignment_error <= 0.040)
            & retention_phase_ready
            & ~retention_latched
        )
        if torch.any(new_retention):
            retention_latched[new_retention] = True
            retention_relation[new_retention] = relation[new_retention]
            current_gripper = robot.data.joint_pos[:, finger_ids]
            current_arm = robot.data.joint_pos[:, arm_ids]
            retention_arm_anchor[new_retention] = current_arm[new_retention]
            retention_reference_anchor[new_retention] = (
                reference_arm_target[new_retention]
            )
            retention_reference_step[new_retention] = reference_step[
                new_retention
            ]
            retention_gripper_target[new_retention, 0] = torch.clamp(
                current_gripper[new_retention, 0] - 0.004,
                min=0.002,
            )
            retention_gripper_target[new_retention, 1] = torch.clamp(
                current_gripper[new_retention, 1] + 0.004,
                max=-0.002,
            )
        retained_contact = retention_latched & ~new_retention
        tighten_left = retained_contact & (
            left_force < RETENTION_FORCE_TARGET_N
        )
        tighten_right = retained_contact & (
            right_force < RETENTION_FORCE_TARGET_N
        )
        retention_gripper_target[:, 0] = torch.where(
            tighten_left,
            torch.clamp(
                retention_gripper_target[:, 0] - RETENTION_GRIP_STEP_M,
                min=0.002,
            ),
            retention_gripper_target[:, 0],
        )
        retention_gripper_target[:, 1] = torch.where(
            tighten_right,
            torch.clamp(
                retention_gripper_target[:, 1] + RETENTION_GRIP_STEP_M,
                max=-0.002,
            ),
            retention_gripper_target[:, 1],
        )
        set_relation = (
            new_retention
            if args.retention_latch
            else bilateral & ~relation_valid
        )
        grasp_relation[set_relation] = relation[set_relation]
        relation_valid |= set_relation
        slip = torch.linalg.vector_norm(relation - grasp_relation, dim=1)
        max_slip = torch.maximum(
            max_slip, torch.where(relation_valid, slip, torch.zeros_like(slip))
        )

        gravity = robot.data.projected_gravity_b
        tilt = torch.asin(
            torch.sqrt(gravity[:, 0].square() + gravity[:, 1].square()).clamp(0.0, 1.0)
        )
        max_tilt = torch.maximum(max_tilt, tilt)
        support = torch.stack(
            [
                _force(env.scene[name]) > 1.0
                for name in (
                    "FL_foot_contact",
                    "FR_foot_contact",
                    "RL_foot_contact",
                    "RR_foot_contact",
                )
            ],
            dim=1,
        ).sum(dim=1)
        min_support = torch.minimum(min_support, support.float())

        if args.event_gated_progress:
            can_advance = (reference_step < 124) | retention_latched
            reference_step[:] = torch.where(
                can_advance,
                (reference_step + 1).clamp_max(
                    GRASP_KEYFRAMES[-1][0]
                ),
                reference_step,
            )

        if step % 2 == 0 or torch.any(new_contact) or torch.any(new_lift):
            applied = robot.data.applied_torque[:, finger_ids]
            for env_id in range(env.num_envs):
                rows.append(
                    {
                        "step": step,
                        "env_id": env_id,
                        "left_force_n": float(left_force[env_id].item()),
                        "right_force_n": float(right_force[env_id].item()),
                        "contact_force_balance": float(
                            force_balance[env_id].item()
                        ),
                        "left_friction_force_n": float(
                            _friction_force(
                                env.scene["left_finger_object_contact"],
                                env.num_envs,
                                env.device,
                            )[env_id].item()
                        ),
                        "right_friction_force_n": float(
                            _friction_force(
                                env.scene["right_finger_object_contact"],
                                env.num_envs,
                                env.device,
                            )[env_id].item()
                        ),
                        "left_joint_torque_nm": float(applied[env_id, 0].item()),
                        "right_joint_torque_nm": float(applied[env_id, 1].item()),
                        "left_joint_position": float(
                            robot.data.joint_pos[env_id, finger_ids[0]].item()
                        ),
                        "right_joint_position": float(
                            robot.data.joint_pos[env_id, finger_ids[1]].item()
                        ),
                        "left_joint_target": float(
                            retention_gripper_target[env_id, 0].item()
                        ),
                        "right_joint_target": float(
                            retention_gripper_target[env_id, 1].item()
                        ),
                        "object_lift_m": float(lift[env_id].item()),
                        "object_gripper_slip_m": float(slip[env_id].item()),
                        "object_relative_x_m": float(
                            relation[env_id, 0].item()
                        ),
                        "object_relative_y_m": float(
                            relation[env_id, 1].item()
                        ),
                        "object_relative_z_m": float(
                            relation[env_id, 2].item()
                        ),
                        "alignment_error_m": float(
                            alignment_error[env_id].item()
                        ),
                        "reference_step": int(
                            reference_step[env_id].item()
                        ),
                        "retention_latched": int(
                            retention_latched[env_id].item()
                        ),
                        "base_tilt_rad": float(tilt[env_id].item()),
                        "support_count": int(support[env_id].item()),
                        "terminated": int(terminated[env_id].item()),
                        "truncated": int(truncated[env_id].item()),
                    }
                )

        if step % 50 == 0:
            print(
                "grasp_replay "
                f"step={step}/{total_steps} "
                f"bilateral={int(bilateral.sum().item())}/{env.num_envs} "
                f"max_lift={float(max_lift.max().item()):.4f} "
                f"reference={int(reference_step.min().item())}-"
                f"{int(reference_step.max().item())}",
                flush=True,
            )

    final_object = grasp_object.data.root_pos_w
    final_lift = final_object[:, 2] - float(OBJECT_START_POSITION[2])
    passed = (
        (first_bilateral >= 0)
        & (first_lift >= 0)
        & (max_bilateral_streak >= 12)
        & (final_lift >= 0.02)
        & (max_slip <= 0.045)
        & (max_tilt <= 0.45)
        & (min_support >= 2.0)
    )
    summary = {
        "gate": "fixed-basin physical grasp calibration",
        "counts_as_policy_evaluation": False,
        "explicit_finger_material": finger_material is not None,
        "finger_static_friction": finger_material[0]
        if finger_material is not None
        else None,
        "finger_dynamic_friction": finger_material[1]
        if finger_material is not None
        else None,
        "closed_loop_alignment": args.closed_loop_alignment,
        "retention_latch": args.retention_latch,
        "event_gated_progress": args.event_gated_progress,
        "iterative_ingress": args.iterative_ingress,
        "resolved_finger_body_ids": finger_body_ids.cpu().tolist(),
        "jacobian_finger_body_ids": jacobian_body_ids.cpu().tolist(),
        "is_fixed_base": robot.is_fixed_base,
        "num_robot_bodies": len(robot.data.body_names),
        "jacobian_shape": list(jacobians.shape),
        "jacobian_joint_offset": jacobian_joint_offset,
        "retention_contact_streak": RETENTION_CONTACT_STREAK,
        "retention_force_balance": RETENTION_FORCE_BALANCE,
        "retention_transfer_steps": RETENTION_TRANSFER_STEPS,
        "retention_force_target_n": RETENTION_FORCE_TARGET_N,
        "retention_grip_step_m": RETENTION_GRIP_STEP_M,
        "num_envs": env.num_envs,
        "passed_envs": int(passed.sum().item()),
        "pass_rate": float(passed.float().mean().item()),
        "first_bilateral_step": first_bilateral.cpu().tolist(),
        "first_lift_step": first_lift.cpu().tolist(),
        "max_bilateral_streak": max_bilateral_streak.cpu().tolist(),
        "max_lift_m": max_lift.cpu().tolist(),
        "final_lift_m": final_lift.cpu().tolist(),
        "max_object_gripper_slip_m": max_slip.cpu().tolist(),
        "max_base_tilt_rad": max_tilt.cpu().tolist(),
        "min_support_count": min_support.cpu().tolist(),
        "final_reference_step": reference_step.cpu().tolist(),
        "source_platform_position": SOURCE_PLATFORM_POSITION,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "grasp_debug.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    env.close()


try:
    main()
finally:
    simulation_app.close()
