"""Evaluate a Jacobian support controller with direct wheel motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--drive_steps", type=int, default=240)
parser.add_argument("--wheel_action", type=float, default=60.0)
parser.add_argument(
    "--wheel_signs",
    type=str,
    default="1,1,1,1",
    help="FL,FR,RL,RR wheel signs",
)
parser.add_argument("--ramp_steps", type=int, default=60)
parser.add_argument("--support_gain", type=float, default=0.55)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.TANDEM_HRL.physical_scene_env_cfg import (
    TANDEMPhysicalSceneEnvCfg,
)


FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEG_JOINT_NAMES = (
    ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
)


def _force(sensor) -> torch.Tensor:
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(
        forces.reshape(forces.shape[0], -1, 3), dim=-1
    ).amax(dim=1)


def _to_base_frame(
    root_quaternion: torch.Tensor, vectors_w: torch.Tensor
) -> torch.Tensor:
    shape = vectors_w.shape
    rotations = root_quaternion.unsqueeze(1).expand(
        -1, shape[1], -1
    )
    return math_utils.quat_apply_inverse(
        rotations.reshape(-1, 4), vectors_w.reshape(-1, 3)
    ).reshape(shape)


def main() -> None:
    wheel_signs = tuple(float(item) for item in args.wheel_signs.split(","))
    if len(wheel_signs) != 4:
        raise ValueError("wheel_signs must contain FL,FR,RL,RR")
    cfg = TANDEMPhysicalSceneEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 4.0
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.episode_length_s = 60.0
    cfg.terminations.time_out = None
    cfg.terminations.bad_contact = None
    cfg.terminations.tilt = None
    cfg.terminations.low_height = None
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    root_position = env.scene.env_origins.clone()
    root_position[:, 1] -= 0.65
    root_position[:, 2] = 0.54
    root_rotation = torch.zeros(env.num_envs, 4, device=env.device)
    root_rotation[:, 0] = 1.0
    robot.write_root_pose_to_sim(
        torch.cat((root_position, root_rotation), dim=1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
    )

    leg_term = env.action_manager.get_term("leg_pos")
    action_leg_ids = torch.as_tensor(
        leg_term._joint_ids, device=env.device, dtype=torch.long
    ).flatten()
    action_leg_names = [
        robot.data.joint_names[int(index)] for index in action_leg_ids
    ]
    action_scales = leg_term._scale.clone()
    action_offsets = leg_term._offset.clone()
    nominal_leg_target = action_offsets.clone()

    foot_ids = []
    jacobian_body_ids = []
    leg_joint_ids = []
    for foot_name, joint_names in zip(FOOT_NAMES, LEG_JOINT_NAMES):
        resolved_foot, _ = robot.find_bodies([foot_name], preserve_order=True)
        resolved_joints, _ = robot.find_joints(
            list(joint_names), preserve_order=True
        )
        foot_id = int(resolved_foot[0])
        foot_ids.append(foot_id)
        leg_joint_ids.append(
            torch.as_tensor(
                resolved_joints, device=env.device, dtype=torch.long
            ).flatten()
        )
        jacobian_body_ids.append(foot_id)
    foot_ids_tensor = torch.tensor(
        foot_ids, device=env.device, dtype=torch.long
    )
    jacobians = robot.root_physx_view.get_jacobians()
    # Isaac Lab omits the root body/DoFs only for fixed-base articulations.
    # B2W is floating-base, so its body indices stay unchanged and the six
    # floating-base columns precede the actuated-joint columns.
    if robot.is_fixed_base:
        jacobian_joint_offset = 0
        jacobian_body_ids = [index - 1 for index in jacobian_body_ids]
    else:
        jacobian_joint_offset = 6
    expected_width = robot.data.joint_pos.shape[-1] + jacobian_joint_offset
    if jacobians.shape[-1] != expected_width:
        raise RuntimeError(
            "Unexpected Jacobian width: "
            f"{tuple(jacobians.shape)} for "
            f"{robot.data.joint_pos.shape[-1]} joints"
        )
    if max(jacobian_body_ids) >= jacobians.shape[1]:
        raise RuntimeError(
            "Resolved foot body index exceeds the Jacobian body dimension: "
            f"{jacobian_body_ids} vs {tuple(jacobians.shape)}"
        )

    actions = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    wheel_pattern = torch.tensor(
        wheel_signs, device=env.device, dtype=actions.dtype
    ).view(1, 4)
    for _ in range(args.settle_steps):
        env.step(actions)

    root_quaternion = robot.data.root_quat_w
    foot_relative_w = (
        robot.data.body_pos_w[:, foot_ids_tensor]
        - robot.data.root_pos_w.unsqueeze(1)
    )
    desired_foot_position_b = _to_base_frame(
        root_quaternion, foot_relative_w
    )
    desired_height = robot.data.root_pos_w[:, 2].clone()
    initial_xy = robot.data.root_pos_w[:, :2].clone()

    max_tilt = torch.zeros(env.num_envs, device=env.device)
    min_height = torch.full_like(max_tilt, float("inf"))
    min_support = torch.full_like(max_tilt, 4.0)
    mean_velocity = torch.zeros(env.num_envs, 3, device=env.device)
    max_leg_correction = torch.zeros_like(max_tilt)
    dls_identity = torch.eye(3, device=env.device).unsqueeze(0)

    for step in range(args.drive_steps):
        root_quaternion = robot.data.root_quat_w
        current_relative_w = (
            robot.data.body_pos_w[:, foot_ids_tensor]
            - robot.data.root_pos_w.unsqueeze(1)
        )
        current_foot_position_b = _to_base_frame(
            root_quaternion, current_relative_w
        )
        position_error = desired_foot_position_b - current_foot_position_b
        height_error = desired_height - robot.data.root_pos_w[:, 2]
        position_error[:, :, 2] -= height_error.unsqueeze(1)

        joint_correction = torch.zeros_like(robot.data.joint_pos)
        jacobians = robot.root_physx_view.get_jacobians()
        for leg_index in range(4):
            joint_ids = leg_joint_ids[leg_index]
            jacobian_joint_ids = joint_ids + jacobian_joint_offset
            jacobian_w = jacobians[
                :, jacobian_body_ids[leg_index], :3, jacobian_joint_ids
            ]
            jacobian_columns_w = jacobian_w.transpose(1, 2)
            jacobian_columns_b = _to_base_frame(
                root_quaternion, jacobian_columns_w
            )
            jacobian_b = jacobian_columns_b.transpose(1, 2)
            transpose = jacobian_b.transpose(1, 2)
            system = torch.bmm(jacobian_b, transpose) + (
                0.045**2
            ) * dls_identity
            delta = torch.bmm(
                transpose,
                torch.linalg.solve(
                    system,
                    args.support_gain
                    * position_error[:, leg_index].unsqueeze(-1),
                ),
            ).squeeze(-1)
            delta.clamp_(-0.08, 0.08)
            joint_correction[:, joint_ids] = delta
            max_leg_correction = torch.maximum(
                max_leg_correction,
                torch.linalg.vector_norm(delta, dim=1),
            )

        limits = robot.data.soft_joint_pos_limits[:, action_leg_ids]
        action_joint_target = torch.maximum(
            torch.minimum(
                nominal_leg_target
                + joint_correction[:, action_leg_ids],
                limits[:, :, 1],
            ),
            limits[:, :, 0],
        )
        actions[:, :12] = (
            action_joint_target - action_offsets
        ) / action_scales
        ramp = (
            min(1.0, float(step + 1) / float(args.ramp_steps))
            if args.ramp_steps > 0
            else 1.0
        )
        actions[:, 12:16] = args.wheel_action * ramp * wheel_pattern
        env.step(actions)

        gravity = robot.data.projected_gravity_b
        tilt = torch.asin(
            torch.sqrt(
                gravity[:, 0].square() + gravity[:, 1].square()
            ).clamp(0.0, 1.0)
        )
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
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, robot.data.root_pos_w[:, 2])
        min_support = torch.minimum(min_support, support.float())
        mean_velocity[:, 0] += robot.data.root_lin_vel_b[:, 0]
        mean_velocity[:, 1] += robot.data.root_lin_vel_b[:, 1]
        mean_velocity[:, 2] += robot.data.root_ang_vel_b[:, 2]

    mean_velocity /= float(args.drive_steps)
    displacement = robot.data.root_pos_w[:, :2] - initial_xy
    stability_passed = (
        (max_tilt <= 0.45)
        & (min_height >= 0.30)
        & (min_support >= 2.0)
    )
    if abs(args.wheel_action) > 1.0e-6:
        passed = stability_passed & (mean_velocity[:, 0] >= 0.08)
    else:
        passed = stability_passed
    summary = {
        "gate": "Jacobian support WBC forward motion",
        "counts_as_hrl_evaluation": False,
        "num_envs": env.num_envs,
        "settle_steps": args.settle_steps,
        "drive_steps": args.drive_steps,
        "wheel_action": args.wheel_action,
        "wheel_signs_FL_FR_RL_RR": list(wheel_signs),
        "wheel_velocity_target_radps": 0.1 * args.wheel_action,
        "ramp_steps": args.ramp_steps,
        "support_gain": args.support_gain,
        "resolved_leg_action_order": action_leg_names,
        "resolved_foot_body_names": list(FOOT_NAMES),
        "resolved_foot_body_ids": foot_ids,
        "jacobian_foot_body_ids": jacobian_body_ids,
        "is_fixed_base": robot.is_fixed_base,
        "num_robot_bodies": len(robot.data.body_names),
        "jacobian_shape": list(jacobians.shape),
        "jacobian_joint_offset": jacobian_joint_offset,
        "leg_action_scale": action_scales[0].cpu().tolist(),
        "leg_action_offset": action_offsets[0].cpu().tolist(),
        "passed_envs": int(passed.sum().item()),
        "pass_rate": float(passed.float().mean().item()),
        "mean_body_vx_mps": mean_velocity[:, 0].cpu().tolist(),
        "mean_body_vy_mps": mean_velocity[:, 1].cpu().tolist(),
        "mean_body_wz_radps": mean_velocity[:, 2].cpu().tolist(),
        "world_displacement_xy_m": displacement.cpu().tolist(),
        "max_tilt_rad": max_tilt.cpu().tolist(),
        "min_base_height_m": min_height.cpu().tolist(),
        "min_support_count": min_support.cpu().tolist(),
        "max_leg_correction_norm_rad": max_leg_correction.cpu().tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
