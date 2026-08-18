"""Identify stable B2W wheel motion bases under the support WBC."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--magnitude", type=float, default=60.0)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--motion_steps", type=int, default=180)
parser.add_argument("--ramp_steps", type=int, default=60)
parser.add_argument("--turn_support_shift_m", type=float, default=0.0)
parser.add_argument(
    "--selected_patterns",
    type=str,
    default="",
    help="Semicolon-separated FL,FR,RL,RR patterns",
)
parser.add_argument("--replicates", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.TANDEM_HRL.controllers import (
    PayloadAwareSupportWBC,
)
from quadruped_arm.tasks.manager_based.TANDEM_HRL.physical_scene_env_cfg import (
    TANDEMPhysicalSceneEnvCfg,
)


SIGN_PATTERNS = list(itertools.product((-1.0, 1.0), repeat=4))
SPARSE_PATTERNS = [
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 1.0),
    (-1.0, 0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0, -1.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
]
PATTERNS = SIGN_PATTERNS + SPARSE_PATTERNS


def _force(sensor) -> torch.Tensor:
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(
        forces.reshape(forces.shape[0], -1, 3), dim=-1
    ).amax(dim=1)


def main() -> None:
    if args.selected_patterns:
        selected = []
        for encoded in args.selected_patterns.split(";"):
            pattern = tuple(float(item) for item in encoded.split(","))
            if len(pattern) != 4:
                raise ValueError("Each selected wheel pattern needs 4 values")
            selected.append(pattern)
        patterns_list = [
            pattern for pattern in selected for _ in range(args.replicates)
        ]
    else:
        patterns_list = PATTERNS
    cfg = TANDEMPhysicalSceneEnvCfg()
    cfg.scene.num_envs = len(patterns_list)
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

    zero_actions = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(args.settle_steps):
        env.step(zero_actions)
    wbc = PayloadAwareSupportWBC(
        env, max_turn_support_shift_m=args.turn_support_shift_m
    )
    wbc.reset_reference()

    patterns = torch.tensor(patterns_list, device=env.device)
    initial_xy = robot.data.root_pos_w[:, :2].clone()
    mean_velocity = torch.zeros(env.num_envs, 3, device=env.device)
    max_tilt = torch.zeros(env.num_envs, device=env.device)
    min_height = torch.full_like(max_tilt, float("inf"))
    min_support = torch.full_like(max_tilt, 4.0)
    for step in range(args.motion_steps):
        ramp = min(1.0, float(step + 1) / float(args.ramp_steps))
        wheel_target = args.magnitude * ramp * patterns
        actions, _ = wbc.compute(wheel_target)
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
        mean_velocity[:, 0] += robot.data.root_lin_vel_b[:, 0]
        mean_velocity[:, 1] += robot.data.root_lin_vel_b[:, 1]
        mean_velocity[:, 2] += robot.data.root_ang_vel_b[:, 2]
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, robot.data.root_pos_w[:, 2])
        min_support = torch.minimum(min_support, support.float())

    mean_velocity /= float(args.motion_steps)
    displacement = robot.data.root_pos_w[:, :2] - initial_xy
    results = []
    for index, pattern in enumerate(patterns_list):
        stable = bool(
            max_tilt[index] <= 0.45
            and min_height[index] >= 0.30
            and min_support[index] >= 2.0
        )
        results.append(
            {
                "pattern_id": index,
                "wheel_FL_FR_RL_RR": list(pattern),
                "mean_body_vx_mps": float(mean_velocity[index, 0].item()),
                "mean_body_vy_mps": float(mean_velocity[index, 1].item()),
                "mean_body_wz_radps": float(mean_velocity[index, 2].item()),
                "world_displacement_xy_m": displacement[index].cpu().tolist(),
                "max_tilt_rad": float(max_tilt[index].item()),
                "min_base_height_m": float(min_height[index].item()),
                "min_support_count": float(min_support[index].item()),
                "stable": stable,
            }
        )
    summary = {
        "gate": "support-WBC wheel-basis identification",
        "counts_as_hrl_evaluation": False,
        "raw_action_magnitude": args.magnitude,
        "joint_velocity_target_magnitude_radps": 0.1 * args.magnitude,
        "ramp_steps": args.ramp_steps,
        "turn_support_shift_m": args.turn_support_shift_m,
        "motion_steps": args.motion_steps,
        "patterns": results,
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
