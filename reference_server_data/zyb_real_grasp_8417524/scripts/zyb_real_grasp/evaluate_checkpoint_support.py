"""Replay a 16-D ZYB-v0 checkpoint and report leg/support behavior."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="ZYB-Real-Grasp-Scene-v0")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--envs_per_command", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import quadruped_arm.tasks  # noqa: F401


FOOT_SENSORS = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)


def _load(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.reshape(forces.shape[0], -1, 3), dim=-1).amax(dim=1)


def _support_metrics(env) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    loads = torch.stack([_load(env, name) for name in FOOT_SENSORS], dim=1)
    return {
        "support_count": (loads > 5.0).float().sum(dim=1),
        "rear_support_count": (loads[:, 2:] > 5.0).float().sum(dim=1),
        "tilt": torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1),
    }


def main() -> None:
    names = ("stationary", "forward", "yaw_left", "yaw_right")
    num_envs = len(names) * args.envs_per_command
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs, use_fabric=True)
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, args.steps * env_cfg.decimation * env_cfg.sim.dt + 2.0
    )
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command_values = torch.tensor(
        ((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), (0.0, 0.0, 0.45), (0.0, 0.0, -0.45)),
        device=env.unwrapped.device,
    )
    commands = command_values.repeat_interleave(args.envs_per_command, dim=0)
    max_tilt = torch.zeros(num_envs, device=env.unwrapped.device)
    min_support = torch.full_like(max_tilt, 4.0)
    min_rear_support = torch.full_like(max_tilt, 2.0)
    velocity_sum = torch.zeros(num_envs, 3, device=env.unwrapped.device)
    done_count = torch.zeros(num_envs, device=env.unwrapped.device)
    sample_count = 0

    for step in range(args.steps):
        env.unwrapped.command_manager.get_term("locomotion").vel_command_b.copy_(commands)
        observations = env.get_observations()
        with torch.inference_mode():
            actions = policy(observations)
            _, _, dones, _ = env.step(actions)
        metrics = _support_metrics(env.unwrapped)
        max_tilt = torch.maximum(max_tilt, metrics["tilt"])
        done_count += dones.float()
        if step >= 30:
            min_support = torch.minimum(min_support, metrics["support_count"])
            min_rear_support = torch.minimum(min_rear_support, metrics["rear_support_count"])
        if step >= args.steps // 2:
            robot = env.unwrapped.scene["robot"]
            velocity_sum[:, :2] += robot.data.root_lin_vel_b[:, :2]
            velocity_sum[:, 2] += robot.data.root_ang_vel_b[:, 2]
            sample_count += 1

    mean_velocity = velocity_sum / max(sample_count, 1)
    result = {"task": args.task, "commands": {}}
    for index, name in enumerate(names):
        rows = slice(index * args.envs_per_command, (index + 1) * args.envs_per_command)
        result["commands"][name] = {
            "mean_vx": float(mean_velocity[rows, 0].mean().item()),
            "mean_vy": float(mean_velocity[rows, 1].mean().item()),
            "mean_wz": float(mean_velocity[rows, 2].mean().item()),
            "done_count": int(done_count[rows].sum().item()),
            "failed_env_rate": float((done_count[rows] > 0).float().mean().item()),
            "p95_max_tilt": float(torch.quantile(max_tilt[rows], 0.95).item()),
            "minimum_support_count": float(min_support[rows].min().item()),
            "rear_support_loss_rate": float(
                (min_rear_support[rows] < 2.0).float().mean().item()
            ),
        }
    print("ZYB_REAL_GRASP_CHECKPOINT_SUPPORT=" + json.dumps(result, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
