"""Smoke test for scene assets, gripper calibration, and four-wheel support."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="ZYB-Real-Grasp-Scene-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import quadruped_arm.tasks  # noqa: F401
from quadruped_arm.tasks.manager_based.zyb_real_grasp.mdp.observations import support_metrics


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.decimation * cfg.sim.dt + 2.0)
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_object = env.unwrapped.scene["grasp_object"].data.root_pos_w.clone()
    max_tilt = torch.zeros(args.num_envs, device=env.unwrapped.device)
    min_height = torch.full_like(max_tilt, float("inf"))
    min_support = torch.full_like(max_tilt, 4.0)
    min_rear_support = torch.full_like(max_tilt, 2.0)
    terminations = torch.zeros_like(max_tilt)
    warmup_steps = min(30, max(1, args.steps // 4))

    for step in range(args.steps):
        _, _, terminated, _, _ = env.step(action)
        metrics = support_metrics(env.unwrapped)
        max_tilt = torch.maximum(max_tilt, metrics["tilt"])
        min_height = torch.minimum(min_height, metrics["base_height"])
        terminations += terminated.float()
        if step >= warmup_steps:
            min_support = torch.minimum(min_support, metrics["support_count"])
            min_rear_support = torch.minimum(min_rear_support, metrics["rear_support_count"])

    final_object = env.unwrapped.scene["grasp_object"].data.root_pos_w
    robot_cfg = env.unwrapped.cfg.scene.robot
    result = {
        "task": args.task,
        "num_envs": args.num_envs,
        "support_warmup_steps": warmup_steps,
        "action_dim": int(env.action_space.shape[-1]),
        "physical_terminations": int(terminations.sum().item()),
        "failed_env_rate": float((terminations > 0).float().mean().item()),
        "p95_max_tilt": float(torch.quantile(max_tilt, 0.95).item()),
        "minimum_base_height": float(min_height.min().item()),
        "minimum_support_count": float(min_support.min().item()),
        "rear_support_loss_rate": float((min_rear_support < 2.0).float().mean().item()),
        "mean_object_drift_m": float(
            torch.linalg.vector_norm(final_object - initial_object, dim=1).mean().item()
        ),
        "gripper_stiffness": float(robot_cfg.actuators["gripper"].stiffness),
        "gripper_damping": float(robot_cfg.actuators["gripper"].damping),
        "wheel_effort_limit": float(robot_cfg.actuators["wheels"].effort_limit),
        "wheel_damping": float(robot_cfg.actuators["wheels"].damping),
    }
    print("ZYB_REAL_GRASP_SCENE_PROBE=" + json.dumps(result, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
