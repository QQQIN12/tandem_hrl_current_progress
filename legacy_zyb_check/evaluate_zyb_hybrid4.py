"""Evaluate the archived ZYB-v0 locomotion checkpoint on fixed body commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--group_size", type=int, default=8)
parser.add_argument("--episodes_per_command", type=int, default=16)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--analytic_wheels",
    action="store_true",
    help="Replace the policy wheel actions with a bounded differential-drive command.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from quadruped_arm.tasks.manager_based.maniploco.agents.rsl_rl_ppo_cfg import (
    ManiPLocoPPORunnerCfg,
)
from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ManipLocoEnvCfg,
)


COMMANDS = (
    ("forward", 0.35, 0.0, 0.0),
    ("reverse", -0.25, 0.0, 0.0),
    ("yaw_left", 0.0, 0.0, 0.45),
    ("yaw_right", 0.0, 0.0, -0.45),
    ("forward_arc", 0.25, 0.0, 0.35),
    ("stationary", 0.0, 0.0, 0.0),
)


def _set_command(term, command: torch.Tensor) -> None:
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    term.vel_command_b[:] = command


def _tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    return torch.asin(
        torch.linalg.vector_norm(gravity[:, :2], dim=1).clamp(0.0, 1.0)
    )


def main() -> None:
    num_groups = len(COMMANDS)
    num_envs = num_groups * args.group_size
    env_cfg = ManipLocoEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = 1042
    env_cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)

    runner_cfg = ManiPLocoPPORunnerCfg()
    runner_cfg.device = args.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device
    )
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    policy_module = runner.alg.policy
    command_term = env.command_manager.get_term("locomotion")

    command = torch.zeros(num_envs, 3, device=env.device)
    group_id = torch.arange(num_envs, device=env.device) // args.group_size
    for index, (_, vx, vy, wz) in enumerate(COMMANDS):
        command[group_id == index] = command.new_tensor((vx, vy, wz))

    episode_steps = torch.zeros(num_envs, device=env.device)
    linear_error_sum = torch.zeros_like(episode_steps)
    yaw_error_sum = torch.zeros_like(episode_steps)
    speed_sum = torch.zeros_like(episode_steps)
    yaw_rate_sum = torch.zeros_like(episode_steps)
    max_tilt = torch.zeros_like(episode_steps)
    completed = torch.zeros(num_groups, dtype=torch.long, device=env.device)
    rows: list[dict] = []

    steps = 0
    while torch.any(completed < args.episodes_per_command) and steps < 1800:
        _set_command(command_term, command)
        observations = wrapped.get_observations()
        with torch.inference_mode():
            actions = policy(observations)
        if args.analytic_wheels:
            # The four wheel actions are normalized targets.  The executor
            # scale is 4 rad/s; map body (vx, wz) to FL, FR, RL, RR targets.
            wheel_radius = 0.11
            track_width = 0.50
            wheel_scale = 4.0
            left = (command[:, 0] - 0.5 * track_width * command[:, 2]) / wheel_radius
            right = (command[:, 0] + 0.5 * track_width * command[:, 2]) / wheel_radius
            wheel_action = torch.stack((left, right, left, right), dim=-1) / wheel_scale
            actions[:, 12:16] = wheel_action.clamp(-1.0, 1.0)
        _, _, dones, _ = wrapped.step(actions)
        policy_module.reset(dones)
        robot = env.scene["robot"]
        linear_error = torch.linalg.vector_norm(
            robot.data.root_lin_vel_b[:, :2] - command[:, :2], dim=1
        )
        yaw_error = (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).abs()
        episode_steps += 1.0
        linear_error_sum += linear_error
        yaw_error_sum += yaw_error
        speed_sum += robot.data.root_lin_vel_b[:, 0]
        yaw_rate_sum += robot.data.root_ang_vel_b[:, 2]
        max_tilt = torch.maximum(max_tilt, _tilt(env))

        if torch.any(dones):
            tilted = env.termination_manager.get_term("tilt") & dones
            low = env.termination_manager.get_term("low_height") & dones
            bad = env.termination_manager.get_term("bad_contact") & dones
            for env_id_tensor in dones.nonzero(as_tuple=False).flatten():
                env_id = int(env_id_tensor.item())
                group = int(group_id[env_id].item())
                if completed[group] >= args.episodes_per_command:
                    continue
                count = max(float(episode_steps[env_id].item()), 1.0)
                mean_linear_error = float(linear_error_sum[env_id].item() / count)
                mean_yaw_error = float(yaw_error_sum[env_id].item() / count)
                stable = not bool(
                    tilted[env_id].item()
                    or low[env_id].item()
                    or bad[env_id].item()
                )
                rows.append(
                    {
                        "command": COMMANDS[group][0],
                        "stable": stable,
                        "tracking_pass": stable
                        and mean_linear_error <= 0.18
                        and mean_yaw_error <= 0.25,
                        "mean_linear_error_mps": mean_linear_error,
                        "mean_yaw_error_radps": mean_yaw_error,
                        "mean_body_vx_mps": float(speed_sum[env_id].item() / count),
                        "mean_body_wz_radps": float(yaw_rate_sum[env_id].item() / count),
                        "max_tilt_rad": float(max_tilt[env_id].item()),
                    }
                )
                completed[group] += 1
            done_ids = dones.nonzero(as_tuple=False).flatten()
            episode_steps[done_ids] = 0.0
            linear_error_sum[done_ids] = 0.0
            yaw_error_sum[done_ids] = 0.0
            speed_sum[done_ids] = 0.0
            yaw_rate_sum[done_ids] = 0.0
            max_tilt[done_ids] = 0.0
        steps += 1

    summary = {}
    for name, *_ in COMMANDS:
        selected = [row for row in rows if row["command"] == name]
        summary[name] = {
            "episodes": len(selected),
            "stable_rate": sum(row["stable"] for row in selected) / len(selected),
            "tracking_pass_rate": sum(row["tracking_pass"] for row in selected) / len(selected),
            "mean_linear_error_mps": sum(row["mean_linear_error_mps"] for row in selected) / len(selected),
            "mean_yaw_error_radps": sum(row["mean_yaw_error_radps"] for row in selected) / len(selected),
            "mean_body_vx_mps": sum(row["mean_body_vx_mps"] for row in selected) / len(selected),
            "mean_body_wz_radps": sum(row["mean_body_wz_radps"] for row in selected) / len(selected),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "environment": "ZYB-v0 direct 16-action executor",
        "steps": steps,
        "commands": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    wrapped.close()


try:
    main()
finally:
    simulation_app.close()
