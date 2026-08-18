"""Short fixed-horizon command response probe for a ZYB checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--analytic_wheels", action="store_true")
parser.add_argument("--analytic_gain", type=float, default=1.0)
parser.add_argument("--only", type=str, default=None)
parser.add_argument("--fixed_reset", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from quadruped_arm.tasks.manager_based.maniploco.agents.rsl_rl_ppo_cfg import ManiPLocoPPORunnerCfg
from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg


COMMANDS = (
    ("forward", 0.35, 0.0, 0.0),
    ("reverse", -0.25, 0.0, 0.0),
    ("yaw_left", 0.0, 0.0, 0.45),
    ("yaw_right", 0.0, 0.0, -0.45),
    ("stationary", 0.0, 0.0, 0.0),
)

if args.only is not None:
    COMMANDS = tuple(row for row in COMMANDS if row[0] == args.only)
    if not COMMANDS:
        raise ValueError(f"Unknown command name: {args.only}")


def set_command(term, command: torch.Tensor) -> None:
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    term.vel_command_b[:] = command


def main() -> None:
    num_envs = len(COMMANDS)
    cfg = ManipLocoEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    cfg.seed = 1042
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    if args.fixed_reset:
        cfg.events.reset_root.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
        cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    runner_cfg = ManiPLocoPPORunnerCfg()
    runner_cfg.device = args.device
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    policy_module = runner.alg.policy
    term = env.command_manager.get_term("locomotion")
    command = torch.tensor([(vx, vy, wz) for _, vx, vy, wz in COMMANDS], device=env.device)

    vx_sum = torch.zeros(num_envs, device=env.device)
    wz_sum = torch.zeros(num_envs, device=env.device)
    min_z = torch.full((num_envs,), float("inf"), device=env.device)
    max_tilt = torch.zeros(num_envs, device=env.device)
    bad = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    low = torch.zeros_like(bad)
    tilt_hit = torch.zeros_like(bad)

    for _ in range(args.steps):
        set_command(term, command)
        observations = wrapped.get_observations()
        with torch.inference_mode():
            actions = policy(observations)
        if args.analytic_wheels:
            radius = 0.11
            track = 0.50
            scale = 4.0
            left = (command[:, 0] - 0.5 * track * command[:, 2]) / radius
            right = (command[:, 0] + 0.5 * track * command[:, 2]) / radius
            actions[:, 12:16] = (
                torch.stack((left, right, left, right), dim=-1)
                .div(scale)
                .mul(args.analytic_gain)
                .clamp(-1.0, 1.0)
            )
        _, _, dones, _ = wrapped.step(actions)
        policy_module.reset(dones)
        robot = env.scene["robot"]
        vx_sum += robot.data.root_lin_vel_b[:, 0]
        wz_sum += robot.data.root_ang_vel_b[:, 2]
        min_z = torch.minimum(min_z, robot.data.root_pos_w[:, 2])
        gravity = robot.data.projected_gravity_b
        tilt = torch.asin(torch.linalg.vector_norm(gravity[:, :2], dim=1).clamp(0.0, 1.0))
        max_tilt = torch.maximum(max_tilt, tilt)
        bad |= env.termination_manager.get_term("bad_contact")
        low |= env.termination_manager.get_term("low_height")
        tilt_hit |= env.termination_manager.get_term("tilt")

    result = {"checkpoint": str(args.checkpoint), "steps": args.steps, "analytic_wheels": args.analytic_wheels, "commands": {}}
    for i, (name, *_cmd) in enumerate(COMMANDS):
        result["commands"][name] = {
            "mean_body_vx_mps": float((vx_sum[i] / args.steps).item()),
            "mean_body_wz_radps": float((wz_sum[i] / args.steps).item()),
            "min_height_m": float(min_z[i].item()),
            "max_tilt_rad": float(max_tilt[i].item()),
            "bad_contact": bool(bad[i].item()),
            "low_height": bool(low[i].item()),
            "tilt": bool(tilt_hit[i].item()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    wrapped.close()


try:
    main()
finally:
    simulation_app.close()
