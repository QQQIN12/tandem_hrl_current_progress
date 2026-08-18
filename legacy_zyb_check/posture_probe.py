"""Probe safe leg posture offsets for a future squat command."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg


def set_command(term, command: torch.Tensor) -> None:
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(True)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    term.vel_command_b[:] = command


def main() -> None:
    cfg = ManipLocoEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = 271828
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    env = ManagerBasedRLEnv(cfg=cfg)
    term = env.command_manager.get_term("locomotion")
    zero = torch.zeros(4, 3, device=env.device)
    patterns = {
        "neutral": (0.0, 0.0),
        "thigh_plus_calf_minus": (0.5, -0.5),
        "thigh_minus_calf_plus": (-0.5, 0.5),
        "both_plus": (0.5, 0.5),
    }
    for name, (thigh, calf) in patterns.items():
        env.reset()
        actions = torch.zeros(4, 16, device=env.device)
        actions[:, 1:3] = actions.new_tensor((thigh, calf))
        actions[:, 4:6] = actions.new_tensor((thigh, calf))
        actions[:, 7:9] = actions.new_tensor((thigh, calf))
        actions[:, 10:12] = actions.new_tensor((thigh, calf))
        z_sum = torch.zeros(4, device=env.device)
        tilt_max = torch.zeros(4, device=env.device)
        for _ in range(args.steps):
            set_command(term, zero)
            env.step(actions)
            robot = env.scene["robot"]
            z_sum += robot.data.root_pos_w[:, 2]
            g = robot.data.projected_gravity_b
            tilt = torch.asin(torch.linalg.vector_norm(g[:, :2], dim=1).clamp(0.0, 1.0))
            tilt_max = torch.maximum(tilt_max, tilt)
        print(name, "mean_z=", (z_sum / args.steps).mean().item(), "max_tilt=", tilt_max.max().item(), flush=True)
    env.close()


try:
    main()
finally:
    simulation_app.close()
