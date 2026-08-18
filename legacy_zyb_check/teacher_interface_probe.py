"""Probe the archived ZYB teacher through the current lower-body action interface.

This is an interface/dynamics diagnostic, not an evaluation of a learned
student.  It compares the archived teacher's raw leg targets, feasible-clipped
targets, and the initial multi-teacher shield on the current MobilityLower
asset.  Wheel outputs are zero because the current wheel action is a bounded
residual around the command-conditioned wheel feed-forward term.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--vx", type=float, default=0.35)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--root_z", type=float, default=0.58)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.mdp.multi_teacher import _FrozenZybTeacher
from quadruped_arm.tasks.manager_based.maniploco.mobility_lower_env_cfg import MobilityLowerEnvCfg


def _policy_obs(value):
    observations = value[0] if isinstance(value, tuple) else value
    if isinstance(observations, dict):
        return observations["policy"]
    if hasattr(observations, "keys") and "policy" in observations.keys():
        return observations["policy"]
    return observations


def _set_command(term, command):
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    term.vel_command_b[:] = command


def _tilt(robot):
    return torch.asin(
        torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1).clamp(0.0, 1.0)
    )


def main() -> None:
    cfg = MobilityLowerEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.render_interval = 100000
    cfg.seed = 424242
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_z)
    cfg.events.reset_root.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)

    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    command_term = env.command_manager.get_term("locomotion")
    teacher = _FrozenZybTeacher(args.teacher_checkpoint, env.device)
    zero_command = torch.zeros(1, 3, device=env.device)
    active_command = torch.tensor([[args.vx, 0.0, args.wz]], device=env.device)
    zero_action = torch.zeros(1, 16, device=env.device)
    leg_term = env.action_manager.get_term("leg_pos")
    leg_scale = leg_term._scale
    feasible_raw = float(cfg.actions.leg_pos.max_policy_residual) / leg_scale.abs().clamp_min(1.0e-6)

    print("teacher_probe_config", {
        "vx": args.vx,
        "wz": args.wz,
        "steps": args.steps,
        "warmup": args.warmup,
        "leg_scale": leg_scale[0].detach().cpu().tolist(),
        "feasible_raw_leg_bound": feasible_raw[0].detach().cpu().tolist(),
    }, flush=True)

    modes = (
        "neutral",
        "teacher_gain_0.05",
        "teacher_gain_0.10",
        "teacher_gain_0.20",
        "teacher_gain_0.30",
        "teacher_raw",
        "teacher_clipped",
        "ensemble_shield",
    )
    for mode in modes:
        env.reset()
        _set_command(command_term, zero_command)
        active_vx = []
        active_wz = []
        active_tilt = []
        active_z = []
        first_done = None
        teacher_max = 0.0
        clipped_fraction = 0.0
        for step in range(args.steps):
            if step < args.warmup:
                _set_command(command_term, zero_command)
                action = zero_action
            else:
                _set_command(command_term, active_command)
                obs = _policy_obs(env.observation_manager.compute())
                with torch.inference_mode():
                    zyb = teacher(obs).to(dtype=zero_action.dtype)
                teacher_max = max(teacher_max, float(zyb[:, :12].abs().max().item()))
                clipped = zyb[:, :12].clamp(-feasible_raw, feasible_raw)
                clipped_fraction += float((zyb[:, :12].abs() > feasible_raw).float().mean().item())
                if mode == "neutral":
                    leg_action = torch.zeros_like(zyb[:, :12])
                elif mode.startswith("teacher_gain_"):
                    gain = float(mode.rsplit("_", 1)[-1])
                    leg_action = zyb[:, :12] * gain
                elif mode == "teacher_raw":
                    leg_action = zyb[:, :12]
                elif mode == "teacher_clipped":
                    leg_action = clipped
                else:
                    # Match the initial shield in MultiTeacherVecEnv at a
                    # settled upright state: ZYB + half-amplitude conservative
                    # candidate + neutral recovery, then alpha=0.95.
                    tilt_now = _tilt(robot)
                    height_now = robot.data.root_pos_w[:, 2]
                    stability = (
                        torch.sigmoid((0.16 - tilt_now) / 0.04)
                        * torch.sigmoid((height_now - 0.40) / 0.03)
                    ).clamp(0.0, 1.0)
                    target = (0.65 * stability + 0.20 * 0.50 * stability)[:, None] * zyb[:, :12]
                    leg_action = target
                action = torch.zeros_like(zero_action)
                action[:, :12] = leg_action

            result = env.step(action)
            observations = result[0]
            terminated = result[2] if len(result) > 2 else torch.zeros(1, dtype=torch.bool, device=env.device)
            truncated = result[3] if len(result) > 3 else torch.zeros(1, dtype=torch.bool, device=env.device)
            if step >= args.warmup:
                vx = float(robot.data.root_lin_vel_b[0, 0].item())
                wz = float(robot.data.root_ang_vel_b[0, 2].item())
                tilt = float(_tilt(robot)[0].item())
                z = float(robot.data.root_pos_w[0, 2].item())
                active_vx.append(vx)
                active_wz.append(wz)
                active_tilt.append(tilt)
                active_z.append(z)
                done = bool((terminated | truncated).any().item())
                if done and first_done is None:
                    first_done = step
            if step in (args.warmup, args.steps - 1):
                print(
                    "mode_step", mode, step,
                    "vx", float(robot.data.root_lin_vel_b[0, 0].item()),
                    "wz", float(robot.data.root_ang_vel_b[0, 2].item()),
                    "z", float(robot.data.root_pos_w[0, 2].item()),
                    "tilt", float(_tilt(robot)[0].item()),
                    "teacher_max", teacher_max,
                    flush=True,
                )
        divisor = max(len(active_vx), 1)
        print(
            "mode_summary", mode,
            "active_steps", len(active_vx),
            "mean_vx", sum(active_vx) / divisor,
            "mean_wz", sum(active_wz) / divisor,
            "min_z", min(active_z) if active_z else None,
            "max_tilt", max(active_tilt) if active_tilt else None,
            "teacher_max", teacher_max,
            "mean_clipped_fraction", clipped_fraction / divisor,
            "first_done_step", first_done,
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
