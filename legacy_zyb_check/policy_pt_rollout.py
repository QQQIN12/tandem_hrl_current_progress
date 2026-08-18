"""Short empirical rollout for the legacy 57-D TorchScript gait policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", type=str, default="ZYB-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--layout", choices=("legacy_guess", "common_scaled"), default="common_scaled")
parser.add_argument("--clip_actions", type=float, default=None)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import quadruped_arm.tasks  # noqa: F401,E402


def set_command(term, command: torch.Tensor) -> None:
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[:] = command


def build_57_obs(env, robot, command_term) -> torch.Tensor:
    """Build one of the two plausible 57-D legged-policy conventions."""
    if args.layout == "common_scaled":
        base_ang_vel = robot.data.root_ang_vel_b * 0.25
    else:
        base_ang_vel = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    command = command_term.vel_command_b
    if args.layout == "common_scaled":
        command = command * command.new_tensor((2.0, 2.0, 0.25))

    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FL_foot_wheel_joint", "FR_foot_wheel_joint",
        "RL_foot_wheel_joint", "RR_foot_wheel_joint",
    ]
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    joint_ids = torch.as_tensor(joint_ids, device=robot.device, dtype=torch.long)
    q_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    qd = robot.data.joint_vel[:, joint_ids]
    if args.layout == "common_scaled":
        qd = qd * 0.05
    last_action = env.action_manager.action[:, :16]
    return torch.cat(
        [base_ang_vel, projected_gravity, command[:, :3], q_rel, qd, last_action],
        dim=-1,
    )


def tilt_from_gravity(robot) -> torch.Tensor:
    return torch.asin(
        torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1).clamp(0.0, 1.0)
    )


try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 1042
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    command_term = base.command_manager.get_term("locomotion")
    policy = torch.jit.load(str(args.checkpoint), map_location=base.device).eval()

    obs, _ = env.reset()
    command = torch.tensor(args.command, device=base.device, dtype=torch.float32).view(1, 3)
    command = command.expand(base.num_envs, -1)
    test_obs = build_57_obs(base, robot, command_term)
    test_action = policy(test_obs)
    print("policy_input_shape", tuple(test_obs.shape), "policy_output_shape", tuple(test_action.shape), flush=True)
    print("action_terms", list(base.action_manager.active_terms), "action_dim", int(base.action_manager.total_action_dim), flush=True)

    ever_tilt = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    ever_low = torch.zeros_like(ever_tilt)
    ever_bad = torch.zeros_like(ever_tilt)
    done_count = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    max_tilt = torch.zeros(base.num_envs, device=base.device)
    min_height = torch.full((base.num_envs,), float("inf"), device=base.device)
    vx_sum = torch.zeros(base.num_envs, device=base.device)
    wz_sum = torch.zeros(base.num_envs, device=base.device)

    for step in range(args.num_steps):
        set_command(command_term, command)
        obs57 = build_57_obs(base, robot, command_term)
        with torch.inference_mode():
            actions = policy(obs57)
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise RuntimeError(f"unexpected policy output shape: {tuple(actions.shape)}")
        if args.clip_actions is not None:
            actions = actions.clamp(-args.clip_actions, args.clip_actions)
        _, _, terminated, truncated, _ = env.step(actions)

        tilt = tilt_from_gravity(robot)
        height = robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, height)
        vx_sum += robot.data.root_lin_vel_b[:, 0]
        wz_sum += robot.data.root_ang_vel_b[:, 2]
        if hasattr(base.termination_manager, "get_term"):
            try:
                ever_tilt |= base.termination_manager.get_term("tilt")
                ever_low |= base.termination_manager.get_term("low_height")
                ever_bad |= base.termination_manager.get_term("bad_contact")
            except Exception:
                pass
        done_count += (terminated | truncated).to(torch.long)

    result = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "num_envs": int(base.num_envs),
        "num_steps": int(args.num_steps),
        "command": list(args.command),
        "layout": args.layout,
        "clip_actions": args.clip_actions,
        "policy_input_shape": list(test_obs.shape),
        "policy_output_shape": list(test_action.shape),
        "action_dim": int(base.action_manager.total_action_dim),
        "tilt_envs": int(ever_tilt.sum().item()),
        "low_height_envs": int(ever_low.sum().item()),
        "bad_contact_envs": int(ever_bad.sum().item()),
        "envs_with_done": int((done_count > 0).sum().item()),
        "max_tilt_mean_rad": float(max_tilt.mean().item()),
        "max_tilt_max_rad": float(max_tilt.max().item()),
        "min_height_mean_m": float(min_height.mean().item()),
        "min_height_min_m": float(min_height.min().item()),
        "mean_body_vx_mps": float((vx_sum / args.num_steps).mean().item()),
        "mean_body_wz_radps": float((wz_sum / args.num_steps).mean().item()),
        "action_abs_mean": float(actions.abs().mean().item()),
        "action_abs_max": float(actions.abs().max().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
finally:
    simulation_app.close()
