"""Short direct replay of the archived 876->16 ZYB teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--vx", type=float, default=0.35)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--root_z", type=float, default=0.58)
parser.add_argument("--zero_policy_wheels", action="store_true")
parser.add_argument("--wheel_damping", type=float, default=None)
parser.add_argument("--wheel_stiffness", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg
from quadruped_arm.tasks.manager_based.maniploco.mdp.multi_teacher import _FrozenZybTeacher


def main() -> None:
    cfg = ManipLocoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 1042
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_z)
    cfg.events.reset_root.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    if args.wheel_damping is not None or args.wheel_stiffness is not None:
        cfg.scene.robot.actuators["wheels"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=args.wheel_stiffness if args.wheel_stiffness is not None else 0.0,
            damping=args.wheel_damping if args.wheel_damping is not None else 0.5,
            armature=0.0,
            friction=0.01,
        )

    env = ManagerBasedRLEnv(cfg=cfg)
    teacher = _FrozenZybTeacher(str(args.checkpoint), env.device)
    robot = env.scene["robot"]
    term = env.command_manager.get_term("locomotion")
    command = torch.tensor([[args.vx, 0.0, args.wz]], device=env.device)
    zero_command = torch.zeros_like(command)
    wheel_ids = robot.find_joints(
        [
            "FL_foot_wheel_joint", "FR_foot_wheel_joint",
            "RL_foot_wheel_joint", "RR_foot_wheel_joint",
        ],
        preserve_order=True,
    )[0]

    env.reset()
    term.vel_command_b[:] = zero_command if args.warmup > 0 else command
    observations = env.get_observations()
    policy_obs = observations["policy"] if isinstance(observations, dict) else observations
    vx_sum = 0.0
    wz_sum = 0.0
    max_tilt = 0.0
    min_z = float("inf")

    for step in range(args.steps):
        if hasattr(term, "is_standing_env"):
            term.is_standing_env.fill_(False)
        if hasattr(term, "is_heading_env"):
            term.is_heading_env.fill_(False)
        term.vel_command_b[:] = zero_command if step < args.warmup else command
        with torch.inference_mode():
            actions = teacher(policy_obs)
        if args.zero_policy_wheels:
            actions[:, 12:16] = 0.0
        result = env.step(actions)
        observations = result[0]
        policy_obs = observations["policy"] if isinstance(observations, dict) else observations

        vx = float(robot.data.root_lin_vel_b[0, 0].item())
        wz = float(robot.data.root_ang_vel_b[0, 2].item())
        z = float(robot.data.root_pos_w[0, 2].item())
        tilt = float(
            torch.asin(
            torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2])
                .clamp(0.0, 1.0)
            ).item()
        )
        vx_sum += vx
        wz_sum += wz
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, z)
        if step in (0, args.steps // 2, args.steps - 1):
            torque = getattr(robot.data, "computed_torque", None)
            if torque is None:
                torque = getattr(robot.data, "applied_torque", None)
            print(
                "step", step,
                "vx", vx, "wz", wz, "z", z, "tilt", tilt,
                "wheel_target", getattr(
                    env, "safe_wheel_target", torch.zeros(1, 4, device=env.device)
                )[0].tolist(),
                "wheel_vel", robot.data.joint_vel[0, wheel_ids].tolist(),
                "wheel_torque", torque[0, wheel_ids].tolist() if torque is not None else None,
                "teacher_leg", actions[0, :12].tolist(),
                flush=True,
            )

    print(
        "summary",
        "mean_vx", vx_sum / args.steps,
        "mean_wz", wz_sum / args.steps,
        "min_z", min_z,
        "max_tilt", max_tilt,
        "tilt_term", bool(env.termination_manager.get_term("tilt")[0].item()),
        "low_term", bool(env.termination_manager.get_term("low_height")[0].item()),
        "bad_term", bool(env.termination_manager.get_term("bad_contact")[0].item()),
        flush=True,
    )
    env.close()


try:
    main()
finally:
    simulation_app.close()
