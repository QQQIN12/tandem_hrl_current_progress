"""Probe the ZYB wheel action scale and left/right wheel signs."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--implicit_wheels", action="store_true")
parser.add_argument("--root_z", type=float, default=0.58)
parser.add_argument("--no_gravity", action="store_true")
parser.add_argument("--wheel_scale", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg


def set_command(term, command: torch.Tensor) -> None:
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    term.vel_command_b[:] = command


def main() -> None:
    cfg = ManipLocoEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    if args.no_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.seed = 271828
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_z)
    if args.wheel_scale is not None:
        cfg.actions.wheel_vel.scale = args.wheel_scale
    if args.implicit_wheels:
        cfg.scene.robot.actuators["wheels"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
        )
        print("wheel_actuator=ImplicitActuator", flush=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    term = env.command_manager.get_term("locomotion")
    robot = env.scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(
        [
            "FL_foot_wheel_joint",
            "FR_foot_wheel_joint",
            "RL_foot_wheel_joint",
            "RR_foot_wheel_joint",
        ],
        preserve_order=True,
    )
    wheel_ids = torch.as_tensor(wheel_ids, device=env.device, dtype=torch.long)
    print("wheel_ids=", wheel_ids.tolist(), "wheel_names=", wheel_names, flush=True)
    foot_ids, foot_names = robot.find_bodies(
        ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
    )
    foot_ids = torch.as_tensor(foot_ids, device=env.device, dtype=torch.long)
    env.reset()
    contact_sensor = env.scene["contact_forces"]
    reset_forces = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
    sensor_names = getattr(contact_sensor, "body_names", None)
    if sensor_names is None:
        sensor_names = getattr(getattr(contact_sensor, "cfg", None), "body_names", None)
    if reset_forces is not None:
        force_by_body = reset_forces.abs().amax(dim=(0, 2))
        print(
            "reset_force_by_body=",
            [(sensor_names[i] if sensor_names is not None and i < len(sensor_names) else i, float(v.item())) for i, v in enumerate(force_by_body)],
            flush=True,
        )
    print(
        "reset_root_z=", robot.data.root_pos_w[:, 2].mean().item(),
        "reset_foot_z=", robot.data.body_pos_w[:, foot_ids, 2].mean(dim=0).tolist(),
        "reset_foot_names=", foot_names,
        "reset_contact_force_shape=", tuple(reset_forces.shape) if reset_forces is not None else None,
        "reset_contact_force_max=", float(reset_forces.abs().max().item()) if reset_forces is not None else None,
        flush=True,
    )
    zero_cmd = torch.zeros(4, 3, device=env.device)
    patterns = {
        "zero": (0.0, 0.0, 0.0, 0.0),
        "all_plus": (1.0, 1.0, 1.0, 1.0),
        "all_minus": (-1.0, -1.0, -1.0, -1.0),
        "yaw_positive": (-1.0, 1.0, -1.0, 1.0),
        "yaw_negative": (1.0, -1.0, 1.0, -1.0),
    }

    for name, pattern in patterns.items():
        env.reset()
        print(name, "post_reset_wheel_vel", robot.data.joint_vel[:, wheel_ids].mean(dim=0).tolist(), flush=True)
        actions = torch.zeros(4, 16, device=env.device)
        actions[:, 12:16] = actions.new_tensor(pattern)
        set_command(term, zero_cmd)
        vx_sum = torch.zeros(4, device=env.device)
        wz_sum = torch.zeros(4, device=env.device)
        wheel_sum = torch.zeros(4, 4, device=env.device)
        first_wheel = None
        last_wheel = None
        first_vx = None
        last_vx = None
        valid = 0
        for _ in range(args.steps):
            set_command(term, zero_cmd)
            env.step(actions)
            vx_sum += robot.data.root_lin_vel_b[:, 0]
            wz_sum += robot.data.root_ang_vel_b[:, 2]
            wheel_sum += robot.data.joint_vel[:, wheel_ids]
            current_wheel = robot.data.joint_vel[:, wheel_ids].clone()
            current_vx = robot.data.root_lin_vel_b[:, 0].clone()
            if first_wheel is None:
                first_wheel = current_wheel
                first_vx = current_vx
                for attr_name in ("joint_vel_target", "computed_torque", "applied_torque"):
                    value = getattr(robot.data, attr_name, None)
                    if value is not None:
                        print(
                            name,
                            attr_name,
                            value[:, wheel_ids].mean(dim=0).tolist(),
                            flush=True,
                        )
                contact = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
                if contact is not None:
                    print(name, "contact_force_max", float(contact.abs().max().item()), flush=True)
            last_wheel = current_wheel
            last_vx = current_vx
            valid += 1
        print(
            name,
            "mean_vx=", (vx_sum / valid).mean().item(),
            "mean_wz=", (wz_sum / valid).mean().item(),
            "mean_wheel=", (wheel_sum / valid).mean(dim=0).tolist(),
            "first_wheel=", first_wheel.mean(dim=0).tolist(),
            "last_wheel=", last_wheel.mean(dim=0).tolist(),
            "first_vx=", first_vx.mean().item(),
            "last_vx=", last_vx.mean().item(),
            flush=True,
        )

    env.close()


try:
    main()
finally:
    simulation_app.close()
