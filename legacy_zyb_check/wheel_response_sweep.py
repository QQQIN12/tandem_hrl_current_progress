"""One-process wheel action response sweep for the current TANDEM asset.

The sweep keeps the arm frozen and the nominal leg posture, settles with zero
wheel residual, then applies one direct four-wheel velocity residual pattern.
It is a diagnostic for the current asset/action interface, not a training
script and not an ideal-differential-drive assumption.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--warmup", type=int, default=45)
parser.add_argument("--magnitude", type=float, default=1.0)
parser.add_argument("--wheel_damping", type=float, default=1.0)
parser.add_argument("--leg_stiffness", type=float, default=None)
parser.add_argument("--leg_damping", type=float, default=None)
parser.add_argument("--calf_stiffness", type=float, default=None)
parser.add_argument("--calf_damping", type=float, default=None)
parser.add_argument("--ground_friction", type=float, default=None)
parser.add_argument(
    "--stable_lower",
    action="store_true",
    help="Use the separately registered stable lower-body teacher config.",
)
parser.add_argument(
    "--patterns",
    type=str,
    default="forward_rearflip,ideal_yaw,tactic_yaw_plus,tactic_yaw_minus,allplus_yaw,front_yaw,rear_yaw",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ManipLocoEnvCfg,
)
from quadruped_arm.tasks.manager_based.maniploco.stable_lower_env_cfg import (
    StableLowerEnvCfg,
)


PATTERNS = {
    "forward_rearflip": (1.0, 1.0, -1.0, -1.0),
    "forward_tactic": (1.0, 1.0, -1.0, 1.0),
    "ideal_yaw": (-1.0, 1.0, 1.0, -1.0),
    "tactic_yaw_plus": (-1.0, 1.0, 1.0, 1.0),
    "tactic_yaw_minus": (1.0, -1.0, -1.0, 1.0),
    "allplus_yaw": (-1.0, 1.0, -1.0, 1.0),
    "front_yaw": (-1.0, 1.0, 0.0, 0.0),
    "rear_yaw": (0.0, 0.0, 1.0, -1.0),
    "fl_plus": (1.0, 0.0, 0.0, 0.0),
    "fl_minus": (-1.0, 0.0, 0.0, 0.0),
    "fr_plus": (0.0, 1.0, 0.0, 0.0),
    "fr_minus": (0.0, -1.0, 0.0, 0.0),
    "rl_plus": (0.0, 0.0, 1.0, 0.0),
    "rl_minus": (0.0, 0.0, -1.0, 0.0),
    "rr_plus": (0.0, 0.0, 0.0, 1.0),
    "rr_minus": (0.0, 0.0, 0.0, -1.0),
}


def _tilt(robot) -> float:
    return float(
        torch.asin(
            torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2]).clamp(
                0.0, 1.0
            )
        ).item()
    )


def main() -> None:
    cfg = StableLowerEnvCfg() if args.stable_lower else ManipLocoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 271828
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    if args.ground_friction is not None:
        cfg.scene.terrain.physics_material.static_friction = args.ground_friction
        cfg.scene.terrain.physics_material.dynamic_friction = args.ground_friction
    cfg.actions.leg_pos.posture_feedback_enabled = False
    cfg.actions.arm_ik.max_joint_delta = 0.0
    cfg.actions.wheel_vel.residual_scale = 1.0
    cfg.actions.wheel_vel.vx_feedback_gain = 0.0
    cfg.actions.wheel_vel.wz_feedback_gain = 0.0
    if args.leg_stiffness is not None:
        cfg.scene.robot.actuators["M107-24-2"].stiffness = args.leg_stiffness
    if args.leg_damping is not None:
        cfg.scene.robot.actuators["M107-24-2"].damping = args.leg_damping
    if args.calf_stiffness is not None:
        cfg.scene.robot.actuators["2"].stiffness = args.calf_stiffness
    if args.calf_damping is not None:
        cfg.scene.robot.actuators["2"].damping = args.calf_damping
    cfg.scene.robot.actuators["wheels"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_foot_wheel_joint"],
        effort_limit_sim=23.5,
        velocity_limit_sim=30.0,
        stiffness=0.0,
        damping=args.wheel_damping,
        friction=0.01,
    )
    cfg.events.reset_root.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)

    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    term = env.command_manager.get_term("locomotion")
    contact_sensor = env.scene["contact_forces"] if "contact_forces" in env.scene.keys() else None
    contact_names = list(getattr(contact_sensor, "body_names", ())) if contact_sensor is not None else []
    foot_contact_ids = [
        contact_names.index(name)
        for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        if name in contact_names
    ]
    wheel_ids = robot.find_joints(
        [
            "FL_foot_wheel_joint",
            "FR_foot_wheel_joint",
            "RL_foot_wheel_joint",
            "RR_foot_wheel_joint",
        ],
        preserve_order=True,
    )[0]
    zero_command = torch.zeros(1, 3, device=env.device)
    zero_actions = torch.zeros(1, 16, device=env.device)
    direct_actions = torch.zeros_like(zero_actions)

    selected = [name.strip() for name in args.patterns.split(",") if name.strip()]
    unknown = [name for name in selected if name not in PATTERNS]
    if unknown:
        raise ValueError(f"Unknown patterns: {unknown}; available={sorted(PATTERNS)}")

    print(
        "sweep_config",
        {
            "steps": args.steps,
            "warmup": args.warmup,
            "magnitude": args.magnitude,
            "wheel_damping": args.wheel_damping,
            "leg_stiffness": args.leg_stiffness,
            "leg_damping": args.leg_damping,
            "calf_stiffness": args.calf_stiffness,
            "calf_damping": args.calf_damping,
            "ground_friction": args.ground_friction,
            "patterns": selected,
        },
        flush=True,
    )

    for name in selected:
        pattern = torch.tensor(
            PATTERNS[name], device=env.device, dtype=torch.float32
        ) * float(args.magnitude)
        direct_actions.zero_()
        direct_actions[:, 12:16] = pattern
        env.reset()
        vx_values: list[float] = []
        wz_values: list[float] = []
        wheel_values: list[list[float]] = []
        contact_values: list[list[float]] = []
        min_z = float("inf")
        max_tilt = 0.0
        active_start = min(args.warmup, args.steps)
        for step in range(args.steps):
            if hasattr(term, "is_standing_env"):
                term.is_standing_env.fill_(False)
            if hasattr(term, "is_heading_env"):
                term.is_heading_env.fill_(False)
            term.vel_command_b[:] = zero_command
            action = zero_actions if step < active_start else direct_actions
            env.step(action)
            vx = float(robot.data.root_lin_vel_b[0, 0].item())
            wz = float(robot.data.root_ang_vel_b[0, 2].item())
            z = float(robot.data.root_pos_w[0, 2].item())
            tilt = _tilt(robot)
            if step >= active_start:
                vx_values.append(vx)
                wz_values.append(wz)
                wheel_values.append(robot.data.joint_vel[:, wheel_ids][0].tolist())
                if contact_sensor is not None and foot_contact_ids:
                    net_forces = getattr(contact_sensor.data, "net_forces_w", None)
                    if net_forces is not None:
                        contact_values.append(
                            torch.linalg.vector_norm(
                                net_forces[0, foot_contact_ids], dim=-1
                            ).tolist()
                        )
                min_z = min(min_z, z)
                max_tilt = max(max_tilt, tilt)
            if step in (active_start, args.steps - 1):
                wheel_vel = robot.data.joint_vel[:, wheel_ids][0].tolist()
                wheel_target = getattr(
                    env, "safe_wheel_target", torch.zeros(1, 4, device=env.device)
                )[0].tolist()
                computed_torque = getattr(robot.data, "computed_torque", None)
                applied_torque = getattr(robot.data, "applied_torque", None)
                torque = computed_torque if computed_torque is not None else applied_torque
                wheel_torque = torque[0, wheel_ids].tolist() if torque is not None else None
                contact_force = None
                if contact_sensor is not None and foot_contact_ids:
                    net_forces = getattr(contact_sensor.data, "net_forces_w", None)
                    if net_forces is not None:
                        contact_force = torch.linalg.vector_norm(
                            net_forces[0, foot_contact_ids], dim=-1
                        ).tolist()
                print(
                    "sample",
                    name,
                    "step",
                    step,
                    "vx",
                    vx,
                    "wz",
                    wz,
                    "z",
                    z,
                    "tilt",
                    tilt,
                    "wheel_vel",
                    wheel_vel,
                    "wheel_target",
                    wheel_target,
                    "wheel_torque",
                    wheel_torque,
                    "foot_contact_force",
                    contact_force,
                    flush=True,
                )
        mean_wheel = (
            torch.tensor(wheel_values, dtype=torch.float32).mean(dim=0).tolist()
            if wheel_values
            else None
        )
        mean_contact = (
            torch.tensor(contact_values, dtype=torch.float32).mean(dim=0).tolist()
            if contact_values
            else None
        )
        print(
            "response",
            name,
            "pattern",
            pattern.tolist(),
            "mean_vx_active",
            sum(vx_values) / max(len(vx_values), 1),
            "mean_wz_active",
            sum(wz_values) / max(len(wz_values), 1),
            "final_vx",
            vx_values[-1] if vx_values else None,
            "final_wz",
            wz_values[-1] if wz_values else None,
            "mean_wheel_active",
            mean_wheel,
            "final_wheel",
            wheel_values[-1] if wheel_values else None,
            "mean_foot_contact_force_active",
            mean_contact,
            "min_z_active",
            min_z,
            "max_tilt_active",
            max_tilt,
            "tilt_term",
            bool(env.termination_manager.get_term("tilt")[0].item()),
            "low_term",
            bool(env.termination_manager.get_term("low_height")[0].item()),
            "bad_term",
            bool(env.termination_manager.get_term("bad_contact")[0].item()),
            flush=True,
        )

    env.close()


try:
    main()
finally:
    simulation_app.close()
