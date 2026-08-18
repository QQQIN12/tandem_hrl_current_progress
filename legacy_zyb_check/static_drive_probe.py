"""Check whether the nominal fixed leg posture can safely drive the wheels."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vx", type=float, default=0.10)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--root_z", type=float, default=0.58)
parser.add_argument("--no_gravity", action="store_true")
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--wheel_damping", type=float, default=None)
parser.add_argument("--wheel_stiffness", type=float, default=None)
parser.add_argument("--wheel_armature", type=float, default=None)
parser.add_argument("--track_width", type=float, default=None)
parser.add_argument("--wheel_accel", type=float, default=None)
parser.add_argument("--turn_breakaway_wz", type=float, default=None)
parser.add_argument("--turn_breakaway_threshold", type=float, default=None)
parser.add_argument("--wheel_residual_scale", type=float, default=None)
parser.add_argument("--direct_wheel", type=str, default=None)
parser.add_argument("--vx_feedback_gain", type=float, default=None)
parser.add_argument("--wz_feedback_gain", type=float, default=None)
parser.add_argument("--leg_stiffness", type=float, default=None)
parser.add_argument("--leg_damping", type=float, default=None)
parser.add_argument("--calf_stiffness", type=float, default=None)
parser.add_argument("--calf_damping", type=float, default=None)
parser.add_argument("--ground_friction", type=float, default=None)
parser.add_argument("--arm_ik_damping", type=float, default=None)
parser.add_argument("--disable_arm_ik", action="store_true")
parser.add_argument("--disable_posture_feedback", action="store_true")
parser.add_argument("--posture_authority", type=float, default=None)
parser.add_argument("--base_height_target", type=float, default=None)
parser.add_argument("--base_height_gain", type=float, default=None)
parser.add_argument("--orientation_gain", type=float, default=None)
parser.add_argument("--wheel_sign", type=float, default=None)
parser.add_argument("--wheel_signs", type=str, default=None)
parser.add_argument("--wz_sign", type=float, default=None)
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--foot_inertia", action="store_true")
parser.add_argument("--foot_ix", type=float, default=0.0032)
parser.add_argument("--foot_iy", type=float, default=0.0059)
parser.add_argument("--foot_iz", type=float, default=0.0032)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg


def main() -> None:
    cfg = ManipLocoEnvCfg()
    print("probe_ground_friction_arg", args.ground_friction, flush=True)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    if args.no_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        print("probe_gravity", cfg.sim.gravity, flush=True)
    cfg.seed = 271828
    cfg.commands.locomotion.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    if args.ground_friction is not None:
        print(
            "ground_friction_before",
            type(cfg.scene.terrain.physics_material).__name__,
            cfg.scene.terrain.physics_material.static_friction,
            cfg.scene.terrain.physics_material.dynamic_friction,
            flush=True,
        )
        cfg.scene.terrain.physics_material.static_friction = args.ground_friction
        cfg.scene.terrain.physics_material.dynamic_friction = args.ground_friction
        print(
            "ground_friction_after",
            cfg.scene.terrain.physics_material.static_friction,
            cfg.scene.terrain.physics_material.dynamic_friction,
            flush=True,
        )
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_z)
    if args.usd is not None:
        cfg.scene.robot.spawn.usd_path = args.usd
    if args.wheel_sign is not None:
        cfg.actions.wheel_vel.wheel_dir_signs = (args.wheel_sign,) * 4
    if args.wheel_signs is not None:
        signs = tuple(float(value) for value in args.wheel_signs.split(","))
        if len(signs) != 4:
            raise ValueError("--wheel_signs must contain four comma-separated values")
        cfg.actions.wheel_vel.wheel_dir_signs = signs
    if args.wz_sign is not None:
        cfg.actions.wheel_vel.wz_sign = args.wz_sign
    if args.track_width is not None:
        cfg.actions.wheel_vel.track_width = args.track_width
    if args.wheel_accel is not None:
        cfg.actions.wheel_vel.max_wheel_accel = args.wheel_accel
    if args.turn_breakaway_wz is not None:
        cfg.actions.wheel_vel.turn_breakaway_wz = args.turn_breakaway_wz
    if args.turn_breakaway_threshold is not None:
        cfg.actions.wheel_vel.turn_breakaway_threshold = args.turn_breakaway_threshold
    if args.wheel_residual_scale is not None:
        cfg.actions.wheel_vel.residual_scale = args.wheel_residual_scale
    if args.vx_feedback_gain is not None:
        cfg.actions.wheel_vel.vx_feedback_gain = args.vx_feedback_gain
    if args.wz_feedback_gain is not None:
        cfg.actions.wheel_vel.wz_feedback_gain = args.wz_feedback_gain
    if args.leg_stiffness is not None:
        cfg.scene.robot.actuators["M107-24-2"].stiffness = args.leg_stiffness
    if args.leg_damping is not None:
        cfg.scene.robot.actuators["M107-24-2"].damping = args.leg_damping
    if args.calf_stiffness is not None:
        cfg.scene.robot.actuators["2"].stiffness = args.calf_stiffness
    if args.calf_damping is not None:
        cfg.scene.robot.actuators["2"].damping = args.calf_damping
    if args.arm_ik_damping is not None:
        cfg.actions.arm_ik.damping = args.arm_ik_damping
    if args.disable_arm_ik:
        cfg.actions.arm_ik = None
    if args.disable_posture_feedback:
        cfg.actions.leg_pos.posture_feedback_enabled = False
    if args.posture_authority is not None:
        cfg.actions.leg_pos.posture_feedback_authority = args.posture_authority
    if args.base_height_target is not None:
        cfg.actions.leg_pos.base_height_target = args.base_height_target
    if args.base_height_gain is not None:
        cfg.actions.leg_pos.base_height_gain = args.base_height_gain
    if args.orientation_gain is not None:
        cfg.actions.leg_pos.orientation_gain = args.orientation_gain
    if args.wheel_damping is not None or args.wheel_stiffness is not None:
        cfg.scene.robot.actuators["wheels"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=args.wheel_stiffness if args.wheel_stiffness is not None else 0.0,
            damping=args.wheel_damping if args.wheel_damping is not None else 0.5,
            armature=args.wheel_armature if args.wheel_armature is not None else 0.0,
            friction=0.01,
        )
    cfg.events.reset_root.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    term = env.command_manager.get_term("locomotion")
    command = torch.tensor([[args.vx, 0.0, args.wz]], device=env.device)
    zero_command = torch.zeros_like(command)
    actions = torch.zeros(1, 16, device=env.device)
    if args.direct_wheel is not None:
        direct_wheel = [float(value) for value in args.direct_wheel.split(",")]
        if len(direct_wheel) != 4:
            raise ValueError("--direct_wheel must contain four comma-separated values")
        actions[:, 12:16] = torch.tensor(direct_wheel, device=env.device)
        print("direct_wheel_action", direct_wheel, flush=True)
    direct_actions = actions.clone()
    zero_actions = torch.zeros_like(actions)
    env.reset()
    if args.foot_inertia:
        foot_ids = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)[0]
        inertias = robot.root_physx_view.get_inertias().clone()
        inertias[:, foot_ids, 0] = args.foot_ix
        inertias[:, foot_ids, 4] = args.foot_iy
        inertias[:, foot_ids, 8] = args.foot_iz
        robot.root_physx_view.set_inertias(inertias, torch.arange(cfg.scene.num_envs))
        print("runtime_foot_inertia", inertias[0, foot_ids].tolist(), flush=True)
    wheel_ids = robot.find_joints(
        ["FL_foot_wheel_joint", "FR_foot_wheel_joint", "RL_foot_wheel_joint", "RR_foot_wheel_joint"],
        preserve_order=True,
    )[0]
    leg_term = env.action_manager.get_term("leg_pos")
    leg_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    leg_ids = robot.find_joints(leg_names, preserve_order=True)[0]
    print(
        "leg_action_joint_names", leg_names,
        "leg_ids", leg_ids,
        "robot_joint_names", list(robot.joint_names),
        flush=True,
    )
    contact_sensor = env.scene["contact_forces"] if "contact_forces" in env.scene.keys() else None
    contact_names = list(getattr(contact_sensor, "body_names", ())) if contact_sensor is not None else []
    body_names = list(getattr(robot, "body_names", ()))
    print("contact_body_names", contact_names, flush=True)
    print("robot_body_names", body_names, flush=True)
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
        step_actions = (
            direct_actions
            if args.direct_wheel is not None and step >= args.warmup
            else zero_actions
            if args.direct_wheel is not None
            else actions
        )
        env.step(step_actions)
        vx = float(robot.data.root_lin_vel_b[0, 0].item())
        wz = float(robot.data.root_ang_vel_b[0, 2].item())
        z = float(robot.data.root_pos_w[0, 2].item())
        tilt = float(torch.asin(torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2]).clamp(0.0, 1.0)).item())
        vx_sum += vx
        wz_sum += wz
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, z)
        if step in (0, args.steps // 2, args.steps - 1):
            wheel_vel = robot.data.joint_vel[:, wheel_ids][0].tolist()
            wheel_target = getattr(env, "safe_wheel_target", torch.zeros(1, 4, device=env.device))[0].tolist()
            computed_torque = getattr(robot.data, "computed_torque", None)
            applied_torque = getattr(robot.data, "applied_torque", None)
            torque = computed_torque if computed_torque is not None else applied_torque
            wheel_torque = torque[:, wheel_ids][0].tolist() if torque is not None else None
            leg_torque = torque[:, leg_ids][0].tolist() if torque is not None else None
            leg_target = getattr(leg_term, "_processed_actions", None)
            leg_target = leg_target[0].tolist() if leg_target is not None else None
            contact_forces = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
            contact_force = None
            if contact_forces is not None:
                selected = [contact_names.index(name) for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot") if name in contact_names]
                contact_force = torch.linalg.vector_norm(contact_forces[0, selected], dim=-1).tolist() if selected else None
            foot_z = None
            body_pos_w = getattr(robot.data, "body_pos_w", None)
            if body_pos_w is not None:
                selected = [body_names.index(name) for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot") if name in body_names]
                foot_z = body_pos_w[0, selected, 2].tolist() if selected else None
            print("step", step, "vx", vx, "wz", wz, "z", z, "tilt", tilt,
                  "wheel_vel", wheel_vel, "wheel_target", wheel_target,
                  "wheel_torque", wheel_torque, "foot_z", foot_z,
                  "foot_contact_force", contact_force,
                  "leg_pos", robot.data.joint_pos[0, leg_ids].tolist(),
                  "leg_target", leg_target,
                  "leg_torque", leg_torque,
                  flush=True)
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
