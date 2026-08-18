#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug: freeze base + disable all ActionTerms + manually set 6 arm joint targets (q_cmd).
Optionally visualize a target point and current EE.
"""

import argparse
from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="DEBUG: freeze robot + manually set arm q targets.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)

# arm joints / ee
parser.add_argument("--ee_body", type=str, default="link7", help="EE body name for visualization/print.")
parser.add_argument(
    "--q",
    type=str,
    default="0.0,1.2,-0.6,-1.2,-0.4,0.0",
    help="Arm joint target in rad, 6 numbers: j1..j6, e.g. '0,1.2,-0.6,-1.2,-0.4,0'",
)
parser.add_argument(
    "--target",
    type=str,
    default=None,
    help="Optional target point in world, 'x,y,z' (for a green marker).",
)
parser.add_argument("--print_every", type=int, default=30)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------- start SimulationApp FIRST ----------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- imports AFTER app ----------------
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import quadruped_arm.tasks  # noqa: F401


def _parse_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    # ---- env cfg ----
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if hasattr(env_cfg.actions, "arm_ik"):
        env_cfg.actions.arm_ik = None

    # 只跑 1 个 env 看清楚
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 4.0

    # 固定 base（底盘完全不动）
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # 禁止 push
    if hasattr(env_cfg.events, "push"):
        env_cfg.events.push = None

    # 关闭 termination，防止 debug reset
    if hasattr(env_cfg.terminations, "bad_contact"):
        env_cfg.terminations.bad_contact = None
    if hasattr(env_cfg.terminations, "tilt"):
        env_cfg.terminations.tilt = None
    if hasattr(env_cfg.terminations, "low_height"):
        env_cfg.terminations.low_height = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # 加长 episode
    env_cfg.episode_length_s = 1.0e6

    # 冻结 locomotion command（虽然底盘 fixed 了，但也让 command 不乱变）
    env_cfg.commands.locomotion.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.locomotion.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.locomotion.ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    # ✅ 关键：禁用所有 ActionTerms（否则 action_manager 会覆盖你手动 set 的 q）
    if hasattr(env_cfg, "actions"):
        if hasattr(env_cfg.actions, "leg_pos"):
            env_cfg.actions.leg_pos = None
        if hasattr(env_cfg.actions, "wheel_vel"):
            env_cfg.actions.wheel_vel = None
        if hasattr(env_cfg.actions, "arm_ik"):
            env_cfg.actions.arm_ik = None

    # ---- make env ----
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: observation space: {env.observation_space}")
    print(f"[INFO]: action space: {env.action_space}")

    obs, _ = env.reset()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    # ---- arm joint ids ----
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
    # -> tensor shape (6,)
    arm_joint_ids = torch.as_tensor(arm_joint_ids, device=base_env.device, dtype=torch.long)

    # ---- ee body id ----
    ee_ids, _ = robot.find_bodies(args_cli.ee_body, preserve_order=True)
    ee_body_id = int(torch.as_tensor(ee_ids)[0].item())

    # ---- parse q target ----
    q_list = _parse_floats(args_cli.q)
    assert len(q_list) == 6, f"--q must have 6 numbers, got {len(q_list)}"
    q_cmd = torch.tensor(q_list, device=base_env.device, dtype=torch.float).view(1, 6)
    q_cmd = q_cmd.repeat(base_env.num_envs, 1)

    jnames = list(robot.data.joint_names)
    for name in ["joint1","joint2","joint3","joint4","joint5","joint6"]:
        jid = jnames.index(name)
        lo, hi = robot.data.joint_pos_limits[0, jid].tolist()
        print(name, "id=", jid, "lim=", (lo,hi))


    # ---- markers ----
    ee_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/debug_ee",
            markers={
                "ee": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                )
            },
        )
    )

    target_marker = None
    target_pos = None
    if args_cli.target is not None:
        t = _parse_floats(args_cli.target)
        assert len(t) == 3
        target_pos = torch.tensor(t, device=base_env.device, dtype=torch.float).view(1, 3)
        target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/debug_target",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )
                },
            )
        )

    quat_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=base_env.device)

    step_count = 0

    # action 维度可能是 (num_envs, 0)（因为我们关了所有 action terms）
    # 这里安全处理：
    act_shape = (base_env.num_envs, int(env.action_space.shape[-1])) if hasattr(env.action_space, "shape") else (base_env.num_envs, 0)
    actions = torch.zeros(act_shape, device=base_env.device)

    # ---------------- main loop ----------------
    while simulation_app.is_running():
        with torch.inference_mode():
            # 1) 先把你指定的 arm q 写入 position target（每步写一次最稳）
            robot.set_joint_position_target(q_cmd, joint_ids=arm_joint_ids)

            # 2) step 仿真（ActionManager 没 term，不会覆盖你的 target）
            obs, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # 3) visualize
            ee_pos = robot.data.body_pos_w[:1, ee_body_id]  # (1,3)
            ee_marker.visualize(ee_pos, quat_identity)

            if target_marker is not None:
                target_marker.visualize(target_pos, quat_identity)

            # 4) print
            if step_count % args_cli.print_every == 0:
                q_now = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().numpy()
                ee_p = ee_pos[0].detach().cpu().numpy()
                msg = f"[step {step_count:06d}] q_now={q_now} | ee=({ee_p[0]:+.3f},{ee_p[1]:+.3f},{ee_p[2]:+.3f})"
                if target_pos is not None:
                    tp = target_pos[0].detach().cpu().numpy()
                    err = float(torch.norm(ee_pos[0] - target_pos[0]).item())
                    msg += f" | target=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f}) err={err:.4f}"
                print(msg)

                arm_ids, _ = robot.find_joints(["joint1","joint2","joint3","joint4","joint5","joint6"])
                q_tgt = robot.data.joint_pos_target[0, arm_ids]  # 有的版本叫 joint_pos_target
                print("q_tgt =", q_tgt.detach().cpu().numpy())

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()