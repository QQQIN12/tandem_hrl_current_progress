from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoEnvCfg


def main() -> None:
    cfg = ManipLocoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.commands.locomotion.debug_vis = False
    cfg.commands.ee_goal.debug_vis = False
    cfg.sim.render_interval = 100000
    cfg.events.reset_root.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    policy = torch.jit.load(args.checkpoint, map_location=env.device).eval()
    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    print("policy_obs_shape", tuple(policy_obs.shape), flush=True)
    print("policy_obs_numel_per_env", int(policy_obs.shape[-1]), flush=True)
    try:
        with torch.inference_mode():
            actions = policy(policy_obs)
        print("policy_action_shape", tuple(actions.shape), flush=True)
    except Exception as exc:
        print("policy_call_error", type(exc).__name__, str(exc), flush=True)
        return
    for step in range(args.steps):
        obs, _, terminated, truncated, _ = env.step(actions)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        with torch.inference_mode():
            actions = policy(policy_obs)
        if step in (0, args.steps // 2, args.steps - 1):
            robot = env.scene["robot"]
            tilt = torch.asin(
                torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2])
                .clamp(0.0, 1.0)
            )
            print(
                "step", step, "vx", float(robot.data.root_lin_vel_b[0, 0]),
                "wz", float(robot.data.root_ang_vel_b[0, 2]),
                "z", float(robot.data.root_pos_w[0, 2]),
                "tilt", float(tilt),
                "terminated", bool(terminated[0]),
                "truncated", bool(truncated[0]),
                flush=True,
            )
    env.close()


try:
    main()
finally:
    simulation_app.close()
