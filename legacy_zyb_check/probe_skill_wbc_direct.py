"""Directly probe the 20-D TANDEM locomotion skill executor.

This is deliberately not a learning run.  It checks whether the local
SupportWBCAction interface can transmit signed wheel references while keeping
the four-foot support and leg correction diagnostics observable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--magnitude", type=float, default=60.0)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--motion_steps", type=int, default=180)
parser.add_argument("--ramp_steps", type=int, default=60)
parser.add_argument("--patterns", type=str, default="forward,ideal_yaw,reverse_yaw")
parser.add_argument(
    "--support_modes",
    type=str,
    default="0,0,0,0",
    help="Semicolon-separated FL,FR,RL,RR support-release vectors.",
)
parser.add_argument("--wheel_damping", type=float, default=None)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from quadruped_arm.tasks.manager_based.TANDEM_HRL.locomotion_skill_env_cfg import (
    TANDEMLocomotionSkillEnvCfg,
)


PATTERNS = {
    "forward": (1.0, 1.0, -1.0, -1.0),
    "ideal_yaw": (-1.0, 1.0, 1.0, -1.0),
    "reverse_yaw": (1.0, -1.0, -1.0, 1.0),
}
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _tilt(robot) -> torch.Tensor:
    gravity = robot.data.projected_gravity_b
    return torch.asin(
        torch.linalg.vector_norm(gravity[:, :2], dim=1).clamp(0.0, 1.0)
    )


def main() -> None:
    selected = [item.strip() for item in args.patterns.split(",") if item.strip()]
    unknown = [item for item in selected if item not in PATTERNS]
    if unknown:
        raise ValueError(f"Unknown patterns {unknown}; available={sorted(PATTERNS)}")
    support_modes = []
    for encoded in args.support_modes.split(";"):
        values = tuple(float(item.strip()) for item in encoded.split(","))
        if len(values) != 4:
            raise ValueError("Each support mode needs four comma-separated values")
        support_modes.append(values)
    entries = [
        (pattern_name, support_mode)
        for pattern_name in selected
        for support_mode in support_modes
    ]

    cfg = TANDEMLocomotionSkillEnvCfg()
    cfg.scene.num_envs = len(entries)
    cfg.scene.env_spacing = 4.0
    cfg.sim.device = args.device
    cfg.seed = 271828
    cfg.episode_length_s = 60.0
    cfg.sim.render_interval = 100000
    cfg.terminations.time_out = None
    cfg.terminations.bad_contact = None
    cfg.terminations.tilt = None
    cfg.terminations.low_height = None
    if args.wheel_damping is not None:
        cfg.scene.robot.actuators["wheels"].damping = args.wheel_damping
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    root_position = env.scene.env_origins.clone()
    root_position[:, 1] -= 0.65
    root_position[:, 2] = 0.54
    root_rotation = torch.zeros(env.num_envs, 4, device=env.device)
    root_rotation[:, 0] = 1.0
    robot.write_root_pose_to_sim(
        torch.cat((root_position, root_rotation), dim=1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        torch.zeros_like(robot.data.default_joint_vel),
        env_ids=env_ids,
    )

    action_dim = env.action_manager.total_action_dim
    if action_dim != 20:
        raise RuntimeError(f"Expected 20-D SupportWBCAction, got {action_dim}")
    zero_actions = torch.zeros(env.num_envs, action_dim, device=env.device)
    for _ in range(args.settle_steps):
        env.step(zero_actions)

    contact_sensor = env.scene["contact_forces"]
    contact_names = list(contact_sensor.body_names)
    foot_sensor_ids = torch.tensor(
        [contact_names.index(name) for name in FOOT_NAMES],
        device=env.device,
        dtype=torch.long,
    )
    wheel_ids = robot.find_joints(
        [
            "FL_foot_wheel_joint",
            "FR_foot_wheel_joint",
            "RL_foot_wheel_joint",
            "RR_foot_wheel_joint",
        ],
        preserve_order=True,
    )[0]
    patterns = torch.tensor(
        [PATTERNS[name] for name, _ in entries],
        device=env.device,
        dtype=torch.float32,
    )
    supports = torch.tensor(
        [support for _, support in entries],
        device=env.device,
        dtype=torch.float32,
    )
    actions = torch.zeros_like(zero_actions)
    velocity_sum = torch.zeros(env.num_envs, 3, device=env.device)
    max_tilt = torch.zeros(env.num_envs, device=env.device)
    min_height = torch.full_like(max_tilt, float("inf"))
    min_support = torch.full_like(max_tilt, 4.0)
    leg_correction_max = torch.zeros_like(max_tilt)
    initial_xy = robot.data.root_pos_w[:, :2].clone()

    print(
        "probe_config",
        {
            "patterns": selected,
            "support_modes": support_modes,
            "entries": entries,
            "magnitude": args.magnitude,
            "settle_steps": args.settle_steps,
            "motion_steps": args.motion_steps,
            "ramp_steps": args.ramp_steps,
            "action_dim": action_dim,
        },
        flush=True,
    )
    for step in range(args.motion_steps):
        ramp = min(1.0, float(step + 1) / float(max(args.ramp_steps, 1)))
        actions.zero_()
        actions[:, 12:16] = args.magnitude * ramp * patterns
        actions[:, 16:20] = supports
        env.step(actions)

        velocity_sum[:, 0] += robot.data.root_lin_vel_b[:, 0]
        velocity_sum[:, 1] += robot.data.root_lin_vel_b[:, 1]
        velocity_sum[:, 2] += robot.data.root_ang_vel_b[:, 2]
        tilt = _tilt(robot)
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, robot.data.root_pos_w[:, 2])
        forces = contact_sensor.data.net_forces_w[:, foot_sensor_ids]
        contact_norm = torch.linalg.vector_norm(forces, dim=-1)
        min_support = torch.minimum(
            min_support, (contact_norm > 1.5).sum(dim=1).float()
        )
        diagnostics = getattr(env, "tandem_wbc_diagnostics", None)
        if diagnostics is not None:
            leg_correction_max = torch.maximum(
                leg_correction_max, diagnostics.leg_correction_norm
            )
        if step in (0, args.motion_steps // 2, args.motion_steps - 1):
            term = env.action_manager.get_term("leg_pos")
            print(
                "sample",
                step,
                "body_vx",
                robot.data.root_lin_vel_b[:, 0].tolist(),
                "body_wz",
                robot.data.root_ang_vel_b[:, 2].tolist(),
                "height",
                robot.data.root_pos_w[:, 2].tolist(),
                "tilt",
                tilt.tolist(),
                "wheel_vel",
                robot.data.joint_vel[:, wheel_ids].tolist(),
                "wheel_target",
                getattr(term, "_processed_actions", torch.zeros_like(actions))[
                    :, 12:16
                ].tolist(),
                "contact_force",
                contact_norm.tolist(),
                "leg_correction",
                getattr(diagnostics, "leg_correction_norm", None).tolist()
                if diagnostics is not None
                else None,
                flush=True,
            )

    velocity_mean = velocity_sum / float(max(args.motion_steps, 1))
    displacement = robot.data.root_pos_w[:, :2] - initial_xy
    rows = []
    for index, (name, support) in enumerate(entries):
        rows.append(
            {
                "pattern": name,
                "wheel_FL_FR_RL_RR": PATTERNS[name],
                "support_FL_FR_RL_RR": support,
                "mean_body_vx_mps": float(velocity_mean[index, 0].item()),
                "mean_body_vy_mps": float(velocity_mean[index, 1].item()),
                "mean_body_wz_radps": float(velocity_mean[index, 2].item()),
                "world_displacement_xy_m": displacement[index].cpu().tolist(),
                "max_tilt_rad": float(max_tilt[index].item()),
                "min_base_height_m": float(min_height[index].item()),
                "min_support_count": float(min_support[index].item()),
                "max_leg_correction_norm_rad": float(
                    leg_correction_max[index].item()
                ),
            }
        )
    result = {
        "gate": "20-D SupportWBC direct executor probe",
        "counts_as_final_hrl_evaluation": False,
        "magnitude": args.magnitude,
        "settle_steps": args.settle_steps,
        "motion_steps": args.motion_steps,
        "ramp_steps": args.ramp_steps,
        "wheel_damping": args.wheel_damping,
        "patterns": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    env.close()


try:
    main()
finally:
    simulation_app.close()
