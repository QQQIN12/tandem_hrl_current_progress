"""Reset and contact-material events for the real-grasp task."""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import EventCfg

from .physics_contract import FINGER_SIMULATION_MATERIAL, WHEEL_GROUND_MATERIAL


def reset_collision_free_root(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    exclusion_asset_names: tuple[str, ...],
    exclusion_radii: tuple[float, ...],
    max_resample_attempts: int = 32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample the robot pose without placing its footprint inside a platform."""

    if len(exclusion_asset_names) != len(exclusion_radii):
        raise ValueError("exclusion assets and radii must have equal length")

    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    pose_bounds = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device
    )

    def sample(count: int) -> torch.Tensor:
        return math_utils.sample_uniform(
            pose_bounds[:, 0], pose_bounds[:, 1], (count, len(pose_keys)), device=asset.device
        )

    pose = sample(len(env_ids))
    best_pose = pose.clone()
    best_clearance = torch.full((len(env_ids),), -float("inf"), device=asset.device)
    valid = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
    clearance = best_clearance.clone()

    for _ in range(max(1, int(max_resample_attempts))):
        positions = root_states[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
        candidates = []
        for asset_name, radius in zip(exclusion_asset_names, exclusion_radii):
            obstacle = env.scene[asset_name]
            obstacle_data = getattr(obstacle, "data", None)
            obstacle_position = getattr(obstacle_data, "root_pos_w", None)
            if isinstance(obstacle_position, torch.Tensor):
                obstacle_xy = obstacle_position[env_ids, :2]
            else:
                cfg = getattr(env.scene.cfg, asset_name)
                obstacle_xy = env.scene.env_origins[env_ids, :2] + positions.new_tensor(
                    cfg.init_state.pos[:2]
                )
            candidates.append(
                torch.linalg.vector_norm(positions[:, :2] - obstacle_xy, dim=1) - float(radius)
            )
        clearance = torch.stack(candidates, dim=1).amin(dim=1)
        improved = clearance > best_clearance
        best_clearance[improved] = clearance[improved]
        best_pose[improved] = pose[improved]
        valid = clearance >= 0.0
        if torch.all(valid):
            break
        pose[~valid] = sample(int((~valid).sum().item()))

    pose[~valid] = best_pose[~valid]
    clearance[~valid] = best_clearance[~valid]
    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_bounds = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys], device=asset.device
    )
    velocity = math_utils.sample_uniform(
        velocity_bounds[:, 0],
        velocity_bounds[:, 1],
        (len(env_ids), len(velocity_keys)),
        device=asset.device,
    )
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13] + velocity, env_ids=env_ids)

    if not hasattr(env, "zyb_real_grasp_start_clearance"):
        env.zyb_real_grasp_start_clearance = torch.zeros(env.num_envs, device=asset.device)
    env.zyb_real_grasp_start_clearance[env_ids] = clearance


@configclass
class ZYBRealGraspEventCfg(EventCfg):
    reset_root = EventTerm(
        func=reset_collision_free_root,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.00, 1.00),
                "y": (-0.90, 0.90),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "exclusion_asset_names": ("source_platform", "target_platform"),
            "exclusion_radii": (0.80, 0.90),
            "max_resample_attempts": 32,
        },
    )
    reset_object = EventTerm(
        func=locomotion_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("grasp_object"),
            "pose_range": {"x": (-0.030, 0.030), "y": (-0.025, 0.025), "yaw": (-0.35, 0.35)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_left_finger_open = EventTerm(
        func=locomotion_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint7"]),
            "position_range": (0.032, 0.034),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_right_finger_open = EventTerm(
        func=locomotion_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint8"]),
            "position_range": (-0.034, -0.032),
            "velocity_range": (0.0, 0.0),
        },
    )
    wheel_contact_material = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "static_friction_range": (
                WHEEL_GROUND_MATERIAL.static_friction,
                WHEEL_GROUND_MATERIAL.static_friction,
            ),
            "dynamic_friction_range": (
                WHEEL_GROUND_MATERIAL.dynamic_friction,
                WHEEL_GROUND_MATERIAL.dynamic_friction,
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    finger_contact_material = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link7", "link8"]),
            "static_friction_range": (
                FINGER_SIMULATION_MATERIAL.static_friction,
                FINGER_SIMULATION_MATERIAL.static_friction,
            ),
            "dynamic_friction_range": (
                FINGER_SIMULATION_MATERIAL.dynamic_friction,
                FINGER_SIMULATION_MATERIAL.dynamic_friction,
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    push = None
