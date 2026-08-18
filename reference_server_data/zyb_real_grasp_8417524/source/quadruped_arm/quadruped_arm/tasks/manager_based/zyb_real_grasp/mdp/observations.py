"""Object, gripper, and support observations appended to ZYB-v0 proprioception."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch


FOOT_SENSORS = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)


def _asset_position_w(env, name: str) -> torch.Tensor:
    asset = env.scene[name]
    data = getattr(asset, "data", None)
    position = getattr(data, "root_pos_w", None)
    if isinstance(position, torch.Tensor):
        return position
    cfg = getattr(env.scene.cfg, name)
    offset = env.scene.env_origins.new_tensor(cfg.init_state.pos).view(1, 3)
    return env.scene.env_origins + offset


def _load(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    forces = sensor.data.force_matrix_w
    if forces is None:
        forces = sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.reshape(forces.shape[0], -1, 3), dim=-1).amax(dim=1)


def real_grasp_state(env) -> torch.Tensor:
    """Return a compact privileged state for the baseline grasp task."""

    robot = env.scene["robot"]
    grasp_object = env.scene["grasp_object"]
    target_w = _asset_position_w(env, "target_platform")
    object_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, grasp_object.data.root_pos_w - robot.data.root_pos_w
    )
    target_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, target_w - robot.data.root_pos_w
    )
    link_ids, _ = robot.find_bodies(["link6"], preserve_order=True)
    wrist_object_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w,
        grasp_object.data.root_pos_w - robot.data.body_pos_w[:, link_ids[0]],
    )
    finger_ids, _ = robot.find_joints(["joint7", "joint8"], preserve_order=True)
    finger_loads = torch.stack(
        (_load(env, "left_finger_object_contact"), _load(env, "right_finger_object_contact")),
        dim=1,
    )
    foot_loads = torch.stack([_load(env, name) for name in FOOT_SENSORS], dim=1) / 400.0
    object_velocity_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, grasp_object.data.root_lin_vel_w
    )
    return torch.cat(
        (
            object_b,
            target_b,
            wrist_object_b,
            object_velocity_b,
            robot.data.joint_pos[:, finger_ids],
            (finger_loads / 20.0).clamp(0.0, 2.0),
            foot_loads.clamp(0.0, 2.0),
            robot.data.projected_gravity_b[:, :2],
        ),
        dim=1,
    )


def support_metrics(env) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    foot_loads = torch.stack([_load(env, name) for name in FOOT_SENSORS], dim=1)
    finger_loads = torch.stack(
        (_load(env, "left_finger_object_contact"), _load(env, "right_finger_object_contact")),
        dim=1,
    )
    return {
        "foot_loads": foot_loads,
        "support_count": (foot_loads > 5.0).float().sum(dim=1),
        "rear_support_count": (foot_loads[:, 2:] > 5.0).float().sum(dim=1),
        "finger_loads": finger_loads,
        "bilateral_contact": (finger_loads > 0.20).all(dim=1),
        "tilt": torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1),
        "base_height": robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
    }
