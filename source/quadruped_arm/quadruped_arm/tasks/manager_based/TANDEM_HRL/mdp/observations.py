"""Compact privileged observations shared by Task and Skill learning."""

from __future__ import annotations

import torch

from .geometry import navigation_error


LEG_JOINTS = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
WHEEL_JOINTS = (
    "FL_foot_wheel_joint", "FR_foot_wheel_joint",
    "RL_foot_wheel_joint", "RR_foot_wheel_joint",
)
FOOT_SENSORS = (
    "FL_foot_contact", "FR_foot_contact",
    "RL_foot_contact", "RR_foot_contact",
)


def _resolved_joint_ids(env) -> tuple[list[int], list[int]]:
    if not hasattr(env, "tandem_navigation_leg_ids"):
        robot = env.scene["robot"]
        leg_ids, _ = robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        wheel_ids, _ = robot.find_joints(
            list(WHEEL_JOINTS), preserve_order=True
        )
        env.tandem_navigation_leg_ids = leg_ids
        env.tandem_navigation_wheel_ids = wheel_ids
    return env.tandem_navigation_leg_ids, env.tandem_navigation_wheel_ids


def contact_loads(env) -> torch.Tensor:
    """Return per-wheel contact-force magnitudes in FL, FR, RL, RR order."""

    loads = []
    for name in FOOT_SENSORS:
        sensor = env.scene[name]
        forces = sensor.data.force_matrix_w
        if forces is None:
            forces = sensor.data.net_forces_w
        magnitude = torch.linalg.vector_norm(
            forces.reshape(forces.shape[0], -1, 3), dim=-1
        ).amax(dim=1)
        loads.append(magnitude)
    return torch.stack(loads, dim=1)


def _contact_state(env) -> torch.Tensor:
    """Preserve binary contacts and add a small continuous load residual."""

    loads = contact_loads(env)
    mean_load = loads.mean(dim=1, keepdim=True).clamp_min(1.0)
    relative_load = (loads / mean_load).clamp(0.0, 2.0)
    contact = (loads > 1.0).float()
    return contact * (1.0 + 0.20 * (relative_load - 1.0))


def privileged_navigation_state(env) -> torch.Tensor:
    """State for the learned approach Skill without visual shortcuts."""

    robot = env.scene["robot"]
    leg_ids, wheel_ids = _resolved_joint_ids(env)
    delta_b, distance, yaw_error = navigation_error(env)
    leg_pos = (
        robot.data.joint_pos[:, leg_ids]
        - robot.data.default_joint_pos[:, leg_ids]
    )
    leg_vel = 0.05 * robot.data.joint_vel[:, leg_ids]
    wheel_vel = 0.05 * robot.data.joint_vel[:, wheel_ids]
    # Keep the 67-D state contract used by existing locomotion checkpoints.
    # The new support head is observable through contact/WBC diagnostics and
    # is intentionally excluded from the recurrent action history here.
    last_action = env.action_manager.action[:, :16].clamp(-1.0, 1.0)
    diagnostics = getattr(env, "tandem_wbc_diagnostics", None)
    if diagnostics is None:
        executor_state = torch.zeros(env.num_envs, 7, device=env.device)
    else:
        executor_state = torch.cat(
            (
                0.01 * diagnostics.wheel_command,
                diagnostics.leg_correction_norm.unsqueeze(1),
                diagnostics.policy_leg_residual_norm.unsqueeze(1),
                diagnostics.support_position_error_norm.unsqueeze(1),
            ),
            dim=1,
        )
    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            leg_pos,
            leg_vel,
            wheel_vel,
            last_action,
            executor_state,
            _contact_state(env),
            delta_b,
            distance.unsqueeze(1),
            torch.sin(yaw_error).unsqueeze(1),
            torch.cos(yaw_error).unsqueeze(1),
        ),
        dim=1,
    )


def privileged_locomotion_state(env) -> torch.Tensor:
    """State and requested body twist for the reusable learned Skill."""

    robot = env.scene["robot"]
    leg_ids, wheel_ids = _resolved_joint_ids(env)
    leg_pos = (
        robot.data.joint_pos[:, leg_ids]
        - robot.data.default_joint_pos[:, leg_ids]
    )
    leg_vel = 0.05 * robot.data.joint_vel[:, leg_ids]
    wheel_vel = 0.05 * robot.data.joint_vel[:, wheel_ids]
    last_action = env.action_manager.action[:, :16].clamp(-1.0, 1.0)
    diagnostics = getattr(env, "tandem_wbc_diagnostics", None)
    if diagnostics is None:
        executor_state = torch.zeros(env.num_envs, 7, device=env.device)
    else:
        executor_state = torch.cat(
            (
                0.01 * diagnostics.wheel_command,
                diagnostics.leg_correction_norm.unsqueeze(1),
                diagnostics.policy_leg_residual_norm.unsqueeze(1),
                diagnostics.support_position_error_norm.unsqueeze(1),
            ),
            dim=1,
        )
    command = env.command_manager.get_command("locomotion")
    return torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            leg_pos,
            leg_vel,
            wheel_vel,
            last_action,
            executor_state,
            _contact_state(env),
            command,
        ),
        dim=1,
    )
