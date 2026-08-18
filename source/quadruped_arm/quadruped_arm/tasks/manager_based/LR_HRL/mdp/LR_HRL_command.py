"""Structured task and skill commands for LR_HRL.

The command term implements the high-level route schedule, phase selector, and
skill-mode selector while preserving the baseline interfaces consumed by the
existing action terms.  ``locomotion`` still returns a 3-D base velocity command,
and ``ee_goal`` still returns the spherical end-effector target used by the
baseline IK action.
"""

import math
from typing import Optional

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.mdp.utils import (
    _normalize,
    _quat_apply,
    _quat_from_euler_xyz,
    _quat_from_tool_z,
    _quat_from_yaw,
    _quat_mul,
    _sphere2cart,
)


PHASE_APPROACH = 0
PHASE_PRE_ALIGN = 1
PHASE_REACH = 2
PHASE_STABILIZE = 3
PHASE_GRASP = 4
PHASE_RECOVER = 5
NUM_PHASES = 6

SKILL_WHEEL_LOCO = 0
SKILL_LEG_LOCO = 1
SKILL_BASE_ALIGN = 2
SKILL_ARM_REACH = 3
SKILL_STABILIZE = 4
SKILL_GRIPPER = 5
SKILL_RECOVER = 6
NUM_SKILLS = 7

TASK_FAMILY_IDS = {
    "route": 0,
    "slalom": 1,
    "narrow": 2,
    "manip": 3,
    "grasp": 4,
    "recovery": 5,
}

TAU_DOWN_DIM = 32
TAU_UP_DIM = 16


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_rotate(yaw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], dim=-1)


def _yaw_rotate_inverse(yaw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack([c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]], dim=-1)


def _one_hot(index: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.zeros(index.numel(), width, device=index.device, dtype=torch.float32)
    out.scatter_(1, index.long().view(-1, 1), 1.0)
    return out


@configclass
class LRHrlRouteCommandCfg(CommandTermCfg):
    """High-level route command with explicit phase/skill decomposition.

    The term selects a task phase and a skill mode, then adapts the selected
    subgoal back to the baseline locomotion and EE-goal APIs.
    """

    class_type: Optional[type] = None
    asset_name: str = "robot"
    ee_body_name: str = "link6"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    task_family: str = "route"
    num_waypoints: int = 3
    num_obstacles: int = 6
    route_horizon_s: float = 10.0

    goal_radius: float = 0.55
    align_radius: float = 0.85
    yaw_align_tolerance: float = 0.35
    ee_reach_radius: float = 0.13

    kp_pos: float = 0.95
    kp_yaw: float = 1.35
    max_lin_speed: float = 0.8
    max_yaw_rate: float = 1.0
    min_approach_speed: float = 0.08

    obstacle_slow_margin: float = 0.85
    obstacle_stop_margin: float = 0.34
    recovery_tilt_threshold: float = 0.48
    low_support_threshold: float = 1.5

    ee_sphere_center_base: tuple[float, float, float] = (-0.215, 0.0, 0.7)
    ee_default_sphere: tuple[float, float, float] = (0.56, -0.22, 0.0)
    frame_yaw_offset: float = math.pi
    goal_roll_about_tool_z: float = -math.pi

    enable_decomposition_packet: bool = True

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = LRHrlRouteCommand


class LRHrlRouteCommand(CommandTerm):
    """Top-down LR_HRL route command with legacy velocity command output."""

    cfg: LRHrlRouteCommandCfg

    def __init__(self, cfg: LRHrlRouteCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        max_wp = max(1, int(cfg.num_waypoints))
        max_obs = max(1, int(cfg.num_obstacles))
        self._waypoints_w = torch.zeros(self.num_envs, max_wp, 3, device=self.device)
        self._obstacles_w = torch.zeros(self.num_envs, max_obs, 2, device=self.device)
        self._obstacle_mask = torch.zeros(self.num_envs, max_obs, dtype=torch.bool, device=self.device)

        self._wp_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._num_waypoints = torch.full((self.num_envs,), max_wp, dtype=torch.long, device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._skill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gripper_target = torch.zeros(self.num_envs, 1, device=self.device)

        self._ee_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        self.tau_down_packet = torch.zeros(self.num_envs, TAU_DOWN_DIM, device=self.device)
        self.tau_up_packet = torch.zeros(self.num_envs, TAU_UP_DIM, device=self.device)

        self._route_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_goal_dist = torch.full((self.num_envs,), 10.0, device=self.device)
        self._last_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self._last_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self._last_obstacle_margin = torch.full((self.num_envs,), 10.0, device=self.device)
        self._last_ee_error = torch.zeros(self.num_envs, device=self.device)
        self._support_count = torch.zeros(self.num_envs, device=self.device)
        self._tilt = torch.zeros(self.num_envs, device=self.device)

        self._center_offset = torch.tensor(cfg.ee_sphere_center_base, device=self.device).view(1, 3)
        self._ee_body_id = None
        self._feet_body_ids = None
        self._resolve_robot_ids()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def ee_command(self) -> torch.Tensor:
        return self._ee_command

    @property
    def phase_ids(self) -> torch.Tensor:
        return self._phase

    @property
    def skill_ids(self) -> torch.Tensor:
        return self._skill

    def _resolve_robot_ids(self):
        ee_ids, _ = self._robot.find_bodies(self.cfg.ee_body_name)
        self._ee_body_id = int(ee_ids[0]) if isinstance(ee_ids, (list, tuple)) else int(ee_ids.flatten()[0].item())

        foot_ids, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        if isinstance(foot_ids, (list, tuple)):
            foot_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)
        self._feet_body_ids = foot_ids.to(self.device).long().flatten()

    def _task_templates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        family = self.cfg.task_family
        if family == "slalom":
            waypoints = [
                (0.9, 0.42, 0.35), (1.7, -0.42, -0.35), (2.5, 0.42, 0.35), (3.3, -0.28, -0.20)
            ]
            obstacles = [(0.8, -0.05), (1.2, 0.28), (1.7, 0.05), (2.2, -0.28), (2.7, 0.05), (3.1, 0.30)]
        elif family == "narrow":
            waypoints = [(0.9, 0.0, 0.0), (1.8, 0.06, 0.0), (2.8, -0.04, 0.0), (3.5, 0.0, 0.0)]
            obstacles = [(1.1, 0.42), (1.1, -0.42), (2.0, 0.38), (2.0, -0.38), (2.9, 0.42), (2.9, -0.42)]
        elif family == "manip":
            waypoints = [(0.8, 0.0, 0.0), (1.18, 0.0, 0.0), (1.25, 0.0, 0.0)]
            obstacles = [(0.95, 0.44), (0.95, -0.44), (1.35, 0.40), (1.35, -0.40)]
        elif family == "grasp":
            waypoints = [(0.75, 0.22, 0.18), (1.16, -0.08, -0.05), (1.46, 0.16, 0.10)]
            obstacles = [(0.85, -0.30), (1.15, 0.34), (1.55, -0.32), (1.75, 0.28)]
        elif family == "recovery":
            waypoints = [(0.9, 0.32, 0.25), (1.8, -0.30, -0.25), (2.7, 0.10, 0.0)]
            obstacles = [(0.8, -0.22), (1.4, 0.26), (2.0, -0.22), (2.5, 0.30)]
        else:
            waypoints = [(1.0, 0.0, 0.0), (2.0, 0.55, 0.25), (3.0, -0.35, -0.20)]
            obstacles = [(1.35, 0.36), (1.75, -0.30), (2.35, 0.30)]

        wp = torch.tensor(waypoints[: self.cfg.num_waypoints], device=self.device, dtype=torch.float32)
        obs = torch.tensor(obstacles[: self.cfg.num_obstacles], device=self.device, dtype=torch.float32)
        ee = torch.tensor(self.cfg.ee_default_sphere, device=self.device, dtype=torch.float32).view(1, 3).repeat(wp.shape[0], 1)
        if family in ("manip", "grasp"):
            ee[:, 0] = torch.linspace(0.50, 0.64, wp.shape[0], device=self.device)
            ee[:, 1] = torch.linspace(-0.10, -0.26, wp.shape[0], device=self.device)
            ee[:, 2] = torch.linspace(-0.18, 0.18, wp.shape[0], device=self.device)
        return wp, obs, ee

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        wp_local, obs_local, ee_local = self._task_templates()
        n = env_ids.numel()
        root_pos = self._robot.data.root_pos_w[env_ids]
        root_yaw = self._robot.data.heading_w[env_ids]

        jitter_xy = torch.empty(n, wp_local.shape[0], 2, device=self.device).uniform_(-0.08, 0.08)
        local_xy = wp_local[:, :2].view(1, -1, 2).repeat(n, 1, 1) + jitter_xy
        local_yaw = wp_local[:, 2].view(1, -1).repeat(n, 1)

        for i in range(wp_local.shape[0]):
            world_xy = root_pos[:, :2] + _yaw_rotate(root_yaw, local_xy[:, i, :])
            self._waypoints_w[env_ids, i, 0:2] = world_xy
            self._waypoints_w[env_ids, i, 2] = _wrap_to_pi(root_yaw + local_yaw[:, i])

        self._num_waypoints[env_ids] = wp_local.shape[0]
        self._wp_index[env_ids] = 0
        self._phase[env_ids] = PHASE_APPROACH
        self._skill[env_ids] = SKILL_WHEEL_LOCO
        self._route_progress[env_ids] = 0.0
        self._prev_goal_dist[env_ids] = 10.0
        self._gripper_target[env_ids] = 0.0

        self._obstacle_mask[env_ids] = False
        if obs_local.numel() > 0:
            obs_jitter = torch.empty(n, obs_local.shape[0], 2, device=self.device).uniform_(-0.03, 0.03)
            obs_xy = obs_local.view(1, -1, 2).repeat(n, 1, 1) + obs_jitter
            for i in range(obs_local.shape[0]):
                self._obstacles_w[env_ids, i, :] = root_pos[:, :2] + _yaw_rotate(root_yaw, obs_xy[:, i, :])
            self._obstacle_mask[env_ids, : obs_local.shape[0]] = True

        self._ee_command[env_ids] = ee_local[0].view(1, 3)
        self.time_left[env_ids] = self.cfg.route_horizon_s

    def _support_and_tilt(self):
        if "contact_forces" in self._env.scene.keys():
            sensor = self._env.scene["contact_forces"]
            forces = sensor.data.net_forces_w[:, self._feet_body_ids, :]
            self._support_count[:] = (forces.norm(dim=-1) > 1.5).float().sum(dim=1)
        else:
            self._support_count[:] = 4.0

        if hasattr(self._robot.data, "projected_gravity_b"):
            self._tilt[:] = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=-1)
        else:
            self._tilt[:] = 0.0

    def _obstacle_margin(self, root_xy: torch.Tensor) -> torch.Tensor:
        diff = self._obstacles_w - root_xy[:, None, :]
        dist = torch.norm(diff, dim=-1)
        dist = torch.where(self._obstacle_mask, dist, torch.full_like(dist, 10.0))
        return torch.amin(dist, dim=1)

    def _set_ee_goal_from_sphere(self):
        root_pos_w = self._robot.data.root_pos_w
        yaw = self._robot.data.heading_w
        yaw_q = _quat_from_yaw(yaw)
        base_xy0 = torch.stack([root_pos_w[:, 0], root_pos_w[:, 1], torch.zeros_like(root_pos_w[:, 2])], dim=-1)
        self.center_w[:] = base_xy0 + _quat_apply(yaw_q, self._center_offset.repeat(self.num_envs, 1))

        cmd_for_cart = self._ee_command.clone()
        cmd_for_cart[:, 2] += self.cfg.frame_yaw_offset
        cart_w = _quat_apply(yaw_q, _sphere2cart(cmd_for_cart))
        self.curr_goal_pos_w[:] = self.center_w + cart_w

        tool_z_w = _normalize(self.curr_goal_pos_w - self.center_w)
        q_align = _quat_from_tool_z(tool_z_w)
        zero = torch.zeros(self.num_envs, device=self.device, dtype=tool_z_w.dtype)
        q_spin_local = _quat_from_euler_xyz(zero, zero, torch.full_like(zero, self.cfg.goal_roll_about_tool_z))
        self.curr_goal_quat_w[:] = _normalize(_quat_mul(q_align, q_spin_local))

        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_id, :]
        self._last_ee_error[:] = torch.norm(ee_pos - self.curr_goal_pos_w, dim=-1)

    def _update_phase_and_skill(self, goal_dist: torch.Tensor, yaw_error: torch.Tensor):
        family = self.cfg.task_family
        recover = (self._tilt > self.cfg.recovery_tilt_threshold) | (self._support_count < self.cfg.low_support_threshold)

        final_wp = self._wp_index >= (self._num_waypoints - 1)
        reached = goal_dist < self.cfg.goal_radius
        advance = reached & (~final_wp)
        self._wp_index[advance] += 1

        phase = torch.full_like(self._phase, PHASE_APPROACH)
        skill = torch.full_like(self._skill, SKILL_WHEEL_LOCO)

        # Task layer: switch from route following to base alignment near each
        # waypoint when heading error is the dominant residual.
        needs_align = (goal_dist < self.cfg.align_radius) & (yaw_error.abs() > self.cfg.yaw_align_tolerance)
        phase = torch.where(needs_align, torch.full_like(phase, PHASE_PRE_ALIGN), phase)
        skill = torch.where(needs_align, torch.full_like(skill, SKILL_BASE_ALIGN), skill)

        # Skill layer: manipulation families reserve the final waypoint for
        # arm reach and gripper/stabilization behavior.
        manipulation_task = family in ("manip", "grasp")
        near_final = final_wp & (goal_dist < self.cfg.align_radius)
        if manipulation_task:
            reach = near_final & (self._last_ee_error > self.cfg.ee_reach_radius)
            stable = near_final & (self._last_ee_error <= self.cfg.ee_reach_radius)
            phase = torch.where(reach, torch.full_like(phase, PHASE_REACH), phase)
            skill = torch.where(reach, torch.full_like(skill, SKILL_ARM_REACH), skill)
            stable_phase = PHASE_GRASP if family == "grasp" else PHASE_STABILIZE
            stable_skill = SKILL_GRIPPER if family == "grasp" else SKILL_STABILIZE
            phase = torch.where(stable, torch.full_like(phase, stable_phase), phase)
            skill = torch.where(stable, torch.full_like(skill, stable_skill), skill)
        else:
            stable = final_wp & reached
            phase = torch.where(stable, torch.full_like(phase, PHASE_STABILIZE), phase)
            skill = torch.where(stable, torch.full_like(skill, SKILL_STABILIZE), skill)

        phase = torch.where(recover, torch.full_like(phase, PHASE_RECOVER), phase)
        skill = torch.where(recover, torch.full_like(skill, SKILL_RECOVER), skill)
        self._phase[:] = phase
        self._skill[:] = skill
        self._gripper_target[:, 0] = (phase == PHASE_GRASP).float()

    def _update_command(self):
        self._support_and_tilt()

        root_pos = self._robot.data.root_pos_w
        root_xy = root_pos[:, :2]
        root_yaw = self._robot.data.heading_w
        idx = self._wp_index.clamp(max=self._waypoints_w.shape[1] - 1)
        goal = self._waypoints_w[torch.arange(self.num_envs, device=self.device), idx]

        delta_w = goal[:, :2] - root_xy
        delta_b = _yaw_rotate_inverse(root_yaw, delta_w)
        goal_dist = torch.norm(delta_w, dim=-1)
        target_heading = torch.atan2(delta_w[:, 1], delta_w[:, 0])
        heading_error = _wrap_to_pi(target_heading - root_yaw)
        yaw_error = _wrap_to_pi(goal[:, 2] - root_yaw)

        self._last_goal_dist[:] = goal_dist
        self._last_yaw_error[:] = yaw_error
        self._last_obstacle_margin[:] = self._obstacle_margin(root_xy)

        self._set_ee_goal_from_sphere()
        self._update_phase_and_skill(goal_dist, yaw_error)

        obstacle_scale = ((self._last_obstacle_margin - self.cfg.obstacle_stop_margin) /
                          max(1e-4, self.cfg.obstacle_slow_margin - self.cfg.obstacle_stop_margin)).clamp(0.18, 1.0)
        base_speed = (self.cfg.kp_pos * goal_dist).clamp(self.cfg.min_approach_speed, self.cfg.max_lin_speed)
        forward_gate = torch.cos(heading_error).clamp(0.0, 1.0)
        vx = base_speed * forward_gate * obstacle_scale
        wz = (self.cfg.kp_yaw * heading_error + 0.35 * yaw_error).clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

        slow_phase = (self._phase == PHASE_REACH) | (self._phase == PHASE_STABILIZE) | (self._phase == PHASE_GRASP)
        recover_phase = self._phase == PHASE_RECOVER
        vx = torch.where(slow_phase, 0.25 * vx, vx)
        wz = torch.where(slow_phase, 0.35 * wz, wz)
        vx = torch.where(recover_phase, torch.zeros_like(vx), vx)
        wz = torch.where(recover_phase, -0.65 * torch.sign(self._robot.data.root_ang_vel_b[:, 2]), wz)
        vx = torch.where(goal_dist < self.cfg.goal_radius, torch.zeros_like(vx), vx)

        self._command[:, 0] = vx
        self._command[:, 1] = 0.0
        self._command[:, 2] = wz

        improvement = (self._prev_goal_dist - goal_dist).clamp(min=-0.05, max=0.12)
        route_index_progress = self._wp_index.float() / torch.clamp((self._num_waypoints - 1).float(), min=1.0)
        local_progress = (1.0 - goal_dist / 3.0).clamp(0.0, 1.0) / torch.clamp(self._num_waypoints.float(), min=1.0)
        self._route_progress[:] = (route_index_progress + local_progress).clamp(0.0, 1.0)
        self._prev_goal_dist[:] = goal_dist

        self._update_packets(delta_b, goal_dist, yaw_error, improvement)

    def _update_packets(self, delta_b: torch.Tensor, goal_dist: torch.Tensor, yaw_error: torch.Tensor, improvement: torch.Tensor):
        if not self.cfg.enable_decomposition_packet:
            self.tau_down_packet.zero_()
        else:
            self.tau_down_packet.zero_()
            # tau_down is the task-to-skill interface observed by LR_HRL.
            # Indices are fixed so that checkpoints remain compatible across
            # the six benchmark families.
            self.tau_down_packet[:, 0:NUM_PHASES] = _one_hot(self._phase, NUM_PHASES)
            self.tau_down_packet[:, 6:6 + NUM_SKILLS] = _one_hot(self._skill, NUM_SKILLS)
            self.tau_down_packet[:, 13:16] = torch.stack(
                [delta_b[:, 0].clamp(-3.0, 3.0) / 3.0, delta_b[:, 1].clamp(-1.5, 1.5) / 1.5, yaw_error / math.pi],
                dim=-1,
            )
            self.tau_down_packet[:, 16:19] = self._ee_command
            obstacle_risk = (1.0 - self._last_obstacle_margin / self.cfg.obstacle_slow_margin).clamp(0.0, 1.0)
            tilt_risk = (self._tilt / max(1e-4, self.cfg.recovery_tilt_threshold)).clamp(0.0, 1.0)
            support_risk = ((2.5 - self._support_count) / 2.5).clamp(0.0, 1.0)
            self.tau_down_packet[:, 19:22] = torch.stack([obstacle_risk, tilt_risk, support_risk], dim=-1)
            self.tau_down_packet[:, 22] = self._route_progress
            self.tau_down_packet[:, 23] = (self.time_left / max(1e-4, self.cfg.route_horizon_s)).clamp(0.0, 1.0)
            self.tau_down_packet[:, 24] = (self._wp_index.float() / torch.clamp((self._num_waypoints - 1).float(), min=1.0))
            self.tau_down_packet[:, 25:31] = torch.stack(
                [
                    goal_dist.clamp(0.0, 4.0) / 4.0,
                    yaw_error.abs().clamp(0.0, math.pi) / math.pi,
                    self._last_ee_error.clamp(0.0, 1.2) / 1.2,
                    self._support_count / 4.0,
                    (self._phase == PHASE_RECOVER).float(),
                    self._gripper_target[:, 0],
                ],
                dim=-1,
            )
            self.tau_down_packet[:, 31] = float(TASK_FAMILY_IDS.get(self.cfg.task_family, 0)) / 5.0

        feasible = 1.0 - torch.maximum(
            torch.maximum((self._tilt / 0.6).clamp(0.0, 1.0), ((2.0 - self._support_count) / 2.0).clamp(0.0, 1.0)),
            (1.0 - self._last_obstacle_margin / self.cfg.obstacle_slow_margin).clamp(0.0, 1.0),
        )
        mismatch = (goal_dist.clamp(0.0, 4.0) / 4.0) * (1.0 - improvement.clamp(0.0, 0.12) / 0.12)

        self.tau_up_packet.zero_()
        # tau_up summarizes execution feedback for reward terms and logging.
        self.tau_up_packet[:, 0] = self._route_progress
        self.tau_up_packet[:, 1] = goal_dist
        self.tau_up_packet[:, 2] = yaw_error.abs()
        self.tau_up_packet[:, 3] = self._last_ee_error
        self.tau_up_packet[:, 4] = self._tilt
        self.tau_up_packet[:, 5] = self._support_count
        self.tau_up_packet[:, 6] = self._last_obstacle_margin
        self.tau_up_packet[:, 7] = feasible.clamp(0.0, 1.0)
        self.tau_up_packet[:, 8] = mismatch.clamp(0.0, 1.0)
        self.tau_up_packet[:, 9] = (self._phase == PHASE_RECOVER).float()
        self.tau_up_packet[:, 10] = self._gripper_target[:, 0]
        self.tau_up_packet[:, 11] = improvement
        self.tau_up_packet[:, 12] = self._command[:, 0]
        self.tau_up_packet[:, 13] = self._command[:, 2]
        self.tau_up_packet[:, 14] = self._wp_index.float()
        self.tau_up_packet[:, 15] = self._phase.float()

    def _update_metrics(self):
        self.metrics["LR_HRL/route_progress"] = self._route_progress
        self.metrics["LR_HRL/goal_distance"] = self._last_goal_dist
        self.metrics["LR_HRL/yaw_error"] = self._last_yaw_error.abs()
        self.metrics["LR_HRL/ee_error"] = self._last_ee_error
        self.metrics["LR_HRL/obstacle_margin"] = self._last_obstacle_margin
        self.metrics["LR_HRL/support_count"] = self._support_count
        self.metrics["LR_HRL/recovery_active"] = (self._phase == PHASE_RECOVER).float()


@configclass
class LRHrlEeGoalCommandCfg(CommandTermCfg):
    """EE-goal adapter that reads the LR_HRL route command packet."""

    class_type: Optional[type] = None
    asset_name: str = "robot"
    source_command_name: str = "locomotion"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = LRHrlEeGoalCommand


class LRHrlEeGoalCommand(CommandTerm):
    """Legacy ``ee_goal`` command backed by the LR_HRL route command."""

    cfg: LRHrlEeGoalCommandCfg

    def __init__(self, cfg: LRHrlEeGoalCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _source(self) -> LRHrlRouteCommand:
        return self._env.command_manager.get_term(self.cfg.source_command_name)

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.time_left[env_ids] = self._source().time_left[env_ids]

    def _update_command(self):
        src = self._source()
        self._command[:] = src.ee_command
        self.center_w[:] = src.center_w
        self.curr_goal_pos_w[:] = src.curr_goal_pos_w
        self.curr_goal_quat_w[:] = src.curr_goal_quat_w
        self.time_left[:] = src.time_left

    def _update_metrics(self):
        return
