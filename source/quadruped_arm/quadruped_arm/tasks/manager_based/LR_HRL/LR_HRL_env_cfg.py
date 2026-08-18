"""LR_HRL task configurations built directly on the ZYB-v0 baseline."""

import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ActionsCfg,
    EventCfg,
    ManipLocoEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from quadruped_arm.tasks.manager_based.maniploco.mdp.observations import VbcPolicyObsTerm

from .mdp.LR_HRL_command import LRHrlEeGoalCommandCfg, LRHrlRouteCommandCfg
from .mdp import LR_HRL_observations as lr_obs
from .mdp import LR_HRL_rewards as lr_rew


OBS_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FL_foot_wheel_joint", "FR_foot_wheel_joint", "RL_foot_wheel_joint", "RR_foot_wheel_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]


@configclass
class LRHRLCommandsCfg:
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="route",
        num_waypoints=3,
        num_obstacles=4,
        goal_radius=0.55,
        align_radius=0.85,
        enable_decomposition_packet=True,
    )
    ee_goal = LRHrlEeGoalCommandCfg(asset_name="robot", source_command_name="locomotion")


@configclass
class LRHRLSlalomCommandsCfg(LRHRLCommandsCfg):
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="slalom",
        num_waypoints=4,
        num_obstacles=6,
        goal_radius=0.48,
        align_radius=0.78,
        max_lin_speed=0.72,
        obstacle_slow_margin=0.95,
        enable_decomposition_packet=True,
    )


@configclass
class LRHRLNarrowCommandsCfg(LRHRLCommandsCfg):
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="narrow",
        num_waypoints=4,
        num_obstacles=6,
        goal_radius=0.46,
        align_radius=0.74,
        max_lin_speed=0.58,
        obstacle_slow_margin=1.05,
        obstacle_stop_margin=0.38,
        enable_decomposition_packet=True,
    )


@configclass
class LRHRLManipCommandsCfg(LRHRLCommandsCfg):
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="manip",
        num_waypoints=3,
        num_obstacles=4,
        goal_radius=0.40,
        align_radius=0.68,
        ee_reach_radius=0.12,
        max_lin_speed=0.52,
        enable_decomposition_packet=True,
    )


@configclass
class LRHRLGraspCommandsCfg(LRHRLCommandsCfg):
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="grasp",
        num_waypoints=3,
        num_obstacles=4,
        goal_radius=0.38,
        align_radius=0.66,
        ee_reach_radius=0.12,
        max_lin_speed=0.50,
        enable_decomposition_packet=True,
    )


@configclass
class LRHRLRecoveryCommandsCfg(LRHRLCommandsCfg):
    locomotion = LRHrlRouteCommandCfg(
        asset_name="robot",
        ee_body_name="link6",
        task_family="recovery",
        num_waypoints=3,
        num_obstacles=4,
        goal_radius=0.50,
        align_radius=0.82,
        max_lin_speed=0.68,
        enable_decomposition_packet=True,
    )


def _baseline_command(command_cfg: LRHrlRouteCommandCfg) -> LRHrlRouteCommandCfg:
    command_cfg.enable_decomposition_packet = False
    return command_cfg


@configclass
class LRBaselineRouteCommandsCfg(LRHRLCommandsCfg):
    locomotion = _baseline_command(LRHrlRouteCommandCfg(task_family="route", num_waypoints=3, num_obstacles=4))


@configclass
class LRBaselineSlalomCommandsCfg(LRHRLSlalomCommandsCfg):
    locomotion = _baseline_command(
        LRHrlRouteCommandCfg(
            task_family="slalom",
            num_waypoints=4,
            num_obstacles=6,
            goal_radius=0.48,
            align_radius=0.78,
            max_lin_speed=0.72,
            obstacle_slow_margin=0.95,
        )
    )


@configclass
class LRBaselineNarrowCommandsCfg(LRHRLNarrowCommandsCfg):
    locomotion = _baseline_command(
        LRHrlRouteCommandCfg(
            task_family="narrow",
            num_waypoints=4,
            num_obstacles=6,
            goal_radius=0.46,
            align_radius=0.74,
            max_lin_speed=0.58,
            obstacle_slow_margin=1.05,
            obstacle_stop_margin=0.38,
        )
    )


@configclass
class LRBaselineManipCommandsCfg(LRHRLManipCommandsCfg):
    locomotion = _baseline_command(
        LRHrlRouteCommandCfg(
            task_family="manip",
            num_waypoints=3,
            num_obstacles=4,
            goal_radius=0.40,
            align_radius=0.68,
            ee_reach_radius=0.12,
            max_lin_speed=0.52,
        )
    )


@configclass
class LRBaselineGraspCommandsCfg(LRHRLGraspCommandsCfg):
    locomotion = _baseline_command(
        LRHrlRouteCommandCfg(
            task_family="grasp",
            num_waypoints=3,
            num_obstacles=4,
            goal_radius=0.38,
            align_radius=0.66,
            ee_reach_radius=0.12,
            max_lin_speed=0.50,
        )
    )


@configclass
class LRBaselineRecoveryCommandsCfg(LRHRLRecoveryCommandsCfg):
    locomotion = _baseline_command(
        LRHrlRouteCommandCfg(
            task_family="recovery",
            num_waypoints=3,
            num_obstacles=4,
            goal_radius=0.50,
            align_radius=0.82,
            max_lin_speed=0.68,
        )
    )


@configclass
class LRBenchmarkActionsCfg(ActionsCfg):
    """Baseline actions plus the Piper gripper joints for grasp-style tasks."""

    gripper_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        scale=0.03,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class LRHRLObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(
            func=VbcPolicyObsTerm,
            params={
                "asset_name": "robot",
                "obs_joint_names": OBS_JOINT_NAMES,
                "contact_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                "history_len": 10,
                "use_priv": True,
                "arm_base_offset": (-0.3, 0.0, 0.09),
            },
        )
        LR_HRL_tau_down = ObsTerm(func=lr_obs.LR_HRL_tau_down_obs, params={"command_name": "locomotion"})
        LR_HRL_tau_up = ObsTerm(func=lr_obs.LR_HRL_tau_up_obs, params={"command_name": "locomotion"})
        LR_HRL_phase_skill = ObsTerm(func=lr_obs.LR_HRL_phase_skill_obs, params={"command_name": "locomotion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LRHRLRewardsCfg(RewardsCfg):
    LR_HRL_route_progress = RewTerm(func=lr_rew.LR_HRL_route_progress, weight=0.0)
    LR_HRL_goal_tracking = RewTerm(func=lr_rew.LR_HRL_goal_tracking, weight=0.0)
    LR_HRL_yaw_alignment = RewTerm(func=lr_rew.LR_HRL_yaw_alignment, weight=0.0)
    LR_HRL_ee_tracking = RewTerm(func=lr_rew.LR_HRL_ee_tracking, weight=0.0)
    LR_HRL_stability_margin = RewTerm(func=lr_rew.LR_HRL_stability_margin, weight=0.0)
    LR_HRL_obstacle_clearance = RewTerm(func=lr_rew.LR_HRL_obstacle_clearance, weight=0.0)
    LR_HRL_feasibility = RewTerm(func=lr_rew.LR_HRL_feasibility, weight=0.0)
    LR_HRL_mismatch_penalty = RewTerm(func=lr_rew.LR_HRL_mismatch_penalty, weight=0.0)
    LR_HRL_recovery_penalty = RewTerm(func=lr_rew.LR_HRL_recovery_penalty, weight=0.0)
    LR_HRL_grasp_proxy = RewTerm(func=lr_rew.LR_HRL_grasp_proxy, weight=0.0)
    LR_HRL_forward_efficiency = RewTerm(func=lr_rew.LR_HRL_forward_efficiency, weight=0.0)


@configclass
class LRBaselineRewardsCfg(RewardsCfg):
    LR_HRL_route_progress = RewTerm(func=lr_rew.LR_HRL_route_progress, weight=0.0)
    LR_HRL_goal_tracking = RewTerm(func=lr_rew.LR_HRL_goal_tracking, weight=0.0)
    LR_HRL_yaw_alignment = RewTerm(func=lr_rew.LR_HRL_yaw_alignment, weight=0.0)
    LR_HRL_ee_tracking = RewTerm(func=lr_rew.LR_HRL_ee_tracking, weight=0.0)
    LR_HRL_stability_margin = RewTerm(func=lr_rew.LR_HRL_stability_margin, weight=0.0)
    LR_HRL_obstacle_clearance = RewTerm(func=lr_rew.LR_HRL_obstacle_clearance, weight=0.0)
    LR_HRL_grasp_proxy = RewTerm(func=lr_rew.LR_HRL_grasp_proxy, weight=0.0)


LR_HRL_REWARD_SCALES = {
    "LR_HRL_route_progress": 2.6,
    "LR_HRL_goal_tracking": 1.8,
    "LR_HRL_yaw_alignment": 0.7,
    "LR_HRL_ee_tracking": 1.4,
    "LR_HRL_stability_margin": 1.2,
    "LR_HRL_obstacle_clearance": 0.9,
    "LR_HRL_feasibility": 0.7,
    "LR_HRL_mismatch_penalty": -1.2,
    "LR_HRL_recovery_penalty": -0.6,
    "LR_HRL_grasp_proxy": 1.8,
    "LR_HRL_forward_efficiency": 1.0,
}

LR_BASELINE_REWARD_SCALES = {
    "LR_HRL_route_progress": 2.0,
    "LR_HRL_goal_tracking": 1.5,
    "LR_HRL_yaw_alignment": 0.45,
    "LR_HRL_ee_tracking": 1.0,
    "LR_HRL_stability_margin": 0.8,
    "LR_HRL_obstacle_clearance": 0.6,
    "LR_HRL_grasp_proxy": 1.2,
}


def _apply_lr_reward_scales(env_cfg, scales: dict[str, float]):
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    factor = 1.0 / (100.0 * step_dt)
    for name, scale in scales.items():
        if hasattr(env_cfg.rewards, name):
            getattr(env_cfg.rewards, name).weight = float(scale) * factor
    if hasattr(env_cfg.rewards, "action_rate"):
        env_cfg.rewards.action_rate.params["action_dim"] = 18


@configclass
class LRHRLEnvCfg(ManipLocoEnvCfg):
    """LR_HRL route benchmark with explicit task/skill packet observations."""

    commands: LRHRLCommandsCfg = LRHRLCommandsCfg()
    actions: LRBenchmarkActionsCfg = LRBenchmarkActionsCfg()
    observations: LRHRLObservationsCfg = LRHRLObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: LRHRLRewardsCfg = LRHRLRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_lr_reward_scales(self, LR_HRL_REWARD_SCALES)


@configclass
class LRHRLRouteEnvCfg(LRHRLEnvCfg):
    commands: LRHRLCommandsCfg = LRHRLCommandsCfg()


@configclass
class LRHRLSlalomEnvCfg(LRHRLEnvCfg):
    commands: LRHRLSlalomCommandsCfg = LRHRLSlalomCommandsCfg()


@configclass
class LRHRLNarrowEnvCfg(LRHRLEnvCfg):
    commands: LRHRLNarrowCommandsCfg = LRHRLNarrowCommandsCfg()


@configclass
class LRHRLManipEnvCfg(LRHRLEnvCfg):
    commands: LRHRLManipCommandsCfg = LRHRLManipCommandsCfg()


@configclass
class LRHRLGraspEnvCfg(LRHRLEnvCfg):
    commands: LRHRLGraspCommandsCfg = LRHRLGraspCommandsCfg()


@configclass
class LRHRLRecoveryEnvCfg(LRHRLEnvCfg):
    commands: LRHRLRecoveryCommandsCfg = LRHRLRecoveryCommandsCfg()


@configclass
class LRHRLPlayEnvCfg(LRHRLRouteEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.episode_length_s = 30.0


@configclass
class LRBaselineEnvCfg(ManipLocoEnvCfg):
    """Flat baseline on the same LR benchmark tasks, without packet observations."""

    commands: LRBaselineRouteCommandsCfg = LRBaselineRouteCommandsCfg()
    actions: LRBenchmarkActionsCfg = LRBenchmarkActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: LRBaselineRewardsCfg = LRBaselineRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_lr_reward_scales(self, LR_BASELINE_REWARD_SCALES)


@configclass
class LRBaselineRouteEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineRouteCommandsCfg = LRBaselineRouteCommandsCfg()


@configclass
class LRBaselineSlalomEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineSlalomCommandsCfg = LRBaselineSlalomCommandsCfg()


@configclass
class LRBaselineNarrowEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineNarrowCommandsCfg = LRBaselineNarrowCommandsCfg()


@configclass
class LRBaselineManipEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineManipCommandsCfg = LRBaselineManipCommandsCfg()


@configclass
class LRBaselineGraspEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineGraspCommandsCfg = LRBaselineGraspCommandsCfg()


@configclass
class LRBaselineRecoveryEnvCfg(LRBaselineEnvCfg):
    commands: LRBaselineRecoveryCommandsCfg = LRBaselineRecoveryCommandsCfg()


@configclass
class LRBaselinePlayEnvCfg(LRBaselineRouteEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.episode_length_s = 30.0
