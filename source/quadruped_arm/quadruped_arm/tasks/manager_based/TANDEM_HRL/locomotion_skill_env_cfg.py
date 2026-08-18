"""Velocity-conditioned locomotion Skill trained below Task decomposition."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ManipLocoEnvCfg,
)

from . import mdp
from .navigation_skill_env_cfg import (
    LEG_ACTION_SCALES,
    LEG_JOINTS,
    WHEEL_JOINTS,
)


@configclass
class LocomotionSkillActionsCfg:
    leg_pos = mdp.SupportWBCActionCfg(
        asset_name="robot",
        leg_joint_names=LEG_JOINTS,
        wheel_joint_names=WHEEL_JOINTS,
        leg_action_scales=LEG_ACTION_SCALES,
        wheel_velocity_scale=0.1,
        wheel_policy_limit=100.0,
        wheel_slew_per_step=100.0,
        support_gain=0.55,
        max_policy_joint_residual=0.0,
        max_turn_xy_relaxation=0.0,
        # Do not freeze the foot XY coordinates: yaw requires the learned
        # Skill to coordinate leg unloading and wheel differential motion.
        # The WBC still retains vertical support and joint-limit projection.
        support_xy_tracking_scale=0.0,
        wheel_coordinate_mode="independent_support",
        max_learned_unload_shift_m=0.08,
        max_learned_support_relaxation=0.75,
        max_learned_unload_joint_correction=0.14,
    )


@configclass
class LocomotionSkillObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(
            func=mdp.privileged_locomotion_state,
            history_length=1,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LocomotionSkillRewardsCfg:
    velocity_tracking = RewTerm(
        func=mdp.locomotion_velocity_tracking, weight=6.0
    )
    yaw_tracking = RewTerm(func=mdp.locomotion_yaw_tracking, weight=4.0)
    command_alignment = RewTerm(
        func=mdp.locomotion_command_alignment, weight=1.5
    )
    yaw_load_transfer = RewTerm(
        func=mdp.yaw_load_redistribution,
        weight=2.0,
        params={"target_effective_support": 3.0},
    )
    support = RewTerm(func=mdp.support_fraction, weight=0.5)
    stability = RewTerm(func=mdp.base_stability, weight=-4.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0045)
    leg_residual = RewTerm(func=mdp.leg_residual_l2, weight=-0.006)
    wheel_coordinate = RewTerm(
        func=mdp.wheel_coordinate_l2, weight=-0.002
    )
    support_allocation = RewTerm(
        func=mdp.support_allocation_l2, weight=-0.001
    )


@configclass
class TANDEMLocomotionSkillEnvCfg(ManipLocoEnvCfg):
    actions: LocomotionSkillActionsCfg = LocomotionSkillActionsCfg()
    observations: LocomotionSkillObservationsCfg = (
        LocomotionSkillObservationsCfg()
    )
    rewards: LocomotionSkillRewardsCfg = LocomotionSkillRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.episode_length_s = 12.0
        self.commands.ee_goal = None
        self.commands.locomotion = mdp.SkillVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 5.0),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            debug_vis=False,
            ranges=mdp.SkillVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.45, 0.45),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.65, 0.65),
                heading=None,
            ),
        )
        self.events.init_priv = None
        self.events.reset_rewards = None
