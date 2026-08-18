"""Diagnostic environment for the learned approach-and-align Skill."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as loco_mdp

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    HARD_BODY_CONTACT_SENSORS,
)
from quadruped_arm.tasks.manager_based.maniploco.mdp.terminations import (
    bad_contacts,
    base_height_low,
    base_tilt,
)

from . import mdp
from .physical_scene_env_cfg import TANDEMPhysicalSceneEnvCfg


LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINTS = [
    "FL_foot_wheel_joint", "FR_foot_wheel_joint",
    "RL_foot_wheel_joint", "RR_foot_wheel_joint",
]
LEG_ACTION_SCALES = (
    0.4, 0.45, 0.45,
    0.4, 0.45, 0.45,
    0.4, 0.45, 0.45,
    0.4, 0.45, 0.45,
)


@configclass
class NavigationSkillActionsCfg:
    leg_pos = mdp.SupportWBCActionCfg(
        asset_name="robot",
        leg_joint_names=LEG_JOINTS,
        wheel_joint_names=WHEEL_JOINTS,
        leg_action_scales=LEG_ACTION_SCALES,
        wheel_velocity_scale=0.1,
        wheel_policy_limit=100.0,
        wheel_slew_per_step=100.0,
        max_policy_joint_residual=0.0,
        support_xy_tracking_scale=0.0,
    )


@configclass
class NavigationSkillObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.privileged_navigation_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavigationSkillRewardsCfg:
    progress = RewTerm(func=mdp.navigation_progress, weight=120.0)
    target_pose = RewTerm(func=mdp.navigation_target_pose, weight=3.0)
    goal_velocity = RewTerm(
        func=mdp.navigation_velocity_profile, weight=3.0
    )
    heading = RewTerm(func=mdp.navigation_heading_alignment, weight=2.0)
    arrival = RewTerm(func=mdp.navigation_arrival, weight=35.0)
    stability = RewTerm(func=mdp.base_stability, weight=-4.0)
    braking = RewTerm(func=mdp.navigation_braking, weight=-4.0)
    policy_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)


@configclass
class NavigationSkillTerminationsCfg:
    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)
    reached = DoneTerm(func=mdp.navigation_reached)
    bad_contact = DoneTerm(
        func=bad_contacts,
        params={
            "sensor_names": HARD_BODY_CONTACT_SENSORS,
            "thresh": 10.0,
        },
    )
    tilt = DoneTerm(
        func=base_tilt,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": 0.65},
    )
    low_height = DoneTerm(
        func=base_height_low,
        params={"asset_cfg": SceneEntityCfg("robot"), "z_min": 0.24},
    )


@configclass
class TANDEMNavigationSkillEnvCfg(TANDEMPhysicalSceneEnvCfg):
    """Natural-start physical gate; it is not a separately registered task."""

    actions: NavigationSkillActionsCfg = NavigationSkillActionsCfg()
    observations: NavigationSkillObservationsCfg = (
        NavigationSkillObservationsCfg()
    )
    rewards: NavigationSkillRewardsCfg = NavigationSkillRewardsCfg()
    terminations: NavigationSkillTerminationsCfg = (
        NavigationSkillTerminationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.episode_length_s = 20.0
        self.commands.locomotion.debug_vis = False
        self.commands.ee_goal.debug_vis = False
