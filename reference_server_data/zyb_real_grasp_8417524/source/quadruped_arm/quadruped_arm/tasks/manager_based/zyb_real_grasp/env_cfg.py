"""ZYB-v0 environments with physical grasp assets."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    ManipLocoEnvCfg,
    ObservationsCfg,
)

from . import mdp
from .event_cfg import ZYBRealGraspEventCfg
from .physics_contract import (
    REAL_GRASP_GRIPPER_DAMPING,
    REAL_GRASP_GRIPPER_STIFFNESS,
    ZYB_V0_ARM_DISTAL_DAMPING,
    ZYB_V0_ARM_DISTAL_STIFFNESS,
    ZYB_V0_ARM_PROXIMAL_DAMPING,
    ZYB_V0_ARM_PROXIMAL_STIFFNESS,
    ZYB_V0_LEG_DAMPING,
    ZYB_V0_LEG_STIFFNESS,
    ZYB_V0_WHEEL_DAMPING,
    ZYB_V0_WHEEL_EFFORT_LIMIT,
    ZYB_V0_WHEEL_JOINT_FRICTION,
    ZYB_V0_WHEEL_STIFFNESS,
    ZYB_V0_WHEEL_VELOCITY_LIMIT,
)
from .scene_cfg import ZYBRealGraspSceneCfg


ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]


@configclass
class ZYBRealGraspSceneActionsCfg(ActionsCfg):
    """Original 16-D leg/wheel interface used for checkpoint-compatible replay."""

    arm_ik = None


@configclass
class ZYBRealGraspActionsCfg(ZYBRealGraspSceneActionsCfg):
    """Full-body baseline interface: 12 legs, 4 wheels, 6 arm, 2 gripper."""

    arm_pos = mdp_core.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        scale=0.30,
        use_default_offset=True,
        preserve_order=True,
    )
    gripper = mdp.ContactHoldGripperActionCfg(asset_name="robot")


@configclass
class ZYBRealGraspObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        grasp_state = ObsTerm(func=mdp.real_grasp_state)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ZYBRealGraspRewardsCfg:
    wrist_object_proximity = RewTerm(func=mdp.wrist_object_proximity, weight=1.0)
    bilateral_grasp = RewTerm(func=mdp.bilateral_grasp, weight=2.5)
    object_lift = RewTerm(func=mdp.object_lift, weight=3.0)
    payload_retention = RewTerm(func=mdp.payload_retention, weight=3.0)
    target_transport_progress = RewTerm(func=mdp.target_transport_progress, weight=2.0)
    placement_quality = RewTerm(func=mdp.placement_quality, weight=5.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=30.0)
    support_quality = RewTerm(func=mdp.support_quality, weight=0.5)
    rear_support_deficit = RewTerm(func=mdp.rear_support_deficit, weight=-1.0)
    vertical_motion = RewTerm(func=mdp.vertical_motion_cost, weight=-0.2)
    action_rate = RewTerm(func=mdp_core.action_rate_l2, weight=-0.002)


@configclass
class ZYBRealGraspTerminationsCfg:
    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)
    excessive_tilt = DoneTerm(func=mdp.excessive_tilt)
    low_base = DoneTerm(func=mdp.low_base)
    success = DoneTerm(func=mdp.placed_object)


def _assert_equal(name: str, actual: float, expected: float) -> None:
    if abs(float(actual) - expected) > 1.0e-8:
        raise RuntimeError(f"{name} changed from ZYB-v0: expected {expected}, got {actual}")


def _validate_robot_contract(robot_cfg) -> None:
    hip = robot_cfg.actuators["M107-24-2"]
    calf = robot_cfg.actuators["2"]
    wheel = robot_cfg.actuators["wheels"]
    arm_proximal = robot_cfg.actuators["arm_proximal"]
    arm_distal = robot_cfg.actuators["arm_distal"]
    for name, actuator in (("hip", hip), ("calf", calf)):
        _assert_equal(f"{name}.stiffness", actuator.stiffness, ZYB_V0_LEG_STIFFNESS)
        _assert_equal(f"{name}.damping", actuator.damping, ZYB_V0_LEG_DAMPING)
    _assert_equal("wheel.effort_limit", wheel.effort_limit, ZYB_V0_WHEEL_EFFORT_LIMIT)
    _assert_equal("wheel.velocity_limit", wheel.velocity_limit, ZYB_V0_WHEEL_VELOCITY_LIMIT)
    _assert_equal("wheel.stiffness", wheel.stiffness, ZYB_V0_WHEEL_STIFFNESS)
    _assert_equal("wheel.damping", wheel.damping, ZYB_V0_WHEEL_DAMPING)
    _assert_equal("wheel.friction", wheel.friction, ZYB_V0_WHEEL_JOINT_FRICTION)
    _assert_equal("arm_proximal.stiffness", arm_proximal.stiffness, ZYB_V0_ARM_PROXIMAL_STIFFNESS)
    _assert_equal("arm_proximal.damping", arm_proximal.damping, ZYB_V0_ARM_PROXIMAL_DAMPING)
    _assert_equal("arm_distal.stiffness", arm_distal.stiffness, ZYB_V0_ARM_DISTAL_STIFFNESS)
    _assert_equal("arm_distal.damping", arm_distal.damping, ZYB_V0_ARM_DISTAL_DAMPING)


@configclass
class ZYBRealGraspSceneEnvCfg(ManipLocoEnvCfg):
    """Physical-scene gate that accepts an existing 16-D ZYB-v0 policy."""

    scene: ZYBRealGraspSceneCfg = ZYBRealGraspSceneCfg(num_envs=16, env_spacing=4.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ZYBRealGraspSceneActionsCfg = ZYBRealGraspSceneActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: ZYBRealGraspEventCfg = ZYBRealGraspEventCfg()
    rewards: ZYBRealGraspRewardsCfg = ZYBRealGraspRewardsCfg()
    terminations: ZYBRealGraspTerminationsCfg = ZYBRealGraspTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.replicate_physics = False
        gripper = self.scene.robot.actuators["gripper"]
        gripper.stiffness = REAL_GRASP_GRIPPER_STIFFNESS
        gripper.damping = REAL_GRASP_GRIPPER_DAMPING
        _validate_robot_contract(self.scene.robot)
        self.commands.locomotion.debug_vis = False
        self.commands.ee_goal.debug_vis = False


@configclass
class ZYBRealGraspEnvCfg(ZYBRealGraspSceneEnvCfg):
    """Trainable 24-D full-body baseline with object-relative observations."""

    actions: ZYBRealGraspActionsCfg = ZYBRealGraspActionsCfg()
    observations: ZYBRealGraspObservationsCfg = ZYBRealGraspObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos.update(
            {
                "joint1": 0.0,
                "joint2": 1.20,
                "joint3": -1.00,
                "joint4": -0.50,
                "joint5": -0.55,
                "joint6": 0.0,
            }
        )


@configclass
class ZYBRealGraspPlayEnvCfg(ZYBRealGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
