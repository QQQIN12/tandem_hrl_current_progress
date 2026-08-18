"""Minimal environment used to validate the new physical scene contract."""

from __future__ import annotations

from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    ManipLocoEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

from .event_cfg import TANDEMMainlineEventCfg
from .physics_contract import (
    MAINLINE_GRIPPER_DAMPING,
    MAINLINE_GRIPPER_STIFFNESS,
)
from .scene_cfg import TANDEMMainlineSceneCfg


@configclass
class TANDEMPhysicalSceneActionsCfg(ActionsCfg):
    """ZYB-v0 locomotion actions with the legacy automatic arm IK disabled."""

    arm_ik = None


@configclass
class TANDEMPhysicalSceneEnvCfg(ManipLocoEnvCfg):
    """Non-training environment for scene and reset validation."""

    scene: TANDEMMainlineSceneCfg = TANDEMMainlineSceneCfg(
        num_envs=16,
        env_spacing=4.0,
    )
    commands: CommandsCfg = CommandsCfg()
    actions: TANDEMPhysicalSceneActionsCfg = TANDEMPhysicalSceneActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: TANDEMMainlineEventCfg = TANDEMMainlineEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.54)
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
        gripper = self.scene.robot.actuators["gripper"]
        gripper.stiffness = MAINLINE_GRIPPER_STIFFNESS
        gripper.damping = MAINLINE_GRIPPER_DAMPING
        self.commands.locomotion.debug_vis = False
        self.commands.ee_goal.debug_vis = False
