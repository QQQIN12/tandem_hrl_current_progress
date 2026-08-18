"""Stable lower-body teacher configuration for the TANDEM robot.

This is deliberately a separate environment configuration.  The existing
``ZYB-v0`` entry remains the baseline so that experiments can be compared
without silently changing its actuator parameters.
"""

from isaaclab.utils import configclass

from .maniploco_env_cfg import ManipLocoEnvCfg


@configclass
class StableLowerEnvCfg(ManipLocoEnvCfg):
    """Conservative lower-body teacher used before command-range expansion."""

    def __post_init__(self):
        super().__post_init__()

        # The lower-body teacher must not receive moving-arm reaction torques.
        self.actions.arm_ik.max_joint_delta = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False

        # Current-asset calibration from single-wheel response sweeps:
        # positive joint velocity produced positive forward response for all
        # four joints (FL/FR/RL/RR).  Keep the legacy base environment
        # unchanged, but make this teacher agree with the measured actuator
        # directions rather than the older mirrored-rear assumption.
        wheel_signs = (1.0, 1.0, 1.0, 1.0)
        wheel_track = 0.4693
        self.actions.wheel_vel.wheel_dir_signs = wheel_signs
        self.actions.wheel_vel.track_width = wheel_track
        self.actions.wheel_vel.turn_speed_gain = 8.0
        self.rewards.wheel_forward_use.params["wheel_dir_signs"] = wheel_signs
        self.rewards.wheel_turn_support.params["wheel_dir_signs"] = wheel_signs
        self.rewards.wheel_turn_support.params["track_width"] = wheel_track
        self.rewards.wheel_turn_support.params["wz_sign"] = 1.0
        self.rewards.wheel_turn_support.params["wz_clip"] = 0.08
        self.rewards.wheel_forward_use.params["wz_small"] = 0.10
        self.rewards.teacher_ensemble_match.weight = 0.0

        # This is the most stable contact configuration found in the current
        # asset probes.  It is a teacher anchor, not yet a claim of final
        # locomotion performance.
        self.scene.robot.actuators["wheels"].damping = 2.0
        self.scene.robot.actuators["M107-24-2"].stiffness = 300.0
        self.scene.robot.actuators["M107-24-2"].damping = 10.0
        self.scene.robot.actuators["2"].stiffness = 300.0
        self.scene.robot.actuators["2"].damping = 10.0

        # Start with the command box that remained upright in the fixed
        # response sweeps.  Expand only after the hold-out command matrix
        # passes height, tilt, support, and tracking gates.
        self.commands.locomotion.ranges.lin_vel_x = (-0.35, 0.35)
        self.commands.locomotion.ranges.ang_vel_z = (-0.20, 0.20)
