"""Mobility-oriented lower-body teacher candidate.

The stable teacher remains the safety anchor.  This configuration deliberately
uses a separate, slightly more compliant leg drive so PPO can learn the leg
motion needed to roll and turn instead of only holding a rigid stance.
"""

from isaaclab.utils import configclass

from .maniploco_env_cfg import ManipLocoEnvCfg


@configclass
class MobilityLowerEnvCfg(ManipLocoEnvCfg):
    """Candidate environment for learning command-conditioned leg motion."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_ik.max_joint_delta = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False

        # Match the reward's wheel convention to the measured current asset.
        wheel_signs = (1.0, 1.0, -1.0, -1.0)
        wheel_track = 0.4693
        self.actions.wheel_vel.wheel_dir_signs = wheel_signs
        self.actions.wheel_vel.track_width = wheel_track
        self.rewards.wheel_forward_use.params["wheel_dir_signs"] = wheel_signs
        self.rewards.wheel_turn_support.params["wheel_dir_signs"] = wheel_signs
        self.rewards.wheel_turn_support.params["track_width"] = wheel_track
        self.rewards.wheel_turn_support.params["wz_sign"] = -1.0

        # Intermediate values are selected from the current asset sweep:
        # safer than nominal 160/5 during turning, but less rigid than the
        # 300/10 stability anchor.
        self.scene.robot.actuators["wheels"].damping = 2.0
        self.scene.robot.actuators["M107-24-2"].stiffness = 220.0
        self.scene.robot.actuators["M107-24-2"].damping = 8.0
        self.scene.robot.actuators["2"].stiffness = 220.0
        self.scene.robot.actuators["2"].damping = 8.0

        # The current contact response reverses the standard positive-yaw
        # convention for the ideal differential pattern.  Keep this correction
        # explicit and easy to ablate.
        self.actions.wheel_vel.wz_sign = -1.0
        self.actions.wheel_vel.residual_scale = 0.25

        self.commands.locomotion.ranges.lin_vel_x = (-0.35, 0.35)
        self.commands.locomotion.ranges.ang_vel_z = (-0.35, 0.35)
