"""Physically scaled, bounded lower-body teacher for TANDEM.

This task is intentionally separate from the earlier gain-8/gain-12
diagnostics.  The command box is mapped with ordinary differential-drive
kinematics, while the wheel actuator is given conservative simulation guards
so a reference-speed limit is not mistaken for an actual-speed guarantee.
"""

from isaaclab.utils import configclass

from .mobility_lower_env_cfg import MobilityLowerEnvCfg


@configclass
class PhysicalSafeLowerEnvCfg(MobilityLowerEnvCfg):
    """Low-speed physical wheel teacher with bounded actuation."""

    def __post_init__(self):
        super().__post_init__()

        # Freeze the arm and leg residual while identifying the safe wheel
        # command path.  Leg posture remains a strong support anchor.
        self.actions.arm_ik.max_joint_delta = 0.0
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = False

        # Current asset calibration: positive joint velocity is the forward
        # direction for all four wheels.  Use the measured track width.
        self.actions.wheel_vel.wheel_dir_signs = (1.0, 1.0, 1.0, 1.0)
        self.actions.wheel_vel.track_width = 0.4693
        self.actions.wheel_vel.turn_speed_gain = 1.0
        self.actions.wheel_vel.max_ref_vx = 0.25
        self.actions.wheel_vel.max_ref_wz = 0.10
        self.actions.wheel_vel.max_wheel_speed = 3.0
        self.actions.wheel_vel.max_wheel_accel = 4.0
        self.actions.wheel_vel.residual_scale = 0.05
        self.actions.wheel_vel.vx_feedback_gain = 0.0
        self.actions.wheel_vel.wz_feedback_gain = 0.0
        self.actions.wheel_vel.turn_breakaway_wz = 0.0

        # These are simulation safety envelopes, not claims about the real
        # motor limits.  The physical reference is <=2.49 rad/s; the small
        # margin allows tracking without retaining the old 5 rad/s clamp.
        self.scene.robot.actuators["wheels"].effort_limit_sim = 8.0
        self.scene.robot.actuators["wheels"].velocity_limit_sim = 4.0
        self.scene.robot.actuators["wheels"].damping = 6.0

        # Keep the support joints at the stable-teacher gains during this
        # actuator identification stage.
        self.scene.robot.actuators["M107-24-2"].stiffness = 300.0
        self.scene.robot.actuators["M107-24-2"].damping = 10.0
        self.scene.robot.actuators["2"].stiffness = 300.0
        self.scene.robot.actuators["2"].damping = 10.0

        self.rewards.wheel_forward_use.params["wheel_dir_signs"] = (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self.rewards.wheel_turn_support.params["wheel_dir_signs"] = (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self.rewards.wheel_turn_support.params["track_width"] = 0.4693
        self.rewards.wheel_turn_support.params["wz_sign"] = 1.0
        self.rewards.teacher_ensemble_match.weight = 0.0

        self.commands.locomotion.ranges.lin_vel_x = (-0.25, 0.25)
        self.commands.locomotion.ranges.ang_vel_z = (-0.10, 0.10)


@configclass
class PhysicalSafeLearningLowerEnvCfg(PhysicalSafeLowerEnvCfg):
    """Safe teacher envelope with a small learnable leg residual."""

    def __post_init__(self):
        super().__post_init__()

        # Reintroduce only a bounded leg residual.  The safety gate blends
        # back to the default stance when tilt or height margin deteriorates.
        self.actions.leg_pos.max_policy_residual = 0.08
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.safety_tilt_soft_limit = 0.10
        self.actions.leg_pos.safety_tilt_gate_width = 0.03
        self.actions.leg_pos.safety_min_height = 0.30
        self.actions.leg_pos.safety_height_gate_width = 0.04

        # Keep the wheel residual smaller while the leg branch is learning;
        # the command-conditioned physical wheel teacher remains primary.
        self.actions.wheel_vel.residual_scale = 0.03


@configclass
class PhysicalSafeTeacherLearningLowerEnvCfg(PhysicalSafeLearningLowerEnvCfg):
    """Teacher-shielded safe student for bounded lower-body adaptation."""

    def __post_init__(self):
        super().__post_init__()
        # MultiTeacherVecEnv publishes the ZYB, conservative, and neutral
        # candidates.  This reward makes the shielded student stay near the
        # selected candidate while still allowing the base tracking rewards
        # to refine the residual policy.
        self.rewards.teacher_ensemble_match.weight = 0.25
        self.rewards.teacher_ensemble_match.params["sigma"] = 0.15


@configclass
class PhysicalSafeGain3LowerEnvCfg(PhysicalSafeLowerEnvCfg):
    """Bounded yaw-authority ablation below the rejected gain-8 regime."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.turn_speed_gain = 3.0


@configclass
class PhysicalSafeGain3ReverseYawLowerEnvCfg(PhysicalSafeGain3LowerEnvCfg):
    """Gain-3 sign ablation for the measured body-yaw convention."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.wz_sign = -1.0


@configclass
class PhysicalSafeWheel5LowerEnvCfg(PhysicalSafeLowerEnvCfg):
    """Safety-instrumented diagnostic that permits a 5 rad/s wheel target."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the measured low-level command box, but deliberately exercise
        # the requested 5 rad/s wheel-reference regime.  The gain is chosen
        # so the high wheel in the (+0.25 m/s, +0.10 rad/s) corner is about
        # 5 rad/s before the reference clamp.
        self.actions.wheel_vel.turn_speed_gain = 12.8
        self.actions.wheel_vel.max_wheel_speed = 5.0
        self.actions.wheel_vel.max_wheel_accel = 4.0
        self.actions.wheel_vel.residual_scale = 0.0
        self.actions.wheel_vel.actual_speed_limit = 5.0
        # Once measured speed crosses 5 rad/s, command zero effective wheel
        # speed until it comes back inside the envelope.  The previous 0.5
        # rad/s margin still allowed a loaded wheel to overshoot appreciably.
        self.actions.wheel_vel.actual_speed_brake_margin = 5.0

        # Keep the torque and posture envelope from PhysicalSafeLower, while
        # allowing the requested reference to reach the actuator path.  The
        # evaluator records actual velocity and applied torque separately.
        self.scene.robot.actuators["wheels"].velocity_limit_sim = 6.0


@configclass
class PhysicalSafeWheel5FeedbackLowerEnvCfg(PhysicalSafeWheel5LowerEnvCfg):
    """5-rad/s reference with bounded body-velocity feedback."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.vx_feedback_gain = 0.15
        self.actions.wheel_vel.wz_feedback_gain = 0.50


@configclass
class PhysicalSafeWheel5CoordLowerEnvCfg(PhysicalSafeWheel5LowerEnvCfg):
    """5-rad/s wheel command with bounded command-conditioned hip support."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_coordination_enabled = True
        self.actions.leg_pos.turn_command_name = "locomotion"
        self.actions.leg_pos.turn_max_vx = 0.25
        self.actions.leg_pos.turn_max_wz = 0.10
        self.actions.leg_pos.turn_hip_offset_gain = 0.08
        self.actions.leg_pos.turn_hip_offset_limit = 0.08
        self.actions.leg_pos.turn_hip_offset_sign = 1.0
        self.actions.leg_pos.turn_signal_smoothing = 0.20
        self.actions.leg_pos.turn_stiffness_enabled = False


@configclass
class PhysicalSafeWheel5LearningLowerEnvCfg(PhysicalSafeWheel5LowerEnvCfg):
    """Learnable lower body under the requested 5 rad/s wheel envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.08
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.safety_tilt_soft_limit = 0.10
        self.actions.leg_pos.safety_tilt_gate_width = 0.03
        self.actions.leg_pos.safety_min_height = 0.30
        self.actions.leg_pos.safety_height_gate_width = 0.04
        self.actions.wheel_vel.residual_scale = 0.03


@configclass
class PhysicalSafeWheel5TeacherLearningLowerEnvCfg(PhysicalSafeWheel5LearningLowerEnvCfg):
    """Multi-teacher student under the requested 5 rad/s envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.teacher_ensemble_match.weight = 0.25
        self.rewards.teacher_ensemble_match.params["sigma"] = 0.15


@configclass
class PhysicalSafeWheel5WheelOnlyLearningLowerEnvCfg(PhysicalSafeWheel5LowerEnvCfg):
    """5-rad/s student with frozen support legs and learnable wheel residual."""

    def __post_init__(self):
        super().__post_init__()
        # Keep the leg branch at the explicitly frozen default posture.  Only
        # the four wheel residuals can adapt the command response.
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.wheel_vel.residual_scale = 0.03


@configclass
class PhysicalSafeWheel5WheelOnlyTeacherLearningLowerEnvCfg(
    PhysicalSafeWheel5WheelOnlyLearningLowerEnvCfg
):
    """Teacher-shielded wheel-only student under the 5-rad/s envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.teacher_ensemble_match.weight = 0.25
        self.rewards.teacher_ensemble_match.params["sigma"] = 0.15
