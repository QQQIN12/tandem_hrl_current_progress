"""Wheel-led lower-body turning with a bounded leg coordinator.

The wheel differential remains the primary motion channel.  During combined
translation and yaw, the leg action adds a small, smoothed left/right hip
offset proportional to ``vx * wz`` and can lower leg impedance within a
bounded envelope.  Pure in-place yaw does not request a persistent body lean.
"""

from isaaclab.utils import configclass

from .mobility_lower_env_cfg import MobilityLowerEnvCfg


@configclass
class TurnCoordLowerEnvCfg(MobilityLowerEnvCfg):
    """Deterministic turn-coordination teacher candidate."""

    def __post_init__(self):
        super().__post_init__()

        # Start from the verified default stance.  The turn coordinator is the
        # only leg-side modification in this diagnostic task.
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = True
        self.actions.leg_pos.turn_command_name = "locomotion"
        self.actions.leg_pos.turn_max_vx = 0.25
        self.actions.leg_pos.turn_max_wz = 0.10
        self.actions.leg_pos.turn_hip_offset_gain = 0.08
        self.actions.leg_pos.turn_hip_offset_limit = 0.08
        self.actions.leg_pos.turn_hip_offset_sign = 1.0
        self.actions.leg_pos.turn_signal_smoothing = 0.20
        self.actions.leg_pos.turn_stiffness_enabled = False

        # Keep the wheel residual small in this teacher diagnostic.  The
        # command-conditioned wheel feed-forward remains authoritative.
        self.actions.wheel_vel.residual_scale = 0.05


@configclass
class TurnCoordSoftLowerEnvCfg(TurnCoordLowerEnvCfg):
    """Turn coordinator with bounded impedance modulation enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_stiffness_enabled = True
        self.actions.leg_pos.turn_stiffness_min_factor = 0.75
        self.actions.leg_pos.turn_stiffness_smoothing = 0.20


@configclass
class TurnCoordOppositeLowerEnvCfg(TurnCoordLowerEnvCfg):
    """Opposite lean sign used only to identify the physical sign convention."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_hip_offset_sign = -1.0


@configclass
class TurnKneeCoordLowerEnvCfg(TurnCoordLowerEnvCfg):
    """Use calf/knee joints as active suspension for turn load transfer."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_coord_joint_names = (
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        )
        self.actions.leg_pos.turn_coord_joint_weights = (1.0, 1.0, 1.0, 1.0)
        self.actions.leg_pos.turn_hip_offset_gain = 0.06
        self.actions.leg_pos.turn_hip_offset_limit = 0.06


@configclass
class TurnKneeCoordOppositeLowerEnvCfg(TurnKneeCoordLowerEnvCfg):
    """Opposite calf/knee sign used for physical sign identification."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_hip_offset_sign = -1.0


@configclass
class TurnGain12LowerEnvCfg(MobilityLowerEnvCfg):
    """Wheel-led diagnostic with stronger calibrated differential yaw."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = False
        self.actions.wheel_vel.residual_scale = 0.05
        self.actions.wheel_vel.turn_speed_gain = 12.0


@configclass
class TurnGain16LowerEnvCfg(MobilityLowerEnvCfg):
    """Wheel-led diagnostic at the near-saturation differential-yaw gain."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = False
        self.actions.wheel_vel.residual_scale = 0.05
        self.actions.wheel_vel.turn_speed_gain = 16.0


@configclass
class TurnGain12FeedbackLowerEnvCfg(TurnGain12LowerEnvCfg):
    """Gain-12 wheel teacher with bounded body-velocity feedback."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.vx_feedback_gain = 0.15
        self.actions.wheel_vel.wz_feedback_gain = 0.50


@configclass
class TurnLoadBalanceLowerEnvCfg(TurnGain12FeedbackLowerEnvCfg):
    """Gain-12 teacher with contact-force-based calf load balancing."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_coord_joint_names = (
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        )
        self.actions.leg_pos.turn_coord_joint_weights = (1.0, 1.0, 1.0, 1.0)
        self.actions.leg_pos.turn_coordination_enabled = False
        self.actions.leg_pos.turn_load_balance_enabled = True
        self.actions.leg_pos.turn_load_balance_gain = 0.16
        self.actions.leg_pos.turn_load_balance_limit = 0.05
        self.actions.leg_pos.turn_load_balance_smoothing = 0.12
        self.actions.leg_pos.turn_load_balance_min_total_force = 40.0


@configclass
class TurnLoadBalanceGentleLowerEnvCfg(TurnLoadBalanceLowerEnvCfg):
    """Reduced load-transfer gain for yaw-preserving ablation testing."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.turn_load_balance_gain = 0.08
        self.actions.leg_pos.turn_load_balance_limit = 0.03
        self.actions.leg_pos.turn_load_balance_smoothing = 0.08


@configclass
class TurnGain12FeedbackWideWheelLowerEnvCfg(TurnGain12FeedbackLowerEnvCfg):
    """Gain-12 feedback teacher with a 6 rad/s wheel safety envelope."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.max_wheel_speed = 6.0


@configclass
class TurnKinematicFeedback4LowerEnvCfg(MobilityLowerEnvCfg):
    """Physical differential mapping with moderate yaw-rate feedback."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = False
        self.actions.wheel_vel.residual_scale = 0.05
        self.actions.wheel_vel.turn_speed_gain = 1.0
        self.actions.wheel_vel.max_ref_wz = 0.65
        self.actions.wheel_vel.vx_feedback_gain = 0.15
        self.actions.wheel_vel.wz_feedback_gain = 4.0


@configclass
class TurnKinematicFeedback8LowerEnvCfg(TurnKinematicFeedback4LowerEnvCfg):
    """Physical differential mapping with stronger bounded yaw feedback."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.wz_feedback_gain = 8.0
