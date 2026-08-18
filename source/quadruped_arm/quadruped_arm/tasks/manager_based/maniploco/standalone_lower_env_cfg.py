"""Deployable lower-body configuration with an intrinsic safety envelope.

The external multi-teacher wrapper is useful during training, but a frozen
checkpoint must remain bounded when evaluated or deployed without that proxy.
This configuration therefore enables the leg-authority gate and strengthens
the existing wheel tilt gate while preserving the MobilityLower actuator
settings for an apples-to-apples comparison.
"""

from isaaclab.utils import configclass

from .mobility_lower_env_cfg import MobilityLowerEnvCfg


@configclass
class StandaloneLowerEnvCfg(MobilityLowerEnvCfg):
    """Mobility lower body with intrinsic tilt/height recovery authority."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.safety_tilt_soft_limit = 0.16
        self.actions.leg_pos.safety_tilt_gate_width = 0.04
        self.actions.leg_pos.safety_min_height = 0.30
        self.actions.leg_pos.safety_height_gate_width = 0.04

        # The standalone training pass uses a neutral safety shield rather
        # than the invalid archived ZYB action target.  Do not reward matching
        # the zero residual itself; command tracking and physical safety must
        # supply the learning signal as the external shield is annealed away.
        self.rewards.teacher_ensemble_match.weight = 0.0

        # The wheel action already has a tilt-dependent feed-forward gate.
        # Increase its authority only in this standalone configuration so a
        # large learned leg transient cannot keep driving the base after tilt.
        self.actions.wheel_vel.tilt_gate_gain = 8.0
