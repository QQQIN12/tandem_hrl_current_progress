"""Calibrated wheel-only lower-body candidate.

This stage keeps the proven default leg posture and exposes only the
command-conditioned wheel channel.  It is an intermediate diagnostic and not
the final walking/gait policy: it separates wheel-command authority from
learned leg residuals before the latter are reintroduced.
"""

from isaaclab.utils import configclass

from .mobility_lower_env_cfg import MobilityLowerEnvCfg


@configclass
class WheelOnlyLowerEnvCfg(MobilityLowerEnvCfg):
    """Mobility calibration with learned leg residuals frozen."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
