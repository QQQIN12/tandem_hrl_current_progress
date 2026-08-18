"""MDP terms used by the privileged-state TANDEM-HRL mainline."""

from .actions import SupportWBCAction, SupportWBCActionCfg
from .commands import SkillVelocityCommand, SkillVelocityCommandCfg
from .observations import privileged_locomotion_state, privileged_navigation_state
from .rewards import (
    action_rate_l2,
    base_stability,
    leg_residual_l2,
    locomotion_command_alignment,
    locomotion_velocity_tracking,
    locomotion_yaw_tracking,
    yaw_load_redistribution,
    navigation_arrival,
    navigation_braking,
    navigation_heading_alignment,
    navigation_progress,
    navigation_target_pose,
    navigation_velocity_profile,
    support_count,
    support_allocation_l2,
    support_fraction,
    wheel_coordinate_l2,
)
from .terminations import navigation_reached

__all__ = [
    "SupportWBCAction",
    "SupportWBCActionCfg",
    "SkillVelocityCommand",
    "SkillVelocityCommandCfg",
    "action_rate_l2",
    "base_stability",
    "leg_residual_l2",
    "locomotion_command_alignment",
    "locomotion_velocity_tracking",
    "locomotion_yaw_tracking",
    "yaw_load_redistribution",
    "navigation_arrival",
    "navigation_braking",
    "navigation_heading_alignment",
    "navigation_progress",
    "navigation_reached",
    "navigation_target_pose",
    "navigation_velocity_profile",
    "privileged_navigation_state",
    "support_count",
    "support_allocation_l2",
    "support_fraction",
    "wheel_coordinate_l2",
]
