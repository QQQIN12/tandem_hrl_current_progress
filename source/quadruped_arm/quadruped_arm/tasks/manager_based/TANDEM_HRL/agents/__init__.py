"""Training configuration for TANDEM-HRL diagnostic gates."""

from .rsl_rl_ppo_cfg import (
    TANDEMLocomotionSkillPPORunnerCfg,
    TANDEMNavigationSkillPPORunnerCfg,
)

__all__ = [
    "TANDEMLocomotionSkillPPORunnerCfg",
    "TANDEMNavigationSkillPPORunnerCfg",
]
