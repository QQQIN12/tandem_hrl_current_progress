"""LR_HRL task registrations.

Existing baseline tasks remain registered by the `maniploco` package.  This
package registers the LR_HRL tasks and the flat baseline controls used for the
same benchmark families.
"""

import gymnasium as gym

from . import agents


def _register(id_name: str, env_cfg: str, runner_cfg: str):
    gym.register(
        id=id_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": runner_cfg},
    )


HRL_RUNNER = f"{agents.__name__}.rsl_rl_LR_HRL_cfg:LRHRLPPORunnerCfg"
BASELINE_RUNNER = f"{agents.__name__}.rsl_rl_LR_HRL_cfg:LRBaselinePPORunnerCfg"
ENV = f"{__name__}.LR_HRL_env_cfg"


_register("LR-HRL-v0", f"{ENV}:LRHRLEnvCfg", HRL_RUNNER)
_register("LR-HRL-Route-v0", f"{ENV}:LRHRLRouteEnvCfg", HRL_RUNNER)
_register("LR-HRL-Slalom-v0", f"{ENV}:LRHRLSlalomEnvCfg", HRL_RUNNER)
_register("LR-HRL-Narrow-v0", f"{ENV}:LRHRLNarrowEnvCfg", HRL_RUNNER)
_register("LR-HRL-Manip-v0", f"{ENV}:LRHRLManipEnvCfg", HRL_RUNNER)
_register("LR-HRL-Grasp-v0", f"{ENV}:LRHRLGraspEnvCfg", HRL_RUNNER)
_register("LR-HRL-Recovery-v0", f"{ENV}:LRHRLRecoveryEnvCfg", HRL_RUNNER)
_register("LR-HRL-Play-v0", f"{ENV}:LRHRLPlayEnvCfg", HRL_RUNNER)

_register("LR-Baseline-v0", f"{ENV}:LRBaselineEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Route-v0", f"{ENV}:LRBaselineRouteEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Slalom-v0", f"{ENV}:LRBaselineSlalomEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Narrow-v0", f"{ENV}:LRBaselineNarrowEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Manip-v0", f"{ENV}:LRBaselineManipEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Grasp-v0", f"{ENV}:LRBaselineGraspEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Recovery-v0", f"{ENV}:LRBaselineRecoveryEnvCfg", BASELINE_RUNNER)
_register("LR-Baseline-Play-v0", f"{ENV}:LRBaselinePlayEnvCfg", BASELINE_RUNNER)
