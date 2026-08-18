"""TACTIC-HRL task registration."""

import gymnasium as gym
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from . import agents
from .agents import TACTICActorCritic, TACTICPPO


rsl_on_policy_runner.TACTICActorCritic = TACTICActorCritic
rsl_on_policy_runner.TACTICPPO = TACTICPPO

RUNNER = f"{agents.__name__}.rsl_rl_TACTIC_HRL_cfg:TACTICRunnerCfg"
ENV = f"{__name__}.TACTIC_HRL_env_cfg"

gym.register(
    id="TACTIC-HRL-Unified-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV}:TACTICEnvCfg",
        "rsl_rl_cfg_entry_point": RUNNER,
    },
)
gym.register(
    id="TACTIC-HRL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV}:TACTICPlayEnvCfg",
        "rsl_rl_cfg_entry_point": RUNNER,
    },
)
gym.register(
    id="TACTIC-HRL-Stress-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV}:TACTICStressEnvCfg",
        "rsl_rl_cfg_entry_point": RUNNER,
    },
)
gym.register(
    id="TACTIC-HRL-Payload-Calibrate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{ENV}:TACTICPayloadCalibrationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": RUNNER,
    },
)
