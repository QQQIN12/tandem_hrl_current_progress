"""Gym registrations for the ZYB-v0 real-grasp extension."""

import gymnasium as gym

from ..maniploco import agents


gym.register(
    id="ZYB-Real-Grasp-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZYBRealGraspSceneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-Real-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZYBRealGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-Real-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZYBRealGraspPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)
