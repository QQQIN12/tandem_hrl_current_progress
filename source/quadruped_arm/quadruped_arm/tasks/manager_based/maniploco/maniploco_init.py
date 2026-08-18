# tasks/manager_based/maniploco/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="ZYB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maniploco_env_cfg:ManipLocoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PointFoot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maniploco_point_foot_env_cfg:ManipLocoPointFootEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)


gym.register(
    id="ZYB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maniploco_env_cfg:ManipLocoPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PointFoot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maniploco_point_foot_env_cfg:ManipLocoPointFootPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-StableLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stable_lower_env_cfg:StableLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)
