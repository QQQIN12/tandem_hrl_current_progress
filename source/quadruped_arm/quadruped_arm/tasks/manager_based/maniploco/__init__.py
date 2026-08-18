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

gym.register(
    id="ZYB-MobilityLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mobility_lower_env_cfg:MobilityLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeLearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeLearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeTeacherLearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeTeacherLearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeGain3Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeGain3LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeGain3ReverseYawLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeGain3ReverseYawLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5FeedbackLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5FeedbackLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5CoordLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5CoordLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5LearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5LearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5TeacherLearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5TeacherLearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5WheelOnlyLearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5WheelOnlyLearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-PhysicalSafeWheel5WheelOnlyTeacherLearningLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.physical_safe_lower_env_cfg:PhysicalSafeWheel5WheelOnlyTeacherLearningLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-WheelOnlyLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_only_lower_env_cfg:WheelOnlyLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnCoordLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnCoordLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnCoordSoftLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnCoordSoftLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnCoordOppositeLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnCoordOppositeLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnKneeCoordLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnKneeCoordLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnKneeCoordOppositeLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnKneeCoordOppositeLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnGain12Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnGain12LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnGain16Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnGain16LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnGain12FeedbackLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnGain12FeedbackLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnLoadBalanceLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnLoadBalanceLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnLoadBalanceGentleLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnLoadBalanceGentleLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnGain12FeedbackWideWheelLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnGain12FeedbackWideWheelLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnKinematicFeedback4Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnKinematicFeedback4LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-TurnKinematicFeedback8Lower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_coord_lower_env_cfg:TurnKinematicFeedback8LowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-WheelTorqueLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_torque_lower_env_cfg:WheelTorqueLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-WheelTorqueHighYawLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_torque_lower_env_cfg:WheelTorqueHighYawLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)

gym.register(
    id="ZYB-StandaloneLower-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standalone_lower_env_cfg:StandaloneLowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManiPLocoPPORunnerCfg",
    },
)
