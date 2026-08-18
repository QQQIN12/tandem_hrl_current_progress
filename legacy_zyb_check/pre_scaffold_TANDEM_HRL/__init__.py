"""Public task registrations for TANDEM-HRL.

TANDEM-HRL stands for Task-and-skill Adaptive Neural Decomposition. The
implementation remains in the TACTIC_HRL compatibility namespace so
checkpoints trained during development retain their original class paths.
"""

import gymnasium as gym

from ..TACTIC_HRL import agents


RUNNER = f"{agents.__name__}.rsl_rl_TACTIC_HRL_cfg:TACTICRunnerCfg"
ENV = "quadruped_arm.tasks.manager_based.TACTIC_HRL.TACTIC_HRL_env_cfg"


def _register(task_id: str, env_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{ENV}:{env_cfg}",
            "rsl_rl_cfg_entry_point": RUNNER,
        },
    )


_register("TANDEM-HRL-Unified-v0", "TACTICEnvCfg")
_register("TANDEM-HRL-Play-v0", "TACTICPlayEnvCfg")
_register(
    "TANDEM-HRL-Single-Object-v0",
    "TACTICSingleObjectCurriculumEnvCfg",
)
_register("TANDEM-HRL-Stress-v0", "TACTICStressEnvCfg")
_register(
    "TANDEM-HRL-Payload-Calibrate-v0",
    "TACTICPayloadCalibrationEnvCfg",
)
