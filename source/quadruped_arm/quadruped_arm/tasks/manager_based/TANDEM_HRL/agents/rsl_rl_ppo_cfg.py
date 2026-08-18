"""PPO settings for the learned locomotion Skill gate."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TANDEMNavigationSkillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 128
    save_interval = 16
    experiment_name = "tandem_navigation_skill_gate"
    run_name = ""
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # Preserve the original ZYB-v0 wrapper range for wheel velocity actions.
    # The action term applies the narrower leg/support limits itself.
    clip_actions = 100.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.012,
        max_grad_norm=1.0,
    )


@configclass
class TANDEMLocomotionSkillPPORunnerCfg(TANDEMNavigationSkillPPORunnerCfg):
    experiment_name = "tandem_locomotion_skill_gate"
    save_interval = 64
