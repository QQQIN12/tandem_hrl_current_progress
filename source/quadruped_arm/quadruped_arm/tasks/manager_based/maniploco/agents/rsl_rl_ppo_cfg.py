# agents/rsl_rl_ppo_cfg.py
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ManiPLocoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner config for ManiPLoco (aligned with VBC PPO hyperparams)."""

    # --- runner ---
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 45000
    save_interval = 200
    experiment_name = "maniploco"
    run_name = ""

    # IMPORTANT:
    # 你的 ObservationsCfg 里目前只有 group 名叫 "policy"
    # 所以先把 policy/critic 都映射到 ["policy"]（等你以后拆出 privileged critic group 再改）
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # 可选：wrapper 内部对 action 再 clip 一次（与 env 的 action clip 是两回事）
    # 这个字段在 RslRlVecEnvWrapper 里生效。:contentReference[oaicite:2]{index=2}
    clip_actions = 100.0

    # --- policy network ---
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,#0.8,                 # VBC 里 init_std 是按维度列表；这里用标量初始化（rsl_rl内部是每维可学的std，但同初值）
        actor_obs_normalization=True,      # 先保持简单；如果你想对齐 legged_gym 的 RMS，可改 True
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],#[128],
        critic_hidden_dims=[256, 128],#[128],
        activation="elu",   # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    )

    # --- PPO algorithm ---
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,#0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,#2e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  #None,        # VBC 是 None；如果你的类型检查不允许 None，就设成 0.0
        max_grad_norm=1.0,
    )