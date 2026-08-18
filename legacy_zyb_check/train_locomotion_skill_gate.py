"""Train the reusable velocity-conditioned locomotion Skill."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_root", type=Path, required=True)
parser.add_argument("--resume_checkpoint", type=Path)
parser.add_argument("--load_optimizer", action="store_true")
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument(
    "--wheel_coordinate_mode",
    choices=(
        "structured",
        "independent",
        "structured_support",
        "independent_support",
    ),
    default="independent_support",
)
parser.add_argument("--wheel_policy_limit", type=float, default=100.0)
parser.add_argument("--wheel_slew_per_step", type=float, default=100.0)
parser.add_argument("--support_gain", type=float, default=0.55)
parser.add_argument("--support_xy_tracking_scale", type=float, default=1.0)
parser.add_argument(
    "--max_policy_joint_residual",
    type=float,
    default=0.0,
    help="Optional absolute radian cap; zero preserves ZYB per-joint scales.",
)
parser.add_argument("--max_learned_unload_shift_m", type=float, default=0.0)
parser.add_argument(
    "--max_learned_support_relaxation", type=float, default=0.0
)
parser.add_argument(
    "--max_learned_unload_joint_correction", type=float, default=0.14
)
parser.add_argument("--reset_support_output", action="store_true")
parser.add_argument("--stationary_fraction", type=float, default=0.10)
parser.add_argument("--yaw_only_fraction", type=float, default=0.30)
parser.add_argument("--straight_fraction", type=float, default=0.30)
parser.add_argument("--yaw_tracking_weight", type=float, default=4.0)
parser.add_argument("--command_alignment_weight", type=float, default=1.5)
parser.add_argument("--yaw_load_transfer_weight", type=float, default=2.0)
parser.add_argument("--action_rate_weight", type=float, default=-0.0045)
parser.add_argument("--init_noise_std", type=float, default=0.60)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--entropy_coef", type=float, default=0.004)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from rsl_rl.runners import OnPolicyRunner
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from quadruped_arm.tasks.manager_based.TANDEM_HRL.agents import (
    TANDEMLocomotionSkillPPORunnerCfg,
)
from quadruped_arm.tasks.manager_based.TANDEM_HRL.locomotion_skill_env_cfg import (
    TANDEMLocomotionSkillEnvCfg,
)


def _checkpoint_architecture(path: Path) -> tuple[int, list[int]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    layers = []
    for name, value in state.items():
        if name.startswith("actor.") and name.endswith(".weight"):
            index = int(name.split(".")[1])
            layers.append((index, value))
    layers.sort(key=lambda item: item[0])
    if len(layers) < 2:
        raise RuntimeError(f"Cannot infer actor architecture from {path}")
    observation_dim = int(layers[0][1].shape[1])
    hidden_dims = [int(value.shape[0]) for _, value in layers[:-1]]
    return observation_dim, hidden_dims


def _load_checkpoint(
    runner: OnPolicyRunner,
    path: Path,
    load_optimizer: bool,
    reset_support_output: bool,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    current_state = runner.alg.policy.state_dict()
    old_actor_layers = [
        (int(name.split(".")[1]), name)
        for name in state
        if name.startswith("actor.") and name.endswith(".weight")
    ]
    new_actor_layers = [
        (int(name.split(".")[1]), name)
        for name in current_state
        if name.startswith("actor.") and name.endswith(".weight")
    ]
    old_output_index, old_output_name = max(old_actor_layers)
    new_output_index, new_output_name = max(new_actor_layers)
    old_output_dim = int(state[old_output_name].shape[0])
    new_output_dim = int(current_state[new_output_name].shape[0])

    if old_output_dim != new_output_dim:
        # Existing locomotion checkpoints have the original 16 outputs. The
        # new policy appends four support outputs while preserving all old
        # leg and wheel rows byte-for-byte.
        if old_output_dim != 16 or new_output_dim != 20:
            raise RuntimeError(
                f"Unsupported action migration {old_output_dim} -> "
                f"{new_output_dim} for {path}"
            )
        old_weight = state[old_output_name]
        old_bias = state[f"actor.{old_output_index}.bias"]
        new_weight = torch.zeros(
            (new_output_dim, old_weight.shape[1]), dtype=old_weight.dtype
        )
        new_bias = torch.zeros(new_output_dim, dtype=old_bias.dtype)
        new_weight[:old_output_dim] = old_weight
        new_bias[:old_output_dim] = old_bias
        state[new_output_name] = new_weight
        state[f"actor.{new_output_index}.bias"] = new_bias
        state["std"] = torch.cat(
            (
                state["std"],
                torch.full((4,), 0.05, dtype=state["std"].dtype),
            ),
            dim=0,
        )

    if reset_support_output:
        state[new_output_name][16:20] = 0.0
        state[f"actor.{new_output_index}.bias"][16:20] = 0.0
        state["std"][16:20] = 0.05

    if old_output_dim == new_output_dim and not reset_support_output:
        runner.load(
            str(path),
            load_optimizer=load_optimizer,
            map_location=runner.device,
        )
        return

    # Optimizer state is not shape-compatible after adding the support head.
    runner.alg.policy.load_state_dict(state)
    runner.current_learning_iteration = checkpoint["iter"]


def main() -> None:
    observation_dim = 67
    hidden_dims = [256, 128]
    if args.resume_checkpoint is not None:
        observation_dim, hidden_dims = _checkpoint_architecture(
            args.resume_checkpoint
        )
    history_length = args.history_length
    if history_length <= 0:
        if observation_dim % 67 != 0:
            raise ValueError(
                f"Checkpoint observation dimension {observation_dim} is not "
                "compatible with the 67-D locomotion state"
            )
        history_length = observation_dim // 67

    command_fraction = (
        args.stationary_fraction
        + args.yaw_only_fraction
        + args.straight_fraction
    )
    if not 0.0 <= command_fraction <= 1.0:
        raise ValueError("Command fractions must sum to a value in [0, 1]")

    env_cfg = TANDEMLocomotionSkillEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.observations.policy.state.history_length = history_length
    action_cfg = env_cfg.actions.leg_pos
    action_cfg.wheel_coordinate_mode = args.wheel_coordinate_mode
    action_cfg.wheel_policy_limit = args.wheel_policy_limit
    action_cfg.wheel_slew_per_step = args.wheel_slew_per_step
    action_cfg.support_gain = args.support_gain
    action_cfg.support_xy_tracking_scale = args.support_xy_tracking_scale
    action_cfg.max_policy_joint_residual = args.max_policy_joint_residual
    action_cfg.max_learned_unload_shift_m = args.max_learned_unload_shift_m
    action_cfg.max_learned_support_relaxation = (
        args.max_learned_support_relaxation
    )
    action_cfg.max_learned_unload_joint_correction = (
        args.max_learned_unload_joint_correction
    )
    command_cfg = env_cfg.commands.locomotion
    command_cfg.stationary_fraction = args.stationary_fraction
    command_cfg.yaw_only_fraction = args.yaw_only_fraction
    command_cfg.straight_fraction = args.straight_fraction
    env_cfg.rewards.yaw_tracking.weight = args.yaw_tracking_weight
    env_cfg.rewards.command_alignment.weight = args.command_alignment_weight
    env_cfg.rewards.yaw_load_transfer.weight = args.yaw_load_transfer_weight
    env_cfg.rewards.action_rate.weight = args.action_rate_weight

    runner_cfg = TANDEMLocomotionSkillPPORunnerCfg()
    runner_cfg.device = args.device
    runner_cfg.seed = args.seed
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.policy.init_noise_std = args.init_noise_std
    runner_cfg.policy.actor_hidden_dims = hidden_dims
    runner_cfg.policy.critic_hidden_dims = hidden_dims
    runner_cfg.algorithm.learning_rate = args.learning_rate
    runner_cfg.algorithm.entropy_coef = args.entropy_coef
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_root / f"{stamp}_seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "gate_config.json").write_text(
        json.dumps(
            {
                "gate": "velocity-conditioned learned locomotion Skill",
                "counts_as_final_hrl_training": False,
                "seed": args.seed,
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "resume_checkpoint": (
                    str(args.resume_checkpoint)
                    if args.resume_checkpoint is not None
                    else None
                ),
                "load_optimizer": args.load_optimizer,
                "action_dim": 20,
                "history_length": history_length,
                "leg_residual_limit_rad": args.max_policy_joint_residual,
                "wheel_action": args.wheel_coordinate_mode,
                "wheel_policy_limit": args.wheel_policy_limit,
                "wheel_slew_per_step": args.wheel_slew_per_step,
                "executor": "payload-aware support WBC",
                "support_gain": args.support_gain,
                "support_xy_tracking_scale": args.support_xy_tracking_scale,
                "max_learned_unload_shift_m": args.max_learned_unload_shift_m,
                "max_learned_support_relaxation": (
                    args.max_learned_support_relaxation
                ),
                "max_learned_unload_joint_correction": (
                    args.max_learned_unload_joint_correction
                ),
                "reset_support_output": args.reset_support_output,
                "contact_observation": (
                    "binary contact plus 0.20 relative-load residual"
                ),
                "yaw_load_transfer_target_effective_support": 3.0,
                "command_fractions": {
                    "stationary": args.stationary_fraction,
                    "yaw_only": args.yaw_only_fraction,
                    "straight": args.straight_fraction,
                    "arc": 1.0 - command_fraction,
                },
                "reward_weights": {
                    "yaw_tracking": args.yaw_tracking_weight,
                    "command_alignment": args.command_alignment_weight,
                    "yaw_load_transfer": args.yaw_load_transfer_weight,
                    "action_rate": args.action_rate_weight,
                },
                "policy_hidden_dims": hidden_dims,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped,
        runner_cfg.to_dict(),
        log_dir=str(log_dir),
        device=runner_cfg.device,
    )
    if args.resume_checkpoint is not None:
        _load_checkpoint(
            runner,
            args.resume_checkpoint,
            args.load_optimizer,
            args.reset_support_output,
        )
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
