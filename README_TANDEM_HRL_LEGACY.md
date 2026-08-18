# TANDEM-HRL

TANDEM-HRL (Task-and-skill Adaptive Neural Decomposition) is a hierarchical
reinforcement-learning system for the Unitree B2-W, Piper arm, and parallel
gripper. It extends the repository's `ZYB-v0` physical policy with learned task
selection, learned skill composition, and measured rigid-object interaction.

## Architecture

- The task layer scores twelve relation-conditioned task slots and predicts a
  continuous subgoal and termination hazard for the selected slot.
- The skill layer factorizes four motion options and three interaction options.
  Their Cartesian product gives twelve reusable task-conditioned skills.
- Task and skill options have separate semi-Markov state and termination, so
  either layer can switch without resetting the other one.
- One actor-critic contains the task layer, skill layer, gripper action, and the
  migrated 16-dimensional ZYB-v0 physical executor.
- Six rigid objects provide bilateral finger contact, lift, carry, transport,
  release, drop, and placement events from simulator state. A proximity reward
  alone is not counted as grasp success.

The public Gym task IDs use the `TANDEM-HRL` name. The implementation keeps the
internal `TACTIC_HRL` module path so checkpoints produced during development
remain loadable.

## Environments

| Gym task | Use |
| --- | --- |
| `TANDEM-HRL-Unified-v0` | Joint multi-task curriculum training |
| `TANDEM-HRL-Single-Object-v0` | Strict contact-to-placement validation |
| `TANDEM-HRL-Play-v0` | Deterministic evaluation and rendering |
| `TANDEM-HRL-Stress-v0` | Disturbance and robustness evaluation |
| `TANDEM-HRL-Payload-Calibrate-v0` | Payload interaction calibration |

## Checkpoint

The branch contains one 1024-iteration policy shared by every TANDEM-HRL task:
[`checkpoints/TANDEM_HRL/model_1023.pt`](checkpoints/TANDEM_HRL/model_1023.pt).
Its training record, checksum, and fixed evaluation results are listed in the
[checkpoint manifest](checkpoints/TANDEM_HRL/README.md).

## Layout

```text
source/quadruped_arm/quadruped_arm/tasks/manager_based/
  TANDEM_HRL/              public task registration
  TACTIC_HRL/
    agents/                actor-critic, PPO, and checkpoint migration
    mdp/                   mission, action, observation, and reward terms
    TACTIC_HRL_env_cfg.py  robot, objects, and randomization
scripts/
  rsl_rl/                  train, play, and fixed-budget evaluation
  TACTIC_HRL/              smoke tests, checkpoint checks, and plots
```

## Setup

Install the project in the Isaac Lab Python environment:

```bash
python -m pip install -e source/quadruped_arm
export OMNI_KIT_ACCEPT_EULA=YES
```

The policy expects the physical actor contained in the original `ZYB-v0`
checkpoint. To initialize a new hierarchy from that checkpoint:

```bash
python scripts/rsl_rl/train.py \
  --task TANDEM-HRL-Unified-v0 \
  --tactic_init_checkpoint /path/to/zyb_v0/model_1023.pt \
  --training_stage upper \
  --pure_hrl_objectives \
  --num_envs 384 \
  --max_iterations 1024 \
  --save_interval 64 \
  --headless \
  --device cuda:0
```

`--pure_hrl_objectives` disables the CLF/CBF-inspired auxiliary losses and
margin-driven recovery bias. It is used to establish the task/skill
decomposition before adding control-derived objectives.

To continue from a TANDEM-HRL checkpoint while rebuilding optimizer state:

```bash
python scripts/rsl_rl/train.py \
  --task TANDEM-HRL-Unified-v0 \
  --resume_checkpoint /path/to/tandem/model.pt \
  --resume_model_only \
  --resume_iteration_override 0 \
  --training_stage upper \
  --pure_hrl_objectives \
  --num_envs 384 \
  --max_iterations 1024 \
  --save_interval 64 \
  --headless \
  --device cuda:0
```

## Evaluation

Strict single-object evaluation:

```bash
python scripts/rsl_rl/evaluate_checkpoint.py \
  --task TANDEM-HRL-Single-Object-v0 \
  --checkpoint /path/to/model_1023.pt \
  --num_envs 128 \
  --num_steps 600 \
  --seed 271828 \
  --compact_metrics \
  --out_csv results/single_object_summary.csv \
  --per_env_csv results/single_object_per_env.csv \
  --headless \
  --device cuda:0
```

Multi-task composition evaluation uses one checkpoint and a set of task slots:

```bash
python scripts/rsl_rl/evaluate_checkpoint.py \
  --task TANDEM-HRL-Unified-v0 \
  --checkpoint /path/to/model_1023.pt \
  --required_task_sets \
  "0+5;1+6;2+7;0+3+5;1+4+9;0+1+2+3+4+5+6+7+8+9+10+11" \
  --force_curriculum_level 4 \
  --num_envs 120 \
  --num_steps 600 \
  --compact_metrics \
  --out_csv results/composition_summary.csv \
  --per_env_csv results/composition_per_env.csv \
  --headless \
  --device cuda:0
```

Run the numerical architecture and gradient checks without Isaac Sim:

```bash
python scripts/TACTIC_HRL/smoke_actor.py \
  --checkpoint /path/to/zyb_v0/model_1023.pt \
  --device cpu
```
