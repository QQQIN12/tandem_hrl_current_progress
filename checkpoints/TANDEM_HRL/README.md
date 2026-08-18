# TANDEM-HRL shared checkpoint

`model_1023.pt` is the single policy produced by the 1024-iteration unified
task/skill training run. The checkpoint is used unchanged for every registered
TANDEM-HRL task.

## Training record

| Item | Value |
| --- | --- |
| Environment | `TANDEM-HRL-Unified-v0` |
| Iterations | 1024 |
| Parallel environments | 384 |
| Environment steps | 9,437,184 |
| Seed | 47 |
| Stage | learned task and skill layers |
| Physical executor | frozen |
| SHA-256 | `24f3f30a247ec824b846fe32080f8183c776aeea1d0c1de672cf293f6fc18096` |

Control-derived auxiliary losses were disabled with
`--pure_hrl_objectives`. Relative to the initial checkpoint, the learned task
parameters changed by 26.36% in relative L2 norm and the skill parameters by
9.49%. The physical executor changed by exactly zero.

## Fixed evaluations

| Evaluation | Result |
| --- | --- |
| Strict single object, 2 seeds, 128 environments, 600 steps | 27 contact, 16 lift, 16 carry, 16 transport, 2 valid placement completions |
| Six task compositions, 60 environments, 450 steps | 54 task switches, 3 carry, 1 transport, 0 complete compositions |
| Twelve task slots, 120 environments, 600 steps | 114 task switches, 4 carry, 1 transport; slot 11 completed in 10/10 environments |
| Strict-chain base tracking error | 0.0541 m mean |
| Strict-chain rear-pair airborne fraction | 0.80% of evaluated steps |
| Strict-chain all-feet-airborne fraction | 0.047% of evaluated steps |

The strict placement success rate is 2/128 (1.56%). This checkpoint establishes
one learned hierarchy and a real rigid-object grasp chain, but far-start
delivery and multi-task completion remain open training targets.
