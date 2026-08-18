# LR_HRL 复现运行教程

本分支在原有 manager-based baseline 的基础上新增 `LR_HRL` 任务包。代码包含层级任务/技能命令接口、同任务设置下的 flat baseline 对照任务、六组 1024 iterations 训练队列脚本，以及训练日志解析和曲线绘制脚本。

## 1. 拉取代码

使用 `LR_HRL` 分支：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch LR_HRL \
  https://github.com/yibozhou0505/quadruped_arm.git \
  /root/gpufree-data/quadruped_arm_LR_HRL
cd /root/gpufree-data/quadruped_arm_LR_HRL
git lfs pull
```

如果仓库是私有仓库，需要使用具有仓库读取权限的 GitHub 账号或 token。建议将工程放在数据盘，例如 `/root/gpufree-data`，因为 Isaac Lab 资产、训练日志和 checkpoint 会占用较多空间。

## 2. 环境配置

以下命令默认 Isaac Lab Python 环境位于：

```text
/root/gpufree-data/envs/isaaclab/bin/python
```

启动训练或回放前，需要设置 Isaac Sim EULA 标志和工程源码路径：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/root/gpufree-data/quadruped_arm_LR_HRL/source/quadruped_arm:${PYTHONPATH:-}
```

安装工程扩展：

```bash
cd /root/gpufree-data/quadruped_arm_LR_HRL
/root/gpufree-data/envs/isaaclab/bin/python -m pip install -e source/quadruped_arm
/root/gpufree-data/envs/isaaclab/bin/python -m compileall \
  source/quadruped_arm/quadruped_arm/tasks/manager_based/LR_HRL
```

当前 `LR_HRL` 不依赖外部预训练 skill 库。task/skill decomposition 通过工程内的 command packet、phase/skill ID、observation terms 和 reward terms 实现。

## 3. 任务 ID

原 baseline 任务仍由已有的 `maniploco` 包注册。新增的 LR_HRL benchmark 任务如下：

| 任务类型 | LR_HRL 任务 | Flat baseline 对照任务 |
|---|---|---|
| 路线跟踪 | `LR-HRL-Route-v0` | `LR-Baseline-Route-v0` |
| 多障碍绕行 | `LR-HRL-Slalom-v0` | `LR-Baseline-Slalom-v0` |
| 狭窄通道 | `LR-HRL-Narrow-v0` | `LR-Baseline-Narrow-v0` |
| 移动操作 | `LR-HRL-Manip-v0` | `LR-Baseline-Manip-v0` |
| 夹爪预抓取 | `LR-HRL-Grasp-v0` | `LR-Baseline-Grasp-v0` |
| 扰动恢复 | `LR-HRL-Recovery-v0` | `LR-Baseline-Recovery-v0` |

`LR-Baseline-*` 任务使用与 `LR_HRL` 相同的路线、障碍物、移动操作和夹爪目标命令设置，但不接收 `LR_HRL` 使用的显式 `tau_down/tau_up` packet 观测。

## 4. Smoke Test

先运行一个短 HRL 任务，确认环境可以正常初始化并进入 PPO 训练：

```bash
cd /root/gpufree-data/quadruped_arm_LR_HRL
/root/gpufree-data/envs/isaaclab/bin/python scripts/rsl_rl/train.py \
  --task LR-HRL-Route-v0 \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

再运行一个短 baseline 对照任务：

```bash
cd /root/gpufree-data/quadruped_arm_LR_HRL
/root/gpufree-data/envs/isaaclab/bin/python scripts/rsl_rl/train.py \
  --task LR-Baseline-Grasp-v0 \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

上述两条命令应能够完成 Isaac Lab 初始化、环境构建、RSL-RL runner 构建，并跑完 1 个 PPO iteration。

## 5. 六组 1024 Iterations 训练

训练队列脚本会依次运行六组任务。每组任务中，`LR_HRL` 和 flat baseline 同时运行；默认 `LR_HRL` 使用 GPU 0，flat baseline 使用 GPU 1。

```bash
cd /root/gpufree-data/quadruped_arm_LR_HRL
STAMP=LR_HRL_repro_v1 nohup scripts/LR_HRL/run_LR_HRL_1024_queue.sh \
  > logs/LR_HRL_queue_LR_HRL_repro_v1.master.log 2>&1 &
```

查看队列进度：

```bash
tail -f logs/LR_HRL_queue_LR_HRL_repro_v1.master.log
```

每组任务的训练日志保存到：

```text
logs/LR_HRL_queue_<STAMP>/
```

RSL-RL checkpoint 保存到：

```text
logs/rsl_rl/LR_HRL/
logs/rsl_rl/LR_Baseline/
```

## 6. 绘制训练曲线

六组训练结束后，运行日志解析和绘图脚本：

```bash
cd /root/gpufree-data/quadruped_arm_LR_HRL
/root/gpufree-data/envs/isaaclab/bin/python \
  scripts/LR_HRL/plot_LR_HRL_training_results.py \
  --queue LR_HRL_queue_LR_HRL_repro_v1
```

输出目录如下：

```text
results/LR_HRL_queue_<STAMP>/
  LR_HRL_all_training_scalars.csv
  LR_HRL_training_summary.md
  curves/*.png
  curves/*.pdf
```

其中 `LR_HRL_all_training_scalars.csv` 保存解析后的训练指标；`curves/` 中保存 PNG 和 PDF 两种格式的对比曲线。

## 7. Baseline 兼容补丁说明

本分支包含一个很小的 baseline 兼容补丁，位于 `maniploco` 配置中。当前使用的 USD 资产没有为所有 legacy trunk/leg contact sensor 名称提供独立的 contact reporter prim。补丁保留 reward 和 termination 中使用的 legacy sensor 字段名，同时将缺失的非足端 contact reporter 映射到有效 body prim。足端 contact sensors 保持不变。

termination helper 也去除了训练过程中的无条件调试打印，避免 1024 iterations 长训练产生过多无关日志。
