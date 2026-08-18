# TANDEM-HRL 当前可复现进度

快照日期：2026-08-18  
当前服务器：L40，Isaac Sim 5.1，Isaac Lab 2.3.2 workstation，ROS 2 Jazzy  
服务器工作目录：`/root/gpufree-data/tandem_hrl_620d798`

## 0. 先看结论

本快照不是“已经得到最终稳定学习策略”的版本，而是完成了底盘稳定性定位、物理缩放、轮速执行器保护、教师策略接口和多组对照测试后的可复现研究快照。

当前最可靠的结果是：冻结机械臂和腿部策略残差后，`ZYB-PhysicalSafeLower-v0` 可以在仿真中保持较小倾角并完成低速直行类测试，不触发倾覆、低高度、坏接触或失败终止。它可以作为后续学习的安全锚点，但它的 yaw 指令跟踪仍然很弱，因此不能称为“已经完成的移动底盘策略”。

当前没有被批准为最终冻结底盘的学习型 checkpoint。尤其不能把下面两个文件误认为已经可用的最终模型：

- `physical_safe_wheel5_multiteacher_smoke/model_4.pt`：只完成 5 次 PPO 迭代，奖励和速度误差恶化。
- `physical_safe_wheel5_wheelonly_mt_30/model_29.pt`：短时看似稳定，但 300 步长测出现最高约 `6.017 rad/s` 的实测轮速，并且 yaw 误差没有消失。

本快照保留了这些失败/否定结果，因为后续调试需要知道哪些方案已经被排除，而不是只保留看起来最好的曲线。

## 1. 项目范围与当前边界

项目包含两条相互关联但不能混为一谈的路径：

1. 原有 TANDEM/TACTIC HRL 任务：包含任务槽、技能选择、物体交互和层次化 actor-critic 的实验框架。原始 `README.md` 中描述的 `TANDEM-HRL-Unified-v0`、`TANDEM-HRL-Single-Object-v0` 等任务仍在代码中。
2. 本次稳定性工作：在 `manager_based/maniploco` 下建立了一个可单独测试、可冻结的下层移动执行器。它先解决 B2-W + Piper 平台的底盘支撑、轮速映射和命令跟踪，再作为上层 HRL 的物理执行层。

第二条路径目前是工作的主线。机械臂动作在下层稳定性测试中被显式冻结；当前测试也没有把真实刚体抓取作为已完成成果。`model_2999.pt` 和 `policy.pt` 只作为旧 ZYB 仓库的参考模型保留，不能直接假设与当前 TANDEM 的 observation/action layout 兼容。

当前仿真使用的机器人配置入口是：

```text
source/quadruped_arm/quadruped_arm/robots/robot_cfg.py
source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3_tandem_inertia.usd
```

这是当前工作树中为 TANDEM 保留的惯量修正版 USD。代码不会自动加载 Isaac Lab 安装目录中的另一份同名资产；复现时必须确认上面的相对路径存在，并确认没有被本地改成其他 USD。

## 2. 代码和框架设计

### 2.1 原有 HRL 层

原有 TANDEM-HRL 的设计意图是把“完成什么任务”和“用什么运动/交互技能完成”分开：

- 任务层对关系条件化的任务槽进行选择，并预测子目标和任务终止风险。
- 技能层组合运动与交互选项，形成可复用的任务条件化技能。
- 任务和技能拥有相对独立的半马尔可夫状态与终止逻辑。
- 上层 actor-critic 同时接收任务/技能状态、机械臂/夹爪动作和迁移后的物理执行器。
- 物体交互必须以刚体接触、抬升、搬运、释放和放置等仿真状态为依据；仅靠距离奖励不能证明抓取成功。

相关目录：

```text
source/quadruped_arm/quadruped_arm/tasks/manager_based/TANDEM_HRL/
source/quadruped_arm/quadruped_arm/tasks/manager_based/TACTIC_HRL/
source/quadruped_arm/quadruped_arm/tasks/manager_based/maniploco/
```

原有 TANDEM checkpoint：

```text
checkpoints/TANDEM_HRL/model_1023.pt
```

它是历史 HRL/交互方向的 checkpoint，不是本次“冻结底盘”候选，也不能单独证明底盘稳定。

### 2.2 当前下层执行器

当前下层的动作链可以概括为：

```text
上层命令 (vx, vy, wz)
        |
        +--> SafeDifferentialWheelVelocityAction
        |       - 物理前馈轮速
        |       - 轮速残差（可冻结/限幅）
        |       - 目标变化率限制
        |       - 可选实测超速反馈制动
        |
        +--> StabilizedLegPositionAction
        |       - 默认支撑姿态
        |       - 腿部策略残差（可冻结/限幅）
        |       - 倾角/高度安全门
        |       - 可选转向时髋部偏置协调
        |
        +--> arm_ik
                - 稳定性阶段显式冻结
```

轮速前馈采用差速轮近似。对左、右侧参考轮速，核心公式是：

```text
v_left  = (vx - 0.5 * track_width * turn_speed_gain * wz) / wheel_radius
v_right = (vx + 0.5 * track_width * turn_speed_gain * wz) / wheel_radius
```

当前标定参数为：

- `wheel_radius = 0.11 m`
- `track_width = 0.4693 m`
- 四个轮子的方向符号当前均为 `(+1,+1,+1,+1)`
- 左右轮模式按 `(left,right,left,right)` 写入四个轮关节
- 普通物理安全任务 `turn_speed_gain = 1.0`
- 5 rad/s 诊断任务 `turn_speed_gain = 12.8`
- 命令盒为 `vx ∈ [-0.25,0.25] m/s`，`wz ∈ [-0.10,0.10] rad/s`

这里的 `turn_speed_gain` 是当前仿真接触/动力学响应的校准系数，不是轮式差速器理论公式本身的一部分。理论公式给出的轮速不等于机器人一定能产生对应的车体 yaw 速度，尤其当腿部承载、轮地接触、轮子惯量和隐式驱动共同变化时。

### 2.3 腿部支撑和安全门

`StabilizedLegPositionAction` 的作用不是直接学习完整步态，而是把下层策略限制在可以诊断的支撑姿态附近：

- `max_policy_residual > 0` 时，腿部目标相对默认姿态限幅。
- `max_policy_residual = 0` 时，现在会显式使用默认姿态，真正冻结腿部残差。
- 安全门根据底盘倾角和高度衰减腿部残差权限，倾斜或下沉时回到默认支撑姿态。
- 机械臂在本阶段设置 `max_joint_delta = 0`，避免上层 arm 动作改变底盘惯性和接触条件。
- 转向协调分支只添加小幅、平滑、命令条件化的髋部偏置；当前没有启用“转弯时自动降低腿刚度”的假设性策略。

后一点很重要：降低腿刚度可能有助于吸收侧向载荷，也可能导致支撑力下降、轮地法向力改变、车体侧倾加剧。当前数据尚不足以支持直接降低刚度，因此本分支把髋部几何协调和刚度调制分开，避免把推测写成结论。

### 2.4 轮子执行器和保护

当前轮子执行器使用有限的仿真 effort/velocity envelope、目标变化率限制和阻尼。主要安全参数如下：

| 参数 | 普通 `PhysicalSafe` | `Wheel5` 诊断 |
| --- | ---: | ---: |
| wheel effort limit | 8 Nm | 8 Nm |
| sim velocity limit | 4 rad/s | 6 rad/s |
| wheel damping | 6 | 6 |
| max wheel reference | 3 rad/s | 5 rad/s |
| max reference acceleration | 4 rad/s² | 4 rad/s² |
| leg support stiffness/damping | 300/10 | 300/10 |

`actual_speed_limit` 是在读到关节实测速度后把有效目标压低的反馈制动，不是 PhysX 的硬限速。隐式关节驱动在有负载时仍可能越过目标，因此测试同时记录了 reference、actual velocity、applied torque 和 brake flag，不能只看 action target。

### 2.5 多教师学习

当前训练脚本支持多教师接口，把旧 ZYB 物理策略、保守支撑策略和中性策略作为候选，并按 blend schedule 形成教师保护层。教师匹配奖励当前测试使用：

- `teacher_ensemble_match.weight = 0.25`
- `sigma = 0.15`
- 训练时可用 `--multi_teacher`
- 训练参数可用 `--teacher_checkpoint`、`--teacher_blend_start`、`--teacher_blend_end`、`--teacher_blend_steps`

这只验证了“训练接口能运行”，没有证明教师集合本身具有正确的 yaw 映射。教师如果共享同一个错误的动作符号或接触假设，多个教师不会自动产生正确监督。

## 3. 已注册的主要任务

注册位置：

```text
source/quadruped_arm/quadruped_arm/tasks/manager_based/maniploco/__init__.py
```

本次工作新增/保留的主要任务如下：

| 任务 | 作用 | 当前结论 |
| --- | --- | --- |
| `ZYB-PhysicalSafeLower-v0` | 低速、冻结腿/臂、物理轮速教师 | 当前最可靠的稳定锚点；yaw 仍不足 |
| `ZYB-PhysicalSafeLearningLower-v0` | 小腿残差学习 | 可训练接口，未收敛 |
| `ZYB-PhysicalSafeTeacherLearningLower-v0` | 腿残差 + 多教师约束 | 只完成 smoke，不升格 |
| `ZYB-PhysicalSafeGain3Lower-v0` | 低于 gain-8 的 yaw 权限消融 | 未解决 yaw 问题 |
| `ZYB-PhysicalSafeGain3ReverseYawLower-v0` | yaw 符号消融 | 只反转符号不能修复问题 |
| `ZYB-PhysicalSafeWheel5Lower-v0` | 5 rad/s 轮速参考诊断 | 稳定性短测可行，非真实硬限速 |
| `ZYB-PhysicalSafeWheel5FeedbackLower-v0` | Wheel5 + 车体速度反馈 | 长时仍有 yaw 漂移和轮速超调 |
| `ZYB-PhysicalSafeWheel5CoordLower-v0` | Wheel5 + 髋部转向协调 | 短测安全，但未证明 yaw 改善 |
| `ZYB-PhysicalSafeWheel5LearningLower-v0` | Wheel5 + 腿/轮小残差学习 | smoke 结果为负 |
| `ZYB-PhysicalSafeWheel5TeacherLearningLower-v0` | Wheel5 + 多教师学生 | 训练接口可用，未收敛 |
| `ZYB-PhysicalSafeWheel5WheelOnlyLearningLower-v0` | 冻结腿，只学习轮残差 | 30 iter 后长时不合格 |
| `ZYB-PhysicalSafeWheel5WheelOnlyTeacherLearningLower-v0` | wheel-only 多教师学生 | 30 iter smoke，未收敛 |

## 4. 测试证据和判定

所有测试的追加式记录在：

```text
legacy_zyb_check/debug_runs_20260818.md
```

每次测试还保存了 summary CSV、per-environment CSV 和可选的 step trace。以下是当前最重要的证据摘要。

### 4.1 物理安全低速锚点

`ZYB-PhysicalSafeLower-v0` 使用兼容的 Mobility `model_25.pt` 作为载入参考，冻结 arm 和腿部策略残差。低速长测中：

- 四个环境均以 time-out 结束，没有 failure、tilt、low-height 或 bad-contact 终止。
- 最大倾角约 `0.030--0.034 rad`。
- 直行类测试的 `wz` 均值接近 0，yaw tracking error 约 `0.098--0.101 rad/s`。
- 因此它证明的是支撑稳定性，不证明转弯跟踪。

对应文件：

```text
legacy_zyb_check/eval_physical_safe_lower_long_per_env.csv
legacy_zyb_check/eval_physical_safe_lower_long_steps.csv
```

参考 checkpoint：

```text
logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt
```

### 4.2 5 rad/s 参考轮速诊断

5 rad/s 是按当前调试阶段要求保留的诊断工况，不应直接解释成真实机器人正常转速。Wheel5 零残差四象限短测中：

- 四个命令角点均无 tilt、low-height、bad-contact 或 failure 终止。
- 最大倾角约 `0.086--0.154 rad`。
- 实测轮速峰值约 `4.23--4.96 rad/s`。
- 这说明参考路径和短时支撑可以运行，但没有证明长时间限速或 yaw 命令正确。

对应文件：

```text
legacy_zyb_check/eval_physical_safe_wheel5_grid.csv
legacy_zyb_check/eval_physical_safe_wheel5_grid_per_env.csv
legacy_zyb_check/eval_physical_safe_wheel5_grid_steps.csv
```

### 4.3 学习训练 smoke

5 次迭代的 Wheel5 多教师 PPO 使用 32 个环境，seed 151，旧 ZYB teacher 和 Mobility `model_25.pt` 做部分初始化：

- mean reward 约从 `-0.55` 降到 `-4.37`。
- teacher match 从 `0.0036` 增到 `0.0396`，但 `error_vel_xy` 从 `0.0069` 增到 `0.0419`。
- `error_vel_yaw` 从 `0.0061` 增到 `0.0265`。
- safety termination 保持 0，但这不足以证明策略学对了。

运行目录：

```text
logs/rsl_rl/maniploco/2026-08-18_11-38-41_physical_safe_wheel5_multiteacher_smoke
```

wheel-only 多教师训练运行 30 iter 后，奖励约从 `-16` 恶化到 `-18.12`。它证明训练进程可以完整启动、保存和评估，不证明已收敛。

运行目录：

```text
logs/rsl_rl/maniploco/2026-08-18_11-53-30_physical_safe_wheel5_wheelonly_mt_30
```

### 4.4 长时否定结果

wheel-only `model_29.pt` 的 100 步四象限短测没有触发安全终止，但 300 步长测最高实测轮速达到约 `6.017 rad/s`，yaw error 约 `0.091--0.102 rad/s`。因此它不能作为冻结底盘。

Wheel5 + 车体速度反馈的 300 步测试最高实测轮速达到约 `6.052` 和 `5.987 rad/s`，同样没有解决 yaw 漂移。测速反馈在当前实现中是改善目标的反馈项，不是硬限速器。

腿部转向协调短测的最大倾角约 `0.083--0.144 rad`，所有安全终止项为 0，yaw error 约 `0.065--0.085 rad/s`。相比无协调基线没有形成足够清晰的改善证据，因此只保留为实验分支。

## 5. 复现方法

### 5.1 环境检查

以下命令假设已经进入服务器工作站，并且 Isaac Lab 安装位置与本次服务器一致：

```bash
cd /root/gpufree-data/tandem_hrl_620d798
export PYTHONPATH="$PWD/source/quadruped_arm:$PYTHONPATH"
export ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh

$ISAAC_PY -c "import isaaclab, isaaclab_tasks, quadruped_arm; print('imports ok')"
$ISAAC_PY -m pip install -e source/quadruped_arm
```

ROS 2 Jazzy 当前没有参与这些 headless 仿真测试；它是服务器的系统环境，不应被误认为本实验已完成 ROS 控制器联调。

### 5.2 运行稳定性四象限评估

当前诊断脚本位于 `legacy_zyb_check/evaluate_checkpoint_stability_test.py`。示例：

```bash
mkdir -p reproduction_results/physical_safe_lower

$ISAAC_PY legacy_zyb_check/evaluate_checkpoint_stability_test.py \
  --task ZYB-PhysicalSafeLower-v0 \
  --checkpoint logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt \
  --num_envs 4 \
  --num_steps 300 \
  --zero_actions \
  --base_command_grid=-0.25,0.25,2,-0.10,0.10,2 \
  --out_csv reproduction_results/physical_safe_lower/summary.csv \
  --per_env_csv reproduction_results/physical_safe_lower/per_env.csv \
  --trace_csv reproduction_results/physical_safe_lower/steps.csv \
  --headless \
  --device cuda:0
```

`--base_command_grid` 的格式是 `vx_min,vx_max,vx_count,wz_min,wz_max,wz_count`。运行前应先确认脚本支持的参数：

```bash
$ISAAC_PY legacy_zyb_check/evaluate_checkpoint_stability_test.py --help
```

### 5.3 运行当前 Wheel5 诊断

```bash
mkdir -p reproduction_results/wheel5

$ISAAC_PY legacy_zyb_check/evaluate_checkpoint_stability_test.py \
  --task ZYB-PhysicalSafeWheel5Lower-v0 \
  --checkpoint logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt \
  --num_envs 4 \
  --num_steps 100 \
  --zero_actions \
  --base_command_grid=-0.25,0.25,2,-0.10,0.10,2 \
  --out_csv reproduction_results/wheel5/summary.csv \
  --per_env_csv reproduction_results/wheel5/per_env.csv \
  --trace_csv reproduction_results/wheel5/steps.csv \
  --headless \
  --device cuda:0
```

需要严格做长时判定时，把 `--num_steps` 改为 300，并同时检查实际轮速字段，而不是只看 `State/wheel_*_vel_target`。

### 5.4 重新启动小规模多教师训练

训练入口是 `scripts/rsl_rl/train.py`。下面是低风险 smoke 模板；它不是最终训练预算：

```bash
$ISAAC_PY scripts/rsl_rl/train.py \
  --task ZYB-PhysicalSafeWheel5WheelOnlyTeacherLearningLower-v0 \
  --num_envs 32 \
  --seed 152 \
  --max_iterations 30 \
  --save_interval 10 \
  --multi_teacher \
  --teacher_checkpoint /path/to/zyb_v0_model_1023.pt \
  --partial_init_checkpoint logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt \
  --teacher_blend_start 0.99 \
  --teacher_blend_end 0.95 \
  --teacher_blend_steps 50000 \
  --action_std_override 0.02 \
  --leg_action_std_override 0.01 \
  --wheel_action_std_override 0.02 \
  --policy_learning_rate_override 1e-5 \
  --headless \
  --device cuda:0
```

真实复现时把 `/path/to/zyb_v0_model_1023.pt` 换成包内或服务器上的 reference checkpoint。训练结束后必须使用独立的 100/300 步四象限评估，不能仅凭训练日志中的 reward 选模型。

## 6. 当前面临的问题

### 6.1 yaw 通道没有被证明正确

底盘直行和底盘转向不是同一个问题。当前轮子符号、轮距和轮半径已经做过校准，但车体 yaw 响应仍可能受以下因素影响：

- 轮关节顺序与左右侧几何位置的映射。
- USD 中轮子旋转轴、碰撞体和摩擦方向。
- 腿部支撑力改变轮子的法向载荷和侧向滑移。
- 隐式速度驱动、阻尼、effort 和 PhysX solver 的耦合。
- 真实资产/修正版资产之间的轮子惯量和接触差异。

仅仅把 `wz_sign` 反过来已经被测试否定，单纯把 turn gain 提高也不能等价于解决 yaw authority。下一步应先做逐轮锁定的动力学/接触辨识，再决定是修正符号、轮距、摩擦、惯量还是控制器结构。

### 6.2 “5 rad/s”不是实测硬限制

当前 Wheel5 任务最多只能保证参考目标被 clamp 到 5 rad/s。实际驱动可能因为负载和目标变化率继续加速，已观测到约 6 rad/s。若未来要把它用于真实机器人，必须把限制下沉到真实电机控制器/驱动层，仿真侧还要建立电机扭矩、速度、减速器和通信周期一致的模型。

### 6.3 学习接口能运行，但学习目标没有收敛

当前学生策略同时面对物理前馈、支撑姿态、安全门、教师匹配和原有 locomotion reward。若教师的 yaw 映射本身错误，teacher matching 会让学生更稳定地复制错误。建议后续按“逐轮辨识 -> 物理前馈闭环 -> 冻结底盘 -> 小残差学习 -> 上层 HRL”的顺序推进。

### 6.4 转向时降低腿刚度尚未得到支持

当前只测试了小幅、平滑的髋部偏置协调，尚未证明降低刚度会提高转向跟踪。降低刚度至少会同时改变支撑力、质心高度、轮地法向力、车体侧倾和轮速响应；在没有接触力/姿态/轮速对照实验前，不应直接把刚度降低作为默认方案。

### 6.5 还没有完成上层 HRL 与真实抓取的闭环验证

本快照的底盘工作还没有形成“上层任务/技能策略输出 -> 冻结下层底盘 -> 真实刚体抓取成功”的完整端到端证据。TANDEM 的交互代码、历史 checkpoint 和 real-grasp 资产可以复现/继续开发，但当前稳定性结论不能外推成真实抓取结论。

### 6.6 还没有 Sim-to-Real 结论

所有当前结论来自 L40 上的 Isaac Sim 仿真。没有真实电机电流、编码器、IMU、轮胎摩擦、控制周期和跌倒保护数据，因此不能直接上机运行。

## 7. 后续推荐顺序

1. 保持 `PhysicalSafeLower` 的腿/臂冻结，单独建立正向和反向的逐轮 wheel response 表。
2. 记录每个轮的关节速度、驱动目标、施加扭矩、接触法向力和底盘 yaw rate；先确定 yaw 符号和幅值的机械原因。
3. 在不改变腿刚度的情况下调通低速 `wz` 跟踪，再逐步增加命令范围。
4. 将真实速度限制建模成电机/驱动约束，而不是只依赖 action clamp 或事后 brake。
5. 选出通过至少 300 步、四象限、多随机种子的底盘策略后冻结其参数。
6. 冻结底盘后再把上层 TANDEM/TACTIC task-skill policy 接回去，单独评估上层是否破坏底盘稳定。
7. 最后才把真实刚体抓取和 ROS 2 控制链加入端到端测试。

## 8. 文件索引

```text
README.md                                      当前说明（本文件）
README_TANDEM_HRL_LEGACY.md                   原服务器 README 的保留副本
CODEX_HANDOVER.md                             后续 Codex 交接文档
source/quadruped_arm/.../maniploco/            当前任务和动作实现
source/quadruped_arm/.../robots/assets/        机器人 USD 资产
checkpoints/TANDEM_HRL/model_1023.pt          历史 TANDEM HRL checkpoint
logs/rsl_rl/maniploco/                        训练运行和 checkpoint
legacy_zyb_check/                             稳定性脚本、reference 模型、CSV、日志
legacy_zyb_check/debug_runs_20260818.md       追加式完整调试记录
reference_server_data/                        主分支中的参考资产；完整历史快照在 Release 资产
SERVER_SNAPSHOT_SCOPE.md                      服务器内容打包范围与归档恢复说明
```

GitHub 当前发布仓库：

```text
https://github.com/QQQIN12/tandem_hrl_current_progress
```

仓库不包含服务器登录密码、GitHub token 或桌面的 `1.txt`。复现者需要自行准备 Isaac Sim/Isaac Lab 安装，并按本文件中的外部环境路径或等价路径修改启动脚本。

服务器数据盘中与本项目相关的参考资产已经在主分支的
`reference_server_data/`；其余历史结果和原始归档分片打包为 GitHub Release
中的 `server_snapshot_20260818.tar`。下载该 Release 资产并解压后，可得到
完整的 `server_snapshot/` 目录。详细边界和归档恢复命令见
`SERVER_SNAPSHOT_SCOPE.md`。Isaac Sim/Isaac Lab 安装本体不在 GitHub 项目内。
