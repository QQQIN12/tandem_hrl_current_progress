# TANDEM-HRL Codex 交接文档

交接日期：2026-08-18  
用途：后续继续推进本项目时，先阅读本文件，再阅读项目根目录的 `README.md` 和 `legacy_zyb_check/debug_runs_20260818.md`。

## 1. 当前工作位置

本地工作目录：

```text
D:\CODEX\TEST1
```

服务器项目目录：

```text
/root/gpufree-data/tandem_hrl_620d798
```

服务器诊断和历史参考目录：

```text
/root/gpufree-data/legacy_zyb_check
```

服务器环境是 L40、Isaac Sim 5.1、Isaac Lab 2.3.2 workstation、ROS 2 Jazzy。Isaac Lab Python 入口为：

```text
/workspace/isaaclab/_isaac_sim/python.sh
```

远程连接必须使用本地项目中的 PuTTY 工具：

```text
D:\CODEX\TEST1\.tools\plink.exe
D:\CODEX\TEST1\.tools\pscp.exe
```

不要改用 PowerShell Remoting。远程主机信息和密码不写入此文档；桌面的 `1.txt` 只用于 GitHub 操作，也不要把它复制进仓库。

## 2. 用户目标和当前判断

用户的目标是先得到“不倾覆、能响应上层 vx/wz 指令”的稳定底盘，再冻结底盘，继续做学习型 HRL 和真实刚体抓取。

当前判断：

- “站不稳”问题在冻结 arm/leg residual 的 `PhysicalSafeLower` 低速仿真路径上已经显著缓解，现有长测没有触发 tilt、low-height、bad-contact 或 failure termination。
- “跟不住”问题仍未解决，尤其是 yaw。低速物理锚点的直行测试很稳定，但 `wz` 响应弱；反转 `wz_sign`、提高 turn gain、加车体速度反馈、加小幅髋部协调，都没有形成充分的 yaw 解决证据。
- 5 rad/s 当前是用户要求的诊断工况，不应被解释为真实机器人正常转速。Wheel5 的 action reference 可以被 clamp，但实测轮速仍可能到约 6 rad/s。
- 当前没有最终可冻结的学习型底盘 checkpoint。
- 后续不能因为“没有安全终止”就宣布策略可用；必须同时看实测轮速、底盘速度跟踪和长时行为。

## 3. 最重要的路径和代码

当前机器人配置使用：

```text
source/quadruped_arm/quadruped_arm/robots/robot_cfg.py
source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3_tandem_inertia.usd
```

下层主要代码：

```text
source/quadruped_arm/quadruped_arm/tasks/manager_based/maniploco/
  physical_safe_lower_env_cfg.py
  mobility_lower_env_cfg.py
  stabilized_leg_action.py
  safe_wheel_action.py
  multi_teacher.py
  __init__.py
```

历史/上层代码：

```text
source/quadruped_arm/quadruped_arm/tasks/manager_based/TANDEM_HRL/
source/quadruped_arm/quadruped_arm/tasks/manager_based/TACTIC_HRL/
```

本地 `_remote_aug17_stability` 目录是调试时的副本和证据汇总，不是独立的 Python 包。服务器上的 active source 才是复现依据；修改后要同步编译/启动验证，不能只改本地副本。

## 4. 当前使用的安全下层逻辑

### 4.1 轮速映射

当前 wheel signs 是 `(1,1,1,1)`，wheel radius 是 `0.11`，track width 是 `0.4693`，四轮写入顺序是 `(left,right,left,right)`。核心映射是：

```text
left  = (vx - 0.5 * track_width * turn_speed_gain * wz) / wheel_radius
right = (vx + 0.5 * track_width * turn_speed_gain * wz) / wheel_radius
```

`PhysicalSafeLower` 使用 `turn_speed_gain=1.0`；`PhysicalSafeWheel5` 使用 `12.8`，后者只是为了在 `(vx,wz)=(0.25,0.10)` 角点得到约 5 rad/s 的参考轮速。

### 4.2 腿部冻结和安全门

本轮关键 bug 已修复：`max_policy_residual=0` 时，`stabilized_leg_action.py` 现在会显式把腿部目标设为默认姿态。此前“配置上冻结”但代码跳过 clamp，腿部仍可能被策略输出驱动，这会改变支撑、惯量和轮速，导致错误归因。

安全门根据 tilt 和 root height 衰减腿部 residual；arm IK 在稳定性阶段 `max_joint_delta=0`。当前 turn coordination 只做有限髋部偏置，Wheel5Coord 中：

```text
hip offset gain = 0.08
hip offset limit = 0.08
signal smoothing = 0.20
stiffness modulation = False
```

不要在没有接触力和姿态对照数据时直接打开转弯降刚度。

### 4.3 轮子保护

普通任务主要参数：wheel effort 8 Nm、sim velocity 4 rad/s、damping 6、reference cap 3、reference acceleration 4。Wheel5 放宽 reference cap 到 5、sim velocity 到 6，并记录 actual joint velocity、target、applied torque 和 brake flag。

`actual_speed_limit` 逻辑是测得超速后降低 effective target 的反馈制动，不是硬限速。当前测得它不能完全阻止超调。

## 5. 主要任务和 checkpoint 判定

任务注册在 `maniploco/__init__.py`。重点任务：

```text
ZYB-PhysicalSafeLower-v0
ZYB-PhysicalSafeLearningLower-v0
ZYB-PhysicalSafeTeacherLearningLower-v0
ZYB-PhysicalSafeWheel5Lower-v0
ZYB-PhysicalSafeWheel5FeedbackLower-v0
ZYB-PhysicalSafeWheel5CoordLower-v0
ZYB-PhysicalSafeWheel5LearningLower-v0
ZYB-PhysicalSafeWheel5TeacherLearningLower-v0
ZYB-PhysicalSafeWheel5WheelOnlyLearningLower-v0
ZYB-PhysicalSafeWheel5WheelOnlyTeacherLearningLower-v0
```

稳定性参考 checkpoint：

```text
logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt
```

它是兼容 Mobility 分支的参考/教师载入，不是已经验证为最终底盘策略的模型。

明确不升级的模型：

```text
logs/rsl_rl/maniploco/2026-08-18_11-38-41_physical_safe_wheel5_multiteacher_smoke/model_4.pt
logs/rsl_rl/maniploco/2026-08-18_11-53-30_physical_safe_wheel5_wheelonly_mt_30/model_29.pt
```

第一个只训练 5 iter，reward 由约 `-0.55` 到 `-4.37`；第二个 300 步长测最高实测轮速约 `6.017 rad/s`，yaw error 约 `0.091--0.102 rad/s`。二者都不能用于“冻结底盘”。

用户提供的桌面参考文件：

```text
D:\4399\model_2999.pt
D:\4399\policy.pt
```

它们只完成了文件/兼容性层面的参考检查，不能直接当作当前 TANDEM 下层 checkpoint。

## 6. 已完成测试证据

完整追加式日志：

```text
legacy_zyb_check/debug_runs_20260818.md
```

最后一批关键记录是 147--151：

- 147：30 iter wheel-only 多教师训练，reward 下降，未收敛。
- 148：wheel-only model29 100 步短测，安全项为 0，但未足以晋级。
- 149：wheel-only model29 300 步长测，实测轮速到约 6.017，拒绝作为冻结候选。
- 150：Wheel5 feedback 100/300 步，长时轮速到约 6.052/5.987，yaw 漂移仍在。
- 151：Wheel5Coord 四象限短测无安全终止，但 yaw 改善不充分，只保留实验分支。

主要 CSV：

```text
legacy_zyb_check/eval_physical_safe_lower_long_per_env.csv
legacy_zyb_check/eval_physical_safe_wheel5_grid_per_env.csv
legacy_zyb_check/eval_physical_safe_wheel5_wheelonly_model29_long_per_env.csv
legacy_zyb_check/eval_physical_safe_wheel5_feedback_long_per_env.csv
legacy_zyb_check/eval_physical_safe_wheel5_coord_grid_per_env.csv
```

CSV 中需要同时查看：

```text
State/base_tilt
State/base_vx
State/base_wz
State/wheel_*_vel
State/wheel_*_vel_target
State/wheel_*_applied_torque
Tracking/base_vx_error
Tracking/base_wz_error
Termination/failure_any
Termination/tilt
Termination/low_height
Termination/bad_contact
```

## 7. 推荐的下一次工作

不要先继续大规模 PPO。推荐按以下顺序：

1. 复现 `PhysicalSafeLower` 的 300 步直行/低速测试，确认环境、资产和 freeze bug 没有回归。
2. 保持腿和臂冻结，做单轮正负速度 sweep；输出每轮 joint velocity、target、torque、contact force 和 base yaw rate。
3. 用相同的 wheel sign、相同的 USD、相同的 physics step 做前后轮/左右轮模式对照，确认 yaw 的机械方向和幅值。
4. 只在物理前馈能稳定产生正确 yaw 后，重新启用小 wheel residual；先不用 leg residual。
5. 至少做四象限、300 步、多 seed 评估，且 actual wheel speed 必须纳入验收条件。
6. 通过后复制 checkpoint 为明确的 frozen-lower candidate，再接回上层 HRL。
7. 上层接回后重复底盘测试，确认 task/skill 输出没有破坏下层稳定。
8. 最后再做真实刚体抓取和 ROS 2 联调。

当前最值得先查的不是“惯量应该改成多少”，而是轮子关节/碰撞体/符号/接触载荷和隐式驱动共同造成的 yaw authority。惯量可能是变量，但现有数据还不能把它定为唯一主因。

## 8. 复现命令模板

GitHub 主分支包含 Git LFS 管理的 USD 和 checkpoint。新服务器先执行：

```bash
git lfs install
git clone https://github.com/QQQIN12/tandem_hrl_current_progress.git
cd tandem_hrl_current_progress
git lfs pull
```

```bash
cd /root/gpufree-data/tandem_hrl_620d798
export PYTHONPATH="$PWD/source/quadruped_arm:$PYTHONPATH"
export ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
$ISAAC_PY -m pip install -e source/quadruped_arm

mkdir -p reproduction_results
$ISAAC_PY legacy_zyb_check/evaluate_checkpoint_stability_test.py \
  --task ZYB-PhysicalSafeLower-v0 \
  --checkpoint logs/rsl_rl/maniploco/2026-08-18_09-08-24_wheel_only_signfix_gain8_damp6_teacher_model100/model_25.pt \
  --num_envs 4 --num_steps 300 --zero_actions \
  --base_command_grid=-0.25,0.25,2,-0.10,0.10,2 \
  --out_csv reproduction_results/summary.csv \
  --per_env_csv reproduction_results/per_env.csv \
  --trace_csv reproduction_results/steps.csv \
  --headless --device cuda:0
```

如果脚本参数或任务注册报错，先检查：

```bash
grep -R "ZYB-PhysicalSafe" -n source/quadruped_arm/quadruped_arm/tasks/manager_based/maniploco
ls -lh source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3_tandem_inertia.usd
$ISAAC_PY legacy_zyb_check/evaluate_checkpoint_stability_test.py --help
```

## 9. 调试记录规则

- 每次测试都在 `debug_runs_20260818.md` 追加新编号；不要覆盖旧结果。
- 记录任务、checkpoint、seed、num_envs、num_steps、命令范围、代码变更、所有 safety term 和关键 tracking/actual-speed 字段。
- 不要只记录 reward；reward 好坏不能替代底盘稳定性和速度跟踪判定。
- 不需要为普通调试生成 SHA256 等哈希；只记录路径、配置、时间和可复现命令即可。
- 修改 active source 后重新运行 import/compile smoke，并通过 pscp 同步诊断脚本和日志。
- 不要在 README、交接文档、Git 提交或日志中写服务器密码、GitHub token 或 `1.txt` 内容。

## 10. GitHub 发布状态

目标仓库：

```text
https://github.com/QQQIN12/tandem_hrl_current_progress
```

README 和本交接文档应同时放在服务器项目根目录和本地桌面。上传内容应包括当前项目源代码、当前训练/评估证据、legacy reference 与相关 USD；不包括外部 Isaac Lab 安装、系统缓存、登录凭据和桌面凭据文件。

服务器数据盘的其余相关历史结果已经打包为 GitHub Release 资产
`server_snapshot_20260818.tar`；其中包含 `reference_server_data/server_snapshot/`
和原始 `tandem_620d798_gitarchive.tar` 的分片。恢复方法见
`SERVER_SNAPSHOT_SCOPE.md`。服务器本地工作树曾保留完整分片提交，但该大提交
未进入 main，避免后续 clone 被超大普通 Git pack 阻塞。

后续 Codex 接手时，第一步应检查远程仓库的 `git status`、当前分支和 README 是否已推送，再检查本交接文档中的模型路径是否存在。不要从一个“看起来最新”的 `model_*.pt` 直接开始训练，先按第 7 节重新验证。
