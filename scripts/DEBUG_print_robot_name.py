# source/quadruped_arm/scripts/debug_print_robot_names.py
"""
用途：只加载你的机器人 USD，然后打印：
- joint names（关节名）
- body names（刚体名）
- 夹爪相关 joint（用于 mimic / 单自由度控制）
"""

from isaaclab.app import AppLauncher

# 你也可以把 headless=False 看可视化
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from quadruped_arm.robots.robot_cfg import B2W_Cfg  # 你自己的 ArticulationCfg

@configclass
class SceneCfg(InteractiveSceneCfg):
    # 1 个环境就够
    num_envs = 1
    env_spacing = 2.0

    # 地面（如果你这里 import 报错，就先删掉 ground，照样能打印 joint）
    # 
    
    

    # 注意 prim_path 必须用 ENV_REGEX_NS（manager-based 体系习惯）
    robot = B2W_Cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

def _get_attr(obj, names):
    """兼容不同 IsaacLab 版本字段名。"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def main():
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")  # 没 GPU 改成 "cpu"
    )
    scene = InteractiveScene(SceneCfg())
    sim.reset()
    scene.reset()

    robot = scene["robot"]

    # ---- 兼容不同版本：有的在 robot.data.*，有的直接 robot.* ----
    data = getattr(robot, "data", robot)

    joint_names = _get_attr(data, ["joint_names", "dof_names"])
    body_names = _get_attr(data, ["body_names", "link_names"])

    print("\n================ Robot Names ================\n")
    print("[JOINT NAMES]")
    print(joint_names)
    print("\n[BODY NAMES]")
    print(body_names)

    # 夹爪相关：你后面要做 single gripper + mimic，需要先知道两个半爪 joint 叫什么
    if joint_names is not None:
        gripper_like = [n for n in joint_names if ("gripper" in n.lower() or "finger" in n.lower() or "jaw" in n.lower())]
        print("\n[GRIPPER-LIKE JOINTS]")
        print(gripper_like)

    print("\n============================================\n")

    simulation_app.close()

if __name__ == "__main__":
    main()
