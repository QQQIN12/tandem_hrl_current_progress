# script/DEBUG_ik_only.py

import argparse

from isaaclab.app import AppLauncher

# Parser
parser = argparse.ArgumentParser(description="Debug IK only for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. ZYB-v0.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# start app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# !!先启动app再import相关的
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import quadruped_arm.tasks  # noqa: F401

from quadruped_arm.tasks.manager_based.maniploco.mdp.utils import TCP_POS_OFFSET, TCP_QUAT_OFFSET


def _quat_apply(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """apply quat to vector. q: (...,4) wxyz, v: (...,3)"""
    w, x, y, z = q_wxyz.unbind(-1)
    vx, vy, vz = v.unbind(-1)

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([vpx, vpy, vpz], dim=-1)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)


def orientation_error(q_des_wxyz: torch.Tensor, q_cur_wxyz: torch.Tensor) -> torch.Tensor:
    """
    q_err = q_des * conj(q_cur)
    drot = 2 * sign(q_err.w) * q_err.xyz
    """
    q_des_wxyz = _normalize_quat(q_des_wxyz)
    q_cur_wxyz = _normalize_quat(q_cur_wxyz)

    w1, x1, y1, z1 = q_des_wxyz.unbind(-1)
    w2, x2, y2, z2 = q_cur_wxyz.unbind(-1)

    # q_des * conj(q_cur)
    w =  w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    x = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    y = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    z = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2

    sign = torch.where(w.unsqueeze(-1) < 0.0, -1.0, 1.0)
    return 2.0 * sign * torch.stack([x, y, z], dim=-1)

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)

def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def print_foot_contacts(base_env, sensor_names):
    for sname in sensor_names:
        sensor = base_env.scene[sname]

        msg = [f"[{sname}]"]

        if sensor.data.net_forces_w is not None:
            nf = sensor.data.net_forces_w[0]
            nf_norm = torch.norm(nf.reshape(-1, 3), dim=-1)
            msg.append(f"net_norm={nf_norm.detach().cpu().tolist()}")

        if sensor.data.force_matrix_w is not None:
            fm = sensor.data.force_matrix_w[0]
            fm_norm = torch.norm(fm.reshape(-1, 3), dim=-1)
            msg.append(f"filtered_norm={fm_norm.detach().cpu().tolist()}")
        else:
            msg.append("filtered_norm=None")

        print(" | ".join(msg))

def main():
    # env_cfg
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # froze base
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 4.0

    # 固定 base
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # 不允许自碰撞
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    # 禁止push
    if hasattr(env_cfg.events, "push"):
        env_cfg.events.push = None

    # 关闭termination 防止debug频繁reset
    # if hasattr(env_cfg.terminations, "bad_contact"):
    #     env_cfg.terminations.bad_contact = None
    if hasattr(env_cfg.terminations, "tilt"):
        env_cfg.terminations.tilt = None
    if hasattr(env_cfg.terminations, "low_height"):
        env_cfg.terminations.low_height = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # 加长 episode
    env_cfg.episode_length_s = 1.0e6

    # 冻结 locomotion command
    env_cfg.commands.locomotion.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.locomotion.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.locomotion.ranges.ang_vel_z = (0.0, 0.0)

    # 打开 ee goal 可视化
    env_cfg.commands.ee_goal.debug_vis = True

    # 为了方便观察，让目标移动慢一点
    env_cfg.commands.ee_goal.traj_time = (3.0, 3.0)
    env_cfg.commands.ee_goal.hold_time = (2.0, 2.0)

    # env设立
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs, _ = env.reset()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    # 12+4 dim actions; ik only
    actions = torch.zeros(env.action_space.shape, device=base_env.device)

    # 末端 body 名和你现在 arm_ik 配置一致
    ee_ids, _ = robot.find_bodies("link6")
    ee_body_id = int(ee_ids[0])

    # 与 action.py 保持一致
    tcp_offset = torch.tensor(TCP_POS_OFFSET, device=base_env.device).view(1, 3)
    tcp_quat_offset = torch.tensor(TCP_QUAT_OFFSET, device=base_env.device).view(1, 4)

    step_count = 0

    # ----------------------- main -----------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # TODO：DEBUG ======= 每 20 步打印一次脚底接触信息 =======
            if step_count % 20 == 0:
                print_foot_contacts(
                    base_env,
                    ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
                )

            # 每 30 步打印一次当前 goal / ee / err
            if step_count % 100 == 0:
                ee_term = base_env.command_manager.get_term("ee_goal")

                # ===== command goal (this is TCP goal) =====
                goal_tcp_pos_w = ee_term.curr_goal_pos_w[0:1]      # (1,3)
                goal_tcp_quat_w = ee_term.curr_goal_quat_w[0:1]    # (1,4)

                # ===== current link6 pose =====
                link6_pos_w = robot.data.body_pos_w[0:1, ee_body_id, :]      # (1,3)
                link6_quat_w = robot.data.body_quat_w[0:1, ee_body_id, :]     # (1,4)

                # ===== current TCP pose =====
                tcp_offset_w_cur = _quat_apply(link6_quat_w, tcp_offset)      # (1,3)
                curr_tcp_pos_w = link6_pos_w + tcp_offset_w_cur
                # curr_tcp_quat_w = link6_quat_w
                curr_tcp_quat_w = _quat_mul(link6_quat_w, tcp_quat_offset)
                curr_tcp_quat_w = _normalize_quat(curr_tcp_quat_w)

                # ===== "true" TCP-space errors =====
                tcp_pos_err = torch.norm(goal_tcp_pos_w - curr_tcp_pos_w, dim=-1).item()
                tcp_orn_vec = orientation_error(goal_tcp_quat_w, curr_tcp_quat_w)
                tcp_orn_err = torch.norm(tcp_orn_vec, dim=-1).item()

                # ===== extra diagnostics: inferred link6 goal =====
                # 用目标姿态反推 link6 goal（更合理）
                tcp_offset_w_goal = _quat_apply(goal_tcp_quat_w, tcp_offset)
                goal_link6_from_goal_quat = goal_tcp_pos_w - tcp_offset_w_goal
                link6_err_from_goal_quat = torch.norm(goal_link6_from_goal_quat - link6_pos_w, dim=-1).item()

                # 用当前姿态反推 link6 goal（对应你旧 action 默认 use_goal_quat_for_offset=False）
                goal_link6_from_curr_quat = goal_tcp_pos_w - tcp_offset_w_cur
                link6_err_from_curr_quat = torch.norm(goal_link6_from_curr_quat - link6_pos_w, dim=-1).item()

                g = goal_tcp_pos_w[0]
                c = curr_tcp_pos_w[0]
                l = link6_pos_w[0]



                print(
                    f"[step {step_count:06d}] "
                    f"goal_tcp=({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}) | "
                    f"curr_tcp=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) | "
                    f"link6=({l[0]:+.3f}, {l[1]:+.3f}, {l[2]:+.3f})"
                )
                print(
                    f"               "
                    f"tcp_pos_err={tcp_pos_err:.4f} | "
                    f"tcp_orn_err={tcp_orn_err:.4f} | "
                    f"link6_err(goal_quat)={link6_err_from_goal_quat:.4f} | "
                    f"link6_err(curr_quat)={link6_err_from_curr_quat:.4f}"
                )

                q_offset_est = _quat_mul(_quat_conj(link6_quat_w), goal_tcp_quat_w)
                q_offset_est = _normalize_quat(q_offset_est)
                qo = q_offset_est[0]

                print(
                    f"               "
                    f"q_offset_est=({qo[0]:+.4f}, {qo[1]:+.4f}, {qo[2]:+.4f}, {qo[3]:+.4f})"
                )
                

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()