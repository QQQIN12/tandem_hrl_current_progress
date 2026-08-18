import math
import torch

import math
import torch

from .utils import TCP_POS_OFFSET, TCP_QUAT_OFFSET


FOOT_SENSOR_NAMES = ("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact")


def _sensor_force_w(env, sensor_name: str, use_filtered: bool = True) -> torch.Tensor:
    """返回单个 sensor 的 (N, 3) 接触力。优先返回对地 filtered force。"""
    sensor = env.scene[sensor_name]

    if use_filtered and sensor.data.force_matrix_w is not None:
        f = sensor.data.force_matrix_w  # 可能是 (N, 1, M, 3) 或类似
        return f.reshape(f.shape[0], -1, 3).sum(dim=1)

    f = sensor.data.net_forces_w
    return f.reshape(f.shape[0], -1, 3).sum(dim=1)


def _sensor_force_norm(env, sensor_name: str, use_filtered: bool = True) -> torch.Tensor:
    return torch.norm(_sensor_force_w(env, sensor_name, use_filtered=use_filtered), dim=-1)


def _foot_force_tensor(env, sensor_names=FOOT_SENSOR_NAMES, use_filtered: bool = True) -> torch.Tensor:
    """返回四只脚的接触力 (N, 4, 3)。"""
    forces = []
    for sname in sensor_names:
        forces.append(_sensor_force_w(env, sname, use_filtered=use_filtered))
    return torch.stack(forces, dim=1)


def _foot_contact_bool(env, sensor_names=FOOT_SENSOR_NAMES, thresh: float = 1.5, use_filtered: bool = True) -> torch.Tensor:
    """返回四只脚是否接地 (N, 4)。"""
    contacts = []
    for sname in sensor_names:
        contacts.append(_sensor_force_norm(env, sname, use_filtered=use_filtered) > thresh)
    return torch.stack(contacts, dim=1)


def _get_tcp_pose_w(env, asset_name="robot", ee_body_name="link6"):
    """当前 TCP 位姿，而不是裸 link6。"""
    robot = _robot(env, asset_name)

    if not hasattr(env, "_vbc_ee_body_id"):
        ids, _ = robot.find_bodies(ee_body_name)
        env._vbc_ee_body_id = int(ids[0]) if isinstance(ids, (list, tuple)) else int(ids.flatten()[0].item())

    link_pos_w = robot.data.body_pos_w[:, env._vbc_ee_body_id, :]
    link_quat_w = robot.data.body_quat_w[:, env._vbc_ee_body_id, :]

    tcp_offset = torch.tensor(TCP_POS_OFFSET, device=env.device, dtype=link_pos_w.dtype).view(1, 3).repeat(env.num_envs, 1)
    tcp_quat_offset = torch.tensor(TCP_QUAT_OFFSET, device=env.device, dtype=link_quat_w.dtype).view(1, 4).repeat(env.num_envs, 1)

    tcp_pos_w = link_pos_w + quat_apply(link_quat_w, tcp_offset)
    tcp_quat_w = quat_mul(link_quat_w, tcp_quat_offset)
    tcp_quat_w = tcp_quat_w / torch.clamp(torch.norm(tcp_quat_w, dim=-1, keepdim=True), min=1e-8)

    return tcp_pos_w, tcp_quat_w


def _quat_angle_error(q_des, q_cur):
    """四元数角误差，返回 (N,) 角度。"""
    q_err = quat_mul(q_des, quat_conjugate(q_cur))
    q_err = q_err / torch.clamp(torch.norm(q_err, dim=-1, keepdim=True), min=1e-8)
    w = torch.clamp(torch.abs(q_err[:, 0]), max=1.0)
    return 2.0 * torch.acos(w)


# --------------------------
# math utils (wxyz quat)
# --------------------------
def quat_conjugate(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_apply(q, v):
    # v' = q * (0,v) * q_conj
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:4]

def quat_rotate_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)

def quat_from_yaw(yaw):
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), 0.0*half, 0.0*half, torch.sin(half)], dim=-1)

def quat_to_euler_xyz(q):
    w, x, y, z = q.unbind(-1)
    t0 = 2.0 * (w*x + y*z)
    t1 = 1.0 - 2.0 * (x*x + y*y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w*y - z*x)
    t2 = torch.clamp(t2, -1.0 + 1e-6, 1.0 - 1e-6)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = torch.atan2(t3, t4)
    return torch.stack([roll, pitch, yaw], dim=-1)

def wrap_to_pi(x):
    return (x + math.pi) % (2*math.pi) - math.pi


# --------------------------
# sphere/cart (VBC)
# --------------------------
def sphere2cart(sph):
    l = sph[:, 0]
    p = sph[:, 1]
    y = sph[:, 2]
    x = l * torch.cos(p) * torch.cos(y)
    yy = l * torch.cos(p) * torch.sin(y)
    z = l * torch.sin(p)
    return torch.stack([x, yy, z], dim=-1)

def cart2sphere(cart):
    x, y, z = cart.unbind(-1)
    l = torch.sqrt(x*x + y*y + z*z).clamp(min=1e-6)
    pitch = torch.asin((z / l).clamp(-1.0+1e-6, 1.0-1e-6))
    yaw = torch.atan2(y, x)
    return torch.stack([l, pitch, yaw], dim=-1)



# --------------------------
# resolve helpers (cache)
# --------------------------
def _robot(env, asset_name="robot"):
    return env.scene[asset_name]

def _cmd(env, name="locomotion"):
    if hasattr(env.command_manager, "get_command"):
        return env.command_manager.get_command(name)
    return env.command_manager.get_term(name).command

def _get_joint_ids(env, robot, names, cache_key):
    if not hasattr(env, cache_key):
        jnames = list(robot.data.joint_names)
        ids = []
        for n in names:
            if n not in jnames:
                raise RuntimeError(f"[reward] joint '{n}' not in joint_names")
            ids.append(jnames.index(n))
        setattr(env, cache_key, torch.tensor(ids, device=env.device, dtype=torch.long))
    return getattr(env, cache_key)

def _get_body_ids(env, robot, names, cache_key):
    if not hasattr(env, cache_key):
        bnames = list(robot.data.body_names)
        ids = []
        for n in names:
            if n not in bnames:
                raise RuntimeError(f"[reward] body '{n}' not in body_names")
            ids.append(bnames.index(n))
        setattr(env, cache_key, torch.tensor(ids, device=env.device, dtype=torch.long))
    return getattr(env, cache_key)

def _default_joint_pos(robot):
    # IsaacLab 通常是 (N,num_dofs) 或 (num_dofs,)
    if hasattr(robot.data, "default_joint_pos"):
        dj = robot.data.default_joint_pos
        if dj.dim() == 2:
            return dj[0]
        return dj
    # fallback: capture once
    if not hasattr(robot, "_captured_default_joint_pos"):
        robot._captured_default_joint_pos = robot.data.joint_pos[0].clone().detach()
    return robot._captured_default_joint_pos

def _applied_torque(robot):
    if hasattr(robot.data, "applied_torque"):
        return robot.data.applied_torque
    if hasattr(robot.data, "joint_effort"):
        return robot.data.joint_effort
    return torch.zeros_like(robot.data.joint_pos)


# --------------------------
# buffers init/reset helpers
# --------------------------
def ensure_reward_buffers(env, feet_body_names):
    robot = _robot(env)
    # last actions
    if not hasattr(env, "vbc_last_actions"):
        env.vbc_last_actions = torch.zeros_like(env.action_manager.action)
    # last dof vel/torques
    if not hasattr(env, "vbc_last_dof_vel"):
        env.vbc_last_dof_vel = robot.data.joint_vel.clone()
    if not hasattr(env, "vbc_last_torques"):
        env.vbc_last_torques = _applied_torque(robot).clone()
    # feet forces cache
    if not hasattr(env, "vbc_last_contact_forces"):
        env.vbc_last_contact_forces = torch.zeros(env.num_envs, 4, 6, device=env.device)
    # feet air time
    if not hasattr(env, "feet_air_time"):
        env.feet_air_time = torch.zeros(env.num_envs, 4, device=env.device)

    # cache feet body ids
    _get_body_ids(env, robot, feet_body_names, "_vbc_feet_body_ids")



def update_reward_buffers(env, sensor_names=FOOT_SENSOR_NAMES):
    """每步更新一次缓存。"""
    robot = _robot(env)
    if not hasattr(env, "vbc_last_actions"):
        env.vbc_last_actions = torch.zeros_like(env.action_manager.action)
    if not hasattr(env, "vbc_last_dof_vel"):
        env.vbc_last_dof_vel = robot.data.joint_vel.clone()
    if not hasattr(env, "vbc_last_torques"):
        env.vbc_last_torques = _applied_torque(robot).clone()
    if not hasattr(env, "vbc_last_contact_forces"):
        env.vbc_last_contact_forces = torch.zeros(env.num_envs, 4, 6, device=env.device)
    if not hasattr(env, "feet_air_time"):
        env.feet_air_time = torch.zeros(env.num_envs, 4, device=env.device)

    foot_forces = _foot_force_tensor(env, sensor_names=sensor_names, use_filtered=True)   # (N,4,3)

    fs = torch.zeros(env.num_envs, 4, 6, device=env.device)
    fs[:, :, :3] = foot_forces

    env.vbc_force_sensor_tensor = fs
    env.vbc_foot_contacts_from_sensor = (foot_forces.norm(dim=-1) > 1.5)

    env.vbc_last_actions = env.action_manager.action.clone()
    env.vbc_last_dof_vel = robot.data.joint_vel.clone()
    env.vbc_last_torques = _applied_torque(robot).clone()
    env.vbc_last_contact_forces = fs.clone()

    env.vbc_root_pos_w = robot.data.root_pos_w
    env.vbc_root_quat_w = robot.data.root_quat_w
    env.vbc_root_lin_vel_b = robot.data.root_lin_vel_b
    env.vbc_root_ang_vel_b = robot.data.root_ang_vel_b



# ==========================================================
# Below: one-to-one mapping from VBC `_reward_xxx`
# Each returns (N,) reward value (metric omitted in IsaacLab terms)
# ==========================================================

# ---- walking mask (VBC _get_walking_cmd_mask) ----

def walking_cmd_mask(env, vx_clip=0.2, wz_clip=0.2):
    c = _cmd(env)[:, :3]
    return (c[:, 0].abs() > vx_clip) | (c[:, 1].abs() > vx_clip) | (c[:, 2].abs() > wz_clip)

# ------------------ Piper ------------------
def tracking_ee_world(env, asset_name="robot", ee_body_name="link6", command_name="ee_goal", tracking_ee_sigma=0.2):
    term = env.command_manager.get_term(command_name)
    tcp_pos_w, _ = _get_tcp_pose_w(env, asset_name=asset_name, ee_body_name=ee_body_name)
    err = torch.norm(tcp_pos_w - term.curr_goal_pos_w, dim=1)
    return torch.exp(-err / tracking_ee_sigma)


def tracking_ee_sphere(env, asset_name="robot", ee_body_name="link6", command_name="ee_goal",
                       sphere_error_scale=(1.0, 1.0, 1.0), tracking_ee_sigma=1.0):
    robot = _robot(env, asset_name)
    term = env.command_manager.get_term(command_name)
    tcp_pos_w, _ = _get_tcp_pose_w(env, asset_name=asset_name, ee_body_name=ee_body_name)

    yaw = robot.data.heading_w
    yaw_q = quat_from_yaw(yaw)

    center = term.center_w
    tcp_local = quat_rotate_inverse(yaw_q, tcp_pos_w - center)   # yaw-frame

    sph = cart2sphere(tcp_local)

    # 和 commands.py 保持一致：command 本身不带 frame_yaw_offset，所以转回比较时要减掉它
    cmd_sph = term.command.clone()
    frame_yaw_offset = float(term.cfg.frame_yaw_offset) if hasattr(term.cfg, "frame_yaw_offset") else 0.0
    sph[:, 2] = wrap_to_pi(sph[:, 2] - frame_yaw_offset)

    scale = torch.tensor(sphere_error_scale, device=env.device, dtype=sph.dtype).view(1, 3)
    diff = sph - cmd_sph
    diff[:, 2] = wrap_to_pi(diff[:, 2])
    err = torch.sum(torch.abs(diff) * scale, dim=1)

    return torch.exp(-err / tracking_ee_sigma)

def tracking_ee_cart(env, asset_name="robot", ee_body_name="link6", command_name="ee_goal", tracking_ee_sigma=0.2):
    robot = _robot(env, asset_name)
    term = env.command_manager.get_term(command_name)
    tcp_pos_w, _ = _get_tcp_pose_w(env, asset_name=asset_name, ee_body_name=ee_body_name)

    yaw = robot.data.heading_w
    yaw_q = quat_from_yaw(yaw)

    tcp_local = quat_rotate_inverse(yaw_q, tcp_pos_w - term.center_w)

    cmd_for_cart = term.command.clone()
    frame_yaw_offset = float(term.cfg.frame_yaw_offset) if hasattr(term.cfg, "frame_yaw_offset") else 0.0
    cmd_for_cart[:, 2] += frame_yaw_offset
    target_local = sphere2cart(cmd_for_cart)

    err = torch.norm(tcp_local - target_local, dim=1)
    return torch.exp(-err / tracking_ee_sigma)


def tracking_ee_orn(env, asset_name="robot", ee_body_name="link6", command_name="ee_goal",
                    tracking_ee_sigma=0.25):
    term = env.command_manager.get_term(command_name)
    _, tcp_quat_w = _get_tcp_pose_w(env, asset_name=asset_name, ee_body_name=ee_body_name)

    ang_err = _quat_angle_error(term.curr_goal_quat_w, tcp_quat_w)
    return torch.exp(-ang_err / tracking_ee_sigma)


def tracking_ee_orn_ry(env, asset_name="robot", ee_body_name="link6", command_name="ee_goal",
                       tracking_ee_sigma=0.25):
    term = env.command_manager.get_term(command_name)
    _, tcp_quat_w = _get_tcp_pose_w(env, asset_name=asset_name, ee_body_name=ee_body_name)

    e_des = quat_to_euler_xyz(term.curr_goal_quat_w)
    e_cur = quat_to_euler_xyz(tcp_quat_w)
    d = wrap_to_pi(e_des - e_cur)

    # 只看 roll + yaw
    err = torch.abs(d[:, 0]) + torch.abs(d[:, 2])
    return torch.exp(-err / tracking_ee_sigma)

def arm_energy_abs_sum(env, asset_name="robot", arm_joint_names=None, num_gripper_joints=0):
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if arm_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)
    arm_ids = _get_joint_ids(env, robot, arm_joint_names, "_vbc_arm_joint_ids")
    # exclude gripper joints if you include them in arm_joint_names (建议 arm_joint_names 只给 joint1..6)
    e = (tau[:, arm_ids] * dq[:, arm_ids]).abs().sum(dim=1)
    return e


# ------------------ B2W locomotion ------------------
def hip_action_l2(env, hip_action_indices=(0, 3, 6, 9)):
    """VBC: sum(actions[:, [0,3,6,9]]^2)"""
    a = env.action_manager.action
    idx = torch.tensor(hip_action_indices, device=env.device, dtype=torch.long)
    return torch.sum(a[:, idx] ** 2, dim=1)


def leg_action_l2(env, leg_action_dim=12):
    """VBC: sum(actions[:, :12]^2)"""
    a = env.action_manager.action
    return torch.sum(a[:, :leg_action_dim] ** 2, dim=1)


def leg_energy_abs_sum(env, asset_name="robot", leg_joint_names=None):
    """VBC: sum(abs(torque * dof_vel)) on leg joints"""
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if leg_joint_names is None:
        # 默认前12个关节是腿
        return torch.sum(torch.abs(tau[:, :12] * dq[:, :12]), dim=1)
    ids = _get_joint_ids(env, robot, list(leg_joint_names), "_vbc_leg_joint_ids_energy_abs_sum")
    return torch.sum(torch.abs(tau[:, ids] * dq[:, ids]), dim=1)


def leg_energy_sum_abs(env, asset_name="robot", leg_joint_names=None):
    """VBC: abs(sum(torque * dof_vel)) on leg joints"""
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if leg_joint_names is None:
        return torch.abs(torch.sum(tau[:, :12] * dq[:, :12], dim=1))
    ids = _get_joint_ids(env, robot, list(leg_joint_names), "_vbc_leg_joint_ids_energy_sum_abs")
    return torch.abs(torch.sum(tau[:, ids] * dq[:, ids], dim=1))


def leg_energy(env, asset_name="robot", leg_joint_names=None):
    """VBC: sum(torque * dof_vel) on leg joints (signed)"""
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if leg_joint_names is None:
        return torch.sum(tau[:, :12] * dq[:, :12], dim=1)
    ids = _get_joint_ids(env, robot, list(leg_joint_names), "_vbc_leg_joint_ids_energy")
    return torch.sum(tau[:, ids] * dq[:, ids], dim=1)


def tracking_lin_vel(env, tracking_sigma=0.2):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    err = ((c[:, :2] - v[:, :2]) ** 2).sum(dim=1)
    return torch.exp(-err / tracking_sigma)

def tracking_lin_vel_x_l1(env):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    zero = c[:, 0].abs() < 1e-5
    err = (c[:, 0] - v[:, 0]).abs()
    rew = torch.zeros_like(err)
    rew_x = -err + c[:, 0].abs()
    rew[~zero] = rew_x[~zero] / (c[~zero, 0].abs() + 0.01)
    rew[zero] = 0.0
    return rew

def tracking_lin_vel_x_exp(env, tracking_sigma=0.2):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    err = (c[:, 0] - v[:, 0]).abs()
    return torch.exp(-err / tracking_sigma)

def tracking_ang_vel_yaw_l1(env):
    c = _cmd(env)[:, :3]
    w = _robot(env).data.root_ang_vel_b
    err = (c[:, 2] - w[:, 2]).abs()
    return -err + c[:, 2].abs()

def tracking_ang_vel_yaw_exp(env, tracking_sigma=0.2):
    c = _cmd(env)[:, :3]
    w = _robot(env).data.root_ang_vel_b
    err = (c[:, 2] - w[:, 2]).abs()
    return torch.exp(-err / tracking_sigma)

def tracking_lin_vel_y_l2(env):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    return (c[:, 1] - v[:, 1]) ** 2

def tracking_lin_vel_z_l2(env):
    v = _robot(env).data.root_lin_vel_b
    return v[:, 2] ** 2


def survive(env):
    return torch.ones(env.num_envs, device=env.device)

def foot_contacts_z(env, sensor_names=FOOT_SENSOR_NAMES):
    foot_forces = _foot_force_tensor(env, sensor_names=sensor_names, use_filtered=True)  # (N,4,3)
    return torch.sum(torch.abs(foot_forces[:, :, 2]), dim=1)

def torques(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    if joint_names is None:
        return (tau ** 2).sum(dim=1)
    ids = _get_joint_ids(env, robot, joint_names, "_vbc_torque_joint_ids")
    return (tau[:, ids] ** 2).sum(dim=1)

def energy_square(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if joint_names is None:
        return ((tau * dq) ** 2).sum(dim=1)
    ids = _get_joint_ids(env, robot, joint_names, "_vbc_energy_sq_joint_ids")
    return ((tau[:, ids] * dq[:, ids]) ** 2).sum(dim=1)

def tracking_lin_vel_y(env, tracking_sigma=0.2):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    err = (c[:, 1] - v[:, 1]) ** 2
    return torch.exp(-err / tracking_sigma)

def lin_vel_z(env):
    v = _robot(env).data.root_lin_vel_b
    return v[:, 2] ** 2

def ang_vel_xy(env):
    w = _robot(env).data.root_ang_vel_b
    return (w[:, :2] ** 2).sum(dim=1)

def tracking_ang_vel(env, tracking_sigma=0.2):
    c = _cmd(env)[:, :3]
    w = _robot(env).data.root_ang_vel_b
    err = (c[:, 2] - w[:, 2]) ** 2
    return torch.exp(-err / tracking_sigma)

def work(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    dq = robot.data.joint_vel
    if joint_names is None:
        return (tau[:, :12] * dq[:, :12]).sum(dim=1).abs()
    ids = _get_joint_ids(env, robot, joint_names, "_vbc_work_joint_ids")
    return (tau[:, ids] * dq[:, ids]).sum(dim=1).abs()

def dof_acc(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    dq = robot.data.joint_vel
    dt = float(env.step_dt)
    if not hasattr(env, "vbc_last_dof_vel"):
        env.vbc_last_dof_vel = dq.clone()
    if joint_names is None:
        acc = (env.vbc_last_dof_vel[:, :12] - dq[:, :12]) / dt
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_dofacc_joint_ids")
        acc = (env.vbc_last_dof_vel[:, ids] - dq[:, ids]) / dt
    return (acc ** 2).sum(dim=1)

def action_rate(env, action_dim=None):
    a = env.action_manager.action
    if not hasattr(env, "vbc_last_actions"):
        env.vbc_last_actions = a.clone()
    if action_dim is None:
        da = a - env.vbc_last_actions
    else:
        da = a[:, :action_dim] - env.vbc_last_actions[:, :action_dim]
    return (da ** 2).sum(dim=1)


from isaaclab.managers import SceneEntityCfg
def dof_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Penalty for joint position limit violations. Supports selecting a subset via asset_cfg.joint_names."""
    robot = env.scene[asset_cfg.name]
    asset_cfg.resolve(env.scene)  # 关键：把 joint_names -> joint_ids

    q_all = robot.data.joint_pos  # (N, J)

    # 取 limits（优先 soft）
    if hasattr(robot.data, "soft_joint_pos_limits") and robot.data.soft_joint_pos_limits is not None:
        limits = robot.data.soft_joint_pos_limits
    elif hasattr(robot.data, "joint_pos_limits") and robot.data.joint_pos_limits is not None:
        limits = robot.data.joint_pos_limits
    else:
        raise RuntimeError("No joint position limits found in robot.data")

    # joint 子集
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        q = q_all
        lim = limits
    else:
        q = q_all[:, joint_ids]
        if limits.dim() == 3:
            lim = limits[:, joint_ids, :]
        else:
            lim = limits[joint_ids, :]

    # 最后一维才是 [lo, hi]
    lo = lim[..., 0]
    hi = lim[..., 1]

    out = (lo - q).clamp(min=0.0) + (q - hi).clamp(min=0.0)
    return out.sum(dim=1)

def delta_torques(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    tau = _applied_torque(robot)
    if not hasattr(env, "vbc_last_torques"):
        env.vbc_last_torques = tau.clone()
    if joint_names is None:
        dtau = tau[:, :12] - env.vbc_last_torques[:, :12]
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_deltatau_joint_ids")
        dtau = tau[:, ids] - env.vbc_last_torques[:, ids]
    return (dtau ** 2).sum(dim=1)

def collision(env, sensor_name="contact_forces", asset_name="robot",
              penalize_tokens=("thigh","calf","trunk"), thresh=0.1):
    robot = _robot(env, asset_name)
    forces = env.scene[sensor_name].data.net_forces_w  # (N,B,3)
    bnames = list(robot.data.body_names)
    ids = []
    for i, n in enumerate(bnames):
        ln = n.lower()
        if any(t in ln for t in penalize_tokens):
            ids.append(i)
    # trunk not existing -> base as proxy
    if ("trunk" in penalize_tokens) and all("trunk" not in n.lower() for n in bnames):
        if "base" in bnames:
            ids.append(bnames.index("base"))
    if len(ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    hit = (forces[:, ids, :].norm(dim=-1) > thresh).float()
    return hit.sum(dim=1)

def stand_still(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    robot = _robot(env, asset_name)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    q = robot.data.joint_pos
    q0 = _default_joint_pos(robot)
    if joint_names is None:
        err = (q[:, :12] - q0[:12]).abs().sum(dim=1)
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_stand_joint_ids")
        err = (q[:, ids] - q0[ids]).abs().sum(dim=1)
    rew = torch.exp(-0.05 * err)
    rew[mask] = 0.0
    return rew

def walking_dof(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    robot = _robot(env, asset_name)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    q = robot.data.joint_pos
    q0 = _default_joint_pos(robot)
    if joint_names is None:
        err = (q[:, :12] - q0[:12]).abs().sum(dim=1)
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_walkdof_joint_ids")
        err = (q[:, ids] - q0[ids]).abs().sum(dim=1)
    rew = torch.exp(-0.05 * err)
    rew[~mask] = 0.0
    return rew

def hip_pos(env, asset_name="robot", hip_joint_names=("FR_hip_joint","FL_hip_joint","RR_hip_joint","RL_hip_joint")):
    robot = _robot(env, asset_name)
    ids = _get_joint_ids(env, robot, list(hip_joint_names), "_vbc_hip_ids")
    q = robot.data.joint_pos[:, ids]
    q0 = _default_joint_pos(robot)[ids]
    return ((q - q0) ** 2).sum(dim=1)


def feet_jerk(env, sensor_names=FOOT_SENSOR_NAMES):
    update_reward_buffers(env, sensor_names=sensor_names)

    cur = env.vbc_force_sensor_tensor[:, :, :3]
    last = env.vbc_last_contact_forces[:, :, :3]
    jerk = torch.norm(cur - last, dim=-1).sum(dim=1)
    return jerk

def alive(env):
    return torch.ones(env.num_envs, device=env.device)


def feet_drag(env, asset_name="robot", feet_body_names=("FL_foot","FR_foot","RL_foot","RR_foot"),
              sensor_names=FOOT_SENSOR_NAMES):
    robot = _robot(env, asset_name)
    feet_ids = _get_body_ids(env, robot, list(feet_body_names), "_vbc_feet_body_ids")

    contacts = _foot_contact_bool(env, sensor_names=sensor_names, thresh=1.5, use_filtered=True)  # (N,4)
    foot_vel = robot.data.body_lin_vel_w[:, feet_ids, :2].norm(dim=-1)  # (N,4)

    return torch.sum(contacts.float() * foot_vel, dim=1)


def feet_contact_forces(env, max_contact_force=40.0, sensor_names=FOOT_SENSOR_NAMES):
    foot_forces = _foot_force_tensor(env, sensor_names=sensor_names, use_filtered=True)
    f = foot_forces.norm(dim=-1)
    excess = (f - max_contact_force).clamp(min=0.0).sum(dim=1)
    return excess

def orientation(env, asset_name="robot"):
    robot = _robot(env, asset_name)
    g = robot.data.projected_gravity_b
    return (g[:, :2] ** 2).sum(dim=1)

def roll(env, asset_name="robot"):
    robot = _robot(env, asset_name)
    rpy = quat_to_euler_xyz(robot.data.root_quat_w)
    return rpy[:, 0].abs()

def base_height(env, asset_name="robot", base_height_target=0.55):
    robot = _robot(env, asset_name)
    z = robot.data.root_pos_w[:, 2]
    return (z - base_height_target).abs()

def orientation_walking(env, asset_name="robot", vx_clip=0.2, wz_clip=0.5):
    r = orientation(env, asset_name)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[~mask] = 0.0
    return r

def orientation_standing(env, asset_name="robot", vx_clip=0.2, wz_clip=0.5):
    r = orientation(env, asset_name)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[mask] = 0.0
    return r

def torques_walking(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    r = torques(env, asset_name, joint_names)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[~mask] = 0.0
    return r

def torques_standing(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    r = torques(env, asset_name, joint_names)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[mask] = 0.0
    return r

def energy_square_walking(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    r = energy_square(env, asset_name, joint_names)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[~mask] = 0.0
    return r

def energy_square_standing(env, asset_name="robot", joint_names=None, vx_clip=0.2, wz_clip=0.5):
    r = energy_square(env, asset_name, joint_names)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[mask] = 0.0
    return r

def base_height_walking(env, asset_name="robot", base_height_target=0.55, vx_clip=0.2, wz_clip=0.5):
    r = base_height(env, asset_name, base_height_target)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[~mask] = 0.0
    return r

def base_height_standing(env, asset_name="robot", base_height_target=0.55, vx_clip=0.2, wz_clip=0.5):
    r = base_height(env, asset_name, base_height_target)
    mask = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    r[mask] = 0.0
    return r

def dof_default_pos(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    q = robot.data.joint_pos
    q0 = _default_joint_pos(robot)
    if joint_names is None:
        err = (q[:, :12] - q0[:12]).abs().sum(dim=1)
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_defpos_joint_ids")
        err = (q[:, ids] - q0[ids]).abs().sum(dim=1)
    return torch.exp(-0.05 * err)

def dof_error(env, asset_name="robot", joint_names=None):
    robot = _robot(env, asset_name)
    q = robot.data.joint_pos
    q0 = _default_joint_pos(robot)
    if joint_names is None:
        err = ((q[:, :12] - q0[:12]) ** 2).sum(dim=1)
    else:
        ids = _get_joint_ids(env, robot, joint_names, "_vbc_doferr_joint_ids")
        err = ((q[:, ids] - q0[ids]) ** 2).sum(dim=1)
    return err

def tracking_lin_vel_max(env, vx_clip=0.2):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    vx_cmd = c[:, 0]
    vx = v[:, 0]
    rew = torch.where(
        vx_cmd > 0,
        torch.minimum(vx, vx_cmd) / (vx_cmd + 1e-5),
        torch.minimum(-vx, -vx_cmd) / (-vx_cmd + 1e-5),
    )
    zero = vx_cmd.abs() < vx_clip
    rew[zero] = torch.exp(-vx.abs())[zero]
    return rew

def penalty_lin_vel_y(env, wz_clip=0.5):
    c = _cmd(env)[:, :3]
    v = _robot(env).data.root_lin_vel_b
    rew = v[:, 1].abs()
    rot = c[:, 2].abs() > wz_clip
    rew[rot] = 0.0
    return rew

# 机体平正
def flat_orientation_l2(env, asset_name="robot"):
    robot = _robot(env, asset_name)
    g = robot.data.projected_gravity_b
    return g[:, 0] ** 2 + g[:, 1] ** 2

# 四脚 air/contact 时间方差惩罚：借 Go2 的 air_time_variance_penalty 结构
def air_time_variance_penalty_from_sensors(
    env,
    sensor_names=("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"),
    max_clip: float = 0.5,
):
    last_air = []
    last_contact = []
    for sname in sensor_names:
        sensor = env.scene[sname]
        if not sensor.cfg.track_air_time:
            raise RuntimeError(f"{sname} 没开 track_air_time")
        # 单 body sensor，通常 shape 是 (N,1)，统一拉平
        air_t = sensor.data.last_air_time.reshape(sensor.data.last_air_time.shape[0], -1).mean(dim=1)
        contact_t = sensor.data.last_contact_time.reshape(sensor.data.last_contact_time.shape[0], -1).mean(dim=1)
        last_air.append(torch.clamp(air_t, max=max_clip))
        last_contact.append(torch.clamp(contact_t, max=max_clip))

    last_air = torch.stack(last_air, dim=1)
    last_contact = torch.stack(last_contact, dim=1)
    return torch.var(last_air, dim=1) + torch.var(last_contact, dim=1)


# ------------------ gait terms (keep, but zero if not enabled) ------------------
def tracking_contacts_shaped_force(env, cfg_flag=False):
    # VBC: if not observe_gait_commands return 0
    if not cfg_flag or not hasattr(env, "desired_contact_states"):
        return torch.zeros(env.num_envs, device=env.device)
    # expects env.contact_forces + feet indices etc -> you can wire later
    return torch.zeros(env.num_envs, device=env.device)

def tracking_contacts_shaped_vel(env, cfg_flag=False):
    if not cfg_flag or not hasattr(env, "desired_contact_states"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.zeros(env.num_envs, device=env.device)


# def feet_height(env, asset_name="robot", feet_body_names=("FL_foot","FR_foot","RL_foot","RR_foot"),
#                 feet_height_target=0.08, allfeet=True, vx_clip=0.05, wz_clip=0.2):
#     robot = _robot(env, asset_name)
#     feet_ids = _get_body_ids(env, robot, list(feet_body_names), "_vbc_feet_body_ids")

#     h = robot.data.body_pos_w[:, feet_ids, 2]  # (N,4)

#     if not allfeet:
#         h = h[:, :2]

#     # 只在走路时启用；高于 target 的 swing 脚有奖励
#     walk = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
#     rew = torch.clamp(h - feet_height_target, min=0.0).sum(dim=1)
#     rew[~walk] = 0.0
#     return rew

def feet_height(env,asset_name="robot",feet_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
    sensor_names=("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"),feet_height_target=0.08,
    feet_height_sigma=0.03,feet_height_max=0.16,allfeet=True,vx_clip=0.05,wz_clip=0.2,):

    robot = _robot(env, asset_name)
    feet_ids = _get_body_ids(env, robot, list(feet_body_names), "_vbc_feet_body_ids")

    # 世界坐标下足端高度
    h = robot.data.body_pos_w[:, feet_ids, 2]  # (N,4)
    if not allfeet:
        h = h[:, :2]

    # 只奖励摆动脚
    contacts = _foot_contact_bool(env, sensor_names=sensor_names, thresh=1.5, use_filtered=True)
    if not allfeet:
        contacts = contacts[:, :2]
    swing = (~contacts).float()

    # 接近 target 最好：高斯型奖励
    err = h - feet_height_target
    rew = torch.exp(-(err ** 2) / (2 * feet_height_sigma ** 2)) * swing

    # 太高额外惩罚
    penalty_high = torch.clamp(h - feet_height_max, min=0.0) * swing

    rew = rew.sum(dim=1) - 2.0 * penalty_high.sum(dim=1)

    walk = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    rew[~walk] = 0.0
    return rew



def feet_air_time(env,sensor_names=("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"),dt=None,
    air_time_target=0.5,contact_thresh=1.5,allfeet=True,vx_clip=0.2,wz_clip=0.5,):

    if dt is None:
        dt = float(env.step_dt)

    num_envs = env.num_envs
    num_feet = len(sensor_names)

    # 初始化 buffer
    if not hasattr(env, "feet_air_time") or env.feet_air_time.shape != (num_envs, num_feet):
        env.feet_air_time = torch.zeros(num_envs, num_feet, device=env.device)
    # 逐个 foot sensor 判断是否与地面接触
    contact_list = []
    for sname in sensor_names:
        sensor = env.scene[sname]

        if sensor.data.force_matrix_w is not None:
            f = sensor.data.force_matrix_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1).max(dim=1).values
        else:
            f = sensor.data.net_forces_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1).max(dim=1).values

        contact_list.append(f_norm > contact_thresh)

    contacts = torch.stack(contact_list, dim=1)   # (N, 4)
    # 先记录落地瞬间
    first_contact = (env.feet_air_time > 0.0) & contacts
    # 离地脚继续累计 air time
    env.feet_air_time += dt
    # 落地时奖励
    rew_per_foot = (env.feet_air_time - air_time_target) * first_contact.float()
    if allfeet:
        rew = rew_per_foot.sum(dim=1)
    else:
        rew = rew_per_foot[:, :2].sum(dim=1)

    # 只有在走路命令下才启用
    walk = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    rew = rew * walk.float()
    # 接触脚 air time 清零
    env.feet_air_time *= (~contacts).float()

    return rew


# ------------------ weight=0 updater term ------------------

def update_last_buffers(env, sensor_names=FOOT_SENSOR_NAMES):
    update_reward_buffers(env, sensor_names=sensor_names)
    return torch.zeros(env.num_envs, device=env.device)


# ------------------ wheels ------------------
def wheel_joint_speed_l2(env, asset_name="robot", wheel_joint_names=None):
    """sum(omega_wheel^2)"""
    robot = _robot(env, asset_name)
    if wheel_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)
    ids = _get_joint_ids(env, robot, list(wheel_joint_names), "_vbc_wheel_joint_ids_speed")
    omega = robot.data.joint_vel[:, ids]
    return torch.sum(omega ** 2, dim=1)


def wheel_idle_speed(env, asset_name="robot", wheel_joint_names=None, vx_clip=0.2, wz_clip=0.5):
    """
    站立/弱命令时惩罚轮子速度（轮子不该动的时候）
    返回 penalty 值（正数），在 cfg 里给负权重
    """
    robot = _robot(env, asset_name)
    if wheel_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    ids = _get_joint_ids(env, robot, list(wheel_joint_names), "_vbc_wheel_joint_ids_idle")
    omega = robot.data.joint_vel[:, ids]
    p = torch.sum(omega ** 2, dim=1)

    walk = walking_cmd_mask(env, vx_clip=vx_clip, wz_clip=wz_clip)
    p[walk] = 0.0
    return p


def wheel_action_l2(env, wheel_action_slice=(12, 16)):
    """sum(a_wheel^2)"""
    a = env.action_manager.action
    s, e = wheel_action_slice
    return torch.sum(a[:, s:e] ** 2, dim=1)


def wheel_action_rate(env, wheel_action_slice=(12, 16)):
    """sum((a_wheel - a_wheel_last)^2)"""
    a = env.action_manager.action
    if not hasattr(env, "vbc_last_actions"):
        env.vbc_last_actions = a.clone()
    s, e = wheel_action_slice
    da = a[:, s:e] - env.vbc_last_actions[:, s:e]
    return torch.sum(da ** 2, dim=1)


def wheel_forward_use(
    env,
    asset_name="robot",
    wheel_joint_names=None,
    wheel_radius=0.08,
    wheel_dir_signs=(1.0, 1.0, 1.0, 1.0),
    tracking_sigma=0.25,
    vx_clip=0.2,
    wz_small=0.35,
    use_yaw_alignment_gate=True,
    yaw_gate_sigma=0.20,
):
    """
    平稳直行时鼓励轮子推进（轮子线速度 ~ cmd_vx）
    注意：wheel_dir_signs 需要按关节正方向校准
    """
    robot = _robot(env, asset_name)
    cmd = _cmd(env)[:, :3]
    vx_cmd = cmd[:, 0]
    wz_cmd = cmd[:, 2]
    base_wz = robot.data.root_ang_vel_b[:, 2]

    if wheel_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    ids = _get_joint_ids(env, robot, list(wheel_joint_names), "_vbc_wheel_joint_ids_fwd")
    omega = robot.data.joint_vel[:, ids]  # (N,4)
    signs = torch.tensor(wheel_dir_signs, device=env.device).view(1, -1)
    omega_eff = omega * signs
    v_wheel = wheel_radius * torch.mean(omega_eff, dim=1)  # 估计轮子提供的前向线速度

    err = (v_wheel - vx_cmd) ** 2
    rew = torch.exp(-err / tracking_sigma)

    # 仅在“直行命令明显 + 转向命令不大”时启用
    mask = (torch.abs(vx_cmd) > vx_clip) & (torch.abs(wz_cmd) < wz_small)
    rew[~mask] = 0.0

    # 软门控：先把 yaw 对齐，再鼓励前进轮速（你说的“先转向再动轮子”）
    if use_yaw_alignment_gate:
        yaw_err = (wz_cmd - base_wz) ** 2
        gate = torch.exp(-yaw_err / yaw_gate_sigma)
        rew = rew * gate

    return rew


def wheel_turn_support(
    env,
    asset_name="robot",
    wheel_joint_names=None,  # order: FL, FR, RL, RR
    wheel_radius=0.08,
    track_width=0.38,
    wheel_dir_signs=(1.0, 1.0, 1.0, 1.0),
    tracking_sigma=0.30,
    wz_clip=0.35,
):
    """
    转向时鼓励左右轮差速支持 yaw
    wz_pred ~ (v_right - v_left) / track_width0.0,
    """
    robot = _robot(env, asset_name)
    cmd = _cmd(env)[:, :3]
    wz_cmd = cmd[:, 2]

    if wheel_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    ids = _get_joint_ids(env, robot, list(wheel_joint_names), "_vbc_wheel_joint_ids_turn")
    omega = robot.data.joint_vel[:, ids]  # expected FL,FR,RL,RR
    signs = torch.tensor(wheel_dir_signs, device=env.device).view(1, -1)
    omega_eff = omega * signs
    v = wheel_radius * omega_eff

    v_left = 0.5 * (v[:, 0] + v[:, 2])   # FL, RL
    v_right = 0.5 * (v[:, 1] + v[:, 3])  # FR, RR
    wz_pred = (v_right - v_left) / (track_width + 1e-6)

    err = (wz_pred - wz_cmd) ** 2
    rew = torch.exp(-err / tracking_sigma)

    mask = torch.abs(wz_cmd) > wz_clip
    rew[~mask] = 0.0
    return rew

# ----------- penalty for termination -----------
def penalty_bad_contact(env, sensor_names: list[str], thresh: float = 20.0, use_filter: bool = True) -> torch.Tensor:
    # 直接复用 termination 判断逻辑
    from .terminations import bad_contacts
    hit = bad_contacts(env, sensor_names=sensor_names, thresh=thresh)
    return hit.float()   # 命中=1.0，没命中=0.0


def penalty_tilt(env, asset_cfg, limit: float = 0.8) -> torch.Tensor:
    from .terminations import base_tilt
    hit = base_tilt(env, asset_cfg=asset_cfg, limit=limit)
    return hit.float()


def penalty_low_height(env, asset_cfg, z_min: float = 0.2) -> torch.Tensor:
    from .terminations import base_height_low
    hit = base_height_low(env, asset_cfg=asset_cfg, z_min=z_min)
    return hit.float()