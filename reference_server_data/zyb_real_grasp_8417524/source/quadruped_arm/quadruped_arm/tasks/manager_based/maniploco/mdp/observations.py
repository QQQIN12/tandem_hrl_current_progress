# tasks/manager_based/maniploco/mdp/observations.py
import torch
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase

from .utils import (
    _quat_apply,
    _quat_from_yaw,
    _quat_conj,
    _quat_rotate_inverse,
    _quat_to_euler_xyz,
)

@configclass
class VbcObsCfg:
    asset_name: str = "robot"

    # 你要纳入 obs 的 joints（建议：12腿 + 4轮 + 6臂；不含夹爪 7/8）
    obs_joint_names: list[str] = MISSING  # type: ignore

    # 足/轮 的 body 名称（用于 contact sensor 取四个接触）
    contact_body_names: list[str] = MISSING  # type: ignore

    # TODO:arm base offset
    arm_base_offset: tuple[float, float, float] = (-0.3, 0.0, 0.09)

    # scales（对齐 VBC normalization.obs_scales）
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05

    # history
    history_len: int = 10

    # priv（先按 VBC: mass(5)+friction(1)+motor(12)）特权信息维度
    use_priv: bool = True
    priv_dim: int = 18


class VbcPolicyObsTerm(ManagerTermBase):
    """一个 term 直接输出：proprio + priv + history_flat（按 VBC 拼接顺序）。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # ----- get params from cfg -----
        p = cfg.params

        asset_name = p.get("asset_name", "robot")
        obs_joint_names = p["obs_joint_names"]
        contact_body_names = p["contact_body_names"]
        history_len = p.get("history_len", 10)
        use_priv = p.get("use_priv", True)
        arm_base_offset = p.get("arm_base_offset", (-0.3, 0.0, 0.09))


        self.ang_vel_scale = p.get("ang_vel_scale", 1.0)    # 默认值 
        self.dof_pos_scale = p.get("dof_pos_scale", 1.0)    # TODO: 显示加在env_cfg中的obscfg的params,可以方便后面调参
        self.dof_vel_scale = p.get("dof_vel_scale", 0.05)

        self.priv_dim = p.get("priv_dim", 18)   # priviledge dimension--特权观测维度，VBC中mass(5)+friction(1)+motor(12)=18 TODO:是否调整

        self._env = env
        self._robot = env.scene[asset_name]
        self._device = env.device

        self.history_len = history_len
        self.use_priv = use_priv

        # resolve joints
        self._joint_cfg = SceneEntityCfg(asset_name, joint_names=obs_joint_names, preserve_order=True)
        self._joint_cfg.resolve(env.scene)
        self._jid = self._joint_cfg.joint_ids

        # resolve bodies for contact sensor
        self._body_cfg = SceneEntityCfg(asset_name, body_names=contact_body_names, preserve_order=True)
        self._body_cfg.resolve(env.scene)
        self._bid = self._body_cfg.body_ids

        self._arm_base_offset = torch.tensor(arm_base_offset, device=self._device).view(1, 3)

        # history buffer (proprio only)
        self._hist = None  # (N, H, D)

        # 工具系的z轴基础方向
        self._tool_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self._device)

    def reset(self, env_ids=None, **kwargs):
        if self._hist is None:
            return
        if env_ids is None:
            self._hist[:] = 0
        else:
            self._hist[env_ids] = 0

    def __call__(self,env,asset_name=None,obs_joint_names=None,contact_body_names=None,
                 history_len=None,use_priv=None,arm_base_offset=None):
        # --- base orientation roll/pitch（用 projected gravity 也行，这里保持直观：从 heading/姿态拿） ---
        # 这里不手写 euler，直接用 root_ang_vel_b + projected_gravity 也可；先按最简可跑：roll/pitch 先置 0
        # 你如果必须严格 roll/pitch：后面把 euler_xyz_from_quat 加进来即可
        
        # 角速度、姿态（roll/pitch）
        root_quat_w = self._robot.data.root_quat_w                     # (N,4), wxyz
        roll, pitch, _ = _quat_to_euler_xyz(root_quat_w)
        body_rp = torch.stack([roll, pitch], dim=-1)                  # (N,2)

        base_ang_vel = self._robot.data.root_ang_vel_b * self.ang_vel_scale


        q = self._robot.data.joint_pos[:, self._jid]    # 当前关节位置
        q0 = self._robot.data.default_joint_pos[:, self._jid]  # 默认关节位置
        qd = self._robot.data.joint_vel[:, self._jid]  # 当前关节速度
        dof_pos_rel = (q - q0) * self.dof_pos_scale  # 当前关节位置与默认位置的差值，然后缩放
        dof_vel = qd * self.dof_vel_scale   # 当前的速度值 -> 缩放

        # last actions（VBC 取 action_history_buf[:,-1]）
        # 这里用 action_manager 的 last_action（包括 leg_pos + wheel_vel），你想只取 leg_pos 就切片
        last_act = env.action_manager.action  # (N, total_action_dim)  取得最新actions

        # foot/wheel contacts（用 ContactSensor）
        sensor = env.scene["contact_forces"]  # 接触力传感器 在env_cfg  sensor中配置
        forces = sensor.data.net_forces_w[:, self._bid, :]     # (N,4,3) 获得指定身体部位的全局坐标系力
        foot_contacts = (torch.norm(forces, dim=-1) > 1.5).to(torch.float)  # (N,4) 接触力大于 1.5 判定为接触（二值化）

        # locomotion command
        cmd = env.command_manager.get_command("locomotion")    # (N,3) 获取底盘运动指令

        # ee goal local cart（对齐 VBC：arm_base_pos = base_pos + yaw_rotate(offset)）
        ee_term = env.command_manager.get_term("ee_goal") # 获得ee_goal的term, 这里是取得ee_goal的位置，而action中是用tcp的位置去追这个位置
        goal_w = ee_term.curr_goal_pos_w  # (N,3) 全局坐标系的ee_goal
        goal_quat_w = ee_term.curr_goal_quat_w  # (N,4) 全局坐标系的ee_goal的四元数


        root_pos = self._robot.data.root_pos_w # 基的pos
        yaw = self._robot.data.heading_w # 偏航=朝向
        yaw_q = _quat_from_yaw(yaw) # 把朝向转成四元数

        # ---
        arm_base_pos = root_pos + _quat_apply(yaw_q, self._arm_base_offset.expand(env.num_envs, -1))  # 机械臂base的全局pose
        goal_delta_w = goal_w - arm_base_pos  # 机械臂base的全局坐标系下与ee_goal的位置差
        ee_goal_local = _quat_rotate_inverse(yaw_q, goal_delta_w)  # 局部的ee_goal位置

        tool_z_world = _quat_apply(
            goal_quat_w,
            self._tool_z_axis.expand(env.num_envs, -1),
        )
        ee_goal_tool_z_local = _quat_rotate_inverse(yaw_q, tool_z_world) 



        # 本体感知拼接
        proprio = torch.cat(
            [body_rp, base_ang_vel, dof_pos_rel, dof_vel, last_act, foot_contacts, cmd, ee_goal_local, ee_goal_tool_z_local],
            dim=-1
        )

        # 历史缓冲初始化: 避免在未使用时分配内存
        if self._hist is None:
            D = proprio.shape[1]
            self._hist = torch.zeros(env.num_envs, self.history_len, D, device=self._device)

        # 使用特权信息
        if self.use_priv:
            # 这个vbc_priv_buf是在event中使用func: set_vbc_priv_buf设置的这个property
            priv = getattr(env, "vbc_priv_buf", torch.zeros(env.num_envs, self.priv_dim, device=self._device))
            obs = torch.cat([proprio, priv, self._hist.reshape(env.num_envs, -1)], dim=-1)
        else:
            obs = torch.cat([proprio, self._hist.reshape(env.num_envs, -1)], dim=-1)

        # 更新 history（VBC 是 append 当前帧）
        self._hist = torch.cat([self._hist[:, 1:], proprio[:, None, :]], dim=1)  # (N, H, D): 环境数，历史长度，观测维度

        return obs