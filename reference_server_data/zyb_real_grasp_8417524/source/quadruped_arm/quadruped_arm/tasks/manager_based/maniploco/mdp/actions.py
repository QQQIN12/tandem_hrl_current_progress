# tasks/manager_based/maniploco/mdp/actions.py
"""
位置误差：e_p = P_des - P_cur
姿态误差：q_err = q_des * conj(q_cur)
旋转向量误差：e_o = 2 * sign(q_err.w) * q_err.xyz

组合误差向量：e = [e_p, e_o] （R^6）
末端执行器的6维雅可比矩阵 J R^{6 \times n}

阻尼最小二乘IK：求解关节速度增量\Delta q,使得J \Delta q = e, 并加入阻尼项防止奇异
$$  \Delta q = J^T (J J^T + lambda I_6)^{-1} e  $$

关节位置更新：q_target = q_current + \Delta q
"""




import torch
from dataclasses import MISSING
from typing import Optional

from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm, ActionTermCfg
from .utils import (
    TCP_POS_OFFSET, 
    TCP_QUAT_OFFSET,
    _quat_apply,
    _normalize_quat,
    _skew_batch,
    orientation_error,
    _quat_mul,
    )


@configclass
class ArmIkFromEeGoalActionCfg(ActionTermCfg):
    class_type: Optional[type] = None
    asset_name: str = "robot"
    command_name: str = "ee_goal"  # command term，提供末端执行器末端位姿

    ee_body_name: str = MISSING  # type: ignore
    arm_joint_names: list[str] = MISSING  # type: ignore

    # DLS lambda = I * (0.05 ** 2)
    damping: float = 0.05

    # TCP ee_body在link6下的偏移
    ee_tcp_offset: tuple[float, float, float] = TCP_POS_OFFSET
    ee_tcp_quat_offset: tuple[float, float, float, float] = TCP_QUAT_OFFSET

    # 姿态权重
    orientation_weight: float = 0.2  

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = ArmIkFromEeGoalAction


class ArmIkFromEeGoalAction(ActionTerm):
    """
    DLS:
        dpos = goal_pos_w - ee_pos_w
        drot = orientation_error(goal_quat_w, ee_quat_w)
        dpose = cat([dpos, drot])

        dq = J^T (J J^T + lambda I)^(-1) dpose
        q_target = q_now + dq

    Notes:
    - This intentionally follows VBC logic, so NO dq clamp and NO joint-limit clamp here.
    - For full VBC equivalence, command term should provide BOTH:
          term.curr_goal_pos_w   : (N, 3)
          term.curr_goal_quat_w  : (N, 4), wxyz
      If goal quaternion is missing, this code falls back to current EE orientation, so orientation control
      becomes effectively zero (position IK still works).
    """
    cfg: ArmIkFromEeGoalActionCfg

    def __init__(self, cfg: ArmIkFromEeGoalActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]

        # ==== TCP offset ====
        self._tcp_offset = torch.tensor(cfg.ee_tcp_offset, device=self.device).view(1, 3)
        self._tcp_quat_offset = torch.tensor(cfg.ee_tcp_quat_offset, device=self.device).view(1, 4)
        # self._use_goal_quat_for_offset = cfg.use_goal_quat_for_offset

        # Resolve arm joint ids
        arm_ids, _ = self._robot.find_joints(cfg.arm_joint_names)
        if isinstance(arm_ids, (list, tuple)):
            arm_ids = torch.tensor(arm_ids, device=env.device, dtype=torch.long)
        else:
            arm_ids = arm_ids.to(device=env.device, dtype=torch.long).flatten()
        self._arm_joint_ids = arm_ids
        # print(f"Arm joint IDs: {self._arm_joint_ids}")
        # Resolve EE body id
        ee_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if isinstance(ee_ids, (list, tuple)):
            self._ee_body_id = int(ee_ids[0])
        else:
            self._ee_body_id = int(ee_ids.flatten()[0].item())
        # print(f"EE body ID: {self._ee_body_id}")

        # DLS: lambda = I * (0.05^2)
        self._lambda = cfg.damping ** 2
        self._I6 = torch.eye(6, device=env.device).unsqueeze(0)

        # Required by ActionTerm interface
        self._raw = torch.empty((env.num_envs, 0), device=env.device)
        self._proc = torch.empty((env.num_envs, 0), device=env.device)

        # Jacobian body index in PhysX may be shifted by 1 for floating-base articulation
        J_all = self._robot.root_physx_view.get_jacobians()  # （num_envs, num_bodies_in_jac, 6, num_joints）
        num_jac_bodies = J_all.shape[1]   # num_bodies_in_jac jacobian包含的刚体数量
        num_bodies = len(self._robot.data.body_names)   # 实际的刚体数量

        if num_jac_bodies == num_bodies - 1:
            self._ee_jacobi_body_id = self._ee_body_id - 1
        else:
            self._ee_jacobi_body_id = self._ee_body_id

        self._warned_missing_goal_quat = False

    @property
    def action_dim(self) -> int:
        """该动作项需要的动作维度--不从外部接受任何动作输入"""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """返回原始动作(满足接口规范)"""
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        """返回处理后的动作(满足接口规范)"""
        return self._proc

    def process_actions(self, actions: torch.Tensor):
        # ActionTerm的方法，用于处理外部动作
        return

    def _get_goal_quat_w(self, term, current_tcp_quat_w: torch.Tensor) -> torch.Tensor:
        """
        get current goal quaternion from command term
        """
        cand_names = [
            "curr_goal_quat_w",
            # "curr_goal_quat_wxyz",
            # "goal_quat_w",
            # "goal_quat_wxyz",
            # "curr_goal_orn_quat_w",
            # "curr_goal_orn_quat_wxyz",
        ]
        for name in cand_names:
            if hasattr(term, name):
                q = getattr(term, name)
                if q is not None:
                    return q

        # Fallback: keep current EE orientation -> drot = 0
        if not self._warned_missing_goal_quat:
            print(
                "[ArmIkFromEeGoalAction] Warning: command term has no goal quaternion. "
                "Fallback to current EE orientation, so orientation IK is disabled. "
                "To fully match VBC, expose term.curr_goal_quat_w (wxyz)."
            )
            self._warned_missing_goal_quat = True
        return current_tcp_quat_w

    def apply_actions(self):
        # 从env的命令管理器获取命令项"ee_goal" 
        term = self._env.command_manager.get_term(self.cfg.command_name)

        # --- current link6 pose ---
        link6_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]      # (N, 3)
        link6_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]    # (N, 4), wxyz

        # --- current TCP pose ----
        ## ==pose==
        tcp_offset_local = self._tcp_offset.expand(self._env.num_envs, -1)               # (N,3)
        tcp_offset_w = _quat_apply(link6_quat_w, tcp_offset_local)                       # (N,3)
        curr_tcp_pos_w = link6_pos_w + tcp_offset_w
        ## ==quat==
        tcp_quat_local = self._tcp_quat_offset.expand(self._env.num_envs, -1)
        curr_tcp_quat_w = _quat_mul(link6_quat_w, tcp_quat_local)
        curr_tcp_quat_w = _normalize_quat(curr_tcp_quat_w)

        # --- desired TCP pose from command term ---
        goal_tcp_pos_w = term.curr_goal_pos_w                                # (N, 3) in command term
        goal_tcp_quat_w = self._get_goal_quat_w(term, curr_tcp_quat_w)             # (N, 4), wxyz

        
        # --- 6D pose error （vector）---
        dpos = goal_tcp_pos_w - curr_tcp_pos_w                             # (N, 3)  --> 改成修正后的位置误差
        drot = self.cfg.orientation_weight * orientation_error(goal_tcp_quat_w, curr_tcp_quat_w)                 # (N, 3)
        # drot = torch.zeros_like(curr_tcp_pos_w)# DEBUG--ik：pose
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)            # (N, 6, 1)

        # --- full 6D geometric Jacobian for EE wrt arm joints ---
        J_all = self._robot.root_physx_view.get_jacobians()             # all jacobians: (N, num_jac_bodies, 6, num_joints)
        # J_eef = J_all[:, self._ee_jacobi_body_id, 0:6, self._arm_joint_ids]   # (N, 6, n_arm)
        # J_eef_T = J_eef.transpose(1, 2)                                         # (N, n_arm, 6)
        J_link6 = J_all[:, self._ee_jacobi_body_id, 0:6, self._arm_joint_ids]           # (N,6,n)

        Jv_link6 = J_link6[:, 0:3, :]                                                    # (N,3,n)
        Jw_link6 = J_link6[:, 3:6, :]                                                    # (N,3,n)

        # ---------------- convert link6 Jacobian -> TCP Jacobian ----------------
        # v_tcp = v_link6 + w x r = v_link6 - [r]_x w
        S = _skew_batch(tcp_offset_w)                                                    # (N,3,3)
        Jv_tcp = Jv_link6 - torch.bmm(S, Jw_link6)                                       # (N,3,n)
        J_tcp = torch.cat([Jv_tcp, Jw_link6], dim=1)                                     # (N,6,n)

        J_tcp_T = J_tcp.transpose(1, 2) 
        # --- damped least squares ---
        A = J_tcp @ J_tcp_T + self._lambda * self._I6                    # (N, 6, 6)
        dq = (J_tcp_T @ torch.linalg.solve(A, dpose)).squeeze(-1)        # (N, n_arm)

        # --- q_target = q_now + dq ---
        q_now = self._robot.data.joint_pos[:, self._arm_joint_ids]       # (N, n_arm)
        q_tgt = q_now + dq

        # IMPORTANT:
        # To match VBC original logic, do NOT clamp dq or q_tgt here.
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_joint_ids)

        # DEBUG
        # if self._env.common_step_counter % 30 == 0:
        #     pos_err = torch.norm(dpos, dim=-1).mean().item()
        #     orn_err = torch.norm(drot, dim=-1).mean().item()
        #     dq_norm = torch.norm(dq, dim=-1).mean().item()

            # print(f"[IK] pos_err={pos_err:.4f}, orn_err={orn_err:.4f}, dq_norm={dq_norm:.4f}")

            # # print("ee_body_id =", self._ee_body_id)
            # # print("ee_body_name =", self._robot.data.body_names[self._ee_body_id])
            # # print("ee_jacobi_body_id =", self._ee_jacobi_body_id)
            # # print("J_all.shape =", J_all.shape)
            # # print("num_bodies =", len(self._robot.data.body_names))

            # print("q_now[0] =", q_now[0].detach().cpu().numpy())
            # print("dq[0]    =", dq[0].detach().cpu().numpy())
            # print("q_tgt[0] =", q_tgt[0].detach().cpu().numpy())
            # print("=" * 28)