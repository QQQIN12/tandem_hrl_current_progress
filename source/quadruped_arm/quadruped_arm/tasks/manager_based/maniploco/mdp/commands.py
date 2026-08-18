# tasks/manager_based/maniploco/mdp/commands.py
import math
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from typing import Optional

from .utils import (
    TCP_POS_OFFSET, 
    TCP_QUAT_OFFSET, 
    _normalize,
    _quat_apply,
    _quat_mul,
    _quat_from_yaw,
    _quat_from_euler_xyz,
    _quat_from_tool_z,
    _sphere2cart, 
    )



@configclass
class EeSphericalGoalCommandCfg(CommandTermCfg):
    """生成EE期望位姿, sphere(l,p,y) + 轨迹插值(traj) + 保持(hold) + yaw-frame 转 world."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = EeSphericalGoalCommand

    class_type: Optional[type] = None
    asset_name: str = "robot"

    # 必填：给 CommandTerm 基类的 resample 框架用（我们会在 _resample_command 里覆盖 time_left）
    resampling_time_range: tuple[float, float] = (1.0, 5.0)  # 占位，真正的=traj+hold

    traj_time: tuple[float, float] = (1.0, 3.0)
    hold_time: tuple[float, float] = (0.5, 2.0)

    pos_l: tuple[float, float] = (0.4, 0.7)
    pos_p: tuple[float, float] = (-math.pi / 2.5, math.pi / 3.0)
    pos_y: tuple[float, float] = (-1.2, 1.2)

    # sphere_center
    sphere_center_base: tuple[float, float, float] = (-0.215, 0.0, 0.7) # x,y相对于B2W；z相对于terrien

    # TODO:随机扰动范围
    delta_orn_r: tuple[float, float] = (-math.pi / 12, math.pi / 12)
    delta_orn_p: tuple[float, float] = (-math.pi / 12, math.pi / 12)
    delta_orn_y: tuple[float, float] = (-math.pi / 12, math.pi / 12)

    # init range（对齐 VBC init_pos_start/end）
    init_pos_start: tuple[float, float, float] = (0.5, math.pi / 8.0, 0.0)
    init_pos_end: tuple[float, float, float] = (0.4, 0.0, 0.0)

    # TODO:===== collision check ===== 危险区判定
    collision_upper_limits: tuple[float, float, float] = (0.1, 0.2, -0.05)
    collision_lower_limits: tuple[float, float, float] = (-0.8, -0.2, -0.7)
    underground_limit: float = -0.7
    num_collision_check_samples: int = 10
    max_resample_attempts: int = 10

    # ===== debug visualization =====
    debug_vis: bool = False
    debug_ee_body_name: str = "link6"


    # ================== position for Piper ==================
    frame_yaw_offset: float = math.pi

    # ================= tcp offset =================
    debug_tcp_offset: tuple[float, float, float] = TCP_POS_OFFSET
    debug_tcp_quat_offset: tuple[float, float, float, float] = TCP_QUAT_OFFSET

    # ========== rotation local ===========
    goal_roll_about_tool_z: float = -math.pi 


class EeSphericalGoalCommand(CommandTerm):
    """输出 command = curr_sphere(l,p,y)，并额外缓存 curr_goal_pos_w / curr_goal_quat_w 给 action/reward 用"""

    cfg: EeSphericalGoalCommandCfg

    def __init__(self, cfg: EeSphericalGoalCommandCfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]

        self._center_offset = torch.tensor(cfg.sphere_center_base, device=self.device).view(1, 3)

        # command tensor (what CommandManager returns)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)  # curr sphere

        # traj state
        self._sph_start = torch.zeros_like(self._command)
        self._sph_goal = torch.zeros_like(self._command)
        self._traj_t = torch.ones(self.num_envs, device=self.device)
        self._total_t = torch.ones(self.num_envs, device=self.device)

        self._delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)     # 每个环境的目标姿态的随机扰动

        # derived buffers (world-frame)
        self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # 让 reset 第一次用 init_start -> init_end
        self._first_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ===== collision check buffers =====
        self._collision_upper = torch.tensor(self.cfg.collision_upper_limits, device=self.device).view(1, 3)
        self._collision_lower = torch.tensor(self.cfg.collision_lower_limits, device=self.device).view(1, 3)
        self._collision_t = torch.linspace(0.0, 1.0, self.cfg.num_collision_check_samples, device=self.device).view(1, 1, -1)


        # ============== debug tcp offset ==============
        self._debug_tcp_offset = torch.tensor(cfg.debug_tcp_offset, device=self.device).view(1, 3)
        self._debug_tcp_quat_offset = torch.tensor(cfg.debug_tcp_quat_offset, device=self.device).view(1, 4)

        # ===== debug visualization =====
        self._goal_marker = None
        self._center_marker = None
        self._ee_marker = None
        self._debug_ee_body_id = None

        frame_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"

        if self.cfg.debug_vis:
            self._goal_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ee_goal",
                    markers={
                        "goal":sim_utils.UsdFileCfg(
                            usd_path=frame_usd,
                            scale=(0.25, 0.18, 0.18), 
                        
                        # sim_utils.SphereCfg(
                        #     radius=0.045,
                        #     visual_material=sim_utils.PreviewSurfaceCfg(
                        #         diffuse_color=(0.0, 1.0, 0.0)
                        ),
                    },
                )
            )

            self._center_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ee_goal_center",
                    markers={
                        "center": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.2, 0.2)
                            ),
                        ),
                    },
                )
            )

            ee_body_ids, _ = self._robot.find_bodies(self.cfg.debug_ee_body_name)
            self._debug_ee_body_id = int(ee_body_ids[0])

            self._ee_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ee_current",
                    markers={
                        "ee": sim_utils.UsdFileCfg(
                            usd_path=frame_usd,
                            scale=(0.10, 0.10, 0.10), 
                        
                        # sim_utils.SphereCfg(
                        #     radius=0.035,
                        #     visual_material=sim_utils.PreviewSurfaceCfg(
                        #         diffuse_color=(0.1, 0.4, 1.0)   # 蓝色
                        ),
                    },
                )
            )

    @property
    def command(self) -> torch.Tensor:
        """外部读取command接口"""
        return self._command

    # ========= vis function ==========
    def _update_debug_vis(self):
        pos_goal = self.curr_goal_pos_w[:1]
        quat_goal = self.curr_goal_quat_w[:1]

        pos_center = self.center_w[:1]
        quat_identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=self.device,
            dtype=pos_goal.dtype,
        )

        self._goal_marker.visualize(pos_goal, quat_goal)
        self._center_marker.visualize(pos_center, quat_identity)

        if self._ee_marker is not None and self._debug_ee_body_id is not None:
            pos_link6 = self._robot.data.body_pos_w[:1, self._debug_ee_body_id, :]
            quat_link6 = self._robot.data.body_quat_w[:1, self._debug_ee_body_id, :]

            tcp_offset_w = _quat_apply(quat_link6, self._debug_tcp_offset.expand(1, -1))
            pos_tcp = pos_link6 + tcp_offset_w
            quat_tcp = _normalize(
                _quat_mul(quat_link6, self._debug_tcp_quat_offset.expand(1, -1))
            )

            self._ee_marker.visualize(pos_tcp, quat_tcp)

    # -------- required by base class --------
    def _update_metrics(self):
        # TODO: metrics
        return
    
    def _collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        """VBC: 沿 start->goal 轨迹采样，落入 AABB 或低于 underground_limit 则判定为 collision/underground."""
        n = env_ids.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        start = self._sph_start[env_ids]  # (n,3)
        goal = self._sph_goal[env_ids]    # (n,3)

        # (n,3,T)
        sph_path = torch.lerp(start[..., None], goal[..., None], self._collision_t)    # lerp线性插值
        if sph_path.dim() == 2:
            # 兜底：如果出了意外，补一维当作 T=1
            print('=====补一维=====')
            sph_path = sph_path.unsqueeze(-1)
        # reshape -> (T*n,3) -> cart -> (T,n,3)
        sph_flat = sph_path.permute(2, 0, 1).reshape(-1, 3)

        # change logic for Piper offset(Piper倒置安装)
        # cart_flat = _sphere2cart(sph_flat)
        sph_flat_for_check = sph_flat.clone()
        sph_flat_for_check[:, 2] += self.cfg.frame_yaw_offset
        cart_flat = _sphere2cart(sph_flat_for_check)


        cart = cart_flat.view(self.cfg.num_collision_check_samples, n, 3)

        in_bbox = torch.logical_and(
            torch.all(cart < self._collision_upper, dim=-1),
            torch.all(cart > self._collision_lower, dim=-1),
        )  # (T,n)
        collision_mask = torch.any(in_bbox, dim=0)                # (n,)
        underground_mask = torch.any(cart[..., 2] < self.cfg.underground_limit, dim=0)
        # print(f"collision_mask: {collision_mask.sum()}, underground_mask: {underground_mask.sum()}")
        # print("cart min =", cart.amin(dim=(0, 1)))
        # print("cart max =", cart.amax(dim=(0, 1)))
        return collision_mask | underground_mask

    def _resample_command(self, env_ids):
        # env_ids 可能是 list / tensor / slice(None)，统一转 tensor 索引
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        # traj+hold
        traj = torch.empty(n, device=self.device).uniform_(self.cfg.traj_time[0], self.cfg.traj_time[1])
        hold = torch.empty(n, device=self.device).uniform_(self.cfg.hold_time[0], self.cfg.hold_time[1])
        total = traj + hold
        self._traj_t[env_ids] = traj
        self._total_t[env_ids] = total

        # 覆盖基类刚刚设的 time_left（基类会用 resampling_time_range 先写一次）:contentReference[oaicite:2]{index=2}
        self.time_left[env_ids] = total

        # start sphere
        init_mask = self._first_reset[env_ids]
        if torch.any(init_mask):
            self._sph_start[env_ids[init_mask]] = torch.tensor(self.cfg.init_pos_start, device=self.device)
            self._sph_goal[env_ids[init_mask]] = torch.tensor(self.cfg.init_pos_end, device=self.device)
        if torch.any(~init_mask):
            # 先把 start 设为上一次 goal（你原来就这么做）
            self._sph_start[env_ids[~init_mask]] = self._sph_goal[env_ids[~init_mask]].clone()

            # 按 VBC：最多尝试 max_resample_attempts 次，直到 collision_check 通过
            retry_ids = env_ids[~init_mask]
            for _ in range(self.cfg.max_resample_attempts):
                if retry_ids.numel() == 0:
                    break
                m = retry_ids.numel()
                l = torch.empty(m, device=self.device).uniform_(self.cfg.pos_l[0], self.cfg.pos_l[1])
                p = torch.empty(m, device=self.device).uniform_(self.cfg.pos_p[0], self.cfg.pos_p[1])
                y = torch.empty(m, device=self.device).uniform_(self.cfg.pos_y[0], self.cfg.pos_y[1])
                self._sph_goal[retry_ids] = torch.stack([l, p, y], dim=-1)

                bad = self._collision_check(retry_ids)
                bad = torch.zeros(m, dtype=torch.bool, device=self.device)  # TODO：这个collision还要改
                retry_ids = retry_ids[bad]

        # delta orn
        dr = torch.empty(n, device=self.device).uniform_(self.cfg.delta_orn_r[0], self.cfg.delta_orn_r[1])
        dp = torch.empty(n, device=self.device).uniform_(self.cfg.delta_orn_p[0], self.cfg.delta_orn_p[1])
        dy = torch.empty(n, device=self.device).uniform_(self.cfg.delta_orn_y[0], self.cfg.delta_orn_y[1])
        self._delta_rpy[env_ids] = torch.stack([dr, dp, dy], dim=-1)

        self._first_reset[env_ids] = False

    def _update_command(self):
        # elapsed = total - time_left
        elapsed = (self._total_t - self.time_left).clamp(min=0.0)
        t = (elapsed / self._traj_t).clamp(0.0, 1.0)

        # curr sphere
        self._command[:] = torch.lerp(self._sph_start, self._sph_goal, t[:, None])

        # robot pose (world)
        root_pos_w = self._robot.data.root_pos_w  # (N,3)
        yaw = self._robot.data.heading_w          # (N,) 直接给 yaw heading :contentReference[oaicite:3]{index=3}
        yaw_quat = _quat_from_yaw(yaw)

        # spherical center in world: (x,y from base, z=0) + yaw_rotate(center_offset)
        base_xy0 = torch.stack([root_pos_w[:, 0], root_pos_w[:, 1], torch.zeros_like(root_pos_w[:, 2])], dim=-1)
        self.center_w[:] = base_xy0 + _quat_apply(yaw_quat, self._center_offset.repeat(self.num_envs, 1))

        # target position in world
        # cart_yaw = _sphere2cart(self._command)
        # add frame_yaw_offset (Piper倒置安装)
        cmd_for_cart = self._command.clone()
        cmd_for_cart[:, 2] += self.cfg.frame_yaw_offset
        cart_yaw = _sphere2cart(cmd_for_cart)


        # self.curr_ee_goal_cart[:] = cart_yaw

        cart_w = _quat_apply(yaw_quat, cart_yaw)
        self.curr_goal_pos_w[:] = self.center_w + cart_w




        tool_z_w = self.curr_goal_pos_w - self.center_w
        tool_z_w = _normalize(tool_z_w)

        # 1) 让工具局部 +z 指向 center -> tcp
        q_align = _quat_from_tool_z(tool_z_w)

        # 2) 固定绕工具局部 z 轴补一个自转角
        zero = torch.zeros(self.num_envs, device=self.device, dtype=tool_z_w.dtype)
        yaw_offset = torch.full_like(zero, self.cfg.goal_roll_about_tool_z)
        q_spin_local = _quat_from_euler_xyz(zero, zero, yaw_offset)

        # 3) VBC 迁移保留：局部随机扰动
        dr = self._delta_rpy[:, 0]
        dp = self._delta_rpy[:, 1]
        dy = self._delta_rpy[:, 2]
        q_delta_local = _quat_from_euler_xyz(dr, dp, dy)

        # 4) 右乘：都表示在工具局部坐标系里追加旋转
        self.curr_goal_quat_w[:] = _normalize(
            _quat_mul(_quat_mul(q_align, q_spin_local), q_delta_local)
        )


        if self.cfg.debug_vis and self._goal_marker is not None:
            self._update_debug_vis()
           