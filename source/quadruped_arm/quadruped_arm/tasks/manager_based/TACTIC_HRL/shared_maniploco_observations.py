"""ZYB-v0-compatible policy observations used by TANDEM-HRL."""

from dataclasses import MISSING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from .utils import (
    _quat_apply,
    _quat_from_yaw,
    _quat_rotate_inverse,
    _quat_to_euler_xyz,
)


@configclass
class VbcObsCfg:
    asset_name: str = "robot"
    obs_joint_names: list[str] = MISSING  # type: ignore
    contact_body_names: list[str] = MISSING  # type: ignore
    arm_base_offset: tuple[float, float, float] = (-0.3, 0.0, 0.09)
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    history_len: int = 10
    use_priv: bool = True
    priv_dim: int = 18


class VbcPolicyObsTerm(ManagerTermBase):
    """Return proprioception, privileged state, and proprioceptive history."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params

        asset_name = params.get("asset_name", "robot")
        obs_joint_names = params["obs_joint_names"]
        contact_body_names = params["contact_body_names"]

        self.ang_vel_scale = params.get("ang_vel_scale", 1.0)
        self.dof_pos_scale = params.get("dof_pos_scale", 1.0)
        self.dof_vel_scale = params.get("dof_vel_scale", 0.05)
        self.priv_dim = params.get("priv_dim", 18)

        self._env = env
        self._robot = env.scene[asset_name]
        self._device = env.device
        self.history_len = params.get("history_len", 10)
        self.use_priv = params.get("use_priv", True)
        self.physical_action_dim = params.get("physical_action_dim")

        self._joint_cfg = SceneEntityCfg(
            asset_name,
            joint_names=obs_joint_names,
            preserve_order=True,
        )
        self._joint_cfg.resolve(env.scene)
        self._jid = self._joint_cfg.joint_ids

        # Preserve the body ordering used by the original ZYB-v0 checkpoint.
        self._body_cfg = SceneEntityCfg(
            asset_name,
            body_names=contact_body_names,
            preserve_order=True,
        )
        self._body_cfg.resolve(env.scene)
        self._bid = self._body_cfg.body_ids

        arm_base_offset = params.get(
            "arm_base_offset",
            (-0.3, 0.0, 0.09),
        )
        self._arm_base_offset = torch.tensor(
            arm_base_offset,
            device=self._device,
        ).view(1, 3)
        self._tool_z_axis = torch.tensor(
            [[0.0, 0.0, 1.0]],
            device=self._device,
        )
        self._hist = None

    def reset(self, env_ids=None, **kwargs):
        if self._hist is None:
            return
        if env_ids is None:
            self._hist[:] = 0
        else:
            self._hist[env_ids] = 0

    def __call__(
        self,
        env,
        asset_name=None,
        obs_joint_names=None,
        contact_body_names=None,
        history_len=None,
        use_priv=None,
        arm_base_offset=None,
        physical_action_dim=None,
    ):
        root_quat_w = self._robot.data.root_quat_w
        roll, pitch, _ = _quat_to_euler_xyz(root_quat_w)
        body_rp = torch.stack((roll, pitch), dim=-1)
        base_ang_vel = (
            self._robot.data.root_ang_vel_b * self.ang_vel_scale
        )

        joint_pos = self._robot.data.joint_pos[:, self._jid]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._jid]
        joint_vel = self._robot.data.joint_vel[:, self._jid]
        dof_pos_rel = (
            joint_pos - default_joint_pos
        ) * self.dof_pos_scale
        dof_vel = joint_vel * self.dof_vel_scale

        last_action = env.action_manager.action
        if self.physical_action_dim is not None:
            last_action = last_action[:, : int(self.physical_action_dim)]

        contact_sensor = env.scene["contact_forces"]
        contact_forces = contact_sensor.data.net_forces_w[:, self._bid, :]
        foot_contacts = (
            torch.linalg.vector_norm(contact_forces, dim=-1) > 1.5
        ).to(torch.float)
        command = env.command_manager.get_command("locomotion")

        ee_term = env.command_manager.get_term("ee_goal")
        goal_w = ee_term.curr_goal_pos_w
        goal_quat_w = ee_term.curr_goal_quat_w
        root_pos = self._robot.data.root_pos_w
        yaw_quat = _quat_from_yaw(self._robot.data.heading_w)

        arm_base_pos = root_pos + _quat_apply(
            yaw_quat,
            self._arm_base_offset.expand(env.num_envs, -1),
        )
        ee_goal_local = _quat_rotate_inverse(
            yaw_quat,
            goal_w - arm_base_pos,
        )
        tool_z_world = _quat_apply(
            goal_quat_w,
            self._tool_z_axis.expand(env.num_envs, -1),
        )
        ee_goal_tool_z_local = _quat_rotate_inverse(
            yaw_quat,
            tool_z_world,
        )

        proprio = torch.cat(
            (
                body_rp,
                base_ang_vel,
                dof_pos_rel,
                dof_vel,
                last_action,
                foot_contacts,
                command,
                ee_goal_local,
                ee_goal_tool_z_local,
            ),
            dim=-1,
        )

        if self._hist is None:
            self._hist = torch.zeros(
                env.num_envs,
                self.history_len,
                proprio.shape[1],
                device=self._device,
            )

        history = self._hist.reshape(env.num_envs, -1)
        if self.use_priv:
            privileged = getattr(
                env,
                "vbc_priv_buf",
                torch.zeros(
                    env.num_envs,
                    self.priv_dim,
                    device=self._device,
                ),
            )
            observation = torch.cat(
                (proprio, privileged, history),
                dim=-1,
            )
        else:
            observation = torch.cat((proprio, history), dim=-1)

        self._hist = torch.cat(
            (self._hist[:, 1:], proprio[:, None, :]),
            dim=1,
        )
        return observation
