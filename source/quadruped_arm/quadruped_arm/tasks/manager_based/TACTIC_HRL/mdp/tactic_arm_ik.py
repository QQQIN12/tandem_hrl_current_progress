"""Numerically bounded arm IK for physical object interaction."""

from dataclasses import MISSING
from typing import Optional

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.mdp.utils import (
    TCP_POS_OFFSET,
    TCP_QUAT_OFFSET,
    _normalize_quat,
    _quat_apply,
    _quat_mul,
    _skew_batch,
    orientation_error,
)


@configclass
class TANDEMArmIkActionCfg(ActionTermCfg):
    class_type: Optional[type] = None
    asset_name: str = "robot"
    command_name: str = "ee_goal"
    ee_body_name: str = MISSING  # type: ignore
    arm_joint_names: list[str] = MISSING  # type: ignore
    damping: float = 0.06
    max_joint_delta: float = 0.07
    joint_limit_margin: float = 0.02
    ee_tcp_offset: tuple[float, float, float] = TCP_POS_OFFSET
    ee_tcp_quat_offset: tuple[
        float, float, float, float
    ] = TCP_QUAT_OFFSET
    orientation_weight: float = 0.16

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TANDEMArmIkAction


class TANDEMArmIkAction(ActionTerm):
    """Track a learned task-space goal with bounded DLS kinematics."""

    cfg: TANDEMArmIkActionCfg

    def __init__(self, cfg: TANDEMArmIkActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._tcp_offset = torch.tensor(
            cfg.ee_tcp_offset, device=self.device
        ).view(1, 3)
        self._tcp_quat_offset = torch.tensor(
            cfg.ee_tcp_quat_offset, device=self.device
        ).view(1, 4)

        arm_ids, _ = self._robot.find_joints(cfg.arm_joint_names)
        self._arm_joint_ids = torch.as_tensor(
            arm_ids, device=env.device, dtype=torch.long
        ).flatten()
        ee_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        self._ee_body_id = int(
            torch.as_tensor(ee_ids).flatten()[0].item()
        )

        jacobians = self._robot.root_physx_view.get_jacobians()
        body_offset = (
            1
            if jacobians.shape[1]
            == len(self._robot.data.body_names) - 1
            else 0
        )
        joint_offset = (
            jacobians.shape[-1]
            - len(self._robot.data.joint_names)
        )
        if joint_offset not in (0, 6):
            raise RuntimeError(
                "Unsupported floating-base Jacobian layout: "
                f"{tuple(jacobians.shape)}"
            )
        self._jacobian_body_id = self._ee_body_id - body_offset
        self._jacobian_joint_ids = (
            self._arm_joint_ids + joint_offset
        )
        self._damping_squared = float(cfg.damping) ** 2
        self._identity = torch.eye(
            6, device=env.device
        ).unsqueeze(0)
        self._raw = torch.empty(
            (env.num_envs, 0), device=env.device
        )
        self._processed = torch.empty(
            (env.num_envs, 0), device=env.device
        )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        return

    def apply_actions(self):
        command = self._env.command_manager.get_term(
            self.cfg.command_name
        )
        link_position = self._robot.data.body_pos_w[
            :, self._ee_body_id
        ]
        link_quaternion = self._robot.data.body_quat_w[
            :, self._ee_body_id
        ]
        tcp_offset = _quat_apply(
            link_quaternion,
            self._tcp_offset.expand(self._env.num_envs, -1),
        )
        tcp_position = link_position + tcp_offset
        tcp_quaternion = _normalize_quat(
            _quat_mul(
                link_quaternion,
                self._tcp_quat_offset.expand(
                    self._env.num_envs, -1
                ),
            )
        )

        position_error = command.curr_goal_pos_w - tcp_position
        orientation_error_vector = (
            float(self.cfg.orientation_weight)
            * orientation_error(
                command.curr_goal_quat_w, tcp_quaternion
            )
        )
        pose_error = torch.cat(
            (position_error, orientation_error_vector), dim=-1
        ).unsqueeze(-1)

        jacobians = self._robot.root_physx_view.get_jacobians()
        link_jacobian = jacobians[
            :,
            self._jacobian_body_id,
            0:6,
            self._jacobian_joint_ids,
        ]
        angular = link_jacobian[:, 3:6, :]
        tcp_jacobian = torch.cat(
            (
                link_jacobian[:, 0:3, :]
                - torch.bmm(_skew_batch(tcp_offset), angular),
                angular,
            ),
            dim=1,
        )
        transpose = tcp_jacobian.transpose(1, 2)
        system = (
            tcp_jacobian @ transpose
            + self._damping_squared * self._identity
        )
        delta = (
            transpose @ torch.linalg.solve(system, pose_error)
        ).squeeze(-1)
        delta.clamp_(
            -float(self.cfg.max_joint_delta),
            float(self.cfg.max_joint_delta),
        )

        current = self._robot.data.joint_pos[
            :, self._arm_joint_ids
        ]
        limits = self._robot.data.soft_joint_pos_limits[
            :, self._arm_joint_ids
        ]
        target = current + delta
        target = torch.maximum(
            target,
            limits[..., 0] + float(self.cfg.joint_limit_margin),
        )
        target = torch.minimum(
            target,
            limits[..., 1] - float(self.cfg.joint_limit_margin),
        )
        self._robot.set_joint_position_target(
            target, joint_ids=self._arm_joint_ids
        )
