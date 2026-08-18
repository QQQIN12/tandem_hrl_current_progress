"""Semi-Markov execution state for learned task and skill proposals."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..tactic_layout import ACTION_LAYOUT


@configclass
class TACTICSymmetricGripperActionCfg(ActionTermCfg):
    class_type: Optional[type] = None
    asset_name: str = "robot"
    joint_names: tuple[str, str] = ("joint7", "joint8")
    open_positions: tuple[float, float] = (0.033, -0.033)
    closed_positions: tuple[float, float] = (0.003, -0.003)
    action_dim: int = 1

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TACTICSymmetricGripperAction


class TACTICSymmetricGripperAction(ActionTerm):
    """Map one learned closure logit to opposed finger targets."""

    cfg: TACTICSymmetricGripperActionCfg

    def __init__(self, cfg: TACTICSymmetricGripperActionCfg, env):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._asset.find_joints(
            list(cfg.joint_names), preserve_order=True
        )
        if len(joint_ids) != 2:
            raise RuntimeError(
                f"Expected two gripper joints, resolved {joint_names}"
            )
        self._joint_ids = joint_ids
        self._raw_actions = torch.zeros(
            env.num_envs, 1, device=env.device
        )
        self._processed_actions = torch.zeros(
            env.num_envs, 2, device=env.device
        )
        self._open = torch.tensor(
            cfg.open_positions, device=env.device
        ).view(1, 2)
        self._closed = torch.tensor(
            cfg.closed_positions, device=env.device
        ).view(1, 2)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        closure = torch.sigmoid(actions)
        self._processed_actions[:] = torch.lerp(
            self._open.expand_as(self._processed_actions),
            self._closed.expand_as(self._processed_actions),
            closure,
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = -3.0
        self._processed_actions[env_ids] = self._open


@configclass
class TACTICHierarchyActionCfg(ActionTermCfg):
    class_type: Optional[type] = None
    asset_name: str = "robot"
    action_dim: int = ACTION_LAYOUT.hierarchy_dim
    task_min_dwell_s: float = 0.20
    skill_min_dwell_s: float = 0.10
    task_max_dwell_s: float = 12.0
    skill_max_dwell_s: float = 2.5
    task_termination_budget: float = 1.0
    skill_termination_budget: float = 1.0
    task_parameter_time_constant_s: float = 0.35
    skill_parameter_time_constant_s: float = 0.18
    task_parameter_rate_limit: float = 1.50
    skill_parameter_rate_limit: float = 2.50
    clip: float = 8.0

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TACTICHierarchyAction


class TACTICHierarchyAction(ActionTerm):
    """Cache proposals and execute learned semi-Markov termination decisions."""

    cfg: TACTICHierarchyActionCfg

    def __init__(self, cfg: TACTICHierarchyActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._raw_actions = torch.zeros(
            env.num_envs, cfg.action_dim, device=env.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.task_probability = torch.zeros(
            env.num_envs, ACTION_LAYOUT.task_dim, device=env.device
        )
        self.task_valid_mask = torch.ones_like(self.task_probability)
        self.skill_probability = torch.zeros(
            env.num_envs, ACTION_LAYOUT.skill_dim, device=env.device
        )
        self.object_probability = torch.zeros(
            env.num_envs, ACTION_LAYOUT.object_dim, device=env.device
        )
        self.task_id = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.skill_id = torch.zeros_like(self.task_id)
        self.object_id = torch.zeros_like(self.task_id)
        self.task_age = torch.zeros(env.num_envs, device=env.device)
        self.skill_age = torch.zeros_like(self.task_age)
        self.task_hazard = torch.zeros_like(self.task_age)
        self.skill_hazard = torch.zeros_like(self.task_age)
        self.task_switch = torch.zeros_like(self.task_age)
        self.skill_switch = torch.zeros_like(self.task_age)
        self.task_constraint_projection = torch.zeros_like(self.task_age)
        self.recovery_latch_seen = torch.zeros_like(self.task_age)
        self.recovery_valid_seen = torch.zeros_like(self.task_age)
        self.recovery_pressure_seen = torch.zeros_like(self.task_age)
        self.control_recovery_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.control_recovery_constraint_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.control_recovery_pressure = torch.zeros_like(self.task_age)
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.task_subgoal = torch.zeros(
            env.num_envs, ACTION_LAYOUT.task_subgoal_dim, device=env.device
        )
        self.skill_parameter = torch.zeros(
            env.num_envs, ACTION_LAYOUT.skill_param_dim, device=env.device
        )
        self.termination_probability = torch.zeros(
            env.num_envs, ACTION_LAYOUT.termination_dim, device=env.device
        )
        env.tactic_hierarchy = self

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.task_probability[env_ids] = 0.0
        self.skill_probability[env_ids] = 0.0
        self.object_probability[env_ids] = 0.0
        self.task_id[env_ids] = 0
        self.skill_id[env_ids] = 0
        self.object_id[env_ids] = 0
        self.task_age[env_ids] = 0.0
        self.skill_age[env_ids] = 0.0
        self.task_hazard[env_ids] = 0.0
        self.skill_hazard[env_ids] = 0.0
        self.task_switch[env_ids] = 0.0
        self.skill_switch[env_ids] = 0.0
        self.task_constraint_projection[env_ids] = 0.0
        self.recovery_latch_seen[env_ids] = 0.0
        self.recovery_valid_seen[env_ids] = 0.0
        self.recovery_pressure_seen[env_ids] = 0.0
        self.initialized[env_ids] = False
        self.task_subgoal[env_ids] = 0.0
        self.skill_parameter[env_ids] = 0.0
        self.termination_probability[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(
            actions, -float(self.cfg.clip), float(self.cfg.clip)
        )
        s = ACTION_LAYOUT.hierarchy_slices()
        task_probability = self._categorical_probability(
            self._processed_actions[:, s["task"]]
        )
        skill_probability = self._categorical_probability(
            self._processed_actions[:, s["skill"]]
        )
        object_probability = self._categorical_probability(
            self._processed_actions[:, s["object"]]
        )

        valid_mask = self.task_valid_mask
        if (
            isinstance(valid_mask, torch.Tensor)
            and valid_mask.shape == task_probability.shape
        ):
            masked = task_probability * valid_mask
            no_probability = masked.sum(dim=1, keepdim=True) < 1.0e-8
            fallback = valid_mask / valid_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            task_probability = torch.where(
                no_probability,
                fallback,
                masked / masked.sum(dim=1, keepdim=True).clamp_min(1.0e-8),
            )
        task_proposal = torch.argmax(task_probability, dim=-1)
        skill_proposal = torch.argmax(skill_probability, dim=-1)
        object_proposal = torch.argmax(object_probability, dim=-1)
        delivery_task = (task_proposal >= 5) & (task_proposal <= 10)
        object_proposal = torch.where(
            delivery_task,
            task_proposal - 5,
            object_proposal,
        )
        termination = torch.sigmoid(
            self._processed_actions[:, s["termination"]]
        )

        step_dt = float(getattr(self._env, "step_dt", 1.0 / 30.0))
        self.task_age.add_(step_dt)
        self.skill_age.add_(step_dt)
        recovery_pressure = self.control_recovery_pressure.clamp(0.0, 1.0)
        recovery_active = self.control_recovery_active
        recovery_latch = recovery_active.to(
            dtype=self.task_hazard.dtype
        )
        self.recovery_latch_seen[:] = recovery_latch
        self.recovery_pressure_seen[:] = recovery_pressure
        # A binding margin shortens the current option horizon.  The command
        # term opens slot 4, while the policy remains responsible for choosing
        # it; no task identity is projected here.
        if (
            isinstance(valid_mask, torch.Tensor)
            and valid_mask.shape == task_probability.shape
        ):
            self.recovery_valid_seen[:] = valid_mask[:, 4]
        else:
            self.recovery_valid_seen.zero_()
        self.task_constraint_projection.zero_()
        recovery_hazard = torch.maximum(
            recovery_pressure, 0.80 * recovery_latch
        )
        self.task_hazard.add_(
            (termination[:, 0] + 1.25 * recovery_hazard) * step_dt
        )
        self.skill_hazard.add_(
            (termination[:, 1] + recovery_hazard) * step_dt
        )
        task_due = (
            (
                self.task_hazard
                >= float(self.cfg.task_termination_budget)
            )
            & (self.task_age >= float(self.cfg.task_min_dwell_s))
        ) | (self.task_age >= float(self.cfg.task_max_dwell_s))
        skill_due = (
            (
                self.skill_hazard
                >= float(self.cfg.skill_termination_budget)
            )
            & (self.skill_age >= float(self.cfg.skill_min_dwell_s))
        ) | (self.skill_age >= float(self.cfg.skill_max_dwell_s))

        if (
            isinstance(valid_mask, torch.Tensor)
            and valid_mask.shape == task_probability.shape
        ):
            current_valid = valid_mask.gather(
                1, self.task_id.unsqueeze(1)
            ).squeeze(1)
            task_due = task_due | (current_valid < 0.5)
        task_due = task_due | (~self.initialized)
        skill_due = skill_due | task_due | (~self.initialized)
        was_initialized = self.initialized.clone()
        old_task = self.task_id.clone()
        old_skill = self.skill_id.clone()
        self.task_id[:] = torch.where(task_due, task_proposal, self.task_id)
        self.skill_id[:] = torch.where(
            skill_due, skill_proposal, self.skill_id
        )
        self.object_id[:] = torch.where(
            task_due, object_proposal, self.object_id
        )
        self.task_switch[:] = (
            task_due & self.initialized & (self.task_id != old_task)
        ).float()
        self.skill_switch[:] = (
            skill_due & self.initialized & (self.skill_id != old_skill)
        ).float()
        self.task_age[:] = torch.where(
            task_due, torch.zeros_like(self.task_age), self.task_age
        )
        self.skill_age[:] = torch.where(
            skill_due, torch.zeros_like(self.skill_age), self.skill_age
        )
        self.task_hazard[:] = torch.where(
            task_due, torch.zeros_like(self.task_hazard), self.task_hazard
        )
        self.skill_hazard[:] = torch.where(
            skill_due, torch.zeros_like(self.skill_hazard), self.skill_hazard
        )
        self.initialized[:] = True

        self.task_probability[:] = task_probability
        self.skill_probability[:] = skill_probability
        self.object_probability[:] = object_probability
        task_subgoal_target = torch.tanh(
            self._processed_actions[:, s["task_subgoal"]]
        )
        skill_parameter_target = torch.tanh(
            self._processed_actions[:, s["skill_param"]]
        )
        task_alpha = 1.0 - math.exp(
            -step_dt
            / max(float(self.cfg.task_parameter_time_constant_s), 1.0e-4)
        )
        skill_alpha = 1.0 - math.exp(
            -step_dt
            / max(float(self.cfg.skill_parameter_time_constant_s), 1.0e-4)
        )
        task_delta = task_alpha * (
            task_subgoal_target - self.task_subgoal
        )
        skill_delta = skill_alpha * (
            skill_parameter_target - self.skill_parameter
        )
        task_step_limit = (
            float(self.cfg.task_parameter_rate_limit) * step_dt
        )
        skill_step_limit = (
            float(self.cfg.skill_parameter_rate_limit) * step_dt
        )
        filtered_task_subgoal = self.task_subgoal + task_delta.clamp(
            -task_step_limit, task_step_limit
        )
        filtered_skill_parameter = self.skill_parameter + skill_delta.clamp(
            -skill_step_limit, skill_step_limit
        )
        first_proposal = (~was_initialized).unsqueeze(1)
        self.task_subgoal[:] = torch.where(
            first_proposal, task_subgoal_target, filtered_task_subgoal
        )
        self.skill_parameter[:] = torch.where(
            first_proposal, skill_parameter_target, filtered_skill_parameter
        )
        self.termination_probability[:] = termination

    def apply_actions(self):
        return

    @staticmethod
    def _categorical_probability(values: torch.Tensor) -> torch.Tensor:
        """Accept either one-hot/probability codes or unconstrained logits."""

        nonnegative = torch.all(values >= 0.0, dim=1, keepdim=True)
        positive_sum = values.sum(dim=1, keepdim=True) > 1.0e-8
        normalized = values / values.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        return torch.where(
            nonnegative & positive_sum,
            normalized,
            torch.softmax(values, dim=-1),
        )
