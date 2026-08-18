"""PPO objectives for control-aware task and skill representations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rsl_rl.algorithms import PPO

from ..tactic_layout import (
    ACTION_LAYOUT,
    BASE_TILT_INDEX,
    BASE_VX_INDEX,
    BASE_WZ_INDEX,
    CONTROL_TARGET_SLICE,
    CURRICULUM_LEVEL_INDEX,
    EXECUTED_SKILL_INDEX,
    EXECUTED_TASK_INDEX,
    GLOBAL_CONTEXT_DIM,
    INTERACTION_HARD_CONTROL_FLOOR,
    INTERACTION_SKILL_COUNT,
    MOTION_EXECUTION_EFFECT_DIM,
    MOTION_SKILL_COUNT,
    PAYLOAD_SKILL_PREVIEW_RESERVE,
    PAYLOAD_SKILL_SAFETY_RESERVE,
    PAYLOAD_SKILL_SWITCH_MARGIN,
    PREVIEW_MARGIN_INDEX,
    RELEASE_HOVER_HEIGHT,
    RELEASE_READINESS_GAIN,
    RELEASE_READINESS_THRESHOLD,
    RELEASE_TARGET_GAIN,
    RELEASE_TARGET_RADIUS,
    RELEASE_TRANSPORT_GAIN,
    RELEASE_TRANSPORT_THRESHOLD,
    RELEASE_VERTICAL_GAIN,
    RELEASE_VERTICAL_TOLERANCE,
    RELEASE_CBF_TRANSIENT_SLACK,
    SECURE_ENTRY_BARRIER_GAIN,
    SECURE_ENTRY_CENTER_RADIUS,
    SECURE_ENTRY_FINGER_RADIUS,
    SECURE_ENTRY_TCP_RADIUS,
    SECURE_CBF_TRANSIENT_SLACK,
    SELECTED_PROGRESS_DELTA_INDEX,
    SAFETY_MARGIN_INDEX,
    SUPPORT_COUNT_INDEX,
    TASK_SLOT_COUNT,
    TASK_SLOT_AVAILABLE_INDEX,
    TASK_SLOT_CARRYING_INDEX,
    TASK_SLOT_COMPLETED_INDEX,
    TASK_SLOT_CONTACT_SYMMETRY_INDEX,
    TASK_SLOT_DELIVERY_TYPE_INDEX,
    TASK_SLOT_DISTANCE_INDEX,
    TASK_SLOT_FEATURE_DIM,
    TASK_SLOT_GRIPPER_CLOSURE_INDEX,
    TASK_SLOT_HEADING_INDEX,
    TASK_SLOT_INTERACTION_STATE_SLICE,
    TASK_SLOT_LEFT_FINGER_DELTA_SLICE,
    TASK_SLOT_MANIPULATION_TYPE_INDEX,
    TASK_SLOT_OBJECT_DELTA_SLICE,
    TASK_SLOT_REACHABILITY_INDEX,
    TASK_SLOT_REMAINING_PROGRESS_INDEX,
    TASK_SLOT_REQUIRED_INDEX,
    TASK_SLOT_RIGHT_FINGER_DELTA_SLICE,
    TASK_SLOT_TARGET_DELTA_SLICE,
)


class TACTICPPO(PPO):
    """PPO with representation losses, without task or skill labels."""

    def __init__(
        self,
        *args,
        hierarchy_context_group: str = "hierarchy_context",
        control_prediction_coef: float = 0.20,
        skill_feasibility_coef: float = 0.12,
        task_outcome_coef: float = 0.16,
        task_outcome_confidence_coef: float = 0.04,
        skill_outcome_coef: float = 0.20,
        payload_survival_coef: float = 0.30,
        payload_drop_weight: float = 8.0,
        payload_drop_task_credit_penalty: float = 0.25,
        payload_drop_skill_credit_penalty: float = 0.90,
        payload_survival_replay_coef: float = 0.70,
        payload_survival_replay_capacity: int = 8192,
        payload_survival_replay_batch_size: int = 512,
        payload_survival_replay_drop_fraction: float = 0.50,
        payload_survival_replay_max_add: int = 512,
        payload_survival_learning_rate: float = 3.0e-4,
        payload_survival_rank_coef: float = 0.35,
        payload_survival_rank_margin: float = 0.05,
        skill_effect_prediction_coef: float = 0.40,
        skill_effect_confidence_coef: float = 0.05,
        motion_execution_prediction_coef: float = 0.30,
        motion_execution_confidence_coef: float = 0.04,
        embodiment_response_coef: float = 0.35,
        constraint_multiplier_coef: float = 0.05,
        grounded_effect_diversity_coef: float = 0.04,
        motion_objective_diversity_coef: float = 0.03,
        task_option_credit_coef: float = 0.04,
        skill_option_credit_coef: float = 0.06,
        event_hindsight_task_coef: float = 0.12,
        event_hindsight_skill_coef: float = 0.18,
        event_replay_task_coef: float = 0.08,
        event_replay_skill_coef: float = 0.18,
        event_replay_parameter_coef: float = 0.10,
        event_replay_motion_weight: float = 0.55,
        event_replay_interaction_weight: float = 0.45,
        event_replay_task_subgoal_weight: float = 0.35,
        event_replay_delivery_fraction: float = 0.55,
        event_replay_recovery_fraction: float = 0.15,
        event_replay_secure_fraction: float = 0.30,
        event_replay_release_fraction: float = 0.15,
        event_replay_role_oversample_cap: float = 4.0,
        event_replay_phase_oversample_cap: float = 4.0,
        event_replay_capacity: int = 8192,
        event_replay_batch_size: int = 1024,
        event_replay_max_add: int = 256,
        event_replay_min_score: float = 1.0e-3,
        task_transition_coef: float = 0.08,
        skill_transition_coef: float = 0.18,
        skill_information_coef: float = 0.060,
        task_information_coef: float = 0.020,
        information_warmup_updates: int = 16,
        transition_horizon_steps: int = 18,
        task_usage_coef: float = 0.04,
        task_frontier_coverage_coef: float = 0.12,
        interaction_phase_coef: float = 0.12,
        interaction_release_focus: float = 8.0,
        skill_usage_coef: float = 0.05,
        skill_confidence_coef: float = 0.005,
        slot_diversity_coef: float = 0.01,
        skill_diversity_coef: float = 0.01,
        motion_gain_diversity_coef: float = 0.08,
        task_control_objective_coef: float = 0.18,
        skill_predictive_control_coef: float = 0.12,
        relational_subgoal_grounding_coef: float = 0.50,
        task_skill_projection_coef: float = 0.20,
        counterfactual_task_selection_coef: float = 0.10,
        counterfactual_task_temperature: float = 0.25,
        counterfactual_skill_selection_coef: float = 0.20,
        counterfactual_skill_temperature: float = 0.20,
        counterfactual_termination_coef: float = 0.12,
        successor_decoder_coef: float = 0.20,
        physical_warmup_updates: int = 8,
        physical_core_lr_scale: float = 0.12,
        physical_adapter_lr_scale: float = 0.50,
        auxiliary_batches: int = 3,
        auxiliary_batch_size: int = 4096,
        auxiliary_learning_rate: float = 3.0e-4,
        successor_adapter_learning_rate: float = 8.0e-5,
        cbf_violation_budget: float = 0.05,
        clf_violation_budget: float = 0.08,
        constraint_dual_learning_rate: float = 0.05,
        constraint_dual_max: float = 4.0,
        constraint_dual_ema_decay: float = 0.95,
        decomposition_trust_region_radius: float = 0.006,
        freeze_locomotion_executor: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hierarchy_context_group = hierarchy_context_group
        self.control_prediction_coef = float(control_prediction_coef)
        self.skill_feasibility_coef = float(skill_feasibility_coef)
        self.task_outcome_coef = float(task_outcome_coef)
        self.task_outcome_confidence_coef = float(
            task_outcome_confidence_coef
        )
        self.skill_outcome_coef = float(skill_outcome_coef)
        self.payload_survival_coef = float(payload_survival_coef)
        self.payload_drop_weight = float(payload_drop_weight)
        self.payload_drop_task_credit_penalty = float(
            payload_drop_task_credit_penalty
        )
        self.payload_drop_skill_credit_penalty = float(
            payload_drop_skill_credit_penalty
        )
        self.payload_survival_replay_coef = float(
            payload_survival_replay_coef
        )
        self.payload_survival_replay_capacity = max(
            1, int(payload_survival_replay_capacity)
        )
        self.payload_survival_replay_batch_size = max(
            1, int(payload_survival_replay_batch_size)
        )
        self.payload_survival_replay_drop_fraction = float(
            payload_survival_replay_drop_fraction
        )
        if not 0.0 < self.payload_survival_replay_drop_fraction < 1.0:
            raise ValueError(
                "payload_survival_replay_drop_fraction must be in (0, 1)"
            )
        self.payload_survival_replay_max_add = max(
            1, int(payload_survival_replay_max_add)
        )
        self.payload_survival_learning_rate = float(
            payload_survival_learning_rate
        )
        if self.payload_survival_learning_rate <= 0.0:
            raise ValueError(
                "payload_survival_learning_rate must be positive"
            )
        self.payload_survival_rank_coef = max(
            0.0, float(payload_survival_rank_coef)
        )
        self.payload_survival_rank_margin = max(
            0.0, float(payload_survival_rank_margin)
        )
        self.skill_effect_prediction_coef = float(
            skill_effect_prediction_coef
        )
        self.skill_effect_confidence_coef = float(
            skill_effect_confidence_coef
        )
        self.motion_execution_prediction_coef = float(
            motion_execution_prediction_coef
        )
        self.motion_execution_confidence_coef = float(
            motion_execution_confidence_coef
        )
        self.embodiment_response_coef = float(embodiment_response_coef)
        self.constraint_multiplier_coef = float(
            constraint_multiplier_coef
        )
        self.grounded_effect_diversity_coef = float(
            grounded_effect_diversity_coef
        )
        self.motion_objective_diversity_coef = float(
            motion_objective_diversity_coef
        )
        self.task_option_credit_coef = float(task_option_credit_coef)
        self.skill_option_credit_coef = float(skill_option_credit_coef)
        self.event_hindsight_task_coef = float(
            event_hindsight_task_coef
        )
        self.event_hindsight_skill_coef = float(
            event_hindsight_skill_coef
        )
        self.event_replay_task_coef = float(event_replay_task_coef)
        self.event_replay_skill_coef = float(event_replay_skill_coef)
        self.event_replay_parameter_coef = float(
            event_replay_parameter_coef
        )
        factor_weight = max(
            1.0e-6,
            float(event_replay_motion_weight)
            + float(event_replay_interaction_weight),
        )
        self.event_replay_motion_weight = (
            float(event_replay_motion_weight) / factor_weight
        )
        self.event_replay_interaction_weight = (
            float(event_replay_interaction_weight) / factor_weight
        )
        self.event_replay_task_subgoal_weight = max(
            0.0, float(event_replay_task_subgoal_weight)
        )
        self.event_replay_delivery_fraction = float(
            event_replay_delivery_fraction
        )
        if not 0.0 < self.event_replay_delivery_fraction < 1.0:
            raise ValueError("Invalid event replay delivery fraction")
        self.event_replay_recovery_fraction = float(
            event_replay_recovery_fraction
        )
        if (
            self.event_replay_recovery_fraction < 0.0
            or self.event_replay_delivery_fraction
            + self.event_replay_recovery_fraction
            >= 1.0
        ):
            raise ValueError(
                "Invalid event replay recovery fraction"
            )
        self.event_replay_secure_fraction = float(
            event_replay_secure_fraction
        )
        self.event_replay_release_fraction = float(
            event_replay_release_fraction
        )
        if (
            self.event_replay_secure_fraction < 0.0
            or self.event_replay_release_fraction < 0.0
            or self.event_replay_secure_fraction
            + self.event_replay_release_fraction
            >= 1.0
        ):
            raise ValueError("Invalid event replay phase fractions")
        self.event_replay_phase_oversample_cap = max(
            1.0, float(event_replay_phase_oversample_cap)
        )
        self.event_replay_role_oversample_cap = max(
            1.0, float(event_replay_role_oversample_cap)
        )
        self.event_replay_capacity = max(1, int(event_replay_capacity))
        self.event_replay_batch_size = max(
            1, int(event_replay_batch_size)
        )
        self.event_replay_max_add = max(1, int(event_replay_max_add))
        self.event_replay_min_score = max(
            0.0, float(event_replay_min_score)
        )
        self.task_transition_coef = float(task_transition_coef)
        self.skill_transition_coef = float(skill_transition_coef)
        self.skill_information_coef = float(skill_information_coef)
        self.task_information_coef = float(task_information_coef)
        self.information_warmup_updates = int(information_warmup_updates)
        self.transition_horizon_steps = max(2, int(transition_horizon_steps))
        self.task_usage_coef = float(task_usage_coef)
        self.task_frontier_coverage_coef = float(
            task_frontier_coverage_coef
        )
        self.interaction_phase_coef = float(interaction_phase_coef)
        self.interaction_release_focus = float(
            interaction_release_focus
        )
        if self.interaction_release_focus < 1.0:
            raise ValueError(
                "interaction_release_focus must be at least one"
            )
        self.skill_usage_coef = float(skill_usage_coef)
        self.skill_confidence_coef = float(skill_confidence_coef)
        self.slot_diversity_coef = float(slot_diversity_coef)
        self.skill_diversity_coef = float(skill_diversity_coef)
        self.motion_gain_diversity_coef = float(
            motion_gain_diversity_coef
        )
        self.task_control_objective_coef = float(
            task_control_objective_coef
        )
        self.skill_predictive_control_coef = float(
            skill_predictive_control_coef
        )
        self.relational_subgoal_grounding_coef = float(
            relational_subgoal_grounding_coef
        )
        self.task_skill_projection_coef = float(
            task_skill_projection_coef
        )
        self.counterfactual_task_selection_coef = float(
            counterfactual_task_selection_coef
        )
        self.counterfactual_task_temperature = max(
            0.05, float(counterfactual_task_temperature)
        )
        self.counterfactual_skill_selection_coef = float(
            counterfactual_skill_selection_coef
        )
        self.counterfactual_skill_temperature = max(
            0.05, float(counterfactual_skill_temperature)
        )
        self.counterfactual_termination_coef = float(
            counterfactual_termination_coef
        )
        self.successor_decoder_coef = float(successor_decoder_coef)
        self.physical_warmup_updates = int(physical_warmup_updates)
        self.physical_core_lr_scale = float(physical_core_lr_scale)
        self.physical_adapter_lr_scale = float(
            physical_adapter_lr_scale
        )
        self.auxiliary_batches = int(auxiliary_batches)
        self.auxiliary_batch_size = int(auxiliary_batch_size)
        self.auxiliary_learning_rate = float(auxiliary_learning_rate)
        self.successor_adapter_learning_rate = float(
            successor_adapter_learning_rate
        )
        self.cbf_violation_budget = float(cbf_violation_budget)
        self.clf_violation_budget = float(clf_violation_budget)
        self.constraint_dual_learning_rate = float(
            constraint_dual_learning_rate
        )
        self.constraint_dual_max = float(constraint_dual_max)
        self.constraint_dual_ema_decay = float(
            constraint_dual_ema_decay
        )
        self.decomposition_trust_region_radius = float(
            decomposition_trust_region_radius
        )
        self.freeze_locomotion_executor = bool(
            freeze_locomotion_executor
        )
        if not 0.0 <= self.cbf_violation_budget < 1.0:
            raise ValueError("cbf_violation_budget must be in [0, 1)")
        if not 0.0 <= self.clf_violation_budget < 1.0:
            raise ValueError("clf_violation_budget must be in [0, 1)")
        if self.constraint_dual_learning_rate <= 0.0:
            raise ValueError("constraint_dual_learning_rate must be positive")
        if self.constraint_dual_max <= 0.0:
            raise ValueError("constraint_dual_max must be positive")
        if not 0.0 <= self.constraint_dual_ema_decay < 1.0:
            raise ValueError("constraint_dual_ema_decay must be in [0, 1)")
        if self.decomposition_trust_region_radius <= 0.0:
            raise ValueError(
                "decomposition_trust_region_radius must be positive"
            )
        physical_core_prefixes = (
            "physical_backbone.",
            "physical_head.",
        )
        physical_adapter_prefixes = (
            "physical_conditioner.",
            "film_head.",
            "physical_residual_head.",
            "support_skill_encoder.",
            "support_reference_head.",
            "support_gate_head.",
            "support_residual_head.",
            "motion_support_basis",
            "support_gate_logit",
            "gripper_head.",
            "interaction_gripper_basis",
            "wheel_residual_encoder.",
            "wheel_residual_head.",
            "wheel_skill_gate_head.",
            "wheel_skill_gate_logit",
            "embodiment_motion_basis",
            "embodiment_response_matrix",
            "motion_action_capacity",
            "motion_kinematic_gain",
            "wheel_breakaway_action",
        )
        base_learning_rate = float(self.learning_rate)
        core_parameters = []
        adapter_parameters = []
        hierarchy_parameters = []
        for name, parameter in self.policy.named_parameters():
            actor_name = (
                name[len("actor.") :]
                if name.startswith("actor.")
                else name
            )
            if actor_name.startswith(physical_core_prefixes):
                core_parameters.append(parameter)
            elif actor_name.startswith(physical_adapter_prefixes):
                adapter_parameters.append(parameter)
            else:
                hierarchy_parameters.append(parameter)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": core_parameters,
                    "lr": base_learning_rate
                    * self.physical_core_lr_scale,
                },
                {
                    "params": adapter_parameters,
                    "lr": base_learning_rate
                    * self.physical_adapter_lr_scale,
                },
                {
                    "params": hierarchy_parameters,
                    "lr": base_learning_rate,
                },
            ]
        )
        excluded_auxiliary_prefixes = (
            *physical_core_prefixes,
            *physical_adapter_prefixes,
        )
        self._auxiliary_parameters = [
            parameter
            for name, parameter in self.policy.actor.named_parameters()
            if not name.startswith(excluded_auxiliary_prefixes)
        ]
        self.auxiliary_optimizer = torch.optim.Adam(
            self._auxiliary_parameters,
            lr=self.auxiliary_learning_rate,
        )
        self._payload_survival_parameters = [
            *self.policy.actor.payload_survival_encoder.parameters(),
            *self.policy.actor.payload_survival_head.parameters(),
        ]
        self.payload_survival_optimizer = torch.optim.Adam(
            self._payload_survival_parameters,
            lr=self.payload_survival_learning_rate,
        )
        successor_adapter_prefixes = (
            "embodiment_motion_basis",
            "embodiment_response_matrix",
            "motion_action_capacity",
            "motion_kinematic_gain",
        )
        self._successor_adapter_parameters = [
            parameter
            for name, parameter in self.policy.actor.named_parameters()
            if name.startswith(successor_adapter_prefixes)
        ]
        self.successor_adapter_optimizer = torch.optim.Adam(
            self._successor_adapter_parameters,
            lr=self.successor_adapter_learning_rate,
        )
        self.training_stage = "joint"
        self._physical_adaptation_stage: int | None = None
        self._physical_noise_hook = None
        self._hierarchy_noise_hook = None
        self._skill_information_history: list[float] = []
        self._task_information_history: list[float] = []
        self._information_context_history: list[torch.Tensor] = []
        self._information_done_history: list[torch.Tensor] = []
        self._event_replay_context: torch.Tensor | None = None
        self._event_replay_actions: torch.Tensor | None = None
        self._event_replay_priority: torch.Tensor | None = None
        self._event_replay_task_valid: torch.Tensor | None = None
        self._event_replay_task_detail_valid: torch.Tensor | None = None
        self._event_replay_recovery_valid: torch.Tensor | None = None
        self._payload_replay_context: torch.Tensor | None = None
        self._payload_replay_actions: torch.Tensor | None = None
        self._payload_replay_target: torch.Tensor | None = None
        self._payload_replay_drop: torch.Tensor | None = None
        self._decomposition_anchor: dict[str, torch.Tensor] = {}
        self._motion_parameter_hooks: list[
            torch.utils.hooks.RemovableHandle
        ] = []
        self.policy.update_normalization = lambda obs: None
        self._configure_physical_stage()

    def set_training_stage(self, stage: str) -> None:
        """Select joint, hierarchy-only, or recovery-adapter learning."""

        if stage not in (
            "joint",
            "stability",
            "upper",
            "decomposition",
            "motion_selector",
            "interaction_selector",
            "motion_skill",
            "payload_motion",
            "recovery",
            "survival",
        ):
            raise ValueError(
                "TACTIC-HRL supports training_stage='joint', 'stability', 'upper', "
                "'decomposition', 'motion_selector', "
                "'interaction_selector', 'motion_skill', "
                "'payload_motion', "
                "'recovery', or 'survival', "
                f"got {stage!r}"
            )
        if (
            stage != "upper"
            and self._hierarchy_noise_hook is not None
        ):
            self._hierarchy_noise_hook.remove()
            self._hierarchy_noise_hook = None
        for hook in self._motion_parameter_hooks:
            hook.remove()
        self._motion_parameter_hooks.clear()
        self.training_stage = stage
        if stage == "stability":
            # The stability stage is intentionally a lower-level continuation:
            # only the migrated ZYB executor is optimized and all hierarchy
            # conditioned actuator decoders are bypassed.
            self.policy.actor.stability_teacher_only = True
        self.policy.actor.payload_survival_control_enabled = (
            stage != "survival"
        )
        self._physical_adaptation_stage = None
        for parameter in self.policy.parameters():
            parameter.requires_grad_(True)
        self._configure_physical_stage()
        if stage == "decomposition":
            self._capture_decomposition_anchor()
        else:
            self._decomposition_anchor.clear()

    def _decomposition_module_name(self, parameter_name: str) -> str | None:
        for module_name in self.policy.actor.DECOMPOSITION_MODULE_NAMES:
            if parameter_name.startswith(module_name + "."):
                return module_name
        return None

    @torch.no_grad()
    def _capture_decomposition_anchor(self) -> None:
        self._decomposition_anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.policy.actor.named_parameters()
            if self._decomposition_module_name(name) is not None
        }

    def _decomposition_relative_drifts(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._decomposition_anchor:
            return {}
        grouped_delta: dict[str, torch.Tensor] = {}
        grouped_reference: dict[str, torch.Tensor] = {}
        for name, parameter in self.policy.actor.named_parameters():
            module_name = self._decomposition_module_name(name)
            if (
                module_name is None
                or name not in self._decomposition_anchor
            ):
                continue
            anchor = self._decomposition_anchor[name]
            delta_square = (parameter - anchor).square().sum()
            reference_square = anchor.square().sum()
            if module_name in grouped_delta:
                grouped_delta[module_name] = (
                    grouped_delta[module_name] + delta_square
                )
                grouped_reference[module_name] = (
                    grouped_reference[module_name] + reference_square
                )
            else:
                grouped_delta[module_name] = delta_square
                grouped_reference[module_name] = reference_square
        return {
            module_name: torch.sqrt(
                delta_square
                / grouped_reference[module_name].clamp_min(1.0e-12)
            )
            for module_name, delta_square in grouped_delta.items()
        }

    @torch.no_grad()
    def _project_decomposition_trust_region(self) -> None:
        if (
            self.training_stage != "decomposition"
            or not self._decomposition_anchor
        ):
            return
        actor_parameters = dict(self.policy.actor.named_parameters())
        drifts = self._decomposition_relative_drifts()
        for module_name, drift in drifts.items():
            radius = self.decomposition_trust_region_radius
            if float(drift.item()) <= radius:
                continue
            scale = radius / drift.clamp_min(1.0e-12)
            prefix = module_name + "."
            for name, anchor in self._decomposition_anchor.items():
                if not name.startswith(prefix):
                    continue
                parameter = actor_parameters[name]
                parameter.copy_(
                    anchor + scale * (parameter - anchor)
                )

    def _configure_physical_stage(self) -> None:
        actor = self.policy.actor
        if self.training_stage == "stability":
            stage = -10
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            for module in (actor.physical_backbone, actor.physical_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                # Keep the known-safe exploration scale while the student
                # learns the loaded/payload-robust physical mean.
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "payload_motion":
            stage = -8
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            for module_name in actor.PAYLOAD_MOTION_MODULE_NAMES:
                module = getattr(actor, module_name)
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for parameter_name in actor.PAYLOAD_MOTION_PARAMETER_NAMES:
                getattr(actor, parameter_name).requires_grad_(True)
            for parameter in actor.skill_continuous_head.parameters():
                row_mask = torch.zeros_like(parameter)
                row_mask[:2] = 1.0
                self._motion_parameter_hooks.append(
                    parameter.register_hook(
                        lambda gradient, mask=row_mask: gradient * mask
                    )
                )
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "motion_skill":
            stage = -7
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_motion_selector_trainable(True)
            for parameter in actor.skill_continuous_head.parameters():
                parameter.requires_grad_(True)
                row_mask = torch.zeros_like(parameter)
                row_mask[:2] = 1.0
                self._motion_parameter_hooks.append(
                    parameter.register_hook(
                        lambda gradient, mask=row_mask: gradient * mask
                    )
                )
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "motion_selector":
            stage = -6
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_motion_selector_trainable(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "interaction_selector":
            stage = -9
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_interaction_selector_trainable(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "decomposition":
            stage = -5
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_decomposition_trainable(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "recovery":
            stage = -3
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_recovery_adapter_trainable(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "survival":
            stage = -4
            if stage == self._physical_adaptation_stage:
                return
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actor.set_payload_survival_trainable(True)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                noise.requires_grad_(False)
            self._physical_adaptation_stage = stage
            return

        if self.training_stage == "upper":
            stage = -2
            if stage == self._physical_adaptation_stage:
                return
            actor.set_physical_executor_trainable(False)
            actor.set_recovery_adapter_trainable(False)
            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None and self._hierarchy_noise_hook is None:
                mask = torch.ones_like(noise)
                mask[: actor.physical_action_dim] = 0.0
                self._hierarchy_noise_hook = noise.register_hook(
                    lambda gradient, gradient_mask=mask: (
                        gradient * gradient_mask
                    )
                )
            self._physical_adaptation_stage = stage
            return

        if self.freeze_locomotion_executor:
            stage = -1
            if stage == self._physical_adaptation_stage:
                return
            # Preserve the transferred support policy while allowing the
            # lower skill layer to learn motion and grasp execution.
            frozen_modules = (
                actor.physical_backbone,
                actor.physical_head,
                actor.physical_conditioner,
                actor.film_head,
                actor.physical_residual_head,
            )
            for module in frozen_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            for module in (
                actor.gripper_head,
                actor.support_skill_encoder,
                actor.support_reference_head,
                actor.support_gate_head,
                actor.support_residual_head,
                actor.wheel_residual_encoder,
                actor.wheel_residual_head,
                actor.wheel_skill_gate_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for parameter in (
                actor.embodiment_motion_basis,
                actor.embodiment_response_matrix,
                actor.motion_action_capacity,
                actor.motion_kinematic_gain,
                actor.wheel_breakaway_action,
                actor.wheel_skill_gate_logit,
                actor.motion_support_basis,
                actor.support_gate_logit,
                actor.interaction_gripper_basis,
            ):
                parameter.requires_grad_(True)

            noise = getattr(
                self.policy, "std", getattr(self.policy, "log_std", None)
            )
            if noise is not None:
                with torch.no_grad():
                    if hasattr(self.policy, "std"):
                        noise[:12].fill_(0.001)
                        noise[12:16].fill_(0.02)
                    else:
                        noise[:12].fill_(torch.log(noise.new_tensor(0.001)))
                        noise[12:16].fill_(torch.log(noise.new_tensor(0.02)))
                if self._physical_noise_hook is None:
                    mask = torch.ones_like(noise)
                    mask[:16] = 0.0
                    self._physical_noise_hook = noise.register_hook(
                        lambda gradient, gradient_mask=mask: (
                            gradient * gradient_mask
                        )
                    )
            self._physical_adaptation_stage = stage
            return

        update_count = int(self.policy.tactic_training_updates.item())
        warmup = max(1, self.physical_warmup_updates)
        if update_count < warmup:
            stage = 0
        elif update_count < 4 * warmup:
            stage = 1
        elif update_count < 12 * warmup:
            stage = 2
        else:
            stage = 3
        if stage == self._physical_adaptation_stage:
            return

        # Preserve the transferred representation during the short hierarchy
        # warm-up, then release the output layer and backbone from top to
        # bottom.  PPO still applies a smaller optimizer rate to every
        # transferred tensor.
        for parameter in actor.physical_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.physical_head.parameters():
            parameter.requires_grad_(stage >= 2)
        if stage >= 2:
            for parameter in actor.physical_backbone[2].parameters():
                parameter.requires_grad_(True)
        if stage >= 3:
            for parameter in actor.physical_backbone[0].parameters():
                parameter.requires_grad_(True)
        for module in (
            actor.physical_conditioner,
            actor.film_head,
            actor.physical_residual_head,
            actor.support_skill_encoder,
            actor.support_reference_head,
            actor.support_gate_head,
            actor.support_residual_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(stage >= 1)
        for module in (
            actor.wheel_residual_encoder,
            actor.wheel_residual_head,
            actor.wheel_skill_gate_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        actor.embodiment_motion_basis.requires_grad_(True)
        actor.embodiment_response_matrix.requires_grad_(True)
        actor.motion_action_capacity.requires_grad_(True)
        actor.motion_kinematic_gain.requires_grad_(True)
        actor.wheel_breakaway_action.requires_grad_(stage >= 1)
        actor.wheel_skill_gate_logit.requires_grad_(True)
        actor.motion_support_basis.requires_grad_(stage >= 1)
        actor.support_gate_logit.requires_grad_(stage >= 1)
        noise = getattr(
            self.policy, "std", getattr(self.policy, "log_std", None)
        )
        if noise is not None:
            leg_exploration = (0.001, 0.008, 0.012, 0.015)[stage]
            # Task-subgoal and discrete skill sampling provide structured
            # exploration.  Small joint-space noise avoids breaking the
            # coordinated wheel patterns chosen by the skill policy.
            wheel_exploration = (0.01, 0.015, 0.02, 0.03)[stage]
            with torch.no_grad():
                if hasattr(self.policy, "std"):
                    noise[:12].fill_(leg_exploration)
                    noise[12:16].fill_(wheel_exploration)
                else:
                    noise[:12].fill_(
                        torch.log(noise.new_tensor(leg_exploration))
                    )
                    noise[12:16].fill_(
                        torch.log(noise.new_tensor(wheel_exploration))
                    )
        self._physical_adaptation_stage = stage

    def process_env_step(self, obs, rewards, dones, extras):
        """Add safe transition information to the PPO reward.

        The discriminator does not prescribe skill semantics. It rewards an
        option only when its measured transition is identifiable and remains
        inside the learned control envelope.
        """

        current_obs = self.transition.observations
        if current_obs is None:
            return super().process_env_step(obs, rewards, dones, extras)
        current_context = current_obs[self.hierarchy_context_group]
        next_context = obs[self.hierarchy_context_group]
        self._information_context_history.append(
            current_context.detach().clone()
        )
        self._information_done_history.append(
            dones.detach().float().reshape(-1).clone()
        )
        if len(self._information_context_history) > self.transition_horizon_steps:
            self._information_context_history.pop(0)
            self._information_done_history.pop(0)
        if len(self._information_context_history) < self.transition_horizon_steps:
            return super().process_env_step(obs, rewards, dones, extras)

        option_context = self._information_context_history[0]
        done_window = torch.stack(
            self._information_done_history, dim=0
        ).amax(dim=0)
        with torch.no_grad():
            task_logits, skill_logits = self.policy.transition_logits(
                option_context, next_context
            )
            task_target = torch.round(
                option_context[:, EXECUTED_TASK_INDEX]
                * float(ACTION_LAYOUT.task_dim - 1)
            ).long().clamp(0, ACTION_LAYOUT.task_dim - 1)
            skill_target = torch.round(
                option_context[:, EXECUTED_SKILL_INDEX]
                * float(ACTION_LAYOUT.skill_dim - 1)
            ).long().clamp(0, ACTION_LAYOUT.skill_dim - 1)
            next_task = torch.round(
                next_context[:, EXECUTED_TASK_INDEX]
                * float(ACTION_LAYOUT.task_dim - 1)
            ).long().clamp(0, ACTION_LAYOUT.task_dim - 1)
            next_skill = torch.round(
                next_context[:, EXECUTED_SKILL_INDEX]
                * float(ACTION_LAYOUT.skill_dim - 1)
            ).long().clamp(0, ACTION_LAYOUT.skill_dim - 1)
            task_information = (
                torch.log_softmax(task_logits, dim=-1)
                .gather(1, task_target.unsqueeze(1))
                .squeeze(1)
                + math.log(float(ACTION_LAYOUT.task_dim))
            ) / math.log(float(ACTION_LAYOUT.task_dim))
            skill_grid = skill_logits.reshape(
                -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
            )
            motion_logits = torch.logsumexp(skill_grid, dim=2)
            interaction_logits = torch.logsumexp(skill_grid, dim=1)
            motion_target = skill_target // INTERACTION_SKILL_COUNT
            interaction_target = skill_target % INTERACTION_SKILL_COUNT
            motion_information = (
                torch.log_softmax(motion_logits, dim=-1)
                .gather(1, motion_target.unsqueeze(1))
                .squeeze(1)
                + math.log(float(MOTION_SKILL_COUNT))
            ) / math.log(float(MOTION_SKILL_COUNT))
            interaction_information = (
                torch.log_softmax(interaction_logits, dim=-1)
                .gather(1, interaction_target.unsqueeze(1))
                .squeeze(1)
                + math.log(float(INTERACTION_SKILL_COUNT))
            ) / math.log(float(INTERACTION_SKILL_COUNT))
            slots = option_context[:, GLOBAL_CONTEXT_DIM:].reshape(
                -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
            )
            rows = torch.arange(slots.shape[0], device=slots.device)
            selected_slot = slots[rows, task_target]
            interaction_active = (
                selected_slot[:, TASK_SLOT_MANIPULATION_TYPE_INDEX]
                + selected_slot[:, TASK_SLOT_DELIVERY_TYPE_INDEX]
            ).clamp(0.0, 1.0)
            skill_information = (
                0.60 * motion_information
                + 0.40 * interaction_active * interaction_information
            )
            active_tasks = (
                (slots[:, :, 11] > 0.5)
                & (slots[:, :, 12] < 0.5)
                & (slots[:, :, 13] > 0.5)
            ).sum(dim=1)
            multi_task = (active_tasks > 1).float()
            safety = next_context[:, CONTROL_TARGET_SLICE].min(dim=1).values
            positive_progress = (
                8.0 * next_context[:, SELECTED_PROGRESS_DELTA_INDEX]
            ).clamp(0.0, 1.0)
            gate = (
                0.15 + 0.55 * safety + 0.30 * positive_progress
            ).clamp(0.0, 1.0)
            nonterminal = 1.0 - done_window
            stable_task = (task_target == next_task).float()
            stable_skill = stable_task * (skill_target == next_skill).float()
            update_count = int(self.policy.tactic_training_updates.item())
            warmup = min(
                1.0,
                update_count / max(1, self.information_warmup_updates),
            )
            skill_bonus = (
                self.skill_information_coef
                * warmup
                * gate
                * skill_information.clamp(-1.0, 1.0)
                * nonterminal
                * stable_skill
            )
            task_bonus = (
                self.task_information_coef
                * warmup
                * gate
                * task_information.clamp(-1.0, 1.0)
                * multi_task
                * nonterminal
                * stable_task
            )
            augmented_rewards = rewards + skill_bonus + task_bonus
            self._skill_information_history.append(
                float(skill_bonus.mean().item())
            )
            self._task_information_history.append(
                float(task_bonus.mean().item())
            )
        return super().process_env_step(
            obs, augmented_rewards, dones, extras
        )

    @staticmethod
    def _off_diagonal_energy(embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        gram = embeddings @ embeddings.transpose(-1, -2)
        width = gram.shape[-1]
        identity = torch.eye(width, device=gram.device, dtype=gram.dtype)
        return ((gram - identity) ** 2).sum() / max(1, width * (width - 1))

    @staticmethod
    def _centered_option_advantage(
        signal: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        normalizer = valid.sum().clamp_min(1.0)
        mean = (signal * valid).sum() / normalizer
        centered = signal - mean
        variance = (centered.square() * valid).sum() / normalizer
        return (
            centered / torch.sqrt(variance + 0.05**2)
        ).clamp(-3.0, 3.0).detach()

    @staticmethod
    def _payload_survival_targets(
        current_context: torch.Tensor,
        next_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Label whether an active payload relation survives the transition."""

        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        next_slots = next_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        current_payload = current_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].amax(dim=1).clamp(0.0, 1.0)
        next_payload = next_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].amax(dim=1).clamp(0.0, 1.0)
        next_placed = next_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE.stop - 1
        ].amax(dim=1).clamp(0.0, 1.0)
        next_completed = next_slots[
            :, :, TASK_SLOT_COMPLETED_INDEX
        ].amax(dim=1).clamp(0.0, 1.0)
        survival_target = torch.maximum(
            next_payload,
            torch.maximum(next_placed, next_completed),
        )
        payload_drop = (
            current_payload * (1.0 - survival_target)
        ).clamp(0.0, 1.0)
        return current_payload, survival_target, payload_drop

    @staticmethod
    def _current_interaction_targets(
        current_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Classify learning stages without relaxing physical release."""

        contact = current_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE.start
        ].clamp(0.0, 1.0)
        lift = current_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE.start + 1
        ].clamp(0.0, 1.0)
        transport = current_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE.start + 2
        ].clamp(0.0, 1.0)
        place = current_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE.start + 3
        ].clamp(0.0, 1.0)
        tcp_distance = torch.linalg.vector_norm(
            current_slots[:, :, TASK_SLOT_OBJECT_DELTA_SLICE], dim=-1
        )
        left_delta = (
            0.75
            * current_slots[:, :, TASK_SLOT_LEFT_FINGER_DELTA_SLICE]
        )
        right_delta = (
            0.75
            * current_slots[:, :, TASK_SLOT_RIGHT_FINGER_DELTA_SLICE]
        )
        finger_distance = torch.maximum(
            torch.linalg.vector_norm(left_delta, dim=-1),
            torch.linalg.vector_norm(right_delta, dim=-1),
        )
        center_error = torch.linalg.vector_norm(
            0.5 * (left_delta + right_delta), dim=-1
        )
        secure_entry = torch.minimum(
            torch.minimum(
                torch.sigmoid(
                    SECURE_ENTRY_BARRIER_GAIN
                    * (SECURE_ENTRY_CENTER_RADIUS - center_error)
                ),
                torch.sigmoid(
                    SECURE_ENTRY_BARRIER_GAIN
                    * (SECURE_ENTRY_FINGER_RADIUS - finger_distance)
                ),
            ),
            torch.sigmoid(
                SECURE_ENTRY_BARRIER_GAIN
                * (SECURE_ENTRY_TCP_RADIUS - tcp_distance)
            ),
        )
        carrying = current_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        hold_evidence = torch.maximum(
            torch.maximum(contact, lift), carrying
        )
        secure_entry = torch.maximum(
            secure_entry, hold_evidence
        ).clamp(0.0, 1.0)
        target_delta = current_slots[
            :, :, TASK_SLOT_TARGET_DELTA_SLICE
        ]
        target_distance = 1.5 * torch.linalg.vector_norm(
            target_delta[:, :, :2], dim=-1
        )
        target_vertical_error = torch.abs(
            1.5 * target_delta[:, :, 2] + RELEASE_HOVER_HEIGHT
        )
        release_frontier = (
            carrying
            * (0.15 + 0.85 * transport)
            * torch.sqrt(
                (
                    torch.exp(-target_distance / 0.45)
                    * torch.exp(-target_vertical_error / 0.20)
                ).clamp_min(0.0)
            )
        ).clamp(0.0, 1.0)
        release_frontier = torch.maximum(release_frontier, place)
        approach = (
            (1.0 - contact)
            * (1.0 - secure_entry)
            * (1.0 - carrying)
        )
        secure = secure_entry * (1.0 - 0.85 * release_frontier)
        phase_score = torch.stack(
            (approach, secure, release_frontier), dim=-1
        )
        phase_id = torch.argmax(phase_score, dim=-1)
        # Near-release states are replayed before an actual opening event.
        # The physical projection still blocks the gripper outside the strict
        # contact, placement, transport, and stability admissible set.
        near_release = (carrying > 0.5) & (release_frontier >= 0.12)
        return torch.where(
            near_release,
            torch.full_like(phase_id, 2),
            phase_id,
        )

    @staticmethod
    def _interaction_event_targets(
        current_context: torch.Tensor,
        next_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return grounded event scores and their interaction-stage labels."""

        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        next_slots = next_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        delivery_active = current_slots[
            :, :, TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        remaining_progress_delta = (
            current_slots[:, :, TASK_SLOT_REMAINING_PROGRESS_INDEX]
            - next_slots[:, :, TASK_SLOT_REMAINING_PROGRESS_INDEX]
        ).clamp(0.0, 0.05)
        task_progress_credit = (
            remaining_progress_delta / 0.008
        ).clamp(0.0, 1.0)
        task_completion_credit = (
            next_slots[:, :, TASK_SLOT_COMPLETED_INDEX]
            - current_slots[:, :, TASK_SLOT_COMPLETED_INDEX]
        ).clamp(0.0, 1.0)
        state_delta = (
            next_slots[:, :, TASK_SLOT_INTERACTION_STATE_SLICE]
            - current_slots[:, :, TASK_SLOT_INTERACTION_STATE_SLICE]
        ).clamp_min(0.0)
        next_state = next_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE
        ].clamp(0.0, 1.0)
        symmetry = next_slots[
            :, :, TASK_SLOT_CONTACT_SYMMETRY_INDEX
        ].clamp(0.0, 1.0)
        closure = next_slots[
            :, :, TASK_SLOT_GRIPPER_CLOSURE_INDEX
        ].clamp(0.0, 1.0)
        current_tcp_distance = torch.linalg.vector_norm(
            current_slots[:, :, TASK_SLOT_OBJECT_DELTA_SLICE], dim=-1
        )
        tcp_distance = torch.linalg.vector_norm(
            next_slots[:, :, TASK_SLOT_OBJECT_DELTA_SLICE], dim=-1
        )
        near_gripper = torch.sigmoid(30.0 * (0.20 - tcp_distance))
        instantaneous_grasp = torch.sqrt(
            (symmetry * closure).clamp_min(0.0)
        )
        retained_grasp = torch.sqrt(
            (
                next_state[:, :, 0]
                * near_gripper
                * closure
            ).clamp_min(0.0)
        )
        grasp_quality = torch.maximum(
            instantaneous_grasp, retained_grasp
        )

        approach_progress = (
            (current_tcp_distance - tcp_distance) / 0.010
        ).clamp(0.0, 1.0)
        approach_proximity = torch.sigmoid(
            12.0 * (0.60 - tcp_distance)
        )
        approach_credit = (
            approach_progress
            * approach_proximity
            * (1.0 - next_state[:, :, 0])
        )
        contact_credit = state_delta[:, :, 0] * grasp_quality
        lift_credit = state_delta[:, :, 1] * (
            0.15 + 0.85 * grasp_quality
        )
        transport_credit = state_delta[:, :, 2] * (
            0.10 + 0.90 * grasp_quality
        )
        place_credit = state_delta[:, :, 3]
        completion_credit = task_completion_credit
        carry_credit = (
            next_state[:, :, 1]
            * next_state[:, :, 2].clamp_min(0.05)
            * grasp_quality
            * near_gripper
        )
        current_target_distance = 1.5 * torch.linalg.vector_norm(
            current_slots[
                :, :, TASK_SLOT_TARGET_DELTA_SLICE
            ][:, :, :2],
            dim=-1,
        )
        next_target_distance = 1.5 * torch.linalg.vector_norm(
            next_slots[
                :, :, TASK_SLOT_TARGET_DELTA_SLICE
            ][:, :, :2],
            dim=-1,
        )
        target_progress_credit = (
            (
                current_target_distance - next_target_distance
            )
            / 0.030
        ).clamp(0.0, 1.0)
        target_progress_credit = (
            target_progress_credit
            * next_slots[:, :, TASK_SLOT_CARRYING_INDEX].clamp(0.0, 1.0)
            * grasp_quality
        )
        current_target_delta = current_slots[
            :, :, TASK_SLOT_TARGET_DELTA_SLICE
        ]
        next_target_delta = next_slots[
            :, :, TASK_SLOT_TARGET_DELTA_SLICE
        ]
        current_hover_error = torch.abs(
            1.5 * current_target_delta[:, :, 2]
            + RELEASE_HOVER_HEIGHT
        )
        next_hover_error = torch.abs(
            1.5 * next_target_delta[:, :, 2]
            + RELEASE_HOVER_HEIGHT
        )
        current_release_readiness = (
            current_slots[:, :, TASK_SLOT_CARRYING_INDEX]
            * (
                0.20
                + 0.80
                * current_slots[:, :, TASK_SLOT_INTERACTION_STATE_SLICE][
                    :, :, 2
                ]
            )
            * torch.exp(
                -current_target_distance / 0.20
                -0.5 * current_hover_error / 0.10
            )
        ).clamp(0.0, 1.0)
        next_release_readiness = (
            next_slots[:, :, TASK_SLOT_CARRYING_INDEX]
            * (
                0.20
                + 0.80
                * next_slots[:, :, TASK_SLOT_INTERACTION_STATE_SLICE][
                    :, :, 2
                ]
            )
            * torch.exp(
                -next_target_distance / 0.20
                -0.5 * next_hover_error / 0.10
            )
        ).clamp(0.0, 1.0)
        release_readiness_credit = (
            (
                next_release_readiness
                - current_release_readiness
            )
            / 0.025
        ).clamp(0.0, 1.0)
        secure_credit = (
            0.45 * contact_credit
            + 0.80 * lift_credit
            + 1.20 * transport_credit
            + 0.20 * carry_credit
            + 0.85 * target_progress_credit
            + 1.00 * release_readiness_credit
        )
        release_credit = 2.50 * place_credit + 3.00 * completion_credit
        physical_stage_credit = (
            0.30 * approach_credit + secure_credit + release_credit
        )
        general_task_credit = (
            0.35 * task_progress_credit
            + 1.50 * task_completion_credit
        )
        stage_credit = (
            general_task_credit
            + delivery_active * physical_stage_credit
        )
        current_control_quality = current_context[
            :, CONTROL_TARGET_SLICE
        ].clamp(0.0, 1.0)
        next_control_quality = next_context[
            :, CONTROL_TARGET_SLICE
        ].clamp(0.0, 1.0)
        barrier_quality = torch.minimum(
            current_control_quality[:, :2].amin(
                dim=1, keepdim=True
            ),
            next_control_quality[:, :2].amin(
                dim=1, keepdim=True
            ),
        )
        convergence_quality = torch.minimum(
            current_control_quality[:, 2:].mean(
                dim=1, keepdim=True
            ),
            next_control_quality[:, 2:].mean(
                dim=1, keepdim=True
            ),
        )
        barrier_gate = torch.sigmoid(
            20.0 * (barrier_quality - 0.12)
        )
        hard_safe = (
            barrier_quality >= INTERACTION_HARD_CONTROL_FLOOR
        ).to(barrier_quality.dtype)
        safety_discount = (
            hard_safe
            * barrier_quality.square()
            * barrier_gate
            * (0.35 + 0.65 * convergence_quality)
        )
        event_score = (
            stage_credit * safety_discount
        ).clamp(0.0, 1.0)
        interaction_target = TACTICPPO._current_interaction_targets(
            current_slots
        )
        interaction_target = torch.where(
            delivery_active > 0.5,
            interaction_target,
            torch.zeros_like(interaction_target),
        )
        return event_score, interaction_target

    @classmethod
    def _interaction_event_credit(
        cls,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
        task_target: torch.Tensor,
    ) -> torch.Tensor:
        event_score, _ = cls._interaction_event_targets(
            current_context, next_context
        )
        return event_score.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1)

    @staticmethod
    def _recovery_event_credit(
        current_context: torch.Tensor,
        next_context: torch.Tensor,
    ) -> torch.Tensor:
        """Score retained-payload transitions that restore viability."""

        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        next_slots = next_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        current_carrying = current_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].max(dim=1).values.clamp(0.0, 1.0)
        next_carrying = next_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].max(dim=1).values.clamp(0.0, 1.0)
        payload_retention = torch.minimum(
            current_carrying, next_carrying
        )

        def recovery_pressure(context: torch.Tensor) -> torch.Tensor:
            tilt = 0.60 * context[:, BASE_TILT_INDEX]
            support = 4.0 * context[:, SUPPORT_COUNT_INDEX]
            margin = torch.minimum(
                context[:, SAFETY_MARGIN_INDEX],
                context[:, PREVIEW_MARGIN_INDEX],
            )
            return torch.maximum(
                ((tilt - 0.30) / 0.18).clamp(0.0, 1.0),
                torch.maximum(
                    ((2.0 - support) / 1.0).clamp(0.0, 1.0),
                    ((0.16 - margin) / 0.16).clamp(0.0, 1.0),
                ),
            )

        current_pressure = recovery_pressure(current_context)
        next_pressure = recovery_pressure(next_context)
        pressure_drop = (
            (current_pressure - next_pressure) / 0.20
        ).clamp(0.0, 1.0)
        current_margin = torch.minimum(
            current_context[:, SAFETY_MARGIN_INDEX],
            current_context[:, PREVIEW_MARGIN_INDEX],
        )
        next_margin = torch.minimum(
            next_context[:, SAFETY_MARGIN_INDEX],
            next_context[:, PREVIEW_MARGIN_INDEX],
        )
        margin_gain = (
            (next_margin - current_margin) / 0.04
        ).clamp(0.0, 1.0)
        tilt_gain = (
            0.60
            * (
                current_context[:, BASE_TILT_INDEX]
                - next_context[:, BASE_TILT_INDEX]
            )
            / 0.06
        ).clamp(0.0, 1.0)
        support_gain = (
            4.0
            * (
                next_context[:, SUPPORT_COUNT_INDEX]
                - current_context[:, SUPPORT_COUNT_INDEX]
            )
        ).clamp(0.0, 1.0)
        current_motion = (
            current_context[:, BASE_VX_INDEX].abs() / 0.35
            + current_context[:, BASE_WZ_INDEX].abs() / 0.55
        )
        next_motion = (
            next_context[:, BASE_VX_INDEX].abs() / 0.35
            + next_context[:, BASE_WZ_INDEX].abs() / 0.55
        )
        motion_damping = (
            (current_motion - next_motion) / 0.25
        ).clamp(0.0, 1.0)
        improvement = torch.maximum(
            pressure_drop,
            torch.maximum(
                margin_gain,
                torch.maximum(
                    tilt_gain,
                    torch.maximum(support_gain, motion_damping),
                ),
            ),
        )
        non_worsening = (
            next_pressure <= current_pressure + 0.02
        ).to(current_pressure.dtype)
        retained_viability = torch.maximum(
            improvement, 0.08 * non_worsening
        )
        pressure_gate = (
            (current_pressure - 0.45) / 0.35
        ).clamp(0.0, 1.0)
        return (
            payload_retention
            * pressure_gate
            * retained_viability
        ).clamp(0.0, 1.0)

    @classmethod
    def _hindsight_event_actions(
        cls,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Relabel an event with the task slot and phase that occurred."""

        event_score, interaction_targets = cls._interaction_event_targets(
            current_context, next_context
        )
        priority, task_target = event_score.max(dim=1)
        interaction_target = interaction_targets.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1)
        stage_priority = priority.new_tensor((0.50, 1.00, 1.50))
        priority = priority * stage_priority[interaction_target]
        nonterminal = 1.0 - dones.float()
        priority = priority * nonterminal
        recovery_priority = (
            1.50
            * cls._recovery_event_credit(
                current_context, next_context
            )
            * nonterminal
        )
        recovery_event = recovery_priority > priority
        priority = torch.maximum(priority, recovery_priority)

        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        carrying_by_task = current_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        carrying_mass = carrying_by_task.max(dim=1).values
        payload_recovery = recovery_event & (carrying_mass > 0.5)
        task_target = torch.where(
            recovery_event,
            torch.full_like(task_target, 4),
            task_target,
        )

        slices = ACTION_LAYOUT.slices()
        original_skill = torch.argmax(
            actions[:, slices["skill"]], dim=1
        )
        original_interaction = original_skill.remainder(
            INTERACTION_SKILL_COUNT
        )
        interaction_target = torch.where(
            payload_recovery,
            torch.ones_like(interaction_target),
            torch.where(
                recovery_event,
                original_interaction,
                interaction_target,
            ),
        )

        target_slot = current_slots[
            torch.arange(
                current_slots.shape[0], device=current_context.device
            ),
            task_target,
        ]
        task_valid = (
            target_slot[:, TASK_SLOT_REQUIRED_INDEX]
            * target_slot[:, TASK_SLOT_AVAILABLE_INDEX]
            * (1.0 - target_slot[:, TASK_SLOT_COMPLETED_INDEX])
            * nonterminal
        ).clamp(0.0, 1.0)
        task_valid = torch.where(
            recovery_event, nonterminal, task_valid
        )
        # Non-target interactions remain useful for cross-task skill
        # transfer, but they must not dominate the sparse target-aligned
        # events that define the mission.
        priority = priority * (0.10 + 0.90 * task_valid)

        replay_actions = actions.detach().clone()
        replay_actions[:, slices["task"]].zero_()
        replay_actions[:, slices["task"]].scatter_(
            1, task_target.unsqueeze(1), 1.0
        )
        original_object = torch.argmax(
            actions[:, slices["object"]], dim=1
        )
        delivery_task = (task_target >= 5) & (task_target <= 10)
        object_target = torch.where(
            delivery_task,
            (task_target - 5).clamp(0, ACTION_LAYOUT.object_dim - 1),
            original_object,
        )
        replay_actions[:, slices["object"]].zero_()
        replay_actions[:, slices["object"]].scatter_(
            1, object_target.unsqueeze(1), 1.0
        )

        motion_target = torch.div(
            original_skill,
            INTERACTION_SKILL_COUNT,
            rounding_mode="floor",
        )
        skill_target = (
            motion_target * INTERACTION_SKILL_COUNT
            + interaction_target
        )
        replay_actions[:, slices["skill"]].zero_()
        replay_actions[:, slices["skill"]].scatter_(
            1, skill_target.unsqueeze(1), 1.0
        )
        return (
            replay_actions,
            priority,
            task_valid,
            task_target,
            recovery_event,
        )

    @staticmethod
    def _delivery_task_mask(task_id: torch.Tensor) -> torch.Tensor:
        return (task_id >= 5) & (task_id <= 10)

    def _role_balanced_unique_indices(
        self,
        candidate_indices: torch.Tensor,
        task_id: torch.Tensor,
        recovery_valid: torch.Tensor,
        priority: torch.Tensor,
        sample_count: int,
        *,
        stochastic: bool,
    ) -> torch.Tensor:
        """Keep delivery evidence from being displaced by dense route progress."""

        sample_count = min(int(sample_count), candidate_indices.numel())
        if sample_count <= 0:
            return candidate_indices.new_empty((0,))
        selected: list[torch.Tensor] = []
        selected_mask = torch.zeros(
            priority.shape[0],
            dtype=torch.bool,
            device=priority.device,
        )
        delivery_quota = int(
            round(sample_count * self.event_replay_delivery_fraction)
        )
        recovery_quota = min(
            sample_count - delivery_quota,
            int(
                round(
                    sample_count * self.event_replay_recovery_fraction
                )
            ),
        )
        general_quota = sample_count - delivery_quota - recovery_quota
        recovery_mask = recovery_valid.bool()
        delivery_mask = self._delivery_task_mask(task_id) & ~recovery_mask
        role_quotas = (
            (~delivery_mask & ~recovery_mask, general_quota),
            (recovery_mask, recovery_quota),
            (delivery_mask, delivery_quota),
        )
        for role_mask, quota in role_quotas:
            role_indices = candidate_indices[role_mask[candidate_indices]]
            count = min(quota, role_indices.numel())
            if count <= 0:
                continue
            role_priority = priority[role_indices].clamp_min(1.0e-6)
            if stochastic:
                local_indices = torch.multinomial(
                    role_priority, count, replacement=False
                )
            else:
                local_indices = torch.topk(
                    role_priority, k=count, sorted=False
                ).indices
            role_sample = role_indices[local_indices]
            selected.append(role_sample)
            selected_mask[role_sample] = True

        selected_count = sum(chunk.numel() for chunk in selected)
        remaining_count = sample_count - selected_count
        if remaining_count > 0:
            remaining_indices = candidate_indices[
                ~selected_mask[candidate_indices]
            ]
            remaining_priority = priority[
                remaining_indices
            ].clamp_min(1.0e-6)
            if stochastic:
                local_indices = torch.multinomial(
                    remaining_priority,
                    remaining_count,
                    replacement=False,
                )
            else:
                local_indices = torch.topk(
                    remaining_priority,
                    k=remaining_count,
                    sorted=False,
                ).indices
            selected.append(remaining_indices[local_indices])
        return torch.cat(selected, dim=0)

    @torch.no_grad()
    def _append_event_replay(
        self,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        (
            relabeled_actions,
            priority,
            task_valid,
            hindsight_task,
            recovery_event,
        ) = self._hindsight_event_actions(
            current_context,
            next_context,
            actions,
            dones,
        )
        event_indices = torch.nonzero(
            priority > self.event_replay_min_score,
            as_tuple=False,
        ).squeeze(1)
        if event_indices.numel() == 0:
            return {
                "added": 0.0,
                "task_valid_fraction": 0.0,
                "task_relabel_fraction": 0.0,
                "interaction_relabel_fraction": 0.0,
                "delivery_fraction": 0.0,
                "recovery_fraction": 0.0,
                "payload_recovery_fraction": 0.0,
            }
        if event_indices.numel() > self.event_replay_max_add:
            event_indices = self._role_balanced_unique_indices(
                event_indices,
                hindsight_task,
                recovery_event,
                priority,
                self.event_replay_max_add,
                stochastic=False,
            )

        context = current_context[event_indices].detach().clone()
        replay_actions = relabeled_actions[event_indices].detach().clone()
        replay_priority = priority[event_indices].detach().clone()
        replay_task_valid = task_valid[event_indices].detach().clone()
        replay_recovery_valid = (
            recovery_event[event_indices].detach().clone()
        )
        slices = ACTION_LAYOUT.slices()
        original_task = torch.argmax(
            actions[event_indices, slices["task"]], dim=1
        )
        replay_task_detail_valid = (
            replay_task_valid
            * (hindsight_task[event_indices] == original_task).float()
        )
        original_skill = torch.argmax(
            actions[event_indices, slices["skill"]], dim=1
        )
        replay_skill = torch.argmax(
            replay_actions[:, slices["skill"]], dim=1
        )
        task_relabel_fraction = (
            hindsight_task[event_indices] != original_task
        ).float().mean()
        interaction_relabel_fraction = (
            replay_skill.remainder(INTERACTION_SKILL_COUNT)
            != original_skill.remainder(INTERACTION_SKILL_COUNT)
        ).float().mean()
        relabeled_delivery = self._delivery_task_mask(
            hindsight_task[event_indices]
        )
        delivery_fraction = (
            relabeled_delivery & ~replay_recovery_valid
        ).float().mean()
        recovery_fraction = replay_recovery_valid.float().mean()
        replay_slots = context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        replay_payload_active = (
            replay_slots[:, :, TASK_SLOT_CARRYING_INDEX]
            .amax(dim=1)
            .clamp(0.0, 1.0)
            > 0.5
        )
        payload_recovery_fraction = (
            replay_payload_active & replay_recovery_valid
        ).float().mean()
        if self._event_replay_context is None:
            self._event_replay_context = context
            self._event_replay_actions = replay_actions
            self._event_replay_priority = replay_priority
            self._event_replay_task_valid = replay_task_valid
            self._event_replay_task_detail_valid = (
                replay_task_detail_valid
            )
            self._event_replay_recovery_valid = replay_recovery_valid
        else:
            self._event_replay_context = torch.cat(
                (self._event_replay_context, context), dim=0
            )
            self._event_replay_actions = torch.cat(
                (self._event_replay_actions, replay_actions), dim=0
            )
            self._event_replay_priority = torch.cat(
                (self._event_replay_priority, replay_priority), dim=0
            )
            self._event_replay_task_valid = torch.cat(
                (self._event_replay_task_valid, replay_task_valid), dim=0
            )
            self._event_replay_task_detail_valid = torch.cat(
                (
                    self._event_replay_task_detail_valid,
                    replay_task_detail_valid,
                ),
                dim=0,
            )
            self._event_replay_recovery_valid = torch.cat(
                (
                    self._event_replay_recovery_valid,
                    replay_recovery_valid,
                ),
                dim=0,
            )

        size = self._event_replay_priority.numel()
        if size > self.event_replay_capacity:
            buffer_task_id = torch.argmax(
                self._event_replay_actions[:, slices["task"]], dim=1
            )
            keep = self._role_balanced_unique_indices(
                torch.arange(
                    size, device=self._event_replay_priority.device
                ),
                buffer_task_id,
                self._event_replay_recovery_valid,
                self._event_replay_priority,
                self.event_replay_capacity,
                stochastic=True,
            )
            self._event_replay_context = self._event_replay_context[keep]
            self._event_replay_actions = self._event_replay_actions[keep]
            self._event_replay_priority = self._event_replay_priority[keep]
            self._event_replay_task_valid = (
                self._event_replay_task_valid[keep]
            )
            self._event_replay_task_detail_valid = (
                self._event_replay_task_detail_valid[keep]
            )
            self._event_replay_recovery_valid = (
                self._event_replay_recovery_valid[keep]
            )
        return {
            "added": float(event_indices.numel()),
            "task_valid_fraction": float(
                replay_task_valid.mean().item()
            ),
            "task_relabel_fraction": float(
                task_relabel_fraction.item()
            ),
            "interaction_relabel_fraction": float(
                interaction_relabel_fraction.item()
            ),
            "delivery_fraction": float(delivery_fraction.item()),
            "recovery_fraction": float(recovery_fraction.item()),
            "payload_recovery_fraction": float(
                payload_recovery_fraction.item()
            ),
        }

    def _payload_balanced_unique_indices(
        self,
        candidate_indices: torch.Tensor,
        drop: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        """Select unique payload transitions while retaining rare drops."""

        sample_count = min(int(sample_count), candidate_indices.numel())
        if sample_count <= 0:
            return candidate_indices.new_empty((0,))
        is_drop = drop > 0.5
        drop_indices = candidate_indices[is_drop[candidate_indices]]
        retain_indices = candidate_indices[~is_drop[candidate_indices]]
        drop_quota = min(
            drop_indices.numel(),
            int(
                round(
                    sample_count
                    * self.payload_survival_replay_drop_fraction
                )
            ),
        )
        retain_quota = min(
            retain_indices.numel(), sample_count - drop_quota
        )
        selected: list[torch.Tensor] = []
        selected_mask = torch.zeros(
            drop.shape[0], dtype=torch.bool, device=drop.device
        )
        for indices, quota in (
            (drop_indices, drop_quota),
            (retain_indices, retain_quota),
        ):
            if quota <= 0:
                continue
            sample = indices[
                torch.randperm(
                    indices.numel(), device=indices.device
                )[:quota]
            ]
            selected.append(sample)
            selected_mask[sample] = True
        remaining_count = sample_count - sum(
            sample.numel() for sample in selected
        )
        if remaining_count > 0:
            remaining = candidate_indices[
                ~selected_mask[candidate_indices]
            ]
            fill = remaining[
                torch.randperm(
                    remaining.numel(), device=remaining.device
                )[:remaining_count]
            ]
            selected.append(fill)
        return torch.cat(selected, dim=0)

    @torch.no_grad()
    def _append_payload_survival_replay(
        self,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Store nonterminal payload transitions for balanced calibration."""

        current_payload, survival_target, payload_drop = (
            self._payload_survival_targets(
                current_context, next_context
            )
        )
        support = (current_payload > 0.5) & (dones.float() < 0.5)
        indices = torch.nonzero(support, as_tuple=False).squeeze(1)
        if indices.numel() > self.payload_survival_replay_max_add:
            indices = self._payload_balanced_unique_indices(
                indices,
                payload_drop,
                self.payload_survival_replay_max_add,
            )
        if indices.numel() == 0:
            size = (
                0
                if self._payload_replay_target is None
                else self._payload_replay_target.numel()
            )
            return {
                "added": 0.0,
                "drop_fraction": 0.0,
                "size": float(size),
            }

        context = current_context[indices].detach().clone()
        replay_actions = actions[indices].detach().clone()
        target = survival_target[indices].detach().clone()
        drop = payload_drop[indices].detach().clone()
        if self._payload_replay_context is None:
            self._payload_replay_context = context
            self._payload_replay_actions = replay_actions
            self._payload_replay_target = target
            self._payload_replay_drop = drop
        else:
            self._payload_replay_context = torch.cat(
                (self._payload_replay_context, context), dim=0
            )
            self._payload_replay_actions = torch.cat(
                (self._payload_replay_actions, replay_actions), dim=0
            )
            self._payload_replay_target = torch.cat(
                (self._payload_replay_target, target), dim=0
            )
            self._payload_replay_drop = torch.cat(
                (self._payload_replay_drop, drop), dim=0
            )

        size = self._payload_replay_target.numel()
        if size > self.payload_survival_replay_capacity:
            keep = self._payload_balanced_unique_indices(
                torch.arange(
                    size, device=self._payload_replay_target.device
                ),
                self._payload_replay_drop,
                self.payload_survival_replay_capacity,
            )
            self._payload_replay_context = (
                self._payload_replay_context[keep]
            )
            self._payload_replay_actions = (
                self._payload_replay_actions[keep]
            )
            self._payload_replay_target = self._payload_replay_target[
                keep
            ]
            self._payload_replay_drop = self._payload_replay_drop[keep]
            size = self._payload_replay_target.numel()
        return {
            "added": float(indices.numel()),
            "drop_fraction": float(drop.mean().item()),
            "size": float(size),
        }

    def _sample_payload_survival_indices(
        self, sample_count: int
    ) -> torch.Tensor:
        """Draw a class-balanced payload batch with replacement."""

        if self._payload_replay_drop is None:
            raise RuntimeError("Payload survival replay is empty")
        is_drop = self._payload_replay_drop > 0.5
        drop_indices = torch.nonzero(
            is_drop, as_tuple=False
        ).squeeze(1)
        retain_indices = torch.nonzero(
            ~is_drop, as_tuple=False
        ).squeeze(1)
        if drop_indices.numel() == 0:
            drop_count = 0
        elif retain_indices.numel() == 0:
            drop_count = sample_count
        else:
            drop_count = int(
                round(
                    sample_count
                    * self.payload_survival_replay_drop_fraction
                )
            )
        retain_count = sample_count - drop_count
        selected: list[torch.Tensor] = []
        for indices, count in (
            (drop_indices, drop_count),
            (retain_indices, retain_count),
        ):
            if count <= 0:
                continue
            selected.append(
                indices[
                    torch.randint(
                        indices.numel(),
                        (count,),
                        device=indices.device,
                    )
                ]
            )
        sample = torch.cat(selected, dim=0)
        return sample[
            torch.randperm(sample.numel(), device=sample.device)
        ]

    def _payload_survival_replay_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calibrate skill survival from balanced physical-relation events."""

        zero = next(self.policy.parameters()).new_zeros(())
        if (
            self._payload_replay_context is None
            or self._payload_replay_actions is None
            or self._payload_replay_target is None
            or self._payload_replay_drop is None
        ):
            return zero, {
                "payload_survival_replay_size": zero,
                "payload_survival_replay_support": zero,
            }
        replay_size = self._payload_replay_target.numel()
        sample_count = min(
            self.payload_survival_replay_batch_size, replay_size
        )
        indices = self._sample_payload_survival_indices(sample_count)
        context = self._payload_replay_context[indices]
        actions = self._payload_replay_actions[indices]
        target = self._payload_replay_target[indices]
        drop = self._payload_replay_drop[indices]
        self.policy.predict_option_outcomes(context, actions)
        survival = self.policy.actor.last_skill_survival
        if survival is None:
            raise RuntimeError(
                "TACTIC actor did not expose replay payload survival"
            )
        slices = ACTION_LAYOUT.slices()
        skill_id = torch.argmax(actions[:, slices["skill"]], dim=1)
        prediction = survival.gather(
            1, skill_id.unsqueeze(1)
        ).squeeze(1)
        loss = F.binary_cross_entropy(
            prediction.clamp(1.0e-5, 1.0 - 1.0e-5),
            target,
        )
        drop_mask = drop > 0.5
        retain_mask = ~drop_mask

        def masked_mean(
            value: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            weight = mask.to(value.dtype)
            return (value * weight).sum() / weight.sum().clamp_min(1.0)

        accuracy = (
            (prediction >= 0.5) == (target >= 0.5)
        ).float().mean()
        task_id = torch.argmax(actions[:, slices["task"]], dim=1)
        interaction_id = skill_id % INTERACTION_SKILL_COUNT
        pair_gaps: list[torch.Tensor] = []
        for task in torch.unique(task_id):
            for interaction in range(INTERACTION_SKILL_COUNT):
                group = (
                    (task_id == task)
                    & (interaction_id == interaction)
                )
                group_drop = torch.nonzero(
                    group & drop_mask, as_tuple=False
                ).squeeze(1)
                group_retain = torch.nonzero(
                    group & retain_mask, as_tuple=False
                ).squeeze(1)
                pair_count = min(
                    group_drop.numel(), group_retain.numel()
                )
                if pair_count == 0:
                    continue
                group_drop = group_drop[
                    torch.randperm(
                        group_drop.numel(), device=group_drop.device
                    )[:pair_count]
                ]
                group_retain = group_retain[
                    torch.randperm(
                        group_retain.numel(), device=group_retain.device
                    )[:pair_count]
                ]
                pair_gaps.append(
                    prediction[group_retain] - prediction[group_drop]
                )
        if pair_gaps:
            rank_gap = torch.cat(pair_gaps)
            rank_loss = F.softplus(
                self.payload_survival_rank_margin - rank_gap
            ).mean()
            rank_accuracy = (rank_gap > 0.0).float().mean()
            loss = loss + self.payload_survival_rank_coef * rank_loss
        else:
            rank_gap = prediction.new_zeros((0,))
            rank_loss = prediction.new_zeros(())
            rank_accuracy = prediction.new_zeros(())
        return loss, {
            "payload_survival_replay_size": target.new_tensor(
                float(replay_size)
            ),
            "payload_survival_replay_support": target.new_ones(()),
            "payload_survival_replay_drop_fraction": drop.mean().detach(),
            "payload_survival_replay_prediction": (
                prediction.mean().detach()
            ),
            "payload_survival_replay_drop_prediction": masked_mean(
                prediction, drop_mask
            ).detach(),
            "payload_survival_replay_retain_prediction": masked_mean(
                prediction, retain_mask
            ).detach(),
            "payload_survival_replay_separation": (
                masked_mean(prediction, retain_mask)
                - masked_mean(prediction, drop_mask)
            ).detach(),
            "payload_survival_replay_accuracy": accuracy.detach(),
            "payload_survival_replay_rank_loss": rank_loss.detach(),
            "payload_survival_replay_rank_gap": (
                rank_gap.mean().detach()
                if rank_gap.numel() > 0
                else prediction.new_zeros(())
            ),
            "payload_survival_replay_rank_accuracy": (
                rank_accuracy.detach()
            ),
            "payload_survival_replay_rank_pairs": target.new_tensor(
                float(rank_gap.numel())
            ),
        }

    def _sample_event_replay_indices(
        self, sample_count: int
    ) -> torch.Tensor:
        """Balance task roles first, then sparse stages inside delivery."""

        if (
            self._event_replay_actions is None
            or self._event_replay_priority is None
            or self._event_replay_recovery_valid is None
        ):
            raise RuntimeError("Event replay is empty")
        slices = ACTION_LAYOUT.slices()
        skill_id = torch.argmax(
            self._event_replay_actions[:, slices["skill"]], dim=1
        )
        task_id = torch.argmax(
            self._event_replay_actions[:, slices["task"]], dim=1
        )
        recovery_mask = self._event_replay_recovery_valid.bool()
        delivery_mask = self._delivery_task_mask(task_id) & ~recovery_mask
        phase_id = skill_id.remainder(INTERACTION_SKILL_COUNT)
        phase_fraction = (
            1.0
            - self.event_replay_secure_fraction
            - self.event_replay_release_fraction,
            self.event_replay_secure_fraction,
            self.event_replay_release_fraction,
        )
        selected: list[torch.Tensor] = []
        selected_unique = torch.zeros(
            phase_id.shape[0],
            dtype=torch.bool,
            device=phase_id.device,
        )
        delivery_budget = int(
            round(sample_count * self.event_replay_delivery_fraction)
        )
        phase_budget = delivery_budget
        for current_phase, fraction in enumerate(phase_fraction):
            phase_indices = torch.nonzero(
                delivery_mask & (phase_id == current_phase), as_tuple=False
            ).squeeze(1)
            if phase_indices.numel() == 0:
                continue
            quota = int(round(float(delivery_budget) * fraction))
            quota = min(
                quota,
                phase_budget,
                int(
                    math.ceil(
                        self.event_replay_phase_oversample_cap
                        * phase_indices.numel()
                    )
                ),
            )
            if quota <= 0:
                continue
            phase_priority = self._event_replay_priority[
                phase_indices
            ].clamp_min(1.0e-6)
            phase_sample = phase_indices[
                torch.multinomial(
                    phase_priority,
                    quota,
                    replacement=quota > phase_indices.numel(),
                )
            ]
            selected.append(phase_sample)
            selected_unique[phase_sample] = True
            phase_budget -= quota

        selected_count = sum(chunk.numel() for chunk in selected)
        remaining_delivery = delivery_budget - selected_count
        delivery_indices = torch.nonzero(
            delivery_mask, as_tuple=False
        ).squeeze(1)
        if remaining_delivery > 0 and delivery_indices.numel() > 0:
            delivery_quota = min(
                remaining_delivery,
                int(
                    math.ceil(
                        self.event_replay_role_oversample_cap
                        * delivery_indices.numel()
                    )
                ),
            )
            delivery_priority = self._event_replay_priority[
                delivery_indices
            ].clamp_min(1.0e-6)
            delivery_sample = delivery_indices[
                torch.multinomial(
                    delivery_priority,
                    delivery_quota,
                    replacement=delivery_quota > delivery_indices.numel(),
                )
            ]
            selected.append(delivery_sample)
            selected_unique[delivery_sample] = True

        recovery_budget = min(
            sample_count - delivery_budget,
            int(
                round(
                    sample_count * self.event_replay_recovery_fraction
                )
            ),
        )
        recovery_indices = torch.nonzero(
            recovery_mask, as_tuple=False
        ).squeeze(1)
        if recovery_budget > 0 and recovery_indices.numel() > 0:
            recovery_quota = min(
                recovery_budget,
                int(
                    math.ceil(
                        self.event_replay_role_oversample_cap
                        * recovery_indices.numel()
                    )
                ),
            )
            recovery_priority = self._event_replay_priority[
                recovery_indices
            ].clamp_min(1.0e-6)
            recovery_sample = recovery_indices[
                torch.multinomial(
                    recovery_priority,
                    recovery_quota,
                    replacement=(
                        recovery_quota > recovery_indices.numel()
                    ),
                )
            ]
            selected.append(recovery_sample)
            selected_unique[recovery_sample] = True

        general_budget = (
            sample_count - delivery_budget - recovery_budget
        )
        general_indices = torch.nonzero(
            ~delivery_mask & ~recovery_mask, as_tuple=False
        ).squeeze(1)
        if general_budget > 0 and general_indices.numel() > 0:
            general_quota = min(
                general_budget,
                int(
                    math.ceil(
                        self.event_replay_role_oversample_cap
                        * general_indices.numel()
                    )
                ),
            )
            general_priority = self._event_replay_priority[
                general_indices
            ].clamp_min(1.0e-6)
            general_sample = general_indices[
                torch.multinomial(
                    general_priority,
                    general_quota,
                    replacement=general_quota > general_indices.numel(),
                )
            ]
            selected.append(general_sample)
            selected_unique[general_sample] = True

        selected_count = sum(chunk.numel() for chunk in selected)
        remaining_count = sample_count - selected_count
        if remaining_count > 0:
            remaining_indices = torch.nonzero(
                ~selected_unique, as_tuple=False
            ).squeeze(1)
            remaining_priority = self._event_replay_priority[
                remaining_indices
            ].clamp_min(1.0e-6)
            fill = remaining_indices[
                torch.multinomial(
                    remaining_priority,
                    remaining_count,
                    replacement=False,
                )
            ]
            selected.append(fill)
        sample_indices = torch.cat(selected, dim=0)
        order = torch.randperm(
            sample_indices.numel(), device=sample_indices.device
        )
        return sample_indices[order]

    def _event_replay_loss(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        zero = next(self.policy.parameters()).new_zeros(())
        if (
            self._event_replay_context is None
            or self._event_replay_actions is None
            or self._event_replay_priority is None
            or self._event_replay_task_valid is None
            or self._event_replay_task_detail_valid is None
            or self._event_replay_recovery_valid is None
        ):
            return zero, zero, zero, {
                "event_replay_size": zero,
                "event_replay_priority": zero,
            }

        replay_size = self._event_replay_priority.numel()
        sample_count = min(self.event_replay_batch_size, replay_size)
        sample_indices = self._sample_event_replay_indices(
            sample_count
        )
        context = self._event_replay_context[sample_indices]
        actions = self._event_replay_actions[sample_indices]
        priority = self._event_replay_priority[sample_indices]
        task_valid = self._event_replay_task_valid[sample_indices]
        task_detail_valid = self._event_replay_task_detail_valid[
            sample_indices
        ]
        sample_recovery_mask = self._event_replay_recovery_valid[
            sample_indices
        ].bool()
        (
            task_probability,
            object_probability,
            skill_probability,
            task_subgoal_prediction,
            skill_parameter_prediction,
        ) = self.policy.uncommitted_option_replay_outputs(
            context, actions
        )
        slices = ACTION_LAYOUT.slices()
        buffer_skill_id = torch.argmax(
            self._event_replay_actions[:, slices["skill"]], dim=1
        )
        buffer_interaction_id = buffer_skill_id.remainder(
            INTERACTION_SKILL_COUNT
        )
        buffer_task_id = torch.argmax(
            self._event_replay_actions[:, slices["task"]], dim=1
        )
        buffer_recovery_mask = self._event_replay_recovery_valid.bool()
        buffer_delivery_mask = (
            self._delivery_task_mask(buffer_task_id)
            & ~buffer_recovery_mask
        )
        task_id = torch.argmax(actions[:, slices["task"]], dim=1)
        sample_delivery_mask = (
            self._delivery_task_mask(task_id)
            & ~sample_recovery_mask
        )
        object_id = torch.argmax(actions[:, slices["object"]], dim=1)
        skill_id = torch.argmax(actions[:, slices["skill"]], dim=1)
        interaction_id = skill_id % INTERACTION_SKILL_COUNT
        task_nll = -torch.log(
            task_probability.gather(
                1, task_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        object_nll = -torch.log(
            object_probability.gather(
                1, object_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        interaction_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=1)
        motion_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=2)
        motion_id = torch.div(
            skill_id,
            INTERACTION_SKILL_COUNT,
            rounding_mode="floor",
        )
        motion_nll = -torch.log(
            motion_probability.gather(
                1, motion_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        interaction_nll = -torch.log(
            interaction_probability.gather(
                1, interaction_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        skill_nll = (
            self.event_replay_motion_weight * motion_nll
            + self.event_replay_interaction_weight * interaction_nll
        )
        weight = torch.ones_like(priority)
        role_stage_masks = [~sample_delivery_mask]
        role_stage_masks.extend(
            sample_delivery_mask & (interaction_id == phase_id)
            for phase_id in range(INTERACTION_SKILL_COUNT)
        )
        for role_stage_mask in role_stage_masks:
            if torch.any(role_stage_mask):
                role_stage_priority = priority[role_stage_mask]
                weight[role_stage_mask] = (
                    role_stage_priority
                    / role_stage_priority.mean().clamp_min(1.0e-6)
                ).clamp(0.25, 4.0)
        if self.training_stage == "recovery":
            weight = weight * sample_recovery_mask.float()
        weight_sum = weight.sum().clamp_min(1.0)
        task_weight = weight * task_valid
        task_loss = (
            (task_nll + object_nll) * task_weight
        ).sum() / task_weight.sum().clamp_min(1.0)
        skill_loss = (skill_nll * weight).sum() / weight_sum
        task_subgoal_mask = task_subgoal_prediction.new_tensor(
            (1.0, 1.0, 0.75, 0.50, 0.50, 0.50, 0.35, 0.35)
        )
        task_subgoal_error = F.smooth_l1_loss(
            task_subgoal_prediction,
            actions[:, slices["task_subgoal"]],
            reduction="none",
        )
        task_subgoal_error = (
            task_subgoal_error * task_subgoal_mask
        ).sum(dim=1) / task_subgoal_mask.sum().clamp_min(1.0)
        task_detail_weight = weight * task_detail_valid
        task_subgoal_loss = (
            task_subgoal_error * task_detail_weight
        ).sum() / task_detail_weight.sum().clamp_min(1.0)
        parameter_mask = skill_parameter_prediction.new_tensor(
            (0.0, 0.0, 1.0, 0.35, 0.35, 0.35)
        )
        parameter_error = F.smooth_l1_loss(
            skill_parameter_prediction,
            actions[:, slices["skill_param"]],
            reduction="none",
        )
        parameter_error = (
            parameter_error * parameter_mask
        ).sum(dim=1) / parameter_mask.sum().clamp_min(1.0)
        skill_parameter_loss = (
            parameter_error * weight
        ).sum() / weight_sum
        parameter_loss = (
            skill_parameter_loss
            + self.event_replay_task_subgoal_weight * task_subgoal_loss
        )
        sampled_barrier_quality = context[
            :, CONTROL_TARGET_SLICE
        ][:, :2].amin(dim=1)
        recovery_sample_weight = sample_recovery_mask.float()
        recovery_sample_count = recovery_sample_weight.sum().clamp_min(1.0)
        recovery_pressure = (
            self.policy.actor.control_recovery_pressure(context)
        )
        recovery_adapter_gate = getattr(
            self.policy.actor,
            "last_recovery_adapter_gate",
            None,
        )
        if (
            not torch.is_tensor(recovery_adapter_gate)
            or recovery_adapter_gate.shape != recovery_pressure.shape
        ):
            recovery_adapter_gate = torch.zeros_like(recovery_pressure)
        buffer_barrier_quality = self._event_replay_context[
            :, CONTROL_TARGET_SLICE
        ][:, :2].amin(dim=1)
        stats = {
            "event_replay_size": priority.new_tensor(float(replay_size)),
            "event_replay_priority": priority.mean().detach(),
            "event_replay_task_valid": task_valid.mean().detach(),
            "event_replay_task_detail_valid": (
                task_detail_valid.mean().detach()
            ),
            "event_replay_task": task_loss.detach(),
            "event_replay_skill": skill_loss.detach(),
            "event_replay_motion": (
                motion_nll * weight
            ).sum().detach() / weight_sum,
            "event_replay_interaction": (
                interaction_nll * weight
            ).sum().detach() / weight_sum,
            "event_replay_task_subgoal": task_subgoal_loss.detach(),
            "event_replay_skill_parameter": (
                skill_parameter_loss.detach()
            ),
            "event_replay_parameter": parameter_loss.detach(),
            "event_replay_barrier_quality": (
                sampled_barrier_quality * weight
            ).sum().detach()
            / weight_sum,
            "event_replay_unsafe_fraction": (
                (sampled_barrier_quality < 0.12).float() * weight
            ).sum().detach()
            / weight_sum,
            "event_replay_buffer_unsafe_fraction": (
                buffer_barrier_quality < 0.12
            ).float().mean().detach(),
            "event_replay_buffer_delivery": (
                buffer_delivery_mask.float().mean().detach()
            ),
            "event_replay_buffer_recovery": (
                buffer_recovery_mask.float().mean().detach()
            ),
            "event_replay_sample_delivery": (
                sample_delivery_mask.float() * weight
            ).sum().detach()
            / weight_sum,
            "event_replay_sample_recovery": (
                sample_recovery_mask.float() * weight
            ).sum().detach()
            / weight_sum,
            "event_replay_sample_raw_delivery": (
                sample_delivery_mask.float().mean().detach()
            ),
            "event_replay_sample_raw_recovery": (
                sample_recovery_mask.float().mean().detach()
            ),
            "event_replay_recovery_pressure": (
                recovery_pressure * recovery_sample_weight
            ).sum().detach()
            / recovery_sample_count,
            "event_replay_recovery_adapter_gate": (
                recovery_adapter_gate * recovery_sample_weight
            ).sum().detach()
            / recovery_sample_count,
        }
        buffer_delivery_count = buffer_delivery_mask.float().sum().clamp_min(
            1.0
        )
        sample_delivery_weight = (
            sample_delivery_mask.float() * weight
        ).sum().clamp_min(1.0)
        sample_delivery_count = sample_delivery_mask.float().sum().clamp_min(
            1.0
        )
        for phase_id, phase_name in enumerate(
            ("approach", "secure", "release")
        ):
            stats[f"event_replay_buffer_{phase_name}"] = (
                (
                    buffer_delivery_mask
                    & (buffer_interaction_id == phase_id)
                ).float().sum().detach()
                / buffer_delivery_count
            )
            stats[f"event_replay_sample_{phase_name}"] = (
                (
                    (
                        sample_delivery_mask
                        & (interaction_id == phase_id)
                    ).float()
                    * weight
                ).sum()
                / sample_delivery_weight
            ).detach()
            stats[f"event_replay_sample_raw_{phase_name}"] = (
                (
                    sample_delivery_mask
                    & (interaction_id == phase_id)
                ).float().sum().detach()
                / sample_delivery_count
            )
        return task_loss, skill_loss, parameter_loss, stats

    @torch.no_grad()
    def _update_hierarchy_constraint_duals(
        self,
        cbf_violation: float,
        clf_violation: float,
    ) -> None:
        """Project dual ascent on measured hierarchy-level violations."""

        actor = self.policy.actor
        device = actor.hierarchy_cbf_dual.device
        dtype = actor.hierarchy_cbf_dual.dtype
        cbf_value = torch.as_tensor(
            cbf_violation, device=device, dtype=dtype
        )
        clf_value = torch.as_tensor(
            clf_violation, device=device, dtype=dtype
        )
        if not torch.isfinite(cbf_value) or not torch.isfinite(clf_value):
            return

        if int(actor.hierarchy_constraint_updates.item()) == 0:
            actor.hierarchy_cbf_violation_ema.copy_(cbf_value)
            actor.hierarchy_clf_violation_ema.copy_(clf_value)
        else:
            decay = self.constraint_dual_ema_decay
            actor.hierarchy_cbf_violation_ema.mul_(decay).add_(
                cbf_value, alpha=1.0 - decay
            )
            actor.hierarchy_clf_violation_ema.mul_(decay).add_(
                clf_value, alpha=1.0 - decay
            )

        actor.hierarchy_cbf_dual.add_(
            self.constraint_dual_learning_rate
            * (cbf_value - self.cbf_violation_budget)
        ).clamp_(0.0, self.constraint_dual_max)
        actor.hierarchy_clf_dual.add_(
            self.constraint_dual_learning_rate
            * (clf_value - self.clf_violation_budget)
        ).clamp_(0.0, self.constraint_dual_max)
        actor.hierarchy_constraint_updates.add_(1)

    def _auxiliary_loss(
        self,
        obs_batch,
        actions_batch,
        current_context,
        next_context,
        dones,
    ):
        prediction, selected_feasibility = (
            self.policy.predict_auxiliary(obs_batch, actions_batch)
        )
        actor = self.policy.actor
        current_action_mean = self.policy.action_mean
        if current_action_mean is None:
            raise RuntimeError("TACTIC policy did not expose its action mean")
        target = next_context[:, CONTROL_TARGET_SLICE].clamp(0.0, 1.0)
        per_sample = F.smooth_l1_loss(
            prediction, target, reduction="none"
        ).mean(dim=1)
        risk_weight = 1.0 + 2.0 * (
            target.min(dim=1).values < 0.25
        ).float()
        nonterminal = 1.0 - dones.float()
        control_loss = (
            risk_weight * per_sample * nonterminal
        ).sum() / nonterminal.sum().clamp_min(1.0)

        robust_target = (
            0.34 * target[:, 0]
            + 0.26 * target[:, 1]
            + 0.22 * target[:, 2]
            + 0.18 * target[:, 3]
        )
        mission_delta = (
            next_context[:, 0] - current_context[:, 0]
        ).clamp(-0.1, 0.1)
        progress_target = (0.5 + 5.0 * mission_delta).clamp(0.0, 1.0)
        feasibility_target = nonterminal * (
            0.80 * robust_target + 0.20 * progress_target
        )
        skill_feasibility_loss = F.binary_cross_entropy(
            selected_feasibility.clamp(1.0e-5, 1.0 - 1.0e-5),
            feasibility_target.detach(),
        )

        task_target = torch.round(
            current_context[:, EXECUTED_TASK_INDEX]
            * float(ACTION_LAYOUT.task_dim - 1)
        ).long().clamp(0, ACTION_LAYOUT.task_dim - 1)
        skill_target = torch.round(
            current_context[:, EXECUTED_SKILL_INDEX]
            * float(ACTION_LAYOUT.skill_dim - 1)
        ).long().clamp(0, ACTION_LAYOUT.skill_dim - 1)
        next_task = torch.round(
            next_context[:, EXECUTED_TASK_INDEX]
            * float(ACTION_LAYOUT.task_dim - 1)
        ).long().clamp(0, ACTION_LAYOUT.task_dim - 1)
        next_skill = torch.round(
            next_context[:, EXECUTED_SKILL_INDEX]
            * float(ACTION_LAYOUT.skill_dim - 1)
        ).long().clamp(0, ACTION_LAYOUT.skill_dim - 1)
        stable_task = (task_target == next_task).float()
        stable_skill = stable_task * (skill_target == next_skill).float()
        task_valid = nonterminal * stable_task
        skill_valid = nonterminal * stable_skill

        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        next_slots = next_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        rows = torch.arange(current_slots.shape[0], device=current_slots.device)
        selected_slot = current_slots[rows, task_target]
        next_selected_slot = next_slots[rows, task_target]
        interaction_active = (
            selected_slot[:, TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + selected_slot[:, TASK_SLOT_DELIVERY_TYPE_INDEX]
        ).clamp(0.0, 1.0)
        (
            current_payload,
            payload_survival_target,
            payload_drop,
        ) = self._payload_survival_targets(
            current_context, next_context
        )
        payload_drop = payload_drop * nonterminal

        # The upper policy remains fully learned.  A CLF inequality constrains
        # its proposed body-frame subgoal without supplying a runtime command.
        subgoal = torch.tanh(
            current_action_mean[:, ACTION_LAYOUT.slices()["task_subgoal"]]
        )
        distance = (
            4.0 * selected_slot[:, TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 4.0)
        heading = (
            math.pi * selected_slot[:, TASK_SLOT_HEADING_INDEX]
        ).clamp(-math.pi, math.pi)
        distance_n = (distance / 2.0).clamp(0.0, 2.0)
        travel_sign = torch.where(
            heading.cos().abs() > 0.05,
            heading.cos().sign(),
            torch.ones_like(heading),
        )
        travel_heading = torch.atan2(
            travel_sign * heading.sin(),
            travel_sign * heading.cos(),
        )
        heading_n = travel_heading / math.pi
        (
            selected_task_motion,
            _,
            _,
            _,
        ) = actor.task_subgoal_motion_components(subgoal)
        projected_subgoal_yaw = (
            selected_task_motion[:, 1] / 0.55
        ).clamp(-1.5, 1.5)
        radial_speed = (
            subgoal[:, 0] * torch.cos(heading)
            + subgoal[:, 1] * torch.sin(heading)
        )
        lyapunov = 0.5 * (
            distance_n.square() + 0.35 * heading_n.square()
        )
        lyapunov_rate = (
            -distance_n * radial_speed
            - 0.35 * heading_n * projected_subgoal_yaw
        )
        descent_violation = torch.relu(
            lyapunov_rate + 0.16 * lyapunov
        ).square()
        active_goal = (distance > 0.12).float()
        control_margin = current_context[
            :, CONTROL_TARGET_SLICE
        ].min(dim=1).values.clamp(0.0, 1.0)
        base_tilt = 0.60 * current_context[:, BASE_TILT_INDEX]
        tilt_capacity = (
            (0.60 - base_tilt) / 0.35
        ).clamp(0.20, 1.0)
        support_capacity = (
            (
                current_context[:, SUPPORT_COUNT_INDEX] - 0.65
            )
            / 0.35
        ).clamp(0.20, 1.0)
        posture_capacity = torch.minimum(
            tilt_capacity, support_capacity
        )
        descent_weight = task_valid * active_goal * control_margin.detach()
        task_descent_loss = (
            descent_violation * descent_weight
        ).sum() / descent_weight.sum().clamp_min(1.0)
        unsafe_demand = torch.maximum(
            ((0.25 - control_margin) / 0.25).clamp(0.0, 1.0),
            1.0 - posture_capacity,
        )
        subgoal_energy = subgoal[:, :3].square().sum(dim=1)
        task_barrier_loss = (
            unsafe_demand.detach() * subgoal_energy * task_valid
        ).sum() / task_valid.sum().clamp_min(1.0)
        task_control_objective_loss = (
            task_descent_loss + 0.35 * task_barrier_loss
        )

        # Decode every task candidate with the same upper-level policy.  The
        # target is expressed only in robot-relative task coordinates, so the
        # supervision transfers across object identities and scene layouts.
        (
            candidate_task_logits,
            candidate_task_subgoal,
            candidate_raw_slots,
            candidate_task_utility,
            candidate_task_confidence,
        ) = actor.candidate_task_details(current_context)
        candidate_valid_task_mask = (
            (candidate_raw_slots[:, :, 11] > 0.5)
            & (candidate_raw_slots[:, :, 12] < 0.5)
            & (candidate_raw_slots[:, :, 13] > 0.5)
        ).float()
        candidate_distance = (
            4.0
            * candidate_raw_slots[:, :, TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 4.0)
        candidate_manipulation = candidate_raw_slots[
            :, :, TASK_SLOT_MANIPULATION_TYPE_INDEX
        ].clamp(0.0, 1.0)
        candidate_delivery = candidate_raw_slots[
            :, :, TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        candidate_interaction = (
            candidate_manipulation + candidate_delivery
        ).clamp(0.0, 1.0)
        candidate_reachability = candidate_raw_slots[
            :, :, TASK_SLOT_REACHABILITY_INDEX
        ].clamp(0.0, 1.0)
        current_motion_request = actor.feasible_motion_request(
            current_context
        )
        current_tracking_burden = (
            0.65
            * (
                current_motion_request[:, 0]
                - current_context[:, BASE_VX_INDEX]
            ).abs()
            / 0.75
            + 0.35
            * (
                current_motion_request[:, 1]
                - current_context[:, BASE_WZ_INDEX]
            ).abs()
            / 1.50
        )
        transient_capacity = torch.exp(
            -2.0 * current_tracking_burden
        ).clamp(0.20, 1.0)
        (
            target_task_subgoal,
            task_subgoal_authority,
        ) = actor.relational_task_subgoal_prior(
            candidate_raw_slots, current_context
        )
        target_task_xy = target_task_subgoal[:, :, 0:2]
        target_task_yaw = target_task_subgoal[:, :, 2]
        (
            target_task_motion,
            _,
            _,
            _,
        ) = actor.task_subgoal_motion_components(target_task_subgoal)
        (
            candidate_task_motion,
            candidate_semantic_yaw,
            candidate_lateral_yaw,
            _,
        ) = actor.task_subgoal_motion_components(
            candidate_task_subgoal
        )
        task_xy_error = F.smooth_l1_loss(
            candidate_task_subgoal[:, :, 0:2],
            target_task_xy.detach(),
            reduction="none",
        ).mean(dim=-1)
        task_yaw_error = F.smooth_l1_loss(
            candidate_task_subgoal[:, :, 2],
            target_task_yaw.detach(),
            reduction="none",
        )
        predicted_progress_demand = torch.sigmoid(
            2.0 * candidate_task_subgoal[:, :, 6]
        )
        predicted_precision_demand = torch.sigmoid(
            2.0 * candidate_task_subgoal[:, :, 7]
        )
        target_progress_demand = torch.sigmoid(
            2.0 * target_task_subgoal[:, :, 6]
        )
        target_precision_demand = torch.sigmoid(
            2.0 * target_task_subgoal[:, :, 7]
        )
        task_semantic_error = 0.5 * (
            F.smooth_l1_loss(
                predicted_progress_demand,
                target_progress_demand.detach(),
                reduction="none",
            )
            + F.smooth_l1_loss(
                predicted_precision_demand,
                target_precision_demand.detach(),
                reduction="none",
            )
        )
        interaction_scope = (
            candidate_manipulation
            + candidate_delivery * candidate_reachability
        ).clamp(0.0, 1.0)
        interaction_authority = (
            interaction_scope
            * (0.25 + 0.75 * control_margin.detach()).unsqueeze(1)
            * (0.20 + 0.80 * transient_capacity.detach()).unsqueeze(1)
            * (0.25 + 0.75 * posture_capacity.detach()).unsqueeze(1)
        ).clamp(0.0, 1.0)
        arm_offset_error = (
            candidate_task_subgoal[:, :, 4:6].square().mean(dim=-1)
            + (1.0 - candidate_delivery)
            * candidate_task_subgoal[:, :, 3].square()
        )
        task_arm_scope_error = (
            arm_offset_error * (1.0 - interaction_authority)
        )
        relational_subgoal_error = (
            0.55 * task_xy_error
            + 0.20 * task_yaw_error
            + 0.15 * task_semantic_error
            + 0.10 * task_arm_scope_error
        )
        candidate_valid_weight = (
            candidate_valid_task_mask * nonterminal.unsqueeze(1)
        )
        relational_subgoal_grounding_loss = (
            relational_subgoal_error * candidate_valid_weight
        ).sum() / candidate_valid_weight.sum().clamp_min(1.0)
        task_motion_projection_error = (
            0.55
            * F.smooth_l1_loss(
                candidate_task_motion[:, :, 0] / 0.35,
                target_task_motion[:, :, 0].detach() / 0.35,
                reduction="none",
            )
            + 0.45
            * F.smooth_l1_loss(
                candidate_task_motion[:, :, 1] / 0.55,
                target_task_motion[:, :, 1].detach() / 0.55,
                reduction="none",
            )
        )
        candidate_transient_burden = (
            0.65
            * (
                candidate_task_motion[:, :, 0]
                - current_context[:, None, BASE_VX_INDEX]
            ).abs()
            / 0.75
            + 0.35
            * (
                candidate_task_motion[:, :, 1]
                - current_context[:, None, BASE_WZ_INDEX]
            ).abs()
            / 1.50
        )
        transient_burden_limit = (
            0.25
            + 0.75
            * torch.minimum(
                transient_capacity, posture_capacity
            ).detach()
        ).unsqueeze(1)
        task_transient_excess = torch.relu(
            candidate_transient_burden - transient_burden_limit
        ).square()
        task_skill_projection_error = (
            task_motion_projection_error
            + 0.15 * task_transient_excess
        )
        task_skill_projection_loss = (
            task_skill_projection_error * candidate_valid_weight
        ).sum() / candidate_valid_weight.sum().clamp_min(1.0)
        alignment_weight = candidate_valid_weight * (
            candidate_distance > 0.12
        ).float()
        task_subgoal_alignment = (
            F.cosine_similarity(
                candidate_task_subgoal[:, :, 0:2],
                target_task_xy,
                dim=-1,
                eps=1.0e-5,
            )
            * alignment_weight
        ).sum() / alignment_weight.sum().clamp_min(1.0)

        # Early task credit comes from task-relative geometry and interaction
        # state. Calibrated outcome residuals enter only as their confidence
        # and data support mature.
        candidate_task_probability = torch.softmax(
            candidate_task_logits, dim=-1
        )
        grounded_task_utility = actor.last_task_grounded_utility
        task_outcome_reliability = actor.last_task_outcome_reliability
        blended_task_utility = actor.last_task_blended_utility
        if (
            grounded_task_utility is None
            or task_outcome_reliability is None
            or blended_task_utility is None
        ):
            raise RuntimeError(
                "TACTIC actor did not expose grounded task credit"
            )
        counterfactual_task_utility = blended_task_utility.detach()
        counterfactual_task_utility = (
            counterfactual_task_utility
            - 1.0e4 * (1.0 - candidate_valid_task_mask)
        )
        counterfactual_task_target = torch.softmax(
            counterfactual_task_utility
            / self.counterfactual_task_temperature,
            dim=-1,
        )
        counterfactual_task_per_sample = -(
            counterfactual_task_target
            * torch.log(candidate_task_probability.clamp_min(1.0e-8))
        ).sum(dim=1)
        counterfactual_task_weight = (
            nonterminal
            * (candidate_valid_task_mask.sum(dim=1) > 1.0).float()
        )
        counterfactual_task_selection_loss = (
            counterfactual_task_per_sample * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)
        counterfactual_task_target_entropy = -(
            counterfactual_task_target
            * torch.log(counterfactual_task_target.clamp_min(1.0e-8))
        ).sum(dim=1)
        counterfactual_task_target_entropy = (
            counterfactual_task_target_entropy
            * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)
        counterfactual_task_target_peak = (
            counterfactual_task_target.max(dim=1).values
            * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)
        masked_grounded_utility = grounded_task_utility.detach().masked_fill(
            candidate_valid_task_mask < 0.5, -1.0e4
        )
        grounded_task_span = (
            masked_grounded_utility.max(dim=1).values
            - grounded_task_utility.detach().masked_fill(
                candidate_valid_task_mask < 0.5, 1.0e4
            ).min(dim=1).values
        )
        grounded_task_span = (
            grounded_task_span * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)
        task_outcome_blend = (
            task_outcome_reliability.detach()
            * candidate_valid_task_mask
        ).sum() / candidate_valid_task_mask.sum().clamp_min(1.0)
        grounded_target = masked_grounded_utility.argmax(dim=1)
        blended_target = counterfactual_task_utility.argmax(dim=1)
        grounded_target_agreement = (
            (grounded_target == blended_target).float()
            * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)
        sampled_task_target_probability = (
            counterfactual_task_target.gather(
                1, task_target.unsqueeze(1)
            ).squeeze(1)
            * counterfactual_task_weight
        ).sum() / counterfactual_task_weight.sum().clamp_min(1.0)

        # The lower policy owns execution authority through its sampled
        # continuous parameters.  The effect model is calibrated separately
        # from measured transitions, so it cannot reduce this loss by claiming
        # that a weak physical action will work.
        feasible_motion_request = actor.feasible_motion_request(current_context)
        execution_motion_request = actor.last_task_motion_request
        if execution_motion_request is None:
            execution_motion_request = feasible_motion_request
        desired_motion = torch.stack(
            (
                (execution_motion_request[:, 0] / 0.75).clamp(
                    -1.0, 1.0
                ),
                (execution_motion_request[:, 1] / 1.50).clamp(
                    -1.0, 1.0
                ),
            ),
            dim=-1,
        )
        command_active = (
            desired_motion.abs().amax(dim=1) > 0.015
        ).float()
        desired_authority = (
            0.90 + 0.15 * control_margin.detach()
        )
        action_slices = ACTION_LAYOUT.slices()
        mean_skill_parameters = current_action_mean[
            :, action_slices["skill_param"]
        ]
        mean_skill_authority = actor.motion_parameter_authority(
            mean_skill_parameters, execution_motion_request
        )
        wheel_gate = actor.last_wheel_skill_gate
        if wheel_gate is None:
            raise RuntimeError("TACTIC actor did not expose wheel authority")
        authority_error = (
            mean_skill_authority - desired_authority
        ).square()
        authority_weight = skill_valid * command_active
        skill_predictive_control_loss = (
            authority_error * authority_weight
        ).sum() / authority_weight.sum().clamp_min(1.0)

        task_outcome, skill_outcome = self.policy.predict_option_outcomes(
            current_context, actions_batch
        )
        skill_survival = actor.last_skill_survival
        if skill_survival is None:
            raise RuntimeError(
                "TACTIC actor did not expose payload survival"
            )
        selected_skill_survival = skill_survival.gather(
            1, skill_target.unsqueeze(1)
        ).squeeze(1)
        payload_survival_weight = current_payload * nonterminal * (
            1.0 + self.payload_drop_weight * payload_drop
        )
        payload_survival_loss = (
            F.binary_cross_entropy(
                selected_skill_survival.clamp(
                    1.0e-5, 1.0 - 1.0e-5
                ),
                payload_survival_target.detach(),
                reduction="none",
            )
            * payload_survival_weight
        ).sum() / payload_survival_weight.sum().clamp_min(1.0)
        payload_robust_target = robust_target * (
            1.0 - payload_drop
        )
        progress_delta = (
            selected_slot[:, TASK_SLOT_REMAINING_PROGRESS_INDEX]
            - next_selected_slot[:, TASK_SLOT_REMAINING_PROGRESS_INDEX]
        ).clamp(-0.08, 0.08)
        progress_target = (0.5 + 6.0 * progress_delta).clamp(0.0, 1.0)
        completion_target = next_selected_slot[
            :, TASK_SLOT_COMPLETED_INDEX
        ].clamp(0.0, 1.0)
        mission_target = (0.5 + 5.0 * mission_delta).clamp(0.0, 1.0)
        task_outcome_target = torch.stack(
            (
                progress_target,
                completion_target,
                mission_target,
                payload_robust_target,
            ),
            dim=-1,
        )
        task_outcome_per_sample = F.smooth_l1_loss(
            task_outcome, task_outcome_target.detach(), reduction="none"
        ).mean(dim=1)
        task_outcome_loss = (
            task_outcome_per_sample * task_valid
        ).sum() / task_valid.sum().clamp_min(1.0)
        task_outcome_confidence = actor.last_task_outcome_confidence
        if task_outcome_confidence is None:
            raise RuntimeError(
                "TACTIC actor did not expose task-outcome confidence"
            )
        task_probability = actions_batch[
            :, ACTION_LAYOUT.slices()["task"]
        ].clamp_min(0.0)
        task_probability = task_probability / task_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        selected_task_confidence = torch.einsum(
            "bs,bs->b",
            task_probability,
            task_outcome_confidence,
        )
        task_calibrated_error = (
            task_outcome.detach() - task_outcome_target.detach()
        ).abs().mean(dim=1)
        task_confidence_target = torch.exp(
            -2.5 * task_calibrated_error
        ).clamp(0.05, 0.95)
        task_confidence_per_sample = F.binary_cross_entropy(
            selected_task_confidence.clamp(1.0e-5, 1.0 - 1.0e-5),
            task_confidence_target,
            reduction="none",
        )
        task_outcome_confidence_loss = (
            task_confidence_per_sample * task_valid
        ).sum() / task_valid.sum().clamp_min(1.0)

        current_motion_request = actor.feasible_motion_request(
            current_context
        )
        next_motion_request = actor.feasible_motion_request(next_context)
        current_tracking_error = (
            current_motion_request[:, 0]
            - current_context[:, BASE_VX_INDEX]
        ).abs() + 0.35 * (
            current_motion_request[:, 1]
            - current_context[:, BASE_WZ_INDEX]
        ).abs()
        next_tracking_error = (
            next_motion_request[:, 0]
            - next_context[:, BASE_VX_INDEX]
        ).abs() + 0.35 * (
            next_motion_request[:, 1]
            - next_context[:, BASE_WZ_INDEX]
        ).abs()
        tracking_target = (
            0.5 + 3.0 * (current_tracking_error - next_tracking_error)
        ).clamp(0.0, 1.0)
        interaction_delta = (
            next_selected_slot[:, TASK_SLOT_INTERACTION_STATE_SLICE]
            - selected_slot[:, TASK_SLOT_INTERACTION_STATE_SLICE]
        ).clamp_min(0.0).amax(dim=1)
        interaction_target = (
            0.5 + 4.0 * interaction_delta
        ).clamp(0.0, 1.0)
        skill_outcome_target = torch.stack(
            (
                progress_target,
                payload_robust_target,
                tracking_target,
                interaction_target,
            ),
            dim=-1,
        )
        skill_outcome_error = F.smooth_l1_loss(
            skill_outcome, skill_outcome_target.detach(), reduction="none"
        )
        skill_outcome_weights = torch.ones_like(skill_outcome_error)
        skill_outcome_weights[:, -1] = interaction_active
        skill_outcome_per_sample = (
            skill_outcome_error * skill_outcome_weights
        ).sum(dim=1) / (3.0 + interaction_active)
        skill_outcome_loss = (
            skill_outcome_per_sample * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)

        skill_effect, skill_effect_confidence = (
            self.policy.predict_skill_effects(
                current_context, actions_batch
            )
        )
        task_log_prob, skill_log_prob, _ = (
            self.policy.option_log_prob_components(actions_batch)
        )
        (
            discrete_task_log_prob,
            _,
            _,
        ) = self.policy.discrete_factor_log_prob_components(actions_batch)
        current_control = current_context[:, CONTROL_TARGET_SLICE].clamp(
            0.0, 1.0
        )
        current_robust = (
            0.34 * current_control[:, 0]
            + 0.26 * current_control[:, 1]
            + 0.22 * current_control[:, 2]
            + 0.18 * current_control[:, 3]
        )
        task_constraint_multiplier = (
            actor.last_task_constraint_multiplier
        )
        skill_constraint_multipliers = (
            actor.last_skill_constraint_multipliers
        )
        if (
            task_constraint_multiplier is None
            or skill_constraint_multipliers is None
        ):
            raise RuntimeError(
                "TACTIC actor did not expose hierarchy constraints"
            )
        control_floor = actor.control_constraint_floor.unsqueeze(0)
        interaction_state = selected_slot[
            :, TASK_SLOT_INTERACTION_STATE_SLICE
        ].clamp(0.0, 1.0)
        interaction_evidence = interaction_state[:, :3].amax(dim=1)
        carrying_evidence = selected_slot[
            :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        interaction_phase = skill_target.remainder(
            INTERACTION_SKILL_COUNT
        )
        secure_phase = (interaction_phase == 1).to(current_control.dtype)
        release_phase = (interaction_phase == 2).to(current_control.dtype)
        transient_slack = interaction_active * (
            SECURE_CBF_TRANSIENT_SLACK
            * secure_phase
            * (0.35 + 0.65 * interaction_evidence)
            + RELEASE_CBF_TRANSIENT_SLACK
            * release_phase
            * (0.50 + 0.50 * carrying_evidence)
        )
        adaptive_control_floor = control_floor.expand(
            current_control.shape[0], -1
        ).clone()
        adaptive_control_floor[:, :2] = torch.maximum(
            adaptive_control_floor[:, :2]
            - transient_slack.unsqueeze(1),
            adaptive_control_floor.new_tensor(
                INTERACTION_HARD_CONTROL_FLOOR
            ),
        )
        current_constraint_demand = (
            (adaptive_control_floor - current_control)
            / adaptive_control_floor.clamp_min(1.0e-4)
        ).clamp(0.0, 1.0)
        next_constraint_demand = (
            (adaptive_control_floor - target)
            / adaptive_control_floor.clamp_min(1.0e-4)
        ).clamp(0.0, 1.0)
        raw_next_constraint_demand = (
            (control_floor - target)
            / control_floor.clamp_min(1.0e-4)
        ).clamp(0.0, 1.0)
        raw_cbf_violation = raw_next_constraint_demand[
            :, :2
        ].amax(dim=1)
        actual_cbf_violation = next_constraint_demand[
            :, :2
        ].amax(dim=1)
        current_cbf_violation = current_constraint_demand[
            :, :2
        ].amax(dim=1)
        actual_clf_violation = next_constraint_demand[:, 2]
        current_clf_violation = current_constraint_demand[:, 2]
        cbf_regression = torch.relu(
            actual_cbf_violation - current_cbf_violation
        )
        clf_regression = torch.relu(
            actual_clf_violation - current_clf_violation
        )
        cbf_pressure = (
            0.70 * actual_cbf_violation + 0.30 * cbf_regression
        )
        clf_pressure = (
            0.70 * actual_clf_violation + 0.30 * clf_regression
        )
        constraint_demand = torch.maximum(
            current_constraint_demand,
            next_constraint_demand,
        )
        predicted_skill_multiplier = (
            (skill_constraint_multipliers - 0.05) / 0.95
        ).clamp(0.0, 1.0)
        predicted_task_multiplier = (
            (task_constraint_multiplier - 0.05) / 0.95
        ).clamp(0.0, 1.0)
        task_constraint_target = torch.einsum(
            "bo,o->b",
            constraint_demand,
            actor.task_outcome_weights,
        )
        skill_constraint_error = F.smooth_l1_loss(
            predicted_skill_multiplier,
            constraint_demand.detach(),
            reduction="none",
        ).mean(dim=1)
        task_constraint_error = F.smooth_l1_loss(
            predicted_task_multiplier,
            task_constraint_target.detach(),
            reduction="none",
        )
        constraint_multiplier_loss = (
            (
                0.65 * skill_constraint_error
                + 0.35 * task_constraint_error
            )
            * nonterminal
        ).sum() / nonterminal.sum().clamp_min(1.0)
        robust_delta = (
            (robust_target - current_robust) / 0.25
        ).clamp(-1.0, 1.0)
        completion_delta = (
            next_selected_slot[:, TASK_SLOT_COMPLETED_INDEX]
            - selected_slot[:, TASK_SLOT_COMPLETED_INDEX]
        ).clamp(0.0, 1.0)
        normalized_progress = (progress_delta / 0.08).clamp(-1.0, 1.0)
        normalized_mission = (mission_delta / 0.10).clamp(-1.0, 1.0)
        normalized_tracking = (
            (current_tracking_error - next_tracking_error) / 0.40
        ).clamp(-1.0, 1.0)
        normalized_interaction = (
            interaction_delta / 0.25
        ).clamp(0.0, 1.0)
        safety_discount = (
            (1.0 - 0.80 * actual_cbf_violation).clamp(0.0, 1.0)
            * (1.0 - 0.35 * actual_clf_violation).clamp(0.0, 1.0)
        )
        task_achievement = (
            0.38 * normalized_progress
            + 0.18 * completion_delta
            + 0.12 * normalized_mission
            + 0.32
            * interaction_active
            * normalized_interaction
        )
        skill_achievement = (
            0.28 * normalized_progress
            + 0.22 * normalized_tracking
            + 0.40
            * interaction_active
            * normalized_interaction
        )
        safe_task_achievement = (
            task_achievement.clamp_max(0.0)
            + task_achievement.clamp_min(0.0) * safety_discount
        )
        safe_skill_achievement = (
            skill_achievement.clamp_max(0.0)
            + skill_achievement.clamp_min(0.0) * safety_discount
        )
        cbf_dual = actor.hierarchy_cbf_dual.detach()
        clf_dual = actor.hierarchy_clf_dual.detach()
        task_credit_signal = (
            safe_task_achievement
            + 0.15 * robust_delta
            - 0.35 * cbf_dual * cbf_pressure
            - 0.20 * clf_dual * clf_pressure
            - self.payload_drop_task_credit_penalty * payload_drop
        )
        skill_credit_signal = (
            safe_skill_achievement
            + 0.15 * robust_delta
            - 0.75 * cbf_dual * cbf_pressure
            - 0.45 * clf_dual * clf_pressure
            - self.payload_drop_skill_credit_penalty * payload_drop
        )
        task_advantage = self._centered_option_advantage(
            task_credit_signal, task_valid
        )
        skill_advantage = self._centered_option_advantage(
            skill_credit_signal, skill_valid
        )
        task_option_credit_loss = -(
            task_log_prob
            / float(ACTION_LAYOUT.task_subgoal_dim + 2)
            * task_advantage
            * task_valid
        ).sum() / task_valid.sum().clamp_min(1.0)
        skill_option_credit_loss = -(
            skill_log_prob
            / float(ACTION_LAYOUT.skill_param_dim + 1)
            * skill_advantage
            * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)
        event_scores, interaction_targets = (
            self._interaction_event_targets(
                current_context, next_context
            )
        )
        event_credit = event_scores.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1).detach()
        hindsight_interaction_target = interaction_targets.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1)
        skill_distribution = self.policy._skill_distribution
        if skill_distribution is None:
            raise RuntimeError(
                "TACTIC skill distribution was not evaluated"
            )
        interaction_probability = skill_distribution.probs.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=1)
        hindsight_interaction_log_prob = torch.log(
            interaction_probability.gather(
                1, hindsight_interaction_target.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        task_event_weight = event_credit * task_valid
        skill_event_weight = event_credit * skill_valid
        task_event_support = (
            (event_credit > self.event_replay_min_score).float()
            * task_valid
        ).mean()
        skill_event_support = (
            (event_credit > self.event_replay_min_score).float()
            * skill_valid
        ).mean()
        task_support_scale = (
            20.0 * task_event_support
        ).clamp(0.0, 1.0).detach()
        skill_support_scale = (
            20.0 * skill_event_support
        ).clamp(0.0, 1.0).detach()
        event_hindsight_task_loss = -(
            discrete_task_log_prob
            / 2.0
            * task_event_weight
        ).sum() / task_event_weight.sum().clamp_min(1.0)
        event_hindsight_task_loss = (
            event_hindsight_task_loss * task_support_scale
        )
        event_hindsight_skill_loss = -(
            hindsight_interaction_log_prob
            * skill_event_weight
        ).sum() / skill_event_weight.sum().clamp_min(1.0)
        event_hindsight_skill_loss = (
            event_hindsight_skill_loss * skill_support_scale
        )
        tracking_improvement = (
            current_tracking_error - next_tracking_error
        )
        skill_effect_target = torch.stack(
            (
                (next_context[:, BASE_VX_INDEX] / 0.75).clamp(-1.0, 1.0),
                (next_context[:, BASE_WZ_INDEX] / 1.50).clamp(-1.0, 1.0),
                (progress_delta / 0.08).clamp(-1.0, 1.0),
                (tracking_improvement / 0.40).clamp(-1.0, 1.0),
                2.0 * target[:, 0] - 1.0,
                2.0 * target[:, 1] - 1.0,
                2.0 * target[:, 2] - 1.0,
                2.0 * target[:, 3] - 1.0,
                (interaction_delta / 0.25).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        effect_weights = skill_effect.new_tensor(
            (1.0, 0.70, 1.0, 0.80, 0.75, 0.75, 0.65, 0.65, 1.0)
        ).unsqueeze(0).expand_as(skill_effect).clone()
        effect_weights[:, -1] = interaction_active
        effect_error = F.smooth_l1_loss(
            skill_effect,
            skill_effect_target.detach(),
            reduction="none",
        )
        effect_weight_sum = effect_weights.sum(dim=1).clamp_min(1.0)
        skill_effect_per_sample = (
            effect_error * effect_weights
        ).sum(dim=1) / effect_weight_sum
        skill_effect_loss = (
            skill_effect_per_sample * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)

        calibrated_error = (
            (skill_effect.detach() - skill_effect_target.detach()).abs()
            * effect_weights
        ).sum(dim=1) / effect_weight_sum
        effect_confidence_target = torch.exp(
            -2.5 * calibrated_error
        ).clamp(0.05, 0.95)
        effect_confidence_per_sample = F.binary_cross_entropy(
            skill_effect_confidence.clamp(1.0e-5, 1.0 - 1.0e-5),
            effect_confidence_target,
            reduction="none",
        )
        skill_effect_confidence_loss = (
            effect_confidence_per_sample * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)

        (
            motion_execution,
            motion_execution_confidence,
        ) = self.policy.predict_motion_execution(
            current_context, actions_batch[:, 12:16].detach()
        )
        motion_execution_target = torch.stack(
            (
                (next_context[:, BASE_VX_INDEX] / 0.75).clamp(-1.0, 1.0),
                (next_context[:, BASE_WZ_INDEX] / 1.50).clamp(-1.0, 1.0),
                normalized_tracking,
                2.0 * target[:, 0] - 1.0,
                2.0 * target[:, 1] - 1.0,
                2.0 * target[:, 2] - 1.0,
                2.0 * target[:, 3] - 1.0,
            ),
            dim=-1,
        )
        motion_execution_error = F.smooth_l1_loss(
            motion_execution,
            motion_execution_target.detach(),
            reduction="none",
        )
        motion_execution_weights = motion_execution.new_tensor(
            (1.0, 0.75, 0.80, 1.0, 0.90, 0.80, 0.65)
        )
        motion_execution_per_sample = (
            motion_execution_error * motion_execution_weights
        ).sum(dim=1) / motion_execution_weights.sum()
        motion_execution_loss = (
            motion_execution_per_sample * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)
        calibrated_motion_error = (
            (
                motion_execution.detach()
                - motion_execution_target.detach()
            ).abs()
            * motion_execution_weights
        ).sum(dim=1) / motion_execution_weights.sum()
        motion_confidence_target = torch.exp(
            -2.5 * calibrated_motion_error
        ).clamp(0.05, 0.95)
        motion_confidence_error = F.binary_cross_entropy(
            motion_execution_confidence.clamp(1.0e-5, 1.0 - 1.0e-5),
            motion_confidence_target,
            reduction="none",
        )
        motion_execution_confidence_loss = (
            motion_confidence_error * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)
        embodiment_response = actor.embodiment_response_prior(
            actions_batch[:, 12:16].detach()
        )
        embodiment_response_target = motion_execution_target[:, :2].detach()
        response_activity = (
            actions_batch[:, 12:16].detach().abs().amax(dim=1)
            / actor.embodiment_response_action.clamp_min(1.0)
        ).clamp(0.0, 1.0)
        response_weight = (
            skill_valid
            * nonterminal
            * (0.20 + 0.80 * response_activity)
        )
        embodiment_response_error = F.smooth_l1_loss(
            embodiment_response,
            embodiment_response_target,
            reduction="none",
        ).mean(dim=1)
        embodiment_response_loss = (
            embodiment_response_error * response_weight
        ).sum() / response_weight.sum().clamp_min(1.0)
        response_anchor_loss = F.smooth_l1_loss(
            actor.embodiment_response_matrix,
            actor.embodiment_response_anchor,
        )
        embodiment_response_loss = (
            embodiment_response_loss + 0.02 * response_anchor_loss
        )

        candidate_effects = actor.last_skill_effects
        candidate_confidence = actor.last_skill_effect_confidence
        if candidate_effects is None or candidate_confidence is None:
            raise RuntimeError("TACTIC actor did not expose skill effects")
        motion_effects = candidate_effects.reshape(
            -1,
            MOTION_SKILL_COUNT,
            INTERACTION_SKILL_COUNT,
            candidate_effects.shape[-1],
        ).mean(dim=2)
        motion_confidence = candidate_confidence.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).mean(dim=2)
        pair_indices = torch.triu_indices(
            MOTION_SKILL_COUNT,
            MOTION_SKILL_COUNT,
            offset=1,
            device=motion_effects.device,
        )
        pair_distance = torch.linalg.vector_norm(
            motion_effects[:, pair_indices[0], :4]
            - motion_effects[:, pair_indices[1], :4],
            dim=-1,
        )
        pair_confidence = torch.minimum(
            motion_confidence[:, pair_indices[0]],
            motion_confidence[:, pair_indices[1]],
        )
        pair_reliability = (
            (pair_confidence.detach() - 0.35) / 0.65
        ).clamp(0.0, 1.0)
        pair_reliability = pair_reliability * skill_valid.unsqueeze(1)
        grounded_effect_diversity_loss = (
            pair_reliability * torch.relu(0.12 - pair_distance).square()
        ).sum() / pair_reliability.sum().clamp_min(1.0)

        task_transition_logits, skill_transition_logits = (
            self.policy.transition_logits(current_context, next_context)
        )
        task_transition_per_sample = F.cross_entropy(
            task_transition_logits, task_target, reduction="none"
        )
        skill_grid = skill_transition_logits.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        motion_transition_logits = torch.logsumexp(skill_grid, dim=2)
        interaction_transition_logits = torch.logsumexp(skill_grid, dim=1)
        motion_target = skill_target // INTERACTION_SKILL_COUNT
        interaction_target = skill_target % INTERACTION_SKILL_COUNT
        motion_transition = F.cross_entropy(
            motion_transition_logits, motion_target, reduction="none"
        )
        interaction_transition = F.cross_entropy(
            interaction_transition_logits,
            interaction_target,
            reduction="none",
        )
        interaction_valid = skill_valid * interaction_active
        task_transition_loss = (
            task_transition_per_sample * task_valid
        ).sum() / task_valid.sum().clamp_min(1.0)
        motion_transition_loss = (
            motion_transition * skill_valid
        ).sum() / skill_valid.sum().clamp_min(1.0)
        interaction_transition_loss = (
            interaction_transition * interaction_valid
        ).sum() / interaction_valid.sum().clamp_min(1.0)
        skill_transition_loss = (
            0.60 * motion_transition_loss
            + 0.40 * interaction_transition_loss
        )

        task_probability_uncommitted, skill_probability = (
            self.policy.uncommitted_selection_probabilities(
                current_context, actions_batch
            )
        )
        valid_task_mask = (
            (current_slots[:, :, 11] > 0.5)
            & (current_slots[:, :, 12] < 0.5)
            & (current_slots[:, :, 13] > 0.5)
        ).float()
        task_weight = nonterminal.unsqueeze(1) * valid_task_mask
        task_target_marginal = task_weight.sum(dim=0)
        task_target_marginal = (
            task_target_marginal
            / task_target_marginal.sum().clamp_min(1.0)
        )
        task_policy_marginal = (
            task_probability_uncommitted * task_weight
        ).sum(dim=0)
        task_policy_marginal = (
            task_policy_marginal
            / task_policy_marginal.sum().clamp_min(1.0)
        )
        task_usage_loss = (
            task_policy_marginal.clamp_min(1.0e-6)
            * torch.log(
                task_policy_marginal.clamp_min(1.0e-6)
                / task_target_marginal.clamp_min(1.0e-6)
            )
        ).sum()
        task_remaining = current_slots[
            :, :, TASK_SLOT_REMAINING_PROGRESS_INDEX
        ].clamp(0.0, 1.0)
        grounded_frontier = actor.grounded_task_utility(
            current_slots,
            current_context[:, :GLOBAL_CONTEXT_DIM],
        ).detach()
        frontier_score = (
            valid_task_mask
            * (0.35 + 0.65 * task_remaining)
            * (0.70 + 0.30 * torch.sigmoid(2.0 * grounded_frontier))
        )
        frontier_target = frontier_score / frontier_score.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        task_frontier_coverage_per_sample = -(
            frontier_target
            * torch.log(task_probability_uncommitted.clamp_min(1.0e-8))
        ).sum(dim=1)
        task_frontier_coverage_loss = (
            task_frontier_coverage_per_sample * nonterminal
        ).sum() / nonterminal.sum().clamp_min(1.0)
        selected_frontier_probability = (
            task_probability_uncommitted
            * frontier_target
        ).sum(dim=1)
        selected_frontier_probability = (
            selected_frontier_probability * nonterminal
        ).sum() / nonterminal.sum().clamp_min(1.0)
        valid_weight = nonterminal.unsqueeze(1)
        skill_grid_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        motion_probability = skill_grid_probability.sum(dim=2)
        interaction_probability = skill_grid_probability.sum(dim=1)

        # Compare every lower-level motion option under the same task
        # context.  The calibrated short-horizon model supplies an MPC-like
        # counterfactual target, while the detached target prevents the effect
        # model from improving the loss by changing its own predictions.
        candidate_motion_effect = (
            actor.last_motion_execution_prediction
        )
        candidate_motion_confidence = (
            actor.last_motion_execution_confidence
        )
        candidate_motion_action = actor.last_motion_action_candidates
        if (
            candidate_motion_effect is None
            or candidate_motion_confidence is None
            or candidate_motion_action is None
            or actor.last_task_motion_request is None
        ):
            raise RuntimeError(
                "TACTIC actor did not expose counterfactual motion effects"
            )
        counterfactual_desired_motion = torch.stack(
            (
                (
                    actor.last_task_motion_request[:, 0] / 0.75
                ).clamp(-1.0, 1.0),
                (
                    actor.last_task_motion_request[:, 1] / 1.50
                ).clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
        counterfactual_tracking_cost = (
            0.65
            * (
                candidate_motion_effect[:, :, 0]
                - counterfactual_desired_motion[:, None, 0]
            ).square()
            + 0.35
            * (
                candidate_motion_effect[:, :, 1]
                - counterfactual_desired_motion[:, None, 1]
            ).square()
        )
        desired_direction = counterfactual_desired_motion.sign()
        counterfactual_alignment = (
            0.65
            * candidate_motion_effect[:, :, 0]
            * desired_direction[:, None, 0]
            + 0.35
            * candidate_motion_effect[:, :, 1]
            * desired_direction[:, None, 1]
        )
        candidate_action_energy = (
            candidate_motion_action
            / max(float(actor.wheel_action_limit), 1.0e-4)
        ).square().mean(dim=2)
        model_counterfactual_utility = (
            -counterfactual_tracking_cost
            + 0.20 * counterfactual_alignment
            + 0.15 * candidate_motion_effect[:, :, 2]
            + 0.20
            * (
                candidate_motion_effect[:, :, 3:7]
                * candidate_motion_effect.new_tensor(
                    (0.34, 0.26, 0.22, 0.18)
                )
            ).sum(dim=2)
            - 0.03 * candidate_action_energy
        )
        grounded_motion_utility = actor.last_motion_objective_scores
        if grounded_motion_utility is None:
            raise RuntimeError(
                "TACTIC actor did not expose grounded motion utility"
            )
        model_reliability = (
            float(actor.motion_execution_maturity)
            * (
                (candidate_motion_confidence.detach() - 0.25) / 0.55
            ).clamp(0.0, 1.0)
        )
        counterfactual_utility = (
            grounded_motion_utility.detach()
            + model_reliability
            * (
                model_counterfactual_utility
                - grounded_motion_utility.detach()
            )
        )
        counterfactual_target = torch.softmax(
            counterfactual_utility.detach()
            / self.counterfactual_skill_temperature,
            dim=1,
        )
        counterfactual_per_sample = -(
            counterfactual_target
            * torch.log(motion_probability.clamp_min(1.0e-8))
        ).sum(dim=1)
        counterfactual_reliability = model_reliability.mean(dim=1).clamp(
            0.05, 1.0
        )
        counterfactual_weight = (
            task_valid
            * (
                counterfactual_desired_motion.abs().amax(dim=1)
                > 0.015
            ).float()
            * counterfactual_reliability
        )
        counterfactual_skill_selection_loss = (
            counterfactual_per_sample * counterfactual_weight
        ).sum() / counterfactual_weight.sum().clamp_min(1.0)
        counterfactual_target_entropy = -(
            counterfactual_target
            * torch.log(counterfactual_target.clamp_min(1.0e-8))
        ).sum(dim=1)
        counterfactual_target_entropy = (
            counterfactual_target_entropy * counterfactual_weight
        ).sum() / counterfactual_weight.sum().clamp_min(1.0)
        counterfactual_target_peak = (
            counterfactual_target.max(dim=1).values
            * counterfactual_weight
        ).sum() / counterfactual_weight.sum().clamp_min(1.0)
        (
            planned_motion_effect,
            planned_motion_confidence,
        ) = actor.predict_motion_execution_frozen(
            current_context, candidate_motion_action
        )
        desired_tracking_improvement = (
            current_tracking_error / 0.40
        ).clamp(0.10, 0.75)
        control_floor = current_control.new_tensor(
            (0.45, 0.45, 0.50, 0.35)
        )
        required_control = current_control + 0.35 * (
            control_floor - current_control
        ).clamp_min(0.0)
        desired_successor = torch.cat(
            (
                counterfactual_desired_motion,
                desired_tracking_improvement.unsqueeze(1),
                2.0 * required_control - 1.0,
            ),
            dim=1,
        )
        if desired_successor.shape[1] != MOTION_EXECUTION_EFFECT_DIM:
            raise RuntimeError("Invalid TACTIC successor target width")
        successor_error = F.smooth_l1_loss(
            planned_motion_effect,
            desired_successor[:, None, :].expand_as(
                planned_motion_effect
            ).detach(),
            reduction="none",
        )
        successor_cost = (
            successor_error
            * successor_error.new_tensor(
                (0.65, 0.35, 0.35, 0.55, 0.50, 0.45, 0.25)
            )
        ).sum(dim=2) / 3.10
        planned_control = 0.5 * (
            planned_motion_effect[:, :, 3:7] + 1.0
        )
        cbf_violation = (
            required_control[:, None, :2] - planned_control[:, :, :2]
        ).clamp_min(0.0).square().mean(dim=2)
        clf_violation = (
            required_control[:, None, 2] - planned_control[:, :, 2]
        ).clamp_min(0.0).square()
        tracking_decrease_violation = (
            desired_tracking_improvement[:, None]
            - planned_motion_effect[:, :, 2]
        ).clamp_min(0.0).square()
        successor_cost = (
            successor_cost
            + 0.40 * cbf_violation
            + 0.25 * clf_violation
            + 0.20 * tracking_decrease_violation
        )
        successor_reliability = (
            0.10
            + 0.90
            * float(actor.motion_execution_maturity)
            * planned_motion_confidence.detach().clamp(0.0, 1.0)
        )
        successor_weight = (
            motion_probability.detach()
            * successor_reliability
            * counterfactual_weight.unsqueeze(1)
        )
        successor_decoder_loss = (
            successor_cost * successor_weight
        ).sum() / successor_weight.sum().clamp_min(1.0)

        interaction_phase_target = (
            self.policy.actor.interaction_phase_distribution(
                selected_slot
            ).detach()
        )
        release_frontier = actor.last_interaction_release_frontier
        if release_frontier is None:
            raise RuntimeError("Interaction release frontier was not evaluated")
        release_frontier = release_frontier.detach()
        delivery_active = selected_slot[
            :, TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        base_interaction_phase_weight = nonterminal * delivery_active
        current_task_valid = candidate_valid_task_mask.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1)
        task_alternative_mask = candidate_valid_task_mask.bool().clone()
        task_alternative_mask.scatter_(1, task_target.unsqueeze(1), False)
        current_task_utility = counterfactual_task_utility.gather(
            1, task_target.unsqueeze(1)
        ).squeeze(1)
        best_alternative_task_utility = (
            counterfactual_task_utility.masked_fill(
                ~task_alternative_mask, -1.0e4
            ).max(dim=1).values
        )
        task_has_alternative = task_alternative_mask.any(dim=1)
        task_switch_gain = torch.where(
            task_has_alternative,
            best_alternative_task_utility - current_task_utility,
            torch.zeros_like(current_task_utility),
        )
        task_stagnation = torch.sigmoid(
            (-normalized_progress - 0.02) / 0.10
        )
        task_constraint_exit = torch.maximum(
            actual_cbf_violation,
            0.65 * actual_clf_violation,
        )
        task_switch_readiness = torch.maximum(
            task_stagnation,
            task_constraint_exit,
        )
        task_alternative_advantage = torch.sigmoid(
            (task_switch_gain - 0.08) / 0.06
        ) * task_has_alternative.float()
        task_termination_target = (
            0.02
            + 0.93
            * task_alternative_advantage
            * task_switch_readiness
        )
        forced_task_termination = (
            (selected_slot[:, TASK_SLOT_COMPLETED_INDEX] > 0.5)
            | (next_selected_slot[:, TASK_SLOT_COMPLETED_INDEX] > 0.5)
            | (current_task_valid < 0.5)
        )
        task_termination_target = torch.where(
            forced_task_termination,
            torch.full_like(task_termination_target, 0.98),
            torch.where(
                task_has_alternative,
                task_termination_target,
                torch.full_like(task_termination_target, 0.02),
            ),
        ).detach()
        task_termination_weight = (
            task_valid
            * (task_has_alternative | forced_task_termination).float()
        )

        current_motion_skill = skill_target // INTERACTION_SKILL_COUNT
        current_interaction_skill = skill_target % INTERACTION_SKILL_COUNT
        motion_alternative_mask = torch.ones_like(
            counterfactual_utility, dtype=torch.bool
        )
        motion_alternative_mask.scatter_(
            1, current_motion_skill.unsqueeze(1), False
        )
        current_motion_utility = counterfactual_utility.gather(
            1, current_motion_skill.unsqueeze(1)
        ).squeeze(1)
        best_alternative_motion_utility = (
            counterfactual_utility.masked_fill(
                ~motion_alternative_mask, -1.0e4
            ).max(dim=1).values
        )
        skill_switch_gain = (
            best_alternative_motion_utility - current_motion_utility
        )
        skill_termination_target = torch.sigmoid(
            (skill_switch_gain - 0.03) / 0.06
        )
        target_interaction_skill = torch.argmax(
            interaction_phase_target, dim=1
        )
        phase_switch_required = (
            interaction_active
            * (current_interaction_skill != target_interaction_skill).float()
        )
        payload_preview_pressure = (
            (
                PAYLOAD_SKILL_PREVIEW_RESERVE
                - current_context[:, PREVIEW_MARGIN_INDEX]
            )
            / PAYLOAD_SKILL_PREVIEW_RESERVE
        ).clamp(0.0, 1.0)
        payload_safety_pressure = (
            (
                PAYLOAD_SKILL_SAFETY_RESERVE
                - current_context[:, SAFETY_MARGIN_INDEX]
            )
            / PAYLOAD_SKILL_SAFETY_RESERVE
        ).clamp(0.0, 1.0)
        payload_distance = (
            4.0 * selected_slot[:, TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 6.0)
        payload_transient_pressure, _ = actor._payload_transient_demand(
            current_context,
            payload_distance,
        )
        payload_barrier_pressure = (
            selected_slot[:, TASK_SLOT_CARRYING_INDEX].clamp(0.0, 1.0)
            * torch.maximum(
                torch.maximum(
                    payload_preview_pressure,
                    payload_safety_pressure,
                ),
                payload_transient_pressure,
            )
        )
        payload_motion_exit_target = (
            payload_barrier_pressure
            * torch.sigmoid(
                (skill_switch_gain - PAYLOAD_SKILL_SWITCH_MARGIN) / 0.06
            )
        )
        skill_survival = actor.last_skill_survival
        if skill_survival is None:
            raise RuntimeError(
                "TACTIC actor did not expose termination survival"
            )
        skill_survival_grid = skill_survival.detach().reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        survival_interaction_index = current_interaction_skill.view(
            -1, 1, 1
        ).expand(-1, MOTION_SKILL_COUNT, 1)
        motion_survival = skill_survival_grid.gather(
            2, survival_interaction_index
        ).squeeze(2)
        current_motion_survival = motion_survival.gather(
            1, current_motion_skill.unsqueeze(1)
        ).squeeze(1)
        survival_alternative_mask = torch.ones_like(
            motion_survival, dtype=torch.bool
        )
        survival_alternative_mask.scatter_(
            1, current_motion_skill.unsqueeze(1), False
        )
        best_alternative_survival = motion_survival.masked_fill(
            ~survival_alternative_mask, -1.0
        ).max(dim=1).values
        payload_survival_advantage = (
            best_alternative_survival - current_motion_survival
        ).clamp_min(0.0)
        payload_survival_maturity = float(
            actor.payload_survival_maturity
        )
        payload_survival_authority = (
            payload_barrier_pressure * payload_survival_maturity
        )
        payload_survival_exit_target = (
            payload_survival_authority
            * torch.sigmoid(
                (
                    payload_survival_advantage
                    - 0.50 * PAYLOAD_SKILL_SWITCH_MARGIN
                )
                / 0.02
            )
        )
        skill_termination_target = torch.maximum(
            skill_termination_target,
            0.95 * phase_switch_required,
        )
        skill_termination_target = torch.maximum(
            skill_termination_target,
            0.90 * payload_motion_exit_target,
        )
        skill_termination_target = torch.maximum(
            skill_termination_target,
            0.95 * payload_survival_exit_target,
        ).clamp(0.02, 0.98).detach()
        skill_termination_weight = skill_valid * (
            0.25
            + 0.75
            * candidate_motion_confidence.detach().mean(dim=1).clamp(
                0.0, 1.0
            )
        )
        termination_slice = ACTION_LAYOUT.slices()["termination"]
        task_termination_logit = current_action_mean[
            :, termination_slice.start
        ]
        skill_termination_logit = current_action_mean[
            :, termination_slice.start + 1
        ]
        task_termination_error = F.binary_cross_entropy_with_logits(
            task_termination_logit,
            task_termination_target,
            reduction="none",
        )
        skill_termination_error = F.binary_cross_entropy_with_logits(
            skill_termination_logit,
            skill_termination_target,
            reduction="none",
        )
        task_termination_loss = (
            task_termination_error * task_termination_weight
        ).sum() / task_termination_weight.sum().clamp_min(1.0)
        skill_termination_loss = (
            skill_termination_error * skill_termination_weight
        ).sum() / skill_termination_weight.sum().clamp_min(1.0)
        counterfactual_termination_loss = (
            0.35 * task_termination_loss
            + 0.65 * skill_termination_loss
        )
        interaction_phase_error = -(
            interaction_phase_target
            * torch.log(interaction_probability.clamp_min(1.0e-8))
        ).sum(dim=1)
        release_balancing_weight = (
            1.0
            + (self.interaction_release_focus - 1.0)
            * interaction_phase_target[:, 2]
        ).detach()
        interaction_phase_weight = (
            base_interaction_phase_weight * release_balancing_weight
        )
        interaction_phase_loss = (
            interaction_phase_error * interaction_phase_weight
        ).sum() / interaction_phase_weight.sum().clamp_min(1.0)
        valid_normalizer = valid_weight.sum().clamp_min(1.0)
        motion_marginal = (
            motion_probability * valid_weight
        ).sum(dim=0) / valid_normalizer
        interaction_weight = (
            nonterminal * interaction_active
        ).unsqueeze(1)
        interaction_normalizer = interaction_weight.sum().clamp_min(1.0)
        interaction_marginal = (
            interaction_probability * interaction_weight
        ).sum(dim=0) / interaction_normalizer
        manipulation_active = selected_slot[
            :, TASK_SLOT_MANIPULATION_TYPE_INDEX
        ].clamp(0.0, 1.0)
        uniform_phase = torch.full_like(
            interaction_phase_target,
            1.0 / float(INTERACTION_SKILL_COUNT),
        )
        interaction_usage_target = (
            delivery_active.unsqueeze(1) * interaction_phase_target
            + manipulation_active.unsqueeze(1) * uniform_phase
        )
        interaction_target_marginal = (
            interaction_usage_target * interaction_weight
        ).sum(dim=0) / interaction_normalizer

        def marginal_kl(
            marginal: torch.Tensor,
            target: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if target is None:
                target = torch.full_like(
                    marginal, 1.0 / float(marginal.numel())
                )
            target = target / target.sum().clamp_min(1.0e-6)
            return (
                marginal.clamp_min(1.0e-6)
                * torch.log(
                    marginal.clamp_min(1.0e-6)
                    / target.clamp_min(1.0e-6)
                )
            ).sum()

        skill_usage_loss = 0.5 * (
            marginal_kl(motion_marginal)
            + marginal_kl(
                interaction_marginal,
                interaction_target_marginal,
            )
        )
        motion_entropy = -(
            motion_probability
            * torch.log(motion_probability.clamp_min(1.0e-8))
        ).sum(dim=1)
        interaction_entropy = -(
            interaction_probability
            * torch.log(interaction_probability.clamp_min(1.0e-8))
        ).sum(dim=1)
        motion_confidence_loss = (
            motion_entropy * nonterminal
        ).sum() / nonterminal.sum().clamp_min(1.0)
        interaction_confidence_loss = (
            interaction_entropy * nonterminal * interaction_active
        ).sum() / (nonterminal * interaction_active).sum().clamp_min(1.0)
        skill_confidence_loss = (
            0.60 * motion_confidence_loss
            + 0.40 * interaction_confidence_loss
        )

        slot_latent = actor.last_slot_latent
        if slot_latent is None:
            raise RuntimeError("TACTIC actor did not expose task-slot embeddings")
        slot_loss = self._off_diagonal_energy(slot_latent.mean(dim=0))
        skill_loss = 0.5 * (
            self._off_diagonal_energy(actor.motion_skill_embedding.weight)
            + self._off_diagonal_energy(
                actor.interaction_skill_embedding.weight
            )
        )
        motion_gain_distance = torch.pdist(
            actor.motion_kinematic_gain, p=2
        )
        motion_gain_diversity_loss = torch.relu(
            0.30 - motion_gain_distance
        ).square().mean()
        objective_basis = F.normalize(
            torch.softmax(actor.motion_objective_basis, dim=-1),
            dim=-1,
        )
        objective_similarity = (
            objective_basis @ objective_basis.transpose(0, 1)
        )
        objective_mask = ~torch.eye(
            MOTION_SKILL_COUNT,
            device=objective_similarity.device,
            dtype=torch.bool,
        )
        motion_objective_diversity_loss = torch.relu(
            objective_similarity[objective_mask] - 0.55
        ).square().mean()
        total = (
            self.control_prediction_coef * control_loss
            + self.skill_feasibility_coef * skill_feasibility_loss
            + self.task_outcome_coef * task_outcome_loss
            + self.task_outcome_confidence_coef
            * task_outcome_confidence_loss
            + self.skill_outcome_coef * skill_outcome_loss
            + self.payload_survival_coef * payload_survival_loss
            + self.skill_effect_prediction_coef * skill_effect_loss
            + self.skill_effect_confidence_coef
            * skill_effect_confidence_loss
            + self.motion_execution_prediction_coef
            * motion_execution_loss
            + self.motion_execution_confidence_coef
            * motion_execution_confidence_loss
            + self.embodiment_response_coef
            * embodiment_response_loss
            + self.constraint_multiplier_coef
            * constraint_multiplier_loss
            + self.grounded_effect_diversity_coef
            * grounded_effect_diversity_loss
            + self.task_option_credit_coef * task_option_credit_loss
            + self.skill_option_credit_coef * skill_option_credit_loss
            + self.event_hindsight_task_coef
            * event_hindsight_task_loss
            + self.event_hindsight_skill_coef
            * event_hindsight_skill_loss
            + self.task_transition_coef * task_transition_loss
            + self.skill_transition_coef * skill_transition_loss
            + self.task_usage_coef * task_usage_loss
            + self.task_frontier_coverage_coef
            * task_frontier_coverage_loss
            + self.interaction_phase_coef * interaction_phase_loss
            + self.skill_usage_coef * skill_usage_loss
            + self.skill_confidence_coef * skill_confidence_loss
            + self.slot_diversity_coef * slot_loss
            + self.skill_diversity_coef * skill_loss
            + self.motion_gain_diversity_coef
            * motion_gain_diversity_loss
            + self.task_control_objective_coef
            * task_control_objective_loss
            + self.skill_predictive_control_coef
            * skill_predictive_control_loss
            + self.relational_subgoal_grounding_coef
            * relational_subgoal_grounding_loss
            + self.task_skill_projection_coef
            * task_skill_projection_loss
            + self.counterfactual_task_selection_coef
            * counterfactual_task_selection_loss
            + self.counterfactual_skill_selection_coef
            * counterfactual_skill_selection_loss
            + self.counterfactual_termination_coef
            * counterfactual_termination_loss
            + self.successor_decoder_coef * successor_decoder_loss
            + self.motion_objective_diversity_coef
            * motion_objective_diversity_loss
        )
        if self.training_stage == "survival":
            total = self.payload_survival_coef * payload_survival_loss
        return total, {
            "next_control_prediction": control_loss.detach(),
            "skill_feasibility": skill_feasibility_loss.detach(),
            "task_outcome_prediction": task_outcome_loss.detach(),
            "task_outcome_confidence": (
                task_outcome_confidence_loss.detach()
            ),
            "task_outcome_confidence_mean": (
                selected_task_confidence * task_valid
            ).sum().detach()
            / task_valid.sum().clamp_min(1.0),
            "skill_outcome_prediction": skill_outcome_loss.detach(),
            "payload_survival": payload_survival_loss.detach(),
            "payload_survival_support": current_payload.mean().detach(),
            "payload_drop_fraction": payload_drop.mean().detach(),
            "payload_survival_prediction": (
                selected_skill_survival * current_payload
            ).sum().detach()
            / current_payload.sum().clamp_min(1.0),
            "skill_effect_prediction": skill_effect_loss.detach(),
            "skill_effect_confidence": (
                skill_effect_confidence_loss.detach()
            ),
            "skill_effect_confidence_mean": (
                skill_effect_confidence * skill_valid
            ).sum().detach()
            / skill_valid.sum().clamp_min(1.0),
            "motion_execution_prediction": (
                motion_execution_loss.detach()
            ),
            "motion_execution_confidence": (
                motion_execution_confidence_loss.detach()
            ),
            "embodiment_response_prediction": (
                embodiment_response_loss.detach()
            ),
            "embodiment_response_vx_abs": (
                embodiment_response[:, 0].abs() * response_weight
            ).sum().detach()
            / response_weight.sum().clamp_min(1.0),
            "embodiment_response_wz_abs": (
                embodiment_response[:, 1].abs() * response_weight
            ).sum().detach()
            / response_weight.sum().clamp_min(1.0),
            "constraint_multiplier": (
                constraint_multiplier_loss.detach()
            ),
            "task_constraint_multiplier_mean": (
                task_constraint_multiplier.mean().detach()
            ),
            "skill_constraint_multiplier_mean": (
                skill_constraint_multipliers.mean().detach()
            ),
            "hierarchy_cbf_violation": (
                actual_cbf_violation * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "hierarchy_cbf_raw_violation": (
                raw_cbf_violation * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "hierarchy_cbf_transient_slack": (
                transient_slack * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "hierarchy_clf_violation": (
                actual_clf_violation * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "hierarchy_cbf_regression": (
                cbf_regression * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "hierarchy_clf_regression": (
                clf_regression * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "safe_task_achievement": (
                safe_task_achievement * task_valid
            ).sum().detach()
            / task_valid.sum().clamp_min(1.0),
            "safe_skill_achievement": (
                safe_skill_achievement * skill_valid
            ).sum().detach()
            / skill_valid.sum().clamp_min(1.0),
            "unsafe_positive_progress_rate": (
                (
                    (normalized_progress > 0.0)
                    & (actual_cbf_violation > 0.0)
                ).float()
                * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "motion_execution_confidence_mean": (
                motion_execution_confidence * skill_valid
            ).sum().detach()
            / skill_valid.sum().clamp_min(1.0),
            "grounded_effect_diversity": (
                grounded_effect_diversity_loss.detach()
            ),
            "task_option_credit": task_option_credit_loss.detach(),
            "skill_option_credit": skill_option_credit_loss.detach(),
            "event_hindsight_task": (
                event_hindsight_task_loss.detach()
            ),
            "event_hindsight_skill": (
                event_hindsight_skill_loss.detach()
            ),
            "event_credit_mass": event_credit.mean().detach(),
            "task_option_advantage_abs": (
                task_advantage.abs() * task_valid
            ).sum()
            / task_valid.sum().clamp_min(1.0),
            "skill_option_advantage_abs": (
                skill_advantage.abs() * skill_valid
            ).sum()
            / skill_valid.sum().clamp_min(1.0),
            "task_transition": task_transition_loss.detach(),
            "skill_transition": skill_transition_loss.detach(),
            "motion_transition": motion_transition_loss.detach(),
            "interaction_transition": interaction_transition_loss.detach(),
            "interaction_active_fraction": (
                interaction_active * nonterminal
            ).sum().detach()
            / nonterminal.sum().clamp_min(1.0),
            "task_usage": task_usage_loss.detach(),
            "task_frontier_coverage": (
                task_frontier_coverage_loss.detach()
            ),
            "selected_frontier_probability": (
                selected_frontier_probability.detach()
            ),
            "interaction_phase": interaction_phase_loss.detach(),
            "interaction_release_target": (
                interaction_phase_target[:, 2]
                * base_interaction_phase_weight
            ).sum().detach()
            / base_interaction_phase_weight.sum().clamp_min(1.0),
            "interaction_release_frontier": (
                release_frontier * base_interaction_phase_weight
            ).sum().detach()
            / base_interaction_phase_weight.sum().clamp_min(1.0),
            "interaction_release_balance_weight": (
                release_balancing_weight * base_interaction_phase_weight
            ).sum().detach()
            / base_interaction_phase_weight.sum().clamp_min(1.0),
            "task_control_objective": task_control_objective_loss.detach(),
            "task_clf_descent": task_descent_loss.detach(),
            "task_barrier_authority": task_barrier_loss.detach(),
            "skill_predictive_control": (
                skill_predictive_control_loss.detach()
            ),
            "relational_subgoal_grounding": (
                relational_subgoal_grounding_loss.detach()
            ),
            "task_subgoal_alignment": task_subgoal_alignment.detach(),
            "task_subgoal_authority": (
                task_subgoal_authority.mean().detach()
            ),
            "task_skill_projection": task_skill_projection_loss.detach(),
            "task_transient_capacity": transient_capacity.mean().detach(),
            "task_posture_capacity": posture_capacity.mean().detach(),
            "task_projected_turn_demand": (
                candidate_task_motion[:, :, 1].abs()
                * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "task_semantic_turn_demand": (
                candidate_semantic_yaw.abs() * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "task_lateral_turn_demand": (
                candidate_lateral_yaw.abs() * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "task_target_yaw_residual": (
                target_task_yaw.abs() * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "task_interaction_authority": (
                interaction_authority * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "task_arm_scope_error": (
                task_arm_scope_error * candidate_valid_weight
            ).sum().detach()
            / candidate_valid_weight.sum().clamp_min(1.0),
            "counterfactual_task_selection": (
                counterfactual_task_selection_loss.detach()
            ),
            "counterfactual_task_target_entropy": (
                counterfactual_task_target_entropy.detach()
            ),
            "counterfactual_task_target_peak": (
                counterfactual_task_target_peak.detach()
            ),
            "grounded_task_utility_span": grounded_task_span.detach(),
            "task_outcome_blend": task_outcome_blend.detach(),
            "grounded_task_target_agreement": (
                grounded_target_agreement.detach()
            ),
            "sampled_task_target_probability": (
                sampled_task_target_probability.detach()
            ),
            "recovery_candidate_fraction": (
                candidate_valid_task_mask[:, 4].mean().detach()
            ),
            "recovery_task_probability": (
                candidate_task_probability[:, 4].mean().detach()
            ),
            "recovery_target_probability": (
                counterfactual_task_target[:, 4].mean().detach()
            ),
            "counterfactual_termination": (
                counterfactual_termination_loss.detach()
            ),
            "task_termination_target": (
                task_termination_target * task_termination_weight
            ).sum().detach()
            / task_termination_weight.sum().clamp_min(1.0),
            "skill_termination_target": (
                skill_termination_target * skill_termination_weight
            ).sum().detach()
            / skill_termination_weight.sum().clamp_min(1.0),
            "payload_skill_barrier_pressure": (
                payload_barrier_pressure.mean().detach()
            ),
            "payload_transient_demand": (
                payload_transient_pressure.mean().detach()
            ),
            "payload_skill_exit_target": (
                payload_motion_exit_target.mean().detach()
            ),
            "payload_survival_exit_target": (
                payload_survival_exit_target.mean().detach()
            ),
            "payload_survival_authority": (
                payload_survival_authority.mean().detach()
            ),
            "task_switch_gain": (
                task_switch_gain * task_termination_weight
            ).sum().detach()
            / task_termination_weight.sum().clamp_min(1.0),
            "skill_switch_gain": (
                skill_switch_gain * skill_termination_weight
            ).sum().detach()
            / skill_termination_weight.sum().clamp_min(1.0),
            "counterfactual_skill_selection": (
                counterfactual_skill_selection_loss.detach()
            ),
            "counterfactual_target_entropy": (
                counterfactual_target_entropy.detach()
            ),
            "counterfactual_target_peak": (
                counterfactual_target_peak.detach()
            ),
            "successor_decoder": successor_decoder_loss.detach(),
            "successor_predicted_vx": (
                planned_motion_effect[:, :, 0] * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "successor_predicted_wz": (
                planned_motion_effect[:, :, 1] * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "successor_predicted_cbf": (
                planned_control[:, :, :2].amin(dim=2)
                * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "successor_predicted_clf": (
                planned_control[:, :, 2] * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "successor_cbf_violation": (
                cbf_violation * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "successor_clf_violation": (
                clf_violation * successor_weight
            ).sum().detach()
            / successor_weight.sum().clamp_min(1.0),
            "skill_usage": skill_usage_loss.detach(),
            "skill_confidence": skill_confidence_loss.detach(),
            "task_slot_diversity": slot_loss.detach(),
            "skill_embedding_diversity": skill_loss.detach(),
            "motion_gain_diversity": motion_gain_diversity_loss.detach(),
            "motion_objective_diversity": (
                motion_objective_diversity_loss.detach()
            ),
        }

    def update(self):
        self._configure_physical_stage()
        if self.training_stage == "recovery":
            actor = self.policy.actor
            actor.set_recovery_adapter_trainable(False)
            try:
                loss_dict = super().update()
            finally:
                actor.set_recovery_adapter_trainable(True)
        elif self.training_stage == "survival":
            actor = self.policy.actor
            actor.set_payload_survival_trainable(False)
            try:
                loss_dict = super().update()
            finally:
                actor.set_payload_survival_trainable(True)
        else:
            loss_dict = super().update()
        self._project_decomposition_trust_region()
        if self.auxiliary_batches <= 0:
            return loss_dict

        observations = self.storage.observations
        if self.storage.num_transitions_per_env < 2:
            return loss_dict
        horizon = min(
            self.transition_horizon_steps,
            self.storage.num_transitions_per_env - 1,
        )
        current_obs = observations[:-horizon].flatten(0, 1)
        next_obs = observations[horizon:].flatten(0, 1)
        current_actions = self.storage.actions[:-horizon].flatten(0, 1)
        done_steps = self.storage.dones.squeeze(-1)
        window_width = done_steps.shape[0] - horizon
        done_windows = torch.stack(
            [
                done_steps[offset : offset + window_width]
                for offset in range(horizon)
            ],
            dim=0,
        ).amax(dim=0)
        current_dones = done_windows.flatten(0, 1)
        transition_count = current_dones.numel()
        if transition_count == 0:
            return loss_dict
        replay_current_context = current_obs[
            self.hierarchy_context_group
        ]
        replay_next_context = next_obs[
            self.hierarchy_context_group
        ]
        event_replay_append_stats = self._append_event_replay(
            replay_current_context,
            replay_next_context,
            current_actions,
            current_dones,
        )
        payload_replay_append_stats = (
            self._append_payload_survival_replay(
                replay_current_context,
                replay_next_context,
                current_actions,
                current_dones,
            )
        )
        valid_indices = torch.arange(
            transition_count, device=current_dones.device
        )

        totals: dict[str, float] = {}
        updates = 0
        normalizer = self.policy.actor_obs_normalizer
        normalizer_was_training = normalizer.training
        normalizer.eval()
        for _ in range(self.auxiliary_batches):
            sample_count = min(
                self.auxiliary_batch_size, valid_indices.numel()
            )
            selection = torch.randint(
                0,
                valid_indices.numel(),
                (sample_count,),
                device=valid_indices.device,
            )
            batch_indices = valid_indices[selection]
            obs_batch = current_obs[batch_indices]
            actions_batch = current_actions[batch_indices]
            next_context = next_obs[
                self.hierarchy_context_group
            ][batch_indices]
            current_context = obs_batch[
                self.hierarchy_context_group
            ]
            dones = current_dones[batch_indices]
            if self.training_stage == "recovery":
                with torch.no_grad():
                    _, stats = self._auxiliary_loss(
                        obs_batch,
                        actions_batch,
                        current_context,
                        next_context,
                        dones,
                    )
            else:
                auxiliary_loss, stats = self._auxiliary_loss(
                    obs_batch,
                    actions_batch,
                    current_context,
                    next_context,
                    dones,
                )
            if self.training_stage == "survival":
                event_replay_task_loss = auxiliary_loss.new_zeros(())
                event_replay_skill_loss = auxiliary_loss.new_zeros(())
                event_replay_parameter_loss = auxiliary_loss.new_zeros(())
                event_replay_stats = {}
            else:
                (
                    event_replay_task_loss,
                    event_replay_skill_loss,
                    event_replay_parameter_loss,
                    event_replay_stats,
                ) = self._event_replay_loss()
            payload_replay_loss, payload_replay_stats = (
                self._payload_survival_replay_loss()
            )
            if self.training_stage == "recovery":
                auxiliary_loss = (
                    self.event_replay_task_coef
                    * event_replay_task_loss
                    + self.event_replay_skill_coef
                    * event_replay_skill_loss
                )
            elif self.training_stage == "survival":
                auxiliary_loss = (
                    auxiliary_loss
                    + self.payload_survival_replay_coef
                    * payload_replay_loss
                )
            else:
                auxiliary_loss = (
                    auxiliary_loss
                    + self.event_replay_task_coef
                    * event_replay_task_loss
                    + self.event_replay_skill_coef
                    * event_replay_skill_loss
                    + self.event_replay_parameter_coef
                    * event_replay_parameter_loss
                    + self.payload_survival_replay_coef
                    * payload_replay_loss
                )
            stats.update(event_replay_stats)
            stats.update(payload_replay_stats)
            stats["event_replay_added"] = auxiliary_loss.new_tensor(
                event_replay_append_stats["added"]
            )
            stats["event_replay_append_task_valid"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats[
                        "task_valid_fraction"
                    ]
                )
            )
            stats["event_replay_task_relabel"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats[
                        "task_relabel_fraction"
                    ]
                )
            )
            stats["event_replay_interaction_relabel"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats[
                        "interaction_relabel_fraction"
                    ]
                )
            )
            stats["event_replay_append_delivery"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats["delivery_fraction"]
                )
            )
            stats["event_replay_append_recovery"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats["recovery_fraction"]
                )
            )
            stats["event_replay_append_payload_recovery"] = (
                auxiliary_loss.new_tensor(
                    event_replay_append_stats[
                        "payload_recovery_fraction"
                    ]
                )
            )
            stats["payload_survival_replay_added"] = (
                auxiliary_loss.new_tensor(
                    payload_replay_append_stats["added"]
                )
            )
            stats["payload_survival_replay_append_drop_fraction"] = (
                auxiliary_loss.new_tensor(
                    payload_replay_append_stats["drop_fraction"]
                )
            )
            self.optimizer.zero_grad()
            self.auxiliary_optimizer.zero_grad()
            self.successor_adapter_optimizer.zero_grad()
            self.payload_survival_optimizer.zero_grad()
            auxiliary_loss.backward()
            if self.training_stage == "recovery":
                gradient_square = auxiliary_loss.new_zeros(())
                head_gradient_square = auxiliary_loss.new_zeros(())
                for module_name in (
                    self.policy.actor.RECOVERY_ADAPTER_MODULE_NAMES
                ):
                    module = getattr(self.policy.actor, module_name)
                    for parameter in module.parameters():
                        if parameter.grad is None:
                            continue
                        contribution = parameter.grad.detach().square().sum()
                        gradient_square = gradient_square + contribution
                        if module_name.endswith("_head"):
                            head_gradient_square = (
                                head_gradient_square + contribution
                            )
                stats["recovery_adapter_gradient_norm"] = (
                    gradient_square.sqrt()
                )
                stats["recovery_adapter_head_gradient_norm"] = (
                    head_gradient_square.sqrt()
                )
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.training_stage == "survival":
                torch.nn.utils.clip_grad_norm_(
                    self._payload_survival_parameters,
                    self.max_grad_norm,
                )
                self.payload_survival_optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self._auxiliary_parameters, self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self._successor_adapter_parameters,
                    self.max_grad_norm,
                )
                self.auxiliary_optimizer.step()
                self.successor_adapter_optimizer.step()
                self._project_decomposition_trust_region()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            updates += 1
        normalizer.train(normalizer_was_training)

        mean_cbf_violation = totals.get(
            "hierarchy_cbf_violation", 0.0
        ) / max(1, updates)
        mean_clf_violation = totals.get(
            "hierarchy_clf_violation", 0.0
        ) / max(1, updates)
        if self.training_stage not in (
            "recovery",
            "survival",
            "motion_selector",
            "interaction_selector",
            "motion_skill",
            "payload_motion",
        ):
            self._update_hierarchy_constraint_duals(
                mean_cbf_violation,
                mean_clf_violation,
            )
        if (
            self.training_stage not in (
                "recovery",
                "motion_selector",
                "interaction_selector",
                "motion_skill",
                "payload_motion",
            )
            and (
                totals.get("payload_survival_support", 0.0)
                + totals.get(
                    "payload_survival_replay_support", 0.0
                )
            )
            > 0.0
        ):
            self.policy.actor.payload_survival_updates.add_(1)
        for key, value in totals.items():
            loss_dict[key] = value / max(1, updates)
        if self.training_stage not in (
            "recovery",
            "survival",
            "motion_selector",
            "interaction_selector",
            "motion_skill",
            "payload_motion",
        ):
            self.policy.tactic_training_updates.add_(1)
        actor = self.policy.actor
        loss_dict["hierarchy_cbf_dual"] = float(
            actor.hierarchy_cbf_dual.item()
        )
        loss_dict["hierarchy_clf_dual"] = float(
            actor.hierarchy_clf_dual.item()
        )
        loss_dict["hierarchy_cbf_violation_ema"] = float(
            actor.hierarchy_cbf_violation_ema.item()
        )
        loss_dict["hierarchy_clf_violation_ema"] = float(
            actor.hierarchy_clf_violation_ema.item()
        )
        physical_stage = int(self._physical_adaptation_stage or 0)
        loss_dict["physical_adaptation_active"] = float(
            physical_stage > 0
        )
        loss_dict["physical_adaptation_stage"] = float(physical_stage)
        loss_dict["hierarchy_only_training"] = float(
            self.training_stage == "upper"
        )
        loss_dict["decomposition_only_training"] = float(
            self.training_stage == "decomposition"
        )
        loss_dict["motion_selector_only_training"] = float(
            self.training_stage == "motion_selector"
        )
        loss_dict["interaction_selector_only_training"] = float(
            self.training_stage == "interaction_selector"
        )
        loss_dict["motion_skill_only_training"] = float(
            self.training_stage == "motion_skill"
        )
        loss_dict["payload_motion_only_training"] = float(
            self.training_stage == "payload_motion"
        )
        decomposition_drifts = self._decomposition_relative_drifts()
        if decomposition_drifts:
            loss_dict["decomposition_max_relative_drift"] = max(
                float(drift.detach().item())
                for drift in decomposition_drifts.values()
            )
            for module_name, drift in decomposition_drifts.items():
                loss_dict[
                    f"decomposition_drift_{module_name}"
                ] = float(drift.detach().item())
        loss_dict["recovery_adapter_only_training"] = float(
            self.training_stage == "recovery"
        )
        loss_dict["auxiliary_learning_rate"] = self.auxiliary_learning_rate
        loss_dict["payload_survival_learning_rate"] = (
            self.payload_survival_learning_rate
        )
        loss_dict["physical_core_learning_rate"] = self.optimizer.param_groups[
            0
        ]["lr"]
        loss_dict["physical_adapter_learning_rate"] = (
            self.optimizer.param_groups[1]["lr"]
        )
        if self._skill_information_history:
            loss_dict["skill_information_reward"] = sum(
                self._skill_information_history
            ) / len(self._skill_information_history)
            self._skill_information_history.clear()
        if self._task_information_history:
            loss_dict["task_information_reward"] = sum(
                self._task_information_history
            ) / len(self._task_information_history)
            self._task_information_history.clear()
        return loss_dict
