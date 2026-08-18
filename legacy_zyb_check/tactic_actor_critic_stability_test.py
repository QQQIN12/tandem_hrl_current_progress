"""Factorized actor-critic for learned task and skill composition."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rsl_rl.modules import ActorCritic

from ..tactic_layout import (
    ACTION_LAYOUT,
    BASE_HEIGHT_INDEX,
    BASE_TILT_INDEX,
    BASE_VX_INDEX,
    BASE_WZ_INDEX,
    CAPTURE_BARRIER_GAIN,
    CAPTURE_CENTER_RADIUS,
    CAPTURE_FINGER_RADIUS,
    CAPTURE_TCP_RADIUS,
    CLF_DECREASE_INDEX,
    COMMAND_VX_INDEX,
    COMMAND_VY_INDEX,
    COMMAND_WZ_INDEX,
    EXECUTED_OBJECT_INDEX,
    EXECUTED_SKILL_INDEX,
    EXECUTED_TASK_INDEX,
    GLOBAL_CONTEXT_DIM,
    HIERARCHY_CONTEXT_DIM,
    INTERACTION_HARD_CONTROL_FLOOR,
    INTERACTION_SKILL_COUNT,
    CURRICULUM_LEVEL_INDEX,
    DISTURBANCE_QUALITY_INDEX,
    MORPHOLOGY_SLICE,
    MOTION_EXECUTION_EFFECT_DIM,
    MOTION_SKILL_COUNT,
    PAYLOAD_SKILL_PREVIEW_RESERVE,
    PAYLOAD_SKILL_RISK_GAIN,
    PAYLOAD_SKILL_SAFETY_RESERVE,
    PAYLOAD_SKILL_SWITCH_MARGIN,
    PAYLOAD_TRANSIENT_BRAKE_GAIN,
    PAYLOAD_TRANSIENT_DISTANCE_SCALE,
    PAYLOAD_TRANSIENT_GATE_GAIN,
    PAYLOAD_TRANSIENT_SPEED_FLOOR,
    PAYLOAD_TRANSIENT_SPEED_RANGE,
    PAYLOAD_TRANSIENT_SPEED_SCALE,
    PAYLOAD_TRANSIENT_TRACKING_ALLOWANCE,
    PAYLOAD_TRANSIENT_TRACKING_SCALE,
    PREVIEW_MARGIN_INDEX,
    RELEASE_HOVER_HEIGHT,
    RELEASE_READINESS_GAIN,
    RELEASE_READINESS_THRESHOLD,
    RELEASE_SETTLE_VX,
    RELEASE_SETTLE_WZ,
    RELEASE_TARGET_GAIN,
    RELEASE_TARGET_RADIUS,
    RELEASE_TRANSPORT_GAIN,
    RELEASE_TRANSPORT_THRESHOLD,
    RELEASE_VERTICAL_GAIN,
    RELEASE_VERTICAL_TOLERANCE,
    RELEASE_CBF_TRANSIENT_SLACK,
    SAFETY_MARGIN_INDEX,
    SECURE_ENTRY_BARRIER_GAIN,
    SECURE_ENTRY_CENTER_RADIUS,
    SECURE_ENTRY_FINGER_RADIUS,
    SECURE_ENTRY_TCP_RADIUS,
    SECURE_CBF_TRANSIENT_SLACK,
    SKILL_EFFECT_DIM,
    SKILL_OUTCOME_DIM,
    SUPPORT_COUNT_INDEX,
    TASK_SLOT_COUNT,
    TASK_SLOT_AVAILABLE_INDEX,
    TASK_SLOT_CARRYING_INDEX,
    TASK_SLOT_COMPLETED_INDEX,
    TASK_SLOT_DELIVERY_TYPE_INDEX,
    TASK_SLOT_DISTANCE_INDEX,
    TASK_SLOT_FEATURE_DIM,
    TASK_SLOT_INTERACTION_STATE_SLICE,
    TASK_SLOT_LEFT_FINGER_DELTA_SLICE,
    TASK_SLOT_HEADING_INDEX,
    TASK_SLOT_MANIPULATION_TYPE_INDEX,
    TASK_SLOT_OBJECT_DELTA_SLICE,
    TASK_SLOT_REACHABILITY_INDEX,
    TASK_SLOT_RECOVERY_TYPE_INDEX,
    TASK_SLOT_REMAINING_PROGRESS_INDEX,
    TASK_SLOT_REQUIRED_INDEX,
    TASK_SLOT_RIGHT_FINGER_DELTA_SLICE,
    TASK_SLOT_TARGET_DELTA_SLICE,
    TASK_OUTCOME_DIM,
    TERMINATION_STATE_SLICE,
    WHEEL_RADIUS_INDEX,
)


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def _mlp(input_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for next_width in hidden_dims:
        layers.extend((nn.Linear(width, next_width), _activation(activation)))
        width = next_width
    return nn.Sequential(*layers)


class TACTICActor(nn.Module):
    """Autoregressive task, skill, and physical policy.

    Task slots are encoded with shared weights, so a slot denotes a goal from
    the current mission rather than a fixed class tied to one benchmark.
    """

    PHYSICAL_EXECUTOR_MODULE_NAMES = (
        "physical_backbone",
        "physical_head",
        "physical_conditioner",
        "film_head",
        "physical_residual_head",
        "support_skill_encoder",
        "support_reference_head",
        "support_gate_head",
        "support_residual_head",
        "wheel_residual_encoder",
        "wheel_residual_head",
        "wheel_skill_gate_head",
        "gripper_head",
    )
    PHYSICAL_EXECUTOR_PARAMETER_NAMES = (
        "motion_support_basis",
        "support_gate_logit",
        "wheel_skill_gate_logit",
        "embodiment_motion_basis",
        "motion_action_capacity",
        "motion_kinematic_gain",
        "wheel_breakaway_action",
        "interaction_gripper_basis",
    )
    RECOVERY_ADAPTER_MODULE_NAMES = (
        "recovery_adapter_encoder",
        "recovery_task_adapter_head",
        "recovery_motion_adapter_head",
        "recovery_interaction_adapter_head",
    )
    DECOMPOSITION_MODULE_NAMES = (
        "task_detail_encoder",
        "task_continuous_head",
        "motion_skill_logits_head",
        "interaction_skill_logits_head",
        "skill_detail_encoder",
        "skill_continuous_head",
        "interaction_skill_effect_head",
    )
    MOTION_SELECTOR_MODULE_NAMES = ("motion_skill_logits_head",)
    INTERACTION_SELECTOR_MODULE_NAMES = (
        "interaction_skill_logits_head",
    )
    MOTION_SKILL_MODULE_NAMES = (
        "motion_skill_logits_head",
        "skill_continuous_head",
    )
    PAYLOAD_MOTION_MODULE_NAMES = (
        "motion_skill_logits_head",
        "skill_continuous_head",
        "wheel_residual_encoder",
        "wheel_residual_head",
        "wheel_skill_gate_head",
    )
    PAYLOAD_MOTION_PARAMETER_NAMES = (
        "wheel_skill_gate_logit",
        "embodiment_motion_basis",
        "embodiment_response_matrix",
        "motion_action_capacity",
        "motion_kinematic_gain",
        "wheel_breakaway_action",
    )

    def __init__(
        self,
        num_obs: int,
        context_dim: int,
        num_actions: int,
        physical_hidden_dims: list[int],
        activation: str,
        mission_latent_dim: int,
        slot_latent_dim: int,
        task_embedding_dim: int,
        skill_embedding_dim: int,
        conditioner_gain: float,
        physical_residual_gain: float,
        support_skill_residual_gain: float,
        support_skill_adaptation_gain: float,
        physical_core_obs_dim: int,
        skill_feasibility_gain: float,
        skill_effect_scale: float,
        interaction_gripper_gain: float,
        interaction_gripper_phase_gain: float,
        interaction_gripper_release_gain: float,
        interaction_gripper_residual_limit: float,
        interaction_phase_prior_gain: float,
        release_target_radius: float,
        task_residual_scale: float,
        transition_latent_dim: int,
        transition_temperature: float,
        task_affordance_gain: float,
        task_outcome_gain: float,
        recovery_task_margin_gain: float,
        recovery_adapter_task_gain: float,
        recovery_adapter_motion_gain: float,
        recovery_adapter_interaction_gain: float,
        task_outcome_warmup_updates: int,
        skill_outcome_gain: float,
        payload_survival_gain: float,
        payload_survival_warmup_updates: int,
        skill_effect_gain: float,
        constraint_utility_gain: float,
        motion_objective_gain: float,
        motion_execution_utility_gain: float,
        embodiment_response_selection_gain: float,
        embodiment_response_prior_confidence: float,
        wheel_action_scale: float,
        wheel_track_width: float,
        wheel_breakaway_action: float,
        wheel_turn_breakaway_action: float,
        wheel_residual_gain: float,
        wheel_action_limit: float,
    ):
        super().__init__()
        if num_actions != ACTION_LAYOUT.total_dim:
            raise ValueError(
                f"TACTIC action layout expects {ACTION_LAYOUT.total_dim} actions, got {num_actions}"
            )
        if context_dim != HIERARCHY_CONTEXT_DIM:
            raise ValueError(
                f"TACTIC context expects {HIERARCHY_CONTEXT_DIM} values, got {context_dim}"
            )

        self.physical_action_dim = ACTION_LAYOUT.physical_dim
        self.context_dim = context_dim
        self.conditioner_gain = float(conditioner_gain)
        self.physical_residual_gain = float(physical_residual_gain)
        self.support_skill_residual_gain = float(
            support_skill_residual_gain
        )
        self.support_skill_adaptation_gain = float(
            support_skill_adaptation_gain
        )
        self.physical_core_obs_dim = int(physical_core_obs_dim)
        # When enabled, the first 16 actuator outputs are the migrated ZYB-v0
        # executor exactly.  Hierarchy-conditioned FiLM, support and wheel
        # decoders are bypassed so an upper-stage experiment cannot silently
        # replace a validated locomotion teacher.
        self.stability_teacher_only = False
        self.skill_feasibility_gain = float(skill_feasibility_gain)
        self.skill_effect_scale = float(skill_effect_scale)
        self.interaction_gripper_gain = float(interaction_gripper_gain)
        self.interaction_gripper_phase_gain = float(
            interaction_gripper_phase_gain
        )
        self.interaction_gripper_release_gain = float(
            interaction_gripper_release_gain
        )
        self.interaction_gripper_residual_limit = float(
            interaction_gripper_residual_limit
        )
        if self.interaction_gripper_phase_gain <= 0.0:
            raise ValueError(
                "interaction_gripper_phase_gain must be positive"
            )
        if self.interaction_gripper_release_gain <= 0.0:
            raise ValueError(
                "interaction_gripper_release_gain must be positive"
            )
        if self.interaction_gripper_residual_limit < 0.0:
            raise ValueError(
                "interaction_gripper_residual_limit must be nonnegative"
            )
        self.interaction_phase_prior_gain = float(
            interaction_phase_prior_gain
        )
        self.release_target_radius = float(release_target_radius)
        if self.release_target_radius <= 0.0:
            raise ValueError("release_target_radius must be positive")
        self.task_residual_scale = float(task_residual_scale)
        self.transition_temperature = float(transition_temperature)
        self.task_affordance_gain = float(task_affordance_gain)
        self.task_outcome_gain = float(task_outcome_gain)
        self.recovery_task_margin_gain = float(recovery_task_margin_gain)
        if self.recovery_task_margin_gain < 0.0:
            raise ValueError(
                "recovery_task_margin_gain must be nonnegative"
            )
        self.recovery_adapter_task_gain = float(
            recovery_adapter_task_gain
        )
        self.recovery_adapter_motion_gain = float(
            recovery_adapter_motion_gain
        )
        self.recovery_adapter_interaction_gain = float(
            recovery_adapter_interaction_gain
        )
        if self.recovery_adapter_task_gain < 0.0:
            raise ValueError(
                "recovery_adapter_task_gain must be nonnegative"
            )
        if self.recovery_adapter_motion_gain < 0.0:
            raise ValueError(
                "recovery_adapter_motion_gain must be nonnegative"
            )
        if self.recovery_adapter_interaction_gain < 0.0:
            raise ValueError(
                "recovery_adapter_interaction_gain must be nonnegative"
            )
        self.task_outcome_warmup_updates = max(
            1, int(task_outcome_warmup_updates)
        )
        self.task_outcome_maturity = 0.0
        self.motion_execution_maturity = 0.0
        self.skill_outcome_gain = float(skill_outcome_gain)
        self.payload_survival_gain = float(payload_survival_gain)
        if self.payload_survival_gain < 0.0:
            raise ValueError("payload_survival_gain must be nonnegative")
        self.payload_survival_warmup_updates = max(
            1, int(payload_survival_warmup_updates)
        )
        self.payload_survival_maturity = 0.0
        self.payload_survival_control_enabled = True
        self.skill_effect_gain = float(skill_effect_gain)
        self.constraint_utility_gain = float(constraint_utility_gain)
        self.motion_objective_gain = float(motion_objective_gain)
        self.motion_execution_utility_gain = float(
            motion_execution_utility_gain
        )
        self.embodiment_response_selection_gain = float(
            embodiment_response_selection_gain
        )
        if self.embodiment_response_selection_gain < 0.0:
            raise ValueError(
                "embodiment_response_selection_gain must be nonnegative"
            )
        self.embodiment_response_prior_confidence = float(
            embodiment_response_prior_confidence
        )
        if not 0.0 <= self.embodiment_response_prior_confidence <= 1.0:
            raise ValueError(
                "embodiment_response_prior_confidence must be in [0, 1]"
            )
        self.wheel_action_scale = float(wheel_action_scale)
        self.wheel_track_width = float(wheel_track_width)
        self.wheel_residual_gain = float(wheel_residual_gain)
        self.wheel_action_limit = float(wheel_action_limit)
        if self.wheel_action_scale <= 0.0:
            raise ValueError("wheel_action_scale must be positive")
        if self.wheel_track_width <= 0.0:
            raise ValueError("wheel_track_width must be positive")
        if wheel_breakaway_action <= 0.0:
            raise ValueError("wheel_breakaway_action must be positive")
        if wheel_turn_breakaway_action <= 0.0:
            raise ValueError("wheel_turn_breakaway_action must be positive")
        if self.wheel_action_limit <= 0.0:
            raise ValueError("wheel_action_limit must be positive")
        if not 0 < self.physical_core_obs_dim <= num_obs:
            raise ValueError("Invalid TACTIC physical core observation width")

        # The physical core keeps the original ZYB-v0 observation contract.
        self.physical_backbone = _mlp(
            self.physical_core_obs_dim, physical_hidden_dims, activation
        )
        physical_latent_dim = physical_hidden_dims[-1]
        self.physical_head = nn.Linear(physical_latent_dim, ACTION_LAYOUT.physical_dim)

        self.global_encoder = _mlp(
            GLOBAL_CONTEXT_DIM, [mission_latent_dim, mission_latent_dim], activation
        )
        self.morphology_encoder = _mlp(
            MORPHOLOGY_SLICE.stop - MORPHOLOGY_SLICE.start,
            [32, 32],
            activation,
        )
        self.morphology_fusion = _mlp(
            mission_latent_dim + 32,
            [mission_latent_dim],
            activation,
        )
        self.slot_encoder = _mlp(
            TASK_SLOT_FEATURE_DIM, [slot_latent_dim, slot_latent_dim], activation
        )
        # Two fingertip-to-object vectors, their invariant distances, gripper
        # aperture, and contact symmetry form a robot-object relation packet.
        self.relation_encoder = _mlp(
            12, [slot_latent_dim, slot_latent_dim], activation
        )
        self.slot_relation_fusion = _mlp(
            2 * slot_latent_dim,
            [slot_latent_dim],
            activation,
        )
        graph_heads = 4 if slot_latent_dim % 4 == 0 else 1
        self.task_graph_attention = nn.MultiheadAttention(
            slot_latent_dim,
            graph_heads,
            batch_first=True,
        )
        self.task_graph_norm_1 = nn.LayerNorm(slot_latent_dim)
        self.task_graph_ffn = _mlp(
            slot_latent_dim,
            [slot_latent_dim, slot_latent_dim],
            activation,
        )
        self.task_graph_norm_2 = nn.LayerNorm(slot_latent_dim)
        self.task_query = nn.Linear(mission_latent_dim, slot_latent_dim)
        self.task_key = nn.Linear(slot_latent_dim, slot_latent_dim)
        self.task_bias = nn.Linear(slot_latent_dim, 1)
        self.mission_fusion = _mlp(
            mission_latent_dim + slot_latent_dim,
            [mission_latent_dim],
            activation,
        )
        self.task_outcome_encoder = _mlp(
            mission_latent_dim + slot_latent_dim,
            [mission_latent_dim],
            activation,
        )
        self.task_outcome_head = nn.Linear(
            mission_latent_dim, TASK_OUTCOME_DIM
        )
        self.task_outcome_confidence_head = nn.Linear(
            mission_latent_dim, 1
        )
        self.task_constraint_multiplier_head = nn.Linear(
            mission_latent_dim, 1
        )
        self.register_buffer(
            "task_outcome_weights",
            torch.tensor((0.38, 0.22, 0.18, 0.22)),
        )
        self.recovery_adapter_feature_dim = 11
        self.recovery_adapter_encoder = _mlp(
            self.recovery_adapter_feature_dim,
            [32, 32],
            activation,
        )
        self.recovery_task_adapter_head = nn.Linear(32, 1)
        self.recovery_motion_adapter_head = nn.Linear(
            32, MOTION_SKILL_COUNT
        )
        self.recovery_interaction_adapter_head = nn.Linear(
            32, INTERACTION_SKILL_COUNT
        )

        task_block_dim = (
            ACTION_LAYOUT.object_dim
            + ACTION_LAYOUT.task_subgoal_dim
            + 1
        )
        self.task_detail_encoder = _mlp(
            mission_latent_dim + slot_latent_dim,
            [mission_latent_dim],
            activation,
        )
        self.task_continuous_head = nn.Linear(mission_latent_dim, task_block_dim)
        self.object_embedding = nn.Embedding(
            ACTION_LAYOUT.object_dim, task_embedding_dim
        )
        self.task_conditioner = _mlp(
            slot_latent_dim
            + task_embedding_dim
            + ACTION_LAYOUT.task_subgoal_dim
            + 1,
            [mission_latent_dim, mission_latent_dim],
            activation,
        )

        if skill_embedding_dim % 2 != 0:
            raise ValueError("TACTIC skill embedding width must be even")
        self.skill_factor_dim = skill_embedding_dim // 2
        self.motion_skill_embedding = nn.Embedding(
            MOTION_SKILL_COUNT, self.skill_factor_dim
        )
        self.interaction_skill_embedding = nn.Embedding(
            INTERACTION_SKILL_COUNT, self.skill_factor_dim
        )
        self.skill_encoder = _mlp(
            2 * mission_latent_dim,
            [mission_latent_dim, mission_latent_dim],
            activation,
        )
        self.skill_outcome_encoder = _mlp(
            2 * mission_latent_dim + skill_embedding_dim,
            [mission_latent_dim],
            activation,
        )
        self.skill_outcome_head = nn.Linear(
            mission_latent_dim, SKILL_OUTCOME_DIM
        )
        # Kept so checkpoints from the initial calibration experiments load.
        self.skill_survival_head = nn.Linear(mission_latent_dim, 1)
        self.payload_survival_encoder = _mlp(
            2 * mission_latent_dim + skill_embedding_dim,
            [mission_latent_dim, mission_latent_dim],
            activation,
        )
        self.payload_survival_head = nn.Linear(mission_latent_dim, 1)
        self.register_buffer(
            "payload_survival_updates",
            torch.zeros((), dtype=torch.long),
        )
        self.skill_effect_encoder = _mlp(
            2 * mission_latent_dim + skill_embedding_dim,
            [mission_latent_dim, mission_latent_dim],
            activation,
        )
        self.skill_effect_head = nn.Linear(
            mission_latent_dim, SKILL_EFFECT_DIM
        )
        self.skill_effect_confidence_head = nn.Linear(
            mission_latent_dim, 1
        )
        self.skill_constraint_multiplier_head = nn.Sequential(
            nn.Linear(2 * mission_latent_dim, mission_latent_dim),
            _activation(activation),
            nn.Linear(mission_latent_dim, 4),
        )
        self.register_buffer(
            "control_constraint_floor",
            torch.tensor((0.30, 0.28, 0.28, 0.25)),
        )
        self.register_buffer(
            "hierarchy_cbf_dual",
            torch.tensor(0.50),
        )
        self.register_buffer(
            "hierarchy_clf_dual",
            torch.tensor(0.35),
        )
        self.register_buffer(
            "hierarchy_cbf_violation_ema",
            torch.zeros(()),
        )
        self.register_buffer(
            "hierarchy_clf_violation_ema",
            torch.zeros(()),
        )
        self.register_buffer(
            "hierarchy_constraint_updates",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "skill_outcome_weights",
            torch.tensor((0.35, 0.30, 0.20, 0.15)),
        )
        self.motion_skill_logits_head = nn.Linear(
            mission_latent_dim, MOTION_SKILL_COUNT
        )
        self.interaction_skill_logits_head = nn.Linear(
            mission_latent_dim, INTERACTION_SKILL_COUNT
        )
        self.skill_feasibility_head = nn.Linear(
            mission_latent_dim, ACTION_LAYOUT.skill_dim
        )
        self.motion_objective_demand_head = nn.Linear(
            mission_latent_dim, 4
        )
        self.motion_objective_basis = nn.Parameter(
            torch.tensor(
                (
                    (-1.0, -1.0, 2.5, 0.4),
                    (2.5, 0.2, -0.5, -1.0),
                    (0.2, 2.5, -0.2, -1.0),
                    (0.7, 0.5, 0.4, 2.5),
                )
            )
        )
        self.skill_detail_encoder = _mlp(
            mission_latent_dim + skill_embedding_dim,
            [mission_latent_dim],
            activation,
        )
        self.skill_continuous_head = nn.Linear(
            mission_latent_dim, ACTION_LAYOUT.skill_param_dim + 1
        )
        self.motion_skill_effect_head = nn.Linear(
            self.skill_factor_dim,
            ACTION_LAYOUT.skill_param_dim,
            bias=False,
        )
        self.interaction_skill_effect_head = nn.Linear(
            self.skill_factor_dim,
            ACTION_LAYOUT.skill_param_dim,
            bias=False,
        )
        self.register_buffer(
            "motion_parameter_mask",
            torch.tensor((1.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        )
        self.register_buffer(
            "interaction_parameter_mask",
            torch.tensor((0.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
        )
        self.skill_effect_gate_head = nn.Linear(mission_latent_dim, 2)
        self.skill_conditioner = _mlp(
            skill_embedding_dim + ACTION_LAYOUT.skill_param_dim + 1,
            [mission_latent_dim],
            activation,
        )

        transition_input_dim = 2 * GLOBAL_CONTEXT_DIM
        self.transition_encoder = _mlp(
            transition_input_dim,
            [mission_latent_dim, transition_latent_dim],
            activation,
        )
        self.transition_slot_encoder = _mlp(
            2 * TASK_SLOT_FEATURE_DIM,
            [slot_latent_dim, slot_latent_dim],
            activation,
        )
        self.task_transition_query = nn.Linear(
            transition_latent_dim, transition_latent_dim
        )
        self.task_transition_key = nn.Linear(
            slot_latent_dim, transition_latent_dim
        )
        self.motion_transition_indices = (
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            32,
            33,
            39,
            40,
        )
        self.interaction_transition_indices = (
            19,
            20,
            21,
            22,
            23,
            36,
            37,
            38,
            39,
            40,
            41,
            46,
        )
        self.motion_transition_query = _mlp(
            2 * len(self.motion_transition_indices) + slot_latent_dim,
            [transition_latent_dim, transition_latent_dim],
            activation,
        )
        self.interaction_transition_query = _mlp(
            2 * len(self.interaction_transition_indices) + slot_latent_dim,
            [transition_latent_dim, transition_latent_dim],
            activation,
        )
        self.motion_transition_key = nn.Linear(
            self.skill_factor_dim, transition_latent_dim
        )
        self.interaction_transition_key = nn.Linear(
            self.skill_factor_dim, transition_latent_dim
        )

        conditioner_input_dim = (
            mission_latent_dim + mission_latent_dim + mission_latent_dim
        )
        self.physical_conditioner = _mlp(
            conditioner_input_dim,
            [physical_latent_dim],
            activation,
        )
        self.film_head = nn.Linear(physical_latent_dim, 2 * physical_latent_dim)
        self.physical_residual_head = nn.Linear(
            physical_latent_dim, ACTION_LAYOUT.physical_dim
        )
        # The support branch belongs to the shared motion skill decoder.  Its
        # prototypes initialize a four-wheel stance observed to be reachable
        # on B2W, while the task/skill state learns payload- and margin-aware
        # adaptations.  A zero motion command keeps the migrated policy exact.
        self.support_state_dim = 12
        self.support_skill_encoder = _mlp(
            2 * physical_latent_dim + self.support_state_dim,
            [128, 128],
            activation,
        )
        self.support_reference_head = nn.Linear(
            128, MOTION_SKILL_COUNT * 12
        )
        self.support_gate_head = nn.Linear(128, MOTION_SKILL_COUNT)
        self.support_residual_head = nn.Linear(128, 12)
        support_reference = torch.tensor(
            (
                0.0,
                0.0,
                0.0,
                0.0,
                -0.1785714286,
                -0.1785714286,
                -0.1785714286,
                -0.1785714286,
                0.0,
                0.0,
                0.3571428571,
                0.3571428571,
            )
        )
        self.motion_support_basis = nn.Parameter(
            support_reference.repeat(MOTION_SKILL_COUNT, 1)
        )
        self.support_gate_logit = nn.Parameter(
            torch.full((MOTION_SKILL_COUNT,), -4.0)
        )
        self.wheel_feature_dim = 14
        self.wheel_residual_encoder = _mlp(
            physical_latent_dim + self.wheel_feature_dim,
            [64, 64],
            activation,
        )
        self.wheel_residual_head = nn.Linear(64, 4)
        self.wheel_skill_gate_head = nn.Linear(
            64, MOTION_SKILL_COUNT
        )
        self.wheel_skill_gate_logit = nn.Parameter(
            torch.full((MOTION_SKILL_COUNT,), -0.50)
        )
        # These bases come from a signed action-response sweep on the B2W
        # simulator.  They seed a reachable chart; selection, interpolation,
        # gain, and residual adaptation remain learned with the skill policy.
        self.embodiment_motion_basis = nn.Parameter(
            torch.tensor(
                (
                    (
                        (0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 0.0),
                    ),
                    (
                        (1.0, 1.0, -1.0, 1.0),
                        (-1.0, 1.0, 1.0, 1.0),
                        (1.0, -1.0, -1.0, 1.0),
                    ),
                    (
                        (-1.0, 1.0, -1.0, 1.0),
                        (-1.0, 1.0, 1.0, 1.0),
                        (1.0, -1.0, -1.0, 1.0),
                    ),
                    (
                        (1.0, 1.0, 1.0, 1.0),
                        (-1.0, 1.0, 1.0, 1.0),
                        (1.0, -1.0, -1.0, 1.0),
                    ),
                )
            )
        )
        identified_drive_capacity = min(
            float(self.wheel_action_limit),
            max(
                float(wheel_breakaway_action),
                0.375
                * float(self.wheel_action_scale)
                * float(self.wheel_action_limit),
            ),
        )
        identified_turn_capacity = min(
            float(self.wheel_action_limit),
            max(
                float(wheel_turn_breakaway_action),
                0.30
                * float(self.wheel_action_scale)
                * float(self.wheel_action_limit),
            ),
        )
        self.motion_action_capacity = nn.Parameter(
            torch.tensor(
                (identified_drive_capacity, identified_turn_capacity)
            )
        )
        # A short signed sweep identifies the platform response before policy
        # training.  The matrix remains trainable, so the same skill interface
        # can recalibrate itself after a morphology or terrain change.
        response_matrix = torch.tensor(
            (
                (0.0024640, -0.0153460),
                (0.0110827, 0.0154233),
                (-0.0086187, 0.0130750),
                (0.0000000, -0.0092603),
            )
        )
        self.embodiment_response_matrix = nn.Parameter(
            response_matrix.clone()
        )
        self.register_buffer(
            "embodiment_response_anchor", response_matrix.clone()
        )
        self.register_buffer(
            "embodiment_response_action", torch.tensor(18.0)
        )
        self.motion_execution_feature_dim = 16
        self.motion_execution_encoder = _mlp(
            self.motion_execution_feature_dim,
            [64, 64],
            activation,
        )
        self.motion_execution_head = nn.Linear(
            64, MOTION_EXECUTION_EFFECT_DIM
        )
        self.motion_execution_confidence_head = nn.Linear(64, 1)
        self.motion_kinematic_gain = nn.Parameter(
            torch.tensor(
                (
                    (0.55, 0.55),
                    (1.30, 0.85),
                    (0.75, 1.30),
                    (0.95, 0.45),
                )
            )
        )
        # The upper policy emits geometry, while the lower policy works in
        # embodiment-feasible velocity residuals.  These capacities are
        # trainable so the shared skill bank can adapt to another platform.
        self.motion_target_capacity = nn.Parameter(
            torch.tensor((0.20, 0.18))
        )
        self.wheel_breakaway_action = nn.Parameter(
            torch.tensor(
                (
                    float(wheel_breakaway_action),
                    float(wheel_turn_breakaway_action),
                )
            )
        )
        self.control_prediction_head = nn.Sequential(
            nn.Linear(conditioner_input_dim, mission_latent_dim),
            _activation(activation),
            nn.Linear(mission_latent_dim, 4),
            nn.Sigmoid(),
        )
        self.gripper_head = nn.Linear(conditioner_input_dim, 1)
        # Kept in the state dict for compatibility with earlier unified
        # checkpoints.  Its secure entry provides a bounded gain correction;
        # the phase direction itself is fixed by the option semantics.
        self.interaction_gripper_basis = nn.Parameter(
            torch.tensor((0.0, 4.0, 0.0))
        )

        # Baseline migration starts with exactly its 16 physical outputs.
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)
        nn.init.zeros_(self.physical_residual_head.weight)
        nn.init.zeros_(self.physical_residual_head.bias)
        nn.init.zeros_(self.support_reference_head.weight)
        nn.init.zeros_(self.support_reference_head.bias)
        nn.init.zeros_(self.support_gate_head.weight)
        nn.init.zeros_(self.support_gate_head.bias)
        nn.init.zeros_(self.support_residual_head.weight)
        nn.init.zeros_(self.support_residual_head.bias)
        nn.init.zeros_(self.wheel_residual_head.weight)
        nn.init.zeros_(self.wheel_residual_head.bias)
        nn.init.zeros_(self.wheel_skill_gate_head.weight)
        nn.init.zeros_(self.wheel_skill_gate_head.bias)
        nn.init.zeros_(self.motion_execution_head.weight)
        nn.init.zeros_(self.motion_execution_head.bias)
        nn.init.zeros_(self.motion_execution_confidence_head.weight)
        nn.init.constant_(self.motion_execution_confidence_head.bias, -2.0)
        nn.init.zeros_(self.recovery_task_adapter_head.weight)
        nn.init.zeros_(self.recovery_task_adapter_head.bias)
        nn.init.zeros_(self.recovery_motion_adapter_head.weight)
        nn.init.zeros_(self.recovery_motion_adapter_head.bias)
        nn.init.zeros_(self.recovery_interaction_adapter_head.weight)
        nn.init.zeros_(self.recovery_interaction_adapter_head.bias)
        for head in (self.task_continuous_head, self.skill_continuous_head):
            nn.init.orthogonal_(head.weight, gain=0.02)
            nn.init.zeros_(head.bias)
        for head in (
            self.motion_skill_logits_head,
            self.interaction_skill_logits_head,
        ):
            nn.init.orthogonal_(head.weight, gain=0.15)
            nn.init.zeros_(head.bias)
        for head in (self.task_query, self.task_key):
            nn.init.orthogonal_(head.weight, gain=0.05)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.task_bias.weight)
        nn.init.zeros_(self.task_bias.bias)
        nn.init.zeros_(self.skill_feasibility_head.weight)
        nn.init.zeros_(self.skill_feasibility_head.bias)
        nn.init.zeros_(self.motion_objective_demand_head.weight)
        nn.init.zeros_(self.motion_objective_demand_head.bias)
        nn.init.zeros_(self.task_outcome_head.weight)
        nn.init.zeros_(self.task_outcome_head.bias)
        nn.init.zeros_(self.task_outcome_confidence_head.weight)
        nn.init.constant_(self.task_outcome_confidence_head.bias, -2.0)
        nn.init.zeros_(self.task_constraint_multiplier_head.weight)
        nn.init.constant_(self.task_constraint_multiplier_head.bias, -2.0)
        nn.init.zeros_(self.skill_outcome_head.weight)
        nn.init.zeros_(self.skill_outcome_head.bias)
        nn.init.zeros_(self.skill_survival_head.weight)
        nn.init.zeros_(self.skill_survival_head.bias)
        nn.init.zeros_(self.payload_survival_head.weight)
        nn.init.zeros_(self.payload_survival_head.bias)
        nn.init.zeros_(self.skill_effect_head.weight)
        nn.init.zeros_(self.skill_effect_head.bias)
        nn.init.zeros_(self.skill_effect_confidence_head.weight)
        nn.init.constant_(self.skill_effect_confidence_head.bias, -2.0)
        nn.init.zeros_(self.skill_constraint_multiplier_head[-1].weight)
        nn.init.constant_(
            self.skill_constraint_multiplier_head[-1].bias, -2.0
        )
        nn.init.orthogonal_(self.motion_skill_effect_head.weight, gain=0.85)
        nn.init.orthogonal_(
            self.interaction_skill_effect_head.weight, gain=0.85
        )
        nn.init.zeros_(self.skill_effect_gate_head.weight)
        nn.init.zeros_(self.skill_effect_gate_head.bias)
        nn.init.zeros_(self.gripper_head.weight)
        nn.init.constant_(self.gripper_head.bias, -3.0)
        # A negative gripper logit corresponds to an open hand.  Locomotion
        # therefore begins with the same arm load as ZYB-v0.
        with torch.no_grad():
            self.physical_head.weight[16].zero_()
            self.physical_head.bias[16] = -3.0
            self.task_continuous_head.bias[-1] = -2.5
            self.skill_continuous_head.bias[-1] = -0.4

        self.last_control_prediction: torch.Tensor | None = None
        self.last_slot_latent: torch.Tensor | None = None
        self.last_skill_feasibility: torch.Tensor | None = None
        self.last_interaction_active: torch.Tensor | None = None
        self.last_interaction_phase_target: torch.Tensor | None = None
        self.last_interaction_phase_raw_probability: torch.Tensor | None = None
        self.last_interaction_phase_projected_probability: (
            torch.Tensor | None
        ) = None
        self.last_interaction_capture_feasibility: torch.Tensor | None = None
        self.last_interaction_secure_entry_feasibility: (
            torch.Tensor | None
        ) = None
        self.last_interaction_release_feasibility: torch.Tensor | None = None
        self.last_interaction_release_frontier: torch.Tensor | None = None
        self.last_interaction_release_target_gate: torch.Tensor | None = None
        self.last_interaction_release_vertical_gate: torch.Tensor | None = None
        self.last_interaction_release_transport_gate: torch.Tensor | None = None
        self.last_interaction_release_control_gate: torch.Tensor | None = None
        self.last_interaction_hold_evidence: torch.Tensor | None = None
        self.last_interaction_center_error: torch.Tensor | None = None
        self.last_interaction_finger_distance: torch.Tensor | None = None
        self.last_interaction_tcp_distance: torch.Tensor | None = None
        self.last_gripper_semantic_logit: torch.Tensor | None = None
        self.last_gripper_residual: torch.Tensor | None = None
        self.last_gripper_residual_authority: torch.Tensor | None = None
        self.last_gripper_logit: torch.Tensor | None = None
        self.last_global_context: torch.Tensor | None = None
        self.last_raw_task_slots: torch.Tensor | None = None
        self.last_task_outcomes: torch.Tensor | None = None
        self.last_task_outcome_confidence: torch.Tensor | None = None
        self.last_task_constraint_multiplier: torch.Tensor | None = None
        self.last_task_constraint_violation: torch.Tensor | None = None
        self.last_skill_outcomes: torch.Tensor | None = None
        self.last_skill_survival: torch.Tensor | None = None
        self.last_skill_effects: torch.Tensor | None = None
        self.last_skill_effect_confidence: torch.Tensor | None = None
        self.last_skill_effect_utility: torch.Tensor | None = None
        self.last_skill_constraint_multipliers: torch.Tensor | None = None
        self.last_skill_constraint_threshold: torch.Tensor | None = None
        self.last_skill_constraint_violation: torch.Tensor | None = None
        self.last_motion_objective_demand: torch.Tensor | None = None
        self.last_motion_objective_scores: torch.Tensor | None = None
        self.last_task_outcome_utility: torch.Tensor | None = None
        self.last_task_grounded_utility: torch.Tensor | None = None
        self.last_task_outcome_reliability: torch.Tensor | None = None
        self.last_task_blended_utility: torch.Tensor | None = None
        self.last_recovery_task_preference: torch.Tensor | None = None
        self.last_recovery_task_adapter: torch.Tensor | None = None
        self.last_recovery_motion_adapter: torch.Tensor | None = None
        self.last_recovery_interaction_adapter: torch.Tensor | None = None
        self.last_recovery_adapter_gate: torch.Tensor | None = None
        self.last_recovery_payload_mass: torch.Tensor | None = None
        self.last_skill_outcome_utility: torch.Tensor | None = None
        self.last_wheel_prior: torch.Tensor | None = None
        self.last_wheel_residual: torch.Tensor | None = None
        self.last_nominal_wheel_action: torch.Tensor | None = None
        self.last_wheel_skill_gate: torch.Tensor | None = None
        self.last_wheel_prior_gate: torch.Tensor | None = None
        self.last_wheel_control_authority: torch.Tensor | None = None
        self.last_support_reference: torch.Tensor | None = None
        self.last_support_gate: torch.Tensor | None = None
        self.last_support_residual: torch.Tensor | None = None
        self.last_motion_kinematic_gain: torch.Tensor | None = None
        self.last_motion_action_candidates: torch.Tensor | None = None
        self.last_motion_raw_action_candidates: torch.Tensor | None = None
        self.last_task_motion_target: torch.Tensor | None = None
        self.last_task_motion_request: torch.Tensor | None = None
        self.last_motion_execution_prediction: torch.Tensor | None = None
        self.last_motion_execution_confidence: torch.Tensor | None = None
        self.last_embodiment_response_prior: torch.Tensor | None = None
        self.last_embodiment_response_prediction: torch.Tensor | None = None
        self.last_embodiment_response_score: torch.Tensor | None = None
        self.last_embodiment_response_authority: torch.Tensor | None = None
        self.last_motion_execution_utility: torch.Tensor | None = None
        self.last_motion_epistemic_risk: torch.Tensor | None = None
        self.last_motion_activity_demand: torch.Tensor | None = None
        self.last_idle_motion_penalty: torch.Tensor | None = None
        self.last_payload_skill_barrier_pressure: torch.Tensor | None = None
        self.last_payload_transient_demand: torch.Tensor | None = None
        self.last_payload_skill_projection: torch.Tensor | None = None
        self.last_payload_survival_authority: torch.Tensor | None = None
        self.last_payload_survival_projection: torch.Tensor | None = None
        self.last_payload_survival_advantage: torch.Tensor | None = None
        self.last_payload_survival_exit: torch.Tensor | None = None
        self.forced_motion_skill_id: int | None = None
        self.support_gate_override: float | None = None
        self.disable_payload_skill_barrier = False

    def skill_codebook(self) -> torch.Tensor:
        motion = self.motion_skill_embedding.weight[:, None, :].expand(
            -1, INTERACTION_SKILL_COUNT, -1
        )
        interaction = self.interaction_skill_embedding.weight[
            None, :, :
        ].expand(MOTION_SKILL_COUNT, -1, -1)
        return torch.cat((motion, interaction), dim=-1).reshape(
            ACTION_LAYOUT.skill_dim, -1
        )

    def interaction_phase_feasibility(
        self,
        selected_raw_slot: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Evaluate smooth capture and release admissibility margins."""

        contact = selected_raw_slot[:, 22].clamp(0.0, 1.0)
        lift = selected_raw_slot[:, 23].clamp(0.0, 1.0)
        transport = selected_raw_slot[:, 24].clamp(0.0, 1.0)
        place = selected_raw_slot[:, 25].clamp(0.0, 1.0)
        tcp_distance = torch.linalg.vector_norm(
            selected_raw_slot[:, TASK_SLOT_OBJECT_DELTA_SLICE],
            dim=-1,
        )
        left_delta = (
            0.75
            * selected_raw_slot[:, TASK_SLOT_LEFT_FINGER_DELTA_SLICE]
        )
        right_delta = (
            0.75
            * selected_raw_slot[:, TASK_SLOT_RIGHT_FINGER_DELTA_SLICE]
        )
        left_distance = torch.linalg.vector_norm(left_delta, dim=-1)
        right_distance = torch.linalg.vector_norm(right_delta, dim=-1)
        finger_distance = torch.maximum(left_distance, right_distance)
        center_error = torch.linalg.vector_norm(
            0.5 * (left_delta + right_delta),
            dim=-1,
        )
        center_gate = torch.sigmoid(
            CAPTURE_BARRIER_GAIN
            * (CAPTURE_CENTER_RADIUS - center_error)
        )
        finger_gate = torch.sigmoid(
            CAPTURE_BARRIER_GAIN
            * (CAPTURE_FINGER_RADIUS - finger_distance)
        )
        tcp_gate = torch.sigmoid(
            CAPTURE_BARRIER_GAIN
            * (CAPTURE_TCP_RADIUS - tcp_distance)
        )
        geometric_capture = torch.minimum(
            torch.minimum(center_gate, finger_gate),
            tcp_gate,
        )
        secure_entry_feasibility = torch.minimum(
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
        hold_evidence = torch.maximum(contact, lift)
        capture_feasibility = torch.maximum(
            geometric_capture,
            hold_evidence,
        ).clamp(0.0, 1.0)
        secure_entry_feasibility = torch.maximum(
            secure_entry_feasibility,
            hold_evidence,
        ).clamp(0.0, 1.0)
        target_delta = selected_raw_slot[
            :, TASK_SLOT_TARGET_DELTA_SLICE
        ]
        target_distance = 1.5 * torch.linalg.vector_norm(
            target_delta[:, :2],
            dim=-1,
        )
        target_vertical_error = torch.abs(
            1.5 * target_delta[:, 2] + RELEASE_HOVER_HEIGHT
        )
        target_gate = torch.sigmoid(
            RELEASE_TARGET_GAIN
            * (self.release_target_radius - target_distance)
        )
        vertical_gate = torch.sigmoid(
            RELEASE_VERTICAL_GAIN
            * (
                RELEASE_VERTICAL_TOLERANCE
                - target_vertical_error
            )
        )
        transport_gate = torch.sigmoid(
            RELEASE_TRANSPORT_GAIN
            * (transport - RELEASE_TRANSPORT_THRESHOLD)
        )
        carrying = selected_raw_slot[
            :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        raw_release_ready = (
            carrying
            * torch.minimum(target_gate, vertical_gate)
            * transport_gate
        ).clamp(0.0, 1.0)
        release_ready = carrying * torch.sigmoid(
            RELEASE_READINESS_GAIN
            * (raw_release_ready - RELEASE_READINESS_THRESHOLD)
        )
        release_feasibility = torch.maximum(
            release_ready,
            place,
        ).clamp(0.0, 1.0)
        target_progress = torch.exp(-target_distance / 0.45)
        vertical_progress = torch.exp(-target_vertical_error / 0.20)
        transport_progress = (0.15 + 0.85 * transport).clamp(0.0, 1.0)
        release_frontier = (
            carrying
            * transport_progress
            * torch.sqrt(
                (target_progress * vertical_progress).clamp_min(0.0)
            )
        ).clamp(0.0, 1.0)
        release_frontier = torch.maximum(
            release_frontier, release_feasibility
        )
        self.last_interaction_capture_feasibility = capture_feasibility
        self.last_interaction_secure_entry_feasibility = (
            secure_entry_feasibility
        )
        self.last_interaction_release_feasibility = release_feasibility
        self.last_interaction_release_frontier = release_frontier
        self.last_interaction_release_target_gate = target_gate
        self.last_interaction_release_vertical_gate = vertical_gate
        self.last_interaction_release_transport_gate = transport_gate
        self.last_interaction_hold_evidence = hold_evidence
        self.last_interaction_center_error = center_error
        self.last_interaction_finger_distance = finger_distance
        self.last_interaction_tcp_distance = tcp_distance
        return (
            capture_feasibility,
            secure_entry_feasibility,
            release_feasibility,
            release_frontier,
            hold_evidence,
            center_error,
            finger_distance,
            tcp_distance,
        )

    def interaction_phase_distribution(
        self,
        selected_raw_slot: torch.Tensor,
    ) -> torch.Tensor:
        """Infer a soft learning frontier while execution stays hard-gated."""

        contact = selected_raw_slot[:, 22].clamp(0.0, 1.0)
        lift = selected_raw_slot[:, 23].clamp(0.0, 1.0)
        (
            capture_feasibility,
            secure_entry_feasibility,
            release_feasibility,
            release_frontier,
            _,
            _,
            _,
            _,
        ) = self.interaction_phase_feasibility(selected_raw_slot)
        carrying = selected_raw_slot[
            :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        approach = (
            (1.0 - contact)
            * (1.0 - secure_entry_feasibility)
            * (1.0 - carrying)
        )
        release = release_frontier
        secure = (
            torch.maximum(
                secure_entry_feasibility,
                torch.maximum(torch.maximum(contact, lift), carrying),
            )
            * (1.0 - 0.85 * release)
        )
        phase = torch.stack((approach, secure, release), dim=-1) + 0.015
        return phase / phase.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _task_candidate_outcomes(
        self,
        global_latent: torch.Tensor,
        slot_latent: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict calibrated outcomes for every relational task candidate."""

        expanded_global = global_latent.unsqueeze(1).expand(
            -1, TASK_SLOT_COUNT, -1
        )
        features = torch.cat((expanded_global, slot_latent), dim=-1)
        hidden = self.task_outcome_encoder(
            features.reshape(-1, features.shape[-1])
        )
        outcomes = torch.sigmoid(
            self.task_outcome_head(hidden)
        ).reshape(-1, TASK_SLOT_COUNT, TASK_OUTCOME_DIM)
        confidence = torch.sigmoid(
            self.task_outcome_confidence_head(hidden)
        ).reshape(-1, TASK_SLOT_COUNT)
        base_utility = torch.einsum(
            "bso,o->bs", outcomes, self.task_outcome_weights
        )
        current_control = global_context[
            :, SAFETY_MARGIN_INDEX : DISTURBANCE_QUALITY_INDEX + 1
        ].clamp(0.0, 1.0)
        current_robustness = torch.einsum(
            "bo,o->b",
            current_control,
            self.task_outcome_weights,
        )
        robustness_target = (
            0.42 + 0.18 * (1.0 - current_robustness)
        ).clamp(0.42, 0.60)
        constraint_multiplier = (
            0.05
            + 0.95
            * torch.sigmoid(
                self.task_constraint_multiplier_head(global_latent)
            ).squeeze(-1)
        )
        constraint_violation = torch.relu(
            robustness_target.unsqueeze(1) - outcomes[:, :, 3]
        )
        dual_pressure = 0.5 * (
            self.hierarchy_cbf_dual + self.hierarchy_clf_dual
        )
        effective_multiplier = (
            constraint_multiplier + 0.25 * dual_pressure
        )
        utility = confidence * (
            base_utility
            - self.constraint_utility_gain
            * effective_multiplier.unsqueeze(1)
            * constraint_violation
        )
        utility = utility - utility.mean(dim=1, keepdim=True)
        self.last_task_outcome_confidence = confidence
        self.last_task_constraint_multiplier = constraint_multiplier
        self.last_task_constraint_violation = constraint_violation
        return outcomes, confidence, utility

    def _payload_transient_demand(
        self,
        global_context: torch.Tensor,
        distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate payload risk from speed and command-tracking transients."""

        velocity = global_context[:, BASE_VX_INDEX]
        previous_request = global_context[:, COMMAND_VX_INDEX].abs()
        while velocity.ndim < distance.ndim:
            velocity = velocity.unsqueeze(-1)
            previous_request = previous_request.unsqueeze(-1)
        speed = velocity.abs()
        speed_limit = (
            PAYLOAD_TRANSIENT_SPEED_FLOOR
            + PAYLOAD_TRANSIENT_SPEED_RANGE
            * torch.tanh(
                distance / PAYLOAD_TRANSIENT_DISTANCE_SCALE
            )
        )
        speed_demand = (
            (speed - speed_limit) / PAYLOAD_TRANSIENT_SPEED_SCALE
        ).clamp(0.0, 1.0)
        tracking_demand = (
            (
                speed
                - previous_request
                - PAYLOAD_TRANSIENT_TRACKING_ALLOWANCE
            )
            / PAYLOAD_TRANSIENT_TRACKING_SCALE
        ).clamp(0.0, 1.0)
        demand = torch.maximum(speed_demand, tracking_demand)
        if self.disable_payload_skill_barrier:
            demand = torch.zeros_like(demand)
        return demand, velocity

    def relational_task_subgoal_prior(
        self,
        raw_slots: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a relational task description to a transferable subgoal prior."""

        distance = (
            4.0 * raw_slots[..., TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 6.0)
        heading = (
            math.pi * raw_slots[..., TASK_SLOT_HEADING_INDEX]
        ).clamp(-math.pi, math.pi)
        interaction = (
            raw_slots[..., TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + raw_slots[..., TASK_SLOT_DELIVERY_TYPE_INDEX]
        ).clamp(0.0, 1.0)
        delivery = raw_slots[
            ..., TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        carrying = raw_slots[
            ..., TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        reachability = raw_slots[
            ..., TASK_SLOT_REACHABILITY_INDEX
        ].clamp(0.0, 1.0)
        recovery = raw_slots[
            ..., TASK_SLOT_RECOVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)

        control_margin = torch.minimum(
            global_context[:, SAFETY_MARGIN_INDEX],
            global_context[:, PREVIEW_MARGIN_INDEX],
        ).clamp(0.0, 1.0)
        base_tilt = 0.60 * global_context[:, BASE_TILT_INDEX]
        tilt_capacity = (
            (0.60 - base_tilt) / 0.35
        ).clamp(0.20, 1.0)
        support_capacity = (
            (global_context[:, SUPPORT_COUNT_INDEX] - 0.65) / 0.35
        ).clamp(0.20, 1.0)
        posture_capacity = torch.minimum(
            tilt_capacity, support_capacity
        )
        authority = (
            (0.30 + 0.70 * control_margin)
            * (0.40 + 0.60 * posture_capacity)
        )
        authority_cap = (
            0.50
            + 0.50
            * global_context[:, CURRICULUM_LEVEL_INDEX].clamp(0.0, 1.0)
        )
        authority = torch.minimum(authority, authority_cap).clamp(
            0.12, 1.0
        )
        context_authority = authority
        safety_demand = (
            (0.25 - control_margin) / 0.25
        ).clamp(0.0, 1.0)
        while authority.ndim < distance.ndim:
            authority = authority.unsqueeze(-1)
            safety_demand = safety_demand.unsqueeze(-1)
        recovery_pressure = self.control_recovery_pressure(global_context)
        while recovery_pressure.ndim < distance.ndim:
            recovery_pressure = recovery_pressure.unsqueeze(-1)
        recovery_regulation = (
            recovery
            * ((recovery_pressure - 0.20) / 0.35).clamp(0.0, 1.0)
        )

        # Hand over to the arm before wheel drift can carry a reachable
        # object back out of the interaction workspace.
        interaction_locomotion_need = (1.0 - reachability).square()
        locomotion_need = (
            1.0
            - interaction
            * (1.0 - interaction_locomotion_need)
        ).clamp(0.03, 1.0)
        approach = torch.tanh(distance / 0.75)
        approach_authority = (
            0.75 * approach * locomotion_need * authority
        )
        heading_alignment = (
            0.5 * (heading.cos() + 1.0)
        ).square()
        payload_progress_gain = 1.0 + 1.5 * carrying
        payload_safety_gate = (
            1.0 - carrying
            + carrying
            * (
                0.18
                + 0.82 * (1.0 - safety_demand).square()
            )
        )
        payload_transient_demand, payload_velocity = (
            self._payload_transient_demand(global_context, distance)
        )
        payload_transient_gate = (
            1.0
            - PAYLOAD_TRANSIENT_GATE_GAIN
            * carrying
            * payload_transient_demand
        )
        forward_magnitude = (
            approach_authority
            * (0.02 + 0.98 * heading_alignment)
            * payload_progress_gain
            * payload_safety_gate
            * payload_transient_gate
        ).clamp(0.0, 0.95)
        # Preserve the task geometry when handing a delivery subgoal to the
        # motion skill.  With a rear-mounted arm, a correctly aligned object or
        # placement target lies behind the body and therefore requests reverse
        # travel before and after capture.
        longitudinal_relation = raw_slots[..., 0]
        relation_travel_sign = torch.where(
            longitudinal_relation > 0.0,
            torch.ones_like(longitudinal_relation),
            -torch.ones_like(longitudinal_relation),
        )
        travel_sign = torch.where(
            delivery > 0.5,
            relation_travel_sign,
            torch.ones_like(relation_travel_sign),
        )
        payload_brake = (
            PAYLOAD_TRANSIENT_BRAKE_GAIN
            * carrying
            * payload_transient_demand
            * payload_velocity.sign()
        )
        task_forward = travel_sign * forward_magnitude - payload_brake
        task_lateral = (
            (1.0 - delivery)
            * 0.75
            * approach_authority
            * heading.sin()
        )
        task_xy = torch.stack((task_forward, task_lateral), dim=-1)

        heading_authority = (
            0.35 + 0.65 * approach
        ) * torch.sqrt(authority)
        desired_feasible_yaw = (
            0.55
            * 1.25
            * heading
            / math.pi
            * heading_authority
        )
        command_vx = 0.35 * task_xy[..., 0]
        command_vy = 0.18 * task_xy[..., 1]
        lateral_heading = torch.atan2(
            command_vy,
            command_vx.abs() + 0.03,
        )
        lateral_yaw = 0.75 * lateral_heading
        task_yaw = (
            (desired_feasible_yaw - lateral_yaw)
            / (0.55 * travel_sign)
        ).clamp(-1.5, 1.5)
        # Recovery defines a regulation subgoal; the lower policy still owns
        # the physical action and learns which skill realizes that subgoal.
        task_xy = task_xy * (1.0 - recovery_regulation).unsqueeze(-1)
        task_yaw = task_yaw * (1.0 - recovery_regulation)

        progress_demand = (
            approach * (0.55 + 0.40 * authority)
        ).clamp(0.05, 0.95)
        precision_demand = torch.maximum(
            interaction * (0.20 + 0.75 * reachability),
            torch.maximum(safety_demand, 1.0 - authority),
        ).clamp(0.05, 0.95)
        progress_demand = torch.lerp(
            progress_demand,
            torch.full_like(progress_demand, 0.08),
            recovery_regulation,
        )
        precision_demand = torch.maximum(
            precision_demand,
            0.92 * recovery_regulation,
        )
        progress_logit = 0.5 * torch.logit(progress_demand)
        precision_logit = 0.5 * torch.logit(precision_demand)

        prior = torch.zeros(
            *raw_slots.shape[:-1],
            ACTION_LAYOUT.task_subgoal_dim,
            device=raw_slots.device,
            dtype=raw_slots.dtype,
        )
        prior[..., 0:2] = task_xy
        prior[..., 2] = task_yaw
        prior[..., 6] = progress_logit
        prior[..., 7] = precision_logit
        return prior, context_authority

    def compose_task_subgoal(
        self,
        prior: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Compose a learned residual without reversing relational geometry."""

        motion_scale = (
            1.0 + self.task_residual_scale * residual[..., :3]
        ).clamp(0.55, 1.45)
        motion = prior[..., :3] * motion_scale
        interaction = (
            prior[..., 3:]
            + self.task_residual_scale * residual[..., 3:]
        )
        return torch.cat((motion, interaction), dim=-1).clamp(-1.5, 1.5)

    @staticmethod
    def control_recovery_pressure(
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Measure when a recovery task should enter the candidate set."""

        base_tilt = 0.60 * global_context[:, BASE_TILT_INDEX]
        support_count = 4.0 * global_context[:, SUPPORT_COUNT_INDEX]
        tilt_pressure = (
            (base_tilt - 0.30) / 0.18
        ).clamp(0.0, 1.0)
        support_pressure = (
            (2.0 - support_count) / 1.0
        ).clamp(0.0, 1.0)
        margin_pressure = (
            (
                0.16
                - torch.minimum(
                    global_context[:, SAFETY_MARGIN_INDEX],
                    global_context[:, PREVIEW_MARGIN_INDEX],
                )
            )
            / 0.16
        ).clamp(0.0, 1.0)
        return torch.maximum(
            tilt_pressure,
            torch.maximum(support_pressure, margin_pressure),
        )

    def recovery_adapter_outputs(
        self,
        global_context: torch.Tensor,
        raw_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict task and factorized skill residuals from viability state."""

        carrying = raw_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].amax(dim=1).clamp(0.0, 1.0)
        pressure = self.control_recovery_pressure(global_context)
        features = torch.stack(
            (
                pressure,
                global_context[:, SAFETY_MARGIN_INDEX].clamp(0.0, 1.0),
                global_context[:, PREVIEW_MARGIN_INDEX].clamp(0.0, 1.0),
                global_context[:, CLF_DECREASE_INDEX].clamp(0.0, 1.0),
                global_context[
                    :, DISTURBANCE_QUALITY_INDEX
                ].clamp(0.0, 1.0),
                (
                    (
                        global_context[:, BASE_HEIGHT_INDEX] - 0.42
                    )
                    / 0.18
                ).clamp(-1.0, 1.0),
                global_context[:, BASE_TILT_INDEX].clamp(0.0, 1.0),
                global_context[:, SUPPORT_COUNT_INDEX].clamp(0.0, 1.0),
                torch.tanh(global_context[:, BASE_VX_INDEX] / 0.35),
                torch.tanh(global_context[:, BASE_WZ_INDEX] / 0.75),
                carrying,
            ),
            dim=-1,
        )
        hidden = self.recovery_adapter_encoder(features)
        task_residual = torch.tanh(
            self.recovery_task_adapter_head(hidden)
        ).squeeze(-1)
        motion_residual = torch.tanh(
            self.recovery_motion_adapter_head(hidden)
        )
        motion_residual = (
            motion_residual
            - motion_residual.mean(dim=1, keepdim=True)
        )
        interaction_residual = torch.tanh(
            self.recovery_interaction_adapter_head(hidden)
        )
        interaction_residual = (
            interaction_residual
            - interaction_residual.mean(dim=1, keepdim=True)
        )
        return task_residual, motion_residual, interaction_residual

    def grounded_task_utility(
        self,
        raw_slots: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate immediate task affordance from transferable measurements."""

        distance = (
            4.0 * raw_slots[:, :, TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 6.0)
        proximity = torch.exp(-distance / 1.20)
        heading_alignment = torch.exp(
            -2.0
            * raw_slots[:, :, TASK_SLOT_HEADING_INDEX].abs().clamp_max(1.0)
        )
        progress = (
            1.0
            - raw_slots[:, :, TASK_SLOT_REMAINING_PROGRESS_INDEX]
        ).clamp(0.0, 1.0)

        navigation_utility = (
            0.58 * proximity
            + 0.20 * heading_alignment
            + 0.22 * progress
        )

        interaction_state = raw_slots[
            :, :, TASK_SLOT_INTERACTION_STATE_SLICE
        ].clamp(0.0, 1.0)
        contact = interaction_state[:, :, 0]
        lift = interaction_state[:, :, 1]
        transport = interaction_state[:, :, 2]
        place = interaction_state[:, :, 3]
        engagement = torch.maximum(
            contact, torch.maximum(lift, transport)
        )
        reachability = raw_slots[
            :, :, TASK_SLOT_REACHABILITY_INDEX
        ].clamp(0.0, 1.0)
        object_distance = torch.linalg.norm(
            raw_slots[:, :, TASK_SLOT_OBJECT_DELTA_SLICE], dim=-1
        )
        target_distance = 1.5 * torch.linalg.norm(
            raw_slots[:, :, TASK_SLOT_TARGET_DELTA_SLICE][:, :, :2],
            dim=-1,
        )
        object_proximity = torch.exp(-object_distance / 0.55)
        target_proximity = torch.exp(-target_distance / 0.75)
        carrying = raw_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        pre_contact_affordance = (
            0.55 * object_proximity + 0.45 * reachability
        )
        post_contact_affordance = (
            0.35 * reachability + 0.65 * target_proximity
        )
        interaction_affordance = torch.lerp(
            pre_contact_affordance,
            post_contact_affordance,
            engagement,
        )
        delivery_utility = (
            0.20 * proximity
            + 0.10 * heading_alignment
            + 0.14 * progress
            + 0.22 * interaction_affordance
            + 0.14 * engagement
            + 0.12 * carrying * target_proximity
            + 0.08 * place
        )
        delivery = raw_slots[
            :, :, TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        utility = torch.lerp(navigation_utility, delivery_utility, delivery)
        recovery = raw_slots[
            :, :, TASK_SLOT_RECOVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        recovery_pressure = self.control_recovery_pressure(
            global_context
        ).unsqueeze(1)
        recovery_utility = (
            0.35 * recovery_pressure
            - 0.10 * (1.0 - recovery_pressure)
        )
        utility = torch.lerp(utility, recovery_utility, recovery)
        valid = (
            (raw_slots[:, :, TASK_SLOT_REQUIRED_INDEX] > 0.5)
            & (raw_slots[:, :, TASK_SLOT_COMPLETED_INDEX] < 0.5)
            & (raw_slots[:, :, TASK_SLOT_AVAILABLE_INDEX] > 0.5)
        ).float()
        valid_mean = (utility * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return utility - valid_mean

    def _skill_candidate_outcomes(
        self,
        mission: torch.Tensor,
        task_latent: torch.Tensor,
        interaction_active: torch.Tensor,
        payload_carrying: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict task-conditioned outcomes for every factorized skill."""

        codebook = self.skill_codebook()
        skill_count = codebook.shape[0]
        mission_expanded = mission.unsqueeze(1).expand(-1, skill_count, -1)
        task_expanded = task_latent.unsqueeze(1).expand(-1, skill_count, -1)
        code_expanded = codebook.unsqueeze(0).expand(
            mission.shape[0], -1, -1
        )
        features = torch.cat(
            (mission_expanded, task_expanded, code_expanded), dim=-1
        )
        hidden = self.skill_outcome_encoder(
            features.reshape(-1, features.shape[-1])
        )
        outcomes = torch.sigmoid(
            self.skill_outcome_head(hidden)
        ).reshape(-1, ACTION_LAYOUT.skill_dim, SKILL_OUTCOME_DIM)
        survival_hidden = self.payload_survival_encoder(
            features.reshape(-1, features.shape[-1])
        )
        survival = torch.sigmoid(
            self.payload_survival_head(survival_hidden)
        ).reshape(-1, ACTION_LAYOUT.skill_dim)
        weights = self.skill_outcome_weights.unsqueeze(0).expand(
            mission.shape[0], -1
        ).clone()
        weights[:, -1] = weights[:, -1] * interaction_active
        utility = torch.einsum("bso,bo->bs", outcomes, weights)
        utility = utility - utility.mean(dim=1, keepdim=True)
        self.last_skill_survival = survival
        return outcomes, utility

    def _skill_candidate_effects(
        self,
        mission: torch.Tensor,
        task_latent: torch.Tensor,
        interaction_active: torch.Tensor,
        global_context: torch.Tensor,
        selected_raw_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict calibrated, task-conditioned effects for every skill."""

        codebook = self.skill_codebook()
        skill_count = codebook.shape[0]
        mission_expanded = mission.unsqueeze(1).expand(-1, skill_count, -1)
        task_expanded = task_latent.unsqueeze(1).expand(-1, skill_count, -1)
        code_expanded = codebook.unsqueeze(0).expand(
            mission.shape[0], -1, -1
        )
        features = torch.cat(
            (mission_expanded, task_expanded, code_expanded), dim=-1
        )
        hidden = self.skill_effect_encoder(
            features.reshape(-1, features.shape[-1])
        )
        effects = torch.tanh(self.skill_effect_head(hidden)).reshape(
            -1, ACTION_LAYOUT.skill_dim, SKILL_EFFECT_DIM
        )
        confidence = torch.sigmoid(
            self.skill_effect_confidence_head(hidden)
        ).reshape(-1, ACTION_LAYOUT.skill_dim)

        motion_request = self.feasible_motion_request(global_context)
        command_vx = (motion_request[:, 0] / 0.75).clamp(-1.0, 1.0)
        command_wz = (motion_request[:, 1] / 1.50).clamp(-1.0, 1.0)
        tracking_quality = -(
            0.70
            * (effects[:, :, 0] - command_vx.unsqueeze(1)).abs()
            + 0.30
            * (effects[:, :, 1] - command_wz.unsqueeze(1)).abs()
        )
        robust_quality = (
            0.30 * effects[:, :, 4]
            + 0.25 * effects[:, :, 5]
            + 0.25 * effects[:, :, 6]
            + 0.20 * effects[:, :, 7]
        )
        interaction_gain = (
            interaction_active.unsqueeze(1)
            * effects[:, :, 8].clamp_min(0.0)
        )
        current_control = global_context[
            :, SAFETY_MARGIN_INDEX : DISTURBANCE_QUALITY_INDEX + 1
        ].clamp(0.0, 1.0)
        constraint_threshold = (
            self.control_constraint_floor.unsqueeze(0)
            + 0.35
            * torch.relu(
                self.control_constraint_floor.unsqueeze(0)
                - current_control
            )
        ).clamp_max(0.65)
        constraint_multipliers = (
            0.05
            + 0.95
            * torch.sigmoid(
                self.skill_constraint_multiplier_head(
                    torch.cat((mission, task_latent), dim=-1)
                )
            )
        )
        predicted_control = 0.5 * (
            effects[:, :, 4:8] + 1.0
        )
        interaction_state = selected_raw_slot[
            :, TASK_SLOT_INTERACTION_STATE_SLICE
        ].clamp(0.0, 1.0)
        interaction_evidence = interaction_state[:, :3].amax(
            dim=1, keepdim=True
        )
        carrying_evidence = selected_raw_slot[
            :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0).unsqueeze(1)
        phase_id = torch.arange(
            ACTION_LAYOUT.skill_dim, device=mission.device
        ).remainder(INTERACTION_SKILL_COUNT)
        secure_candidate = (phase_id == 1).to(mission.dtype).unsqueeze(0)
        release_candidate = (phase_id == 2).to(mission.dtype).unsqueeze(0)
        transient_slack = interaction_active.unsqueeze(1) * (
            SECURE_CBF_TRANSIENT_SLACK
            * secure_candidate
            * (0.35 + 0.65 * interaction_evidence)
            + RELEASE_CBF_TRANSIENT_SLACK
            * release_candidate
            * (0.50 + 0.50 * carrying_evidence)
        )
        candidate_threshold = constraint_threshold.unsqueeze(1).expand(
            -1, ACTION_LAYOUT.skill_dim, -1
        ).clone()
        candidate_threshold[:, :, :2] = torch.maximum(
            candidate_threshold[:, :, :2]
            - transient_slack.unsqueeze(2),
            candidate_threshold.new_tensor(INTERACTION_HARD_CONTROL_FLOOR),
        )
        constraint_violation = torch.relu(
            candidate_threshold - predicted_control
        )
        constraint_cost = (
            constraint_violation
            * constraint_multipliers.unsqueeze(1)
        ).sum(dim=-1) / constraint_multipliers.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-4)
        primal_dual_cost = (
            self.hierarchy_cbf_dual
            * constraint_violation[:, :, :2].amax(dim=-1)
            + self.hierarchy_clf_dual
            * constraint_violation[:, :, 2]
        )
        raw_utility = (
            0.24 * effects[:, :, 2]
            + 0.18 * tracking_quality
            + 0.10 * effects[:, :, 3]
            + 0.30 * robust_quality
            + 0.18 * interaction_gain
            - self.constraint_utility_gain
            * (constraint_cost + 0.35 * primal_dual_cost)
        )
        # At initialization and outside the data support, confidence is low.
        # Consequently a speculative counterfactual cannot steer the policy.
        utility = confidence * raw_utility
        utility = utility - utility.mean(dim=1, keepdim=True)
        self.last_skill_constraint_multipliers = constraint_multipliers
        self.last_skill_constraint_threshold = candidate_threshold
        self.last_skill_constraint_violation = constraint_violation
        return effects, confidence, utility

    def transition_logits(
        self,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identify executed task and skill from their measured state change."""

        current_global = current_context[:, :GLOBAL_CONTEXT_DIM].clone()
        delta_global = (
            next_context[:, :GLOBAL_CONTEXT_DIM] - current_global
        )
        for index in (
            EXECUTED_TASK_INDEX,
            EXECUTED_SKILL_INDEX,
            EXECUTED_OBJECT_INDEX,
        ):
            current_global[:, index] = 0.0
            delta_global[:, index] = 0.0
        current_global[:, TERMINATION_STATE_SLICE] = 0.0
        delta_global[:, TERMINATION_STATE_SLICE] = 0.0

        transition = self.transition_encoder(
            torch.cat((current_global, delta_global), dim=-1)
        )
        current_slots = current_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        next_slots = next_context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        slot_transition = torch.cat(
            (current_slots, next_slots - current_slots), dim=-1
        )
        slot_latent = self.transition_slot_encoder(
            slot_transition.reshape(-1, 2 * TASK_SLOT_FEATURE_DIM)
        ).reshape(current_context.shape[0], TASK_SLOT_COUNT, -1)
        task_query = F.normalize(
            self.task_transition_query(transition), dim=-1
        )
        task_keys = F.normalize(
            self.task_transition_key(slot_latent), dim=-1
        )
        task_logits = torch.einsum(
            "bd,bsd->bs", task_query, task_keys
        ) / max(self.transition_temperature, 1.0e-4)

        task_attention = torch.softmax(task_logits, dim=-1)
        transition_task = torch.einsum(
            "bs,bsd->bd", task_attention, slot_latent
        )
        motion_indices = list(self.motion_transition_indices)
        interaction_indices = list(self.interaction_transition_indices)
        motion_transition = torch.cat(
            (
                current_global[:, motion_indices],
                delta_global[:, motion_indices],
                transition_task,
            ),
            dim=-1,
        )
        interaction_transition = torch.cat(
            (
                current_global[:, interaction_indices],
                delta_global[:, interaction_indices],
                transition_task,
            ),
            dim=-1,
        )
        motion_query = F.normalize(
            self.motion_transition_query(motion_transition), dim=-1
        )
        interaction_query = F.normalize(
            self.interaction_transition_query(interaction_transition), dim=-1
        )
        motion_keys = F.normalize(
            self.motion_transition_key(self.motion_skill_embedding.weight),
            dim=-1,
        )
        interaction_keys = F.normalize(
            self.interaction_transition_key(
                self.interaction_skill_embedding.weight
            ),
            dim=-1,
        )
        temperature = max(self.transition_temperature, 1.0e-4)
        motion_logits = (
            motion_query @ motion_keys.transpose(0, 1)
        ) / temperature
        interaction_logits = (
            interaction_query @ interaction_keys.transpose(0, 1)
        ) / temperature
        skill_logits = (
            motion_logits.unsqueeze(2) + interaction_logits.unsqueeze(1)
        ).reshape(-1, ACTION_LAYOUT.skill_dim)
        return task_logits, skill_logits

    def _mission(
        self, context: torch.Tensor, apply_commitment: bool = True
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        global_context = context[:, :GLOBAL_CONTEXT_DIM]
        self.last_global_context = global_context
        raw_slots = context[:, GLOBAL_CONTEXT_DIM:].reshape(
            -1, TASK_SLOT_COUNT, TASK_SLOT_FEATURE_DIM
        )
        self.last_raw_task_slots = raw_slots
        global_latent = self.global_encoder(global_context)
        morphology_latent = self.morphology_encoder(
            global_context[:, MORPHOLOGY_SLICE]
        )
        global_latent = self.morphology_fusion(
            torch.cat((global_latent, morphology_latent), dim=-1)
        )
        slot_latent = self.slot_encoder(
            raw_slots.reshape(-1, TASK_SLOT_FEATURE_DIM)
        )
        slot_latent = slot_latent.reshape(
            context.shape[0], TASK_SLOT_COUNT, -1
        )
        relation = raw_slots[:, :, 32:40]
        left = relation[:, :, 0:3]
        right = relation[:, :, 3:6]
        left_distance = torch.linalg.norm(left, dim=-1, keepdim=True)
        right_distance = torch.linalg.norm(right, dim=-1, keepdim=True)
        midpoint_distance = torch.linalg.norm(
            0.5 * (left + right), dim=-1, keepdim=True
        )
        distance_asymmetry = (left_distance - right_distance).abs()
        relation_descriptor = torch.cat(
            (
                relation,
                left_distance,
                right_distance,
                midpoint_distance,
                distance_asymmetry,
            ),
            dim=-1,
        )
        relation_latent = self.relation_encoder(
            relation_descriptor.reshape(-1, 12)
        ).reshape(context.shape[0], TASK_SLOT_COUNT, -1)
        slot_latent = self.slot_relation_fusion(
            torch.cat((slot_latent, relation_latent), dim=-1)
        )
        graph_message, _ = self.task_graph_attention(
            slot_latent,
            slot_latent,
            slot_latent,
            need_weights=False,
        )
        slot_latent = self.task_graph_norm_1(
            slot_latent + graph_message
        )
        slot_latent = self.task_graph_norm_2(
            slot_latent + self.task_graph_ffn(slot_latent)
        )

        query = self.task_query(global_latent).unsqueeze(1)
        keys = self.task_key(slot_latent)
        logits = (query * keys).sum(dim=-1) / keys.shape[-1] ** 0.5
        logits = logits + self.task_bias(slot_latent).squeeze(-1)
        (
            task_outcomes,
            task_outcome_confidence,
            task_utility,
        ) = self._task_candidate_outcomes(
            global_latent,
            slot_latent,
            global_context,
        )
        grounded_utility = self.grounded_task_utility(
            raw_slots, global_context
        )
        outcome_reliability = (
            float(self.task_outcome_maturity)
            * (
                (task_outcome_confidence - 0.35) / 0.50
            ).clamp(0.0, 1.0)
        )
        blended_utility = (
            grounded_utility
            + self.task_outcome_gain
            * outcome_reliability
            * task_utility.detach()
        )
        # Outcome heads are calibrated only by measured transitions.  Their
        # predictions guide option selection without receiving policy-gradient
        # pressure to make a preferred option look artificially successful.
        logits = logits + self.task_affordance_gain * grounded_utility
        logits = logits + (
            self.task_outcome_gain
            * outcome_reliability
            * task_utility.detach()
        )
        recovery_role = raw_slots[
            :, :, TASK_SLOT_RECOVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        recovery_pressure = self.control_recovery_pressure(
            global_context
        ).unsqueeze(1)
        recovery_preference = (
            (recovery_pressure - 0.20) / 0.60
        ).clamp(0.0, 1.0)
        logits = logits + (
            self.recovery_task_margin_gain
            * recovery_role
            * recovery_preference
        )
        (
            recovery_task_adapter,
            recovery_motion_adapter,
            recovery_interaction_adapter,
        ) = self.recovery_adapter_outputs(
            global_context, raw_slots
        )
        logits = logits + (
            self.recovery_adapter_task_gain
            * recovery_role
            * recovery_preference
            * recovery_task_adapter.unsqueeze(1)
        )
        self.last_recovery_task_preference = (
            recovery_role * recovery_preference
        )
        self.last_recovery_task_adapter = recovery_task_adapter
        self.last_recovery_motion_adapter = recovery_motion_adapter
        self.last_recovery_interaction_adapter = (
            recovery_interaction_adapter
        )
        self.last_recovery_adapter_gate = (
            recovery_preference.squeeze(1)
        )

        required = raw_slots[:, :, 11].clamp(0.0, 1.0)
        completed = raw_slots[:, :, 12].clamp(0.0, 1.0)
        available = raw_slots[:, :, 13].clamp(0.0, 1.0)
        invalid = (required < 0.5) | (completed > 0.5) | (available < 0.5)
        all_invalid = invalid.all(dim=1, keepdim=True)
        invalid = invalid & (~all_invalid)
        logits = logits.masked_fill(invalid, -20.0)
        if apply_commitment:
            current_task = torch.round(
                global_context[:, EXECUTED_TASK_INDEX]
                * float(ACTION_LAYOUT.task_dim - 1)
            ).long().clamp(0, ACTION_LAYOUT.task_dim - 1)
            current_valid = (~invalid).gather(
                1, current_task.unsqueeze(1)
            ).squeeze(1)
            committed = (
                (global_context[:, 1] > 0.01)
                & (global_context[:, 1] < 0.97)
                & (global_context[:, TERMINATION_STATE_SLICE.start] < 0.97)
                & current_valid
            )
            committed = committed & (
                self.control_recovery_pressure(global_context) < 0.55
            )
            commitment_mask = torch.ones_like(logits, dtype=torch.bool)
            commitment_mask.scatter_(1, current_task.unsqueeze(1), False)
            logits = logits.masked_fill(
                commitment_mask & committed.unsqueeze(1), -20.0
            )

        attention = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bs,bsd->bd", attention, slot_latent)
        mission = self.mission_fusion(torch.cat((global_latent, pooled), dim=-1))
        self.last_slot_latent = slot_latent
        self.last_task_outcomes = task_outcomes
        self.last_task_outcome_confidence = task_outcome_confidence
        self.last_task_outcome_utility = task_utility
        self.last_task_grounded_utility = grounded_utility
        self.last_task_outcome_reliability = outcome_reliability
        self.last_task_blended_utility = blended_utility
        return mission, slot_latent, raw_slots, logits, attention

    def task_choice(
        self, context: torch.Tensor, apply_commitment: bool = True
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        mission, slots, raw_slots, task_logits, _ = self._mission(
            context, apply_commitment=apply_commitment
        )
        return task_logits, (mission, slots, raw_slots)

    def task_detail_parameters(
        self,
        task_code: torch.Tensor,
        mission_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mission, slots, raw_slots = mission_state
        selected_slot = torch.einsum("bs,bsd->bd", task_code, slots)
        selected_raw_slot = torch.einsum(
            "bs,bsd->bd", task_code, raw_slots
        )
        detail = self.task_detail_encoder(
            torch.cat((mission, selected_slot), dim=-1)
        )
        continuous = self.task_continuous_head(detail)
        object_end = ACTION_LAYOUT.object_dim
        subgoal_end = object_end + ACTION_LAYOUT.task_subgoal_dim
        object_logits = continuous[:, :object_end]
        subgoal_raw = continuous[:, object_end:subgoal_end]
        global_context = self.last_global_context
        if global_context is None:
            raise RuntimeError("TACTIC mission context was not evaluated")
        subgoal_prior, _ = self.relational_task_subgoal_prior(
            selected_raw_slot, global_context
        )
        subgoal_residual = 1.5 * torch.tanh(subgoal_raw / 1.5)
        subgoal = self.compose_task_subgoal(
            subgoal_prior, subgoal_residual
        )
        interaction_gate = (
            selected_raw_slot[:, TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + selected_raw_slot[:, TASK_SLOT_DELIVERY_TYPE_INDEX]
            * selected_raw_slot[:, TASK_SLOT_REACHABILITY_INDEX].clamp(
                0.0, 1.0
            )
        ).clamp(0.0, 1.0)
        subgoal = torch.cat(
            (
                subgoal[:, :3],
                interaction_gate.unsqueeze(1) * subgoal[:, 3:6],
                subgoal[:, 6:],
            ),
            dim=-1,
        )
        termination = continuous[:, subgoal_end:]

        delivery_prior = task_code[:, 5:11].clamp_min(0.0)
        delivery_mass = delivery_prior.sum(dim=-1, keepdim=True).clamp(
            0.0, 1.0
        )
        delivery_prior = (
            delivery_prior + 0.02
        ) / (
            delivery_prior + 0.02
        ).sum(dim=-1, keepdim=True)
        object_logits = object_logits + (
            3.0
            * delivery_mass
            * torch.log(delivery_prior.clamp_min(1.0e-6))
        )
        return object_logits, subgoal, termination

    def candidate_task_details(
        self, context: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Decode every task slot through the shared upper-level policy."""

        (
            mission,
            slots,
            raw_slots,
            task_logits,
            _,
        ) = self._mission(context, apply_commitment=False)
        mission_grid = mission.unsqueeze(1).expand(
            -1, TASK_SLOT_COUNT, -1
        )
        detail_input = torch.cat((mission_grid, slots), dim=-1)
        detail = self.task_detail_encoder(
            detail_input.reshape(-1, detail_input.shape[-1])
        ).reshape(context.shape[0], TASK_SLOT_COUNT, -1)
        continuous = self.task_continuous_head(detail)
        object_end = ACTION_LAYOUT.object_dim
        subgoal_end = object_end + ACTION_LAYOUT.task_subgoal_dim
        subgoal_raw = continuous[:, :, object_end:subgoal_end]
        global_context = self.last_global_context
        if global_context is None:
            raise RuntimeError("TACTIC mission context was not evaluated")
        subgoal_prior, _ = self.relational_task_subgoal_prior(
            raw_slots, global_context
        )
        subgoal_residual = 1.5 * torch.tanh(subgoal_raw / 1.5)
        subgoal = self.compose_task_subgoal(
            subgoal_prior, subgoal_residual
        )
        interaction_gate = (
            raw_slots[:, :, TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + raw_slots[:, :, TASK_SLOT_DELIVERY_TYPE_INDEX]
            * raw_slots[:, :, TASK_SLOT_REACHABILITY_INDEX].clamp(
                0.0, 1.0
            )
        ).clamp(0.0, 1.0)
        subgoal = torch.cat(
            (
                subgoal[:, :, :3],
                interaction_gate.unsqueeze(-1) * subgoal[:, :, 3:6],
                subgoal[:, :, 6:],
            ),
            dim=-1,
        )
        task_utility = self.last_task_outcome_utility
        task_confidence = self.last_task_outcome_confidence
        if task_utility is None or task_confidence is None:
            raise RuntimeError("Task candidate outcomes were not evaluated")
        return (
            task_logits,
            subgoal,
            raw_slots,
            task_utility,
            task_confidence,
        )

    def task_parameters(
        self, context: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        task_logits, mission_state = self.task_choice(context)
        task_code = torch.softmax(task_logits, dim=-1)
        object_logits, subgoal, termination = self.task_detail_parameters(
            task_code, mission_state
        )
        return (
            task_logits,
            object_logits,
            subgoal,
            termination,
            mission_state,
        )

    def task_mean(
        self, context: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        (
            task_logits,
            object_logits,
            subgoal,
            termination,
            mission_state,
        ) = self.task_parameters(context)
        task_block = torch.cat(
            (task_logits, object_logits, subgoal, termination), dim=-1
        )
        return task_block, mission_state

    def _task_latent(
        self,
        task_block: torch.Tensor,
        slots: torch.Tensor,
        raw_slots: torch.Tensor,
    ) -> torch.Tensor:
        task_end = ACTION_LAYOUT.task_dim
        object_end = task_end + ACTION_LAYOUT.object_dim
        subgoal_end = object_end + ACTION_LAYOUT.task_subgoal_dim
        task_probability = task_block[:, :task_end].clamp_min(0.0)
        task_probability = task_probability / task_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        object_probability = task_block[:, task_end:object_end].clamp_min(0.0)
        object_probability = object_probability / object_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        delivery_probability = task_probability[:, 5:11]
        delivery_mass = delivery_probability.sum(
            dim=-1, keepdim=True
        ).clamp(0.0, 1.0)
        delivery_object_probability = delivery_probability / (
            delivery_mass.clamp_min(1.0e-6)
        )
        object_probability = (
            (1.0 - delivery_mass) * object_probability
            + delivery_mass * delivery_object_probability
        )
        selected_slot = torch.einsum(
            "bs,bsd->bd", task_probability, slots
        )
        selected_raw_slot = torch.einsum(
            "bs,bsd->bd", task_probability, raw_slots
        )
        interaction_gate = (
            selected_raw_slot[:, TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + selected_raw_slot[:, TASK_SLOT_DELIVERY_TYPE_INDEX]
            * selected_raw_slot[:, TASK_SLOT_REACHABILITY_INDEX].clamp(
                0.0, 1.0
            )
        ).clamp(0.0, 1.0)
        selected_object = object_probability @ self.object_embedding.weight
        termination_context = torch.sigmoid(
            task_block[:, subgoal_end:].detach()
        )
        task_subgoal = torch.tanh(
            task_block[:, object_end:subgoal_end]
        )
        task_subgoal = torch.cat(
            (
                task_subgoal[:, :3],
                interaction_gate.unsqueeze(1) * task_subgoal[:, 3:6],
                task_subgoal[:, 6:],
            ),
            dim=-1,
        )
        task_input = torch.cat(
            (
                selected_slot,
                selected_object,
                task_subgoal,
                termination_context,
            ),
            dim=-1,
        )
        return self.task_conditioner(task_input)

    def skill_choice(
        self,
        task_block: torch.Tensor,
        mission_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        apply_commitment: bool = True,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        mission, slots, raw_slots = mission_state
        task_latent = self._task_latent(task_block, slots, raw_slots)
        skill_latent = self.skill_encoder(torch.cat((mission, task_latent), dim=-1))
        feasibility_logits = self.skill_feasibility_head(skill_latent)
        motion_logits = self.motion_skill_logits_head(skill_latent)
        interaction_logits = self.interaction_skill_logits_head(skill_latent)
        task_probability = task_block[:, : ACTION_LAYOUT.task_dim].clamp_min(
            0.0
        )
        task_probability = task_probability / task_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        selected_raw_slot = torch.einsum(
            "bs,bsd->bd", task_probability, raw_slots
        )
        payload_carrying = raw_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        payload_slot_id = payload_carrying.argmax(dim=1)
        payload_relation_slot = raw_slots.gather(
            1,
            payload_slot_id[:, None, None].expand(
                -1, 1, raw_slots.shape[-1]
            ),
        ).squeeze(1)
        payload_relation_active = (
            payload_carrying.max(dim=1).values > 0.5
        )
        # The task option may temporarily switch to posture recovery while
        # the skill option must retain the physical payload relation.
        skill_relation_slot = torch.where(
            payload_relation_active.unsqueeze(1),
            payload_relation_slot,
            selected_raw_slot,
        )
        carrying = selected_raw_slot[
            :, TASK_SLOT_CARRYING_INDEX
        ].clamp(0.0, 1.0)
        carrying = torch.maximum(
            carrying, payload_relation_active.float()
        )
        interaction_active = (
            skill_relation_slot[:, TASK_SLOT_MANIPULATION_TYPE_INDEX]
            + skill_relation_slot[:, TASK_SLOT_DELIVERY_TYPE_INDEX]
        ).clamp(0.0, 1.0)
        delivery_active = skill_relation_slot[
            :, TASK_SLOT_DELIVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        interaction_phase_target = self.interaction_phase_distribution(
            skill_relation_slot
        )
        centered_phase_prior = torch.log(
            interaction_phase_target.clamp_min(1.0e-5)
        )
        centered_phase_prior = (
            centered_phase_prior
            - centered_phase_prior.mean(dim=-1, keepdim=True)
        )
        interaction_logits = interaction_logits + (
            self.interaction_phase_prior_gain
            * delivery_active.unsqueeze(1)
            * centered_phase_prior
        )
        skill_logits = (
            motion_logits.unsqueeze(2) + interaction_logits.unsqueeze(1)
        ).reshape(-1, ACTION_LAYOUT.skill_dim)
        skill_logits = skill_logits + self.skill_feasibility_gain * F.logsigmoid(
            feasibility_logits
        )
        global_context = self.last_global_context
        if global_context is None:
            raise RuntimeError("TACTIC mission context was not evaluated")
        current_skill = torch.round(
            global_context[:, EXECUTED_SKILL_INDEX]
            * float(ACTION_LAYOUT.skill_dim - 1)
        ).long().clamp(0, ACTION_LAYOUT.skill_dim - 1)
        current_motion = torch.div(
            current_skill,
            INTERACTION_SKILL_COUNT,
            rounding_mode="floor",
        )
        current_interaction = current_skill.remainder(
            INTERACTION_SKILL_COUNT
        )
        safety_demand = torch.maximum(
            (
                0.12 - global_context[:, SAFETY_MARGIN_INDEX]
            ) / 0.12,
            (
                0.10 - global_context[:, PREVIEW_MARGIN_INDEX]
            ) / 0.10,
        ).clamp(0.0, 1.0)
        task_end = ACTION_LAYOUT.task_dim
        object_end = task_end + ACTION_LAYOUT.object_dim
        task_subgoal = task_block[
            :,
            object_end : object_end + ACTION_LAYOUT.task_subgoal_dim,
        ]
        task_motion_target = self.task_subgoal_motion_components(
            task_subgoal
        )[0]
        task_motion_request = self.skill_tracking_request(
            global_context, task_motion_target
        )
        distance_demand = (
            selected_raw_slot[:, TASK_SLOT_DISTANCE_INDEX] / 0.30
        ).clamp(0.0, 1.0)
        progress_motion_demand = (
            task_motion_request[:, 0].abs() / 0.20
        ).clamp(0.0, 1.0)
        turning_motion_demand = (
            task_motion_request[:, 1].abs() / 0.50
        ).clamp(0.0, 1.0)
        objective_demand = torch.stack(
            (
                distance_demand * progress_motion_demand,
                distance_demand * turning_motion_demand,
                safety_demand,
                interaction_active
                * skill_relation_slot[
                    :, TASK_SLOT_REACHABILITY_INDEX
                ].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        progress_intent = torch.sigmoid(2.0 * task_subgoal[:, 6])
        precision_intent = torch.sigmoid(2.0 * task_subgoal[:, 7])
        intent_scale = torch.stack(
            (
                0.50 + progress_intent,
                torch.ones_like(progress_intent),
                torch.ones_like(progress_intent),
                0.50 + precision_intent,
            ),
            dim=-1,
        )
        objective_demand = objective_demand * intent_scale
        objective_demand = (
            objective_demand
            + 0.25 * torch.tanh(
                self.motion_objective_demand_head(skill_latent)
            )
        ).clamp_min(0.01)
        objective_demand = objective_demand / objective_demand.sum(
            dim=-1, keepdim=True
        )
        objective_basis = torch.softmax(
            self.motion_objective_basis, dim=-1
        )
        motion_objective_scores = (
            objective_demand @ objective_basis.transpose(0, 1)
        )
        motion_objective_scores = (
            motion_objective_scores
            - motion_objective_scores.mean(dim=1, keepdim=True)
        )
        skill_logits = skill_logits + self.motion_objective_gain * (
            motion_objective_scores.unsqueeze(2).expand(
                -1, -1, INTERACTION_SKILL_COUNT
            )
        ).reshape(-1, ACTION_LAYOUT.skill_dim)
        skill_outcomes, skill_utility = self._skill_candidate_outcomes(
            mission, task_latent, interaction_active, carrying
        )
        (
            skill_effects,
            skill_effect_confidence,
            skill_effect_utility,
        ) = self._skill_candidate_effects(
            mission,
            task_latent,
            interaction_active,
            global_context,
            selected_raw_slot,
        )
        motion_actions = self._motion_action_candidates(
            global_context, motion_request=task_motion_request
        )
        # Active options retain their identified breakaway magnitude. Model
        # uncertainty changes option selection, not action amplitude.
        motion_execution_actions = motion_actions
        (
            motion_execution,
            motion_execution_confidence,
        ) = self.predict_motion_execution(
            global_context, motion_execution_actions
        )
        motion_capacity = self.motion_target_capacity.clamp(0.05, 0.50)
        desired_vx = task_motion_request[:, 0].unsqueeze(1)
        desired_wz = task_motion_request[:, 1].unsqueeze(1)
        predicted_vx = 0.75 * motion_execution[:, :, 0]
        predicted_wz = 1.50 * motion_execution[:, :, 1]
        tracking_quality = (
            1.0
            - 0.60
            * (predicted_vx - desired_vx).abs()
            / motion_capacity[0]
            - 0.40
            * (predicted_wz - desired_wz).abs()
            / motion_capacity[1]
        ).clamp(-1.0, 1.0)
        turning_authority = (
            desired_wz.abs() / 0.020
        ).clamp(0.0, 1.0)
        progress_quality = (
            0.65
            * predicted_vx
            / motion_capacity[0]
            * desired_vx.sign()
            + 0.35
            * turning_authority
            * predicted_wz
            / motion_capacity[1]
            * desired_wz.sign()
        ).clamp(-1.0, 1.0)
        predicted_improvement = motion_execution[:, :, 2]
        predicted_control = 0.5 * (
            motion_execution[:, :, 3:7] + 1.0
        )
        predicted_robustness = (
            0.34 * predicted_control[:, :, 0]
            + 0.26 * predicted_control[:, :, 1]
            + 0.22 * predicted_control[:, :, 2]
            + 0.18 * predicted_control[:, :, 3]
        )
        learned_execution_reliability = (
            float(self.motion_execution_maturity)
            * (
                (motion_execution_confidence - 0.25) / 0.55
            ).clamp(0.0, 1.0)
        )
        control_margin = torch.minimum(
            global_context[:, SAFETY_MARGIN_INDEX],
            global_context[:, PREVIEW_MARGIN_INDEX],
        ).clamp(0.0, 1.0)
        identified_response_authority = (
            self.embodiment_response_prior_confidence
            * (0.15 + 0.85 * control_margin)
        ).unsqueeze(1)
        motion_execution_utility = (
            0.65
            * identified_response_authority
            * (0.55 * tracking_quality + 0.45 * progress_quality)
            + 0.35
            * learned_execution_reliability
            * (
                0.40 * predicted_improvement
                + 0.60 * predicted_robustness
            )
        )
        motion_execution_utility = (
            motion_execution_utility
            - motion_execution_utility.mean(dim=1, keepdim=True)
        )
        motion_epistemic_risk = (
            (1.0 - motion_execution_confidence.detach())
            * (1.0 - control_margin.unsqueeze(1))
        )
        skill_logits = (
            skill_logits
            + self.skill_outcome_gain * skill_utility.detach()
            + self.skill_effect_gain * skill_effect_utility.detach()
            + self.motion_execution_utility_gain
            * (
                motion_execution_utility.detach().unsqueeze(2).expand(
                    -1, -1, INTERACTION_SKILL_COUNT
                )
            ).reshape(-1, ACTION_LAYOUT.skill_dim)
        )
        skill_survival = self.last_skill_survival
        if skill_survival is None:
            raise RuntimeError(
                "TACTIC actor did not expose payload survival"
            )
        skill_survival_grid = skill_survival.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        survival_interaction_index = current_interaction.view(
            -1, 1, 1
        ).expand(-1, MOTION_SKILL_COUNT, 1)
        motion_survival = skill_survival_grid.gather(
            2, survival_interaction_index
        ).squeeze(2)
        centered_motion_survival = (
            motion_survival
            - motion_survival.mean(dim=1, keepdim=True)
        )
        survival_maturity = (
            float(self.payload_survival_maturity)
            if self.payload_survival_control_enabled
            else 0.0
        )
        skill_grid = skill_logits.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        identified_response = self.embodiment_response_prior(
            motion_execution_actions
        )
        (
            identified_response_prediction,
            identified_response_score,
            identified_response_authority,
        ) = self.identified_motion_response_score(
            identified_response,
            task_motion_request,
            control_margin,
        )
        # The chart is fitted from measured transitions.  Its supervised
        # calibration remains trainable, while policy gradients cannot distort
        # the chart merely to obtain a larger option logit.
        identified_response_correction = (
            self.embodiment_response_selection_gain
            * identified_response_authority.detach().unsqueeze(1)
            * identified_response_score.detach()
        )
        skill_grid = (
            skill_grid
            + identified_response_correction.unsqueeze(2)
        )
        recovery_role = raw_slots[
            :, :, TASK_SLOT_RECOVERY_TYPE_INDEX
        ].clamp(0.0, 1.0)
        recovery_task_mass = (
            task_probability * recovery_role
        ).sum(dim=1)
        recovery_payload_mass = raw_slots[
            :, :, TASK_SLOT_CARRYING_INDEX
        ].amax(dim=1).clamp(0.0, 1.0)
        recovery_pressure = (
            (
                self.control_recovery_pressure(global_context) - 0.20
            )
            / 0.60
        ).clamp(0.0, 1.0)
        recovery_adapter_gate = (
            torch.maximum(recovery_task_mass, recovery_payload_mass)
            * recovery_pressure
        )
        recovery_motion_adapter = self.last_recovery_motion_adapter
        recovery_interaction_adapter = (
            self.last_recovery_interaction_adapter
        )
        if (
            recovery_motion_adapter is None
            or recovery_motion_adapter.shape[0] != skill_grid.shape[0]
            or recovery_interaction_adapter is None
            or recovery_interaction_adapter.shape[0]
            != skill_grid.shape[0]
        ):
            (
                _,
                recovery_motion_adapter,
                recovery_interaction_adapter,
            ) = self.recovery_adapter_outputs(global_context, raw_slots)
        skill_grid = skill_grid + (
            self.recovery_adapter_motion_gain
            * recovery_adapter_gate[:, None, None]
            * recovery_motion_adapter.unsqueeze(2)
            + self.recovery_adapter_interaction_gain
            * recovery_adapter_gate[:, None, None]
            * recovery_interaction_adapter.unsqueeze(1)
        )
        self.last_recovery_motion_adapter = recovery_motion_adapter
        self.last_recovery_interaction_adapter = (
            recovery_interaction_adapter
        )
        self.last_recovery_adapter_gate = recovery_adapter_gate
        self.last_recovery_payload_mass = recovery_payload_mass
        motion_activity_demand = torch.maximum(
            task_motion_request[:, 0].abs() / 0.08,
            task_motion_request[:, 1].abs() / 0.16,
        ).clamp(0.0, 1.0)
        safe_progress_authority = (1.0 - safety_demand).clamp(0.0, 1.0)
        idle_motion_penalty = (
            4.0 * motion_activity_demand * safe_progress_authority
        )
        # Motion factor zero represents stabilize/hold.  It remains available
        # under a binding safety constraint, but cannot retain an unresolved
        # spatial task merely because the option was active at reset.
        idle_factor = (
            torch.arange(
                MOTION_SKILL_COUNT,
                device=skill_grid.device,
            )
            == 0
        ).to(skill_grid.dtype)
        skill_grid = (
            skill_grid
            - idle_motion_penalty[:, None, None]
            * idle_factor[None, :, None]
        )
        # Near a binding barrier, uncertain active options yield to the hold
        # option. In the interior they remain fully excited and identifiable.
        active_factor = 1.0 - idle_factor
        skill_grid = (
            skill_grid
            - 1.5
            * motion_epistemic_risk.unsqueeze(2)
            * active_factor[None, :, None]
        )
        inactive_interaction = (
            interaction_active < 0.5
        ).view(-1, 1, 1)
        neutral_interaction = torch.arange(
            INTERACTION_SKILL_COUNT,
            device=skill_logits.device,
        ).view(1, 1, -1) == 0
        skill_grid = skill_grid.masked_fill(
            inactive_interaction & (~neutral_interaction), -20.0
        )
        payload_preview_pressure = (
            (
                PAYLOAD_SKILL_PREVIEW_RESERVE
                - global_context[:, PREVIEW_MARGIN_INDEX]
            )
            / PAYLOAD_SKILL_PREVIEW_RESERVE
        ).clamp(0.0, 1.0)
        payload_safety_pressure = (
            (
                PAYLOAD_SKILL_SAFETY_RESERVE
                - global_context[:, SAFETY_MARGIN_INDEX]
            )
            / PAYLOAD_SKILL_SAFETY_RESERVE
        ).clamp(0.0, 1.0)
        payload_distance = (
            4.0 * skill_relation_slot[:, TASK_SLOT_DISTANCE_INDEX]
        ).clamp(0.0, 6.0)
        payload_transient_pressure, _ = self._payload_transient_demand(
            global_context,
            payload_distance,
        )
        payload_barrier_pressure = carrying * torch.maximum(
            torch.maximum(
                payload_preview_pressure,
                payload_safety_pressure,
            ),
            payload_transient_pressure,
        )
        effective_payload_pressure = payload_barrier_pressure
        if self.disable_payload_skill_barrier:
            effective_payload_pressure = torch.zeros_like(
                effective_payload_pressure
            )
        motion_effort = motion_execution_actions.detach().abs().mean(dim=-1)
        payload_candidate_risk = (
            0.55 * (1.0 - predicted_robustness.detach()).clamp(0.0, 1.0)
            + 0.25 * motion_epistemic_risk.detach()
            + 0.20 * motion_effort.clamp(0.0, 1.0)
        )
        centered_payload_risk = (
            payload_candidate_risk
            - payload_candidate_risk.mean(dim=1, keepdim=True)
        )
        payload_logit_correction = (
            PAYLOAD_SKILL_RISK_GAIN
            * effective_payload_pressure.unsqueeze(1)
            * centered_payload_risk
        )
        skill_grid = (
            skill_grid - payload_logit_correction.unsqueeze(2)
        )
        # A learned relation model may rank payload-preserving skills, but it
        # only receives decision authority when a control margin is binding.
        payload_survival_authority = (
            survival_maturity * effective_payload_pressure.detach()
        )
        survival_motion_correction = (
            self.payload_survival_gain
            * payload_survival_authority.unsqueeze(1)
            * centered_motion_survival.detach()
        )
        skill_grid = (
            skill_grid + survival_motion_correction.unsqueeze(2)
        )
        current_payload_risk = payload_candidate_risk.gather(
            1, current_motion.unsqueeze(1)
        ).squeeze(1)
        safer_payload_risk = payload_candidate_risk.min(dim=1).values
        payload_risk_advantage = (
            current_payload_risk - safer_payload_risk
        )
        payload_barrier_binding = (
            effective_payload_pressure * payload_risk_advantage
            > PAYLOAD_SKILL_SWITCH_MARGIN
        )
        current_motion_survival = motion_survival.detach().gather(
            1, current_motion.unsqueeze(1)
        ).squeeze(1)
        survival_alternative_mask = torch.ones_like(
            motion_survival, dtype=torch.bool
        )
        survival_alternative_mask.scatter_(
            1, current_motion.unsqueeze(1), False
        )
        best_alternative_survival = motion_survival.detach().masked_fill(
            ~survival_alternative_mask, -1.0
        ).max(dim=1).values
        payload_survival_advantage = (
            payload_survival_authority
            * (
                best_alternative_survival - current_motion_survival
            ).clamp_min(0.0)
        )
        payload_survival_binding = (
            payload_survival_advantage
            > 0.50 * PAYLOAD_SKILL_SWITCH_MARGIN
        )
        payload_projection = payload_logit_correction.abs().mean(dim=1)
        if self.forced_motion_skill_id is not None:
            forced_motion = int(self.forced_motion_skill_id)
            if forced_motion < 0 or forced_motion >= MOTION_SKILL_COUNT:
                raise ValueError("Forced motion skill is out of range")
            forced_mask = torch.ones_like(
                skill_grid, dtype=torch.bool
            )
            forced_mask[:, forced_motion, :] = False
            skill_grid = skill_grid.masked_fill(forced_mask, -20.0)
        skill_logits = skill_grid.reshape(-1, ACTION_LAYOUT.skill_dim)
        payload_motion_exit = payload_barrier_binding & (
            current_motion != 0
        )
        payload_skill_exit = (
            payload_motion_exit | payload_survival_binding
        )
        if apply_commitment and global_context is not None:
            committed = (
                (global_context[:, 2] > 0.01)
                & (global_context[:, 2] < 0.97)
                & (global_context[:, TERMINATION_STATE_SLICE.start + 1] < 0.97)
            )
            idle_deadlock = (
                (current_motion == 0)
                & (motion_activity_demand > 0.15)
                & (safety_demand < 0.80)
            )
            committed = committed & (~idle_deadlock)
            committed = committed & (~payload_skill_exit)
            commitment_mask = torch.ones_like(
                skill_logits, dtype=torch.bool
            )
            commitment_mask.scatter_(
                1, current_skill.unsqueeze(1), False
            )
            skill_logits = skill_logits.masked_fill(
                commitment_mask & committed.unsqueeze(1), -20.0
            )
        self.last_skill_feasibility = torch.sigmoid(feasibility_logits)
        self.last_interaction_active = interaction_active
        self.last_interaction_phase_target = interaction_phase_target
        self.last_skill_outcomes = skill_outcomes
        self.last_skill_outcome_utility = skill_utility
        self.last_skill_effects = skill_effects
        self.last_skill_effect_confidence = skill_effect_confidence
        self.last_skill_effect_utility = skill_effect_utility
        self.last_motion_objective_demand = objective_demand
        self.last_motion_objective_scores = motion_objective_scores
        self.last_task_motion_target = task_motion_target
        self.last_task_motion_request = task_motion_request
        self.last_motion_raw_action_candidates = motion_actions
        self.last_motion_action_candidates = motion_execution_actions
        self.last_motion_execution_prediction = motion_execution
        self.last_motion_execution_confidence = (
            motion_execution_confidence
        )
        self.last_embodiment_response_prediction = (
            identified_response_prediction
        )
        self.last_embodiment_response_score = identified_response_score
        self.last_embodiment_response_authority = (
            identified_response_authority
        )
        self.last_motion_execution_utility = motion_execution_utility
        self.last_motion_epistemic_risk = motion_epistemic_risk
        self.last_motion_activity_demand = motion_activity_demand
        self.last_idle_motion_penalty = idle_motion_penalty
        self.last_payload_skill_barrier_pressure = payload_barrier_pressure
        self.last_payload_transient_demand = (
            carrying * payload_transient_pressure
        )
        self.last_payload_skill_projection = payload_projection
        self.last_payload_survival_authority = (
            payload_survival_authority
        )
        self.last_payload_survival_projection = (
            survival_motion_correction.abs().mean(dim=1)
        )
        self.last_payload_survival_advantage = (
            payload_survival_advantage
        )
        self.last_payload_survival_exit = payload_skill_exit.float()
        return (
            skill_logits,
            (mission, task_latent, skill_latent, feasibility_logits),
        )

    def skill_detail_parameters(
        self,
        skill_code: torch.Tensor,
        skill_state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        mission, task_latent, skill_latent, _ = skill_state
        selected_skill = skill_code @ self.skill_codebook()
        detail = self.skill_detail_encoder(
            torch.cat((skill_latent, selected_skill), dim=-1)
        )
        continuous = self.skill_continuous_head(detail)
        param_end = ACTION_LAYOUT.skill_param_dim
        motion_skill = selected_skill[:, : self.skill_factor_dim]
        interaction_skill = selected_skill[:, self.skill_factor_dim :]
        effect_gate = torch.sigmoid(
            self.skill_effect_gate_head(task_latent)
        )
        motion_effect = (
            self.motion_skill_effect_head(motion_skill)
            * self.motion_parameter_mask
        )
        interaction_effect = (
            self.interaction_skill_effect_head(interaction_skill)
            * self.interaction_parameter_mask
        )
        skill_effect = self.skill_effect_scale * torch.tanh(
            effect_gate[:, :1]
            * motion_effect
            + effect_gate[:, 1:]
            * interaction_effect
        )
        parameter_raw = continuous[:, :param_end] + skill_effect
        parameters = 1.5 * torch.tanh(parameter_raw / 1.5)
        return (
            parameters,
            continuous[:, param_end:],
            (mission, task_latent, skill_latent),
        )

    def skill_parameters(
        self,
        task_block: torch.Tensor,
        mission_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        skill_logits, skill_state = self.skill_choice(
            task_block, mission_state
        )
        skill_code = torch.softmax(skill_logits, dim=-1)
        parameters, termination, hierarchy_state = (
            self.skill_detail_parameters(skill_code, skill_state)
        )
        return (
            skill_logits,
            parameters,
            termination,
            hierarchy_state,
        )

    def skill_mean(
        self,
        task_block: torch.Tensor,
        mission_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        skill_logits, parameters, termination, hierarchy_state = (
            self.skill_parameters(task_block, mission_state)
        )
        continuous = torch.cat((parameters, termination), dim=-1)
        skill_block = torch.cat((skill_logits, continuous), dim=-1)
        return skill_block, hierarchy_state

    def _skill_latent(
        self, skill_block: torch.Tensor
    ) -> torch.Tensor:
        skill_end = ACTION_LAYOUT.skill_dim
        param_end = skill_end + ACTION_LAYOUT.skill_param_dim
        skill_probability = skill_block[:, :skill_end].clamp_min(0.0)
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        selected_skill = skill_probability @ self.skill_codebook()
        termination_context = torch.sigmoid(
            skill_block[:, param_end:].detach()
        )
        skill_input = torch.cat(
            (
                selected_skill,
                torch.tanh(skill_block[:, skill_end:param_end]),
                termination_context,
            ),
            dim=-1,
        )
        return self.skill_conditioner(skill_input)

    @staticmethod
    def _deadzone_aware_gate(
        value: torch.Tensor,
        onset: float,
        full_scale: float,
    ) -> torch.Tensor:
        level = (
            (value.abs() - onset) / max(full_scale - onset, 1.0e-4)
        ).clamp(0.0, 1.0)
        level = level.square() * (3.0 - 2.0 * level)
        active = (value.abs() > onset).to(value.dtype)
        return (
            value.sign()
            * active
            * (0.65 + 0.35 * level)
        )

    @staticmethod
    def _spatial_motion_components(
        command_vx: torch.Tensor,
        command_vy: torch.Tensor,
        command_wz: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        travel_sign = torch.where(
            command_vx.abs() > 0.015,
            command_vx.sign(),
            torch.ones_like(command_vx),
        )
        lateral_heading = torch.atan2(
            travel_sign * command_vy, command_vx.abs() + 0.03
        )
        semantic_yaw = travel_sign * command_wz
        lateral_yaw = 0.75 * lateral_heading
        feasible_wz = (semantic_yaw + lateral_yaw).clamp(-1.50, 1.50)
        motion = torch.stack((command_vx, feasible_wz), dim=-1)
        return motion, semantic_yaw, lateral_yaw, travel_sign

    @staticmethod
    def _spatial_to_feasible_motion(
        command_vx: torch.Tensor,
        command_vy: torch.Tensor,
        command_wz: torch.Tensor,
    ) -> torch.Tensor:
        motion, _, _, _ = TACTICActor._spatial_motion_components(
            command_vx, command_vy, command_wz
        )
        return motion

    def task_subgoal_motion_components(
        self, task_subgoal: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project a spatial subgoal onto the B2W motion chart."""

        bounded_subgoal = task_subgoal.clamp(-1.5, 1.5)
        return self._spatial_motion_components(
            0.35 * bounded_subgoal[..., 0],
            0.18 * bounded_subgoal[..., 1],
            0.55 * bounded_subgoal[..., 2],
        )

    def feasible_motion_request(
        self, global_context: torch.Tensor
    ) -> torch.Tensor:
        """Read the B2W-feasible twist published by the command layer."""

        return torch.stack(
            (
                global_context[:, COMMAND_VX_INDEX],
                global_context[:, COMMAND_WZ_INDEX],
            ),
            dim=-1,
        )

    def task_motion_request(
        self, task_block: torch.Tensor
    ) -> torch.Tensor:
        """Map the current upper-layer subgoal into the lower skill chart."""

        task_end = ACTION_LAYOUT.task_dim
        object_end = task_end + ACTION_LAYOUT.object_dim
        subgoal = task_block[
            :,
            object_end : object_end + ACTION_LAYOUT.task_subgoal_dim,
        ]
        motion, _, _, _ = self.task_subgoal_motion_components(subgoal)
        return motion

    def skill_tracking_request(
        self,
        global_context: torch.Tensor,
        task_motion_target: torch.Tensor,
    ) -> torch.Tensor:
        """Express an upper-level target as a feasible lower-level residual."""

        capacity = self.motion_target_capacity.clamp(0.08, 0.50)
        target = torch.stack(
            (
                task_motion_target[:, 0].clamp(
                    -capacity[0], capacity[0]
                ),
                task_motion_target[:, 1].clamp(
                    -capacity[1], capacity[1]
                ),
            ),
            dim=-1,
        )
        state = torch.stack(
            (
                global_context[:, BASE_VX_INDEX],
                global_context[:, BASE_WZ_INDEX],
            ),
            dim=-1,
        )
        residual = target - state
        tolerance = residual.new_tensor((0.020, 0.035))
        residual = torch.where(
            residual.abs() > tolerance,
            residual,
            torch.zeros_like(residual),
        )
        # Spatial approach tasks do not request reverse travel merely because
        # the robot briefly exceeds the target speed.  Turning remains signed
        # so the same skill can damp an angular overshoot.
        forward = torch.where(
            task_motion_target[:, 0] >= 0.0,
            residual[:, 0].clamp_min(0.0),
            residual[:, 0],
        )
        return torch.stack((forward, residual[:, 1]), dim=-1)

    def _execution_motion_request(
        self, global_context: torch.Tensor
    ) -> torch.Tensor:
        """Use the current task intent after the environment leaves settling."""

        published_request = self.feasible_motion_request(global_context)
        current_request = self.last_task_motion_request
        if (
            current_request is None
            or current_request.shape[0] != global_context.shape[0]
        ):
            return published_request
        command_active = (
            (global_context[:, COMMAND_VX_INDEX].abs() > 0.005)
            | (global_context[:, COMMAND_VY_INDEX].abs() > 0.005)
            | (global_context[:, COMMAND_WZ_INDEX].abs() > 0.010)
        )
        return torch.where(
            command_active.unsqueeze(1),
            current_request,
            published_request,
        )

    def _motion_action_candidates(
        self,
        global_context: torch.Tensor,
        motion_request: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode all shared motion options into platform actions."""

        if motion_request is None:
            motion_request = self.feasible_motion_request(global_context)
        drive_gate = self._deadzone_aware_gate(
            motion_request[:, 0], 0.010, 0.25
        )
        yaw_gate = self._deadzone_aware_gate(
            motion_request[:, 1], 0.020, 0.35
        )
        positive_yaw = yaw_gate.clamp_min(0.0)
        negative_yaw = (-yaw_gate).clamp_min(0.0)
        gains = self.motion_kinematic_gain.clamp(0.35, 1.35)
        learned_authority = self.motion_action_capacity.clamp(
            0.08 * self.wheel_action_limit, self.wheel_action_limit
        )
        breakaway_authority = self.wheel_breakaway_action.clamp(
            0.0, self.wheel_action_limit
        )
        authority = torch.maximum(
            learned_authority, breakaway_authority
        )
        basis = self.embodiment_motion_basis.clamp(-1.5, 1.5)
        action = (
            drive_gate[:, None, None]
            * authority[None, None, 0:1]
            * gains[None, :, 0:1]
            * basis[None, :, 0, :]
            + positive_yaw[:, None, None]
            * authority[None, None, 1:2]
            * gains[None, :, 1:2]
            * basis[None, :, 1, :]
            + negative_yaw[:, None, None]
            * authority[None, None, 1:2]
            * gains[None, :, 1:2]
            * basis[None, :, 2, :]
        )
        # Keep combined translation and yaw requests inside one transient
        # action envelope instead of letting two feasible basis vectors add
        # into an actuator spike.
        blend_norm = (
            drive_gate.abs() + yaw_gate.abs()
        ).clamp_min(1.0)
        action = action / blend_norm[:, None, None]

        action = action.clamp(
            -self.wheel_action_limit, self.wheel_action_limit
        )
        return action

    @staticmethod
    def motion_parameter_authority(
        skill_parameters: torch.Tensor,
        motion_request: torch.Tensor,
    ) -> torch.Tensor:
        """Decode sampled motion-skill parameters into execution authority."""

        if skill_parameters.shape[1] < 2:
            raise ValueError("Motion skills require two continuous parameters")
        parameter = torch.tanh(skill_parameters[:, :2])
        drive_demand = motion_request[:, 0].abs()
        turn_demand = 0.35 * motion_request[:, 1].abs()
        drive_weight = drive_demand / (
            drive_demand + turn_demand + 1.0e-4
        )
        blended_parameter = (
            drive_weight * parameter[:, 0]
            + (1.0 - drive_weight) * parameter[:, 1]
        )
        return (1.0 + 0.15 * blended_parameter).clamp(0.85, 1.15)

    def _motion_execution_features(
        self,
        global_context: torch.Tensor,
        wheel_action: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, bool]:
        squeeze_option = wheel_action.ndim == 2
        if squeeze_option:
            wheel_action = wheel_action.unsqueeze(1)
        if wheel_action.ndim != 3 or wheel_action.shape[-1] != 4:
            raise ValueError("Wheel actions must have shape [B,4] or [B,K,4]")
        motion_request = self.feasible_motion_request(global_context)
        state = torch.stack(
            (
                motion_request[:, 0],
                motion_request[:, 1],
                global_context[:, BASE_VX_INDEX],
                global_context[:, BASE_WZ_INDEX],
                global_context[:, SAFETY_MARGIN_INDEX],
                global_context[:, PREVIEW_MARGIN_INDEX],
                global_context[:, CLF_DECREASE_INDEX],
                global_context[:, DISTURBANCE_QUALITY_INDEX],
                global_context[:, BASE_HEIGHT_INDEX],
                global_context[:, BASE_TILT_INDEX],
                global_context[:, SUPPORT_COUNT_INDEX],
                global_context[:, CURRICULUM_LEVEL_INDEX],
            ),
            dim=-1,
        )
        state = state.unsqueeze(1).expand(
            -1, wheel_action.shape[1], -1
        )
        features = torch.cat(
            (
                wheel_action / self.wheel_action_limit,
                state,
            ),
            dim=-1,
        )
        return (
            features.reshape(-1, features.shape[-1]),
            wheel_action.shape[0],
            wheel_action.shape[1],
            squeeze_option,
        )

    @staticmethod
    def _detached_sequential(
        module: nn.Sequential, value: torch.Tensor
    ) -> torch.Tensor:
        """Keep action gradients while freezing a learned dynamics model."""

        for layer in module:
            if isinstance(layer, nn.Linear):
                bias = (
                    None if layer.bias is None else layer.bias.detach()
                )
                value = F.linear(value, layer.weight.detach(), bias)
                continue
            if tuple(layer.parameters(recurse=False)):
                raise TypeError(
                    "Detached successor rollout only supports parameter-free "
                    f"layers after Linear, got {type(layer).__name__}"
                )
            value = layer(value)
        return value

    def predict_motion_execution(
        self,
        global_context: torch.Tensor,
        wheel_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict measured short-horizon effects of platform actions."""

        features, batch_size, option_count, squeeze_option = (
            self._motion_execution_features(global_context, wheel_action)
        )
        hidden = self.motion_execution_encoder(
            features
        )
        learned_effect = torch.tanh(
            self.motion_execution_head(hidden)
        ).reshape(
            batch_size, option_count, MOTION_EXECUTION_EFFECT_DIM
        )
        response_prior = self.embodiment_response_prior(wheel_action)
        if response_prior.ndim == 2:
            response_prior = response_prior.unsqueeze(1)
        effect = learned_effect.clone()
        effect[:, :, :2] = (
            response_prior + 0.35 * learned_effect[:, :, :2]
        ).clamp(-1.0, 1.0)
        self.last_embodiment_response_prior = response_prior
        confidence = torch.sigmoid(
            self.motion_execution_confidence_head(hidden)
        ).reshape(batch_size, option_count)
        if squeeze_option:
            return effect[:, 0], confidence[:, 0]
        return effect, confidence

    def embodiment_response_prior(
        self,
        wheel_action: torch.Tensor,
        *,
        detach_chart: bool = False,
    ) -> torch.Tensor:
        """Map wheel effort to the identified signed body response."""

        if wheel_action.ndim not in (2, 3) or wheel_action.shape[-1] != 4:
            raise ValueError(
                "Wheel actions must have shape [B,4] or [B,K,4]"
            )
        matrix = self.embodiment_response_matrix.clamp(-0.50, 0.50)
        if detach_chart:
            matrix = matrix.detach()
        normalized_action = wheel_action / (
            self.embodiment_response_action.clamp_min(1.0)
        )
        return torch.matmul(normalized_action, matrix).clamp(-1.0, 1.0)

    def identified_motion_response_score(
        self,
        normalized_response: torch.Tensor,
        motion_request: torch.Tensor,
        control_margin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank motion options by their identified signed body response."""

        if normalized_response.ndim != 3:
            raise ValueError(
                "Motion response must have shape [batch, option, 2]"
            )
        if normalized_response.shape[-1] != 2:
            raise ValueError("Motion response must contain vx and wz")
        if motion_request.ndim != 2 or motion_request.shape[-1] != 2:
            raise ValueError("Motion request must have shape [batch, 2]")
        if control_margin.ndim != 1:
            raise ValueError("Control margin must have shape [batch]")

        response_scale = normalized_response.new_tensor((0.75, 1.50))
        physical_response = normalized_response * response_scale
        capacity = self.motion_target_capacity.clamp(0.05, 0.50)
        normalized_demand = (
            motion_request.abs() / capacity.unsqueeze(0)
        ).clamp(0.0, 1.0)
        active_demand = torch.stack(
            (
                (
                    motion_request[:, 0].abs() / 0.010
                ).clamp(0.0, 1.0),
                (
                    motion_request[:, 1].abs() / 0.020
                ).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        component_weight = (
            normalized_demand
            / normalized_demand.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        )
        component_weight = component_weight * active_demand
        component_weight = (
            component_weight
            / component_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        )

        signed_alignment = torch.tanh(
            physical_response
            / (0.20 * capacity).view(1, 1, 2)
        )
        signed_alignment = (
            signed_alignment
            * motion_request.sign().unsqueeze(1)
            * active_demand.unsqueeze(1)
        )
        alignment_score = (
            signed_alignment * component_weight.unsqueeze(1)
        ).sum(dim=-1)
        tracking_score = (
            1.0
            - (
                physical_response - motion_request.unsqueeze(1)
            ).abs()
            / capacity.view(1, 1, 2)
        ).clamp(-1.0, 1.0)
        tracking_score = (
            tracking_score * component_weight.unsqueeze(1)
        ).sum(dim=-1)
        response_score = (
            0.75 * alignment_score + 0.25 * tracking_score
        )
        response_score = (
            response_score
            - response_score.mean(dim=1, keepdim=True)
        )

        motion_activity = normalized_demand.amax(dim=1)
        response_authority = (
            self.embodiment_response_prior_confidence
            * motion_activity
            * (0.25 + 0.75 * control_margin.clamp(0.0, 1.0))
        )
        return physical_response, response_score, response_authority

    def predict_motion_execution_frozen(
        self,
        global_context: torch.Tensor,
        wheel_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out candidate skills through a frozen successor model."""

        features, batch_size, option_count, squeeze_option = (
            self._motion_execution_features(global_context, wheel_action)
        )
        hidden = self._detached_sequential(
            self.motion_execution_encoder, features
        )
        bias = (
            None
            if self.motion_execution_head.bias is None
            else self.motion_execution_head.bias.detach()
        )
        learned_effect = torch.tanh(
            F.linear(
                hidden,
                self.motion_execution_head.weight.detach(),
                bias,
            )
        ).reshape(
            batch_size, option_count, MOTION_EXECUTION_EFFECT_DIM
        )
        response_prior = self.embodiment_response_prior(
            wheel_action, detach_chart=True
        )
        if response_prior.ndim == 2:
            response_prior = response_prior.unsqueeze(1)
        effect = learned_effect.clone()
        effect[:, :, :2] = (
            response_prior + 0.35 * learned_effect[:, :, :2]
        ).clamp(-1.0, 1.0)
        confidence_bias = (
            None
            if self.motion_execution_confidence_head.bias is None
            else self.motion_execution_confidence_head.bias.detach()
        )
        confidence = torch.sigmoid(
            F.linear(
                hidden,
                self.motion_execution_confidence_head.weight.detach(),
                confidence_bias,
            )
        ).reshape(batch_size, option_count)
        if squeeze_option:
            return effect[:, 0], confidence[:, 0]
        return effect, confidence

    def _wheel_skill_action(
        self,
        conditioner: torch.Tensor,
        skill_block: torch.Tensor,
    ) -> torch.Tensor:
        global_context = self.last_global_context
        if global_context is None:
            zeros = conditioner.new_zeros(conditioner.shape[0], 4)
            self.last_wheel_prior = zeros
            self.last_wheel_residual = zeros
            self.last_wheel_skill_gate = zeros[:, 0]
            self.last_wheel_prior_gate = zeros[:, 0]
            self.last_wheel_control_authority = zeros[:, 0]
            return zeros

        skill_probability = skill_block[:, : ACTION_LAYOUT.skill_dim].clamp_min(
            0.0
        )
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        motion_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=2)
        kinematic_gain = motion_probability @ self.motion_kinematic_gain.clamp(
            0.35, 1.35
        )

        state_vx = global_context[:, BASE_VX_INDEX]
        state_wz = global_context[:, BASE_WZ_INDEX]
        motion_request = self._execution_motion_request(global_context)
        skill_parameter_start = ACTION_LAYOUT.skill_dim
        skill_parameter_end = (
            skill_parameter_start + ACTION_LAYOUT.skill_param_dim
        )
        skill_authority = self.motion_parameter_authority(
            skill_block[:, skill_parameter_start:skill_parameter_end],
            motion_request,
        )
        command_vx = motion_request[:, 0] * kinematic_gain[:, 0]
        command_wz = motion_request[:, 1] * kinematic_gain[:, 1]
        candidates = self._motion_action_candidates(
            global_context, motion_request=motion_request
        )
        wheel_prior = torch.einsum(
            "bm,bmw->bw", motion_probability, candidates
        )
        safety = global_context[:, SAFETY_MARGIN_INDEX].clamp(0.0, 1.0)
        preview = global_context[:, PREVIEW_MARGIN_INDEX].clamp(0.0, 1.0)
        clf_score = global_context[:, CLF_DECREASE_INDEX].clamp(0.0, 1.0)
        disturbance = global_context[:, DISTURBANCE_QUALITY_INDEX].clamp(
            0.0, 1.0
        )
        support = global_context[:, SUPPORT_COUNT_INDEX].clamp(0.0, 1.0)

        wheel_features = torch.stack(
            (
                command_vx,
                command_wz,
                state_vx,
                state_wz,
                command_vx - state_vx,
                command_wz - state_wz,
                safety,
                preview,
                clf_score,
                disturbance,
                global_context[:, BASE_HEIGHT_INDEX],
                global_context[:, BASE_TILT_INDEX],
                support,
                global_context[:, CURRICULUM_LEVEL_INDEX],
            ),
            dim=-1,
        )
        wheel_hidden = self.wheel_residual_encoder(
            torch.cat((conditioner, wheel_features), dim=-1)
        )
        wheel_residual = (
            0.25
            * self.wheel_residual_gain
            * self.wheel_action_limit
            * torch.tanh(self.wheel_residual_head(wheel_hidden))
        )
        option_gate = torch.sigmoid(
            self.wheel_skill_gate_logit.unsqueeze(0)
            + 2.0 * torch.tanh(
                self.wheel_skill_gate_head(wheel_hidden)
            )
        )
        selected_gate = torch.einsum(
            "bm,bm->b", motion_probability, option_gate
        )
        reference_gate = torch.sigmoid(
            global_context.new_tensor(-0.50)
        )
        relative_option_gate = (
            selected_gate / reference_gate.clamp_min(1.0e-4)
        ).clamp(0.50, 1.50)
        control_margin = torch.minimum(safety, preview)
        barrier_interior = (
            (control_margin - 0.03) / 0.15
        ).clamp(0.0, 1.0)
        barrier_interior = barrier_interior.square() * (
            3.0 - 2.0 * barrier_interior
        )
        control_authority = barrier_interior
        wheel_action_residual = (
            skill_authority.unsqueeze(1)
            * control_authority.unsqueeze(1)
            * relative_option_gate.unsqueeze(1)
            * wheel_prior
            + skill_authority.unsqueeze(1)
            * control_authority.unsqueeze(1)
            * wheel_residual
        ).clamp(
            -self.wheel_action_limit,
            self.wheel_action_limit,
        )
        self.last_wheel_prior = wheel_prior
        self.last_wheel_residual = wheel_residual
        self.last_wheel_skill_gate = relative_option_gate
        self.last_wheel_prior_gate = relative_option_gate
        self.last_wheel_control_authority = control_authority
        self.last_motion_kinematic_gain = kinematic_gain
        return wheel_action_residual

    def _support_skill_action(
        self,
        physical_latent: torch.Tensor,
        conditioner: torch.Tensor,
        skill_block: torch.Tensor,
        nominal_leg_action: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a skill-conditioned, embodiment-shared support stance."""

        global_context = self.last_global_context
        if global_context is None:
            raise RuntimeError("TACTIC mission context was not evaluated")
        skill_probability = skill_block[
            :, : ACTION_LAYOUT.skill_dim
        ].clamp_min(0.0)
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        motion_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=2)

        motion_request = self._execution_motion_request(global_context)
        support_state = torch.stack(
            (
                motion_request[:, 0],
                motion_request[:, 1],
                global_context[:, BASE_VX_INDEX],
                global_context[:, BASE_WZ_INDEX],
                global_context[:, SAFETY_MARGIN_INDEX],
                global_context[:, PREVIEW_MARGIN_INDEX],
                global_context[:, CLF_DECREASE_INDEX],
                global_context[:, DISTURBANCE_QUALITY_INDEX],
                global_context[:, BASE_HEIGHT_INDEX],
                global_context[:, BASE_TILT_INDEX],
                global_context[:, SUPPORT_COUNT_INDEX],
                global_context[:, CURRICULUM_LEVEL_INDEX],
            ),
            dim=-1,
        )
        hidden = self.support_skill_encoder(
            torch.cat(
                (physical_latent, conditioner, support_state), dim=-1
            )
        )
        adaptation = self.support_skill_adaptation_gain * torch.tanh(
            self.support_reference_head(hidden)
        ).reshape(-1, MOTION_SKILL_COUNT, 12)
        references = (
            self.motion_support_basis.clamp(-0.70, 0.90).unsqueeze(0)
            + adaptation
        ).clamp(-1.0, 1.2)
        gates = torch.sigmoid(
            self.support_gate_logit.clamp(-6.0, 6.0).unsqueeze(0)
            + 0.50 * torch.tanh(self.support_gate_head(hidden))
        )
        selected_reference = torch.einsum(
            "bm,bmj->bj", motion_probability, references
        )
        selected_gate = torch.einsum(
            "bm,bm->b", motion_probability, gates
        )
        if self.support_gate_override is not None:
            selected_gate = torch.full_like(
                selected_gate,
                max(0.0, min(1.0, self.support_gate_override)),
            )

        command_activity = (
            motion_request[:, 0].abs()
            + 0.375 * motion_request[:, 1].abs()
        ) / 0.04
        command_activity = command_activity.clamp(0.0, 1.0)
        command_activity = command_activity.square() * (
            3.0 - 2.0 * command_activity
        )
        support_gate = command_activity * selected_gate
        residual = self.support_skill_residual_gain * torch.tanh(
            self.support_residual_head(hidden)
        )
        residual = residual * command_activity.unsqueeze(1)
        leg_action = torch.lerp(
            nominal_leg_action,
            selected_reference,
            support_gate.unsqueeze(1),
        )
        leg_action = leg_action + residual
        self.last_support_reference = selected_reference
        self.last_support_gate = support_gate
        self.last_support_residual = residual
        return leg_action

    def physical_mean(
        self,
        obs: torch.Tensor,
        task_block: torch.Tensor,
        skill_block: torch.Tensor,
        hierarchy_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mission, task_latent, _ = hierarchy_state
        skill_latent = self._skill_latent(skill_block)
        hierarchy_latent = torch.cat(
            (mission, task_latent, skill_latent), dim=-1
        )
        conditioner = self.physical_conditioner(hierarchy_latent)
        gamma, beta = self.film_head(conditioner).chunk(2, dim=-1)

        physical_obs = obs[:, : self.physical_core_obs_dim]
        physical_latent = self.physical_backbone(physical_obs)
        global_context = self.last_global_context
        if global_context is None:
            raise RuntimeError("TACTIC mission context was not evaluated")
        if self.stability_teacher_only:
            # This path is deliberately a direct copy of the migrated ZYB
            # physical policy.  The gripper branch below remains available,
            # while the 12 leg and 4 wheel actions stay teacher-controlled.
            physical = self.physical_head(physical_latent)
            self.last_support_reference = physical[:, :12]
            self.last_support_gate = torch.zeros(
                physical.shape[0], device=physical.device, dtype=physical.dtype
            )
            self.last_support_residual = torch.zeros_like(physical[:, :12])
            self.last_nominal_wheel_action = physical[:, 12:16]
            self.last_wheel_skill_action = physical[:, 12:16]
            self.last_wheel_skill_gate = torch.zeros(
                physical.shape[0], device=physical.device, dtype=physical.dtype
            )
        else:
            conditioned = physical_latent * (
                1.0 + self.conditioner_gain * torch.tanh(gamma)
            )
            conditioned = conditioned + self.conditioner_gain * torch.tanh(beta)
            physical = self.physical_head(conditioned)
            physical = physical + self.physical_residual_gain * torch.tanh(
                self.physical_residual_head(conditioner)
            )
            # The migrated ZYB-v0 core remains the support-policy anchor. Motion
            # options add shared support and wheel decoders because the original
            # checkpoint does not respond to commanded base velocity.
            self.last_support_reference = physical[:, :12]
            leg_action = self._support_skill_action(
                physical_latent,
                conditioner,
                skill_block,
                physical[:, :12],
            )
            self.last_nominal_wheel_action = physical[:, 12:16]
            motion_request = self._execution_motion_request(global_context)
            skill_wheel_action = self._wheel_skill_action(
                conditioner, skill_block
            )
            motion_active = (
                (motion_request[:, 0].abs() > 0.010)
                | (motion_request[:, 1].abs() > 0.020)
            ).unsqueeze(1)
            wheel_action = torch.where(
                motion_active,
                skill_wheel_action,
                torch.zeros_like(physical[:, 12:16]),
            ).clamp(-self.wheel_action_limit, self.wheel_action_limit)
            physical = torch.cat(
                (leg_action, wheel_action, physical[:, 16:]), dim=-1
            )
        interaction_intent = torch.tanh(
            skill_block[:, ACTION_LAYOUT.skill_dim + 2 : ACTION_LAYOUT.skill_dim + 3]
        )
        skill_probability = skill_block[
            :, : ACTION_LAYOUT.skill_dim
        ].clamp_min(0.0)
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        interaction_probability = skill_probability.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        ).sum(dim=1)
        raw_interaction_probability = interaction_probability
        capture_feasibility = self.last_interaction_capture_feasibility
        release_feasibility = self.last_interaction_release_feasibility
        hold_evidence = self.last_interaction_hold_evidence
        raw_slots = self.last_raw_task_slots
        if raw_slots is not None:
            payload_hold = raw_slots[
                :, :, TASK_SLOT_CARRYING_INDEX
            ].amax(dim=1)
            if hold_evidence is not None:
                hold_evidence = torch.maximum(
                    hold_evidence, payload_hold
                )
            if capture_feasibility is not None:
                capture_feasibility = torch.maximum(
                    capture_feasibility, payload_hold
                )
        if release_feasibility is not None:
            barrier_quality = global_context[
                :, SAFETY_MARGIN_INDEX : PREVIEW_MARGIN_INDEX + 1
            ].amin(dim=1)
            barrier_gate = torch.sigmoid(
                20.0
                * (
                    barrier_quality
                    - INTERACTION_HARD_CONTROL_FLOOR
                )
            )
            settle_vx_gate = torch.sigmoid(
                25.0
                * (
                    RELEASE_SETTLE_VX
                    - global_context[:, BASE_VX_INDEX].abs()
                )
            )
            settle_wz_gate = torch.sigmoid(
                12.0
                * (
                    RELEASE_SETTLE_WZ
                    - global_context[:, BASE_WZ_INDEX].abs()
                )
            )
            release_control_gate = (
                barrier_gate
                * torch.minimum(settle_vx_gate, settle_wz_gate)
            )
            release_feasibility = (
                release_feasibility * release_control_gate
            )
            self.last_interaction_release_control_gate = (
                release_control_gate
            )
        if (
            capture_feasibility is not None
            and release_feasibility is not None
            and hold_evidence is not None
        ):
            approach = interaction_probability[:, 0]
            secure = interaction_probability[:, 1]
            release = interaction_probability[:, 2]
            blocked_secure = secure * (1.0 - capture_feasibility)
            blocked_release = release * (1.0 - release_feasibility)
            blocked_approach = (
                approach
                * hold_evidence
                * (1.0 - release_feasibility)
            )
            projected_approach = (
                approach
                - blocked_approach
                + blocked_secure
                + blocked_release * (1.0 - hold_evidence)
            )
            projected_secure = (
                secure * capture_feasibility
                + blocked_approach
                + blocked_release * hold_evidence
            )
            projected_release = release * release_feasibility
            interaction_probability = torch.stack(
                (
                    projected_approach,
                    projected_secure,
                    projected_release,
                ),
                dim=-1,
            )
            interaction_probability = interaction_probability / (
                interaction_probability.sum(dim=-1, keepdim=True)
                .clamp_min(1.0e-6)
            )
        self.last_interaction_phase_raw_probability = (
            raw_interaction_probability
        )
        self.last_interaction_phase_projected_probability = (
            interaction_probability
        )
        phase_margin = (
            interaction_probability[:, 1]
            - interaction_probability[:, 0]
            - self.interaction_gripper_release_gain
            * interaction_probability[:, 2]
        ).unsqueeze(1)
        phase_gain = (
            self.interaction_gripper_phase_gain
            + 0.5 * torch.tanh(self.interaction_gripper_basis[1])
        )
        semantic_gripper = phase_gain * phase_margin
        phase_confidence = interaction_probability.max(
            dim=1, keepdim=True
        ).values
        residual_authority = (1.0 - phase_confidence).clamp(0.05, 0.70)
        unconstrained_residual = (
            self.gripper_head(hierarchy_latent)
            + self.interaction_gripper_gain * interaction_intent
        )
        gripper_residual = (
            self.interaction_gripper_residual_limit
            * residual_authority
            * torch.tanh(unconstrained_residual)
        )
        gripper = semantic_gripper + gripper_residual
        self.last_gripper_semantic_logit = semantic_gripper
        self.last_gripper_residual = gripper_residual
        self.last_gripper_residual_authority = residual_authority
        self.last_gripper_logit = gripper
        physical = torch.cat((physical[:, :16], gripper), dim=-1)
        self.last_control_prediction = self.control_prediction_head(
            hierarchy_latent
        )
        return physical

    def physical_executor_named_parameters(self):
        """Yield parameters that directly map a selected skill to actuation."""

        seen: set[int] = set()
        for module_name in self.PHYSICAL_EXECUTOR_MODULE_NAMES:
            module = getattr(self, module_name)
            for parameter_name, parameter in module.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                yield f"{module_name}.{parameter_name}", parameter
        for parameter_name in self.PHYSICAL_EXECUTOR_PARAMETER_NAMES:
            parameter = getattr(self, parameter_name)
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield parameter_name, parameter

    def set_physical_executor_trainable(self, trainable: bool) -> int:
        """Set the direct actuator decoder boundary and return its size."""

        parameter_count = 0
        for _, parameter in self.physical_executor_named_parameters():
            parameter.requires_grad_(trainable)
            parameter_count += parameter.numel()
        return parameter_count

    def set_recovery_adapter_trainable(self, trainable: bool) -> int:
        """Set the event-conditioned recovery adapter boundary."""

        parameter_count = 0
        for module_name in self.RECOVERY_ADAPTER_MODULE_NAMES:
            module = getattr(self, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                parameter_count += parameter.numel()
        return parameter_count

    def set_decomposition_trainable(self, trainable: bool) -> int:
        """Set the task-detail and skill adaptation boundary."""

        parameter_count = 0
        for module_name in self.DECOMPOSITION_MODULE_NAMES:
            module = getattr(self, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                parameter_count += parameter.numel()
        return parameter_count

    def set_motion_selector_trainable(self, trainable: bool) -> int:
        """Set the isolated motion-skill selection boundary."""

        parameter_count = 0
        for module_name in self.MOTION_SELECTOR_MODULE_NAMES:
            module = getattr(self, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                parameter_count += parameter.numel()
        return parameter_count

    def set_interaction_selector_trainable(self, trainable: bool) -> int:
        """Set the isolated interaction-phase selection boundary."""

        parameter_count = 0
        for module_name in self.INTERACTION_SELECTOR_MODULE_NAMES:
            module = getattr(self, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                parameter_count += parameter.numel()
        return parameter_count

    def set_payload_survival_trainable(self, trainable: bool) -> int:
        """Set the payload-survival calibration boundary."""

        parameter_count = 0
        for module in (
            self.payload_survival_encoder,
            self.payload_survival_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                parameter_count += parameter.numel()
        return parameter_count


class TACTICActorCritic(ActorCritic):
    """RSL-RL policy with exact autoregressive PPO log probabilities."""

    OPTIONAL_RECOVERY_ADAPTER_PREFIXES = (
        "actor.recovery_adapter_encoder.",
        "actor.recovery_task_adapter_head.",
        "actor.recovery_motion_adapter_head.",
        "actor.recovery_interaction_adapter_head.",
        "actor.skill_survival_head.",
        "actor.payload_survival_encoder.",
        "actor.payload_survival_head.",
        "actor.payload_survival_updates",
    )

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        hierarchy_context_group: str = "hierarchy_context",
        mission_latent_dim: int = 128,
        slot_latent_dim: int = 128,
        task_embedding_dim: int = 32,
        skill_embedding_dim: int = 48,
        conditioner_gain: float = 0.12,
        physical_residual_gain: float = 0.16,
        support_skill_residual_gain: float = 0.10,
        support_skill_adaptation_gain: float = 0.12,
        physical_core_obs_dim: int = 876,
        skill_feasibility_gain: float = 0.35,
        skill_effect_scale: float = 0.85,
        interaction_gripper_gain: float = 2.0,
        interaction_gripper_phase_gain: float = 4.5,
        interaction_gripper_release_gain: float = 6.0,
        interaction_gripper_residual_limit: float = 1.0,
        interaction_phase_prior_gain: float = 0.85,
        release_target_radius: float = RELEASE_TARGET_RADIUS,
        task_residual_scale: float = 0.22,
        transition_latent_dim: int = 64,
        transition_temperature: float = 0.12,
        task_affordance_gain: float = 1.00,
        task_outcome_gain: float = 0.80,
        recovery_task_margin_gain: float = 0.60,
        recovery_adapter_task_gain: float = 2.00,
        recovery_adapter_motion_gain: float = 1.50,
        recovery_adapter_interaction_gain: float = 2.00,
        task_outcome_warmup_updates: int = 64,
        skill_outcome_gain: float = 0.25,
        payload_survival_gain: float = 1.20,
        payload_survival_warmup_updates: int = 64,
        skill_effect_gain: float = 1.50,
        constraint_utility_gain: float = 0.80,
        motion_objective_gain: float = 1.20,
        motion_execution_utility_gain: float = 0.85,
        embodiment_response_selection_gain: float = 2.40,
        embodiment_response_prior_confidence: float = 0.70,
        wheel_action_scale: float = 2.00,
        wheel_track_width: float = 0.50,
        wheel_breakaway_action: float = 3.0,
        wheel_turn_breakaway_action: float = 3.5,
        wheel_residual_gain: float = 0.8,
        wheel_action_limit: float = 24.0,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        **kwargs,
    ):
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )
        actor_obs_dim = sum(
            obs[name].shape[-1] for name in obs_groups["policy"]
        )
        self.hierarchy_context_group = hierarchy_context_group
        context_dim = int(obs[hierarchy_context_group].shape[-1])
        self.actor = TACTICActor(
            num_obs=actor_obs_dim,
            context_dim=context_dim,
            num_actions=num_actions,
            physical_hidden_dims=actor_hidden_dims,
            activation=activation,
            mission_latent_dim=mission_latent_dim,
            slot_latent_dim=slot_latent_dim,
            task_embedding_dim=task_embedding_dim,
            skill_embedding_dim=skill_embedding_dim,
            conditioner_gain=conditioner_gain,
            physical_residual_gain=physical_residual_gain,
            support_skill_residual_gain=support_skill_residual_gain,
            support_skill_adaptation_gain=support_skill_adaptation_gain,
            physical_core_obs_dim=physical_core_obs_dim,
            skill_feasibility_gain=skill_feasibility_gain,
            skill_effect_scale=skill_effect_scale,
            interaction_gripper_gain=interaction_gripper_gain,
            interaction_gripper_phase_gain=interaction_gripper_phase_gain,
            interaction_gripper_release_gain=(
                interaction_gripper_release_gain
            ),
            interaction_gripper_residual_limit=(
                interaction_gripper_residual_limit
            ),
            interaction_phase_prior_gain=interaction_phase_prior_gain,
            release_target_radius=release_target_radius,
            task_residual_scale=task_residual_scale,
            transition_latent_dim=transition_latent_dim,
            transition_temperature=transition_temperature,
            task_affordance_gain=task_affordance_gain,
            task_outcome_gain=task_outcome_gain,
            recovery_task_margin_gain=recovery_task_margin_gain,
            recovery_adapter_task_gain=recovery_adapter_task_gain,
            recovery_adapter_motion_gain=recovery_adapter_motion_gain,
            recovery_adapter_interaction_gain=(
                recovery_adapter_interaction_gain
            ),
            task_outcome_warmup_updates=task_outcome_warmup_updates,
            skill_outcome_gain=skill_outcome_gain,
            payload_survival_gain=payload_survival_gain,
            payload_survival_warmup_updates=(
                payload_survival_warmup_updates
            ),
            skill_effect_gain=skill_effect_gain,
            constraint_utility_gain=constraint_utility_gain,
            motion_objective_gain=motion_objective_gain,
            motion_execution_utility_gain=motion_execution_utility_gain,
            embodiment_response_selection_gain=(
                embodiment_response_selection_gain
            ),
            embodiment_response_prior_confidence=(
                embodiment_response_prior_confidence
            ),
            wheel_action_scale=wheel_action_scale,
            wheel_track_width=wheel_track_width,
            wheel_breakaway_action=wheel_breakaway_action,
            wheel_turn_breakaway_action=wheel_turn_breakaway_action,
            wheel_residual_gain=wheel_residual_gain,
            wheel_action_limit=wheel_action_limit,
        )
        self.register_buffer(
            "tactic_training_updates",
            torch.zeros((), dtype=torch.long),
        )

        self._last_actor_obs: torch.Tensor | None = None
        self._last_context: torch.Tensor | None = None
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._entropy: torch.Tensor | None = None
        self._task_distribution: Categorical | None = None
        self._object_distribution: Categorical | None = None
        self._skill_distribution: Categorical | None = None
        self._task_subgoal_distribution: Normal | None = None
        self._task_termination_distribution: Normal | None = None
        self._skill_parameter_distribution: Normal | None = None
        self._skill_termination_distribution: Normal | None = None
        self._physical_distribution: Normal | None = None

        # Continuous subgoals, skill parameters, and physical actions use
        # Gaussian exploration. Option termination is a deterministic hazard
        # trained by its own counterfactual objective.
        with torch.no_grad():
            s = ACTION_LAYOUT.slices()
            if self.noise_std_type == "scalar":
                self.std.fill_(0.10)
                self.std[:12].fill_(0.001)
                self.std[12:16].fill_(0.01)
                self.std[16].fill_(0.04)
                self.std[s["task_subgoal"]].fill_(0.025)
                self.std[s["skill_param"]].fill_(0.06)
                self.std[
                    s["skill_param"].start : s["skill_param"].start + 2
                ].fill_(0.12)
                self.std[s["termination"]].fill_(0.05)
            else:
                target = torch.full(
                    (num_actions,), 0.10, device=self.log_std.device
                )
                target[:12] = 0.001
                target[12:16] = 0.01
                target[16] = 0.04
                target[s["task_subgoal"]] = 0.025
                target[s["skill_param"]] = 0.06
                target[
                    s["skill_param"].start : s["skill_param"].start + 2
                ] = 0.12
                target[s["termination"]] = 0.05
                self.log_std.copy_(target.log())

        noise_parameter = (
            self.std
            if self.noise_std_type == "scalar"
            else self.log_std
        )

        def preserve_physical_exploration(gradient: torch.Tensor):
            gradient = gradient.clone()
            gradient[:16] = 0.0
            return gradient

        noise_parameter.register_hook(preserve_physical_exploration)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load legacy unified checkpoints with a neutral recovery adapter."""

        current_state = super().state_dict()
        compatible_state = state_dict.copy()
        for name, value in current_state.items():
            if (
                name.startswith(
                    self.OPTIONAL_RECOVERY_ADAPTER_PREFIXES
                )
                and name not in compatible_state
            ):
                compatible_state[name] = value
        return super().load_state_dict(
            compatible_state, strict=strict
        )

    def _sync_hierarchy_schedule(self) -> None:
        update_count = int(self.tactic_training_updates.item())
        self.actor.task_outcome_maturity = min(
            1.0,
            float(update_count)
            / float(self.actor.task_outcome_warmup_updates),
        )
        self.actor.motion_execution_maturity = min(
            1.0,
            float(update_count)
            / float(4 * self.actor.task_outcome_warmup_updates),
        )
        self.actor.payload_survival_maturity = min(
            1.0,
            float(self.actor.payload_survival_updates.item())
            / float(self.actor.payload_survival_warmup_updates),
        )

    def _context(self, obs) -> torch.Tensor:
        try:
            return obs[self.hierarchy_context_group]
        except (KeyError, TypeError):
            actor_obs = self.get_actor_obs(obs)
            return torch.zeros(
                actor_obs.shape[0],
                HIERARCHY_CONTEXT_DIM,
                device=actor_obs.device,
                dtype=actor_obs.dtype,
            )

    def _std_vector(self) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.clamp_min(1.0e-4)
        return torch.exp(self.log_std).clamp_min(1.0e-4)

    @staticmethod
    def _task_from_action(actions: torch.Tensor) -> torch.Tensor:
        s = ACTION_LAYOUT.slices()
        return torch.cat(
            (
                actions[:, s["task"]],
                actions[:, s["object"]],
                actions[:, s["task_subgoal"]],
                actions[:, s["termination"].start : s["termination"].start + 1],
            ),
            dim=-1,
        )

    @staticmethod
    def _skill_from_action(actions: torch.Tensor) -> torch.Tensor:
        s = ACTION_LAYOUT.slices()
        return torch.cat(
            (
                actions[:, s["skill"]],
                actions[:, s["skill_param"]],
                actions[:, s["termination"].start + 1 : s["termination"].stop],
            ),
            dim=-1,
        )

    @staticmethod
    def _assemble(
        physical: torch.Tensor,
        task_block: torch.Tensor,
        skill_block: torch.Tensor,
    ) -> torch.Tensor:
        task_end = ACTION_LAYOUT.task_dim
        object_end = task_end + ACTION_LAYOUT.object_dim
        task_subgoal_end = object_end + ACTION_LAYOUT.task_subgoal_dim
        skill_end = ACTION_LAYOUT.skill_dim
        skill_param_end = skill_end + ACTION_LAYOUT.skill_param_dim
        return torch.cat(
            (
                physical,
                task_block[:, :task_end],
                skill_block[:, :skill_end],
                task_block[:, task_end:object_end],
                task_block[:, object_end:task_subgoal_end],
                skill_block[:, skill_end:skill_param_end],
                task_block[:, task_subgoal_end:],
                skill_block[:, skill_param_end:],
            ),
            dim=-1,
        )

    def _component_stds(self, batch_size: int):
        s = ACTION_LAYOUT.slices()
        std = self._std_vector()
        return {
            "physical": std[s["physical"]].expand(batch_size, -1),
            "task_subgoal": std[s["task_subgoal"]].expand(batch_size, -1),
            "task_termination": std[
                s["termination"].start : s["termination"].start + 1
            ].expand(batch_size, -1),
            "skill_param": std[s["skill_param"]].expand(batch_size, -1),
            "skill_termination": std[
                s["termination"].start + 1 : s["termination"].stop
            ].expand(batch_size, -1),
        }

    @staticmethod
    def _one_hot(
        index: torch.Tensor, width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.nn.functional.one_hot(index, num_classes=width).to(
            dtype=dtype
        )

    def _build_distributions(
        self,
        actor_obs: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor | None = None,
    ):
        self._sync_hierarchy_schedule()
        s = ACTION_LAYOUT.slices()
        std = self._component_stds(actor_obs.shape[0])
        task_logits, mission_state = self.actor.task_choice(context)
        task_distribution = Categorical(logits=task_logits)
        if actions is None:
            task_id = task_distribution.sample()
        else:
            task_id = torch.argmax(actions[:, s["task"]], dim=-1)
        task_code = self._one_hot(
            task_id, ACTION_LAYOUT.task_dim, actor_obs.dtype
        )
        (
            object_logits,
            task_subgoal_mean,
            task_termination_mean,
        ) = self.actor.task_detail_parameters(task_code, mission_state)
        object_distribution = Categorical(logits=object_logits)
        task_subgoal_distribution = Normal(
            task_subgoal_mean, std["task_subgoal"]
        )
        task_termination_distribution = Normal(
            task_termination_mean, std["task_termination"]
        )
        if actions is None:
            object_id = object_distribution.sample()
            task_subgoal = task_subgoal_distribution.sample()
            task_termination = task_termination_mean
        else:
            object_id = torch.argmax(actions[:, s["object"]], dim=-1)
            task_subgoal = actions[:, s["task_subgoal"]]
            task_termination = actions[
                :,
                s["termination"].start : s["termination"].start + 1,
            ]
        object_code = self._one_hot(
            object_id, ACTION_LAYOUT.object_dim, actor_obs.dtype
        )
        task_action = torch.cat(
            (task_code, object_code, task_subgoal, task_termination),
            dim=-1,
        )

        skill_logits, skill_state = self.actor.skill_choice(
            task_action, mission_state
        )
        skill_distribution = Categorical(logits=skill_logits)
        if actions is None:
            skill_id = skill_distribution.sample()
        else:
            skill_id = torch.argmax(actions[:, s["skill"]], dim=-1)
        skill_code = self._one_hot(
            skill_id, ACTION_LAYOUT.skill_dim, actor_obs.dtype
        )
        (
            skill_parameter_mean,
            skill_termination_mean,
            hierarchy_state,
        ) = self.actor.skill_detail_parameters(skill_code, skill_state)
        skill_parameter_distribution = Normal(
            skill_parameter_mean, std["skill_param"]
        )
        skill_termination_distribution = Normal(
            skill_termination_mean, std["skill_termination"]
        )
        if actions is None:
            skill_parameter = skill_parameter_distribution.sample()
            skill_termination = skill_termination_mean
        else:
            skill_parameter = actions[:, s["skill_param"]]
            skill_termination = actions[
                :,
                s["termination"].start + 1 : s["termination"].stop,
            ]
        skill_action = torch.cat(
            (skill_code, skill_parameter, skill_termination), dim=-1
        )

        physical_mean = self.actor.physical_mean(
            actor_obs, task_action, skill_action, hierarchy_state
        )
        physical_distribution = Normal(physical_mean, std["physical"])
        physical_action = (
            physical_distribution.sample()
            if actions is None
            else actions[:, s["physical"]]
        )

        self._task_distribution = task_distribution
        self._object_distribution = object_distribution
        self._skill_distribution = skill_distribution
        self._task_subgoal_distribution = task_subgoal_distribution
        self._task_termination_distribution = task_termination_distribution
        self._skill_parameter_distribution = skill_parameter_distribution
        self._skill_termination_distribution = skill_termination_distribution
        self._physical_distribution = physical_distribution
        task_mean = torch.cat(
            (
                task_distribution.probs,
                object_distribution.probs,
                task_subgoal_mean,
                task_termination_mean,
            ),
            dim=-1,
        )
        task_std = torch.cat(
            (
                torch.sqrt(
                    task_distribution.probs
                    * (1.0 - task_distribution.probs)
                    + 1.0e-6
                ),
                torch.sqrt(
                    object_distribution.probs
                    * (1.0 - object_distribution.probs)
                    + 1.0e-6
                ),
                std["task_subgoal"],
                std["task_termination"],
            ),
            dim=-1,
        )
        skill_mean = torch.cat(
            (
                skill_distribution.probs,
                skill_parameter_mean,
                skill_termination_mean,
            ),
            dim=-1,
        )
        skill_std = torch.cat(
            (
                torch.sqrt(
                    skill_distribution.probs
                    * (1.0 - skill_distribution.probs)
                    + 1.0e-6
                ),
                std["skill_param"],
                std["skill_termination"],
            ),
            dim=-1,
        )
        self._action_mean = self._assemble(
            physical_mean, task_mean, skill_mean
        )
        self._action_std = self._assemble(
            std["physical"], task_std, skill_std
        )
        self._entropy = (
            task_distribution.entropy()
            + object_distribution.entropy()
            + skill_distribution.entropy()
            + task_subgoal_distribution.entropy().sum(dim=-1)
            + skill_parameter_distribution.entropy().sum(dim=-1)
            + physical_distribution.entropy().sum(dim=-1)
        )
        return self._assemble(physical_action, task_action, skill_action)

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def action_std(self):
        return self._action_std

    @property
    def entropy(self):
        return self._entropy

    def _release_action_cache(self):
        """Drop outputs from the previous hierarchical forward pass."""

        self._action_mean = None
        self._action_std = None
        self._entropy = None
        self._task_distribution = None
        self._object_distribution = None
        self._skill_distribution = None
        self._task_subgoal_distribution = None
        self._task_termination_distribution = None
        self._skill_parameter_distribution = None
        self._skill_termination_distribution = None
        self._physical_distribution = None
        for name, value in vars(self.actor).items():
            if name.startswith("last_") and torch.is_tensor(value):
                setattr(self.actor, name, None)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        context = self._context(obs)
        self._last_actor_obs = actor_obs
        self._last_context = context
        self._release_action_cache()
        if torch.is_grad_enabled():
            # RSL-RL calls act() once before asking for the likelihood of the
            # stored hierarchical action. That likelihood needs a second,
            # action-conditioned pass, so retaining this first graph only
            # doubles peak memory.
            with torch.no_grad():
                return self._build_distributions(actor_obs, context)
        return self._build_distributions(actor_obs, context)

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        context = self._context(obs)
        self._sync_hierarchy_schedule()
        task_logits, mission_state = self.actor.task_choice(context)
        task_code = self._one_hot(
            torch.argmax(task_logits, dim=-1),
            ACTION_LAYOUT.task_dim,
            actor_obs.dtype,
        )
        (
            object_logits,
            task_subgoal,
            task_termination,
        ) = self.actor.task_detail_parameters(task_code, mission_state)
        object_code = self._one_hot(
            torch.argmax(object_logits, dim=-1),
            ACTION_LAYOUT.object_dim,
            actor_obs.dtype,
        )
        task_action = torch.cat(
            (task_code, object_code, task_subgoal, task_termination), dim=-1
        )
        skill_logits, skill_state = self.actor.skill_choice(
            task_action, mission_state
        )
        skill_code = self._one_hot(
            torch.argmax(skill_logits, dim=-1),
            ACTION_LAYOUT.skill_dim,
            actor_obs.dtype,
        )
        (
            skill_parameter,
            skill_termination,
            hierarchy_state,
        ) = self.actor.skill_detail_parameters(skill_code, skill_state)
        skill_action = torch.cat(
            (skill_code, skill_parameter, skill_termination), dim=-1
        )
        physical_mean = self.actor.physical_mean(
            actor_obs, task_action, skill_action, hierarchy_state
        )
        return self._assemble(physical_mean, task_action, skill_action)

    def predict_control(self, obs, actions):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        context = self._context(obs)
        self._build_distributions(
            actor_obs, context, actions=actions
        )
        prediction = self.actor.last_control_prediction
        if prediction is None:
            raise RuntimeError("TACTIC control predictor was not evaluated")
        return prediction

    def transition_logits(
        self,
        current_context: torch.Tensor,
        next_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.transition_logits(current_context, next_context)

    def uncommitted_selection_probabilities(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_probability, _, skill_probability = (
            self.uncommitted_option_probabilities(context, actions)
        )
        return task_probability, skill_probability

    def uncommitted_option_probabilities(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate task, object, and skill choices without option commitment."""

        (
            task_probability,
            object_probability,
            skill_probability,
            _,
            _,
        ) = (
            self.uncommitted_option_replay_outputs(context, actions)
        )
        return task_probability, object_probability, skill_probability

    def uncommitted_option_replay_outputs(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Decode replayed option choices and both continuous detail levels."""

        self._sync_hierarchy_schedule()
        task_logits, mission_state = self.actor.task_choice(
            context, apply_commitment=False
        )
        task_code = actions[:, ACTION_LAYOUT.slices()["task"]]
        object_logits, task_subgoal, _ = self.actor.task_detail_parameters(
            task_code, mission_state
        )
        task_action = self._task_from_action(actions)
        skill_logits, skill_state = self.actor.skill_choice(
            task_action, mission_state, apply_commitment=False
        )
        skill_code = actions[:, ACTION_LAYOUT.slices()["skill"]]
        skill_parameters, _, _ = self.actor.skill_detail_parameters(
            skill_code, skill_state
        )
        return (
            torch.softmax(task_logits, dim=-1),
            torch.softmax(object_logits, dim=-1),
            torch.softmax(skill_logits, dim=-1),
            task_subgoal,
            skill_parameters,
        )

    def predict_option_outcomes(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted outcomes for the task and skill in a rollout."""

        self._sync_hierarchy_schedule()
        _, mission_state = self.actor.task_choice(
            context, apply_commitment=False
        )
        task_action = self._task_from_action(actions)
        self.actor.skill_choice(
            task_action, mission_state, apply_commitment=False
        )
        task_outcomes = self.actor.last_task_outcomes
        skill_outcomes = self.actor.last_skill_outcomes
        if task_outcomes is None or skill_outcomes is None:
            raise RuntimeError("TACTIC option outcome models were not evaluated")
        slices = ACTION_LAYOUT.slices()
        task_probability = actions[:, slices["task"]].clamp_min(0.0)
        task_probability = task_probability / task_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        skill_probability = actions[:, slices["skill"]].clamp_min(0.0)
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        selected_task = torch.einsum(
            "bs,bso->bo", task_probability, task_outcomes
        )
        selected_skill = torch.einsum(
            "bs,bso->bo", skill_probability, skill_outcomes
        )
        return selected_task, selected_skill

    def predict_skill_effects(
        self, context: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the measured-coordinate effect selected in a rollout."""

        self._sync_hierarchy_schedule()
        _, mission_state = self.actor.task_choice(
            context, apply_commitment=False
        )
        task_action = self._task_from_action(actions)
        self.actor.skill_choice(
            task_action, mission_state, apply_commitment=False
        )
        effects = self.actor.last_skill_effects
        confidence = self.actor.last_skill_effect_confidence
        if effects is None or confidence is None:
            raise RuntimeError("TACTIC skill-effect model was not evaluated")
        skill_probability = actions[
            :, ACTION_LAYOUT.slices()["skill"]
        ].clamp_min(0.0)
        skill_probability = skill_probability / skill_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        selected_effect = torch.einsum(
            "bs,bse->be", skill_probability, effects
        )
        selected_confidence = torch.einsum(
            "bs,bs->b", skill_probability, confidence
        )
        return selected_effect, selected_confidence

    def predict_motion_execution(
        self,
        context: torch.Tensor,
        wheel_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.predict_motion_execution(
            context, wheel_action
        )

    def predict_motion_execution_frozen(
        self,
        context: torch.Tensor,
        wheel_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.predict_motion_execution_frozen(
            context, wheel_action
        )

    def predict_auxiliary(self, obs, actions):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        context = self._context(obs)
        self._build_distributions(
            actor_obs, context, actions=actions
        )
        control = self.actor.last_control_prediction
        feasibility = self.actor.last_skill_feasibility
        if control is None or feasibility is None:
            raise RuntimeError("TACTIC auxiliary heads were not evaluated")
        skill_id = torch.argmax(
            actions[:, ACTION_LAYOUT.slices()["skill"]], dim=-1
        )
        selected_feasibility = feasibility.gather(
            1, skill_id.unsqueeze(1)
        ).squeeze(1)
        return control, selected_feasibility

    def get_actions_log_prob(self, actions):
        if self._last_actor_obs is None or self._last_context is None:
            raise RuntimeError("act() must be called before get_actions_log_prob()")
        self._release_action_cache()
        self._build_distributions(
            self._last_actor_obs, self._last_context, actions=actions
        )
        task_log_prob, skill_log_prob, physical_log_prob = (
            self.option_log_prob_components(actions)
        )
        return task_log_prob + skill_log_prob + physical_log_prob

    def option_log_prob_components(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return task-, skill-, and physical-action log probabilities."""

        distributions = (
            self._task_distribution,
            self._object_distribution,
            self._skill_distribution,
            self._task_subgoal_distribution,
            self._task_termination_distribution,
            self._skill_parameter_distribution,
            self._skill_termination_distribution,
            self._physical_distribution,
        )
        if any(distribution is None for distribution in distributions):
            raise RuntimeError("TACTIC action distributions were not evaluated")
        s = ACTION_LAYOUT.slices()
        task_id = torch.argmax(actions[:, s["task"]], dim=-1)
        object_id = torch.argmax(actions[:, s["object"]], dim=-1)
        skill_id = torch.argmax(actions[:, s["skill"]], dim=-1)
        physical_action = actions[:, s["physical"]]
        task_log_prob = (
            self._task_distribution.log_prob(task_id)
            + self._object_distribution.log_prob(object_id)
            + self._task_subgoal_distribution.log_prob(
                actions[:, s["task_subgoal"]]
            ).sum(dim=-1)
        )
        skill_log_prob = (
            self._skill_distribution.log_prob(skill_id)
            + self._skill_parameter_distribution.log_prob(
                actions[:, s["skill_param"]]
            ).sum(dim=-1)
        )
        physical_log_prob = self._physical_distribution.log_prob(
            physical_action
        ).sum(dim=-1)
        return task_log_prob, skill_log_prob, physical_log_prob

    def discrete_option_log_prob_components(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return categorical task/object and skill log probabilities."""

        distributions = (
            self._task_distribution,
            self._object_distribution,
            self._skill_distribution,
        )
        if any(distribution is None for distribution in distributions):
            raise RuntimeError("TACTIC action distributions were not evaluated")
        s = ACTION_LAYOUT.slices()
        task_id = torch.argmax(actions[:, s["task"]], dim=-1)
        object_id = torch.argmax(actions[:, s["object"]], dim=-1)
        skill_id = torch.argmax(actions[:, s["skill"]], dim=-1)
        task_log_prob = (
            self._task_distribution.log_prob(task_id)
            + self._object_distribution.log_prob(object_id)
        )
        skill_log_prob = self._skill_distribution.log_prob(skill_id)
        return task_log_prob, skill_log_prob

    def discrete_factor_log_prob_components(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return task/object, motion-factor, and interaction-factor logs."""

        distributions = (
            self._task_distribution,
            self._object_distribution,
            self._skill_distribution,
        )
        if any(distribution is None for distribution in distributions):
            raise RuntimeError("TACTIC action distributions were not evaluated")
        slices = ACTION_LAYOUT.slices()
        task_id = torch.argmax(actions[:, slices["task"]], dim=-1)
        object_id = torch.argmax(actions[:, slices["object"]], dim=-1)
        skill_id = torch.argmax(actions[:, slices["skill"]], dim=-1)
        motion_id = torch.div(
            skill_id,
            INTERACTION_SKILL_COUNT,
            rounding_mode="floor",
        )
        interaction_id = skill_id % INTERACTION_SKILL_COUNT
        factor_probability = self._skill_distribution.probs.reshape(
            -1, MOTION_SKILL_COUNT, INTERACTION_SKILL_COUNT
        )
        motion_probability = factor_probability.sum(dim=2)
        interaction_probability = factor_probability.sum(dim=1)
        task_log_prob = (
            self._task_distribution.log_prob(task_id)
            + self._object_distribution.log_prob(object_id)
        )
        motion_log_prob = torch.log(
            motion_probability.gather(
                1, motion_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        interaction_log_prob = torch.log(
            interaction_probability.gather(
                1, interaction_id.unsqueeze(1)
            ).squeeze(1).clamp_min(1.0e-8)
        )
        return task_log_prob, motion_log_prob, interaction_log_prob
