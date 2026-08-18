"""RSL-RL configuration for the unified TANDEM-HRL policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from ..tactic_layout import RELEASE_TARGET_RADIUS


@configclass
class TACTICActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "TACTICActorCritic"
    init_noise_std = 0.10
    noise_std_type = "scalar"
    actor_obs_normalization = True
    critic_obs_normalization = True
    actor_hidden_dims = [256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    hierarchy_context_group = "hierarchy_context"
    mission_latent_dim = 128
    slot_latent_dim = 128
    task_embedding_dim = 32
    skill_embedding_dim = 48
    conditioner_gain = 0.12
    physical_residual_gain = 0.16
    support_skill_residual_gain = 0.10
    support_skill_adaptation_gain = 0.12
    physical_core_obs_dim = 876
    skill_feasibility_gain = 0.35
    skill_effect_scale = 0.85
    interaction_gripper_gain = 2.0
    interaction_gripper_phase_gain = 4.5
    interaction_gripper_release_gain = 6.0
    interaction_gripper_residual_limit = 1.0
    interaction_phase_prior_gain = 0.85
    release_target_radius = RELEASE_TARGET_RADIUS
    task_residual_scale = 0.22
    transition_latent_dim = 64
    transition_temperature = 0.12
    task_affordance_gain = 1.00
    task_outcome_gain = 0.80
    recovery_task_margin_gain = 0.60
    recovery_adapter_task_gain = 2.00
    recovery_adapter_motion_gain = 1.50
    recovery_adapter_interaction_gain = 2.00
    task_outcome_warmup_updates = 64
    skill_outcome_gain = 0.25
    payload_survival_gain = 1.20
    payload_survival_warmup_updates = 64
    skill_effect_gain = 1.50
    constraint_utility_gain = 0.80
    motion_objective_gain = 1.80
    motion_execution_utility_gain = 0.85
    embodiment_response_selection_gain = 2.40
    embodiment_response_prior_confidence = 0.70
    wheel_action_scale = 2.00
    wheel_track_width = 0.50
    wheel_breakaway_action = 3.0
    wheel_turn_breakaway_action = 3.5
    wheel_residual_gain = 0.8
    wheel_action_limit = 24.0


@configclass
class TACTICPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "TACTICPPO"
    hierarchy_context_group = "hierarchy_context"
    control_prediction_coef = 0.20
    skill_feasibility_coef = 0.12
    task_outcome_coef = 0.16
    task_outcome_confidence_coef = 0.04
    skill_outcome_coef = 0.08
    payload_survival_coef = 0.30
    payload_drop_weight = 8.0
    payload_drop_task_credit_penalty = 0.25
    payload_drop_skill_credit_penalty = 0.90
    payload_survival_replay_coef = 0.70
    payload_survival_replay_capacity = 8192
    payload_survival_replay_batch_size = 512
    payload_survival_replay_drop_fraction = 0.50
    payload_survival_replay_max_add = 512
    payload_survival_learning_rate = 3.0e-4
    payload_survival_rank_coef = 0.35
    payload_survival_rank_margin = 0.05
    skill_effect_prediction_coef = 0.40
    skill_effect_confidence_coef = 0.05
    motion_execution_prediction_coef = 0.40
    motion_execution_confidence_coef = 0.04
    embodiment_response_coef = 0.35
    constraint_multiplier_coef = 0.05
    grounded_effect_diversity_coef = 0.04
    motion_objective_diversity_coef = 0.03
    task_option_credit_coef = 0.10
    skill_option_credit_coef = 0.14
    event_hindsight_task_coef = 0.16
    event_hindsight_skill_coef = 0.10
    event_replay_task_coef = 0.08
    event_replay_skill_coef = 0.14
    event_replay_parameter_coef = 0.06
    event_replay_motion_weight = 0.55
    event_replay_interaction_weight = 0.45
    event_replay_task_subgoal_weight = 0.35
    event_replay_delivery_fraction = 0.55
    event_replay_recovery_fraction = 0.15
    event_replay_secure_fraction = 0.30
    event_replay_release_fraction = 0.15
    event_replay_role_oversample_cap = 4.0
    event_replay_phase_oversample_cap = 4.0
    event_replay_capacity = 16384
    event_replay_batch_size = 1024
    event_replay_max_add = 256
    event_replay_min_score = 1.0e-3
    task_transition_coef = 0.08
    skill_transition_coef = 0.10
    skill_information_coef = 0.05
    task_information_coef = 0.02
    information_warmup_updates = 8
    transition_horizon_steps = 12
    task_usage_coef = 0.04
    task_frontier_coverage_coef = 0.12
    interaction_phase_coef = 0.30
    interaction_release_focus = 8.0
    skill_usage_coef = 0.03
    skill_confidence_coef = 0.02
    slot_diversity_coef = 0.01
    skill_diversity_coef = 0.01
    motion_gain_diversity_coef = 0.08
    task_control_objective_coef = 0.60
    skill_predictive_control_coef = 0.40
    relational_subgoal_grounding_coef = 0.50
    task_skill_projection_coef = 0.20
    counterfactual_task_selection_coef = 0.10
    counterfactual_task_temperature = 0.25
    counterfactual_skill_selection_coef = 0.20
    counterfactual_skill_temperature = 0.20
    counterfactual_termination_coef = 0.12
    successor_decoder_coef = 0.20
    physical_warmup_updates = 32
    physical_core_lr_scale = 0.05
    physical_adapter_lr_scale = 1.00
    decomposition_trust_region_radius = 0.006
    auxiliary_batches = 1
    auxiliary_batch_size = 2048
    auxiliary_learning_rate = 2.0e-5
    successor_adapter_learning_rate = 3.0e-5
    cbf_violation_budget = 0.05
    clf_violation_budget = 0.08
    constraint_dual_learning_rate = 0.05
    constraint_dual_max = 4.0
    constraint_dual_ema_decay = 0.95
    freeze_locomotion_executor = True


@configclass
class TACTICRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1024
    save_interval = 64
    experiment_name = "TACTIC_HRL"
    run_name = ""
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "hierarchy_context"],
    }
    clip_actions = 100.0
    policy = TACTICActorCriticCfg()
    algorithm = TACTICPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.0008,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.997,
        lam=0.97,
        desired_kl=0.004,
        max_grad_norm=0.25,
    )
