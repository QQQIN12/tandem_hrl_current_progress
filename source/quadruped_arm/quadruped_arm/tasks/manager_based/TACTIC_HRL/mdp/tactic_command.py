"""Goal-conditioned mission interface for TACTIC-HRL."""

from __future__ import annotations

import math
from typing import Optional

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.mdp.utils import (
    _normalize,
    _quat_apply,
    _quat_from_euler_xyz,
    _quat_from_tool_z,
    _quat_from_yaw,
    _quat_mul,
    _quat_rotate_inverse,
)

from ..tactic_layout import (
    ACTION_LAYOUT,
    GLOBAL_CONTEXT_DIM,
    HIERARCHY_CONTEXT_DIM,
    PAYLOAD_TASK_CONTRACTION_GAIN,
    RELEASE_HOVER_HEIGHT,
    TASK_SLOT_CARRYING_INDEX,
    TASK_SLOT_COUNT,
    TASK_SLOT_FEATURE_DIM,
    TASK_SLOT_REACHABILITY_INDEX,
)


TASK_TYPE_COUNT = 6
TASK_TYPE_ROUTE = 0
TASK_TYPE_SLALOM = 1
TASK_TYPE_NARROW = 2
TASK_TYPE_MANIPULATION = 3
TASK_TYPE_RECOVERY = 4
TASK_TYPE_DELIVERY = 5


def _wrap_to_pi(value: torch.Tensor) -> torch.Tensor:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@configclass
class TACTICMissionCommandCfg(CommandTermCfg):
    class_type: Optional[type] = None
    asset_name: str = "robot"
    ee_body_name: str = "link6"
    # The mission ledger is sampled at episode reset and must remain valid
    # throughout the complete long-horizon trajectory.
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)

    object_names: tuple[str, ...] = (
        "grasp_object",
        "grasp_object_tall",
        "grasp_object_flat",
        "grasp_object_round",
        "grasp_object_slim",
        "grasp_object_wide",
    )
    target_platform_names: tuple[str, ...] = (
        "grasp_place_pad",
        "grasp_place_pad_aux_a",
        "grasp_place_pad_aux_b",
        "grasp_place_pad_aux_c",
        "grasp_place_pad_aux_d",
        "grasp_place_pad_aux_e",
    )
    target_offsets: tuple[tuple[float, float, float], ...] = (
        (-0.05, -0.02, 0.065),
        (0.00, 0.00, 0.082),
        (0.00, 0.00, 0.050),
        (0.00, 0.00, 0.068),
        (0.00, 0.00, 0.098),
        (0.00, 0.00, 0.059),
    )
    object_shape_codes: tuple[float, ...] = (
        0.10,
        0.30,
        0.50,
        0.70,
        0.85,
        1.00,
    )
    # Grasp width, height, mass, and dynamic friction after the six assets are
    # calibrated to the physical finger clearance.
    object_physical_descriptors: tuple[tuple[float, float, float, float], ...] = (
        (0.038, 0.078, 0.044, 1.02),
        (0.036, 0.116, 0.040, 1.02),
        (0.038, 0.044, 0.028, 1.14),
        (0.036, 0.086, 0.037, 0.96),
        (0.034, 0.122, 0.034, 1.10),
        (0.040, 0.050, 0.030, 1.20),
    )

    mission_goal_xy: tuple[tuple[float, float], ...] = (
        (0.80, -0.55),
        (1.75, 0.30),
        (2.70, 0.02),
        (-0.72, 0.82),
        (-0.25, -0.85),
        (0.00, 0.00),
    )
    task_goal_radius: float = 0.34
    final_goal_radius: float = 0.42
    max_forward_speed: float = 0.90
    max_lateral_speed: float = 0.28
    max_yaw_rate: float = 1.25
    ee_center_offset: tuple[float, float, float] = (-0.215, 0.0, 0.70)
    # These two poses reproduce the safe initial arm sweep in ZYB-v0:
    # spherical (0.5, pi/8, 0) -> (0.4, 0, 0).
    ee_reset_offset: tuple[float, float, float] = (
        -0.677,
        0.0,
        0.891,
    )
    ee_default_offset: tuple[float, float, float] = (-0.615, 0.0, 0.70)
    ee_subgoal_scale: tuple[float, float, float] = (0.22, 0.22, 0.18)
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.13)
    interaction_workspace_radius: float = 0.52
    interaction_base_standoff: float = 0.46
    interaction_arm_axis_yaw: float = math.pi
    capture_staging_radius: float = 0.28
    capture_heading_tolerance: float = 0.45
    capture_position_gate_gain: float = 18.0
    capture_heading_gate_gain: float = 9.0
    delivery_side_hysteresis: float = 0.03
    grasp_orientation_activation_radius: float = 0.58
    grasp_orientation_roll_limit: float = 0.65
    gripper_open_gap: float = 0.066
    wheel_radius: float = 0.10
    max_ee_goal_step_m: float = 0.006
    hierarchy_settling_time_s: float = 0.35
    curriculum_boundaries: tuple[int, ...] = (
        0,
        2048,
        6144,
        11264,
        16384,
    )
    composition_probe_probability: float = 0.30
    composition_probe_min_probability: float = 0.18
    composition_probe_levels_ahead: int = 1
    delivery_replay_fraction: float = 0.0
    interaction_start_curriculum_probability: float = 0.0
    interaction_start_curriculum_min_probability: float = 0.0
    interaction_start_curriculum_decay_steps: int = 0
    interaction_start_object_id: int = -1
    interaction_start_platform_map: tuple[int, ...] = (5, 3, 5, 1, 1, 0)
    interaction_start_base_standoff: float = 0.46
    interaction_start_position_jitter: float = 0.035

    # The inherited hard termination is z < 0.20 m and the migrated ZYB-v0
    # policy settles near 0.356 m.  Keep a physical buffer above termination,
    # but do not label the nominal stance as a barrier-boundary state.
    base_height_min: float = 0.23
    base_height_target: float = 0.36
    tilt_limit: float = 0.48
    support_min: float = 1.5
    automatic_recovery_task_candidate: bool = True
    obstacle_stop_margin: float = 0.28
    obstacle_safe_margin: float = 0.75
    preview_horizon_s: float = 0.40
    disturbance_filter: float = 0.92
    contact_force_threshold: float = 0.16
    contact_hold_s: float = 0.06
    lift_height: float = 0.05
    capture_lift_offset: float = 0.10
    transport_distance: float = 0.08
    place_xy_tolerance: float = 0.085
    place_z_tolerance: float = 0.070
    open_gap_threshold: float = 0.050

    def __post_init__(self):
        if not 0.0 <= self.delivery_replay_fraction <= 1.0:
            raise ValueError(
                "delivery_replay_fraction must lie in [0, 1]"
            )
        if not 0.0 <= self.interaction_start_curriculum_probability <= 1.0:
            raise ValueError(
                "interaction_start_curriculum_probability must lie in [0, 1]"
            )
        if not (
            0.0
            <= self.interaction_start_curriculum_min_probability
            <= self.interaction_start_curriculum_probability
        ):
            raise ValueError(
                "interaction_start_curriculum_min_probability must lie "
                "between zero and the initial probability"
            )
        if self.interaction_start_curriculum_decay_steps < 0:
            raise ValueError(
                "interaction_start_curriculum_decay_steps cannot be negative"
            )
        if not -1 <= self.interaction_start_object_id < len(
            self.object_names
        ):
            raise ValueError(
                "interaction_start_object_id must be -1 or a valid object id"
            )
        if len(self.interaction_start_platform_map) != len(
            self.object_names
        ):
            raise ValueError(
                "interaction_start_platform_map must define one source "
                "platform per object"
            )
        if any(
            platform < 0 or platform >= len(self.target_platform_names)
            for platform in self.interaction_start_platform_map
        ):
            raise ValueError(
                "interaction_start_platform_map contains an invalid platform"
            )
        if self.interaction_start_base_standoff <= 0.0:
            raise ValueError(
                "interaction_start_base_standoff must be positive"
            )
        if self.interaction_start_position_jitter < 0.0:
            raise ValueError(
                "interaction_start_position_jitter cannot be negative"
            )
        if self.class_type is None:
            self.class_type = TACTICMissionCommand


class TACTICMissionCommand(CommandTerm):
    """Expose task goals and measured outcomes without choosing their order."""

    cfg: TACTICMissionCommandCfg

    def __init__(self, cfg: TACTICMissionCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.curr_goal_quat_w = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.curr_goal_quat_w[:, 0] = 1.0

        self.task_slots = torch.zeros(
            self.num_envs,
            TASK_SLOT_COUNT,
            TASK_SLOT_FEATURE_DIM,
            device=self.device,
        )
        self.hierarchy_context = torch.zeros(
            self.num_envs, HIERARCHY_CONTEXT_DIM, device=self.device
        )
        self.tau_down_packet = torch.zeros(
            self.num_envs, 77, device=self.device
        )
        self.tau_up_packet = torch.zeros(
            self.num_envs, 48, device=self.device
        )
        self.task_required = torch.ones(
            self.num_envs, TASK_SLOT_COUNT, device=self.device
        )
        self.curriculum_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.curriculum_probe = torch.zeros(
            self.num_envs, device=self.device
        )
        self.interaction_start_curriculum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._interaction_start_probability = 0.0
        self.interaction_start_source_platform = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._curriculum_unlocked_level = 0
        self._curriculum_evidence_initialized = False
        self._composition_probe_probability = 0.0
        self._has_episode_history = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._curriculum_completion_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_progress_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_safety_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_contact_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_lift_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_transport_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_place_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_delivery_completion_ema = torch.zeros(
            (), device=self.device
        )
        self._curriculum_regression_streak = 0
        self.episode_safe_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self.episode_evidence_steps = torch.zeros(
            self.num_envs, device=self.device
        )
        self.task_completed = torch.zeros_like(self.task_required)
        self.task_available = torch.ones_like(self.task_required)
        self.task_progress = torch.zeros_like(self.task_required)
        self.previous_task_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.task_error = torch.zeros_like(self.previous_task_error)
        self.clf_decrease = torch.zeros_like(self.previous_task_error)
        self.clf_decrease_score = torch.full_like(
            self.previous_task_error, 0.5
        )
        self.safety_margin = torch.ones_like(self.previous_task_error)
        self.preview_margin = torch.ones_like(self.previous_task_error)
        self.disturbance_estimate = torch.zeros_like(
            self.previous_task_error
        )
        self.disturbance_quality = torch.ones_like(
            self.previous_task_error
        )
        self.obstacle_margin = torch.ones_like(self.previous_task_error)
        self.support_count = torch.full_like(self.previous_task_error, 4.0)
        self.base_tilt = torch.zeros_like(self.previous_task_error)
        self.base_height = torch.full_like(self.previous_task_error, 0.36)
        self.control_recovery_pressure = torch.zeros_like(
            self.previous_task_error
        )
        self.control_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.control_recovery_constraint_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.payload_posture_projection = torch.zeros_like(
            self.previous_task_error
        )
        self.payload_relation_violation = torch.zeros_like(
            self.previous_task_error
        )
        self.ee_error = torch.zeros_like(self.previous_task_error)
        self.selected_task_progress = torch.zeros_like(
            self.previous_task_error
        )
        self.previous_selected_task_progress = torch.zeros_like(
            self.previous_task_error
        )
        self.selected_task_progress_delta = torch.zeros_like(
            self.previous_task_error
        )
        self.selected_interaction_frontier = torch.zeros_like(
            self.previous_task_error
        )
        self.selected_interaction_frontier_delta = torch.zeros_like(
            self.previous_task_error
        )
        self.mission_completion = torch.zeros_like(
            self.previous_task_error
        )
        self.previous_mission_completion = torch.zeros_like(
            self.previous_task_error
        )
        self.mission_completion_delta = torch.zeros_like(
            self.previous_task_error
        )
        self.mission_success = torch.zeros_like(self.previous_task_error)
        self.previous_mission_success = torch.zeros_like(
            self.previous_task_error
        )
        self.mission_success_event = torch.zeros_like(
            self.previous_task_error
        )

        object_count = len(cfg.object_names)
        if object_count != ACTION_LAYOUT.object_dim:
            raise ValueError("TACTIC expects exactly six object slots")
        self.object_initial_pos_w = torch.zeros(
            self.num_envs, object_count, 3, device=self.device
        )
        self.object_initial_target_distance = torch.zeros(
            self.num_envs, object_count, device=self.device
        )
        self.object_target_pos_w = torch.zeros_like(
            self.object_initial_pos_w
        )
        self.object_contact = torch.zeros(
            self.num_envs, object_count, device=self.device
        )
        self.object_contact_symmetry = torch.zeros_like(self.object_contact)
        self.object_contact_time = torch.zeros_like(self.object_contact)
        self.object_contact_memory = torch.zeros_like(self.object_contact)
        self.object_lift = torch.zeros_like(self.object_contact)
        self.object_lift_memory = torch.zeros_like(self.object_contact)
        self.object_transport = torch.zeros_like(self.object_contact)
        self.object_transport_memory = torch.zeros_like(self.object_contact)
        self.object_carry_memory = torch.zeros_like(self.object_contact)
        self.object_place = torch.zeros_like(self.object_contact)
        self.object_completion = torch.zeros_like(self.object_contact)
        self.object_target_progress = torch.zeros_like(self.object_contact)
        self.object_target_progress_delta = torch.zeros_like(
            self.object_contact
        )
        self.previous_object_target_progress = torch.zeros_like(
            self.object_contact
        )
        self.object_release_readiness = torch.zeros_like(
            self.object_contact
        )
        self.object_release_readiness_delta = torch.zeros_like(
            self.object_contact
        )
        self.previous_object_release_readiness = torch.zeros_like(
            self.object_contact
        )
        self.object_release_event = torch.zeros_like(
            self.object_contact
        )
        self.object_release_event_quality = torch.zeros_like(
            self.object_contact
        )
        self.object_drop_event = torch.zeros_like(self.object_contact)
        self.object_gripper_distance = torch.full_like(
            self.object_contact, 2.0
        )
        self.object_tcp_offset = torch.zeros(
            self.num_envs,
            ACTION_LAYOUT.object_dim,
            3,
            device=self.device,
        )
        self.object_carry_anchor_b = torch.zeros_like(
            self.object_tcp_offset
        )
        self.object_carry_anchor_valid = torch.zeros(
            self.num_envs,
            ACTION_LAYOUT.object_dim,
            dtype=torch.bool,
            device=self.device,
        )
        self.object_delivery_side = torch.ones(
            self.num_envs,
            ACTION_LAYOUT.object_dim,
            device=self.device,
        )
        self.object_interaction_frontier = torch.zeros_like(
            self.object_contact
        )
        self.object_interaction_frontier_delta = torch.zeros_like(
            self.object_contact
        )
        self.object_carrying = torch.zeros(
            self.num_envs,
            object_count,
            dtype=torch.bool,
            device=self.device,
        )
        self.object_place_error_xy = torch.full_like(
            self.object_contact, 4.0
        )
        self.object_place_error_z = torch.full_like(
            self.object_contact, 2.0
        )
        self.gripper_closure = torch.zeros(
            self.num_envs, device=self.device
        )
        self.tcp_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.tcp_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self._ee_goal_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._control_initialized = torch.zeros_like(
            self._ee_goal_initialized
        )
        self._progress_initialized = torch.zeros_like(
            self._ee_goal_initialized
        )
        self.mission_age = torch.zeros(
            self.num_envs, device=self.device
        )
        self.manipulation_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.grasp_active = torch.zeros_like(self.manipulation_active)
        self.grasp_orientation_active = torch.zeros_like(
            self.manipulation_active
        )
        self.grasp_orientation_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.interaction_authority = torch.zeros(
            self.num_envs, device=self.device
        )
        self.workspace_handoff = torch.zeros(
            self.num_envs, device=self.device
        )
        self.grasp_projection_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self.grasp_center_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.capture_staging_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self.capture_heading_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.capture_initiation_margin = torch.zeros(
            self.num_envs, device=self.device
        )
        self.selected_delivery_side = torch.zeros(
            self.num_envs, device=self.device
        )
        self.capture_option_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.capture_option_object = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.capture_lift_active = torch.zeros_like(
            self.manipulation_active
        )

        self._center_offset = torch.tensor(
            cfg.ee_center_offset, device=self.device
        ).view(1, 3)
        self._default_ee_offset = torch.tensor(
            cfg.ee_default_offset, device=self.device
        ).view(1, 3)
        self._reset_ee_offset = torch.tensor(
            cfg.ee_reset_offset, device=self.device
        ).view(1, 3)
        self._ee_subgoal_scale = torch.tensor(
            cfg.ee_subgoal_scale, device=self.device
        ).view(1, 3)
        self._tcp_offset = torch.tensor(
            cfg.tcp_offset, device=self.device
        ).view(1, 3)
        self._target_offsets = torch.tensor(
            cfg.target_offsets, device=self.device
        ).view(1, object_count, 3)
        self._shape_codes = torch.tensor(
            cfg.object_shape_codes, device=self.device
        )
        descriptors = torch.tensor(
            cfg.object_physical_descriptors,
            device=self.device,
            dtype=self.object_initial_pos_w.dtype,
        )
        if descriptors.shape != (object_count, 4):
            raise ValueError(
                "TACTIC object descriptors must have shape (6, 4)"
            )
        self._object_physical_descriptors = descriptors
        descriptor_scale = torch.tensor(
            (0.07, 0.13, 0.05, 1.25),
            device=self.device,
            dtype=descriptors.dtype,
        )
        self._object_descriptors = descriptors / descriptor_scale
        self._mission_goal_xy = torch.tensor(
            cfg.mission_goal_xy, device=self.device
        )

        self._ee_body_id = self._resolve_body(cfg.ee_body_name)
        self._finger_body_ids = torch.tensor(
            [self._resolve_body("link7"), self._resolve_body("link8")],
            device=self.device,
            dtype=torch.long,
        )
        contact_sensor = env.scene["contact_forces"]
        sensor_body_names = list(contact_sensor.body_names)
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        missing_feet = [
            name for name in foot_names if name not in sensor_body_names
        ]
        if missing_feet:
            raise RuntimeError(
                f"Ground-contact sensor is missing feet: {missing_feet}"
            )
        self._foot_sensor_ids = torch.tensor(
            [sensor_body_names.index(name) for name in foot_names],
            device=self.device,
            dtype=torch.long,
        )
        self._finger_joint_ids = self._resolve_joints(
            ["joint7", "joint8"]
        )
        self._object_names = tuple(
            name for name in cfg.object_names if name in env.scene.keys()
        )
        if len(self._object_names) != object_count:
            raise RuntimeError(
                "TACTIC scene must expose all six rigid grasp objects"
            )
        self._target_platform_names = tuple(cfg.target_platform_names)
        self._obstacle_names = tuple(
            name
            for name in env.scene.keys()
            if name.startswith("tactic_obstacle_")
        )
        self._initialized_objects = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.task_valid_mask = self.task_required.clone()
        env.tactic_task_valid_mask = self.task_valid_mask
        env.tactic_control_recovery_pressure = (
            self.control_recovery_pressure
        )
        env.tactic_control_recovery_active = self.control_recovery_active
        env.tactic_control_recovery_constraint_active = (
            self.control_recovery_constraint_active
        )
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            hierarchy.task_valid_mask.copy_(self.task_valid_mask)
            hierarchy.control_recovery_pressure.copy_(
                self.control_recovery_pressure
            )
            hierarchy.control_recovery_active.copy_(
                self.control_recovery_active
            )
            hierarchy.control_recovery_constraint_active.copy_(
                self.control_recovery_constraint_active
            )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resolve_body(self, name: str) -> int:
        ids, _ = self._robot.find_bodies(name)
        if isinstance(ids, (list, tuple)):
            return int(ids[0])
        return int(ids.flatten()[0].item())

    def _resolve_joints(self, names: list[str]) -> torch.Tensor:
        ids, _ = self._robot.find_joints(names, preserve_order=True)
        if isinstance(ids, (list, tuple)):
            ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        return ids.to(self.device).long().flatten()

    def _scene_origins(self) -> torch.Tensor:
        origins = getattr(self._env.scene, "env_origins", None)
        if isinstance(origins, torch.Tensor):
            return origins
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _asset_positions(self, names: tuple[str, ...]) -> torch.Tensor:
        """Read dynamic rigid poses and configured static poses uniformly."""

        positions = []
        origins = self._scene_origins()
        for name in names:
            asset = self._env.scene[name]
            data = getattr(asset, "data", None)
            root_pos_w = getattr(data, "root_pos_w", None)
            if isinstance(root_pos_w, torch.Tensor):
                positions.append(root_pos_w)
                continue

            asset_cfg = getattr(asset, "cfg", None)
            if asset_cfg is None:
                asset_cfg = getattr(self._env.scene.cfg, name)
            init_state = getattr(asset_cfg, "init_state", None)
            local_position = getattr(init_state, "pos", None)
            if local_position is None:
                raise RuntimeError(
                    f"Static scene asset '{name}' has no configured position"
                )
            offset = torch.as_tensor(
                local_position, device=self.device, dtype=origins.dtype
            ).view(1, 3)
            positions.append(origins + offset)
        return torch.stack(positions, dim=1)

    def _object_positions(self) -> torch.Tensor:
        return self._asset_positions(self._object_names)

    def _object_quaternions(self) -> torch.Tensor:
        quaternions = []
        for name in self._object_names:
            root_quat_w = self._env.scene[name].data.root_quat_w
            if not isinstance(root_quat_w, torch.Tensor):
                raise RuntimeError(
                    f"Rigid object '{name}' does not expose a root quaternion"
                )
            quaternions.append(root_quat_w)
        return torch.stack(quaternions, dim=1)

    def _target_positions(self) -> torch.Tensor:
        platform_positions = self._asset_positions(
            self._target_platform_names
        )
        return platform_positions + self._target_offsets

    @staticmethod
    def _placement_hover_positions(
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        hover_positions = target_positions.clone()
        hover_positions[:, :, 2] += RELEASE_HOVER_HEIGHT
        return hover_positions

    def _assign_curriculum(self, env_ids: torch.Tensor):
        """Stratify mission complexity after the frontier is mastered."""

        step = int(getattr(self._env, "common_step_counter", 0))
        time_unlocked_level = sum(
            step >= int(boundary)
            for boundary in self.cfg.curriculum_boundaries[1:]
        )
        time_unlocked_level = min(time_unlocked_level, 4)
        unlocked_level = min(
            time_unlocked_level, self._curriculum_unlocked_level
        )
        count = env_ids.numel()
        replay_draw = torch.rand(count, device=self.device)
        sampled_level = torch.randint(
            0, unlocked_level + 1, (count,), device=self.device
        )
        sampled_level = torch.where(
            replay_draw < 0.40,
            torch.full_like(sampled_level, unlocked_level),
            sampled_level,
        )
        sampled_level = torch.where(
            (replay_draw >= 0.40) & (replay_draw < 0.65),
            torch.full_like(sampled_level, max(0, unlocked_level - 1)),
            sampled_level,
        )
        sampled_level = torch.where(
            (replay_draw >= 0.65) & (replay_draw < 0.85),
            torch.zeros_like(sampled_level),
            sampled_level,
        )
        probe_ceiling = min(
            4,
            max(
                unlocked_level
                + max(0, int(self.cfg.composition_probe_levels_ahead)),
                time_unlocked_level,
            ),
        )
        if probe_ceiling > unlocked_level:
            probe_level = torch.randint(
                unlocked_level + 1,
                probe_ceiling + 1,
                (count,),
                device=self.device,
            )
            farthest_probe = torch.rand(count, device=self.device) < 0.40
            probe_level = torch.where(
                farthest_probe,
                torch.full_like(probe_level, probe_ceiling),
                probe_level,
            )
        else:
            probe_level = torch.full(
                (count,),
                unlocked_level,
                device=self.device,
                dtype=torch.long,
            )
        if self._curriculum_evidence_initialized:
            contact_readiness = min(
                1.0,
                float(self._curriculum_contact_ema.item()) / 0.04,
            )
            lift_readiness = min(
                1.0,
                float(self._curriculum_lift_ema.item()) / 0.025,
            )
            transport_readiness = min(
                1.0,
                float(self._curriculum_transport_ema.item()) / 0.018,
            )
            place_readiness = min(
                1.0,
                float(self._curriculum_place_ema.item()) / 0.001,
            )
            completion_readiness = min(
                1.0,
                float(
                    self._curriculum_delivery_completion_ema.item()
                )
                / 0.002,
            )
            safety_readiness = max(
                0.0,
                min(
                    1.0,
                    (
                        float(self._curriculum_safety_ema.item())
                        - 0.50
                    )
                    / 0.25,
                ),
            )
            interaction_readiness = min(
                contact_readiness,
                lift_readiness,
                transport_readiness,
                safety_readiness,
            )
            terminal_readiness = max(
                place_readiness,
                completion_readiness,
            )
            # Once the interaction chain reaches transport, expose a small
            # number of compositions so the task policy receives comparative
            # credit before terminal placement is common.  Placement evidence
            # then raises the probe rate smoothly to its configured maximum.
            readiness = interaction_readiness * (
                0.20 + 0.80 * terminal_readiness
            )
        else:
            readiness = 0.0
        coverage_start = int(self.cfg.curriculum_boundaries[1])
        coverage_end = int(self.cfg.curriculum_boundaries[3])
        coverage_progress = max(
            0.0,
            min(
                1.0,
                (step - coverage_start)
                / max(1, coverage_end - coverage_start),
            ),
        )
        coverage_floor = min(
            float(self.cfg.composition_probe_probability),
            float(self.cfg.composition_probe_min_probability),
        ) * coverage_progress
        self._composition_probe_probability = max(
            float(self.cfg.composition_probe_probability) * readiness,
            coverage_floor,
        )
        probe_mask = (
            torch.rand(count, device=self.device)
            < self._composition_probe_probability
        ) & (probe_ceiling > unlocked_level)
        sampled_level = torch.where(
            probe_mask,
            probe_level,
            sampled_level,
        )
        self.curriculum_level[env_ids] = sampled_level
        self.curriculum_probe[env_ids] = probe_mask.float()
        self.task_required[env_ids] = 0.0

        for level in range(5):
            level_ids = env_ids[sampled_level == level]
            level_count = level_ids.numel()
            if level_count == 0:
                continue
            if level == 0:
                # Identify motion, manipulation, and delivery primitives
                # before combining them into longer missions.
                draw = torch.rand(level_count, device=self.device)
                selected = torch.randint(
                    0,
                    3,
                    (level_count,),
                    device=self.device,
                )
                selected = torch.where(
                    (draw >= 0.45) & (draw < 0.65),
                    torch.full_like(selected, 3),
                    selected,
                )
                delivery = torch.randint(
                    5,
                    11,
                    (level_count,),
                    device=self.device,
                )
                selected = torch.where(
                    draw >= 0.65,
                    delivery,
                    selected,
                )
                self.task_required[level_ids, selected] = 1.0
                continue

            base_rank = torch.rand(
                level_count, 5, device=self.device
            ).argsort(dim=1)
            delivery_rank = (
                torch.rand(
                    level_count,
                    ACTION_LAYOUT.object_dim,
                    device=self.device,
                ).argsort(dim=1)
                + 5
            )
            if level == 1:
                self.task_required[
                    level_ids[:, None], base_rank[:, :2]
                ] = 1.0
                self.task_required[level_ids, delivery_rank[:, 0]] = 1.0
            elif level == 2:
                self.task_required[
                    level_ids[:, None], base_rank[:, :3]
                ] = 1.0
                self.task_required[
                    level_ids[:, None], delivery_rank[:, :2]
                ] = 1.0
            elif level == 3:
                self.task_required[level_ids, :5] = 1.0
                self.task_required[
                    level_ids[:, None], delivery_rank[:, :4]
                ] = 1.0
            else:
                self.task_required[level_ids] = 1.0

        delivery_replay_fraction = float(
            self.cfg.delivery_replay_fraction
        )
        if delivery_replay_fraction > 0.0:
            replay_mask = (
                torch.rand(count, device=self.device)
                < delivery_replay_fraction
            )
            replay_ids = env_ids[replay_mask]
            if replay_ids.numel() > 0:
                delivery_tasks = torch.randint(
                    5,
                    5 + ACTION_LAYOUT.object_dim,
                    (replay_ids.numel(),),
                    device=self.device,
                )
                self.task_required[replay_ids] = 0.0
                self.task_required[replay_ids, delivery_tasks] = 1.0
                self.curriculum_level[replay_ids] = 0
                self.curriculum_probe[replay_ids] = 0.0

    def _apply_interaction_start_curriculum(
        self, env_ids: torch.Tensor
    ) -> None:
        """Reset a single-object episode on a nearby physical source pad."""

        self.interaction_start_curriculum[env_ids] = 0.0
        self.interaction_start_source_platform[env_ids] = -1
        initial_probability = float(
            self.cfg.interaction_start_curriculum_probability
        )
        minimum_probability = float(
            self.cfg.interaction_start_curriculum_min_probability
        )
        decay_steps = int(
            self.cfg.interaction_start_curriculum_decay_steps
        )
        if decay_steps > 0:
            step = int(getattr(self._env, "common_step_counter", 0))
            decay_fraction = min(max(step / decay_steps, 0.0), 1.0)
            probability = initial_probability + decay_fraction * (
                minimum_probability - initial_probability
            )
        else:
            probability = initial_probability
        self._interaction_start_probability = probability
        if probability <= 0.0 or env_ids.numel() == 0:
            return

        fixed_object_id = int(self.cfg.interaction_start_object_id)
        if fixed_object_id >= 0:
            self.task_required[env_ids] = 0.0
            self.task_required[env_ids, 5 + fixed_object_id] = 1.0
        required_delivery = self.task_required[env_ids, 5:11]
        eligible = required_delivery.sum(dim=1) == 1.0
        selected = eligible & (
            torch.rand(env_ids.numel(), device=self.device) < probability
        )
        curriculum_ids = env_ids[selected]
        if curriculum_ids.numel() == 0:
            return

        object_ids = required_delivery[selected].argmax(dim=1)
        platform_map = torch.as_tensor(
            self.cfg.interaction_start_platform_map,
            device=self.device,
            dtype=torch.long,
        )
        source_platform_ids = platform_map[object_ids]
        platform_positions = self._asset_positions(
            self._target_platform_names
        )[curriculum_ids, source_platform_ids]
        target_offsets = torch.as_tensor(
            self.cfg.target_offsets,
            device=self.device,
            dtype=platform_positions.dtype,
        )
        source_positions = (
            platform_positions + target_offsets[object_ids]
        )
        jitter = float(self.cfg.interaction_start_position_jitter)
        if jitter > 0.0:
            source_positions[:, :2] += (
                2.0
                * torch.rand(
                    curriculum_ids.numel(),
                    2,
                    device=self.device,
                    dtype=source_positions.dtype,
                )
                - 1.0
            ) * jitter

        destination_positions = self._target_positions()[
            curriculum_ids, object_ids
        ]
        delivery_delta = (
            destination_positions[:, :2] - source_positions[:, :2]
        )
        delivery_direction = delivery_delta / torch.norm(
            delivery_delta, dim=1, keepdim=True
        ).clamp_min(1.0e-5)

        object_yaw = (
            2.0
            * torch.rand(
                curriculum_ids.numel(),
                device=self.device,
                dtype=source_positions.dtype,
            )
            - 1.0
        ) * 0.25
        for object_id, object_name in enumerate(self._object_names):
            object_mask = object_ids == object_id
            object_env_ids = curriculum_ids[object_mask]
            if object_env_ids.numel() == 0:
                continue
            asset = self._env.scene[object_name]
            object_pose = torch.cat(
                (
                    asset.data.root_pos_w[object_env_ids].clone(),
                    asset.data.root_quat_w[object_env_ids].clone(),
                ),
                dim=1,
            )
            object_pose[:, :3] = source_positions[object_mask]
            object_pose[:, 3:7] = _quat_from_yaw(
                object_yaw[object_mask]
            )
            asset.write_root_pose_to_sim(
                object_pose, env_ids=object_env_ids
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(
                    object_env_ids.numel(),
                    6,
                    device=self.device,
                    dtype=object_pose.dtype,
                ),
                env_ids=object_env_ids,
            )

        robot_pose = torch.cat(
            (
                self._robot.data.root_pos_w[curriculum_ids].clone(),
                self._robot.data.root_quat_w[curriculum_ids].clone(),
            ),
            dim=1,
        )
        robot_pose[:, :2] = (
            source_positions[:, :2]
            - float(self.cfg.interaction_start_base_standoff)
            * delivery_direction
        )
        approach_axis = delivery_direction
        robot_yaw = _wrap_to_pi(
            torch.atan2(
                approach_axis[:, 1],
                approach_axis[:, 0],
            )
            - float(self.cfg.interaction_arm_axis_yaw)
        )
        robot_pose[:, 3:7] = _quat_from_yaw(
            robot_yaw.to(dtype=robot_pose.dtype)
        )
        self._robot.write_root_pose_to_sim(
            robot_pose, env_ids=curriculum_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.zeros(
                curriculum_ids.numel(),
                6,
                device=self.device,
                dtype=robot_pose.dtype,
            ),
            env_ids=curriculum_ids,
        )
        self.interaction_start_curriculum[curriculum_ids] = 1.0
        self.interaction_start_source_platform[curriculum_ids] = (
            source_platform_ids
        )

    def _update_curriculum_frontier(self, env_ids: torch.Tensor):
        """Advance only from outcomes measured at the current frontier."""

        valid = (
            self._has_episode_history[env_ids]
            & (
                self.curriculum_level[env_ids]
                == self._curriculum_unlocked_level
            )
        )
        evidence_ids = env_ids[valid]
        if evidence_ids.numel() == 0:
            return

        required = self.task_required[evidence_ids]
        completion = (
            self.task_completed[evidence_ids] * required
        ).sum(dim=1) / required.sum(dim=1).clamp_min(1.0)
        progress = (
            self.task_progress[evidence_ids] * required
        ).amax(dim=1)
        safety = (
            self.episode_safe_sum[evidence_ids]
            / self.episode_evidence_steps[evidence_ids].clamp_min(1.0)
        ).clamp(0.0, 1.0)
        contact = self.object_contact_memory[evidence_ids].amax(dim=1)
        lift = self.object_lift_memory[evidence_ids].amax(dim=1)
        transport = self.object_transport_memory[evidence_ids].amax(dim=1)
        place = self.object_place[evidence_ids].amax(dim=1)
        required_delivery = required[:, 5:11]
        has_delivery = required_delivery.sum(dim=1) > 0.5
        delivery_completion = (
            self.object_completion[evidence_ids] * required_delivery
        ).sum(dim=1) / required_delivery.sum(dim=1).clamp_min(1.0)
        if torch.any(has_delivery):
            delivery_completion_observation = delivery_completion[
                has_delivery
            ].mean()
        elif self._curriculum_evidence_initialized:
            delivery_completion_observation = (
                self._curriculum_delivery_completion_ema.clone()
            )
        else:
            delivery_completion_observation = completion.new_zeros(())
        observations = (
            completion.mean(),
            progress.mean(),
            safety.mean(),
            contact.mean(),
            lift.mean(),
            transport.mean(),
            place.mean(),
            delivery_completion_observation,
        )
        buffers = (
            self._curriculum_completion_ema,
            self._curriculum_progress_ema,
            self._curriculum_safety_ema,
            self._curriculum_contact_ema,
            self._curriculum_lift_ema,
            self._curriculum_transport_ema,
            self._curriculum_place_ema,
            self._curriculum_delivery_completion_ema,
        )
        if not self._curriculum_evidence_initialized:
            for buffer, observation in zip(buffers, observations):
                buffer.copy_(observation)
            self._curriculum_evidence_initialized = True
        else:
            for buffer, observation in zip(buffers, observations):
                buffer.lerp_(observation, 0.05)

        step = int(getattr(self._env, "common_step_counter", 0))
        next_level = self._curriculum_unlocked_level + 1
        completion_value = float(
            self._curriculum_completion_ema.item()
        )
        progress_value = float(self._curriculum_progress_ema.item())
        safety_value = float(self._curriculum_safety_ema.item())
        event_values = (
            float(self._curriculum_contact_ema.item()),
            float(self._curriculum_lift_ema.item()),
            float(self._curriculum_transport_ema.item()),
            float(self._curriculum_place_ema.item()),
        )
        event_thresholds = (
            (0.050, 0.020, 0.015, 0.0008),
            (0.070, 0.030, 0.022, 0.0012),
            (0.090, 0.040, 0.030, 0.0018),
            (0.100, 0.050, 0.040, 0.0025),
        )
        delivery_completion_thresholds = (0.002, 0.003, 0.004, 0.006)
        if self._curriculum_unlocked_level > 0:
            retention_index = self._curriculum_unlocked_level - 1
            retention_ready = (
                safety_value >= 0.62
                and float(
                    self._curriculum_delivery_completion_ema.item()
                )
                >= 0.40
                * delivery_completion_thresholds[retention_index]
                and all(
                    value >= 0.40 * threshold
                    for value, threshold in zip(
                        event_values, event_thresholds[retention_index]
                    )
                )
            )
            if retention_ready:
                self._curriculum_regression_streak = max(
                    0, self._curriculum_regression_streak - 2
                )
            else:
                self._curriculum_regression_streak += 1
            if self._curriculum_regression_streak >= 48:
                self._curriculum_unlocked_level -= 1
                self._curriculum_regression_streak = 0
                self._curriculum_evidence_initialized = False
                return
        if next_level > 4:
            return
        if step < int(self.cfg.curriculum_boundaries[next_level]):
            return
        event_ready = all(
            value >= threshold
            for value, threshold in zip(
                event_values, event_thresholds[next_level - 1]
            )
        )
        mastery_ready = (
            safety_value >= 0.70
            and completion_value >= 0.005
            and progress_value >= 0.28
            and float(self._curriculum_delivery_completion_ema.item())
            >= delivery_completion_thresholds[next_level - 1]
            and event_ready
        )
        if mastery_ready:
            self._curriculum_unlocked_level = next_level
            self._curriculum_regression_streak = 0
            self._curriculum_evidence_initialized = False

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        self._update_curriculum_frontier(env_ids)
        self._assign_curriculum(env_ids)
        self._command[env_ids] = 0.0
        self.task_completed[env_ids] = 0.0
        self.task_progress[env_ids] = 0.0
        self.object_contact[env_ids] = 0.0
        self.object_contact_symmetry[env_ids] = 0.0
        self.object_contact_time[env_ids] = 0.0
        self.object_contact_memory[env_ids] = 0.0
        self.object_lift[env_ids] = 0.0
        self.object_lift_memory[env_ids] = 0.0
        self.object_transport[env_ids] = 0.0
        self.object_transport_memory[env_ids] = 0.0
        self.object_carry_memory[env_ids] = 0.0
        self.object_place[env_ids] = 0.0
        self.object_completion[env_ids] = 0.0
        self.object_target_progress[env_ids] = 0.0
        self.object_target_progress_delta[env_ids] = 0.0
        self.previous_object_target_progress[env_ids] = 0.0
        self.object_release_readiness[env_ids] = 0.0
        self.object_release_readiness_delta[env_ids] = 0.0
        self.previous_object_release_readiness[env_ids] = 0.0
        self.object_release_event[env_ids] = 0.0
        self.object_release_event_quality[env_ids] = 0.0
        self.object_drop_event[env_ids] = 0.0
        self.object_gripper_distance[env_ids] = 2.0
        self.object_tcp_offset[env_ids] = 0.0
        self.object_carry_anchor_b[env_ids] = 0.0
        self.object_carry_anchor_valid[env_ids] = False
        self.object_delivery_side[env_ids] = 1.0
        self.object_interaction_frontier[env_ids] = 0.0
        self.object_interaction_frontier_delta[env_ids] = 0.0
        self.object_initial_target_distance[env_ids] = 0.0
        self.object_carrying[env_ids] = False
        self.control_recovery_pressure[env_ids] = 0.0
        self.control_recovery_active[env_ids] = False
        self.control_recovery_constraint_active[env_ids] = False
        self.payload_posture_projection[env_ids] = 0.0
        self.payload_relation_violation[env_ids] = 0.0
        self.previous_task_error[env_ids] = 0.0
        self.previous_selected_task_progress[env_ids] = 0.0
        self.selected_task_progress_delta[env_ids] = 0.0
        self.selected_interaction_frontier[env_ids] = 0.0
        self.selected_interaction_frontier_delta[env_ids] = 0.0
        self.previous_mission_completion[env_ids] = 0.0
        self.mission_completion_delta[env_ids] = 0.0
        self.previous_mission_success[env_ids] = 0.0
        self.mission_success_event[env_ids] = 0.0
        self._ee_goal_initialized[env_ids] = False
        self._control_initialized[env_ids] = False
        self._progress_initialized[env_ids] = False
        self.mission_age[env_ids] = 0.0
        self.manipulation_active[env_ids] = False
        self.grasp_active[env_ids] = False
        self.grasp_orientation_active[env_ids] = False
        self.grasp_orientation_error[env_ids] = 0.0
        self.interaction_authority[env_ids] = 0.0
        self.workspace_handoff[env_ids] = 0.0
        self.grasp_projection_distance[env_ids] = 0.0
        self.grasp_center_error[env_ids] = 0.0
        self.capture_staging_distance[env_ids] = 0.0
        self.capture_heading_error[env_ids] = 0.0
        self.capture_initiation_margin[env_ids] = 0.0
        self.selected_delivery_side[env_ids] = 0.0
        self.capture_option_active[env_ids] = False
        self.capture_option_object[env_ids] = 0
        self.capture_lift_active[env_ids] = False
        self._apply_interaction_start_curriculum(env_ids)
        self.episode_safe_sum[env_ids] = 0.0
        self.episode_evidence_steps[env_ids] = 0.0
        self._initialized_objects[env_ids] = False
        self._has_episode_history[env_ids] = True

    def _hierarchy(self):
        return getattr(self._env, "tactic_hierarchy", None)

    def _selected_ids(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hierarchy = self._hierarchy()
        if hierarchy is None:
            zeros = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            return zeros, zeros, zeros
        return hierarchy.task_id, hierarchy.skill_id, hierarchy.object_id

    def _effective_object_ids(
        self, task_id: torch.Tensor, object_id: torch.Tensor
    ) -> torch.Tensor:
        delivery_task = (task_id >= 5) & (task_id <= 10)
        delivery_object = (task_id - 5).clamp(
            0, ACTION_LAYOUT.object_dim - 1
        )
        selected_object = torch.where(
            delivery_task, delivery_object, object_id
        ).clamp(0, ACTION_LAYOUT.object_dim - 1)
        carrying_any = self.object_carrying.any(dim=1)
        carried_object = self.object_carrying.float().argmax(dim=1)
        return torch.where(
            carrying_any, carried_object, selected_object
        )

    def _gripper_state(self):
        q = self._robot.data.joint_pos[:, self._finger_joint_ids]
        gap = torch.abs(q[:, 0] - q[:, 1])
        self.gripper_closure[:] = (
            1.0 - gap / max(1.0e-4, self.cfg.open_gap_threshold + 0.016)
        ).clamp(0.0, 1.0)

    def _bilateral_contact(self) -> torch.Tensor:
        left = self._env.scene["left_finger_object_contact"].data.force_matrix_w
        right = self._env.scene[
            "right_finger_object_contact"
        ].data.force_matrix_w
        if left is None or right is None:
            raise RuntimeError(
                "TACTIC requires filtered contact force matrices on both fingers"
            )
        left_force = torch.norm(left[:, 0], dim=-1)
        right_force = torch.norm(right[:, 0], dim=-1)
        width = min(left_force.shape[1], right_force.shape[1])
        if width < ACTION_LAYOUT.object_dim:
            raise RuntimeError("Finger contact filters do not cover six objects")
        threshold = float(self.cfg.contact_force_threshold)
        force_sum = left_force[:, :ACTION_LAYOUT.object_dim] + right_force[
            :, :ACTION_LAYOUT.object_dim
        ]
        self.object_contact_symmetry[:] = (
            2.0
            * torch.minimum(
                left_force[:, :ACTION_LAYOUT.object_dim],
                right_force[:, :ACTION_LAYOUT.object_dim],
            )
            / force_sum.clamp_min(1.0e-4)
        ).clamp(0.0, 1.0)
        return (
            (left_force[:, :ACTION_LAYOUT.object_dim] >= threshold)
            & (right_force[:, :ACTION_LAYOUT.object_dim] >= threshold)
        )

    def _update_object_events(self):
        object_pos = self._object_positions()
        target_pos = self._target_positions()
        was_carrying = self.object_carrying.clone()
        previous_release_readiness = (
            self.object_release_readiness.clone()
        )
        self.object_target_pos_w[:] = target_pos
        new_episode = self._env.episode_length_buf <= 1
        initialize = new_episode | (~self._initialized_objects)
        if torch.any(initialize):
            self.object_initial_pos_w[initialize] = object_pos[initialize]
            self.object_initial_target_distance[initialize] = torch.norm(
                object_pos[initialize, :, :2]
                - target_pos[initialize, :, :2],
                dim=-1,
            )
            self._initialized_objects[initialize] = True

        self._gripper_state()
        bilateral = self._bilateral_contact()
        dt = float(getattr(self._env, "step_dt", 1.0 / 30.0))
        self.object_contact_time[:] = torch.where(
            bilateral,
            self.object_contact_time + dt,
            (self.object_contact_time - 2.0 * dt).clamp_min(0.0),
        )
        contact = (
            self.object_contact_time >= float(self.cfg.contact_hold_s)
        ).float()
        self.object_contact[:] = contact
        self.object_contact_memory[:] = torch.maximum(
            0.995 * self.object_contact_memory, contact
        )

        lift_height = object_pos[:, :, 2] - self.object_initial_pos_w[:, :, 2]
        self.object_lift[:] = (
            lift_height / max(1.0e-4, self.cfg.lift_height)
        ).clamp(0.0, 1.0)
        lift_evidence = self.object_lift * self.object_contact_memory
        self.object_lift_memory[:] = torch.maximum(
            0.996 * self.object_lift_memory, lift_evidence
        )
        closure = self.gripper_closure.unsqueeze(1)
        finger_pos = self._robot.data.body_pos_w[:, self._finger_body_ids]
        gripper_center = finger_pos.mean(dim=1)
        self.object_gripper_distance[:] = torch.norm(
            object_pos - gripper_center.unsqueeze(1), dim=-1
        )
        carry_enter = (
            (self.object_contact_memory > 0.45)
            & (self.object_lift > 0.20)
            & (closure > 0.20)
            & (self.object_gripper_distance < 0.18)
        )
        carry_keep = (
            self.object_carrying
            & (closure > 0.12)
            & (
                bilateral
                | (self.object_contact_memory > 0.30)
            )
            & (self.object_gripper_distance < 0.22)
            & (self.object_completion < 0.5)
        )
        new_carry = carry_enter & (~was_carrying)
        measured_tcp_offset = (
            self.tcp_pos_w.unsqueeze(1) - object_pos
        ).clamp(-0.16, 0.16)
        self.object_tcp_offset[:] = torch.where(
            new_carry.unsqueeze(-1),
            measured_tcp_offset,
            self.object_tcp_offset,
        )
        yaw_q = _quat_from_yaw(self._robot.data.heading_w)
        root = self._robot.data.root_pos_w
        base_zero = torch.stack(
            (root[:, 0], root[:, 1], torch.zeros_like(root[:, 2])),
            dim=-1,
        )
        measured_carry_anchor_b = _quat_rotate_inverse(
            yaw_q, self.tcp_pos_w - base_zero
        ).unsqueeze(1)
        self.object_carry_anchor_b[:] = torch.where(
            new_carry.unsqueeze(-1),
            measured_carry_anchor_b,
            self.object_carry_anchor_b,
        )
        self.object_carry_anchor_valid[:] = (
            self.object_carry_anchor_valid | new_carry
        )
        self.object_carrying[:] = carry_enter | carry_keep
        self.object_carry_memory[:] = torch.maximum(
            self.object_carry_memory,
            self.object_carrying.float(),
        )

        displacement = torch.norm(
            object_pos[:, :, :2] - self.object_initial_pos_w[:, :, :2],
            dim=-1,
        )
        displacement_progress = (
            displacement / max(1.0e-4, self.cfg.transport_distance)
        ).clamp(0.0, 1.0)
        target_distance = torch.norm(
            object_pos[:, :, :2] - target_pos[:, :, :2], dim=-1
        )
        target_progress = (
            (
                self.object_initial_target_distance - target_distance
            )
            / self.object_initial_target_distance.clamp_min(1.0e-4)
        ).clamp(-1.0, 1.0)
        self.object_target_progress[:] = target_progress
        self.object_target_progress_delta[:] = torch.where(
            self.object_carrying,
            (
                target_progress - self.previous_object_target_progress
            ).clamp(-0.10, 0.10),
            torch.zeros_like(target_progress),
        )
        self.previous_object_target_progress[:] = target_progress
        target_progress = (
            (
                self.object_initial_target_distance - target_distance
            )
            / max(1.0e-4, self.cfg.transport_distance)
        ).clamp(0.0, 1.0)
        self.object_transport[:] = torch.minimum(
            displacement_progress, target_progress
        )
        transport_evidence = self.object_transport * self.object_lift_memory
        self.object_transport_memory[:] = torch.maximum(
            0.997 * self.object_transport_memory, transport_evidence
        )

        error = object_pos - target_pos
        error_xy = torch.norm(error[:, :, :2], dim=-1)
        error_z = torch.abs(error[:, :, 2])
        self.object_place_error_xy[:] = error_xy
        self.object_place_error_z[:] = error_z
        hover_error_z = torch.abs(
            object_pos[:, :, 2]
            - target_pos[:, :, 2]
            - RELEASE_HOVER_HEIGHT
        )
        release_readiness = (
            self.object_carrying.float()
            * (0.20 + 0.80 * self.object_transport_memory)
            * torch.exp(
                -error_xy / 0.20
                -0.5 * hover_error_z / 0.10
            )
        ).clamp(0.0, 1.0)
        self.object_release_readiness[:] = release_readiness
        self.object_release_readiness_delta[:] = torch.where(
            self.object_carrying,
            (
                release_readiness
                - self.previous_object_release_readiness
            ).clamp(-0.10, 0.10),
            torch.zeros_like(release_readiness),
        )
        self.previous_object_release_readiness[:] = release_readiness
        open_gripper = self.gripper_closure < 0.22
        placed = (
            (error_xy <= float(self.cfg.place_xy_tolerance))
            & (error_z <= float(self.cfg.place_z_tolerance))
            & (self.object_lift_memory > 0.45)
            & (self.object_transport_memory > 0.45)
            & (self.object_carry_memory > 0.5)
            & open_gripper.unsqueeze(1)
        )
        place_score = torch.exp(
            -error_xy / max(1.0e-4, self.cfg.place_xy_tolerance)
            -0.5 * error_z / max(1.0e-4, self.cfg.place_z_tolerance)
        )
        self.object_place[:] = place_score * (
            self.object_lift_memory
            * self.object_transport_memory
            * self.object_carry_memory
        ).sqrt()
        self.object_completion[:] = torch.maximum(
            self.object_completion, placed.float()
        )
        self.object_carrying[self.object_completion > 0.5] = False
        released = was_carrying & (~self.object_carrying)
        successful_placement_release = released & (
            self.object_completion > 0.5
        )
        intended_release = (
            released
            & (
                (previous_release_readiness >= 0.30)
                | successful_placement_release
            )
        )
        self.object_release_event[:] = intended_release.float()
        self.object_release_event_quality[:] = (
            intended_release.float()
            * torch.maximum(
                previous_release_readiness,
                successful_placement_release.float(),
            )
        )
        self.object_drop_event[:] = (
            released & (~intended_release)
        ).float()
        interaction_evidence = (
            0.16 * self.object_contact_memory
            + 0.24 * self.object_lift_memory
            + 0.25 * self.object_transport_memory
            + 0.35 * self.object_place
        ).clamp(0.0, 1.0)
        interaction_evidence = torch.maximum(
            interaction_evidence,
            0.90 * self.object_release_readiness,
        )
        interaction_evidence = torch.maximum(
            interaction_evidence, self.object_completion
        )
        next_frontier = torch.maximum(
            self.object_interaction_frontier, interaction_evidence
        )
        self.object_interaction_frontier_delta[:] = (
            next_frontier - self.object_interaction_frontier
        ).clamp(0.0, 1.0)
        self.object_interaction_frontier[:] = next_frontier

    def _base_measurements(self):
        if hasattr(self._robot.data, "projected_gravity_b"):
            self.base_tilt[:] = torch.norm(
                self._robot.data.projected_gravity_b[:, :2], dim=-1
            )
        self.base_height[:] = self._robot.data.root_pos_w[:, 2]
        sensor = self._env.scene["contact_forces"]
        forces = sensor.data.net_forces_w[:, self._foot_sensor_ids, :]
        self.support_count[:] = (
            torch.norm(forces, dim=-1) > 1.5
        ).float().sum(dim=1)

        if not self._obstacle_names:
            self.obstacle_margin.fill_(2.0)
            return
        obstacle_xy = self._asset_positions(self._obstacle_names)[:, :, :2]
        base_xy = self._robot.data.root_pos_w[:, None, :2]
        self.obstacle_margin[:] = torch.norm(
            base_xy - obstacle_xy, dim=-1
        ).min(dim=1).values

    def _update_control_recovery_admissibility(self):
        """Expose a hysteretic recovery candidate from measured margins."""

        tilt_pressure = (
            (self.base_tilt - 0.30)
            / max(1.0e-4, float(self.cfg.tilt_limit) - 0.30)
        ).clamp(0.0, 1.0)
        support_pressure = (
            (2.0 - self.support_count) / 1.0
        ).clamp(0.0, 1.0)
        margin_pressure = (
            (
                0.16
                - torch.minimum(
                    self.safety_margin, self.preview_margin
                )
            )
            / 0.16
        ).clamp(0.0, 1.0)
        pressure = torch.maximum(
            tilt_pressure,
            torch.maximum(support_pressure, margin_pressure),
        )
        margin_ready = self._env.episode_length_buf > 2
        instantaneous_pressure = torch.maximum(
            tilt_pressure, support_pressure
        )
        pressure = torch.where(
            margin_ready, pressure, instantaneous_pressure
        )
        enter = pressure >= 0.55
        leave = pressure <= 0.20
        self.control_recovery_active[:] = torch.where(
            self.control_recovery_active,
            ~leave,
            enter,
        )
        self.control_recovery_constraint_active[:] = (
            (self.base_tilt >= 0.92 * float(self.cfg.tilt_limit))
            | (self.support_count < 1.5)
            | (
                margin_ready
                & (
                    (pressure >= 0.85)
                    | (self.safety_margin < 0.03)
                    | (self.preview_margin < 0.03)
                )
            )
        )
        self.control_recovery_pressure[:] = pressure
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            hierarchy.control_recovery_pressure.copy_(
                self.control_recovery_pressure
            )
            hierarchy.control_recovery_active.copy_(
                self.control_recovery_active
            )
            hierarchy.control_recovery_constraint_active.copy_(
                self.control_recovery_constraint_active
            )

    def _payload_recovery_candidate(self) -> torch.Tensor:
        """Open the auxiliary recovery task only for a secured payload."""

        carrying = self.object_carrying.max(dim=1).values > 0.5
        return self.control_recovery_active & carrying

    def _recovery_regulation_active(self) -> torch.Tensor:
        """Regulate a required recovery task or an active payload transient."""

        recovery_required = self.task_required[:, 4] > 0.5
        return self.control_recovery_active & (
            recovery_required | (self.object_carrying.max(dim=1).values > 0.5)
        )

    def _task_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self._scene_origins()
        root = self._robot.data.root_pos_w
        heading = self._robot.data.heading_w
        recovery_regulation = self._recovery_regulation_active()
        targets = torch.zeros(
            self.num_envs, TASK_SLOT_COUNT, 3, device=self.device
        )
        headings = torch.zeros(
            self.num_envs, TASK_SLOT_COUNT, device=self.device
        )
        for slot in range(3):
            targets[:, slot, :2] = (
                origins[:, :2] + self._mission_goal_xy[slot]
            )
        targets[:, 3] = self._target_positions().mean(dim=1)
        targets[:, 3, :2] += torch.tensor(
            (0.50, -0.45), device=self.device
        )
        targets[:, 4, :2] = (
            origins[:, :2] + self._mission_goal_xy[4]
        )
        # A transient recovery option regulates the current pose.  Once the
        # hysteretic viability latch clears, slot 4 returns to its mission
        # target without changing the mission ledger.
        targets[:, 4, :2] = torch.where(
            recovery_regulation.unsqueeze(1),
            root[:, :2],
            targets[:, 4, :2],
        )
        headings[:, 4] = torch.where(
            recovery_regulation,
            heading,
            headings[:, 4],
        )

        object_pos = self._object_positions()
        target_pos = self._target_positions()
        hover_pos = self._placement_hover_positions(target_pos)
        carrying = self.object_carrying
        payload_delta = (
            target_pos[:, :, :2] - object_pos[:, :, :2]
        )
        payload_distance = torch.norm(
            payload_delta, dim=-1, keepdim=True
        ).clamp_min(1.0e-5)
        payload_direction = payload_delta / payload_distance
        nominal_delivery_delta = (
            target_pos[:, :, :2]
            - self.object_initial_pos_w[:, :, :2]
        )
        nominal_delivery_direction = nominal_delivery_delta / torch.norm(
            nominal_delivery_delta, dim=-1, keepdim=True
        ).clamp_min(1.0e-5)
        delivery_direction = torch.where(
            payload_distance > 0.08,
            payload_direction,
            nominal_delivery_direction,
        )
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            rows = torch.arange(self.num_envs, device=self.device)
            selected_task = hierarchy.task_id
            selected_object = self._effective_object_ids(
                selected_task, hierarchy.object_id
            )
            selected_delivery = (
                (selected_task >= 5) & (selected_task <= 10)
            )
            selected_carrying = self.object_carrying[
                rows, selected_object
            ]
            current_side = self.object_delivery_side[
                rows, selected_object
            ]
            side_score = hierarchy.task_subgoal[:, 3]
            threshold = float(self.cfg.delivery_side_hysteresis)
            proposed_side = torch.where(
                side_score > threshold,
                torch.ones_like(side_score),
                torch.where(
                    side_score < -threshold,
                    -torch.ones_like(side_score),
                    current_side,
                ),
            )
            update_side = selected_delivery & (~selected_carrying)
            self.object_delivery_side[rows, selected_object] = torch.where(
                update_side,
                proposed_side,
                current_side,
            )
        delivery_side = self.object_delivery_side
        delivery_anchor = torch.where(
            carrying.unsqueeze(-1), hover_pos, object_pos
        )
        root_xy = self._robot.data.root_pos_w[:, None, :2]
        # The task layer selects one of two relation charts.  Positive means
        # the rear arm pulls the object while the base travels forward;
        # negative means the base approaches from behind and reverses.
        pre_contact_target = (
            object_pos[:, :, :2]
            + float(self.cfg.interaction_base_standoff)
            * delivery_side.unsqueeze(-1)
            * delivery_direction
        )
        approach_axis = (
            -delivery_side.unsqueeze(-1) * delivery_direction
        )
        relation_heading = torch.atan2(
            approach_axis[:, :, 1],
            approach_axis[:, :, 0],
        )
        relation_heading = _wrap_to_pi(
            relation_heading - float(self.cfg.interaction_arm_axis_yaw)
        )
        # A payload-relative target has a fixed point at zero placement error.
        # A target reconstructed from the root-to-platform bearing flips sides
        # as the base passes the platform and prevents a stable release.
        carrying_target = (
            root_xy + PAYLOAD_TASK_CONTRACTION_GAIN * payload_delta
        )
        targets[:, 5:11, :2] = torch.where(
            carrying.unsqueeze(-1),
            carrying_target,
            pre_contact_target,
        )
        targets[:, 5:11, 2] = delivery_anchor[:, :, 2]
        pre_contact_heading = _wrap_to_pi(
            relation_heading
        )
        headings[:, 5:11] = torch.where(
            carrying,
            relation_heading,
            pre_contact_heading,
        )
        targets[:, 11, :2] = (
            origins[:, :2] + self._mission_goal_xy[5]
        )
        return targets, headings

    def _update_task_ledger(
        self, targets: torch.Tensor, headings: torch.Tensor
    ):
        root = self._robot.data.root_pos_w
        distance = torch.norm(
            targets[:, :, :2] - root[:, None, :2], dim=-1
        )
        navigation_progress = torch.exp(-distance / 0.75)
        self.task_progress[:, :5] = navigation_progress[:, :5]
        object_pos = self._object_positions()
        target_pos = self._target_positions()
        base_object_distance = torch.norm(
            object_pos[:, :, :2] - root[:, None, :2], dim=-1
        )
        tcp_object_distance = torch.norm(
            object_pos - self.tcp_pos_w.unsqueeze(1), dim=-1
        )
        object_target_distance = torch.norm(
            object_pos[:, :, :2] - target_pos[:, :, :2], dim=-1
        )
        approach_progress = torch.exp(-base_object_distance / 0.80)
        reach_progress = torch.exp(-tcp_object_distance / 0.18)
        target_progress = (
            torch.exp(-object_target_distance / 0.45)
            * self.object_lift_memory
        )
        terminal_progress = torch.maximum(
            torch.maximum(
                self.object_place,
                self.object_release_readiness,
            ),
            self.object_completion,
        )
        dense_reach_progress = (
            0.48 * approach_progress + 0.52 * reach_progress
        )
        interaction_frontier = self.object_interaction_frontier
        delivery_progress = (
            0.22
            * dense_reach_progress
            * (1.0 - 0.85 * interaction_frontier)
            + 0.68 * interaction_frontier
            + 0.10 * target_progress
        ).clamp(0.0, 1.0)
        self.task_progress[:, 5:11] = torch.where(
            self.object_completion > 0.5,
            torch.ones_like(delivery_progress),
            torch.maximum(delivery_progress, 0.95 * terminal_progress),
        )

        reached = distance <= float(self.cfg.task_goal_radius)
        self.task_completed[:, :3] = torch.maximum(
            self.task_completed[:, :3], reached[:, :3].float()
        )
        self.task_completed[:, 3] = torch.maximum(
            self.task_completed[:, 3],
            (
                (self.ee_error < 0.12)
                & (distance[:, 3] < 0.55)
            ).float(),
        )
        stable = (
            (self.safety_margin > 0.65)
            & (self.preview_margin > 0.55)
            & (distance[:, 4] < 0.45)
            & (~self._recovery_regulation_active())
        )
        self.task_completed[:, 4] = torch.maximum(
            self.task_completed[:, 4], stable.float()
        )
        self.task_completed[:, 5:11] = self.object_completion
        prerequisite = (
            self.task_completed[:, :11]
            + (1.0 - self.task_required[:, :11])
        ).min(dim=1).values
        final_reached = (
            distance[:, 11] <= float(self.cfg.final_goal_radius)
        ).float()
        self.task_completed[:, 11] = torch.maximum(
            self.task_completed[:, 11], prerequisite * final_reached
        )
        self.task_progress[:, 11] = prerequisite * torch.exp(
            -distance[:, 11] / 0.85
        )

        self.task_available.fill_(1.0)
        self.task_available[:, 11] = prerequisite
        valid = (
            self.task_required
            * (1.0 - self.task_completed)
            * self.task_available
        )
        valid[:, 4] = torch.maximum(
            valid[:, 4],
            self._payload_recovery_candidate().float(),
        )
        no_valid = valid.sum(dim=1) < 0.5
        valid[no_valid, 11] = 1.0
        self.task_valid_mask.copy_(valid)
        self._env.tactic_task_valid_mask = self.task_valid_mask
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            hierarchy.task_valid_mask.copy_(self.task_valid_mask)
        self.mission_completion[:] = (
            self.task_completed * self.task_required
        ).sum(dim=1) / self.task_required.sum(dim=1).clamp_min(1.0)
        self.mission_success[:] = (
            (
                self.task_completed
                + (1.0 - self.task_required)
            ).min(dim=1).values
            > 0.5
        ).float()

    def _update_control_signals(
        self, targets: torch.Tensor, headings: torch.Tensor
    ):
        task_id, _, _ = self._selected_ids()
        rows = torch.arange(self.num_envs, device=self.device)
        selected_target = targets[rows, task_id]
        root = self._robot.data.root_pos_w
        heading = self._robot.data.heading_w
        delta = selected_target[:, :2] - root[:, :2]
        distance = torch.norm(delta, dim=-1)
        desired_heading = torch.atan2(delta[:, 1], delta[:, 0])
        yaw_error = _wrap_to_pi(desired_heading - heading)
        self.task_error[:] = distance + 0.20 * yaw_error.abs()
        dt = float(getattr(self._env, "step_dt", 1.0 / 30.0))
        decrease = (self.previous_task_error - self.task_error) / max(
            dt, 1.0e-4
        )
        self.clf_decrease[:] = torch.where(
            self._control_initialized,
            decrease,
            torch.zeros_like(decrease),
        )
        self.clf_decrease_score[:] = torch.sigmoid(
            2.5 * self.clf_decrease
        )
        self.previous_task_error[:] = self.task_error
        self._control_initialized.fill_(True)

        height_margin = (
            (self.base_height - self.cfg.base_height_min)
            / max(1.0e-4, self.cfg.base_height_target - self.cfg.base_height_min)
        )
        tilt_margin = 1.0 - self.base_tilt / max(
            1.0e-4, self.cfg.tilt_limit
        )
        support_margin = (
            (self.support_count - self.cfg.support_min)
            / max(1.0e-4, 4.0 - self.cfg.support_min)
        )
        obstacle_margin = (
            (self.obstacle_margin - self.cfg.obstacle_stop_margin)
            / max(
                1.0e-4,
                self.cfg.obstacle_safe_margin
                - self.cfg.obstacle_stop_margin,
            )
        )
        margins = torch.stack(
            (height_margin, tilt_margin, support_margin, obstacle_margin),
            dim=-1,
        )
        self.safety_margin[:] = margins.min(dim=1).values.clamp(0.0, 1.0)

        horizon = float(self.cfg.preview_horizon_s)
        vertical_velocity = self._robot.data.root_lin_vel_b[:, 2]
        tilt_rate = torch.norm(
            self._robot.data.root_ang_vel_b[:, :2], dim=-1
        )
        predicted_height = self.base_height + horizon * vertical_velocity
        predicted_tilt = self.base_tilt + 0.35 * horizon * tilt_rate
        predicted_obstacle = self.obstacle_margin - horizon * (
            self._command[:, 0].abs()
            + 0.20 * self._command[:, 1].abs()
        )
        predicted = torch.stack(
            (
                (predicted_height - self.cfg.base_height_min)
                / max(
                    1.0e-4,
                    self.cfg.base_height_target - self.cfg.base_height_min,
                ),
                1.0 - predicted_tilt / max(1.0e-4, self.cfg.tilt_limit),
                support_margin,
                (predicted_obstacle - self.cfg.obstacle_stop_margin)
                / max(
                    1.0e-4,
                    self.cfg.obstacle_safe_margin
                    - self.cfg.obstacle_stop_margin,
                ),
            ),
            dim=-1,
        )
        self.preview_margin[:] = predicted.min(dim=1).values.clamp(0.0, 1.0)

        tracking_error = (
            (self._command[:, 0] - self._robot.data.root_lin_vel_b[:, 0]).abs()
            + 0.35
            * (
                self._command[:, 2]
                - self._robot.data.root_ang_vel_b[:, 2]
            ).abs()
            + 0.20 * tilt_rate
        )
        alpha = float(self.cfg.disturbance_filter)
        self.disturbance_estimate[:] = (
            alpha * self.disturbance_estimate
            + (1.0 - alpha) * tracking_error
        ).clamp(0.0, 2.0)
        self.disturbance_quality[:] = (
            1.0 - self.disturbance_estimate / 1.25
        ).clamp(0.0, 1.0)

    def _update_task_slots(
        self, targets: torch.Tensor, headings: torch.Tensor
    ):
        root = self._robot.data.root_pos_w
        yaw = self._robot.data.heading_w
        yaw_q = _quat_from_yaw(yaw)
        delta_w = targets - root[:, None, :]
        yaw_q_slots = yaw_q[:, None, :].expand(-1, TASK_SLOT_COUNT, -1)
        delta_b = _quat_rotate_inverse(
            yaw_q_slots.reshape(-1, 4), delta_w.reshape(-1, 3)
        ).reshape(self.num_envs, TASK_SLOT_COUNT, 3)
        distance = torch.norm(delta_b[:, :, :2], dim=-1)
        travel_heading_error = _wrap_to_pi(
            torch.atan2(delta_b[:, :, 1], delta_b[:, :, 0])
        )
        final_heading_error = _wrap_to_pi(
            headings - yaw.unsqueeze(1)
        )
        delivery_mask = torch.zeros(
            TASK_SLOT_COUNT, dtype=torch.bool, device=self.device
        )
        delivery_mask[5:11] = True
        heading_error = torch.where(
            delivery_mask.unsqueeze(0),
            final_heading_error,
            travel_heading_error,
        )
        self.task_slots.zero_()
        self.task_slots[:, :, 0:2] = (delta_b[:, :, 0:2] / 3.0).clamp(
            -2.0, 2.0
        )
        self.task_slots[:, :, 2] = (delta_b[:, :, 2] / 1.0).clamp(
            -2.0, 2.0
        )
        self.task_slots[:, :, 3] = (distance / 4.0).clamp(0.0, 2.0)
        self.task_slots[:, :, 4] = heading_error / math.pi

        type_ids = torch.tensor(
            [
                TASK_TYPE_ROUTE,
                TASK_TYPE_SLALOM,
                TASK_TYPE_NARROW,
                TASK_TYPE_MANIPULATION,
                TASK_TYPE_RECOVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_DELIVERY,
                TASK_TYPE_ROUTE,
            ],
            device=self.device,
        )
        type_one_hot = torch.nn.functional.one_hot(
            type_ids, num_classes=TASK_TYPE_COUNT
        ).float()
        self.task_slots[:, :, 5:11] = type_one_hot.unsqueeze(0)
        policy_required = self.task_required.clone()
        if self.cfg.automatic_recovery_task_candidate:
            policy_required[:, 4] = torch.maximum(
                policy_required[:, 4],
                self._payload_recovery_candidate().float(),
            )
        policy_completed = self.task_completed.clone()
        policy_completed[:, 4] = torch.where(
            self._recovery_regulation_active(),
            torch.zeros_like(policy_completed[:, 4]),
            policy_completed[:, 4],
        )
        self.task_slots[:, :, 11] = policy_required
        self.task_slots[:, :, 12] = policy_completed
        self.task_slots[:, :, 13] = self.task_available
        self.task_slots[:, 5:11, 14] = self._shape_codes.view(1, -1)
        self.task_slots[:, :, 15] = (
            1.0 - self.task_progress
        ).clamp(0.0, 1.0)

        object_pos = self._object_positions()
        target_pos = self._target_positions()
        hover_pos = self._placement_hover_positions(target_pos)
        object_delta_w = object_pos - self.tcp_pos_w.unsqueeze(1)
        target_delta_w = target_pos - object_pos
        yaw_q_objects = yaw_q[:, None, :].expand(
            -1, ACTION_LAYOUT.object_dim, -1
        )
        object_delta_b = _quat_rotate_inverse(
            yaw_q_objects.reshape(-1, 4),
            object_delta_w.reshape(-1, 3),
        ).reshape(self.num_envs, ACTION_LAYOUT.object_dim, 3)
        target_delta_b = _quat_rotate_inverse(
            yaw_q_objects.reshape(-1, 4),
            target_delta_w.reshape(-1, 3),
        ).reshape(self.num_envs, ACTION_LAYOUT.object_dim, 3)
        delivery_slots = self.task_slots[:, 5:11]
        delivery_slots[:, :, 16:19] = object_delta_b.clamp(-2.0, 2.0)
        delivery_slots[:, :, 19:22] = (
            target_delta_b / 1.5
        ).clamp(-2.0, 2.0)
        delivery_slots[:, :, 22] = self.object_contact_memory
        delivery_slots[:, :, 23] = self.object_lift_memory
        delivery_slots[:, :, 24] = self.object_transport_memory
        delivery_slots[:, :, 25] = self.object_place
        delivery_slots[:, :, 26] = self.gripper_closure.unsqueeze(1)
        carrying = self.object_carrying
        interaction_anchor = torch.where(
            carrying.unsqueeze(-1), hover_pos, object_pos
        )
        workspace_distance = torch.norm(
            interaction_anchor - self.center_w.unsqueeze(1), dim=-1
        )
        workspace_gate = torch.sigmoid(
            12.0
            * (
                float(self.cfg.interaction_workspace_radius)
                - workspace_distance
            )
        )
        staging_distance = distance[:, 5:11]
        staging_heading_error = final_heading_error[:, 5:11].abs()
        position_gate = torch.sigmoid(
            float(self.cfg.capture_position_gate_gain)
            * (
                float(self.cfg.capture_staging_radius)
                - staging_distance
            )
        )
        heading_gate = torch.sigmoid(
            float(self.cfg.capture_heading_gate_gain)
            * (
                float(self.cfg.capture_heading_tolerance)
                - staging_heading_error
            )
        )
        initiation_gate = torch.minimum(position_gate, heading_gate)
        delivery_slots[:, :, 27] = torch.where(
            carrying,
            workspace_gate,
            torch.minimum(workspace_gate, initiation_gate),
        )
        delivery_slots[:, :, 28:32] = self._object_descriptors.unsqueeze(0)
        finger_pos = self._robot.data.body_pos_w[:, self._finger_body_ids]
        left_delta_w = object_pos - finger_pos[:, 0].unsqueeze(1)
        right_delta_w = object_pos - finger_pos[:, 1].unsqueeze(1)
        left_delta_b = _quat_rotate_inverse(
            yaw_q_objects.reshape(-1, 4),
            left_delta_w.reshape(-1, 3),
        ).reshape(self.num_envs, ACTION_LAYOUT.object_dim, 3)
        right_delta_b = _quat_rotate_inverse(
            yaw_q_objects.reshape(-1, 4),
            right_delta_w.reshape(-1, 3),
        ).reshape(self.num_envs, ACTION_LAYOUT.object_dim, 3)
        delivery_slots[:, :, 32:35] = (left_delta_b / 0.75).clamp(
            -2.0, 2.0
        )
        delivery_slots[:, :, 35:38] = (right_delta_b / 0.75).clamp(
            -2.0, 2.0
        )
        delivery_slots[:, :, TASK_SLOT_CARRYING_INDEX] = (
            self.object_carrying.float()
        )
        delivery_slots[:, :, 39] = self.object_contact_symmetry

    def _update_learned_commands(
        self,
        targets: torch.Tensor,
        headings: torch.Tensor,
    ):
        hierarchy = self._hierarchy()
        if hierarchy is None:
            self._command.zero_()
            task_id = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            object_id = torch.zeros_like(task_id)
            task_subgoal = torch.zeros(
                self.num_envs,
                ACTION_LAYOUT.task_subgoal_dim,
                device=self.device,
            )
            skill_parameter = torch.zeros(
                self.num_envs,
                ACTION_LAYOUT.skill_param_dim,
                device=self.device,
            )
        else:
            task_id = hierarchy.task_id
            object_id = hierarchy.object_id
            task_subgoal = hierarchy.task_subgoal
            skill_parameter = hierarchy.skill_parameter

        speed_scale = 0.35 + 0.65 * (skill_parameter[:, 0] + 1.0) * 0.5
        yaw_scale = 0.35 + 0.65 * (skill_parameter[:, 1] + 1.0) * 0.5
        command_vx = (
            float(self.cfg.max_forward_speed)
            * speed_scale
            * task_subgoal[:, 0]
        )
        command_vy = (
            float(self.cfg.max_lateral_speed)
            * speed_scale
            * task_subgoal[:, 1]
        )
        semantic_wz = (
            float(self.cfg.max_yaw_rate)
            * yaw_scale
            * task_subgoal[:, 2]
        )
        travel_sign = torch.where(
            command_vx.abs() > 0.015,
            command_vx.sign(),
            torch.ones_like(command_vx),
        )
        lateral_heading = torch.atan2(
            travel_sign * command_vy, command_vx.abs() + 0.03
        )
        self._command[:, 0] = command_vx
        self._command[:, 1] = command_vy
        self._command[:, 2] = (
            travel_sign * semantic_wz + 0.75 * lateral_heading
        ).clamp(
            -float(self.cfg.max_yaw_rate),
            float(self.cfg.max_yaw_rate),
        )
        settling = self.mission_age < float(
            self.cfg.hierarchy_settling_time_s
        )
        self._command[settling] = 0.0

        root = self._robot.data.root_pos_w
        yaw_q = _quat_from_yaw(self._robot.data.heading_w)
        base_zero = torch.stack(
            (root[:, 0], root[:, 1], torch.zeros_like(root[:, 2])),
            dim=-1,
        )
        self.center_w[:] = base_zero + _quat_apply(
            yaw_q, self._center_offset.expand(self.num_envs, -1)
        )
        default_goal = base_zero + _quat_apply(
            yaw_q, self._default_ee_offset.expand(self.num_envs, -1)
        )
        reset_goal = base_zero + _quat_apply(
            yaw_q, self._reset_ee_offset.expand(self.num_envs, -1)
        )

        object_pos = self._object_positions()
        target_pos = self._target_positions()
        hover_pos = self._placement_hover_positions(target_pos)
        rows = torch.arange(self.num_envs, device=self.device)
        delivery_task = (task_id >= 5) & (task_id <= 10)
        selected_task_type = self.task_slots[rows, task_id, 5:11]
        manipulation_task = (
            selected_task_type[:, TASK_TYPE_MANIPULATION] > 0.5
        )
        effective_object = self._effective_object_ids(
            task_id, object_id
        )
        selected_object_pos = object_pos[rows, effective_object]
        selected_target_pos = target_pos[rows, effective_object]
        carrying = self.object_carrying[rows, effective_object]
        selected_tcp_offset = self.object_tcp_offset[
            rows, effective_object
        ]
        selected_carry_anchor_b = self.object_carry_anchor_b[
            rows, effective_object
        ]
        selected_carry_anchor_valid = self.object_carry_anchor_valid[
            rows, effective_object
        ]
        carry_anchor_w = base_zero + _quat_apply(
            yaw_q, selected_carry_anchor_b
        )
        carry_anchor_w = torch.where(
            selected_carry_anchor_valid.unsqueeze(1),
            carry_anchor_w,
            selected_object_pos + selected_tcp_offset,
        )
        selected_base_target = targets[rows, task_id]
        selected_heading_target = headings[rows, task_id]
        staging_distance = torch.norm(
            selected_base_target[:, :2] - self._robot.data.root_pos_w[:, :2],
            dim=-1,
        )
        heading_error = _wrap_to_pi(
            selected_heading_target - self._robot.data.heading_w
        ).abs()
        position_margin = (
            float(self.cfg.capture_staging_radius) - staging_distance
        )
        heading_margin = (
            float(self.cfg.capture_heading_tolerance) - heading_error
        )
        capture_admissible = (
            (position_margin >= 0.0)
            & (heading_margin >= 0.0)
        )
        self.capture_staging_distance[:] = torch.where(
            delivery_task,
            staging_distance,
            torch.zeros_like(staging_distance),
        )
        self.capture_heading_error[:] = torch.where(
            delivery_task,
            heading_error,
            torch.zeros_like(heading_error),
        )
        self.capture_initiation_margin[:] = torch.where(
            delivery_task,
            torch.minimum(
                position_margin
                / max(float(self.cfg.capture_staging_radius), 1.0e-6),
                heading_margin
                / max(float(self.cfg.capture_heading_tolerance), 1.0e-6),
            ),
            torch.zeros_like(staging_distance),
        )
        self.selected_delivery_side[:] = torch.where(
            delivery_task,
            self.object_delivery_side[rows, effective_object],
            torch.zeros_like(staging_distance),
        )
        payload_relation_active = carrying
        self.manipulation_active[:] = (
            manipulation_task | delivery_task | payload_relation_active
        )
        self.grasp_active[:] = delivery_task | payload_relation_active
        skill_probability = getattr(
            hierarchy, "skill_probability", None
        )
        if (
            isinstance(skill_probability, torch.Tensor)
            and skill_probability.shape[1] == ACTION_LAYOUT.skill_dim
        ):
            interaction_probability = skill_probability.reshape(
                -1, 4, 3
            ).sum(dim=1)
            secure_probability = interaction_probability[:, 1]
        else:
            secure_probability = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=selected_object_pos.dtype,
            )
        selected_contact_memory = self.object_contact_memory[
            rows, effective_object
        ]
        capture_lift_enter = (
            delivery_task
            & (~carrying)
            & (selected_contact_memory >= 0.45)
            & (secure_probability >= 0.45)
        )
        capture_lift_retain = (
            self.capture_lift_active
            & (self.capture_option_object == effective_object)
            & delivery_task
            & (~carrying)
            & (selected_contact_memory >= 0.08)
        )
        self.capture_lift_active[:] = (
            capture_lift_enter | capture_lift_retain
        )
        lift_offset = torch.tensor(
            (0.0, 0.0, float(self.cfg.capture_lift_offset)),
            device=self.device,
            dtype=selected_object_pos.dtype,
        )
        acquisition_anchor = torch.where(
            self.capture_lift_active.unsqueeze(1),
            selected_object_pos + lift_offset,
            selected_object_pos,
        )
        desired_anchor = torch.where(
            carrying.unsqueeze(1),
            hover_pos[rows, effective_object] + selected_tcp_offset,
            acquisition_anchor,
        )
        safe_anchor = torch.where(
            carrying.unsqueeze(1),
            carry_anchor_w,
            torch.where(
                self.capture_lift_active.unsqueeze(1),
                selected_object_pos + lift_offset,
                default_goal,
            ),
        )
        interaction_distance = torch.norm(
            desired_anchor - self.center_w, dim=-1
        )
        reachability = torch.sigmoid(
            12.0
            * (
                float(self.cfg.interaction_workspace_radius)
                - interaction_distance
            )
        )
        payload_offset_xy = (
            selected_object_pos[:, :2] - self.center_w[:, :2]
        )
        payload_radius = torch.norm(
            payload_offset_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-5)
        learned_carry_radius = (
            0.26
            + 0.12
            * (0.5 * (skill_parameter[:, 5] + 1.0)).clamp(0.0, 1.0)
        )
        projected_offset_xy = payload_offset_xy * torch.minimum(
            torch.ones_like(payload_radius),
            learned_carry_radius.unsqueeze(1) / payload_radius,
        )
        carry_posture_anchor = torch.cat(
            (
                self.center_w[:, :2] + projected_offset_xy,
                selected_object_pos[:, 2:3],
            ),
            dim=-1,
        ) + selected_tcp_offset
        carry_posture_anchor = torch.where(
            carrying.unsqueeze(1),
            carry_anchor_w,
            carry_posture_anchor,
        )
        relation_violation = (
            (
                payload_radius.squeeze(1) - learned_carry_radius
            )
            / 0.12
        ).clamp(0.0, 1.0)
        self.payload_relation_violation[:] = (
            carrying.float() * relation_violation
        )
        posture_authority = (
            carrying.float()
            * torch.maximum(
                self.control_recovery_pressure,
                torch.maximum(
                    relation_violation,
                    0.35 * self.control_recovery_active.float(),
                ),
            )
        ).clamp(0.0, 1.0)
        projected_safe_anchor = torch.lerp(
            safe_anchor,
            carry_posture_anchor,
            posture_authority.unsqueeze(1),
        )
        self.payload_posture_projection[:] = (
            posture_authority
            * torch.norm(projected_safe_anchor - safe_anchor, dim=-1)
        )
        safe_anchor = projected_safe_anchor
        same_capture_object = (
            self.capture_option_object == effective_object
        )
        capture_enter = (
            delivery_task
            & (~carrying)
            & capture_admissible
            & (reachability >= 0.45)
        )
        capture_retain = (
            self.capture_option_active
            & same_capture_object
            & delivery_task
            & (~carrying)
            & (reachability >= 0.02)
        )
        self.capture_option_active[:] = capture_enter | capture_retain
        self.capture_option_object[:] = torch.where(
            capture_enter,
            effective_object,
            self.capture_option_object,
        )
        handoff = self.capture_option_active.float()
        self.workspace_handoff[:] = handoff
        self._command *= (1.0 - handoff).unsqueeze(1)
        margin_authority = (
            (
                torch.minimum(self.safety_margin, self.preview_margin)
                - 0.02
            )
            / 0.18
        ).clamp(0.0, 1.0)
        tilt_authority = (
            (
                0.90 * float(self.cfg.tilt_limit) - self.base_tilt
            )
            / max(0.15, 0.50 * float(self.cfg.tilt_limit))
        ).clamp(0.0, 1.0)
        support_authority = (
            (self.support_count - float(self.cfg.support_min))
            / max(1.0, 4.0 - float(self.cfg.support_min))
        ).clamp(0.0, 1.0)
        stability_authority = torch.minimum(
            margin_authority,
            torch.minimum(tilt_authority, support_authority),
        )
        # Reaching and gripper closure are separate skill effects.  Coupling
        # them made an open pre-grasp hand retreat from the object.
        # Reachability is a soft classifier, not a Cartesian interpolation
        # coefficient.  Saturate it inside the calibrated workspace while
        # retaining a dead band for genuinely unreachable anchors.
        reach_authority = (
            (reachability - 0.05) / 0.35
        ).clamp(0.0, 1.0)
        reach_authority = reach_authority.square() * (
            3.0 - 2.0 * reach_authority
        )
        reach_authority = torch.where(
            self.capture_option_active,
            torch.ones_like(reach_authority),
            reach_authority,
        )
        interaction_gate = reach_authority * torch.sqrt(
            stability_authority.clamp_min(0.0)
        )
        interaction_anchor = torch.lerp(
            safe_anchor,
            desired_anchor,
            interaction_gate.unsqueeze(1),
        )
        anchor = torch.where(
            (delivery_task | payload_relation_active).unsqueeze(1),
            interaction_anchor,
            default_goal,
        )
        learned_skill_offset = torch.stack(
            (
                0.10 * skill_parameter[:, 3],
                0.10 * skill_parameter[:, 4],
                0.13 * skill_parameter[:, 5],
            ),
            dim=-1,
        )
        task_interaction_authority = torch.maximum(
            manipulation_task.float(),
            delivery_task.float() * reachability,
        )
        payload_skill_authority = payload_relation_active.float()
        semantic_interaction_authority = torch.maximum(
            task_interaction_authority, payload_skill_authority
        )
        interaction_authority = (
            semantic_interaction_authority * stability_authority
        )
        self.interaction_authority[:] = interaction_authority
        task_interaction_offset = task_subgoal[:, 3:6].clone()
        task_interaction_offset[:, 0] = torch.where(
            delivery_task,
            torch.zeros_like(task_interaction_offset[:, 0]),
            task_interaction_offset[:, 0],
        )
        offset_local = (
            task_interaction_authority.unsqueeze(1)
            * stability_authority.unsqueeze(1)
            * task_interaction_offset
            * self._ee_subgoal_scale
            + interaction_authority.unsqueeze(1)
            * learned_skill_offset
        )
        offset_world = _quat_apply(yaw_q, offset_local)
        desired_goal = anchor + offset_world
        object_quat = self._object_quaternions()[rows, effective_object]
        capture_reference = torch.where(
            self.capture_lift_active.unsqueeze(1),
            selected_object_pos + lift_offset,
            selected_object_pos,
        )
        candidate_object_offset = _quat_rotate_inverse(
            object_quat, desired_goal - capture_reference
        )
        object_descriptor = self._object_physical_descriptors[
            effective_object
        ]
        grasp_width = object_descriptor[:, 0]
        object_height = object_descriptor[:, 1]
        grasp_bounds = torch.stack(
            (
                0.006 + 0.044 * (1.0 - secure_probability),
                0.06 * grasp_width
                + 0.006 * (1.0 - secure_probability),
                0.08 * object_height
                + 0.012 * (1.0 - secure_probability),
            ),
            dim=1,
        )
        projected_object_offset = torch.maximum(
            torch.minimum(candidate_object_offset, grasp_bounds),
            -grasp_bounds,
        )
        projected_grasp_center = capture_reference + _quat_apply(
            object_quat, projected_object_offset
        )
        finger_pos = self._robot.data.body_pos_w[:, self._finger_body_ids]
        gripper_center = finger_pos.mean(dim=1)
        tcp_to_gripper_center = gripper_center - self.tcp_pos_w
        projected_tcp_goal = (
            projected_grasp_center - tcp_to_gripper_center
        )
        projected_tcp_goal = torch.where(
            self._ee_goal_initialized.unsqueeze(1),
            projected_tcp_goal,
            projected_grasp_center,
        )
        projection_active = (
            delivery_task
            & (~carrying)
            & self.capture_option_active
        )
        self.grasp_projection_distance[:] = torch.where(
            projection_active,
            torch.norm(
                projected_object_offset - candidate_object_offset,
                dim=-1,
            ),
            torch.zeros_like(secure_probability),
        )
        self.grasp_center_error[:] = torch.where(
            projection_active,
            torch.norm(
                projected_grasp_center - gripper_center, dim=-1
            ),
            torch.zeros_like(secure_probability),
        )
        desired_goal = torch.where(
            projection_active.unsqueeze(1),
            projected_tcp_goal,
            desired_goal,
        )
        desired_goal = torch.where(
            settling.unsqueeze(1), default_goal, desired_goal
        )
        delta_goal = desired_goal - self.curr_goal_pos_w
        delta_norm = torch.norm(delta_goal, dim=-1, keepdim=True)
        max_step = float(self.cfg.max_ee_goal_step_m)
        limited_delta = delta_goal * (
            max_step / delta_norm.clamp_min(max_step)
        )
        self.curr_goal_pos_w[:] = torch.where(
            (~self._ee_goal_initialized).unsqueeze(1),
            reset_goal,
            self.curr_goal_pos_w + limited_delta,
        )
        self._ee_goal_initialized.fill_(True)

        tool_direction = _normalize(self.curr_goal_pos_w - self.center_w)
        q_align = _quat_from_tool_z(tool_direction)
        zero = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=tool_direction.dtype,
        )
        q_spin = _quat_from_euler_xyz(
            zero, zero, torch.full_like(zero, -math.pi)
        )
        goal_quat = _normalize(_quat_mul(q_align, q_spin))

        object_short_axis = torch.tensor(
            (0.0, 1.0, 0.0),
            device=self.device,
            dtype=goal_quat.dtype,
        ).view(1, 3).expand(self.num_envs, -1)
        desired_aperture_axis = _quat_apply(
            object_quat, object_short_axis
        )
        actual_aperture_axis = _normalize(
            finger_pos[:, 0] - finger_pos[:, 1]
        )
        link_quat = self._robot.data.body_quat_w[:, self._ee_body_id]
        aperture_axis_local = _quat_rotate_inverse(
            link_quat, actual_aperture_axis
        )
        predicted_aperture_axis = _quat_apply(
            goal_quat, aperture_axis_local
        )

        actual_projection = actual_aperture_axis - torch.sum(
            actual_aperture_axis * tool_direction,
            dim=-1,
            keepdim=True,
        ) * tool_direction
        predicted_projection = predicted_aperture_axis - torch.sum(
            predicted_aperture_axis * tool_direction,
            dim=-1,
            keepdim=True,
        ) * tool_direction
        desired_projection = desired_aperture_axis - torch.sum(
            desired_aperture_axis * tool_direction,
            dim=-1,
            keepdim=True,
        ) * tool_direction
        actual_projection = _normalize(actual_projection)
        predicted_projection = _normalize(predicted_projection)
        desired_projection = _normalize(desired_projection)
        desired_for_actual = torch.where(
            (
                torch.sum(
                    actual_projection * desired_projection, dim=-1
                )
                < 0.0
            ).unsqueeze(1),
            -desired_projection,
            desired_projection,
        )
        desired_for_goal = torch.where(
            (
                torch.sum(
                    predicted_projection * desired_projection, dim=-1
                )
                < 0.0
            ).unsqueeze(1),
            -desired_projection,
            desired_projection,
        )
        actual_sin = torch.sum(
            tool_direction
            * torch.cross(
                actual_projection, desired_for_actual, dim=-1
            ),
            dim=-1,
        )
        actual_cos = torch.sum(
            actual_projection * desired_for_actual, dim=-1
        ).clamp(-1.0, 1.0)
        roll_sin = torch.sum(
            tool_direction
            * torch.cross(
                predicted_projection, desired_for_goal, dim=-1
            ),
            dim=-1,
        )
        roll_cos = torch.sum(
            predicted_projection * desired_for_goal, dim=-1
        ).clamp(-1.0, 1.0)
        orientation_error = torch.atan2(actual_sin, actual_cos).abs()
        roll_correction = torch.atan2(roll_sin, roll_cos).clamp(
            -float(self.cfg.grasp_orientation_roll_limit),
            float(self.cfg.grasp_orientation_roll_limit),
        )
        round_object = effective_object == 3
        orientation_error = torch.where(
            round_object,
            torch.zeros_like(orientation_error),
            orientation_error,
        )
        roll_correction = torch.where(
            round_object,
            torch.zeros_like(roll_correction),
            roll_correction,
        )
        orientation_active = (
            delivery_task
            & (~carrying)
            & (
                interaction_distance
                <= float(self.cfg.grasp_orientation_activation_radius)
            )
            & (
                self.object_contact_memory[rows, effective_object]
                < 0.45
            )
            & (secure_probability >= 0.45)
            & (stability_authority > 0.25)
            & (~settling)
        )
        half_correction = 0.5 * roll_correction
        correction_quat = torch.cat(
            (
                torch.cos(half_correction).unsqueeze(1),
                tool_direction
                * torch.sin(half_correction).unsqueeze(1),
            ),
            dim=1,
        )
        conditioned_goal_quat = _normalize(
            _quat_mul(correction_quat, goal_quat)
        )
        self.curr_goal_quat_w[:] = torch.where(
            orientation_active.unsqueeze(1),
            conditioned_goal_quat,
            goal_quat,
        )
        self.grasp_orientation_active[:] = orientation_active
        self.grasp_orientation_error[:] = torch.where(
            orientation_active,
            orientation_error,
            torch.zeros_like(orientation_error),
        )
        self.ee_command[:] = _quat_rotate_inverse(
            yaw_q, self.curr_goal_pos_w - self.center_w
        )

        link_pos = self._robot.data.body_pos_w[:, self._ee_body_id]
        link_quat = self._robot.data.body_quat_w[:, self._ee_body_id]
        self.tcp_pos_w[:] = link_pos + _quat_apply(
            link_quat, self._tcp_offset.expand(self.num_envs, -1)
        )
        body_velocity = getattr(self._robot.data, "body_lin_vel_w", None)
        if isinstance(body_velocity, torch.Tensor):
            self.tcp_speed[:] = torch.norm(
                body_velocity[:, self._ee_body_id], dim=-1
            )
        else:
            self.tcp_speed.zero_()
        self.ee_error[:] = torch.norm(
            self.tcp_pos_w - self.curr_goal_pos_w, dim=-1
        )
        step_dt = float(getattr(self._env, "step_dt", 1.0 / 30.0))
        self.mission_age.add_(step_dt)

    def _update_context(self):
        hierarchy = self._hierarchy()
        task_id, skill_id, object_id = self._selected_ids()
        rows = torch.arange(self.num_envs, device=self.device)
        effective_object = self._effective_object_ids(task_id, object_id)
        selected_object_contact = self.object_contact[
            rows, effective_object
        ]
        selected_object_lift = self.object_lift_memory[
            rows, effective_object
        ]
        selected_object_transport = self.object_transport_memory[
            rows, effective_object
        ]
        selected_object_place = self.object_place[
            rows, effective_object
        ]
        selected_interaction_frontier = self.object_interaction_frontier[
            rows, effective_object
        ]
        selected_interaction_frontier_delta = (
            self.object_interaction_frontier_delta[
                rows, effective_object
            ]
        )
        self.selected_interaction_frontier[:] = (
            selected_interaction_frontier
        )
        self.selected_interaction_frontier_delta[:] = (
            selected_interaction_frontier_delta
        )
        next_selected_progress = self.task_progress[rows, task_id]
        same_task = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if hierarchy is not None:
            same_task = hierarchy.task_switch < 0.5
        self.selected_task_progress_delta[:] = torch.where(
            same_task & self._progress_initialized,
            next_selected_progress - self.previous_selected_task_progress,
            torch.zeros_like(next_selected_progress),
        ).clamp(-0.25, 0.25)
        self.selected_task_progress[:] = next_selected_progress
        self.previous_selected_task_progress[:] = next_selected_progress
        self.mission_completion_delta[:] = (
            self.mission_completion - self.previous_mission_completion
        ).clamp(0.0, 1.0)
        self.previous_mission_completion[:] = self.mission_completion
        self.mission_success_event[:] = (
            self.mission_success - self.previous_mission_success
        ).clamp(0.0, 1.0)
        self.previous_mission_success[:] = self.mission_success
        self._progress_initialized.fill_(True)

        global_context = self.hierarchy_context[:, :GLOBAL_CONTEXT_DIM]
        global_context.zero_()
        global_context[:, 0] = self.mission_completion
        if hierarchy is not None:
            global_context[:, 1] = (
                hierarchy.task_age
                / max(1.0e-4, float(hierarchy.cfg.task_max_dwell_s))
            ).clamp(0.0, 1.0)
            global_context[:, 2] = (
                hierarchy.skill_age
                / max(1.0e-4, float(hierarchy.cfg.skill_max_dwell_s))
            ).clamp(0.0, 1.0)
        global_context[:, 3] = (self.base_height / 0.65).clamp(0.0, 1.5)
        global_context[:, 4] = (self.base_tilt / 0.60).clamp(0.0, 1.5)
        global_context[:, 5] = self.support_count / 4.0
        global_context[:, 6] = (self.obstacle_margin / 1.2).clamp(0.0, 2.0)
        global_context[:, 7] = self._robot.data.root_lin_vel_b[:, 0].clamp(-2.0, 2.0)
        global_context[:, 8] = self._robot.data.root_ang_vel_b[:, 2].clamp(-2.0, 2.0)
        global_context[:, 9] = self._command[:, 0]
        global_context[:, 10] = self._command[:, 2]
        global_context[:, 11] = (self.task_error / 4.0).clamp(0.0, 2.0)
        global_context[:, 12] = self.safety_margin
        global_context[:, 13] = self.preview_margin
        global_context[:, 14] = self.clf_decrease_score
        global_context[:, 15] = self.disturbance_quality
        global_context[:, 16] = task_id.float() / max(
            1, ACTION_LAYOUT.task_dim - 1
        )
        global_context[:, 17] = skill_id.float() / max(
            1, ACTION_LAYOUT.skill_dim - 1
        )
        global_context[:, 18] = effective_object.float() / max(
            1, ACTION_LAYOUT.object_dim - 1
        )
        global_context[:, 19] = selected_object_contact
        global_context[:, 20] = selected_object_lift
        global_context[:, 21] = selected_object_transport
        global_context[:, 22] = selected_object_place
        global_context[:, 23] = self.gripper_closure
        global_context[:, 24:30] = self.object_completion
        if hierarchy is not None:
            global_context[:, 30] = (
                hierarchy.task_hazard
                / max(
                    1.0e-4,
                    float(hierarchy.cfg.task_termination_budget),
                )
            ).clamp(0.0, 1.0)
            global_context[:, 31] = (
                hierarchy.skill_hazard
                / max(
                    1.0e-4,
                    float(hierarchy.cfg.skill_termination_budget),
                )
            ).clamp(0.0, 1.0)
        global_context[:, 32] = self.selected_task_progress
        global_context[:, 33] = (
            4.0 * self.selected_task_progress_delta
        ).clamp(-1.0, 1.0)
        global_context[:, 34] = self.curriculum_level.float() / 4.0
        global_context[:, 35] = self.task_required.sum(dim=1) / float(
            TASK_SLOT_COUNT
        )
        root = self._robot.data.root_pos_w
        yaw_q = _quat_from_yaw(self._robot.data.heading_w)
        tcp_b = _quat_rotate_inverse(yaw_q, self.tcp_pos_w - root)
        global_context[:, 36:39] = tcp_b.clamp(-2.0, 2.0)
        global_context[:, 39] = (self.tcp_speed / 2.0).clamp(0.0, 2.0)
        global_context[:, 40] = (self.ee_error / 0.75).clamp(0.0, 2.0)
        selected_slot = self.task_slots[rows, task_id]
        global_context[:, 41] = selected_slot[:, 27]
        required_count = self.task_required.sum(dim=1).clamp_min(1.0)
        global_context[:, 42] = (
            required_count
            - (self.task_completed * self.task_required).sum(dim=1)
        ) / required_count
        global_context[:, 43] = float(
            self.cfg.interaction_workspace_radius
        )
        global_context[:, 44] = float(self.cfg.gripper_open_gap) / 0.10
        global_context[:, 45] = float(self.cfg.wheel_radius) / 0.15
        global_context[:, 46] = self.object_interaction_frontier.mean(dim=1)
        max_episode_length = max(
            1, int(getattr(self._env, "max_episode_length", 1))
        )
        global_context[:, 47] = (
            1.0
            - self._env.episode_length_buf.float()
            / float(max_episode_length)
        ).clamp(0.0, 1.0)
        global_context[:, 48] = self._command[:, 1]

        self.hierarchy_context[:, GLOBAL_CONTEXT_DIM:] = self.task_slots.reshape(
            self.num_envs, -1
        )

    def _update_packets(self):
        task_id, skill_id, object_id = self._selected_ids()
        effective_object = self._effective_object_ids(task_id, object_id)
        self.tau_down_packet.zero_()
        self.tau_down_packet[:, 0:3] = self._command
        self.tau_down_packet[:, 3:6] = self.ee_command
        self.tau_down_packet[:, 6] = task_id.float() / 11.0
        self.tau_down_packet[:, 7] = skill_id.float() / 11.0
        self.tau_down_packet[:, 8] = effective_object.float() / 5.0
        self.tau_down_packet[:, 9] = self.mission_completion
        self.tau_down_packet[:, 10] = self.selected_task_progress
        self.tau_down_packet[:, 11] = self.safety_margin
        self.tau_down_packet[:, 12] = self.preview_margin
        self.tau_down_packet[:, 13] = self.clf_decrease_score
        self.tau_down_packet[:, 14] = self.disturbance_quality
        self.tau_down_packet[:, 15:27] = self.task_completed
        self.tau_down_packet[:, 27:39] = self.task_progress
        self.tau_down_packet[:, 39:45] = self.object_contact_memory
        self.tau_down_packet[:, 45:51] = self.object_lift_memory
        self.tau_down_packet[:, 51:57] = self.object_transport_memory
        self.tau_down_packet[:, 57:63] = self.object_place
        self.tau_down_packet[:, 63] = self.gripper_closure
        self.tau_down_packet[:, 64] = self.base_height
        self.tau_down_packet[:, 65] = self.base_tilt
        self.tau_down_packet[:, 66] = self.support_count / 4.0
        self.tau_down_packet[:, 67] = self.obstacle_margin
        self.tau_down_packet[:, 68:77] = self.hierarchy_context[:, :9]

        self.tau_up_packet.zero_()
        self.tau_up_packet[:, 0] = self.mission_completion
        self.tau_up_packet[:, 1] = self.selected_task_progress
        self.tau_up_packet[:, 2] = self.task_error
        self.tau_up_packet[:, 3] = self.ee_error
        self.tau_up_packet[:, 4] = self.base_tilt
        self.tau_up_packet[:, 5] = self.support_count
        self.tau_up_packet[:, 6] = self.obstacle_margin
        self.tau_up_packet[:, 7] = self.safety_margin
        self.tau_up_packet[:, 8] = self.preview_margin
        self.tau_up_packet[:, 9] = self.clf_decrease
        self.tau_up_packet[:, 10] = self.clf_decrease_score
        self.tau_up_packet[:, 11] = self.disturbance_estimate
        self.tau_up_packet[:, 12] = self.disturbance_quality
        self.tau_up_packet[:, 13] = self.mission_success
        self.tau_up_packet[:, 14:20] = self.object_completion
        self.tau_up_packet[:, 20:26] = self.object_contact_memory
        self.tau_up_packet[:, 26:32] = self.object_lift_memory
        self.tau_up_packet[:, 32:38] = self.object_transport_memory
        self.tau_up_packet[:, 38:44] = self.object_place
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            self.tau_up_packet[:, 44] = hierarchy.task_switch
            self.tau_up_packet[:, 45] = hierarchy.skill_switch
            self.tau_up_packet[:, 46:48] = hierarchy.termination_probability

    def _update_command(self):
        self._base_measurements()
        self._update_object_events()
        self._update_control_recovery_admissibility()
        targets, headings = self._task_targets()
        self._update_learned_commands(targets, headings)
        self._update_control_signals(targets, headings)
        safe_step = (
            (self.base_tilt < 0.90 * float(self.cfg.tilt_limit))
            & (self.support_count >= 2.5)
            & (self.safety_margin > 0.02)
        )
        self.episode_safe_sum.add_(safe_step.float())
        self.episode_evidence_steps.add_(1.0)
        self._update_task_ledger(targets, headings)
        self._update_task_slots(targets, headings)
        self._update_context()
        self._update_packets()

    def _update_metrics(self):
        hierarchy = self._hierarchy()
        self.metrics["TACTIC/mission_completion"] = self.mission_completion
        self.metrics["TACTIC/mission_completion_delta"] = (
            self.mission_completion_delta
        )
        self.metrics["TACTIC/mission_success"] = self.mission_success
        self.metrics["TACTIC/task_progress"] = self.selected_task_progress
        self.metrics["TACTIC/task_progress_delta"] = (
            self.selected_task_progress_delta
        )
        self.metrics["TACTIC/task_error"] = self.task_error
        self.metrics["TACTIC/ee_error"] = self.ee_error
        hierarchy = self._hierarchy()
        if hierarchy is not None:
            rows = torch.arange(self.num_envs, device=self.device)
            selected_object = self._effective_object_ids(
                hierarchy.task_id, hierarchy.object_id
            )
            object_pos = self._object_positions()[rows, selected_object]
            target_pos = self._target_positions()[rows, selected_object]
            finger_pos = self._robot.data.body_pos_w[
                :, self._finger_body_ids
            ]
            self.metrics["TACTIC/tcp_object_distance"] = torch.norm(
                self.tcp_pos_w - object_pos, dim=-1
            )
            self.metrics["TACTIC/goal_object_distance"] = torch.norm(
                self.curr_goal_pos_w - object_pos, dim=-1
            )
            self.metrics["TACTIC/finger_object_distance"] = torch.maximum(
                torch.norm(finger_pos[:, 0] - object_pos, dim=-1),
                torch.norm(finger_pos[:, 1] - object_pos, dim=-1),
            )
            self.metrics["TACTIC/object_target_distance"] = torch.norm(
                object_pos - target_pos, dim=-1
            )
            self.metrics["TACTIC/recovery_task_executed"] = (
                hierarchy.task_id == 4
            ).float()
            self.metrics["TACTIC/recovery_task_probability"] = (
                hierarchy.task_probability[:, 4]
            )
            self.metrics["TACTIC/executed_task_probability"] = (
                hierarchy.task_probability.gather(
                    1, hierarchy.task_id.unsqueeze(1)
                ).squeeze(1)
            )
            self.metrics["TACTIC/task_constraint_projection"] = (
                hierarchy.task_constraint_projection
            )
            self.metrics["TACTIC/recovery_latch_seen"] = (
                hierarchy.recovery_latch_seen
            )
            self.metrics["TACTIC/recovery_valid_seen"] = (
                hierarchy.recovery_valid_seen
            )
            self.metrics["TACTIC/recovery_pressure_seen"] = (
                hierarchy.recovery_pressure_seen
            )
        self.metrics["TACTIC/cbf_margin"] = self.safety_margin
        self.metrics["TACTIC/predicted_margin"] = self.preview_margin
        self.metrics["TACTIC/clf_decrease"] = self.clf_decrease
        self.metrics["TACTIC/disturbance_estimate"] = self.disturbance_estimate
        self.metrics["TACTIC/base_height"] = self.base_height
        self.metrics["TACTIC/base_tilt"] = self.base_tilt
        self.metrics["TACTIC/control_recovery_pressure"] = (
            self.control_recovery_pressure
        )
        self.metrics["TACTIC/control_recovery_active"] = (
            self.control_recovery_active.float()
        )
        self.metrics["TACTIC/control_recovery_constraint_active"] = (
            self.control_recovery_constraint_active.float()
        )
        self.metrics["TACTIC/payload_posture_projection"] = (
            self.payload_posture_projection
        )
        self.metrics["TACTIC/payload_relation_violation"] = (
            self.payload_relation_violation
        )
        self.metrics["TACTIC/base_vx"] = (
            self._robot.data.root_lin_vel_b[:, 0]
        )
        self.metrics["TACTIC/base_vy"] = (
            self._robot.data.root_lin_vel_b[:, 1]
        )
        self.metrics["TACTIC/base_wz"] = (
            self._robot.data.root_ang_vel_b[:, 2]
        )
        self.metrics["TACTIC/support_count"] = self.support_count
        self.metrics["TACTIC/obstacle_margin"] = self.obstacle_margin
        self.metrics["TACTIC/gripper_closure"] = self.gripper_closure
        self.metrics["TACTIC/manipulation_active"] = (
            self.manipulation_active.float()
        )
        self.metrics["TACTIC/grasp_active"] = self.grasp_active.float()
        self.metrics["TACTIC/grasp_orientation_active"] = (
            self.grasp_orientation_active.float()
        )
        self.metrics["TACTIC/grasp_orientation_error"] = (
            self.grasp_orientation_error
        )
        self.metrics["TACTIC/interaction_authority"] = (
            self.interaction_authority
        )
        self.metrics["TACTIC/workspace_handoff"] = self.workspace_handoff
        self.metrics["TACTIC/grasp_projection_distance"] = (
            self.grasp_projection_distance
        )
        self.metrics["TACTIC/grasp_center_error"] = self.grasp_center_error
        self.metrics["TACTIC/capture_staging_distance"] = (
            self.capture_staging_distance
        )
        self.metrics["TACTIC/capture_heading_error"] = (
            self.capture_heading_error
        )
        self.metrics["TACTIC/capture_initiation_margin"] = (
            self.capture_initiation_margin
        )
        self.metrics["TACTIC/selected_delivery_side"] = (
            self.selected_delivery_side
        )
        self.metrics["TACTIC/capture_lift_active"] = (
            self.capture_lift_active.float()
        )
        self.metrics["TACTIC/command_vx"] = self._command[:, 0]
        self.metrics["TACTIC/command_vy"] = self._command[:, 1]
        self.metrics["TACTIC/command_wz"] = self._command[:, 2]
        self.metrics["TACTIC/curriculum_level"] = (
            self.curriculum_level.float()
        )
        self.metrics["TACTIC/curriculum_frontier"] = torch.full_like(
            self.task_error,
            float(self._curriculum_unlocked_level),
        )
        self.metrics["TACTIC/curriculum_probe"] = self.curriculum_probe
        self.metrics["TACTIC/interaction_start_curriculum"] = (
            self.interaction_start_curriculum
        )
        self.metrics["TACTIC/interaction_start_probability"] = (
            torch.full_like(
                self.task_error,
                self._interaction_start_probability,
            )
        )
        self.metrics["TACTIC/composition_probe_probability"] = (
            torch.full_like(
                self.task_error,
                self._composition_probe_probability,
            )
        )
        self.metrics["TACTIC/curriculum_completion_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_completion_ema
        )
        self.metrics["TACTIC/curriculum_progress_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_progress_ema
        )
        self.metrics["TACTIC/curriculum_safety_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_safety_ema
        )
        self.metrics["TACTIC/episode_safety_fraction"] = (
            self.episode_safe_sum
            / self.episode_evidence_steps.clamp_min(1.0)
        )
        self.metrics["TACTIC/curriculum_contact_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_contact_ema
        )
        self.metrics["TACTIC/curriculum_lift_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_lift_ema
        )
        self.metrics["TACTIC/curriculum_transport_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_transport_ema
        )
        self.metrics["TACTIC/curriculum_place_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_place_ema
        )
        self.metrics["TACTIC/curriculum_delivery_completion_ema"] = (
            torch.ones_like(self.task_error)
            * self._curriculum_delivery_completion_ema
        )
        self.metrics["TACTIC/curriculum_regression_streak"] = (
            torch.full_like(
                self.task_error,
                float(self._curriculum_regression_streak),
            )
        )
        self.metrics["TACTIC/required_task_count"] = self.task_required.sum(
            dim=1
        )
        self.metrics["TACTIC/object_completion_mean"] = (
            self.object_completion.mean(dim=1)
        )
        self.metrics["TACTIC/object_contact_mean"] = (
            self.object_contact.mean(dim=1)
        )
        self.metrics["TACTIC/object_lift_mean"] = self.object_lift_memory.mean(
            dim=1
        )
        self.metrics["TACTIC/object_transport_mean"] = (
            self.object_transport_memory.mean(dim=1)
        )
        self.metrics["TACTIC/object_place_mean"] = self.object_place.mean(dim=1)
        self.metrics["TACTIC/object_target_progress_mean"] = (
            self.object_target_progress.mean(dim=1)
        )
        self.metrics["TACTIC/object_target_progress_delta_mean"] = (
            self.object_target_progress_delta.mean(dim=1)
        )
        self.metrics["TACTIC/object_release_readiness_mean"] = (
            self.object_release_readiness.mean(dim=1)
        )
        self.metrics[
            "TACTIC/object_release_readiness_delta_mean"
        ] = self.object_release_readiness_delta.mean(dim=1)
        self.metrics["TACTIC/object_release_event_mean"] = (
            self.object_release_event.mean(dim=1)
        )
        self.metrics["TACTIC/object_drop_event_mean"] = (
            self.object_drop_event.mean(dim=1)
        )
        self.metrics["TACTIC/object_gripper_distance_mean"] = (
            self.object_gripper_distance.mean(dim=1)
        )
        self.metrics["TACTIC/object_carrying_mean"] = (
            self.object_carrying.float().mean(dim=1)
        )
        self.metrics["TACTIC/object_carry_memory_mean"] = (
            self.object_carry_memory.mean(dim=1)
        )
        self.metrics["TACTIC/interaction_frontier_mean"] = (
            self.object_interaction_frontier.mean(dim=1)
        )
        self.metrics["TACTIC/interaction_frontier_delta_mean"] = (
            self.object_interaction_frontier_delta.mean(dim=1)
        )
        self.metrics["TACTIC/contact_symmetry_mean"] = (
            self.object_contact_symmetry.mean(dim=1)
        )
        if hierarchy is not None:
            rows = torch.arange(self.num_envs, device=self.device)
            selected_object = self._effective_object_ids(
                hierarchy.task_id, hierarchy.object_id
            )
            self.metrics["TACTIC/task_id"] = hierarchy.task_id.float()
            self.metrics["TACTIC/skill_id"] = hierarchy.skill_id.float()
            self.metrics["TACTIC/object_id"] = hierarchy.object_id.float()
            self.metrics["TACTIC/effective_object_id"] = (
                selected_object.float()
            )
            self.metrics["TACTIC/selected_object_contact"] = (
                self.object_contact[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_lift"] = (
                self.object_lift_memory[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_transport"] = (
                self.object_transport_memory[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_place"] = (
                self.object_place[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_completion"] = (
                self.object_completion[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_carrying"] = (
                self.object_carrying[rows, selected_object].float()
            )
            self.metrics["TACTIC/selected_object_carry_memory"] = (
                self.object_carry_memory[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_target_progress"] = (
                self.object_target_progress[rows, selected_object]
            )
            self.metrics[
                "TACTIC/selected_object_target_progress_delta"
            ] = self.object_target_progress_delta[rows, selected_object]
            self.metrics["TACTIC/selected_object_release_readiness"] = (
                self.object_release_readiness[rows, selected_object]
            )
            self.metrics[
                "TACTIC/selected_object_release_readiness_delta"
            ] = self.object_release_readiness_delta[
                rows, selected_object
            ]
            self.metrics["TACTIC/selected_object_release_event"] = (
                self.object_release_event[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_drop_event"] = (
                self.object_drop_event[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_gripper_distance"] = (
                self.object_gripper_distance[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_place_error_xy"] = (
                self.object_place_error_xy[rows, selected_object]
            )
            self.metrics["TACTIC/selected_object_place_error_z"] = (
                self.object_place_error_z[rows, selected_object]
            )
            self.metrics["TACTIC/selected_interaction_frontier"] = (
                self.object_interaction_frontier[
                    rows, selected_object
                ]
            )
            self.metrics["TACTIC/selected_interaction_frontier_delta"] = (
                self.object_interaction_frontier_delta[
                    rows, selected_object
                ]
            )
            self.metrics["TACTIC/selected_reachability"] = (
                self.task_slots[
                    rows,
                    hierarchy.task_id,
                    TASK_SLOT_REACHABILITY_INDEX,
                ]
            )
            self.metrics["TACTIC/task_switch"] = hierarchy.task_switch
            self.metrics["TACTIC/skill_switch"] = hierarchy.skill_switch
            self.metrics["TACTIC/task_age"] = hierarchy.task_age
            self.metrics["TACTIC/skill_age"] = hierarchy.skill_age
            self.metrics["TACTIC/task_hazard"] = hierarchy.task_hazard
            self.metrics["TACTIC/skill_hazard"] = hierarchy.skill_hazard
            self.metrics["TACTIC/task_termination_probability"] = (
                hierarchy.termination_probability[:, 0]
            )
            self.metrics["TACTIC/skill_termination_probability"] = (
                hierarchy.termination_probability[:, 1]
            )
            task_occupancy = torch.bincount(
                hierarchy.task_id,
                minlength=ACTION_LAYOUT.task_dim,
            ).to(self.task_error.dtype)
            task_occupancy = task_occupancy / task_occupancy.sum().clamp_min(
                1.0
            )
            skill_occupancy = torch.bincount(
                hierarchy.skill_id,
                minlength=ACTION_LAYOUT.skill_dim,
            ).to(self.task_error.dtype)
            skill_occupancy = skill_occupancy / skill_occupancy.sum().clamp_min(
                1.0
            )
            task_entropy = -(
                task_occupancy
                * torch.log(task_occupancy.clamp_min(1.0e-8))
            ).sum() / math.log(float(ACTION_LAYOUT.task_dim))
            skill_entropy = -(
                skill_occupancy
                * torch.log(skill_occupancy.clamp_min(1.0e-8))
            ).sum() / math.log(float(ACTION_LAYOUT.skill_dim))
            self.metrics["TACTIC/task_entropy"] = task_entropy.expand_as(
                self.task_error
            ).clone()
            self.metrics["TACTIC/skill_entropy"] = skill_entropy.expand_as(
                self.task_error
            ).clone()
            self.metrics[
                "TACTIC/task_occupancy_entropy"
            ] = task_entropy.expand_as(self.task_error).clone()
            self.metrics[
                "TACTIC/skill_occupancy_entropy"
            ] = skill_entropy.expand_as(self.task_error).clone()
            for index in range(ACTION_LAYOUT.skill_param_dim):
                self.metrics[
                    f"TACTIC/skill_parameter_{index}"
                ] = hierarchy.skill_parameter[:, index]
        for index in range(ACTION_LAYOUT.object_dim):
            self.metrics[
                f"TACTIC/object_{index}_complete"
            ] = self.object_completion[:, index]


@configclass
class TACTICEeGoalCommandCfg(CommandTermCfg):
    class_type: Optional[type] = None
    source_command_name: str = "locomotion"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = TACTICEeGoalCommand


class TACTICEeGoalCommand(CommandTerm):
    """Adapter for the baseline arm IK action."""

    cfg: TACTICEeGoalCommandCfg

    def __init__(self, cfg: TACTICEeGoalCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.curr_goal_quat_w = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.curr_goal_quat_w[:, 0] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _source(self) -> TACTICMissionCommand:
        return self._env.command_manager.get_term(
            self.cfg.source_command_name
        )

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        source = self._source()
        self._command[:] = source.ee_command
        self.center_w[:] = source.center_w
        self.curr_goal_pos_w[:] = source.curr_goal_pos_w
        self.curr_goal_quat_w[:] = source.curr_goal_quat_w

    def _update_metrics(self):
        return
