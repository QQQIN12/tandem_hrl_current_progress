"""TANDEM-HRL unified mission built on the ZYB-v0 robot model.

The module path is retained for checkpoint compatibility.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ActionsCfg,
    EventCfg,
    ManipLocoEnvCfg,
    ManipLocoSceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from quadruped_arm.tasks.manager_based.maniploco.mdp.observations import (
    VbcPolicyObsTerm,
)
from .tactic_layout import ACTION_LAYOUT
from .mdp.tactic_actions import (
    TACTICHierarchyActionCfg,
    TACTICSymmetricGripperActionCfg,
)
from .mdp.tactic_arm_ik import TANDEMArmIkActionCfg
from .mdp.tactic_command import (
    TACTICEeGoalCommandCfg,
    TACTICMissionCommandCfg,
)
from .mdp import tactic_observations as tactic_obs
from .mdp import tactic_rewards as tactic_rew


OBS_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "FL_foot_wheel_joint",
    "FR_foot_wheel_joint",
    "RL_foot_wheel_joint",
    "RR_foot_wheel_joint",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


OBJECT_PRIM_PATHS = (
    "{ENV_REGEX_NS}/GraspObject",
    "{ENV_REGEX_NS}/GraspObjectTall",
    "{ENV_REGEX_NS}/GraspObjectFlat",
    "{ENV_REGEX_NS}/GraspObjectRound",
    "{ENV_REGEX_NS}/GraspObjectSlim",
    "{ENV_REGEX_NS}/GraspObjectWide",
)


def _static_cuboid(
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    static_friction: float = 1.0,
    dynamic_friction: float = 0.85,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=position, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.72
            ),
        ),
    )


def _rigid_object(
    name: str,
    position: tuple[float, float, float],
    geometry: str,
    size: tuple[float, float, float],
    mass: float,
    dynamic_friction: float,
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    common = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=20,
            solver_velocity_iteration_count=5,
            max_depenetration_velocity=0.75,
            disable_gravity=False,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "mass_props": sim_utils.MassPropertiesCfg(mass=mass),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=dynamic_friction + 0.34,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, roughness=0.48
        ),
    }
    if geometry == "cylinder":
        spawn = sim_utils.CylinderCfg(
            radius=0.5 * size[1],
            height=size[2],
            axis="Z",
            **common,
        )
    else:
        spawn = sim_utils.CuboidCfg(size=size, **common)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=spawn,
    )


def _finger_object_contact(link_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=False,
        track_friction_forces=False,
        force_threshold=0.20,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=list(OBJECT_PRIM_PATHS),
    )


@configclass
class TACTICSceneCfg(ManipLocoSceneCfg):
    """ZYB-v0 scene extended with physical task objects and route geometry."""

    grasp_source_pad_main = _static_cuboid(
        "GraspSourcePadMain",
        (-0.40, 0.00, 0.70),
        (0.16, 0.11, 0.06),
        (0.22, 0.42, 0.72),
        1.24,
        0.96,
    )
    grasp_source_pad_tall = _static_cuboid(
        "GraspSourcePadTall",
        (-0.62, 0.00, 0.70),
        (0.13, 0.11, 0.06),
        (0.70, 0.26, 0.18),
        1.28,
        0.98,
    )
    grasp_source_pad_flat = _static_cuboid(
        "GraspSourcePadFlat",
        (-0.40, -0.18, 0.70),
        (0.18, 0.12, 0.06),
        (0.10, 0.55, 0.62),
        1.30,
        1.00,
    )
    grasp_source_pad_round = _static_cuboid(
        "GraspSourcePadRound",
        (-0.40, 0.18, 0.70),
        (0.12, 0.12, 0.06),
        (0.72, 0.48, 0.14),
        1.22,
        0.94,
    )
    grasp_source_pad_slim = _static_cuboid(
        "GraspSourcePadSlim",
        (-0.62, 0.18, 0.70),
        (0.12, 0.11, 0.06),
        (0.50, 0.36, 0.72),
        1.24,
        0.96,
    )
    grasp_source_pad_wide = _static_cuboid(
        "GraspSourcePadWide",
        (-0.62, -0.18, 0.70),
        (0.20, 0.12, 0.06),
        (0.18, 0.50, 0.46),
        1.30,
        1.00,
    )

    grasp_place_pad = _static_cuboid(
        "GraspTargetPlatform",
        (0.35, 0.35, 0.70),
        (0.22, 0.20, 0.06),
        (0.12, 0.46, 0.22),
        1.10,
        0.88,
    )
    grasp_place_pad_aux_a = _static_cuboid(
        "GraspTargetPlatformAuxA",
        (0.55, -0.30, 0.70),
        (0.22, 0.20, 0.06),
        (0.58, 0.30, 0.10),
        1.10,
        0.88,
    )
    grasp_place_pad_aux_b = _static_cuboid(
        "GraspTargetPlatformAuxB",
        (0.85, 0.15, 0.70),
        (0.22, 0.20, 0.06),
        (0.42, 0.22, 0.64),
        1.10,
        0.88,
    )
    grasp_place_pad_aux_c = _static_cuboid(
        "GraspTargetPlatformAuxC",
        (0.40, -0.55, 0.70),
        (0.22, 0.20, 0.06),
        (0.38, 0.30, 0.68),
        1.10,
        0.88,
    )
    grasp_place_pad_aux_d = _static_cuboid(
        "GraspTargetPlatformAuxD",
        (0.95, -0.40, 0.70),
        (0.22, 0.20, 0.06),
        (0.18, 0.38, 0.62),
        1.10,
        0.88,
    )
    grasp_place_pad_aux_e = _static_cuboid(
        "GraspTargetPlatformAuxE",
        (0.70, 0.55, 0.70),
        (0.22, 0.20, 0.06),
        (0.62, 0.24, 0.44),
        1.10,
        0.88,
    )

    grasp_object = _rigid_object(
        "GraspObject",
        (-0.40, 0.00, 0.771),
        "cuboid",
        (0.092, 0.038, 0.078),
        0.044,
        1.02,
        (0.06, 0.30, 0.78),
    )
    grasp_object_tall = _rigid_object(
        "GraspObjectTall",
        (-0.62, 0.00, 0.790),
        "cuboid",
        (0.056, 0.036, 0.116),
        0.040,
        1.02,
        (0.78, 0.18, 0.10),
    )
    grasp_object_flat = _rigid_object(
        "GraspObjectFlat",
        (-0.40, -0.18, 0.754),
        "cuboid",
        (0.118, 0.038, 0.044),
        0.028,
        1.14,
        (0.10, 0.56, 0.70),
    )
    grasp_object_round = _rigid_object(
        "GraspObjectRound",
        (-0.40, 0.18, 0.775),
        "cylinder",
        (0.036, 0.036, 0.086),
        0.037,
        0.96,
        (0.86, 0.56, 0.12),
    )
    grasp_object_slim = _rigid_object(
        "GraspObjectSlim",
        (-0.62, 0.18, 0.793),
        "cuboid",
        (0.050, 0.034, 0.122),
        0.034,
        1.10,
        (0.46, 0.22, 0.78),
    )
    grasp_object_wide = _rigid_object(
        "GraspObjectWide",
        (-0.62, -0.18, 0.757),
        "cuboid",
        (0.140, 0.040, 0.050),
        0.030,
        1.20,
        (0.12, 0.62, 0.48),
    )

    left_finger_object_contact = _finger_object_contact("link7")
    right_finger_object_contact = _finger_object_contact("link8")

    tactic_obstacle_0 = _static_cuboid(
        "TACTICObstacle0", (1.10, -0.10, 0.28), (0.18, 0.18, 0.56), (0.74, 0.20, 0.18)
    )
    tactic_obstacle_1 = _static_cuboid(
        "TACTICObstacle1", (1.45, 0.42, 0.28), (0.18, 0.18, 0.56), (0.18, 0.48, 0.72)
    )
    tactic_obstacle_2 = _static_cuboid(
        "TACTICObstacle2", (1.82, -0.28, 0.28), (0.18, 0.18, 0.56), (0.76, 0.48, 0.12)
    )
    tactic_obstacle_3 = _static_cuboid(
        "TACTICObstacle3", (2.18, 0.38, 0.28), (0.18, 0.18, 0.56), (0.34, 0.62, 0.26)
    )
    tactic_obstacle_4 = _static_cuboid(
        "TACTICObstacle4", (2.30, -0.58, 0.30), (1.00, 0.12, 0.60), (0.46, 0.48, 0.52)
    )
    tactic_obstacle_5 = _static_cuboid(
        "TACTICObstacle5", (2.30, 0.62, 0.30), (1.00, 0.12, 0.60), (0.46, 0.48, 0.52)
    )
    tactic_obstacle_6 = _static_cuboid(
        "TACTICObstacle6", (0.30, -1.38, 0.22), (0.42, 0.42, 0.44), (0.50, 0.24, 0.62)
    )
    tactic_obstacle_7 = _static_cuboid(
        "TACTICObstacle7", (-0.72, -1.18, 0.20), (0.34, 0.34, 0.40), (0.12, 0.58, 0.60)
    )


def _object_reset(asset_name: str) -> EventTerm:
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "pose_range": {
                "x": (-0.030, 0.030),
                "y": (-0.030, 0.030),
                "yaw": (-0.40, 0.40),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


def _object_material_randomization(asset_name: str) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "static_friction_range": (0.88, 1.72),
            "dynamic_friction_range": (0.68, 1.38),
            "restitution_range": (0.0, 0.03),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )


def _object_mass_randomization(asset_name: str) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "mass_distribution_params": (0.72, 1.38),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 0.016,
        },
    )


@configclass
class TACTICEventCfg(EventCfg):
    """Nominal ZYB-v0 robot reset plus independent object variation."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "yaw": (-0.14, 0.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_left_finger_open = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint7"]),
            "position_range": (0.032, 0.034),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_right_finger_open = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint8"]),
            "position_range": (-0.034, -0.032),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_grasp_object = _object_reset("grasp_object")
    reset_grasp_object_tall = _object_reset("grasp_object_tall")
    reset_grasp_object_flat = _object_reset("grasp_object_flat")
    reset_grasp_object_round = _object_reset("grasp_object_round")
    reset_grasp_object_slim = _object_reset("grasp_object_slim")
    reset_grasp_object_wide = _object_reset("grasp_object_wide")
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.18, 0.18),
                "y": (-0.14, 0.14),
                "yaw": (-0.25, 0.25),
            },
        },
    )


@configclass
class TACTICPlayEventCfg(TACTICEventCfg):
    push = None


@configclass
class TACTICStressEventCfg(TACTICEventCfg):
    """Held-out dynamics and impulse variation for robustness evaluation."""

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["base"]
            ),
            "mass_distribution_params": (0.90, 1.12),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 1.0,
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_thigh_.*",
                    ".*_calf_.*",
                    "joint[1-6]",
                ],
            ),
            "stiffness_distribution_params": (0.92, 1.10),
            "damping_distribution_params": (0.88, 1.14),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_object_0_material = _object_material_randomization(
        "grasp_object"
    )
    randomize_object_1_material = _object_material_randomization(
        "grasp_object_tall"
    )
    randomize_object_2_material = _object_material_randomization(
        "grasp_object_flat"
    )
    randomize_object_3_material = _object_material_randomization(
        "grasp_object_round"
    )
    randomize_object_4_material = _object_material_randomization(
        "grasp_object_slim"
    )
    randomize_object_5_material = _object_material_randomization(
        "grasp_object_wide"
    )
    randomize_object_0_mass = _object_mass_randomization("grasp_object")
    randomize_object_1_mass = _object_mass_randomization(
        "grasp_object_tall"
    )
    randomize_object_2_mass = _object_mass_randomization(
        "grasp_object_flat"
    )
    randomize_object_3_mass = _object_mass_randomization(
        "grasp_object_round"
    )
    randomize_object_4_mass = _object_mass_randomization(
        "grasp_object_slim"
    )
    randomize_object_5_mass = _object_mass_randomization(
        "grasp_object_wide"
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.5, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.35, 0.35),
                "y": (-0.28, 0.28),
                "yaw": (-0.45, 0.45),
            },
        },
    )


@configclass
class TACTICCommandsCfg:
    locomotion = TACTICMissionCommandCfg()
    ee_goal = TACTICEeGoalCommandCfg()


@configclass
class TACTICTrainingCommandsCfg(TACTICCommandsCfg):
    locomotion = TACTICMissionCommandCfg(
        delivery_replay_fraction=0.25,
        interaction_start_curriculum_probability=0.35,
        interaction_start_curriculum_min_probability=0.08,
        interaction_start_curriculum_decay_steps=6144,
    )


@configclass
class TACTICPayloadCalibrationCommandsCfg(TACTICCommandsCfg):
    locomotion = TACTICMissionCommandCfg(
        delivery_replay_fraction=1.0
    )


@configclass
class TACTICSingleObjectCurriculumCommandsCfg(TACTICCommandsCfg):
    locomotion = TACTICMissionCommandCfg(
        delivery_replay_fraction=1.0,
        interaction_start_curriculum_probability=1.0,
        interaction_start_curriculum_min_probability=1.0,
        interaction_start_object_id=4,
        automatic_recovery_task_candidate=False,
    )


@configclass
class TACTICActionsCfg(ActionsCfg):
    # Keep the actuator contract used to train ZYB-v0.  TACTIC motion skills
    # adapt this shared velocity policy through bounded learned residuals.
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_foot_wheel_joint",
            "FR_foot_wheel_joint",
            "RL_foot_wheel_joint",
            "RR_foot_wheel_joint",
        ],
        scale=0.1,
        use_default_offset=False,
        preserve_order=True,
    )
    arm_ik = TANDEMArmIkActionCfg(
        asset_name="robot",
        command_name="ee_goal",
        ee_body_name="link6",
        arm_joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
    )
    gripper = TACTICSymmetricGripperActionCfg()
    hierarchy = TACTICHierarchyActionCfg()


@configclass
class TACTICObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(
            func=VbcPolicyObsTerm,
            params={
                "asset_name": "robot",
                "obs_joint_names": OBS_JOINT_NAMES,
                "contact_body_names": [
                    "FL_foot",
                    "FR_foot",
                    "RL_foot",
                    "RR_foot",
                ],
                "history_len": 10,
                "use_priv": True,
                "arm_base_offset": (-0.3, 0.0, 0.09),
                # Keep the original ZYB-v0 physical action history.
                "physical_action_dim": 16,
            },
        )
        tactic_tau_down = ObsTerm(
            func=tactic_obs.tactic_tau_down,
            params={"command_name": "locomotion"},
        )
        tactic_tau_up = ObsTerm(
            func=tactic_obs.tactic_tau_up,
            params={"command_name": "locomotion"},
        )
        tactic_task_skill = ObsTerm(
            func=tactic_obs.tactic_task_skill,
            params={"command_name": "locomotion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HierarchyContextCfg(ObsGroup):
        context = ObsTerm(
            func=tactic_obs.tactic_hierarchy_context,
            params={"command_name": "locomotion"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    hierarchy_context: HierarchyContextCfg = HierarchyContextCfg()


@configclass
class TACTICRewardsCfg(RewardsCfg):
    tactic_mission_completion = RewTerm(
        func=tactic_rew.mission_completion, weight=0.0
    )
    tactic_mission_success = RewTerm(
        func=tactic_rew.mission_success, weight=0.0
    )
    tactic_task_progress = RewTerm(
        func=tactic_rew.selected_task_progress, weight=0.0
    )
    tactic_task_tracking = RewTerm(
        func=tactic_rew.selected_task_tracking, weight=0.0
    )
    tactic_ee_tracking = RewTerm(func=tactic_rew.ee_tracking, weight=0.0)
    tactic_valid_task = RewTerm(func=tactic_rew.valid_task_choice, weight=0.0)
    tactic_invalid_task = RewTerm(
        func=tactic_rew.invalid_task_choice, weight=0.0
    )
    tactic_option_switch = RewTerm(
        func=tactic_rew.option_switch_penalty, weight=0.0
    )
    tactic_control_progress = RewTerm(
        func=tactic_rew.control_aware_progress, weight=0.0
    )
    tactic_cbf_margin = RewTerm(func=tactic_rew.cbf_margin, weight=0.0)
    tactic_predicted_margin = RewTerm(
        func=tactic_rew.predicted_margin, weight=0.0
    )
    tactic_clf_decrease = RewTerm(
        func=tactic_rew.clf_decrease, weight=0.0
    )
    tactic_disturbance_rejection = RewTerm(
        func=tactic_rew.disturbance_rejection, weight=0.0
    )
    tactic_safety_violation = RewTerm(
        func=tactic_rew.safety_violation, weight=0.0
    )
    tactic_unsafe_progress = RewTerm(
        func=tactic_rew.unsafe_progress, weight=0.0
    )
    tactic_object_contact = RewTerm(
        func=tactic_rew.object_contact, weight=0.0
    )
    tactic_object_lift = RewTerm(func=tactic_rew.object_lift, weight=0.0)
    tactic_object_transport = RewTerm(
        func=tactic_rew.object_transport, weight=0.0
    )
    tactic_object_place = RewTerm(func=tactic_rew.object_place, weight=0.0)
    tactic_object_completion = RewTerm(
        func=tactic_rew.object_completion, weight=0.0
    )
    tactic_interaction_frontier = RewTerm(
        func=tactic_rew.robust_interaction_frontier, weight=0.0
    )
    tactic_grasp_hold = RewTerm(
        func=tactic_rew.grasp_hold_quality, weight=0.0
    )
    tactic_payload_progress = RewTerm(
        func=tactic_rew.payload_target_progress, weight=0.0
    )
    tactic_release_readiness = RewTerm(
        func=tactic_rew.payload_release_readiness, weight=0.0
    )
    tactic_intended_release = RewTerm(
        func=tactic_rew.intended_payload_release, weight=0.0
    )
    tactic_payload_retention = RewTerm(
        func=tactic_rew.payload_retention, weight=0.0
    )
    tactic_payload_drop = RewTerm(
        func=tactic_rew.payload_drop, weight=0.0
    )
    tactic_release_quality = RewTerm(
        func=tactic_rew.release_quality, weight=0.0
    )
    tactic_wrong_object_interaction = RewTerm(
        func=tactic_rew.wrong_object_interaction, weight=0.0
    )
    tactic_all_objects = RewTerm(
        func=tactic_rew.all_objects_delivered, weight=0.0
    )


@configclass
class TACTICTerminationsCfg(TerminationsCfg):
    mission_success = DoneTerm(
        func=tactic_rew.mission_succeeded,
        params={"command_name": "locomotion"},
    )


TACTIC_REWARD_SCALES = {
    "tactic_mission_completion": 20.0,
    "tactic_mission_success": 120.0,
    "tactic_task_progress": 24.0,
    "tactic_task_tracking": 0.8,
    "tactic_ee_tracking": 2.4,
    "tactic_valid_task": 0.2,
    "tactic_invalid_task": -2.0,
    "tactic_option_switch": -0.10,
    "tactic_control_progress": 28.0,
    "tactic_cbf_margin": 0.0,
    "tactic_predicted_margin": 0.0,
    "tactic_clf_decrease": 0.0,
    "tactic_disturbance_rejection": 0.0,
    "tactic_safety_violation": -2.5,
    "tactic_unsafe_progress": -12.0,
    "tactic_object_contact": 6.0,
    "tactic_object_lift": 10.0,
    "tactic_object_transport": 12.0,
    "tactic_object_place": 18.0,
    "tactic_object_completion": 40.0,
    "tactic_interaction_frontier": 320.0,
    "tactic_grasp_hold": 5.0,
    "tactic_payload_progress": 32.0,
    "tactic_release_readiness": 24.0,
    "tactic_intended_release": 20.0,
    "tactic_payload_retention": 6.0,
    "tactic_payload_drop": -45.0,
    "tactic_release_quality": 6.0,
    "tactic_wrong_object_interaction": -8.0,
    "tactic_all_objects": 0.0,
}


@configclass
class TACTICEnvCfg(ManipLocoEnvCfg):
    scene: TACTICSceneCfg = TACTICSceneCfg(
        num_envs=512, env_spacing=6.0
    )
    commands: TACTICTrainingCommandsCfg = TACTICTrainingCommandsCfg()
    actions: TACTICActionsCfg = TACTICActionsCfg()
    observations: TACTICObservationsCfg = TACTICObservationsCfg()
    events: TACTICEventCfg = TACTICEventCfg()
    rewards: TACTICRewardsCfg = TACTICRewardsCfg()
    terminations: TACTICTerminationsCfg = TACTICTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 240.0
        self.rewards.term_bad_contact.weight = -120.0
        self.rewards.term_tilt.weight = -100.0
        self.rewards.term_low_height.weight = -100.0
        self.actions.wheel_vel.scale = 0.1
        wheel_actuator = self.scene.robot.actuators["wheels"]
        wheel_actuator.effort_limit = 18.0
        if getattr(wheel_actuator, "effort_limit_sim", None) is not None:
            wheel_actuator.effort_limit_sim = 18.0
        gripper_actuator = self.scene.robot.actuators["gripper"]
        gripper_actuator.stiffness = 85.0
        gripper_actuator.damping = 4.5
        self.rewards.action_rate.params["action_dim"] = ACTION_LAYOUT.physical_dim
        self.rewards.wheel_forward_use.weight = 1.10
        self.rewards.wheel_forward_use.params["wheel_dir_signs"] = (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self.rewards.wheel_forward_use.params["vx_clip"] = 0.015
        self.rewards.wheel_turn_support.weight = 0.60
        self.rewards.wheel_turn_support.params["wheel_dir_signs"] = (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self.rewards.wheel_turn_support.params["wz_clip"] = 0.025
        factor = 1.0 / (100.0 * self.sim.dt * self.decimation)
        for name, scale in TACTIC_REWARD_SCALES.items():
            getattr(self.rewards, name).weight = float(scale) * factor


@configclass
class TACTICPlayEnvCfg(TACTICEnvCfg):
    commands: TACTICCommandsCfg = TACTICCommandsCfg()
    events: TACTICPlayEventCfg = TACTICPlayEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 6.0
        self.episode_length_s = 240.0


@configclass
class TACTICSingleObjectCurriculumEnvCfg(TACTICPlayEnvCfg):
    commands: TACTICSingleObjectCurriculumCommandsCfg = (
        TACTICSingleObjectCurriculumCommandsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.grasp_place_pad_aux_a.init_state.pos = (
            0.68,
            -0.33,
            0.70,
        )


@configclass
class TACTICStressEnvCfg(TACTICEnvCfg):
    commands: TACTICCommandsCfg = TACTICCommandsCfg()
    events: TACTICStressEventCfg = TACTICStressEventCfg()


@configclass
class TACTICPayloadCalibrationEnvCfg(TACTICEnvCfg):
    commands: TACTICPayloadCalibrationCommandsCfg = (
        TACTICPayloadCalibrationCommandsCfg()
    )
