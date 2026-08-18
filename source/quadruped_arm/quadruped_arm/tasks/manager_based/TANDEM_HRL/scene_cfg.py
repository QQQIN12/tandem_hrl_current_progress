"""Single-object two-platform scene built on the unmodified ZYB-v0 robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import (
    ManipLocoSceneCfg,
)

from .physics_contract import (
    MAIN_OBJECT_MASS_KG,
    MAIN_OBJECT_MATERIAL,
    MAIN_OBJECT_SIZE_M,
    PLATFORM_MATERIAL,
)


SOURCE_PLATFORM_POSITION = (-0.40, 0.00, 0.70)
SOURCE_PLATFORM_SIZE = (0.16, 0.11, 0.06)
TARGET_PLATFORM_POSITION = (0.65, 0.45, 0.82)
TARGET_PLATFORM_SIZE = (0.22, 0.20, 0.06)
OBJECT_START_POSITION = (
    SOURCE_PLATFORM_POSITION[0],
    SOURCE_PLATFORM_POSITION[1],
    SOURCE_PLATFORM_POSITION[2]
    + 0.5 * SOURCE_PLATFORM_SIZE[2]
    + 0.5 * MAIN_OBJECT_SIZE_M[2],
)


def _platform(
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=position,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PLATFORM_MATERIAL.static_friction,
                dynamic_friction=PLATFORM_MATERIAL.dynamic_friction,
                restitution=PLATFORM_MATERIAL.restitution,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.72,
            ),
        ),
    )


def _main_object() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBJECT_START_POSITION,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=MAIN_OBJECT_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=5,
                max_depenetration_velocity=0.75,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MAIN_OBJECT_MASS_KG),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=MAIN_OBJECT_MATERIAL.static_friction,
                dynamic_friction=MAIN_OBJECT_MATERIAL.dynamic_friction,
                restitution=MAIN_OBJECT_MATERIAL.restitution,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.32, 0.82),
                roughness=0.48,
            ),
        ),
    )


def _object_contact(link_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=False,
        # Full friction-force reporting is prohibitively expensive in the
        # current Isaac Sim build.  Contact duration and object slip provide
        # the material-calibration signal without changing simulation speed.
        track_friction_forces=False,
        force_threshold=0.20,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspObject"],
    )


@configclass
class TANDEMMainlineSceneCfg(ManipLocoSceneCfg):
    """The first complete-chain scene: one object and two raised platforms."""

    source_platform = _platform(
        "SourcePlatform",
        SOURCE_PLATFORM_POSITION,
        SOURCE_PLATFORM_SIZE,
        (0.20, 0.42, 0.72),
    )
    target_platform = _platform(
        "TargetPlatform",
        TARGET_PLATFORM_POSITION,
        TARGET_PLATFORM_SIZE,
        (0.12, 0.50, 0.24),
    )
    grasp_object = _main_object()

    left_finger_object_contact = _object_contact("link7")
    right_finger_object_contact = _object_contact("link8")
    wrist_object_contact = _object_contact("link6")
    arm_link1_object_contact = _object_contact("link1")
    arm_link2_object_contact = _object_contact("link2")
    arm_link3_object_contact = _object_contact("link3")
    arm_link4_object_contact = _object_contact("link4")
    arm_link5_object_contact = _object_contact("link5")
