"""Dynamic object and two-platform scene for ZYB-v0."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from quadruped_arm.tasks.manager_based.maniploco.maniploco_env_cfg import ManipLocoSceneCfg

from .physics_contract import OBJECT_MASS_KG, OBJECT_MATERIAL, OBJECT_SIZE_M, PLATFORM_MATERIAL


SOURCE_PLATFORM_POSITION = (-0.40, 0.00, 0.70)
SOURCE_PLATFORM_SIZE = (0.16, 0.11, 0.06)
TARGET_PLATFORM_POSITION = (0.65, 0.45, 0.82)
TARGET_PLATFORM_SIZE = (0.22, 0.20, 0.06)
OBJECT_START_POSITION = (
    SOURCE_PLATFORM_POSITION[0],
    SOURCE_PLATFORM_POSITION[1],
    SOURCE_PLATFORM_POSITION[2] + 0.5 * SOURCE_PLATFORM_SIZE[2] + 0.5 * OBJECT_SIZE_M[2],
)


def _material(cfg):
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
        restitution=cfg.restitution,
    )


def _platform(name, position, size, color) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_material(PLATFORM_MATERIAL),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.72),
        ),
    )


def _grasp_object() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START_POSITION),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=5,
                max_depenetration_velocity=0.75,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
            physics_material=_material(OBJECT_MATERIAL),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.32, 0.82), roughness=0.48
            ),
        ),
    )


def _object_contact(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
        track_contact_points=False,
        track_friction_forces=False,
        force_threshold=0.20,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspObject"],
    )


@configclass
class ZYBRealGraspSceneCfg(ManipLocoSceneCfg):
    """ZYB-v0 robot with physical grasp assets and contact instrumentation."""

    # These legacy names do not exist in the USD.  The new task uses the base
    # sensor directly and keeps the original task untouched.
    trunk_contact = None
    head_contact = None

    source_platform = _platform(
        "SourcePlatform", SOURCE_PLATFORM_POSITION, SOURCE_PLATFORM_SIZE, (0.20, 0.42, 0.72)
    )
    target_platform = _platform(
        "TargetPlatform", TARGET_PLATFORM_POSITION, TARGET_PLATFORM_SIZE, (0.12, 0.50, 0.24)
    )
    grasp_object = _grasp_object()

    left_finger_object_contact = _object_contact("link7")
    right_finger_object_contact = _object_contact("link8")
    wrist_object_contact = _object_contact("link6")

