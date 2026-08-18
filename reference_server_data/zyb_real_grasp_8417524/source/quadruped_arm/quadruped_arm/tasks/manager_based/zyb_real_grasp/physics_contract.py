"""Physical constants for the ZYB-v0 real-grasp extension.

Robot mass, inertia, joint limits, and leg/wheel/arm actuator parameters remain
defined by ``ZYB_QUADRUPED_ARM_Cfg``.  This module only records task assets and
the two gripper gains validated with a dynamic object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactMaterial:
    static_friction: float
    dynamic_friction: float
    restitution: float = 0.0

    def validate(self) -> None:
        if self.static_friction < self.dynamic_friction:
            raise ValueError("static friction must not be below dynamic friction")
        if self.dynamic_friction < 0.0:
            raise ValueError("dynamic friction must be non-negative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")


# ZYB-v0 values used as an executable regression contract.
ZYB_V0_LEG_STIFFNESS = 160.0
ZYB_V0_LEG_DAMPING = 5.0
ZYB_V0_WHEEL_EFFORT_LIMIT = 23.5
ZYB_V0_WHEEL_VELOCITY_LIMIT = 30.0
ZYB_V0_WHEEL_STIFFNESS = 0.0
ZYB_V0_WHEEL_DAMPING = 0.5
ZYB_V0_WHEEL_JOINT_FRICTION = 0.01
ZYB_V0_ARM_PROXIMAL_STIFFNESS = 1000.0
ZYB_V0_ARM_PROXIMAL_DAMPING = 80.0
ZYB_V0_ARM_DISTAL_STIFFNESS = 1000.0
ZYB_V0_ARM_DISTAL_DAMPING = 80.0

# The original 17/0.02 gains did not establish repeatable bilateral contact.
# These values are the only actuator calibration applied by the new task.
REAL_GRASP_GRIPPER_STIFFNESS = 85.0
REAL_GRASP_GRIPPER_DAMPING = 4.5

WHEEL_GROUND_MATERIAL = ContactMaterial(1.00, 0.85)
FINGER_SIMULATION_MATERIAL = ContactMaterial(0.90, 0.70)
PLATFORM_MATERIAL = ContactMaterial(1.00, 0.85)
OBJECT_MATERIAL = ContactMaterial(1.36, 1.02)

OBJECT_MASS_KG = 0.044
OBJECT_SIZE_M = (0.092, 0.038, 0.078)
FINGER_MATERIAL_HARDWARE_CALIBRATED = False


def validate_physics_contract() -> None:
    for material in (
        WHEEL_GROUND_MATERIAL,
        FINGER_SIMULATION_MATERIAL,
        PLATFORM_MATERIAL,
        OBJECT_MATERIAL,
    ):
        material.validate()
    if OBJECT_MASS_KG <= 0.0:
        raise ValueError("object mass must be positive")
    if REAL_GRASP_GRIPPER_STIFFNESS <= 17.0:
        raise ValueError("dynamic-grasp gripper calibration is missing")
    if REAL_GRASP_GRIPPER_DAMPING <= 0.02:
        raise ValueError("dynamic-grasp gripper damping is missing")


validate_physics_contract()
