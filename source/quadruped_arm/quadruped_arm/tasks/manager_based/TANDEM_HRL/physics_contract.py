"""Task-level physical parameters allowed to differ from ZYB-v0."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactMaterial:
    static_friction: float
    dynamic_friction: float
    restitution: float = 0.0

    def validate(self) -> None:
        if self.static_friction < self.dynamic_friction:
            raise ValueError("Static friction must not be below dynamic friction")
        if self.dynamic_friction < 0.0:
            raise ValueError("Dynamic friction must be non-negative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("Restitution must lie in [0, 1]")


ZYB_V0_WHEEL_EFFORT_LIMIT = 23.5
ZYB_V0_WHEEL_VELOCITY_LIMIT = 30.0
ZYB_V0_WHEEL_STIFFNESS = 0.0
ZYB_V0_WHEEL_DAMPING = 0.5
ZYB_V0_WHEEL_JOINT_FRICTION = 0.01

ZYB_V0_ARM_PROXIMAL_STIFFNESS = 1000.0
ZYB_V0_ARM_PROXIMAL_DAMPING = 80.0
ZYB_V0_ARM_DISTAL_STIFFNESS = 1000.0
ZYB_V0_ARM_DISTAL_DAMPING = 80.0

# The ZYB-v0 gains did not establish bilateral contact with a dynamic object.
# These two values are the only actuator-level task calibration retained from
# the physical contact, lift, and carry gate recorded in the parameter audit.
MAINLINE_GRIPPER_STIFFNESS = 85.0
MAINLINE_GRIPPER_DAMPING = 4.5

WHEEL_GROUND_MATERIAL = ContactMaterial(1.00, 0.85)
PLATFORM_MATERIAL = ContactMaterial(1.00, 0.85)
MAIN_OBJECT_MATERIAL = ContactMaterial(1.36, 1.02)
MAIN_OBJECT_MASS_KG = 0.044
MAIN_OBJECT_SIZE_M = (0.092, 0.038, 0.078)

# The imported finger collision bodies have no explicit material.  A fixed
# trajectory sweep showed 0/1 retention with the inherited behavior, 5/6 at
# 0.90/0.70, and 4/6 at 1.36/1.02.  Use the lower candidate in simulation;
# real hardware identification is still required before sim-to-real claims.
FINGER_SIMULATION_MATERIAL = ContactMaterial(0.90, 0.70)
FINGER_MATERIAL_CALIBRATED = False


def validate_physics_contract() -> None:
    for material in (
        WHEEL_GROUND_MATERIAL,
        PLATFORM_MATERIAL,
        MAIN_OBJECT_MATERIAL,
        FINGER_SIMULATION_MATERIAL,
    ):
        material.validate()
    if MAIN_OBJECT_MASS_KG <= 0.0:
        raise ValueError("Object mass must be positive")
    if MAINLINE_GRIPPER_STIFFNESS <= 17.0:
        raise ValueError("The validated dynamic-grasp calibration is missing")
    if MAINLINE_GRIPPER_DAMPING <= 0.02:
        raise ValueError("The validated gripper damping calibration is missing")
    if FINGER_MATERIAL_CALIBRATED:
        raise ValueError("No real finger material calibration has been approved")


validate_physics_contract()
