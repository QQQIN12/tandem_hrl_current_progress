"""Shared tensor layout for the TACTIC-HRL policy and environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TACTICActionLayout:
    # ZYB-v0 contributes 16 locomotion actions.  A single symmetric gripper
    # logit is appended so the two opposed fingers cannot fight each other.
    physical_dim: int = 17
    task_dim: int = 12
    skill_dim: int = 12
    object_dim: int = 6
    task_subgoal_dim: int = 8
    skill_param_dim: int = 6
    termination_dim: int = 2

    @property
    def hierarchy_dim(self) -> int:
        return (
            self.task_dim
            + self.skill_dim
            + self.object_dim
            + self.task_subgoal_dim
            + self.skill_param_dim
            + self.termination_dim
        )

    @property
    def total_dim(self) -> int:
        return self.physical_dim + self.hierarchy_dim

    def slices(self) -> dict[str, slice]:
        offset = 0
        result: dict[str, slice] = {}
        for name, width in (
            ("physical", self.physical_dim),
            ("task", self.task_dim),
            ("skill", self.skill_dim),
            ("object", self.object_dim),
            ("task_subgoal", self.task_subgoal_dim),
            ("skill_param", self.skill_param_dim),
            ("termination", self.termination_dim),
        ):
            result[name] = slice(offset, offset + width)
            offset += width
        return result

    def hierarchy_slices(self) -> dict[str, slice]:
        result = self.slices()
        base = self.physical_dim
        return {
            name: slice(value.start - base, value.stop - base)
            for name, value in result.items()
            if name != "physical"
        }


ACTION_LAYOUT = TACTICActionLayout()

GLOBAL_CONTEXT_DIM = 49
TASK_SLOT_COUNT = ACTION_LAYOUT.task_dim
MOTION_SKILL_COUNT = 4
INTERACTION_SKILL_COUNT = 3
if MOTION_SKILL_COUNT * INTERACTION_SKILL_COUNT != ACTION_LAYOUT.skill_dim:
    raise ValueError("TACTIC skill factors must span the discrete skill space")
# The final eight values form a robot-object interaction descriptor:
# left/right fingertip-to-object vectors, carrying state, and bilateral
# contact symmetry.  These quantities are independent of joint indexing.
TASK_SLOT_FEATURE_DIM = 40
HIERARCHY_CONTEXT_DIM = GLOBAL_CONTEXT_DIM + TASK_SLOT_COUNT * TASK_SLOT_FEATURE_DIM
TASK_OUTCOME_DIM = 4
SKILL_OUTCOME_DIM = 4
# Candidate skill effects are expressed in measurable, transferable state
# coordinates: vx, wz, task progress, tracking improvement, four robustness
# quantities, and robot-object interaction progress.
SKILL_EFFECT_DIM = 9
# vx, wz, tracking improvement, CBF margin, preview margin, CLF decrease,
# and disturbance rejection quality.
MOTION_EXECUTION_EFFECT_DIM = 7

# One-hot task-role entries inside each raw slot.  These indices are used only
# to decide which skill factor is relevant; the option identity itself remains
# learned.
TASK_SLOT_MANIPULATION_TYPE_INDEX = 8
TASK_SLOT_RECOVERY_TYPE_INDEX = 9
TASK_SLOT_DELIVERY_TYPE_INDEX = 10
TASK_SLOT_REQUIRED_INDEX = 11
TASK_SLOT_COMPLETED_INDEX = 12
TASK_SLOT_AVAILABLE_INDEX = 13
TASK_SLOT_REMAINING_PROGRESS_INDEX = 15
TASK_SLOT_INTERACTION_STATE_SLICE = slice(22, 26)
TASK_SLOT_DISTANCE_INDEX = 3
TASK_SLOT_HEADING_INDEX = 4
TASK_SLOT_OBJECT_DELTA_SLICE = slice(16, 19)
TASK_SLOT_TARGET_DELTA_SLICE = slice(19, 22)
TASK_SLOT_GRIPPER_CLOSURE_INDEX = 26
TASK_SLOT_REACHABILITY_INDEX = 27
TASK_SLOT_LEFT_FINGER_DELTA_SLICE = slice(32, 35)
TASK_SLOT_RIGHT_FINGER_DELTA_SLICE = slice(35, 38)
TASK_SLOT_CARRYING_INDEX = 38
TASK_SLOT_CONTACT_SYMMETRY_INDEX = 39

# Differentiable relation-space capture set.  Distances are in metres after
# undoing the slot normalization.  A secure skill is admissible only when the
# gripper midpoint, both fingers, and the TCP are simultaneously close enough
# to the selected object.
CAPTURE_CENTER_RADIUS = 0.028
CAPTURE_FINGER_RADIUS = 0.065
CAPTURE_TCP_RADIUS = 0.080
CAPTURE_BARRIER_GAIN = 70.0
# Entering the secure option first triggers the final insertion motion.  The
# tighter capture set above remains the only condition that permits closure.
SECURE_ENTRY_CENTER_RADIUS = 0.065
SECURE_ENTRY_FINGER_RADIUS = 0.090
SECURE_ENTRY_TCP_RADIUS = 0.080
SECURE_ENTRY_BARRIER_GAIN = 70.0
# Interaction skills use a phase-dependent CBF envelope inside the hierarchy
# objective.  The hard floor remains active in every phase; measured contact
# or carrying evidence only enlarges the soft transient budget.
INTERACTION_HARD_CONTROL_FLOOR = 0.12
SECURE_CBF_TRANSIENT_SLACK = 0.08
RELEASE_CBF_TRANSIENT_SLACK = 0.04
# The strict placement set has a 0.085 m planar and 0.070 m vertical
# tolerance.  The release option first moves the object to a reachable hover
# set, then opens the fingers and lets the event detector verify settling.
RELEASE_TARGET_RADIUS = 0.12
RELEASE_TARGET_GAIN = 35.0
RELEASE_HOVER_HEIGHT = 0.18
RELEASE_VERTICAL_TOLERANCE = 0.12
RELEASE_VERTICAL_GAIN = 22.0
RELEASE_TRANSPORT_THRESHOLD = 0.20
RELEASE_TRANSPORT_GAIN = 8.0
RELEASE_READINESS_THRESHOLD = 0.82
RELEASE_READINESS_GAIN = 24.0
RELEASE_SETTLE_VX = 0.10
RELEASE_SETTLE_WZ = 0.25
# A carried payload reduces the admissible motion envelope.  These reserves
# define the skill-level barrier used to interrupt an active motion option
# before the body-level CBF becomes binding.
PAYLOAD_SKILL_PREVIEW_RESERVE = 0.18
PAYLOAD_SKILL_SAFETY_RESERVE = 0.15
PAYLOAD_SKILL_RISK_GAIN = 2.0
PAYLOAD_SKILL_SWITCH_MARGIN = 0.08
# Payload transport uses a contracted task-space target and a learned
# transient-risk signal shared by the task and skill layers.
PAYLOAD_TASK_CONTRACTION_GAIN = 1.5
PAYLOAD_TRANSIENT_SPEED_FLOOR = 0.18
PAYLOAD_TRANSIENT_SPEED_RANGE = 0.12
PAYLOAD_TRANSIENT_DISTANCE_SCALE = 0.75
PAYLOAD_TRANSIENT_SPEED_SCALE = 0.18
PAYLOAD_TRANSIENT_TRACKING_ALLOWANCE = 0.08
PAYLOAD_TRANSIENT_TRACKING_SCALE = 0.20
PAYLOAD_TRANSIENT_GATE_GAIN = 0.75
PAYLOAD_TRANSIENT_BRAKE_GAIN = 0.45

# Four measured quantities supervised by the control-prediction auxiliary loss:
# CBF margin, preview margin, CLF decrease, and disturbance rejection quality.
CONTROL_TARGET_SLICE = slice(12, 16)

# Robot-state and command fields in the embodiment-independent global packet.
BASE_HEIGHT_INDEX = 3
BASE_TILT_INDEX = 4
SUPPORT_COUNT_INDEX = 5
BASE_VX_INDEX = 7
BASE_WZ_INDEX = 8
COMMAND_VX_INDEX = 9
COMMAND_WZ_INDEX = 10
# Lateral task intent is appended to preserve the existing global packet
# layout while exposing the nonholonomic task-to-skill interface.
COMMAND_VY_INDEX = 48
SAFETY_MARGIN_INDEX = 12
PREVIEW_MARGIN_INDEX = 13
CLF_DECREASE_INDEX = 14
DISTURBANCE_QUALITY_INDEX = 15
CURRICULUM_LEVEL_INDEX = 34
MORPHOLOGY_SLICE = slice(43, 46)
WHEEL_RADIUS_INDEX = 45

# These identities and the task/skill hazard state are masked from the
# transition encoder so its auxiliary task cannot copy the target label.
EXECUTED_TASK_INDEX = 16
EXECUTED_SKILL_INDEX = 17
EXECUTED_OBJECT_INDEX = 18
TERMINATION_STATE_SLICE = slice(30, 32)
SELECTED_PROGRESS_INDEX = 32
SELECTED_PROGRESS_DELTA_INDEX = 33
