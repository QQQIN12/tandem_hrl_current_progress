"""Authoritative architecture contract for the TANDEM-HRL mainline."""

from __future__ import annotations

from dataclasses import dataclass


TASK_PHASES = (
    "approach",
    "align",
    "acquire",
    "lift",
    "transport",
    "place",
    "release",
    "verify",
)

SKILL_FAMILIES = (
    "base_navigation",
    "whole_body_alignment",
    "grasp_interaction",
    "payload_transport",
    "placement",
    "recovery",
)


@dataclass(frozen=True)
class MainlineContract:
    task_id: str
    state_source: str
    task_phases: tuple[str, ...]
    skill_families: tuple[str, ...]
    learned_task_decomposition: bool
    learned_skill_decomposition: bool
    payload_aware_wbc: bool
    natural_episode_start: bool
    single_evolving_checkpoint: bool
    camera_enabled: bool
    visual_distillation_enabled: bool

    def validate(self) -> None:
        if self.task_id != "TANDEM-HRL-Privileged-Mainline-v0":
            raise ValueError("The privileged mainline must have one canonical task ID")
        if self.state_source != "privileged_relation_state":
            raise ValueError("The first mainline must use direct simulator relation state")
        if self.task_phases != TASK_PHASES:
            raise ValueError("Task decomposition must implement the eight-phase chain")
        if self.skill_families != SKILL_FAMILIES:
            raise ValueError("Skill decomposition does not match the execution contract")
        if not self.learned_task_decomposition:
            raise ValueError("Task decomposition must be learned")
        if not self.learned_skill_decomposition:
            raise ValueError("Skill decomposition must be learned")
        if not self.payload_aware_wbc:
            raise ValueError("The physical executor must be payload-aware")
        if not self.natural_episode_start:
            raise ValueError("Formal episodes cannot start from staged interaction states")
        if not self.single_evolving_checkpoint:
            raise ValueError("All stages must continue from one checkpoint lineage")
        if self.camera_enabled or self.visual_distillation_enabled:
            raise ValueError("Vision is not part of the privileged-state HRL stage")


MAINLINE_CONTRACT = MainlineContract(
    task_id="TANDEM-HRL-Privileged-Mainline-v0",
    state_source="privileged_relation_state",
    task_phases=TASK_PHASES,
    skill_families=SKILL_FAMILIES,
    learned_task_decomposition=True,
    learned_skill_decomposition=True,
    payload_aware_wbc=True,
    natural_episode_start=True,
    single_evolving_checkpoint=True,
    camera_enabled=False,
    visual_distillation_enabled=False,
)

MAINLINE_CONTRACT.validate()
