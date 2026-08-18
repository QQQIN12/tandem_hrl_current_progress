"""Summarize TANDEM-HRL per-environment evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


PREFIX = "Command/locomotion/TACTIC/"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate_output", type=Path, default=None)
    return parser.parse_args()


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _number(row, key):
    value = row.get(key, "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _mean_field(rows, key):
    return _mean([_number(row, key) for row in rows])


def _event_rate(rows, key, threshold=0.5):
    maximum_key = key + "__max"
    mean_key = key + "__mean"
    values = []
    for row in rows:
        value = _number(row, maximum_key)
        if not math.isfinite(value):
            value = _number(row, mean_key)
            active = value > 1.0e-4
        else:
            active = value >= threshold
        values.append(float(active))
    return _mean(values)


def _low_rate(rows, key, threshold):
    values = []
    for row in rows:
        value = _number(row, key)
        if math.isfinite(value):
            values.append(float(value <= threshold))
    return _mean(values)


def _high_rate(rows, key, threshold):
    values = []
    for row in rows:
        value = _number(row, key)
        if math.isfinite(value):
            values.append(float(value >= threshold))
    return _mean(values)


def _required_object_event_stats(
    rows,
    object_ids,
    event_name,
    threshold=0.5,
):
    if not object_ids:
        return math.nan, math.nan, math.nan
    fractions = []
    any_success = []
    all_success = []
    for row in rows:
        hits = []
        for object_id in object_ids:
            value = _number(
                row,
                "Diagnostic/TACTIC/object_{}/{}__max".format(
                    object_id,
                    event_name,
                ),
            )
            hits.append(
                float(math.isfinite(value) and value >= threshold)
            )
        fractions.append(sum(hits) / len(hits))
        any_success.append(float(any(hits)))
        all_success.append(float(all(hits)))
    return (
        _mean(any_success),
        _mean(fractions),
        _mean(all_success),
    )


def _event_time_stats(rows, keys):
    if not keys:
        return math.nan, math.nan
    observed = []
    restricted = []
    for row in rows:
        horizon = (
            _number(row, "num_steps")
            * _number(row, "step_dt_s")
        )
        values = [_number(row, key) for key in keys]
        if all(math.isfinite(value) for value in values):
            event_time = max(values)
            observed.append(event_time)
            restricted.append(event_time)
        elif math.isfinite(horizon):
            restricted.append(horizon)
    return _mean(observed), _mean(restricted)


def _checkpoint_iteration(checkpoint):
    match = re.search(r"model_(\d+)\.pt$", checkpoint)
    return int(match.group(1)) if match else -1


def _summarize(
    path,
    rows,
    task_set_id=-1,
    scenario_name="",
    required_task_ids=(),
):
    if not rows:
        raise RuntimeError("Empty per-environment CSV: {}".format(path))
    first = rows[0]
    if not required_task_ids:
        task_spec = (
            first.get("required_task_grid", "")
            or first.get("required_task_sets", "")
        )
        if task_spec and ";" not in task_spec:
            required_task_ids = tuple(
                int(value)
                for value in re.split(r"[+,]", task_spec)
                if value.strip()
            )

    contact_rate = _event_rate(
        rows, PREFIX + "selected_object_contact"
    )
    lift_rate = _event_rate(
        rows, PREFIX + "selected_object_lift"
    )
    carry_rate = _event_rate(
        rows, PREFIX + "selected_object_carrying"
    )
    transport_rate = _event_rate(
        rows, PREFIX + "selected_object_transport"
    )
    place_rate = _event_rate(
        rows, PREFIX + "selected_object_place"
    )
    intended_release_env_rate = _event_rate(
        rows, PREFIX + "selected_object_release_event"
    )
    drop_env_rate = _event_rate(
        rows, PREFIX + "selected_object_drop_event"
    )
    mission_success_rate = _event_rate(
        rows, PREFIX + "mission_completion", threshold=0.99
    )
    object_completion = _mean_field(
        rows, PREFIX + "object_completion_mean__max"
    )
    if not math.isfinite(object_completion):
        object_completion = _mean_field(
            rows, PREFIX + "object_completion_mean__mean"
        )
    cbf_margin = _mean_field(
        rows, PREFIX + "cbf_margin__mean"
    )
    predicted_margin = _mean_field(
        rows, PREFIX + "predicted_margin__mean"
    )
    safety_failure_rate = _mean_field(
        rows, "Safety/cbf_violation__mean"
    )
    if not math.isfinite(safety_failure_rate):
        safety_failure_rate = _low_rate(
            rows, PREFIX + "cbf_margin__min", 0.02
        )
    preview_violation_rate = _mean_field(
        rows, "Safety/preview_violation__mean"
    )
    tilt_failure_rate = _mean_field(
        rows, "Termination/tilt__mean"
    )
    if not math.isfinite(tilt_failure_rate):
        tilt_failure_rate = _mean_field(
            rows, "Safety/tilt_violation__mean"
        )
    if not math.isfinite(tilt_failure_rate):
        tilt_failure_rate = _high_rate(
            rows, PREFIX + "base_tilt__max", 0.60
        )
    low_height_failure_rate = _mean_field(
        rows, "Termination/low_height__mean"
    )
    bad_contact_failure_rate = _mean_field(
        rows, "Termination/bad_contact__mean"
    )
    failure_step_rate = _mean_field(
        rows, "Termination/failure_any__mean"
    )
    time_out_step_rate = _mean_field(
        rows, "Termination/time_out_any__mean"
    )
    done_rate = _mean_field(rows, "done__mean")
    done_any_rate = _event_rate(rows, "done", threshold=0.5)
    required_object_ids = tuple(
        sorted(
            {
                task_id - 5
                for task_id in required_task_ids
                if 5 <= task_id <= 10
            }
        )
    )
    (
        physical_contact_any,
        physical_contact_fraction,
        physical_contact_all,
    ) = _required_object_event_stats(
        rows,
        required_object_ids,
        "object_contact_memory",
    )
    (
        physical_lift_any,
        physical_lift_fraction,
        physical_lift_all,
    ) = _required_object_event_stats(
        rows,
        required_object_ids,
        "object_lift_memory",
        threshold=0.20,
    )
    (
        physical_carry_any,
        physical_carry_fraction,
        physical_carry_all,
    ) = _required_object_event_stats(
        rows,
        required_object_ids,
        "object_carrying",
    )
    (
        physical_transport_any,
        physical_transport_fraction,
        physical_transport_all,
    ) = _required_object_event_stats(
        rows,
        required_object_ids,
        "object_transport_memory",
    )
    (
        object_completion_any,
        object_completion_fraction,
        object_completion_all,
    ) = _required_object_event_stats(
        rows,
        required_object_ids,
        "object_completion",
    )
    mission_success_time, mission_success_restricted_time = (
        _event_time_stats(
            rows,
            ("EventTime/mission_success_s",),
        )
    )
    event_time_stats = {}
    for event_name in (
        "contact",
        "lift",
        "carry",
        "transport",
        "completion",
    ):
        keys = tuple(
            "EventTime/object_{}_{}_s".format(
                object_id,
                event_name,
            )
            for object_id in required_object_ids
        )
        observed_time, restricted_time = _event_time_stats(rows, keys)
        event_time_stats[
            "required_{}_time_s".format(event_name)
        ] = observed_time
        event_time_stats[
            "required_{}_restricted_time_s".format(event_name)
        ] = restricted_time

    bounded_cbf = min(max(cbf_margin, 0.0), 1.0)
    strict_completion = (
        object_completion_all
        if math.isfinite(object_completion_all)
        else mission_success_rate
    )
    score = (
        0.35 * mission_success_rate
        + 0.20 * min(max(strict_completion, 0.0), 1.0)
        + 0.10 * place_rate
        + 0.08 * transport_rate
        + 0.05 * carry_rate
        + 0.04 * lift_rate
        + 0.03 * contact_rate
        + 0.08 * bounded_cbf
        + 0.04 * (1.0 - tilt_failure_rate)
        + 0.03 * (1.0 - failure_step_rate)
    )

    checkpoint = first.get("checkpoint", "")
    return {
        "source": str(path),
        "scenario": scenario_name or first.get("scenario", ""),
        "task_set_id": task_set_id,
        "method": first.get("method", ""),
        "task": first.get("task", ""),
        "regime": (
            "stress"
            if "Stress" in first.get("task", "")
            else "nominal"
        ),
        "seed": first.get("seed", ""),
        "num_steps": first.get("num_steps", ""),
        "step_dt_s": first.get("step_dt_s", ""),
        "ablation": first.get("ablation", "none"),
        "release_target_radius": first.get(
            "release_target_radius_override", ""
        ),
        "interaction_phase_prior_gain": first.get(
            "interaction_phase_prior_gain_override", ""
        ),
        "checkpoint": checkpoint,
        "iteration": _checkpoint_iteration(checkpoint),
        "num_envs": len(rows),
        "reward_mean": _mean_field(rows, "reward__mean"),
        "done_rate": done_rate,
        "done_any_rate": done_any_rate,
        "failure_step_rate": failure_step_rate,
        "time_out_step_rate": time_out_step_rate,
        "mission_success_rate": mission_success_rate,
        "mission_success_time_s": mission_success_time,
        "mission_success_restricted_time_s": (
            mission_success_restricted_time
        ),
        "object_completion": object_completion,
        "required_object_count": len(required_object_ids),
        "physical_contact_any_rate": physical_contact_any,
        "physical_contact_fraction": physical_contact_fraction,
        "physical_contact_all_rate": physical_contact_all,
        "physical_lift_any_rate": physical_lift_any,
        "physical_lift_fraction": physical_lift_fraction,
        "physical_lift_all_rate": physical_lift_all,
        "physical_carry_any_rate": physical_carry_any,
        "physical_carry_fraction": physical_carry_fraction,
        "physical_carry_all_rate": physical_carry_all,
        "physical_transport_any_rate": physical_transport_any,
        "physical_transport_fraction": physical_transport_fraction,
        "physical_transport_all_rate": physical_transport_all,
        "object_completion_any_rate": object_completion_any,
        "object_completion_fraction": object_completion_fraction,
        "object_completion_all_rate": object_completion_all,
        **event_time_stats,
        "contact_rate": contact_rate,
        "lift_rate": lift_rate,
        "carry_rate": carry_rate,
        "transport_rate": transport_rate,
        "place_rate": place_rate,
        "selected_target_progress_mean": _mean_field(
            rows, PREFIX + "selected_object_target_progress__mean"
        ),
        "release_readiness_mean": _mean_field(
            rows, PREFIX + "selected_object_release_readiness__mean"
        ),
        "intended_release_env_rate": intended_release_env_rate,
        "drop_env_rate": drop_env_rate,
        "place_error_xy_mean": _mean_field(
            rows, PREFIX + "selected_object_place_error_xy__mean"
        ),
        "place_error_z_mean": _mean_field(
            rows, PREFIX + "selected_object_place_error_z__mean"
        ),
        "gripper_distance_mean": _mean_field(
            rows, PREFIX + "selected_object_gripper_distance__mean"
        ),
        "cbf_margin": cbf_margin,
        "predicted_margin": predicted_margin,
        "safety_failure_rate": safety_failure_rate,
        "preview_violation_rate": preview_violation_rate,
        "tilt_failure_rate": tilt_failure_rate,
        "low_height_failure_rate": low_height_failure_rate,
        "bad_contact_failure_rate": bad_contact_failure_rate,
        "base_tilt_mean": _mean_field(
            rows, PREFIX + "base_tilt__mean"
        ),
        "control_recovery_active_rate": _mean_field(
            rows, PREFIX + "control_recovery_active__mean"
        ),
        "control_recovery_constraint_rate": _mean_field(
            rows, PREFIX + "control_recovery_constraint_active__mean"
        ),
        "recovery_task_executed_rate": _mean_field(
            rows, PREFIX + "recovery_task_executed__mean"
        ),
        "recovery_task_probability_mean": _mean_field(
            rows, PREFIX + "recovery_task_probability__mean"
        ),
        "executed_task_probability_mean": _mean_field(
            rows, PREFIX + "executed_task_probability__mean"
        ),
        "task_constraint_projection_rate": _mean_field(
            rows, PREFIX + "task_constraint_projection__mean"
        ),
        "recovery_latch_seen_rate": _mean_field(
            rows, PREFIX + "recovery_latch_seen__mean"
        ),
        "recovery_valid_seen_rate": _mean_field(
            rows, PREFIX + "recovery_valid_seen__mean"
        ),
        "recovery_pressure_seen_mean": _mean_field(
            rows, PREFIX + "recovery_pressure_seen__mean"
        ),
        "payload_posture_projection_mean": _mean_field(
            rows, PREFIX + "payload_posture_projection__mean"
        ),
        "payload_posture_projection_max": _mean_field(
            rows, PREFIX + "payload_posture_projection__max"
        ),
        "payload_relation_violation_mean": _mean_field(
            rows, PREFIX + "payload_relation_violation__mean"
        ),
        "support_count_mean": _mean_field(
            rows, PREFIX + "support_count__mean"
        ),
        "task_error_mean": _mean_field(
            rows, PREFIX + "task_error__mean"
        ),
        "ee_error_mean": _mean_field(
            rows, PREFIX + "ee_error__mean"
        ),
        "base_xy_tracking_error": _mean_field(
            rows, "Tracking/base_xy_error__mean"
        ),
        "base_vx_tracking_error": _mean_field(
            rows, "Tracking/base_vx_error__mean"
        ),
        "base_vy_tracking_error": _mean_field(
            rows, "Tracking/base_vy_error__mean"
        ),
        "base_wz_tracking_error": _mean_field(
            rows, "Tracking/base_wz_error__mean"
        ),
        "rear_contact_fraction": _mean_field(
            rows, "State/rear_contact_fraction__mean"
        ),
        "rear_any_airborne_rate": _mean_field(
            rows, "State/rear_any_airborne__mean"
        ),
        "rear_both_airborne_rate": _mean_field(
            rows, "State/rear_both_airborne__mean"
        ),
        "all_feet_airborne_rate": _mean_field(
            rows, "State/all_feet_airborne__mean"
        ),
        "rear_both_airborne_env_rate": _event_rate(
            rows, "State/rear_both_airborne", threshold=0.5
        ),
        "object_target_distance_last": _mean_field(
            rows, PREFIX + "object_target_distance__last"
        ),
        "object_target_distance_min": _mean_field(
            rows, PREFIX + "object_target_distance__min"
        ),
        "gripper_closure_mean": _mean_field(
            rows, PREFIX + "gripper_closure__mean"
        ),
        "task_switch_mean": _mean_field(
            rows, PREFIX + "task_switch__mean"
        ),
        "skill_switch_mean": _mean_field(
            rows, PREFIX + "skill_switch__mean"
        ),
        "actor_task_entropy": _mean_field(
            rows, "Actor/task_entropy__mean"
        ),
        "actor_skill_entropy": _mean_field(
            rows, "Actor/skill_entropy__mean"
        ),
        "actor_task_confidence": _mean_field(
            rows, "Actor/task_confidence__mean"
        ),
        "actor_skill_confidence": _mean_field(
            rows, "Actor/skill_confidence__mean"
        ),
        "actor_motion_entropy": _mean_field(
            rows, "Actor/motion_entropy__mean"
        ),
        "actor_interaction_entropy": _mean_field(
            rows, "Actor/interaction_entropy__mean"
        ),
        "actor_constraint_violation": _mean_field(
            rows, "Actor/constraint_violation_mean__mean"
        ),
        "actor_payload_risk_pressure": _mean_field(
            rows, "Actor/payload_risk_pressure__mean"
        ),
        "actor_payload_risk_correction": _mean_field(
            rows, "Actor/payload_risk_logit_correction__mean"
        ),
        "validation_score": score,
    }


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    paths = sorted(args.root.rglob("per_env.csv"))
    if not paths:
        raise FileNotFoundError(
            "No per_env.csv files under {}".format(args.root)
        )
    scenario_order_by_case = {}
    scenario_task_ids = {}
    manifest_path = args.root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        for case in manifest.get("cases", []):
            scenario_order_by_case[case.get("case", "")] = case.get(
                "scenario_names", []
            )
        for scenario_name, task_spec in manifest.get(
            "scenarios", {}
        ).items():
            scenario_task_ids[scenario_name] = tuple(
                int(value)
                for value in task_spec.split("+")
                if value
            )

    summaries = []
    set_id_key = "Diagnostic/required_task_set_id__last"
    for path in paths:
        rows = _read_rows(path)
        grouped_rows = defaultdict(list)
        for row in rows:
            set_id = _number(row, set_id_key)
            grouped_rows[
                int(set_id) if math.isfinite(set_id) else -1
            ].append(row)
        scenario_order = scenario_order_by_case.get(
            path.parent.name, []
        )
        for task_set_id, group in sorted(grouped_rows.items()):
            scenario_name = (
                scenario_order[task_set_id]
                if 0 <= task_set_id < len(scenario_order)
                else ""
            )
            summaries.append(
                _summarize(
                    path,
                    group,
                    task_set_id=task_set_id,
                    scenario_name=scenario_name,
                    required_task_ids=scenario_task_ids.get(
                        scenario_name, ()
                    ),
                )
            )
    _write_csv(args.output, summaries)

    aggregate_path = args.aggregate_output
    if aggregate_path is None:
        aggregate_path = args.output.with_name(
            args.output.stem + "_by_iteration.csv"
        )
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["iteration"]].append(row)
    aggregate = []
    for iteration, group in sorted(grouped.items()):
        aggregate.append(
            {
                "iteration": iteration,
                "screen_count": len(group),
                "validation_score": _mean(
                    [row["validation_score"] for row in group]
                ),
                "mission_success_rate": _mean(
                    [row["mission_success_rate"] for row in group]
                ),
                "contact_rate": _mean(
                    [row["contact_rate"] for row in group]
                ),
                "lift_rate": _mean(
                    [row["lift_rate"] for row in group]
                ),
                "carry_rate": _mean(
                    [row["carry_rate"] for row in group]
                ),
                "transport_rate": _mean(
                    [row["transport_rate"] for row in group]
                ),
                "place_rate": _mean(
                    [row["place_rate"] for row in group]
                ),
                "selected_target_progress_mean": _mean(
                    [row["selected_target_progress_mean"] for row in group]
                ),
                "release_readiness_mean": _mean(
                    [row["release_readiness_mean"] for row in group]
                ),
                "intended_release_env_rate": _mean(
                    [row["intended_release_env_rate"] for row in group]
                ),
                "drop_env_rate": _mean(
                    [row["drop_env_rate"] for row in group]
                ),
                "physical_contact_fraction": _mean(
                    [row["physical_contact_fraction"] for row in group]
                ),
                "physical_lift_fraction": _mean(
                    [row["physical_lift_fraction"] for row in group]
                ),
                "physical_carry_fraction": _mean(
                    [row["physical_carry_fraction"] for row in group]
                ),
                "physical_transport_fraction": _mean(
                    [row["physical_transport_fraction"] for row in group]
                ),
                "object_completion_fraction": _mean(
                    [row["object_completion_fraction"] for row in group]
                ),
                "object_completion_all_rate": _mean(
                    [row["object_completion_all_rate"] for row in group]
                ),
                "cbf_margin": _mean(
                    [row["cbf_margin"] for row in group]
                ),
                "tilt_failure_rate": _mean(
                    [row["tilt_failure_rate"] for row in group]
                ),
                "rear_both_airborne_rate": _mean(
                    [row["rear_both_airborne_rate"] for row in group]
                ),
                "base_xy_tracking_error": _mean(
                    [row["base_xy_tracking_error"] for row in group]
                ),
            }
        )
    _write_csv(aggregate_path, aggregate)
    print(
        "evaluation_summary={} aggregate={}".format(
            args.output,
            aggregate_path,
        )
    )


if __name__ == "__main__":
    main()
