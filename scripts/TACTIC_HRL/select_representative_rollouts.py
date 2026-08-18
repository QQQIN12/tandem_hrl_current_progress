"""Select paired formal-evaluation rollouts for qualitative figures.

The selector never changes the quantitative result set.  It identifies an
event-rich TANDEM-HRL environment from the completed formal cases, then pairs
it with the baseline environment having the same seed and environment index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PREFIX = "Command/locomotion/TACTIC/"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrl_root", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_tsv", type=Path, default=None)
    parser.add_argument(
        "--scenarios",
        default=(
            "mobile_single_delivery,triple_object_delivery,"
            "shape_diverse_delivery,grand_mission"
        ),
    )
    parser.add_argument(
        "--regime",
        choices=("nominal", "stress"),
        default="nominal",
    )
    return parser.parse_args()


def _number(row, key, default=math.nan):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _event(row, object_id, name, threshold=0.5):
    value = _number(
        row,
        "Diagnostic/TACTIC/object_{}/{}__max".format(
            object_id,
            name,
        ),
        default=-math.inf,
    )
    return float(value >= threshold)


def _fraction(row, object_ids, name, threshold=0.5):
    if not object_ids:
        return math.nan
    return sum(
        _event(row, object_id, name, threshold)
        for object_id in object_ids
    ) / len(object_ids)


def _load_manifest(root):
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _case_rows(root, manifest, regime):
    records = []
    for case in manifest.get("cases", []):
        if case.get("regime") != regime:
            continue
        if case.get("ablation", "none") != "none":
            continue
        case_name = case.get("case", "")
        path = root / "cases" / case_name / "per_env.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        scenario_names = case.get("scenario_names", [])
        for row in _read_csv(path):
            task_set_id = int(
                _number(
                    row,
                    "Diagnostic/required_task_set_id__last",
                    default=-1,
                )
            )
            if not 0 <= task_set_id < len(scenario_names):
                continue
            records.append(
                {
                    "case_name": case_name,
                    "scenario": scenario_names[task_set_id],
                    "seed": int(case["seed"]),
                    "env_id": int(row["env_id"]),
                    "row": row,
                }
            )
    return records


def _candidate_metrics(row, object_ids):
    mission_success = float(
        _number(
            row,
            PREFIX + "mission_completion__max",
            default=0.0,
        )
        >= 0.99
    )
    metrics = {
        "mission_success": mission_success,
        "completion_fraction": _fraction(
            row, object_ids, "object_completion"
        ),
        "transport_fraction": _fraction(
            row, object_ids, "object_transport_memory"
        ),
        "carry_fraction": _fraction(
            row, object_ids, "object_carrying"
        ),
        "lift_fraction": _fraction(
            row, object_ids, "object_lift_memory", threshold=0.20
        ),
        "contact_fraction": _fraction(
            row, object_ids, "object_contact_memory"
        ),
        "drop_rate": _number(
            row,
            PREFIX + "selected_object_drop_event__mean",
            default=1.0,
        ),
        "safety_failure_rate": _number(
            row, "Safety/cbf_violation__mean", default=1.0
        ),
        "tilt_failure_rate": _number(
            row, "Termination/tilt__mean", default=1.0
        ),
        "task_error": _number(
            row, PREFIX + "task_error__mean", default=math.inf
        ),
        "tracking_error": _number(
            row, "Tracking/base_xy_error__mean", default=math.inf
        ),
    }
    if not object_ids:
        for name in (
            "completion_fraction",
            "transport_fraction",
            "carry_fraction",
            "lift_fraction",
            "contact_fraction",
        ):
            metrics[name] = 0.0
    return metrics


def _rank_key(candidate):
    metrics = candidate["metrics"]
    return (
        metrics["mission_success"],
        metrics["completion_fraction"],
        metrics["transport_fraction"],
        metrics["carry_fraction"],
        metrics["lift_fraction"],
        metrics["contact_fraction"],
        -metrics["drop_rate"],
        -metrics["safety_failure_rate"],
        -metrics["tilt_failure_rate"],
        -metrics["task_error"],
        -metrics["tracking_error"],
        -candidate["seed"],
        -candidate["env_id"],
    )


def _write_rows(path, rows, delimiter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    scenarios = [
        value.strip()
        for value in args.scenarios.split(",")
        if value.strip()
    ]
    hrl_manifest = _load_manifest(args.hrl_root)
    baseline_manifest = _load_manifest(args.baseline_root)
    scenario_specs = hrl_manifest.get("scenarios", {})
    if scenario_specs != baseline_manifest.get("scenarios", {}):
        raise RuntimeError("HRL and baseline scenario manifests differ")

    hrl_records = _case_rows(
        args.hrl_root, hrl_manifest, args.regime
    )
    baseline_records = _case_rows(
        args.baseline_root, baseline_manifest, args.regime
    )
    baseline_lookup = {
        (
            record["scenario"],
            record["seed"],
            record["env_id"],
        ): record
        for record in baseline_records
    }

    output_rows = []
    for scenario in scenarios:
        if scenario not in scenario_specs:
            raise ValueError("Unknown scenario: {}".format(scenario))
        task_ids = [
            int(value)
            for value in scenario_specs[scenario].split("+")
            if value
        ]
        object_ids = sorted(
            {task_id - 5 for task_id in task_ids if 5 <= task_id <= 10}
        )
        candidates = []
        for record in hrl_records:
            if record["scenario"] != scenario:
                continue
            candidates.append(
                {
                    **record,
                    "metrics": _candidate_metrics(
                        record["row"], object_ids
                    ),
                }
            )
        if not candidates:
            raise RuntimeError(
                "No completed HRL cases for {}".format(scenario)
            )
        selected = max(candidates, key=_rank_key)
        pair_key = (
            scenario,
            selected["seed"],
            selected["env_id"],
        )
        if pair_key not in baseline_lookup:
            raise RuntimeError(
                "Missing paired baseline row for {}".format(pair_key)
            )
        baseline = baseline_lookup[pair_key]
        baseline_metrics = _candidate_metrics(
            baseline["row"], object_ids
        )
        output_rows.append(
            {
                "scenario": scenario,
                "regime": args.regime,
                "seed": selected["seed"],
                "env_id": selected["env_id"],
                "case_name": selected["case_name"],
                "required_tasks": scenario_specs[scenario],
                "required_objects": "+".join(
                    str(value) for value in object_ids
                ),
                **{
                    "hrl_" + key: value
                    for key, value in selected["metrics"].items()
                },
                **{
                    "baseline_" + key: value
                    for key, value in baseline_metrics.items()
                },
                "selection_rule": (
                    "formal event-chain lexicographic; paired seed/env"
                ),
            }
        )

    if not output_rows:
        raise RuntimeError("No representative rollouts selected")
    _write_rows(args.output_csv, output_rows, delimiter=",")
    output_tsv = (
        args.output_tsv
        if args.output_tsv is not None
        else args.output_csv.with_suffix(".tsv")
    )
    _write_rows(output_tsv, output_rows, delimiter="\t")
    print(
        "representative_csv={} representative_tsv={}".format(
            args.output_csv,
            output_tsv,
        )
    )


if __name__ == "__main__":
    main()
