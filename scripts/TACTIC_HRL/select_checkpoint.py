"""Select one shared TANDEM-HRL checkpoint from fixed validation screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HIGHER_IS_BETTER = (
    "validation_score",
    "object_completion_fraction",
    "physical_transport_fraction",
    "physical_lift_fraction",
    "cbf_margin",
)
LOWER_IS_BETTER = (
    "tilt_failure_rate",
    "rear_both_airborne_rate",
    "base_xy_tracking_error",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen_root",
        type=Path,
        required=True,
        help="Directory containing model_NNNN screen summaries.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run containing model_N.pt checkpoints.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_screens", type=int, default=4)
    parser.add_argument(
        "--min_iteration",
        type=int,
        default=0,
        help="Exclude diagnostic warm-start checkpoints before this iteration.",
    )
    parser.add_argument(
        "--min_object_completion_fraction",
        type=float,
        default=0.0,
        help="Required fixed-screen physical completion fraction.",
    )
    parser.add_argument(
        "--min_physical_transport_fraction",
        type=float,
        default=0.0,
        help="Required fixed-screen physical transport fraction.",
    )
    return parser.parse_args()


def _number(row, key, default):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_candidates(root, expected_screens, min_iteration):
    candidates = []
    for path in sorted(root.glob("model_*/final_aggregate_by_iteration.csv")):
        with path.open("r", encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        if len(rows) != 1:
            raise RuntimeError(
                "{} must contain exactly one aggregate row".format(path)
            )
        row = rows[0]
        screen_count = int(_number(row, "screen_count", -1))
        if screen_count != expected_screens:
            continue
        iteration = int(_number(row, "iteration", -1))
        if iteration < min_iteration:
            continue
        row["_source"] = str(path.resolve())
        candidates.append(row)
    if not candidates:
        raise FileNotFoundError(
            "No complete checkpoint screens under {}".format(root)
        )
    return candidates


def _selection_key(row):
    higher = tuple(
        _number(row, key, -math.inf) for key in HIGHER_IS_BETTER
    )
    lower = tuple(
        -_number(row, key, math.inf) for key in LOWER_IS_BETTER
    )
    iteration = int(_number(row, "iteration", -1))
    return higher + lower + (iteration,)


def main():
    args = _parse_args()
    if args.expected_screens < 1:
        raise ValueError("--expected_screens must be positive")
    if args.min_iteration < 0:
        raise ValueError("--min_iteration cannot be negative")
    if args.min_object_completion_fraction < 0.0:
        raise ValueError(
            "--min_object_completion_fraction cannot be negative"
        )
    if args.min_physical_transport_fraction < 0.0:
        raise ValueError(
            "--min_physical_transport_fraction cannot be negative"
        )
    candidates = _read_candidates(
        args.screen_root,
        args.expected_screens,
        args.min_iteration,
    )
    ranked = sorted(candidates, key=_selection_key, reverse=True)
    eligible = [
        row
        for row in ranked
        if _number(
            row,
            "object_completion_fraction",
            -math.inf,
        )
        >= args.min_object_completion_fraction
        and _number(
            row,
            "physical_transport_fraction",
            -math.inf,
        )
        >= args.min_physical_transport_fraction
    ]
    acceptance_passed = bool(eligible)
    selected = eligible[0] if eligible else ranked[0]
    iteration = int(_number(selected, "iteration", -1))
    checkpoint = args.run_dir / "model_{}.pt".format(iteration)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    serializable = []
    for rank, row in enumerate(ranked, 1):
        serializable.append(
            {
                "rank": rank,
                "eligible": int(row in eligible),
                **{
                    key: row.get(key, "")
                    for key in (
                        "iteration",
                        "screen_count",
                        *HIGHER_IS_BETTER,
                        *LOWER_IS_BETTER,
                    )
                },
                "aggregate_source": row["_source"],
            }
        )
    manifest = {
        "method": "TANDEM-HRL",
        "selection_split": "four fixed checkpoint screens",
        "formal_test_used_for_selection": False,
        "ordering": {
            "higher_is_better": list(HIGHER_IS_BETTER),
            "lower_is_better": list(LOWER_IS_BETTER),
            "final_tie_breaker": "later iteration",
        },
        "expected_screens": args.expected_screens,
        "minimum_candidate_iteration": args.min_iteration,
        "acceptance_thresholds": {
            "object_completion_fraction": (
                args.min_object_completion_fraction
            ),
            "physical_transport_fraction": (
                args.min_physical_transport_fraction
            ),
        },
        "acceptance_passed": acceptance_passed,
        "selected_iteration": iteration,
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": _sha256(checkpoint),
        "ranking": serializable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        "selected_checkpoint={} sha256={}".format(
            checkpoint,
            manifest["selected_checkpoint_sha256"],
        )
    )
    if not acceptance_passed:
        raise RuntimeError(
            "No checkpoint met the fixed-screen task-capability thresholds"
        )


if __name__ == "__main__":
    main()
