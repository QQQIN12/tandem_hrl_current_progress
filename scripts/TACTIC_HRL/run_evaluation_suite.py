"""Run the registered TANDEM-HRL evaluation and ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCENARIOS = {
    "navigation_recovery": "0+1+2+4+11",
    "mobile_single_delivery": "0+3+5+11",
    "dual_object_delivery": "0+1+5+7+11",
    "triple_object_delivery": "0+1+2+5+7+9+11",
    "shape_diverse_delivery": "0+2+3+6+8+10+11",
    "grand_mission": "0+1+2+3+4+5+6+7+8+9+10+11",
    "heldout_composition_a": "1+3+6+9+11",
    "heldout_composition_b": "2+4+5+8+10+11",
}

HARD_SCENARIOS = (
    "triple_object_delivery",
    "shape_diverse_delivery",
    "grand_mission",
)

ABLATIONS = (
    "fixed_task",
    "fixed_skill",
    "no_relational_state",
    "no_predictive_models",
    "no_control_objective",
    "no_payload_option_barrier",
    "no_payload_relation_authority",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one shared TANDEM-HRL checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--zyb_baseline_checkpoint",
        type=Path,
        default=None,
        help=(
            "Use the original ZYB-v0 physical checkpoint with the fixed "
            "task/skill dispatcher; --checkpoint remains the TANDEM shell."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable that owns the Isaac Lab environment.",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).parents[1] / "rsl_rl" / "evaluate_checkpoint.py",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--envs_per_scenario",
        type=int,
        default=12,
        help="Parallel replicas assigned to each registered scenario.",
    )
    parser.add_argument("--num_steps", type=int, default=1800)
    parser.add_argument(
        "--seeds",
        default="42,73,101",
        help="Comma-separated evaluation seeds.",
    )
    parser.add_argument(
        "--regimes",
        default="nominal,stress",
        help="Comma-separated regimes: nominal and/or stress.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated registered scenario names.",
    )
    parser.add_argument("--include_ablations", action="store_true")
    parser.add_argument(
        "--ablations_only",
        action="store_true",
        help="Skip full-model cases and run only the requested ablation shard.",
    )
    parser.add_argument(
        "--ablations",
        default=",".join(ABLATIONS),
        help=(
            "Comma-separated ablations used with --include_ablations. "
            "This permits disjoint evaluation shards on multiple GPUs."
        ),
    )
    parser.add_argument(
        "--actor_diagnostics",
        action="store_true",
        help="Record policy-internal task, skill, and constraint metrics.",
    )
    parser.add_argument(
        "--actor_diagnostics_stride",
        type=int,
        default=1,
        help="Sample policy-internal diagnostics every N simulation steps.",
    )
    parser.add_argument("--trace_stride", type=int, default=12)
    parser.add_argument(
        "--full_metrics",
        action="store_true",
        help="Disable compact evidence-only metric collection.",
    )
    parser.add_argument("--fixed_skill_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _parse_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_case(
    args,
    scenario_names,
    regime,
    seed,
    ablation="none",
):
    task_name = (
        "TANDEM-HRL-Stress-v0"
        if regime == "stress"
        else "TANDEM-HRL-Play-v0"
    )
    case_name = "{}scenarios__{}__seed{:03d}__{}".format(
        len(scenario_names),
        regime,
        seed,
        ablation,
    )
    case_dir = args.output_dir / "cases" / case_name
    summary_path = case_dir / "summary.csv"
    if args.resume and summary_path.exists():
        return {
            "case": case_name,
            "scenario_names": list(scenario_names),
            "regime": regime,
            "seed": seed,
            "ablation": ablation,
            "status": "skipped",
            "summary": str(summary_path),
        }

    case_dir.mkdir(parents=True, exist_ok=True)
    method = (
        "ZYB-v0+Fixed-Dispatcher"
        if args.zyb_baseline_checkpoint is not None
        else "TANDEM-HRL"
    )
    if ablation != "none":
        method += "[{}]".format(ablation)
    required_tasks = ";".join(
        SCENARIOS[name] for name in scenario_names
    )
    num_envs = args.envs_per_scenario * len(scenario_names)
    command = [
        args.python,
        str(args.evaluator),
        "--task",
        task_name,
        "--checkpoint",
        str(args.checkpoint),
        "--num_envs",
        str(num_envs),
        "--num_steps",
        str(args.num_steps),
        "--required_task_sets",
        required_tasks,
        "--force_curriculum_level",
        "4",
        "--seed",
        str(seed),
        "--ablation",
        ablation,
        "--fixed_skill_id",
        str(args.fixed_skill_id),
        "--scenario",
        "batched:{}".format(regime),
        "--method",
        method,
        "--out_csv",
        str(summary_path),
        "--per_env_csv",
        str(case_dir / "per_env.csv"),
        "--trace_csv",
        str(case_dir / "trace_env0.csv"),
        "--trace_stride",
        str(args.trace_stride),
        "--deterministic_reset",
        "--compact_metrics",
        "--headless",
        "--device",
        args.device,
    ]
    if args.zyb_baseline_checkpoint is not None:
        command.extend(
            (
                "--zyb_baseline_checkpoint",
                str(args.zyb_baseline_checkpoint),
            )
        )
    if args.actor_diagnostics:
        command.append("--actor_diagnostics")
        command.extend(
            (
                "--actor_diagnostics_stride",
                str(args.actor_diagnostics_stride),
            )
        )
    if args.full_metrics:
        command.remove("--compact_metrics")
    log_path = case_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMNI_KIT_ACCEPT_EULA": "YES",
            },
        )
    return {
        "case": case_name,
        "scenario_names": list(scenario_names),
        "regime": regime,
        "seed": seed,
        "ablation": ablation,
        "status": "completed",
        "summary": str(summary_path),
        "log": str(log_path),
    }


def main():
    args = _parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if (
        args.zyb_baseline_checkpoint is not None
        and not args.zyb_baseline_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.zyb_baseline_checkpoint)
    if not args.evaluator.is_file():
        raise FileNotFoundError(args.evaluator)
    if args.fixed_skill_id < 0 or args.fixed_skill_id >= 12:
        raise ValueError("--fixed_skill_id must be between 0 and 11")
    if args.envs_per_scenario < 1:
        raise ValueError("--envs_per_scenario must be positive")
    if args.trace_stride < 1:
        raise ValueError("--trace_stride must be positive")
    if args.actor_diagnostics_stride < 1:
        raise ValueError("--actor_diagnostics_stride must be positive")
    if args.zyb_baseline_checkpoint is not None and args.include_ablations:
        raise ValueError(
            "Fixed-dispatcher baseline does not define TANDEM-HRL ablations"
        )
    if args.ablations_only and not args.include_ablations:
        raise ValueError("--ablations_only requires --include_ablations")

    seeds = [int(value) for value in _parse_list(args.seeds)]
    regimes = _parse_list(args.regimes)
    invalid_regimes = sorted(set(regimes) - {"nominal", "stress"})
    if invalid_regimes:
        raise ValueError("Unknown regimes: {}".format(invalid_regimes))
    scenario_names = _parse_list(args.scenarios)
    invalid_scenarios = sorted(set(scenario_names) - set(SCENARIOS))
    if invalid_scenarios:
        raise ValueError("Unknown scenarios: {}".format(invalid_scenarios))
    ablations = _parse_list(args.ablations)
    invalid_ablations = sorted(set(ablations) - set(ABLATIONS))
    if invalid_ablations:
        raise ValueError("Unknown ablations: {}".format(invalid_ablations))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "zyb_baseline_checkpoint": (
            str(args.zyb_baseline_checkpoint.resolve())
            if args.zyb_baseline_checkpoint is not None
            else None
        ),
        "envs_per_scenario": args.envs_per_scenario,
        "num_steps": args.num_steps,
        "seeds": seeds,
        "regimes": regimes,
        "scenarios": {
            name: SCENARIOS[name] for name in scenario_names
        },
        "task_set_id_by_scenario": {
            name: index for index, name in enumerate(scenario_names)
        },
        "fixed_skill_id": args.fixed_skill_id,
        "actor_diagnostics": args.actor_diagnostics,
        "actor_diagnostics_stride": args.actor_diagnostics_stride,
        "trace_stride": args.trace_stride,
        "full_metrics": args.full_metrics,
        "ablations": ablations if args.include_ablations else [],
        "ablations_only": args.ablations_only,
        "cases": [],
    }
    manifest_path = args.output_dir / "manifest.json"

    if not args.ablations_only:
        for regime in regimes:
            for seed in seeds:
                result = _run_case(
                    args,
                    scenario_names,
                    regime,
                    seed,
                )
                manifest["cases"].append(result)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )

    if args.include_ablations:
        hard_scenarios = [
            scenario
            for scenario in HARD_SCENARIOS
            if scenario in scenario_names
        ]
        if not hard_scenarios:
            raise ValueError(
                "--include_ablations requires at least one hard scenario"
            )
        for ablation in ablations:
            for seed in seeds:
                result = _run_case(
                    args,
                    hard_scenarios,
                    "stress",
                    seed,
                    ablation=ablation,
                )
                manifest["cases"].append(result)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )

    print("evaluation_manifest={}".format(manifest_path))


if __name__ == "__main__":
    main()
