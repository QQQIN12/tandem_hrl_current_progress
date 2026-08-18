"""Screen selected checkpoints while a TANDEM-HRL run is training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCREENS = (
    {
        "name": "navigation_recovery",
        "task": "TANDEM-HRL-Play-v0",
        "tasks_argument": "--required_task_sets",
        "tasks": "0+1+2+4+11",
        "steps": 900,
    },
    {
        "name": "single_delivery_slot9",
        "task": "TANDEM-HRL-Play-v0",
        "tasks_argument": "--required_task_grid",
        "tasks": "9",
        "steps": 1800,
    },
    {
        "name": "triple_object_delivery",
        "task": "TANDEM-HRL-Play-v0",
        "tasks_argument": "--required_task_sets",
        "tasks": "0+1+2+5+7+9+11",
        "steps": 2700,
    },
    {
        "name": "grand_mission_stress",
        "task": "TANDEM-HRL-Stress-v0",
        "tasks_argument": "--required_task_sets",
        "tasks": "0+1+2+3+4+5+6+7+8+9+10+11",
        "steps": 3600,
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Wait for and evaluate selected TANDEM-HRL checkpoints."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        default="256,512,768,1023",
        help="Comma-separated model iteration numbers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=24)
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).parents[1] / "rsl_rl" / "evaluate_checkpoint.py",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _wait_for_checkpoint(path, poll_seconds):
    while not path.is_file() or path.stat().st_size == 0:
        time.sleep(poll_seconds)
    previous_size = path.stat().st_size
    while True:
        time.sleep(2.0)
        current_size = path.stat().st_size
        if current_size == previous_size:
            return
        previous_size = current_size


def _run_screen(args, checkpoint, iteration, screen):
    case_dir = args.output_dir / "model_{:04d}".format(iteration) / screen["name"]
    summary_path = case_dir / "summary.csv"
    if args.resume and summary_path.is_file():
        return {
            "iteration": iteration,
            "screen": screen["name"],
            "status": "skipped",
            "summary": str(summary_path),
        }

    case_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(args.evaluator),
        "--task",
        screen["task"],
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.num_envs),
        "--num_steps",
        str(screen["steps"]),
        screen["tasks_argument"],
        screen["tasks"],
        "--force_curriculum_level",
        "4",
        "--seed",
        str(args.seed),
        "--deterministic_reset",
        "--actor_diagnostics",
        "--scenario",
        screen["name"],
        "--method",
        "TANDEM_model_{:04d}".format(iteration),
        "--out_csv",
        str(summary_path),
        "--per_env_csv",
        str(case_dir / "per_env.csv"),
        "--headless",
        "--device",
        args.device,
    ]
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
        "iteration": iteration,
        "screen": screen["name"],
        "status": "completed",
        "summary": str(summary_path),
        "per_env": str(case_dir / "per_env.csv"),
        "log": str(log_path),
    }


def main():
    args = _parse_args()
    iterations = [
        int(value.strip())
        for value in args.iterations.split(",")
        if value.strip()
    ]
    if not iterations:
        raise ValueError("--iterations cannot be empty")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(args.run_dir)
    if not args.evaluator.is_file():
        raise FileNotFoundError(args.evaluator)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": str(args.run_dir.resolve()),
        "iterations": iterations,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "screens": SCREENS,
        "cases": [],
    }
    manifest_path = args.output_dir / "manifest.json"

    for iteration in iterations:
        checkpoint = args.run_dir / "model_{}.pt".format(iteration)
        _wait_for_checkpoint(checkpoint, args.poll_seconds)
        for screen in SCREENS:
            result = _run_screen(
                args,
                checkpoint,
                iteration,
                screen,
            )
            manifest["cases"].append(result)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

    print("checkpoint_screen_manifest={}".format(manifest_path))


if __name__ == "__main__":
    main()
