#!/usr/bin/env python3
"""Parse LR_HRL training logs and generate comparison curves."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ["route", "slalom", "narrow", "manip", "grasp", "recovery"]

METRICS = {
    "mean_reward": "Mean reward",
    "mean_episode_length": "Episode length",
    "Episode_Reward/LR_HRL_route_progress": "Route progress reward",
    "Episode_Reward/LR_HRL_goal_tracking": "Goal tracking reward",
    "Episode_Reward/LR_HRL_yaw_alignment": "Yaw alignment reward",
    "Episode_Reward/LR_HRL_ee_tracking": "EE tracking reward",
    "Episode_Reward/LR_HRL_stability_margin": "Stability reward",
    "Episode_Reward/LR_HRL_obstacle_clearance": "Obstacle clearance reward",
    "Episode_Reward/LR_HRL_grasp_proxy": "Grasp proxy reward",
    "Episode_Termination/tilt": "Tilt termination",
    "Episode_Termination/bad_contact": "Bad-contact termination",
    "Episode_Termination/low_height": "Low-height termination",
}


def _clean_line(line: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", line).strip()


def parse_log(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    iter_re = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
    scalar_re = re.compile(r"([A-Za-z_]+/[A-Za-z0-9_./-]+):\s*([-+0-9.eE]+)")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = _clean_line(raw)
            m_iter = iter_re.search(line)
            if m_iter:
                if current is not None:
                    records.append(current)
                current = {
                    "iteration": float(m_iter.group(1)),
                    "max_iteration": float(m_iter.group(2)),
                }
                continue

            if current is None:
                continue

            if line.startswith("Mean reward:"):
                current["mean_reward"] = float(line.split(":", 1)[1].strip())
                continue
            if line.startswith("Mean episode length:"):
                current["mean_episode_length"] = float(line.split(":", 1)[1].strip())
                continue
            if line.startswith("Mean action noise std:"):
                current["mean_action_noise_std"] = float(line.split(":", 1)[1].strip())
                continue

            m_scalar = scalar_re.search(line)
            if m_scalar:
                current[m_scalar.group(1)] = float(m_scalar.group(2))

    if current is not None:
        records.append(current)
    return records


def infer_run(path: Path) -> tuple[str, str]:
    name = path.stem
    method = "LR_HRL" if name.startswith("LR_HRL_") else "LR_Baseline"
    family = "unknown"
    for item in FAMILIES:
        if f"_{item}_" in name:
            family = item
            break
    return method, family


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    if values.size < 3:
        return values
    window = min(window, max(3, values.size // 8))
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(data: dict[tuple[str, str], list[dict[str, float]]], out_dir: Path, metric: str):
    colors = {"LR_HRL": "#0072B2", "LR_Baseline": "#D55E00"}
    for family in FAMILIES:
        fig, ax = plt.subplots(figsize=(4.3, 2.9))
        has_data = False
        for method in ("LR_Baseline", "LR_HRL"):
            records = data.get((method, family), [])
            xs = np.array([r.get("iteration", np.nan) for r in records], dtype=float)
            ys = np.array([r.get(metric, np.nan) for r in records], dtype=float)
            mask = ~(np.isnan(xs) | np.isnan(ys))
            if not np.any(mask):
                continue
            has_data = True
            xs = xs[mask]
            ys = ys[mask]
            ax.plot(xs, smooth(ys), color=colors[method], label=method.replace("_", "-"))
            ax.plot(xs, ys, color=colors[method], alpha=0.16, linewidth=0.7)
        if not has_data:
            plt.close(fig)
            continue
        ax.set_title(f"{family.capitalize()} | {METRICS.get(metric, metric)}")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(METRICS.get(metric, metric))
        ax.legend(frameon=False)
        fig.tight_layout()
        safe_metric = metric.replace("/", "__").replace(" ", "_")
        fig.savefig(out_dir / f"{family}_{safe_metric}.png")
        fig.savefig(out_dir / f"{family}_{safe_metric}.pdf")
        plt.close(fig)


def final_mean(records: list[dict[str, float]], metric: str, tail: int = 50) -> float:
    vals = [r[metric] for r in records[-tail:] if metric in r]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def write_csv(data: dict[tuple[str, str], list[dict[str, float]]], out_dir: Path):
    all_keys = {"method", "family"}
    for records in data.values():
        for r in records:
            all_keys.update(r.keys())
    keys = ["method", "family", "iteration"] + sorted(k for k in all_keys if k not in {"method", "family", "iteration"})
    with (out_dir / "LR_HRL_all_training_scalars.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for (method, family), records in sorted(data.items()):
            for r in records:
                row = {"method": method, "family": family}
                row.update(r)
                writer.writerow(row)


def write_markdown(data: dict[tuple[str, str], list[dict[str, float]]], out_dir: Path):
    rows = []
    for family in FAMILIES:
        for metric in [
            "mean_reward",
            "Episode_Reward/LR_HRL_route_progress",
            "Episode_Reward/LR_HRL_goal_tracking",
            "Episode_Reward/LR_HRL_ee_tracking",
            "Episode_Reward/LR_HRL_stability_margin",
            "Episode_Termination/tilt",
        ]:
            base = final_mean(data.get(("LR_Baseline", family), []), metric)
            hrl = final_mean(data.get(("LR_HRL", family), []), metric)
            gain = hrl - base if not (np.isnan(base) or np.isnan(hrl)) else float("nan")
            rows.append((family, METRICS.get(metric, metric), base, hrl, gain))

    lines = [
        "# LR_HRL Training Summary",
        "",
        "Values are tail means over the last available 50 training iterations. Positive gain means LR_HRL is higher than the flat baseline for that scalar.",
        "",
        "| Task | Metric | Baseline | LR_HRL | Gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, metric, base, hrl, gain in rows:
        fmt = lambda x: "n/a" if np.isnan(x) else f"{x:.4f}"
        lines.append(f"| {family} | {metric} | {fmt(base)} | {fmt(hrl)} | {fmt(gain)} |")
    lines.append("")
    lines.append("Generated files:")
    lines.append("- `LR_HRL_all_training_scalars.csv`: parsed scalar table.")
    lines.append("- `curves/*.png` and `curves/*.pdf`: comparison curves in publication-style formatting.")
    (out_dir / "LR_HRL_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/gpufree-data/quadruped_arm_LR_HRL"))
    parser.add_argument("--queue", type=str, default=None, help="Queue directory name under logs, e.g. LR_HRL_queue_...")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log_root = args.root / "logs"
    if args.queue is None:
        queues = sorted(log_root.glob("LR_HRL_queue_*"), key=lambda p: p.stat().st_mtime)
        if not queues:
            raise SystemExit("No LR_HRL_queue_* directory found.")
        queue_dir = queues[-1]
    else:
        queue_dir = log_root / args.queue

    out_dir = args.out or (args.root / "results" / queue_dir.name)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    data: dict[tuple[str, str], list[dict[str, float]]] = {}
    for log_path in sorted(queue_dir.glob("LR_*_1024_*.log")):
        method, family = infer_run(log_path)
        records = parse_log(log_path)
        if records:
            data[(method, family)] = records

    if not data:
        raise SystemExit(f"No parsed LR_HRL records in {queue_dir}")

    setup_style()
    for metric in METRICS:
        plot_metric(data, curves_dir, metric)
    write_csv(data, out_dir)
    write_markdown(data, out_dir)
    print(f"Wrote LR_HRL results to {out_dir}")


if __name__ == "__main__":
    main()
