"""Plot system and hierarchy traces aligned with physical keyframes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFIX = "Command/locomotion/TACTIC/"
METHODS = (
    ("TANDEM-HRL", "#0072B2", "-"),
    ("ZYB-v0 + fixed dispatcher", "#D55E00", "--"),
)
EVENT_COLORS = {
    "contact": "#009E73",
    "lift": "#56B4E9",
    "transport": "#CC79A7",
    "release": "#E69F00",
    "completion": "#000000",
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrl_trace", type=Path, required=True)
    parser.add_argument("--baseline_trace", type=Path, required=True)
    parser.add_argument("--hrl_alignment", type=Path, default=None)
    parser.add_argument("--baseline_alignment", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _number(row, key, default=math.nan):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read(path):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _series(rows, key):
    return np.asarray([_number(row, key) for row in rows], dtype=float)


def _object_event(rows, suffix):
    result = []
    for row in rows:
        values = [
            _number(row, key)
            for key in row
            if key.startswith("Diagnostic/TACTIC/object_")
            and key.endswith(suffix)
        ]
        values = [value for value in values if math.isfinite(value)]
        result.append(max(values) if values else math.nan)
    return np.asarray(result, dtype=float)


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.55,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _events(path):
    if path is None or not path.is_file():
        return []
    result = []
    for row in _read(path):
        if (
            row.get("event") in EVENT_COLORS
            and row.get("available") == "1"
        ):
            result.append(
                (
                    row["event"],
                    _number(row, "sim_time_s"),
                )
            )
    return result


def _annotate_events(axes, events, method_prefix, linestyle, at_top):
    for event, time_s in events:
        if not math.isfinite(time_s):
            continue
        color = EVENT_COLORS[event]
        for axis in axes:
            axis.axvline(
                time_s,
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.70 if at_top else 0.48,
                zorder=0,
            )
        label_axis = axes[0] if at_top else axes[-1]
        y = 1.0 if at_top else 0.0
        y_offset = -2 if at_top else 2
        label_axis.annotate(
            "{}:{}".format(method_prefix, event),
            xy=(time_s, y),
            xycoords=("data", "axes fraction"),
            xytext=(2, y_offset),
            textcoords="offset points",
            rotation=90,
            va="top" if at_top else "bottom",
            ha="left",
            color=color,
            fontsize=7.0,
        )


def _plot_method(axes, rows, label, color, linestyle):
    time_s = _series(rows, "sim_time_s")
    task_error = _series(rows, PREFIX + "task_error")
    target_distance = _series(rows, PREFIX + "object_target_distance")
    axes[0].plot(
        time_s, task_error, color=color, linestyle=linestyle, label=label
    )
    if np.isfinite(target_distance).any():
        axes[0].plot(
            time_s,
            target_distance,
            color=color,
            linestyle=":" if linestyle == "-" else "-.",
            alpha=0.75,
        )

    axes[1].plot(
        time_s,
        _series(rows, "Tracking/base_xy_error"),
        color=color,
        linestyle=linestyle,
        label=label,
    )
    axes[2].plot(
        time_s,
        _series(rows, PREFIX + "cbf_margin"),
        color=color,
        linestyle=linestyle,
        label=label + " CBF",
    )
    axes[2].plot(
        time_s,
        _series(rows, PREFIX + "predicted_margin"),
        color=color,
        linestyle=":" if linestyle == "-" else "-.",
        alpha=0.75,
        label=label + " preview",
    )
    axes[3].plot(
        time_s,
        _series(rows, PREFIX + "base_tilt"),
        color=color,
        linestyle=linestyle,
        label=label + " tilt",
    )
    axes[3].plot(
        time_s,
        _series(rows, "State/rear_both_airborne"),
        color=color,
        linestyle=":" if linestyle == "-" else "-.",
        alpha=0.75,
        label=label + " rear airborne",
    )

    stage_colors = (
        "#009E73",
        "#56B4E9",
        "#CC79A7",
        "#E69F00",
        "#000000",
    )
    stage_names = (
        "Contact",
        "Lift",
        "Transport",
        "Release readiness",
        "Completion",
    )
    stage_suffixes = (
        "/object_contact_memory",
        "/object_lift_memory",
        "/object_transport_memory",
        "/object_release_readiness",
        "/object_completion",
    )
    for stage, suffix, stage_color in zip(
        stage_names, stage_suffixes, stage_colors
    ):
        axes[4].plot(
            time_s,
            _object_event(rows, suffix),
            color=stage_color,
            linestyle=linestyle,
            alpha=1.0 if label == METHODS[0][0] else 0.48,
            label=(
                stage
                if label == METHODS[0][0]
                else stage + " baseline"
            ),
        )

    if label == METHODS[0][0]:
        axes[6].step(
            time_s,
            _series(rows, PREFIX + "task_id"),
            where="post",
            color="#0072B2",
            label="Task id",
        )
        axes[6].step(
            time_s,
            _series(rows, PREFIX + "skill_id"),
            where="post",
            color="#D55E00",
            alpha=0.85,
            label="Skill id",
        )
    axes[5].plot(
        time_s,
        _series(rows, PREFIX + "selected_object_release_readiness"),
        color=color,
        linestyle=linestyle,
        label=label + " readiness",
    )
    axes[5].step(
        time_s,
        _series(rows, PREFIX + "selected_object_release_event"),
        where="post",
        color=color,
        linestyle=":" if linestyle == "-" else "-.",
        alpha=0.85,
        label=label + " release",
    )
    axes[5].step(
        time_s,
        _series(rows, PREFIX + "selected_object_drop_event"),
        where="post",
        color="#7A7A7A",
        linestyle=linestyle,
        alpha=0.65,
        label=label + " drop",
    )


def main():
    args = _parse_args()
    traces = (_read(args.hrl_trace), _read(args.baseline_trace))
    if not all(traces):
        raise RuntimeError("Trace CSV files must be non-empty")
    _style()
    figure, axes = plt.subplots(
        7,
        1,
        figsize=(9.3, 11.8),
        sharex=True,
        constrained_layout=True,
    )
    for rows, method in zip(traces, METHODS):
        _plot_method(axes, rows, *method)

    axes[0].set_ylabel("Task / target\nerror")
    axes[1].set_ylabel("Base tracking\nerror (m/s)")
    axes[2].set_ylabel("Safety margin")
    axes[3].set_ylabel("Tilt / rear\nairborne")
    axes[4].set_ylabel("Physical event\nstate")
    axes[5].set_ylabel("Release\nstate")
    axes[6].set_ylabel("Option id")
    axes[6].set_xlabel("Simulation time (s)")
    axes[4].set_ylim(-0.03, 1.03)
    axes[5].set_ylim(-0.03, 1.03)
    axes[6].set_ylim(-0.5, 11.5)

    for axis in axes:
        axis.grid(color="#D8D8D8", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].legend(frameon=False, ncol=2)
    axes[2].legend(frameon=False, ncol=2)
    axes[3].legend(frameon=False, ncol=2)
    axes[4].legend(frameon=False, ncol=5)
    axes[5].legend(frameon=False, ncol=3)
    axes[6].legend(frameon=False, ncol=2)

    _annotate_events(
        axes,
        _events(args.hrl_alignment),
        method_prefix="T",
        linestyle="-",
        at_top=True,
    )
    _annotate_events(
        axes,
        _events(args.baseline_alignment),
        method_prefix="B",
        linestyle="--",
        at_top=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)
    print("trace_figure={}".format(args.output))


if __name__ == "__main__":
    main()
