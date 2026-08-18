"""Create publication figures from TANDEM-HRL evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIO_LABELS = {
    "navigation_recovery": "Navigation\nrecovery",
    "mobile_single_delivery": "Single-object\ndelivery",
    "dual_object_delivery": "Dual-object\ndelivery",
    "triple_object_delivery": "Triple-object\ndelivery",
    "shape_diverse_delivery": "Shape-diverse\ndelivery",
    "grand_mission": "Grand\nmission",
    "heldout_composition_a": "Held-out\ncomposition A",
    "heldout_composition_b": "Held-out\ncomposition B",
}

METHOD_LABELS = {
    "TANDEM-HRL": "TANDEM-HRL",
    "ZYB-v0+Fixed-Dispatcher": "ZYB-v0 + fixed dispatcher",
}

ABLATION_LABELS = {
    "none": "Full model",
    "fixed_task": "Fixed task",
    "fixed_skill": "Fixed skill",
    "no_relational_state": "No relational state",
    "no_predictive_models": "No predictive models",
    "no_control_objective": "No control objective",
    "no_payload_option_barrier": "No payload-aware objective",
}

COLORS = {
    "TANDEM-HRL": "#0072B2",
    "ZYB-v0+Fixed-Dispatcher": "#D55E00",
    "Full model": "#0072B2",
    "Fixed task": "#CC79A7",
    "Fixed skill": "#E69F00",
    "No relational state": "#009E73",
    "No predictive models": "#56B4E9",
    "No control objective": "#7A7A7A",
    "No payload-aware objective": "#F0E442",
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrl_summary", type=Path, required=True)
    parser.add_argument("--baseline_summary", type=Path, required=True)
    parser.add_argument(
        "--ablation_summary",
        type=Path,
        nargs="*",
        default=None,
        help="One or more difficult-scene ablation summary files.",
    )
    parser.add_argument("--training_scalars", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _scenario_order(rows):
    return [
        name
        for name in SCENARIO_LABELS
        if any(row.get("scenario") == name for row in rows)
    ]


def _strict_success(row):
    return _number(row.get("mission_success_rate"))


def _metric_value(row, metric):
    if metric == "strict_success":
        return _strict_success(row)
    return _number(row.get(metric))


def _mean_ci(values):
    values = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    t95 = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(int(values.size), 1.96)
    return mean, float(t95 * values.std(ddof=1) / math.sqrt(values.size))


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.dpi": 400,
            "figure.dpi": 130,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_figure(figure, output_stem):
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)


def _method_bars(
    axes,
    rows,
    scenarios,
    regime,
    metrics,
    methods,
):
    subset = [
        row
        for row in rows
        if row.get("ablation", "none") == "none"
        and row.get("regime", "nominal") == regime
    ]
    width = 0.34
    x = np.arange(len(scenarios), dtype=np.float64)
    for metric_id, (metric, ylabel, lower_better) in enumerate(metrics):
        axis = axes.flat[metric_id]
        for method_id, method in enumerate(methods):
            means = []
            errors = []
            for scenario in scenarios:
                values = [
                    _metric_value(row, metric)
                    for row in subset
                    if row.get("scenario") == scenario
                    and row.get("method") == method
                ]
                mean, error = _mean_ci(values)
                means.append(mean)
                errors.append(error)
            offset = (method_id - (len(methods) - 1) / 2.0) * width
            label = METHOD_LABELS.get(method, method)
            axis.bar(
                x + offset,
                means,
                width=width,
                yerr=errors,
                capsize=2.2,
                linewidth=0.6,
                edgecolor="black",
                color=COLORS.get(method, "#777777"),
                label=label,
                zorder=3,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SCENARIO_LABELS.get(name, name) for name in scenarios]
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in {
            "strict_success",
            "physical_transport_fraction",
            "rear_both_airborne_rate",
            "safety_failure_rate",
            "tilt_failure_rate",
        }:
            axis.set_ylim(bottom=0.0)
        if not lower_better and metric != "reward_mean":
            axis.set_ylim(top=max(1.0, axis.get_ylim()[1]))
    axes.flat[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.05, 1.26),
        ncol=len(methods),
        frameon=False,
    )


def _plot_multiscenario(rows, output_dir):
    scenarios = _scenario_order(rows)
    methods = [
        method
        for method in (
            "TANDEM-HRL",
            "ZYB-v0+Fixed-Dispatcher",
        )
        if any(row.get("method") == method for row in rows)
    ]
    outcome_metrics = (
        ("strict_success", "Mission success within budget", False),
        (
            "physical_transport_fraction",
            "Required objects transported",
            False,
        ),
        ("reward_mean", "Return per step", False),
        ("task_error_mean", "Task error", True),
    )
    system_metrics = (
        ("base_xy_tracking_error", "Base tracking error (m/s)", True),
        ("safety_failure_rate", "CBF violation rate", True),
        ("tilt_failure_rate", "Tilt termination rate", True),
        (
            "rear_both_airborne_rate",
            "Both-rear-airborne rate",
            True,
        ),
    )
    for regime in ("nominal", "stress"):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(13.2, 6.5),
            constrained_layout=True,
        )
        _method_bars(
            axes,
            rows,
            scenarios,
            regime,
            outcome_metrics,
            methods,
        )
        figure.suptitle(
            "{} multi-scenario outcomes".format(regime.capitalize()),
            y=1.03,
            fontsize=11,
        )
        _finish_figure(
            figure,
            output_dir / f"multiscenario_outcomes_{regime}",
        )

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(13.2, 6.5),
            constrained_layout=True,
        )
        _method_bars(
            axes,
            rows,
            scenarios,
            regime,
            system_metrics,
            methods,
        )
        figure.suptitle(
            "{} transient and robustness metrics".format(
                regime.capitalize()
            ),
            y=1.03,
            fontsize=11,
        )
        _finish_figure(
            figure,
            output_dir / f"system_robustness_{regime}",
        )


def _plot_physical_manipulation(rows, output_dir):
    scenarios = [
        scenario
        for scenario in _scenario_order(rows)
        if any(
            row.get("scenario") == scenario
            and _number(row.get("required_object_count")) > 0
            for row in rows
        )
    ]
    methods = [
        method
        for method in (
            "TANDEM-HRL",
            "ZYB-v0+Fixed-Dispatcher",
        )
        if any(row.get("method") == method for row in rows)
    ]
    metrics = (
        ("physical_contact_fraction", "Required objects contacted", False),
        ("physical_lift_fraction", "Required objects lifted", False),
        ("physical_carry_fraction", "Required objects carried", False),
        ("physical_transport_fraction", "Required objects transported", False),
        ("object_completion_fraction", "Required objects placed", False),
        ("drop_env_rate", "Unintended-drop environment rate", True),
    )
    for regime in ("nominal", "stress"):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(13.2, 6.5),
            constrained_layout=True,
        )
        _method_bars(
            axes,
            rows,
            scenarios,
            regime,
            metrics,
            methods,
        )
        figure.suptitle(
            "{} physical manipulation chain".format(regime.capitalize()),
            y=1.03,
            fontsize=11,
        )
        _finish_figure(
            figure,
            output_dir / f"physical_manipulation_{regime}",
        )


def _plot_metric_points(
    rows,
    scenarios,
    regime,
    metrics,
    methods,
    output,
    title,
):
    subset = [
        row
        for row in rows
        if row.get("ablation", "none") == "none"
        and row.get("regime", "nominal") == regime
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 6.5),
        constrained_layout=True,
    )
    x = np.arange(len(scenarios), dtype=np.float64)
    offsets = np.linspace(-0.12, 0.12, max(1, len(methods)))
    for metric_id, (metric, ylabel, lower_better) in enumerate(metrics):
        axis = axes.flat[metric_id]
        for method_id, method in enumerate(methods):
            means = []
            errors = []
            for scenario in scenarios:
                values = [
                    _metric_value(row, metric)
                    for row in subset
                    if row.get("scenario") == scenario
                    and row.get("method") == method
                ]
                mean, error = _mean_ci(values)
                means.append(mean)
                errors.append(error)
            axis.errorbar(
                x + offsets[method_id],
                means,
                yerr=errors,
                color=COLORS.get(method, "#777777"),
                marker="o" if method_id == 0 else "s",
                markersize=4.2,
                capsize=2.4,
                linewidth=1.3,
                label=METHOD_LABELS.get(method, method),
                zorder=3,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SCENARIO_LABELS.get(name, name) for name in scenarios]
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        if lower_better:
            axis.set_ylim(bottom=0.0)
    axes.flat[0].legend(frameon=False, ncol=len(methods))
    figure.suptitle(title, y=1.03, fontsize=11)
    _finish_figure(figure, output)


def _plot_transient_times(rows, output_dir):
    scenarios = [
        scenario
        for scenario in _scenario_order(rows)
        if any(
            row.get("scenario") == scenario
            and _number(row.get("required_object_count")) > 0
            for row in rows
        )
    ]
    methods = [
        method
        for method in (
            "TANDEM-HRL",
            "ZYB-v0+Fixed-Dispatcher",
        )
        if any(row.get("method") == method for row in rows)
    ]
    metrics = (
        (
            "required_contact_restricted_time_s",
            "Time to required contact (s)",
            True,
        ),
        (
            "required_lift_restricted_time_s",
            "Time to required lift (s)",
            True,
        ),
        (
            "required_transport_restricted_time_s",
            "Time to required transport (s)",
            True,
        ),
        (
            "required_completion_restricted_time_s",
            "Time to strict completion (s)",
            True,
        ),
    )
    for regime in ("nominal", "stress"):
        _plot_metric_points(
            rows,
            scenarios,
            regime,
            metrics,
            methods,
            output_dir / f"transient_event_times_{regime}",
            "{} event-reaching time".format(regime.capitalize()),
        )


def _plot_hierarchy_diagnostics(rows, output_dir):
    rows = [
        row
        for row in rows
        if row.get("method") == "TANDEM-HRL"
        and row.get("ablation", "none") == "none"
    ]
    if not rows:
        return
    scenarios = _scenario_order(rows)
    metrics = (
        ("actor_task_confidence", "Task-selection confidence"),
        ("actor_skill_confidence", "Skill-selection confidence"),
        ("task_switch_mean", "Task switches per step"),
        ("skill_switch_mean", "Skill switches per step"),
        ("actor_constraint_violation", "Constraint violation"),
        ("actor_payload_risk_pressure", "Payload risk pressure"),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.0, 6.0),
        constrained_layout=True,
    )
    x = np.arange(len(scenarios), dtype=np.float64)
    for metric_id, (metric, ylabel) in enumerate(metrics):
        axis = axes.flat[metric_id]
        for regime, color, marker in (
            ("nominal", "#0072B2", "o"),
            ("stress", "#D55E00", "s"),
        ):
            means = []
            errors = []
            for scenario in scenarios:
                values = [
                    _metric_value(row, metric)
                    for row in rows
                    if row.get("scenario") == scenario
                    and row.get("regime") == regime
                ]
                mean, error = _mean_ci(values)
                means.append(mean)
                errors.append(error)
            axis.errorbar(
                x,
                means,
                yerr=errors,
                color=color,
                marker=marker,
                markersize=4.0,
                capsize=2.2,
                linewidth=1.3,
                label=regime.capitalize(),
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SCENARIO_LABELS.get(name, name) for name in scenarios],
            rotation=10,
            ha="right",
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=2)
    figure.suptitle(
        "Learned task and skill decomposition diagnostics",
        y=1.03,
        fontsize=11,
    )
    _finish_figure(figure, output_dir / "hierarchy_diagnostics")


def _plot_stress_retention(rows, output_dir):
    scenarios = _scenario_order(rows)
    methods = [
        method
        for method in (
            "TANDEM-HRL",
            "ZYB-v0+Fixed-Dispatcher",
        )
        if any(row.get("method") == method for row in rows)
    ]
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(9.6, 3.6),
        constrained_layout=True,
    )
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.34
    for method_id, method in enumerate(methods):
        retention = []
        for scenario in scenarios:
            nominal = [
                _strict_success(row)
                for row in rows
                if row.get("method") == method
                and row.get("scenario") == scenario
                and row.get("regime") == "nominal"
                and row.get("ablation", "none") == "none"
            ]
            stress = [
                _strict_success(row)
                for row in rows
                if row.get("method") == method
                and row.get("scenario") == scenario
                and row.get("regime") == "stress"
                and row.get("ablation", "none") == "none"
            ]
            nominal_mean, _ = _mean_ci(nominal)
            stress_mean, _ = _mean_ci(stress)
            retention.append(
                stress_mean / nominal_mean
                if math.isfinite(nominal_mean) and nominal_mean > 1.0e-6
                else math.nan
            )
        offset = (method_id - (len(methods) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            retention,
            width=width,
            color=COLORS.get(method, "#777777"),
            edgecolor="black",
            linewidth=0.6,
            label=METHOD_LABELS.get(method, method),
        )
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    axis.set_ylabel("Stress / nominal success")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [SCENARIO_LABELS.get(name, name) for name in scenarios]
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=len(methods))
    _finish_figure(figure, output_dir / "stress_success_retention")


def _plot_ablation(rows, output_dir):
    rows = [
        row
        for row in rows
        if row.get("regime") == "stress"
        and row.get("scenario")
        in {
            "triple_object_delivery",
            "shape_diverse_delivery",
            "grand_mission",
        }
    ]
    if not rows or not any(row.get("ablation") != "none" for row in rows):
        return
    ablation_seeds = {
        row.get("seed", "")
        for row in rows
        if row.get("ablation", "none") != "none"
    }
    rows = [
        row
        for row in rows
        if row.get("ablation", "none") != "none"
        or row.get("seed", "") in ablation_seeds
    ]
    ablations = [
        name
        for name in ABLATION_LABELS
        if any(row.get("ablation", "none") == name for row in rows)
    ]
    metrics = (
        ("strict_success", "Mission success within budget", False),
        (
            "physical_transport_fraction",
            "Required objects transported",
            False,
        ),
        ("safety_failure_rate", "CBF violation rate", True),
        ("reward_mean", "Return per step", False),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 6.4),
        constrained_layout=True,
    )
    x = np.arange(len(ablations))
    for metric_id, (metric, ylabel, lower_better) in enumerate(metrics):
        axis = axes.flat[metric_id]
        means = []
        errors = []
        for ablation in ablations:
            values = [
                _metric_value(row, metric)
                for row in rows
                if row.get("ablation", "none") == ablation
            ]
            mean, error = _mean_ci(values)
            means.append(mean)
            errors.append(error)
        labels = [ABLATION_LABELS[name] for name in ablations]
        axis.bar(
            x,
            means,
            yerr=errors,
            capsize=2.5,
            color=[COLORS[label] for label in labels],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        if metric != "reward_mean":
            axis.set_ylim(bottom=0.0)
        if not lower_better and metric != "reward_mean":
            axis.set_ylim(top=max(1.0, axis.get_ylim()[1]))
    figure.suptitle(
        "Difficult-scene ablation study",
        y=1.03,
        fontsize=11,
    )
    _finish_figure(figure, output_dir / "difficult_scene_ablation")


def _smooth(values, window=21):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values
    window = min(window, values.size)
    result = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - window // 2)
        stop = min(values.size, index + window // 2 + 1)
        result[index] = np.median(values[start:stop])
    return result


def _plot_training(path, output_dir):
    if path is None or not path.is_file():
        return
    grouped = defaultdict(list)
    for row in _read_rows(path):
        grouped[row["tag"]].append(
            (int(row["step"]), _number(row["value"]))
        )
    panels = (
        (("Train/mean_reward",), "Training return"),
        (
            (
                "Metrics/locomotion/TACTIC/curriculum_contact_ema",
                "Metrics/locomotion/TACTIC/curriculum_lift_ema",
                "Metrics/locomotion/TACTIC/curriculum_transport_ema",
                "Metrics/locomotion/TACTIC/curriculum_place_ema",
                "Metrics/locomotion/TACTIC/curriculum_delivery_completion_ema",
            ),
            "Interaction curriculum EMA",
        ),
        (
            (
                "Metrics/locomotion/TACTIC/composition_probe_probability",
                "Metrics/locomotion/TACTIC/curriculum_level",
            ),
            "Task-composition curriculum",
        ),
        (
            (
                "Loss/counterfactual_task_selection",
                "Loss/counterfactual_skill_selection",
            ),
            "Task / skill selection loss",
        ),
        (
            (
                "Episode_Reward/tactic_release_readiness",
                "Episode_Reward/tactic_intended_release",
                "Episode_Reward/tactic_object_completion",
            ),
            "Release and completion reward",
        ),
        (
            (
                "Episode_Termination/tilt",
                "Episode_Termination/low_height",
                "Episode_Termination/time_out",
                "Loss/payload_skill_barrier_pressure",
                "Loss/payload_transient_demand",
            ),
            "Failure and payload-risk signal",
        ),
    )
    label_map = {
        "Train/mean_reward": "Return",
        "Metrics/locomotion/TACTIC/curriculum_contact_ema": "Contact",
        "Metrics/locomotion/TACTIC/curriculum_lift_ema": "Lift",
        "Metrics/locomotion/TACTIC/curriculum_transport_ema": "Transport",
        "Metrics/locomotion/TACTIC/curriculum_place_ema": "Place",
        "Metrics/locomotion/TACTIC/curriculum_delivery_completion_ema": (
            "Strict placement"
        ),
        "Metrics/locomotion/TACTIC/composition_probe_probability": (
            "Composition probe rate"
        ),
        "Metrics/locomotion/TACTIC/curriculum_level": "Curriculum level",
        "Loss/counterfactual_task_selection": "Task selection",
        "Loss/counterfactual_skill_selection": "Skill selection",
        "Episode_Reward/tactic_release_readiness": "Release readiness",
        "Episode_Reward/tactic_intended_release": "Intended release",
        "Episode_Reward/tactic_object_completion": "Strict completion",
        "Episode_Termination/tilt": "Tilt",
        "Episode_Termination/low_height": "Low height",
        "Episode_Termination/time_out": "Time out",
        "Loss/payload_skill_barrier_pressure": "Risk pressure",
        "Loss/payload_transient_demand": "Transient demand",
    }
    palette = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.2, 8.0),
        constrained_layout=True,
    )
    for panel_id, (tags, ylabel) in enumerate(panels):
        axis = axes.flat[panel_id]
        for tag_id, tag in enumerate(tags):
            values = sorted(grouped.get(tag, ()))
            if not values:
                continue
            steps = np.asarray([item[0] for item in values])
            raw = np.asarray([item[1] for item in values])
            color = palette[tag_id % len(palette)]
            axis.plot(steps, raw, color=color, alpha=0.16, linewidth=0.8)
            axis.plot(
                steps,
                _smooth(raw),
                color=color,
                label=label_map.get(tag, tag),
            )
        axis.set_xlabel("Training iteration")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D8D8D8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        if len(tags) > 1:
            axis.legend(frameon=False, ncol=2)
    _finish_figure(figure, output_dir / "training_diagnostics")


def _write_aggregate_table(rows, output_path):
    keys = (
        "strict_success",
        "physical_contact_fraction",
        "physical_lift_fraction",
        "physical_carry_fraction",
        "physical_transport_fraction",
        "object_completion_fraction",
        "selected_target_progress_mean",
        "release_readiness_mean",
        "intended_release_env_rate",
        "drop_env_rate",
        "place_error_xy_mean",
        "place_error_z_mean",
        "reward_mean",
        "task_error_mean",
        "ee_error_mean",
        "base_xy_tracking_error",
        "safety_failure_rate",
        "tilt_failure_rate",
        "rear_both_airborne_rate",
        "required_contact_restricted_time_s",
        "required_lift_restricted_time_s",
        "required_transport_restricted_time_s",
        "required_completion_restricted_time_s",
        "actor_task_confidence",
        "actor_skill_confidence",
        "task_switch_mean",
        "skill_switch_mean",
        "actor_constraint_violation",
        "actor_payload_risk_pressure",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("method", ""),
                row.get("regime", ""),
                row.get("scenario", ""),
                row.get("ablation", "none"),
            )
        ].append(row)
    output_rows = []
    for group_key, group_rows in sorted(grouped.items()):
        output = dict(
            zip(("method", "regime", "scenario", "ablation"), group_key)
        )
        output["seed_count"] = len(group_rows)
        for metric in keys:
            values = [
                _metric_value(row, metric) for row in group_rows
            ]
            mean, ci = _mean_ci(values)
            output[metric + "_mean"] = mean
            output[metric + "_ci95"] = ci
        output_rows.append(output)
    if not output_rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(output_rows[0])
        )
        writer.writeheader()
        writer.writerows(output_rows)


def _write_paired_comparisons(rows, output_path):
    metrics = (
        ("strict_success", True),
        ("physical_transport_fraction", True),
        ("object_completion_fraction", True),
        ("intended_release_env_rate", True),
        ("drop_env_rate", False),
        ("reward_mean", True),
        ("task_error_mean", False),
        ("ee_error_mean", False),
        ("base_xy_tracking_error", False),
        ("safety_failure_rate", False),
        ("tilt_failure_rate", False),
        ("rear_both_airborne_rate", False),
    )
    output_rows = []
    for regime in ("nominal", "stress"):
        for scenario in _scenario_order(rows):
            subset = [
                row
                for row in rows
                if row.get("regime") == regime
                and row.get("scenario") == scenario
                and row.get("ablation", "none") == "none"
            ]
            hrl_by_seed = {
                row.get("seed", ""): row
                for row in subset
                if row.get("method") == "TANDEM-HRL"
            }
            baseline_by_seed = {
                row.get("seed", ""): row
                for row in subset
                if row.get("method") == "ZYB-v0+Fixed-Dispatcher"
            }
            seeds = sorted(set(hrl_by_seed) & set(baseline_by_seed))
            for metric, higher_is_better in metrics:
                hrl_values = [
                    _metric_value(hrl_by_seed[seed], metric)
                    for seed in seeds
                ]
                baseline_values = [
                    _metric_value(baseline_by_seed[seed], metric)
                    for seed in seeds
                ]
                paired = [
                    (hrl - baseline)
                    * (1.0 if higher_is_better else -1.0)
                    for hrl, baseline in zip(
                        hrl_values, baseline_values
                    )
                    if math.isfinite(hrl) and math.isfinite(baseline)
                ]
                if not paired:
                    continue
                improvement, half_width = _mean_ci(paired)
                lower = improvement - half_width
                upper = improvement + half_width
                hrl_mean = float(
                    np.mean(
                        [value for value in hrl_values if math.isfinite(value)]
                    )
                )
                baseline_mean = float(
                    np.mean(
                        [
                            value
                            for value in baseline_values
                            if math.isfinite(value)
                        ]
                    )
                )
                baseline_scale = abs(baseline_mean)
                relative = (
                    100.0 * improvement / baseline_scale
                    if baseline_scale > 1.0e-6
                    else math.nan
                )
                paired_array = np.asarray(paired, dtype=np.float64)
                effect_size = (
                    float(
                        paired_array.mean()
                        / paired_array.std(ddof=1)
                    )
                    if paired_array.size > 1
                    and paired_array.std(ddof=1) > 1.0e-12
                    else math.nan
                )
                output_rows.append(
                    {
                        "regime": regime,
                        "scenario": scenario,
                        "metric": metric,
                        "higher_is_better": int(higher_is_better),
                        "paired_seed_count": len(paired),
                        "hrl_mean": hrl_mean,
                        "baseline_mean": baseline_mean,
                        "signed_improvement_mean": improvement,
                        "signed_improvement_ci95": half_width,
                        "signed_improvement_ci95_low": lower,
                        "signed_improvement_ci95_high": upper,
                        "ci95_excludes_zero": int(
                            lower > 0.0 or upper < 0.0
                        ),
                        "paired_win_rate": float(
                            np.mean(paired_array > 0.0)
                        ),
                        "relative_improvement_percent": relative,
                        "paired_effect_size_dz": effect_size,
                    }
                )
    if not output_rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(output_rows[0])
        )
        writer.writeheader()
        writer.writerows(output_rows)


def _deduplicate(rows):
    unique = {}
    for row in rows:
        key = (
            row.get("source", ""),
            row.get("scenario", ""),
            row.get("task_set_id", ""),
            row.get("method", ""),
            row.get("ablation", ""),
        )
        unique[key] = row
    return list(unique.values())


def main():
    args = _parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.hrl_summary) + _read_rows(
        args.baseline_summary
    )
    _plot_multiscenario(rows, args.output_dir)
    _plot_physical_manipulation(rows, args.output_dir)
    _plot_transient_times(rows, args.output_dir)
    _plot_hierarchy_diagnostics(rows, args.output_dir)
    _plot_stress_retention(rows, args.output_dir)
    ablation_rows = _read_rows(args.hrl_summary)
    for path in args.ablation_summary or ():
        ablation_rows.extend(_read_rows(path))
    _plot_ablation(ablation_rows, args.output_dir)
    _plot_training(args.training_scalars, args.output_dir)
    _write_aggregate_table(
        _deduplicate(rows + ablation_rows),
        args.output_dir / "aggregate_metrics_with_ci95.csv",
    )
    _write_paired_comparisons(
        _deduplicate(rows),
        args.output_dir / "paired_hrl_vs_baseline.csv",
    )
    print("figure_dir={}".format(args.output_dir))


if __name__ == "__main__":
    main()
