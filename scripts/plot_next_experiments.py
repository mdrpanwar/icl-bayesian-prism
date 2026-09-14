#!/usr/bin/env python
"""Plot the next ICL/MAML experiments from the latest available W&B history.

Missing, queued, and partially trained runs are tolerated. Every curve ends at
its most recent logged evaluation, so this script is safe to rerun while jobs
are training.
"""

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ICL_METRIC = "linear_regression_eval/pointwise/loss"
MAML_METRIC = "maml_eval/pointwise/loss"
OLD_L21_RUN_ID = "e5e9d6b8-e334-4e67-a386-c535744f745b"
OLD_L21_RUN_NAME = "icl_lr_len21_last"
OLD_L21_RUN_PATH = (
    REPO_ROOT
    / "models/linear_regression"
    / f"{OLD_L21_RUN_NAME}-{OLD_L21_RUN_ID}"
)
POSITION_WINDOW_SEEDS = (0, 154645467, 65765443)


def warn(message):
    warnings.warn(message, stacklevel=2)


def configure_wandb_key(key_file):
    if os.environ.get("WANDB_API_KEY"):
        return
    candidates = []
    if key_file is not None:
        candidates.append(key_file)
    if os.environ.get("WANDB_API_KEY_FILE_AT"):
        candidates.append(Path(os.environ["WANDB_API_KEY_FILE_AT"]))
    candidates.append(REPO_ROOT.parent / ".wandb-api-key")
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            key = candidate.read_text().strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
                return


def pointwise_values(row, metric):
    """Extract integer-indexed values from nested or flattened W&B rows."""
    values = {}
    nested = row.get(metric)
    if isinstance(nested, Mapping):
        items = nested.items()
    elif isinstance(nested, (list, tuple)):
        items = enumerate(nested)
    else:
        items = []
    for key, value in items:
        try:
            value = float(value)
            if math.isfinite(value):
                values[int(key)] = value
        except (TypeError, ValueError):
            pass

    for key, value in row.items():
        suffix = None
        for separator in (".", "/"):
            prefix = f"{metric}{separator}"
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                break
        if suffix is None:
            continue
        try:
            value = float(value)
            if math.isfinite(value):
                values[int(suffix)] = value
        except (TypeError, ValueError):
            pass
    return values


class HistoryCache:
    def __init__(self):
        self._cache = {}

    def pointwise(self, run, metric):
        key = (run.id, metric)
        if key not in self._cache:
            rows = []
            started = time.monotonic()
            print(
                f"[history] fetching {run.name} ({run.id}): {metric}",
                flush=True,
            )
            try:
                if metric == MAML_METRIC:
                    indices = (0, 20, 40)
                elif run.id == OLD_L21_RUN_ID:
                    indices = range(21)
                else:
                    training_config = dict(run.config).get("training", {})
                    eval_n_points = training_config.get("eval_n_points", 41) or 41
                    indices = range(int(eval_n_points))
                keys = [f"{metric}.{index}" for index in indices]
                history = run.history(keys=keys, pandas=False, samples=10000)
                for row in history:
                    step = row.get("_step")
                    values = pointwise_values(row, metric)
                    if step is not None and values:
                        rows.append((int(step), values))
            except Exception as exc:
                warn(f"Could not read history for {run.name!r}: {exc}")
            # Multiple wandb.log calls can share a step; merge their values.
            merged = {}
            for step, values in rows:
                merged.setdefault(step, {}).update(values)
            self._cache[key] = sorted(merged.items())
            print(
                f"[history] loaded {run.name}: "
                f"{len(self._cache[key])} evaluation steps "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        return self._cache[key]

    def dynamics(self, run, metric, position):
        points = {}
        for step, values in self.pointwise(run, metric):
            if position in values:
                points[step] = values[position]
        return sorted(points), [points[step] for step in sorted(points)]

    def latest(self, run, metric, max_context=40):
        history = self.pointwise(run, metric)
        if not history:
            return None, [], []
        step, values = history[-1]
        xs = sorted(index for index in values if 0 <= index <= max_context)
        return step, xs, [values[index] for index in xs]


def fetch_runs(api, entity, project, names):
    names = sorted(set(names))
    by_name = {}

    def rank(run):
        summary_step = run.summary.get("_step", -1)
        try:
            summary_step = int(summary_step)
        except (TypeError, ValueError):
            summary_step = -1
        return summary_step, str(run.created_at)

    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": {"$in": names}},
        )
        for run in runs:
            if run.name not in names:
                continue
            previous = by_name.get(run.name)
            if previous is None or rank(run) > rank(previous):
                by_name[run.name] = run
    except Exception as exc:
        warn(f"Could not list W&B runs: {exc}")

    missing = [name for name in names if name not in by_name]
    if missing:
        warn("Runs not available yet (skipping): " + ", ".join(missing))
    return by_name


def fetch_old_l21_run(api, entity, project, runs):
    try:
        runs[OLD_L21_RUN_NAME] = api.run(
            f"{entity}/{project}/{OLD_L21_RUN_ID}"
        )
    except Exception as exc:
        warn(f"Could not load existing length-21 baseline by ID: {exc}")


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {output_dir / stem}.png/.pdf", flush=True)


def plot_trajectory(ax, xs, ys, **kwargs):
    """Draw a trajectory while keeping sparse and one-point runs visible."""
    marker_stride = 1 if len(xs) <= 25 else math.ceil(len(xs) / 25)
    return ax.plot(
        xs,
        ys,
        marker="o",
        markersize=3.5,
        markevery=int(marker_stride),
        linewidth=1.8,
        **kwargs,
    )


def spec_run_names(spec):
    if "run_names" in spec:
        return list(spec["run_names"])
    return [spec["run_name"]]


def plot_dynamics(
    runs,
    cache,
    specs,
    position,
    output_dir,
    stem,
    ylabel,
    xlabel="Training steps",
    normalization=1.0,
    task_axis=False,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for spec in specs:
        label = spec["label"]
        metric = spec["metric"]
        kind = spec["kind"]
        max_steps = spec.get("max_steps")
        if max_steps is not None and max_steps < 0:
            raise ValueError(f"max_steps must be non-negative for {label!r}")

        series = []
        for run_name in spec_run_names(spec):
            run = runs.get(run_name)
            if run is None:
                continue
            steps, values = cache.dynamics(run, metric, position)
            if not steps:
                warn(f"{run_name!r} has no evaluation at context {position} yet")
                continue

            points = [
                (step, value)
                for step, value in zip(steps, values)
                if max_steps is None or step <= max_steps
            ]
            if not points:
                warn(
                    f"{run_name!r} has no evaluation at or before "
                    f"max_steps={max_steps}"
                )
                continue

            if task_axis:
                config = dict(run.config)
                if kind == "maml":
                    multiplier = config.get("meta", {}).get("meta_batch_size", 64)
                else:
                    multiplier = config.get("training", {}).get("batch_size", 64)
            else:
                multiplier = 1
            series.append(
                {
                    int(step) * int(multiplier): float(value) / normalization
                    for step, value in points
                }
            )

        if not series:
            continue
        common_steps = sorted(set.intersection(*(set(item) for item in series)))
        if not common_steps:
            warn(f"No common evaluation steps yet for {label}")
            continue
        matrix = np.asarray(
            [[run_values[step] for step in common_steps] for run_values in series]
        )
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if len(series) > 1 else None
        curve_label = label if len(series) == 1 else f"{label} (n={len(series)})"
        plot_trajectory(ax, common_steps, mean, label=curve_label)
        if std is not None:
            ax.fill_between(common_steps, mean - std, mean + std, alpha=0.2)
        plotted += 1

    if not plotted:
        plt.close(fig)
        warn(f"No available curves for {stem}; figure skipped")
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, stem)


def plot_individual_dynamics(
    runs,
    cache,
    specs,
    position,
    output_dir,
    stem,
    ylabel,
    xlabel="Training steps",
    normalization=1.0,
    task_axis=False,
):
    """Plot every available run directly, grouped by spec color."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    linestyles = ("-", "--", ":", "-.")
    markers = ("o", "s", "^", "D")
    plotted = 0

    for spec_index, spec in enumerate(specs):
        color = f"C{spec_index}"
        for run_index, run_name in enumerate(spec_run_names(spec)):
            run = runs.get(run_name)
            if run is None:
                continue
            steps, values = cache.dynamics(run, spec["metric"], position)
            max_steps = spec.get("max_steps")
            points = [
                (step, value / normalization)
                for step, value in zip(steps, values)
                if max_steps is None or step <= max_steps
            ]
            if not points:
                continue

            xs, ys = zip(*points)
            if task_axis:
                config = dict(run.config)
                if spec["kind"] == "maml":
                    multiplier = config.get("meta", {}).get("meta_batch_size", 64)
                else:
                    multiplier = config.get("training", {}).get("batch_size", 64)
            else:
                multiplier = 1
            xs = tuple(int(step) * int(multiplier) for step in xs)
            marker_stride = 1 if len(xs) <= 25 else math.ceil(len(xs) / 25)
            seed_label = run_name.rsplit("_s", 1)[-1]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyles[run_index % len(linestyles)],
                marker=markers[run_index % len(markers)],
                markersize=3.5,
                markevery=marker_stride,
                linewidth=1.5,
                alpha=0.9,
                label=f"{spec['label']}, seed {seed_label}",
            )
            plotted += 1

    if not plotted:
        plt.close(fig)
        warn(f"No available curves for {stem}; figure skipped")
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, stem)


def evaluate_old_l21_context(output_dir, skip_local_eval):
    """Evaluate the reused old checkpoint through context 40 once and cache it."""
    cache_path = output_dir / "old_l21_context41.json"
    if cache_path.is_file():
        try:
            values = json.loads(cache_path.read_text())
            if len(values) >= 41:
                return list(range(41)), [float(value) for value in values[:41]]
        except Exception as exc:
            warn(f"Ignoring invalid old-baseline cache {cache_path}: {exc}")
    if skip_local_eval or not OLD_L21_RUN_PATH.is_dir():
        return None

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from eval import get_run_metrics

        metrics = get_run_metrics(
            OLD_L21_RUN_PATH,
            cache=False,
            skip_baselines=True,
            eval_names=["standard"],
            eval_kwargs_overrides={"n_points": 41, "parallel_seed": 0},
        )
        model_metrics = next(iter(metrics["standard"].values()))
        values = [float(value) for value in model_metrics["mean"][:41]]
        if len(values) != 41:
            raise RuntimeError(f"expected 41 points, received {len(values)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(values))
        return list(range(41)), values
    except Exception as exc:
        warn(f"Could not evaluate old length-21 checkpoint through context 40: {exc}")
        return None


def plot_positional_context(
    runs, cache, output_dir, normalization, skip_old_local_eval
):
    specs = [
        (OLD_L21_RUN_NAME, "Length 21, no position embedding"),
        ("icl_lr20_l21_last1_pos_s0", "Length 21, position embedding"),
        ("icl_lr20_l41_last1_nopos_s0", "Length 41, no position embedding"),
        ("icl_lr20_l41_last1_pos_s0", "Length 41, position embedding"),
    ]
    old_local = evaluate_old_l21_context(output_dir, skip_old_local_eval)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for run_name, label in specs:
        if run_name == OLD_L21_RUN_NAME and old_local is not None:
            xs, values = old_local
        else:
            run = runs.get(run_name)
            if run is None:
                continue
            _, xs, values = cache.latest(run, ICL_METRIC, max_context=40)
        if not xs:
            warn(f"{run_name!r} has no pointwise evaluation yet")
            continue
        ax.plot(xs, np.asarray(values) / normalization, label=label)
        plotted += 1
    if not plotted:
        plt.close(fig)
        warn("No available positional context curves; figure skipped")
        return
    ax.set_xlabel("In-context examples")
    ax.set_ylabel("Normalized squared error")
    ax.set_xlim(0, 40)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(
        fig,
        output_dir,
        "positional_embeddings_error_vs_context_length",
    )


def plot_position_windows(runs, cache, output_dir, max_steps=None):
    windows = [
        ("icl_pos_terms01_15_s{}", "Terms 1–15"),
        ("icl_pos_terms06_20_s{}", "Terms 6–20"),
        ("icl_pos_terms21_35_s{}", "Terms 21–35"),
        ("icl_pos_terms26_40_s{}", "Terms 26–40"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for pattern, label in windows:
        series = []
        for seed in POSITION_WINDOW_SEEDS:
            run = runs.get(pattern.format(seed))
            if run is None:
                continue
            steps, values = cache.dynamics(run, ICL_METRIC, 20)
            if steps:
                series.append(dict(zip(steps, values)))
        if not series:
            continue
        # Use steps present in every currently available seed. This prevents
        # the effective seed count from silently changing along a partial curve.
        common_steps = sorted(
            step
            for step in set.intersection(*(set(item) for item in series))
            if max_steps is None or step <= max_steps
        )
        if not common_steps:
            warn(f"No common evaluation steps yet for {label}")
            continue
        matrix = np.asarray(
            [[seed_values[step] for step in common_steps] for seed_values in series]
        )
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if len(series) > 1 else np.zeros_like(mean)
        plot_trajectory(
            ax, common_steps, mean, label=f"{label} (n={len(series)})"
        )
        if len(series) > 1:
            ax.fill_between(common_steps, mean - std, mean + std, alpha=0.2)
        plotted += 1
    if not plotted:
        plt.close(fig)
        warn("No available position-window curves; figure skipped")
        return
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Squared error at context length 20")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, "position_windows_mean_std_training_dynamics")


def plot_position_windows_individual(runs, cache, output_dir, max_steps=None):
    windows = [
        ("icl_pos_terms01_15_s{}", "Terms 1–15"),
        ("icl_pos_terms06_20_s{}", "Terms 6–20"),
        ("icl_pos_terms21_35_s{}", "Terms 21–35"),
        ("icl_pos_terms26_40_s{}", "Terms 26–40"),
    ]
    fig, axes_grid = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True, sharey=True
    )
    axes = axes_grid.ravel()
    plotted = 0

    for (pattern, label), ax in zip(windows, axes):
        series = []
        for seed in POSITION_WINDOW_SEEDS:
            run = runs.get(pattern.format(seed))
            if run is None:
                continue
            steps, values = cache.dynamics(run, ICL_METRIC, 20)
            points = [
                (step, value)
                for step, value in zip(steps, values)
                if max_steps is None or step <= max_steps
            ]
            if not points:
                continue
            seed_values = dict(points)
            series.append((seed, seed_values))
            ax.plot(
                list(seed_values),
                list(seed_values.values()),
                linewidth=1.2,
                alpha=0.8,
                label=f"Seed {seed}",
            )

        if not series:
            ax.set_visible(False)
            continue

        common_steps = sorted(
            set.intersection(*(set(values) for _, values in series))
        )
        if common_steps:
            matrix = np.asarray(
                [
                    [seed_values[step] for step in common_steps]
                    for _, seed_values in series
                ]
            )
            plot_trajectory(
                ax,
                common_steps,
                matrix.mean(axis=0),
                color="black",
                label=f"Mean (n={len(series)})",
            )
        ax.set_title(label)
        ax.grid(alpha=0.25)
        plotted += 1

    if not plotted:
        plt.close(fig)
        warn("No available position-window curves; figure skipped")
        return

    for ax in axes[2:]:
        ax.set_xlabel("Training steps")
    for ax in axes[::2]:
        ax.set_ylabel("Squared error at context length 20")

    handles, labels = next(
        (
            ax.get_legend_handles_labels()
            for ax in axes
            if ax.get_visible() and ax.lines
        ),
        ([], []),
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.0),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        output_dir,
        "position_windows_individual_training_dynamics",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/followup_experiments")
    )
    parser.add_argument("--wandb-api-key-file", type=Path)
    parser.add_argument(
        "--skip-old-baseline-eval",
        action="store_true",
        help=(
            "Do not locally evaluate the old length-21 checkpoint through "
            "context 40. Its W&B curve (usually through context 20) is used."
        ),
    )
    parser.add_argument(
        "--context-normalization",
        type=float,
        default=20.0,
        help="Normalization for 20D comparison/context plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_wandb_key(args.wandb_api_key_file)

    # max_steps is measured in optimizer steps, including when a figure's
    # x-axis is converted to tasks seen. None keeps every completed evaluation.
    lr5_specs = [
        {
            "run_name": "icl_lr5_l11_last5_200k_s0",
            "label": "Last 5 terms, 200k steps",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr5_l11_last2_500k_s0",
            "label": "Last 2 terms, 500k steps",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr5_l11_last1_1000k_s0",
            "label": "Last 1 term, 1000k steps",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
    ]
    matched_specs = [
        {
            "run_names": [
                "icl_lr20_causal_c20_l20_s0",
                "icl_lr20_causal_c20_l20_s154645467",
                "icl_lr20_causal_c20_l20_s65765443",
            ],
            "label": "ICL, causal prefix",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr20_prefixbidir_c20_l20_s0",
            "label": "ICL, bidirectional prefix",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "maml_lr20_s20_q20_steps2_stable_s0",
            "label": "MAML Meta-SGD, 2 inner steps (stabilized)",
            "metric": MAML_METRIC,
            "kind": "maml",
            "max_steps": None,
        },
        {
            "run_name": "maml_lr20_s20_q20_steps5_stable_s0",
            "label": "MAML Meta-SGD, 5 inner steps (stabilized)",
            "metric": MAML_METRIC,
            "kind": "maml",
            "max_steps": None,
        },
    ]
    positional_specs = [
        {
            "run_name": OLD_L21_RUN_NAME,
            "label": "Length 21, no position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr20_l21_last1_pos_s0",
            "label": "Length 21, position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr20_l41_last1_nopos_s0",
            "label": "Length 41, no position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr20_l41_last1_pos_s0",
            "label": "Length 41, position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
    ]
    terms21_35_positional_specs = [
        {
            "run_names": [
                "icl_pos_terms21_35_s154645467",
                "icl_pos_terms21_35_s65765443",
            ],
            "label": "Terms 21–35, no position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": 400_000,
        },
        {
            "run_names": [
                "icl_pos_terms21_35_pos_s154645467",
                "icl_pos_terms21_35_pos_s65765443",
            ],
            "label": "Terms 21–35, position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": 400_000,
        },
    ]
    window_names = [
        pattern.format(seed)
        for pattern in (
            "icl_pos_terms01_15_s{}",
            "icl_pos_terms06_20_s{}",
            "icl_pos_terms21_35_s{}",
            "icl_pos_terms26_40_s{}",
        )
        for seed in POSITION_WINDOW_SEEDS
    ]
    requested_names = [
        run_name
        for spec in (
            lr5_specs
            + matched_specs
            + positional_specs
            + terms21_35_positional_specs
        )
        for run_name in spec_run_names(spec)
        if run_name != OLD_L21_RUN_NAME
    ] + window_names

    try:
        api = wandb.Api()
        runs = fetch_runs(api, args.entity, args.project, requested_names)
        fetch_old_l21_run(api, args.entity, args.project, runs)
    except Exception as exc:
        raise SystemExit(f"Unable to initialize W&B API: {exc}") from exc

    cache = HistoryCache()
    plot_dynamics(
        runs,
        cache,
        lr5_specs,
        position=5,
        output_dir=args.output_dir,
        stem="lr5_loss_budget_training_dynamics",
        ylabel="Squared error at context length 5",
    )
    plot_individual_dynamics(
        runs,
        cache,
        matched_specs,
        position=20,
        output_dir=args.output_dir,
        stem="icl_maml_tasks_seen_individual_training_dynamics",
        ylabel="Normalized squared error after 20 examples",
        xlabel="Training tasks seen",
        normalization=args.context_normalization,
        task_axis=True,
    )
    plot_dynamics(
        runs,
        cache,
        matched_specs,
        position=20,
        output_dir=args.output_dir,
        stem="icl_maml_tasks_seen_training_dynamics",
        ylabel="Normalized squared error after 20 examples",
        xlabel="Training tasks seen",
        normalization=args.context_normalization,
        task_axis=True,
    )
    plot_dynamics(
        runs,
        cache,
        positional_specs,
        position=20,
        output_dir=args.output_dir,
        stem="positional_embeddings_training_dynamics",
        ylabel="Squared error at context length 20",
    )
    plot_dynamics(
        runs,
        cache,
        terms21_35_positional_specs,
        position=20,
        output_dir=args.output_dir,
        stem="terms21_35_positional_training_dynamics",
        ylabel="Squared error at context length 20",
    )
    plot_individual_dynamics(
        runs,
        cache,
        terms21_35_positional_specs,
        position=20,
        output_dir=args.output_dir,
        stem="terms21_35_positional_individual_training_dynamics",
        ylabel="Squared error at context length 20",
    )
    plot_positional_context(
        runs,
        cache,
        args.output_dir,
        args.context_normalization,
        args.skip_old_baseline_eval,
    )
    plot_position_windows(runs, cache, args.output_dir, max_steps=400_000)
    plot_position_windows_individual(runs, cache, args.output_dir, max_steps=400_000)
    print(f"plots available in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
