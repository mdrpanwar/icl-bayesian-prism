#!/usr/bin/env python
"""Generate the experiment figures agreed on 2026-09-11.

The W&B figures tolerate missing and partially trained runs. Curves stop at
their latest available evaluation. Local post-training evaluation reads atomic
state checkpoints and caches by checkpoint signature, so it is also safe while
training continues.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from plot_next_experiments import (
    HistoryCache,
    ICL_METRIC,
    MAML_METRIC,
    configure_wandb_key,
    fetch_runs,
    plot_dynamics,
    plot_individual_dynamics,
    plot_trajectory,
    save_figure,
    spec_run_names,
    warn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SEEDS = (0, 154645467, 65765443)
WINDOW_CONTEXTS = (20, 22, 25, 30)
STABLE_FIVE_RUN_ID = "b92d63fa-31ba-4a24-9860-1728b3e5c32a"
STABLE_FIVE_RUN_NAME = "maml_lr20_s20_q20_steps5_stable_s0"
POSTTRAINING_CACHE_VERSION = 2


def progress(message):
    print(f"[plot {time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/experiments-2026-09-11"),
    )
    parser.add_argument("--wandb-api-key-file", type=Path)
    parser.add_argument("--context-normalization", type=float, default=20.0)
    parser.add_argument("--window-max-steps", type=int, default=400_000)
    parser.add_argument("--posttraining-eval-examples", type=int, default=1280)
    parser.add_argument("--posttraining-eval-batch-size", type=int, default=64)
    parser.add_argument("--posttraining-eval-seed", type=int, default=0)
    parser.add_argument(
        "--posttraining-ymax",
        type=float,
        default=1.3,
        help="Upper y-limit for the normalized post-training comparison.",
    )
    parser.add_argument(
        "--refresh-posttraining-cache",
        action="store_true",
        help="Re-evaluate checkpoints instead of reusing matching cached results.",
    )
    parser.add_argument(
        "--skip-posttraining-eval",
        action="store_true",
        help="Skip the local error-versus-context evaluation.",
    )
    return parser.parse_args()


def common_series(runs, cache, spec, position, normalization):
    series = []
    for run_name in spec_run_names(spec):
        run = runs.get(run_name)
        if run is None:
            continue
        steps, values = cache.dynamics(run, spec["metric"], position)
        limit = spec.get("max_steps")
        points = {
            int(step): float(value) / normalization
            for step, value in zip(steps, values)
            if limit is None or step <= limit
        }
        if points:
            series.append((run, points))
    return series


def profile_seconds(run):
    if run is None:
        return None
    value = run.summary.get("compute_profile/seconds_per_step")
    try:
        value = float(value)
        if math.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError):
        pass
    try:
        history = run.history(
            keys=["compute_profile/seconds_per_step"],
            pandas=False,
            samples=100,
        )
        values = [
            float(row["compute_profile/seconds_per_step"])
            for row in history
            if row.get("compute_profile/seconds_per_step") is not None
        ]
        return values[-1] if values else None
    except Exception:
        return None


def transformed_x(step, run, spec, axis, runs):
    config = dict(run.config)
    if spec["kind"] == "maml":
        batch_size = int(config.get("meta", {}).get("meta_batch_size", 64))
    else:
        batch_size = int(config.get("training", {}).get("batch_size", 64))

    if axis == "tasks":
        return step * batch_size
    if axis == "gradient_pass":
        return step * batch_size * float(spec["gradient_pass_multiplier"])
    if axis == "gpu_hours":
        seconds = profile_seconds(runs.get(spec["profile_run_name"]))
        if seconds is None:
            return None
        return step * seconds / 3600.0
    raise ValueError(f"Unknown comparison axis: {axis}")


def plot_matched_axis(
    runs,
    cache,
    specs,
    output_dir,
    axis,
    individual,
    normalization,
):
    labels = {
        "tasks": "Training tasks seen",
        "gradient_pass": "Task-gradient-pass equivalents",
        "gpu_hours": "Calibrated H100 GPU-hours",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    linestyles = ("-", "--", ":", "-.")
    markers = ("o", "s", "^", "D")
    plotted = 0

    for spec_index, spec in enumerate(specs):
        series = common_series(runs, cache, spec, 20, normalization)
        transformed = []
        for run, values in series:
            points = {}
            for step, value in values.items():
                x_value = transformed_x(step, run, spec, axis, runs)
                if x_value is not None:
                    points[x_value] = value
            if points:
                transformed.append((run, points))

        if not transformed:
            if axis == "gpu_hours":
                warn(
                    f"Skipping {spec['label']!r} in GPU-hours plot until "
                    f"{spec['profile_run_name']!r} finishes"
                )
            continue

        color = f"C{spec_index}"
        if individual:
            for run_index, (run, values) in enumerate(transformed):
                xs = sorted(values)
                ys = [values[x] for x in xs]
                seed = dict(run.config).get("training", {}).get("seed", "?")
                stride = 1 if len(xs) <= 25 else math.ceil(len(xs) / 25)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyles[run_index % len(linestyles)],
                    marker=markers[run_index % len(markers)],
                    markersize=3.5,
                    markevery=stride,
                    linewidth=1.5,
                    label=f"{spec['label']}, seed {seed}",
                )
                plotted += 1
        else:
            common_x = sorted(
                set.intersection(*(set(values) for _, values in transformed))
            )
            if not common_x:
                continue
            matrix = np.asarray(
                [[values[x] for x in common_x] for _, values in transformed]
            )
            mean = matrix.mean(axis=0)
            plot_trajectory(
                ax,
                common_x,
                mean,
                color=color,
                label=(
                    spec["label"]
                    if len(transformed) == 1
                    else f"{spec['label']} (n={len(transformed)})"
                ),
            )
            if len(transformed) > 1:
                std = matrix.std(axis=0, ddof=1)
                ax.fill_between(common_x, mean - std, mean + std, color=color, alpha=0.2)
            plotted += 1

    suffix = "individual" if individual else "mean_std"
    if not plotted:
        plt.close(fig)
        warn(f"No data available for {axis} {suffix} plot")
        return
    ax.set_xlabel(labels[axis])
    ax.set_ylabel("Normalized squared error after 20 examples")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, f"icl_maml_{axis}_{suffix}_training_dynamics")


def window_specs():
    return [
        ("icl_pos_terms01_15_s{}", "Terms 1–15"),
        ("icl_pos_terms05_19_s{}", "Terms 5–19"),
        ("icl_pos_terms06_20_s{}", "Terms 6–20"),
        ("icl_pos_terms21_35_s{}", "Terms 21–35"),
        ("icl_pos_terms26_40_s{}", "Terms 26–40"),
    ]


def plot_windows(runs, cache, output_dir, context, max_steps, individual):
    windows = window_specs()
    if individual:
        fig, axes_grid = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey=True)
        axes = axes_grid.ravel()
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        axes = None
    plotted = 0

    for window_index, (pattern, label) in enumerate(windows):
        series = []
        for seed in SEEDS:
            run = runs.get(pattern.format(seed))
            if run is None:
                continue
            steps, values = cache.dynamics(run, ICL_METRIC, context)
            points = {
                step: value
                for step, value in zip(steps, values)
                if max_steps is None or step <= max_steps
            }
            if points:
                series.append((seed, points))
        if not series:
            if individual:
                axes[window_index].set_visible(False)
            continue

        common_steps = sorted(
            set.intersection(*(set(values) for _, values in series))
        )
        if not common_steps:
            continue
        matrix = np.asarray(
            [[values[step] for step in common_steps] for _, values in series]
        )
        mean = matrix.mean(axis=0)

        if individual:
            ax = axes[window_index]
            for seed, values in series:
                xs = sorted(values)
                ax.plot(
                    xs,
                    [values[x] for x in xs],
                    linewidth=1.1,
                    alpha=0.8,
                    label=f"Seed {seed}",
                )
            plot_trajectory(
                ax,
                common_steps,
                mean,
                color="black",
                label=f"Mean (n={len(series)})",
            )
            ax.set_title(label)
            ax.grid(alpha=0.25)
        else:
            color = f"C{window_index}"
            plot_trajectory(
                ax,
                common_steps,
                mean,
                color=color,
                label=f"{label} (n={len(series)})",
            )
            if len(series) > 1:
                std = matrix.std(axis=0, ddof=1)
                ax.fill_between(
                    common_steps,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.18,
                )
        plotted += 1

    kind = "individual" if individual else "mean_std"
    if not plotted:
        plt.close(fig)
        warn(f"No position-window data at context {context}")
        return

    if individual:
        axes[-1].set_visible(False)
        for ax in axes:
            if ax.get_visible():
                ax.set_xlabel("Training steps")
                ax.set_ylabel(f"Squared error at context length {context}")
        handles, labels = next(
            (
                ax.get_legend_handles_labels()
                for ax in axes
                if ax.get_visible() and ax.lines
            ),
            ([], []),
        )
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        ax.set_xlabel("Training steps")
        ax.set_ylabel(f"Squared error at context length {context}")
        ax.grid(alpha=0.25)
        ax.legend()
    save_figure(
        fig,
        output_dir,
        f"position_windows_context{context}_{kind}_training_dynamics",
    )


def plot_windows_by_seed_contexts(
    runs,
    cache,
    output_dir,
    seed,
    contexts,
    max_steps,
):
    """One figure per seed; one panel per loss window; one line per context."""
    windows = window_specs()
    fig, axes_grid = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey=True)
    axes = axes_grid.ravel()
    linestyles = ("-", "--", ":", "-.")
    plotted_panels = 0

    for window_index, (pattern, label) in enumerate(windows):
        ax = axes[window_index]
        run = runs.get(pattern.format(seed))
        if run is None:
            ax.set_visible(False)
            continue

        plotted_contexts = 0
        for context_index, context in enumerate(contexts):
            steps, values = cache.dynamics(run, ICL_METRIC, context)
            points = [
                (step, value)
                for step, value in zip(steps, values)
                if max_steps is None or step <= max_steps
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            marker_stride = 1 if len(xs) <= 25 else math.ceil(len(xs) / 25)
            ax.plot(
                xs,
                ys,
                color=f"C{context_index}",
                linestyle=linestyles[context_index % len(linestyles)],
                marker="o",
                markersize=2.5,
                markevery=marker_stride,
                linewidth=1.35,
                label=f"Context {context}",
            )
            plotted_contexts += 1

        if not plotted_contexts:
            ax.set_visible(False)
            continue
        ax.set_title(label)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Squared error")
        ax.grid(alpha=0.25)
        plotted_panels += 1

    axes[-1].set_visible(False)
    if not plotted_panels:
        plt.close(fig)
        warn(f"No position-window data available for seed {seed}")
        return

    handles, labels = next(
        (
            ax.get_legend_handles_labels()
            for ax in axes
            if ax.get_visible() and ax.lines
        ),
        ([], []),
    )
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.suptitle(f"Position-window training dynamics — seed {seed}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        output_dir,
        f"position_windows_seed{seed}_all_contexts_training_dynamics",
    )


def plot_latest_context(runs, cache, specs, output_dir, stem, ylabel, max_context):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for spec in specs:
        run = runs.get(spec["run_name"])
        if run is None:
            continue
        step, xs, values = cache.latest(run, spec["metric"], max_context=max_context)
        if not xs:
            continue
        ax.plot(xs, values, marker="o", markersize=3, label=f"{spec['label']} (step {step})")
        plotted += 1
    if not plotted:
        plt.close(fig)
        warn(f"No data available for {stem}")
        return
    ax.set_xlabel("In-context examples")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, stem)


POSTTRAINING_SPECS = [
    {
        "label": "TF (ICL)",
        "kind": "icl",
        "path": "models/linear_regression/12cc04e3-c831-488a-ac34-0d8388192559",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, 1 step, learned LR, multi-k)",
        "kind": "maml",
        "path": "models/meta_linear_regression/b7db4b1e-0f26-42b3-aa9b-6b9df4793916",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, 2 steps, learned LR, multi-k)",
        "kind": "maml",
        "path": "models/meta_linear_regression/d0a4c860-5f8d-445b-997f-13543b957363",
        "checkpoint": 350000,
    },
    {
        "label": "TF (MAML, 1 step, learned LR, fixed length 25)",
        "kind": "maml",
        "path": "models/meta_linear_regression/6f6c6d3e-54c7-4964-ab65-8696f046d322",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, 2 steps, learned LR, fixed length 25)",
        "kind": "maml",
        "path": "models/meta_linear_regression/72bae06d-71bc-4479-9613-731907b15fa2",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, fixed length 21, 5 steps; old)",
        "kind": "maml",
        "path": "models/meta_linear_regression/maml_lr_fixed21-50bead48-f5c6-4d32-a3d8-99acf6697030",
        "checkpoint": "latest",
    },
    {
        "label": "TF (ICL, fixed length 21, last)",
        "kind": "icl",
        "path": "models/linear_regression/icl_lr_len21_last-e5e9d6b8-e334-4e67-a386-c535744f745b",
        "checkpoint": "latest",
    },
    {
        "label": "TF (ICL, fixed length 21, all)",
        "kind": "icl",
        "path": "models/linear_regression/icl_lr_len21_all-8d6e8f4c-3380-4740-b612-cc1d0d8e0f73",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, fixed length 21, 2 steps)",
        "kind": "maml",
        "path": "models/meta_linear_regression/maml_lr_fixed21_2step-f3bd835d-70f7-44f2-bfd4-d8275eb10185",
        "checkpoint": "latest",
    },
    {
        "label": "TF (MAML, 5-step bounded Meta-SGD; resumed)",
        "kind": "maml",
        "path": (
            "models/meta_linear_regression/"
            "maml_lr20_s20_q20_steps5_stable_s0-"
            "b92d63fa-31ba-4a24-9860-1728b3e5c32a"
        ),
        "checkpoint": "latest",
    },
]


def checkpoint_signature(run_path, checkpoint):
    checkpoint_path = (
        run_path / "state.pt"
        if checkpoint == "latest"
        else run_path / f"model_{checkpoint}.pt"
    )
    stat = checkpoint_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def posttraining_curves(args):
    cache_path = args.output_dir / "posttraining_context_cache.json"
    cached = {}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception as exc:
            warn(f"Ignoring invalid post-training cache: {exc}")
    if cached.get("version") != POSTTRAINING_CACHE_VERSION:
        cached = {"version": POSTTRAINING_CACHE_VERSION, "entries": {}}
    entries = cached.setdefault("entries", {})

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from eval import get_run_metrics

    curves = []
    for spec in POSTTRAINING_SPECS:
        run_path = REPO_ROOT / spec["path"]
        if not run_path.is_dir():
            warn(f"Missing post-training run directory: {run_path}")
            continue
        try:
            signature = checkpoint_signature(run_path, spec["checkpoint"])
        except OSError as exc:
            warn(f"Missing checkpoint for {spec['label']}: {exc}")
            continue

        cache_key = (
            spec["path"]
            + ":"
            + str(spec["checkpoint"])
            + f":n{args.posttraining_eval_examples}"
            + f":b{args.posttraining_eval_batch_size}"
            + f":s{args.posttraining_eval_seed}"
        )
        entry = entries.get(cache_key)
        if (
            not args.refresh_posttraining_cache
            and entry is not None
            and entry.get("signature") == signature
        ):
            curves.append((spec["label"], entry))
            progress(
                f"post-training cache hit (reusing matching evaluation): "
                f"{spec['label']}"
            )
            continue

        random.seed(args.posttraining_eval_seed)
        np.random.seed(args.posttraining_eval_seed)
        torch.manual_seed(args.posttraining_eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.posttraining_eval_seed)

        eval_name = "maml" if spec["kind"] == "maml" else "standard"
        overrides = {
            "n_points": 41,
            "num_eval_examples": args.posttraining_eval_examples,
            "batch_size": args.posttraining_eval_batch_size,
            "data_sampler_kwargs": {"data_seed": args.posttraining_eval_seed},
        }
        if spec["kind"] == "maml":
            overrides["stride"] = 1
        else:
            overrides["parallel_seed"] = args.posttraining_eval_seed

        metrics = None
        started = time.monotonic()
        progress(
            f"evaluating checkpoint: {spec['label']} "
            f"({spec['checkpoint']}, {args.posttraining_eval_examples} examples)"
        )
        try:
            metrics = get_run_metrics(
                run_path,
                step=(-1 if spec["checkpoint"] == "latest" else spec["checkpoint"]),
                cache=False,
                skip_baselines=True,
                eval_names=[eval_name],
                eval_kwargs_overrides=overrides,
            )
            values = next(iter(metrics[eval_name].values()))
            entry = {
                "signature": signature,
                "mean": values["mean"][:41],
                "low": values["bootstrap_low"][:41],
                "high": values["bootstrap_high"][:41],
            }
            entries[cache_key] = entry
            args.output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached, indent=2))
            curves.append((spec["label"], entry))
            progress(
                f"finished checkpoint: {spec['label']} "
                f"in {time.monotonic() - started:.1f}s"
            )
        except Exception as exc:
            warn(f"Post-training evaluation failed for {spec['label']}: {exc}")
            progress(
                f"failed checkpoint: {spec['label']} "
                f"after {time.monotonic() - started:.1f}s"
            )
        finally:
            metrics = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return curves


def plot_posttraining(args):
    curves = posttraining_curves(args)
    if not curves:
        warn("No post-training curves available")
        return

    normalization = float(args.context_normalization)
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("context normalization must be finite and positive")
    if not math.isfinite(args.posttraining_ymax) or args.posttraining_ymax <= 0:
        raise ValueError("post-training ymax must be finite and positive")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig_log, ax_log = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(41)
    plotted = 0
    for curve_index, (label, values) in enumerate(curves):
        mean = np.asarray(values["mean"], dtype=float) / normalization
        low = np.asarray(values["low"], dtype=float) / normalization
        high = np.asarray(values["high"], dtype=float) / normalization
        finite = np.isfinite(mean)
        if not finite.any():
            warn(f"Skipping all-nonfinite post-training curve: {label}")
            continue
        color = f"C{curve_index}"
        curve_xs = xs[: len(mean)]
        ax.plot(curve_xs, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(curve_xs, low, high, color=color, alpha=0.12)
        ax_log.plot(curve_xs, mean, color=color, linewidth=1.8, label=label)
        ax_log.fill_between(curve_xs, low, high, color=color, alpha=0.12)
        above = int(np.count_nonzero(finite & (mean > args.posttraining_ymax)))
        if above:
            warn(
                f"{label!r} has {above} points above the primary plot's "
                f"y-limit; see the log-scale companion plot"
            )
        plotted += 1

    if not plotted:
        plt.close(fig)
        plt.close(fig_log)
        warn("No finite post-training curves available")
        return

    for current_ax in (ax, ax_log):
        current_ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        current_ax.set_xlabel("In-context examples")
        current_ax.set_ylabel("Normalized squared error")
        current_ax.set_xlim(0, 40)
        current_ax.grid(alpha=0.25)
        current_ax.legend(fontsize=7.5, ncol=2)
    ax.set_ylim(0, args.posttraining_ymax)
    ax_log.set_yscale("log")
    save_figure(fig, args.output_dir, "linear_regression_posttraining_error_vs_context")
    save_figure(
        fig_log,
        args.output_dir,
        "linear_regression_posttraining_error_vs_context_logscale",
    )


def main():
    args = parse_args()
    configure_wandb_key(args.wandb_api_key_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lr5_specs = [
        {
            "run_name": "icl_lr5_l11_last5_200k_s0",
            "label": "Last 5 terms, 200k steps",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr5_l11_last4_250k_s0",
            "label": "Last 4 terms, 250k steps",
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
    lr5_position_specs = [
        {
            "run_name": "icl_lr5_l11_last1_1000k_s0",
            "label": "Last 1 term, no position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_name": "icl_lr5_l11_last1_pos_1000k_s0",
            "label": "Last 1 term, learned position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
    ]
    matched_specs = [
        {
            "run_names": [f"icl_lr20_isolated_causal_s{seed}" for seed in SEEDS],
            "label": "ICL, causal support",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
            "profile_run_name": "profile_icl_isolated_causal_h100",
            "gradient_pass_multiplier": 1,
        },
        {
            "run_names": [f"icl_lr20_isolated_bidir_s{seed}" for seed in SEEDS],
            "label": "ICL, bidirectional support",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
            "profile_run_name": "profile_icl_isolated_bidir_h100",
            "gradient_pass_multiplier": 1,
        },
        {
            "run_name": "maml_lr20_s20_q20_steps2_stable_s0",
            "label": "MAML Meta-SGD, 2 inner steps (stabilized)",
            "metric": MAML_METRIC,
            "kind": "maml",
            "max_steps": None,
            "profile_run_name": "profile_maml_steps2_h100",
            "gradient_pass_multiplier": 3,
        },
        {
            "run_name": STABLE_FIVE_RUN_NAME,
            "label": "MAML Meta-SGD, 5 inner steps (stabilized)",
            "metric": MAML_METRIC,
            "kind": "maml",
            "max_steps": None,
            "profile_run_name": "profile_maml_steps5_h100",
            "gradient_pass_multiplier": 6,
        },
    ]
    query_interaction_specs = [
        {
            "run_names": [f"icl_lr20_causal_c20_l20_s{seed}" for seed in SEEDS],
            "label": "ICL, default causal (queries interact)",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
        {
            "run_names": [f"icl_lr20_isolated_causal_s{seed}" for seed in SEEDS],
            "label": "ICL, causal (queries isolated)",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": None,
        },
    ]
    terms21_35_specs = [
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
            "label": "Terms 21–35, learned position embedding",
            "metric": ICL_METRIC,
            "kind": "icl",
            "max_steps": 400_000,
        },
    ]

    window_names = [
        pattern.format(seed)
        for pattern, _ in window_specs()
        for seed in SEEDS
    ]
    requested_names = sorted(
        set(
            [
                run_name
                for spec in (
                    lr5_specs
                    + lr5_position_specs
                    + matched_specs
                    + query_interaction_specs
                    + terms21_35_specs
                )
                for run_name in spec_run_names(spec)
            ]
            + [spec["profile_run_name"] for spec in matched_specs]
            + window_names
        )
    )

    progress(
        f"querying W&B for {len(requested_names)} named runs in "
        f"{args.entity}/{args.project}"
    )
    try:
        api = wandb.Api()
        runs = fetch_runs(api, args.entity, args.project, requested_names)
        try:
            runs[STABLE_FIVE_RUN_NAME] = api.run(
                f"{args.entity}/{args.project}/{STABLE_FIVE_RUN_ID}"
            )
        except Exception as exc:
            warn(f"Could not load stable 5-step run by ID: {exc}")
    except Exception as exc:
        raise SystemExit(f"Unable to initialize W&B API: {exc}") from exc

    progress(f"loaded {len(runs)} runs; generating live-history figures")
    cache = HistoryCache()
    progress("plotting 5D objective-budget training dynamics")
    plot_dynamics(
        runs,
        cache,
        lr5_specs,
        position=5,
        output_dir=args.output_dir,
        stem="lr5_objective_budget_training_dynamics",
        ylabel="Squared error at context length 5",
    )
    progress("plotting 5D last-1 positional training dynamics")
    plot_dynamics(
        runs,
        cache,
        lr5_position_specs,
        position=5,
        output_dir=args.output_dir,
        stem="lr5_last1_positional_training_dynamics",
        ylabel="Squared error at context length 5",
    )
    progress("plotting 5D last-1 final error versus context length")
    plot_latest_context(
        runs,
        cache,
        lr5_position_specs,
        args.output_dir,
        "lr5_last1_positional_error_vs_context",
        "Squared error",
        max_context=10,
    )

    for axis in ("tasks", "gpu_hours", "gradient_pass"):
        for individual in (False, True):
            curve_kind = "individual runs" if individual else "aggregate"
            progress(
                f"plotting ICL–MAML comparison: {axis}, {curve_kind}"
            )
            plot_matched_axis(
                runs,
                cache,
                matched_specs,
                args.output_dir,
                axis,
                individual,
                args.context_normalization,
            )

    progress("plotting ICL query-interaction comparison: aggregate")
    plot_dynamics(
        runs,
        cache,
        query_interaction_specs,
        position=20,
        output_dir=args.output_dir,
        stem="icl_query_interactions_mean_std_training_dynamics",
        ylabel="Normalized squared error after 20 examples",
        xlabel="Training tasks seen",
        normalization=args.context_normalization,
        task_axis=True,
    )
    progress("plotting ICL query-interaction comparison: individual runs")
    plot_individual_dynamics(
        runs,
        cache,
        query_interaction_specs,
        position=20,
        output_dir=args.output_dir,
        stem="icl_query_interactions_individual_training_dynamics",
        ylabel="Normalized squared error after 20 examples",
        xlabel="Training tasks seen",
        normalization=args.context_normalization,
        task_axis=True,
    )

    progress("plotting terms 21–35 positional mean and variance")
    plot_dynamics(
        runs,
        cache,
        terms21_35_specs,
        position=20,
        output_dir=args.output_dir,
        stem="terms21_35_positional_mean_std_training_dynamics",
        ylabel="Squared error at context length 20",
    )
    progress("plotting terms 21–35 positional individual runs")
    plot_individual_dynamics(
        runs,
        cache,
        terms21_35_specs,
        position=20,
        output_dir=args.output_dir,
        stem="terms21_35_positional_individual_training_dynamics",
        ylabel="Squared error at context length 20",
    )

    for context in WINDOW_CONTEXTS:
        progress(
            f"plotting position windows at context {context}: aggregate"
        )
        plot_windows(
            runs,
            cache,
            args.output_dir,
            context=context,
            max_steps=args.window_max_steps,
            individual=False,
        )
        progress(
            f"plotting position windows at context {context}: individual runs"
        )
        plot_windows(
            runs,
            cache,
            args.output_dir,
            context=context,
            max_steps=args.window_max_steps,
            individual=True,
        )

    for seed in SEEDS:
        progress(
            f"plotting position windows for seed {seed}: all evaluation contexts"
        )
        plot_windows_by_seed_contexts(
            runs,
            cache,
            args.output_dir,
            seed=seed,
            contexts=WINDOW_CONTEXTS,
            max_steps=args.window_max_steps,
        )

    if not args.skip_posttraining_eval:
        progress("starting local post-training checkpoint evaluations")
        plot_posttraining(args)
    else:
        progress("skipping local post-training checkpoint evaluations")
    progress(f"complete; plots available in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
