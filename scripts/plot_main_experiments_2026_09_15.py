#!/usr/bin/env python3
"""Refresh and plot the complete breadth-first 2026-09-15 experiment suite.

All ordinary and dense inter-query training curves are fetched live from W&B.
Crossed-mask checkpoint evaluations are refreshed through evaluate_interquery.py
before their cache is read. The script tolerates missing, queued, and partial
runs and overwrites the same figures with the latest available results.
"""

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch


ICL_LR = "linear_regression_eval/pointwise/loss"
ICL_DT = "decision_tree_eval/pointwise/loss"
MAML = "maml_eval/pointwise/loss"
MULTIFUNCTION = "multi_function_linear_eval/pointwise/loss"
SEEDS_2 = (0, 154645467)
SEEDS_3 = (0, 154645467, 65765443)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAML_INTERQUERY_RUN_ID = "b92d63fa-31ba-4a24-9860-1728b3e5c32a"
INTERQUERY_SPECS = (
    (
        "causal", 0, "0897d9d4-6bc5-4ecf-92da-eec06de6bfdb",
        "models/linear_regression/icl_lr20_causal_c20_l20_s0-0897d9d4-6bc5-4ecf-92da-eec06de6bfdb",
    ),
    (
        "causal", 154645467, "31212caa-f704-4531-a3e3-8fe95435f65b",
        "models/linear_regression/icl_lr20_causal_c20_l20_s154645467-31212caa-f704-4531-a3e3-8fe95435f65b",
    ),
    (
        "causal", 65765443, "c0a03422-4711-4a13-94f9-ce58245e287a",
        "models/linear_regression/icl_lr20_causal_c20_l20_s65765443-c0a03422-4711-4a13-94f9-ce58245e287a",
    ),
    (
        "isolated", 0, "exp20260911-icl-iso-causal-s0",
        "models/linear_regression/icl_lr20_isolated_causal_s0-exp20260911-icl-iso-causal-s0",
    ),
    (
        "isolated", 154645467, "exp20260911-icl-iso-causal-s154645467",
        "models/linear_regression/icl_lr20_isolated_causal_s154645467-exp20260911-icl-iso-causal-s154645467",
    ),
    (
        "isolated", 65765443, "exp20260911-icl-iso-causal-s65765443",
        "models/linear_regression/icl_lr20_isolated_causal_s65765443-exp20260911-icl-iso-causal-s65765443",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("all", "followups", "legacy"), default="all")
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/main-experiments-2026-09-15"),
    )
    parser.add_argument("--wandb-api-key-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--interquery-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--interquery-num-eval-examples", type=int, default=1280)
    parser.add_argument("--interquery-eval-batch-size", type=int, default=64)
    parser.add_argument("--interquery-validation-seed", type=int, default=81173)
    parser.add_argument("--interquery-test-seed", type=int, default=104729)
    parser.add_argument("--context-num-eval-examples", type=int, default=1280)
    parser.add_argument("--context-eval-batch-size", type=int, default=32)
    parser.add_argument("--context-eval-seed", type=int, default=0)
    parser.add_argument("--refresh-context-comparison", action="store_true")
    parser.add_argument("--context-comparison-only", action="store_true")
    parser.add_argument(
        "--interquery-thresholds",
        nargs="+",
        type=float,
        default=(0.8, 0.5, 0.2),
    )
    return parser.parse_args()


def configure_wandb_key(explicit_path):
    if os.environ.get("WANDB_API_KEY"):
        return
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    if os.environ.get("WANDB_API_KEY_FILE_AT"):
        candidates.append(Path(os.environ["WANDB_API_KEY_FILE_AT"]))
    candidates.append(Path(__file__).resolve().parents[2] / ".wandb-api-key")
    for path in candidates:
        path = Path(path).expanduser()
        if path.is_file():
            value = path.read_text().strip()
            if value:
                os.environ["WANDB_API_KEY"] = value
                return


def run_rank(run):
    try:
        step = int(run.summary.get("_step", -1))
    except (TypeError, ValueError):
        step = -1
    return step, str(run.created_at)


def fetch_runs(api, entity, project, names):
    names = sorted(set(names))
    result = {}
    try:
        for run in api.runs(
            f"{entity}/{project}", filters={"display_name": {"$in": names}}
        ):
            if run.name in names and (
                run.name not in result or run_rank(run) > run_rank(result[run.name])
            ):
                result[run.name] = run
    except Exception as exc:
        warnings.warn(f"could not list W&B runs: {exc}")
    missing = [name for name in names if name not in result]
    if missing:
        warnings.warn("runs not available yet: " + ", ".join(missing))
    print(f"[wandb] found {len(result)}/{len(names)} requested runs", flush=True)
    return result


def fetch_exact_interquery_runs(api, entity, project):
    runs = {}
    for mode, seed, run_id, _ in INTERQUERY_SPECS:
        try:
            runs[(mode, seed)] = api.run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            warnings.warn(
                f"could not fetch inter-query {mode} seed={seed} ({run_id}): {exc}"
            )
    try:
        maml = api.run(f"{entity}/{project}/{MAML_INTERQUERY_RUN_ID}")
    except Exception as exc:
        warnings.warn(f"could not fetch inter-query MAML reference: {exc}")
        maml = None
    print(
        f"[wandb] found {len(runs)}/{len(INTERQUERY_SPECS)} exact inter-query ICL runs",
        flush=True,
    )
    return runs, maml


class History:
    def __init__(self):
        self.scalar_cache = {}
        self.pointwise_cache = {}

    def scalar(self, run, key):
        cache_key = (run.id, key)
        if cache_key not in self.scalar_cache:
            points = {}
            try:
                rows = run.history(keys=[key], pandas=False, samples=10000)
                for row in rows:
                    step, value = row.get("_step"), row.get(key)
                    try:
                        value = float(value)
                        if step is not None and math.isfinite(value):
                            points[int(step)] = value
                    except (TypeError, ValueError):
                        pass
            except Exception as exc:
                warnings.warn(f"could not read {run.name}: {key}: {exc}")
            steps = np.asarray(sorted(points), dtype=int)
            self.scalar_cache[cache_key] = (
                steps,
                np.asarray([points[step] for step in steps]),
            )
        return self.scalar_cache[cache_key]

    def pointwise(self, run, prefix, indices):
        indices = tuple(indices)
        cache_key = (run.id, prefix, indices)
        if cache_key not in self.pointwise_cache:
            flattened = [f"{prefix}.{index}" for index in indices]
            by_step = {}
            try:
                rows = run.history(keys=flattened, pandas=False, samples=10000)
                for row in rows:
                    step = row.get("_step")
                    if step is None:
                        continue
                    values = by_step.setdefault(int(step), {})
                    nested = row.get(prefix)
                    if isinstance(nested, Mapping):
                        for key, value in nested.items():
                            try:
                                values[int(key)] = float(value)
                            except (TypeError, ValueError):
                                pass
                    for index, key in zip(indices, flattened):
                        try:
                            value = float(row.get(key))
                            if math.isfinite(value):
                                values[index] = value
                        except (TypeError, ValueError):
                            pass
            except Exception as exc:
                warnings.warn(f"could not read {run.name}: {prefix}: {exc}")
            self.pointwise_cache[cache_key] = by_step
        return self.pointwise_cache[cache_key]

    def point_series(self, run, prefix, index):
        rows = self.pointwise(run, prefix, (index,))
        points = {
            step: values[index]
            for step, values in rows.items()
            if index in values and math.isfinite(values[index])
        }
        steps = np.asarray(sorted(points), dtype=int)
        return steps, np.asarray([points[step] for step in steps])

    def latest_curve(self, run, prefix, indices):
        rows = self.pointwise(run, prefix, indices)
        complete = [(step, values) for step, values in rows.items() if values]
        if not complete:
            return np.asarray([]), np.asarray([])
        _, values = max(complete, key=lambda item: item[0])
        xs = np.asarray(sorted(values))
        return xs, np.asarray([values[index] for index in xs])

    def point_mean_series(self, run, prefix, indices):
        indices = tuple(indices)
        rows = self.pointwise(run, prefix, indices)
        points = {
            step: float(np.mean([values[index] for index in indices]))
            for step, values in rows.items()
            if all(
                index in values and math.isfinite(values[index])
                for index in indices
            )
        }
        steps = np.asarray(sorted(points), dtype=int)
        return steps, np.asarray([points[step] for step in steps])


def save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {stem}.png", flush=True)


def aligned_mean(series):
    series = [(steps, values) for steps, values in series if len(steps)]
    if not series:
        return np.asarray([]), np.asarray([]), np.asarray([])
    common = sorted(set.intersection(*(set(steps.tolist()) for steps, _ in series)))
    if not common:
        return np.asarray([]), np.asarray([]), np.asarray([])
    stacked = []
    for steps, values in series:
        by_step = dict(zip(steps.tolist(), values.tolist()))
        stacked.append([by_step[step] for step in common])
    stacked = np.asarray(stacked)
    return np.asarray(common), stacked.mean(axis=0), stacked.std(axis=0)


def refresh_interquery_evaluations(args):
    """Refresh the expensive crossed-mask cache before any plot reads it."""
    cache_dir = args.output_dir / "interquery"
    cache_dir.mkdir(parents=True, exist_ok=True)
    evaluator = REPO_ROOT / "scripts" / "evaluate_interquery.py"
    records = []
    for mode, seed, _, relative_run_dir in INTERQUERY_SPECS:
        output = cache_dir / f"{mode}_s{seed}.json"
        command = [
            sys.executable,
            str(evaluator),
            "--run-dir",
            str(REPO_ROOT / relative_run_dir),
            "--training-mode",
            mode,
            "--seed",
            str(seed),
            "--output",
            str(output),
            "--num-eval-examples",
            str(args.interquery_num_eval_examples),
            "--eval-batch-size",
            str(args.interquery_eval_batch_size),
            "--validation-seed",
            str(args.interquery_validation_seed),
            "--test-seed",
            str(args.interquery_test_seed),
            "--device",
            args.interquery_device,
        ]
        print(f"[interquery] refresh {mode} seed={seed}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        record = json.loads(output.read_text())
        if "cache" not in record:
            raise RuntimeError(f"inter-query evaluator did not refresh cache metadata: {output}")
        records.append(record)
    return records


def refresh_context_comparison(args):
    """Validate or refresh all four dense checkpoint curves."""
    cache_path = args.output_dir / "context_comparison.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_context_comparison.py"),
        "--output",
        str(cache_path),
        "--num-eval-examples",
        str(args.context_num_eval_examples),
        "--eval-batch-size",
        str(args.context_eval_batch_size),
        "--eval-seed",
        str(args.context_eval_seed),
    ]
    if args.refresh_context_comparison:
        command.append("--force")
    print("[context] validating dense checkpoint evaluation", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(cache_path.read_text())


def query_curve(record, split, mask, checkpoint=-1):
    support = record["support"]
    return np.asarray(record["checkpoints"][checkpoint][split][mask][support:])


def selection_score(record):
    return float(np.mean(query_curve(record, "validation", "isolated")))


def first_crossing(steps, values, threshold, batch_size):
    for step, value in zip(steps, values):
        if math.isfinite(float(value)) and float(value) <= threshold:
            return int(step) * batch_size
    return None


def curve_area(steps, values, batch_size):
    steps = np.asarray(steps)
    values = np.asarray(values)
    keep = (steps > 0) & np.isfinite(values)
    steps, values = steps[keep], values[keep]
    if len(steps) < 2:
        return None
    log_episodes = np.log(steps.astype(float) * batch_size)
    return float(
        np.trapezoid(values, log_episodes)
        / (log_episodes[-1] - log_episodes[0])
    )


def dense_icl_seed_curves(records, exact_runs, history, normalization=20.0):
    curves = {"causal": {}, "isolated": {}}
    missing = []
    for record in records:
        mode, seed = record["training_mode"], record["seed"]
        run = exact_runs.get((mode, seed))
        if run is None:
            missing.append(f"{mode} seed={seed}")
            continue
        start = record["support"]
        stop = start + record["queries"]
        steps, values = history.point_mean_series(
            run, ICL_LR, range(start, stop)
        )
        if not len(steps):
            missing.append(f"{mode} seed={seed} (no complete pointwise history)")
            continue
        curves[mode][seed] = (steps, values / normalization)
    if missing:
        raise RuntimeError(
            "dense inter-query W&B histories unavailable: " + ", ".join(missing)
        )
    return curves


def maml_interquery_curve(run, history, normalization=20.0):
    if run is None:
        return None
    steps, values = history.point_series(run, MAML, 20)
    if not len(steps):
        warnings.warn("inter-query MAML reference has no pointwise history")
        return None
    return {
        "run_id": run.id,
        "run_name": run.name,
        "batch_size": run_batch_size(run, 8),
        "steps": steps,
        "values": values / normalization,
    }


def threshold_summary(
    curves, seed_curves, groups, thresholds, batch_size, maml
):
    result = {}
    for threshold in thresholds:
        n_causal = first_crossing(*curves["causal"], threshold, batch_size)
        n_isolated = first_crossing(*curves["isolated"], threshold, batch_size)
        n_maml = (
            first_crossing(maml["steps"], maml["values"], threshold, maml["batch_size"])
            if maml is not None
            else None
        )
        r_iq = (
            n_isolated / n_causal
            if n_isolated is not None and n_causal not in (None, 0)
            else None
        )
        denominator = (
            n_maml - n_causal
            if n_maml is not None and n_causal is not None
            else None
        )
        c_iq = (
            (n_isolated - n_causal) / denominator
            if n_isolated is not None and denominator is not None and denominator > 0
            else None
        )
        crossings = {}
        for mode in ("causal", "isolated"):
            seed_values = {}
            for record in groups[mode]:
                seed = record["seed"]
                curve = seed_curves[mode].get(seed)
                seed_values[str(seed)] = (
                    first_crossing(*curve, threshold, batch_size)
                    if curve is not None
                    else None
                )
            crossings[mode] = {
                "episodes_by_seed": seed_values,
                "converged": sum(value is not None for value in seed_values.values()),
                "total": len(seed_values),
            }
        result[str(threshold)] = {
            "N_causal": n_causal,
            "N_isolated": n_isolated,
            "N_MAML": n_maml,
            "R_IQ": r_iq,
            "C_IQ": c_iq,
            "C_IQ_denominator_positive": denominator is not None and denominator > 0,
            "seed_crossings": crossings,
        }
    return result


def plot_interquery(
    records, exact_runs, maml_run, history, output_dir, batch_size, thresholds
):
    groups = {
        mode: sorted(
            (
                record
                for record in records
                if record["training_mode"] == mode
            ),
            key=lambda record: record["seed"],
        )
        for mode in ("causal", "isolated")
    }
    if any(not group for group in groups.values()):
        raise ValueError("both causal- and isolated-trained records are required")
    best = {mode: min(group, key=selection_score) for mode, group in groups.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, evaluation_mask in zip(axes, ("natural", "isolated")):
        for mode, color in (("causal", "tab:blue"), ("isolated", "tab:orange")):
            values = np.stack(
                [
                    query_curve(record, "test", evaluation_mask)
                    for record in groups[mode]
                ]
            )
            mean, std = values.mean(axis=0), values.std(axis=0)
            x = np.arange(1, values.shape[1] + 1)
            axis.plot(x, mean, color=color, label=f"trained {mode}")
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
            axis.plot(
                x,
                query_curve(best[mode], "test", evaluation_mask),
                color=color,
                linestyle="--",
                label=f"best {mode}",
            )
        axis.set_title(f"evaluation: {evaluation_mask}")
        axis.set_xlabel("query offset")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("normalized squared error")
    axes[1].legend(fontsize=8)
    save(fig, output_dir, "interquery_query_offset")

    seed_curves = dense_icl_seed_curves(records, exact_runs, history)
    all_seed_stats = {}
    all_seed_curves = {}
    for mode in ("causal", "isolated"):
        steps, mean, std = aligned_mean(list(seed_curves[mode].values()))
        all_seed_stats[mode] = (steps, mean, std)
        all_seed_curves[mode] = (steps, mean)
    best_curves = {
        mode: seed_curves[mode][best[mode]["seed"]]
        for mode in ("causal", "isolated")
    }
    maml = maml_interquery_curve(maml_run, history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for mode, color in (("causal", "tab:blue"), ("isolated", "tab:orange")):
        steps, mean, std = all_seed_stats[mode]
        x = steps * batch_size
        axes[0].plot(x, mean, color=color, label=f"ICL {mode}")
        axes[0].fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        best_steps, best_values = best_curves[mode]
        axes[1].plot(
            best_steps * batch_size,
            best_values,
            color=color,
            label=f"ICL {mode}",
        )
    if maml is not None:
        for axis in axes:
            axis.plot(
                maml["steps"] * maml["batch_size"],
                maml["values"],
                color="tab:red",
                label="MAML-5 (seed 0)",
            )
    for axis, title in zip(
        axes, ("all-seed mean", "validation-selected best")
    ):
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("training episodes")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("held-out 20-query mean NMSE")
    axes[1].legend(fontsize=8)
    save(fig, output_dir, "interquery_threshold_dynamics")

    summary = {
        "definitions": {
            "N": "first observed fixed held-out W&B evaluation crossing, in training episodes",
            "R_IQ": "N_isolated / N_causal",
            "C_IQ": "(N_isolated - N_causal) / (N_MAML - N_causal); only when denominator > 0",
            "protocol": "causal-trained/natural inference versus isolated-trained/isolated inference",
            "selection": "best seed chosen only by final validation isolated-query mean NMSE",
            "training_curve_source": (
                "live W&B pointwise evaluations, averaged over query indices 20:40"
            ),
            "crossed_mask_source": (
                "evaluate_interquery.py cache, validated and refreshed by this plotter"
            ),
        },
        "best": {},
        "all_seed": {},
        "cross_evaluation_cache": [
            {
                "training_mode": record["training_mode"],
                "seed": record["seed"],
                **record["cache"],
            }
            for record in records
        ],
    }
    for mode, group in groups.items():
        summary["best"][mode] = {
            "seed": best[mode]["seed"],
            "validation_selection_score": selection_score(best[mode]),
            "test": best[mode]["final_test_query_nmse"],
            "inference_relative_gain": best[mode][
                "final_test_inference_relative_gain"
            ],
            "dense_wandb_points": len(best_curves[mode][0]),
        }
        gains = [
            record["final_test_inference_relative_gain"] for record in group
        ]
        summary["all_seed"][mode] = {
            "seeds": [record["seed"] for record in group],
            "inference_relative_gain_mean": float(np.mean(gains)),
            "inference_relative_gain_std": float(np.std(gains)),
            "dense_wandb_points": len(all_seed_curves[mode][0]),
        }

    summary["all_seed"]["thresholds"] = threshold_summary(
        all_seed_curves,
        seed_curves,
        groups,
        thresholds,
        batch_size,
        maml,
    )
    summary["best"]["thresholds"] = threshold_summary(
        best_curves,
        seed_curves,
        groups,
        thresholds,
        batch_size,
        maml,
    )
    for label, curves in (("all_seed", all_seed_curves), ("best", best_curves)):
        areas = {
            mode: curve_area(*curves[mode], batch_size)
            for mode in ("causal", "isolated")
        }
        areas["relative_causal_AULC_reduction"] = (
            (areas["isolated"] - areas["causal"]) / areas["isolated"]
            if areas["isolated"] not in (None, 0)
            else None
        )
        summary[label]["log_episode_AULC"] = areas

    first_query_differences = []
    for record in records:
        for checkpoint in record["checkpoints"]:
            support = record["support"]
            first_query_differences.append(
                abs(
                    checkpoint["test"]["natural"][support]
                    - checkpoint["test"]["isolated"][support]
                )
            )
    summary["first_query_correctness_control"] = {
        "max_absolute_nmse_difference": float(max(first_query_differences)),
        "expected": 0.0,
    }
    summary["maml_reference"] = (
        {
            "run_id": maml["run_id"],
            "run_name": maml["run_name"],
            "metric": f"{MAML}.20",
            "dense_wandb_points": len(maml["steps"]),
        }
        if maml is not None
        else None
    )

    summary_path = output_dir / "interquery_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    print("[plot] interquery_summary.json", flush=True)



def trailing_mean(steps, values, window=51):
    """Return a trailing rolling mean without using future observations."""
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=float)
    window = min(int(window), len(values))
    if window <= 1:
        return steps, values
    kernel = np.ones(window, dtype=float) / window
    return steps[window - 1 :], np.convolve(values, kernel, mode="valid")


def run_batch_size(run, fallback):
    try:
        meta = run.config.get("meta", {})
        if isinstance(meta, Mapping) and meta.get("meta_batch_size") is not None:
            return int(meta["meta_batch_size"])
        return int(run.config.get("training", {}).get("batch_size", fallback))
    except (AttributeError, TypeError, ValueError):
        return fallback


def plot_mean_line(ax, runs, history, key, label, color, normalization=1.0,
                   x_multiplier=64, linestyle="-"):
    available = [run for name in runs if (run := RUNS.get(name)) is not None]
    steps, mean, std = aligned_mean(
        [history.scalar(run, key) for run in available]
    )
    if not len(steps):
        return False
    x = steps * x_multiplier
    mean, std = mean / normalization, std / normalization
    ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
    if len(available) > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    return True


def plot_mean_point_line(ax, names, history, prefix, index, label, color,
                         normalization, x_multiplier, linestyle="-"):
    available = [run for name in names if (run := RUNS.get(name)) is not None]
    steps, mean, std = aligned_mean(
        [history.point_series(run, prefix, index) for run in available]
    )
    if not len(steps):
        return False
    x = steps * x_multiplier
    mean, std = mean / normalization, std / normalization
    ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
    if len(available) > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    return True


def supervision_specs():
    return [
        ("ICL q=1", [f"icl_lr20_s20_q1_s{s}" for s in SEEDS_2], ICL_LR, "tab:blue", "-", 1),
        ("ICL q=5", [f"icl_lr20_s20_q5_s{s}" for s in SEEDS_2], ICL_LR, "tab:green", "-", 5),
        ("ICL q=5 isolated", [f"icl_lr20_s20_q5_isolated_s{s}" for s in SEEDS_2], ICL_LR, "tab:green", "--", 5),
        ("ICL q=20", [f"icl_lr20_causal_c20_l20_s{s}" for s in SEEDS_3], ICL_LR, "tab:purple", "-", 20),
        ("ICL q=20 isolated", [f"icl_lr20_isolated_causal_s{s}" for s in SEEDS_3], ICL_LR, "tab:purple", "--", 20),
        ("MAML-5 q=1", [f"maml_lr20_q1_guard_s{s}" for s in SEEDS_2], MAML, "tab:orange", "-", 1),
        ("MAML-5 q=5", [f"maml_lr20_q5_guard_s{s}" for s in SEEDS_2], MAML, "tab:red", "-", 5),
        ("MAML-5 q=20", [f"maml_lr20_q20_guard_s{s}" for s in SEEDS_2], MAML, "tab:brown", "-", 20),
    ]


def plot_supervision(history, output_dir, batch_size, target_axis=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    any_line = False
    for label, names, prefix, color, style, q in supervision_specs():
        axis = axes[1] if label.startswith("MAML") else axes[0]
        available = next((RUNS[name] for name in names if name in RUNS), None)
        fallback = 8 if label.startswith("MAML") else batch_size
        multiplier = run_batch_size(available, fallback) * (q if target_axis else 1) if available else fallback * (q if target_axis else 1)
        any_line |= plot_mean_point_line(
            axis, names, history, prefix, 20, label, color, 20.0, multiplier, style
        )
    if not any_line:
        plt.close(fig)
        return
    for axis, title in zip(axes, ("meta-ICL", "five-step MAML")):
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("supervised query targets" if target_axis else "training episodes")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("first-query NMSE after 20 supports")
    save(fig, output_dir, "supervision_sweep_targets" if target_axis else "supervision_sweep_episodes")


def plot_context_generalization(history, output_dir):
    regimes = {
        "mixed": ("10, 15, 20", (10, 15, 20)),
        "under": ("5, 10, 15", (5, 10, 15)),
        "over": ("25, 30, 35", (25, 30, 35)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    found = False
    for axis, (regime, (label, train_sizes)) in zip(axes, regimes.items()):
        specs = (
            (f"icl_lr20_context_joint_{regime}_s0_v2", ICL_LR,
             "meta-ICL joint", "tab:blue", range(41)),
            (f"maml_lr20_context_{regime}_parallel_i5_s0_v2", MAML,
             "MAML-5 independent", "tab:red", range(0, 41, 5)),
            (f"maml_lr20_context_{regime}_sequential_i5_s0_v2", MAML,
             "MAML-5 sequential", "tab:purple", range(0, 41, 5)),
        )
        for name, prefix, method, color, indices in specs:
            run = RUNS.get(name)
            if run is None:
                continue
            label_to_show = method
            if prefix == MAML:
                snapshots = []
                for step, values in history.pointwise(run, prefix, indices).items():
                    observed = [value for index, value in values.items()
                                if index > 0 and np.isfinite(value)]
                    if observed and max(observed) / 20.0 < 3.0:
                        snapshots.append((step, values))
                if snapshots:
                    selected_step, values = max(snapshots, key=lambda item: item[0])
                    xs = np.asarray(sorted(values))
                    ys = np.asarray([values[index] for index in xs])
                    label_to_show = f"{method} (stable eval {selected_step:,})"
                else:
                    xs, ys = np.asarray([]), np.asarray([])
            else:
                xs, ys = history.latest_curve(run, prefix, indices)
            if len(xs):
                axis.plot(xs, ys / 20.0, marker="o", markersize=3, color=color, label=label_to_show)
                found = True
        for size in train_sizes:
            axis.axvline(size, color="0.75", linewidth=0.8, linestyle=":")
        axis.set_title(f"{regime}: joint train {{{label}}}")
        axis.set_xlabel("support examples at inference")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("held-out NMSE")
    if not found:
        axes[1].text(
            0.5, 0.5, "corrected v2 runs pending",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    save(fig, output_dir, "context_generalization_final")


def plot_multifunction(history, output_dir, batch_size):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    found = False
    for count in (1, 2, 3):
        icl_run = RUNS.get(f"icl_lr5_multifunction_m{count}_s0")
        if icl_run is not None:
            steps, values = history.point_series(icl_run, MULTIFUNCTION, 5 * count)
            if len(steps):
                ax.plot(
                    steps * run_batch_size(icl_run, batch_size), values / 5.0, color=colors[count],
                    label=f"meta-ICL M={count} (held-out)",
                )
                found = True

        maml_run = RUNS.get(f"maml_lr5_multifunction_m{count}_i5_s0")
        if maml_run is not None:
            steps, values = history.point_series(maml_run, MAML, 5 * count)
            if len(steps):
                ax.plot(
                    steps * run_batch_size(maml_run, 8), values / 5.0, color=colors[count],
                    linestyle="--", label=f"MAML-5 M={count} (held-out)",
                )
                found = True
            else:
                # Existing runs predate tagged held-out evaluation. Retain a
                # clearly labelled provisional training trace until they resume.
                steps, values = history.scalar(maml_run, "meta_train/query_loss")
                if len(steps):
                    smoothed_steps, smoothed = trailing_mean(steps, values, 51)
                    ax.plot(
                        smoothed_steps * run_batch_size(maml_run, 8), smoothed / 5.0,
                        color=colors[count], linestyle=":",
                        label=f"MAML-5 M={count} (train, rolling mean)",
                    )
                    found = True
    if not found:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("query NMSE (MSE / 5)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, output_dir, "multifunction_training_dynamics")


def plot_pretraining(history, output_dir, batch_size):
    run = RUNS.get("segmented_pretraining_s0_v2")
    if run is None:
        return
    families = (
        "coordinate_affine", "coordinate_offset", "coordinate_quadratic",
        "decision_stump", "linear_regression", "decision_tree",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    found = False
    for name in families:
        steps, values = history.scalar(run, f"segment_family/{name}")
        if len(steps):
            smooth_steps, smooth_values = trailing_mean(steps, values, 51)
            axes[0].plot(steps * batch_size, values, color="0.75", alpha=0.08, linewidth=0.4)
            axes[0].plot(smooth_steps * batch_size, smooth_values, label=name)
            found = True
    for offset in range(1, 6):
        steps, values = history.scalar(run, f"segment_offset/loss_{offset}")
        if len(steps):
            smooth_steps, smooth_values = trailing_mean(steps, values, 51)
            axes[1].plot(steps * batch_size, values, color="0.75", alpha=0.08, linewidth=0.4)
            axes[1].plot(smooth_steps * batch_size, smooth_values, label=f"offset {offset}")
            found = True
    if not found:
        plt.close(fig)
        return
    for axis, title in zip(axes, ("function family", "position within segment")):
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("training sequences")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("NMSE (unit-variance outputs)")
    save(fig, output_dir, "segmented_pretraining_dynamics")


def plot_stability(history, output_dir, batch_size):
    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for seed, color in zip(SEEDS_2, ("tab:red", "tab:pink")):
        run = RUNS.get(f"maml_lr20_s20_q20_steps5_stable_s{seed}")
        if run is None:
            continue
        steps, values = history.point_series(run, MAML, 20)
        keep = np.isfinite(values)
        if seed == 154645467:
            keep &= steps <= 150000  # latest retained pre-divergence checkpoint
        if np.any(keep):
            label = f"seed {seed}" + (" (through 150k)" if seed == 154645467 else "")
            ax.plot(steps[keep] * run_batch_size(run, 8), values[keep] / 20.0,
                    color=color, label=label)
            found = True
    pilot = RUNS.get("maml_lr20_control_pilot_s154")
    if pilot is not None:
        steps, values = history.point_series(pilot, MAML, 20)
        keep = np.isfinite(values) & (steps <= 65000)
        if np.any(keep):
            pilot_batch = run_batch_size(pilot, 64)
            ax.axvline(150000 * pilot_batch, color="0.5", linestyle=":", linewidth=1)
            ax.plot((150000 + steps[keep]) * pilot_batch, values[keep] / 20.0,
                    color="tab:pink", linestyle="--",
                    label="seed 154645467: +65k unguarded (optimizer reset)")
            found = True
    if not found:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("MAML-5 NMSE after 20 supports")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output_dir, "maml5_stability")


def plot_decision_trees(history, output_dir, batch_size):
    specs = (
        ("icl_dt_s65_q20_causal_s0", ICL_DT, "ICL causal", "tab:blue", range(101)),
        ("icl_dt_s65_q20_isolated_s0", ICL_DT, "ICL isolated", "tab:orange", range(101)),
        ("maml_dt_s65_q20_i5_s0_v2", MAML, "MAML-5", "tab:red", (0, 65, 100)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    found = False
    for name, prefix, label, color, indices in specs:
        run = RUNS.get(name)
        if run is None:
            continue
        steps, values = history.point_series(run, prefix, 65)
        if len(steps):
            axes[0].plot(steps * run_batch_size(run, batch_size), values, color=color, label=label)
            found = True
        xs, ys = history.latest_curve(run, prefix, indices)
        if len(xs):
            axes[1].plot(xs, ys, color=color, label=label)
            found = True
    if not found:
        plt.close(fig)
        return
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training episodes")
    axes[0].set_ylabel("NMSE after 65 supports")
    axes[1].set_xlabel("support examples at inference")
    axes[1].set_ylabel("final NMSE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(fig, output_dir, "decision_tree_results")


def plot_context_comparison(record, output_dir):
    """Compare the selected 20/20 ICL checkpoints with five-step MAML."""
    normalization = float(record["metadata"]["normalization"])
    curves = record["curves"]
    order = (
        ("icl_causal", "#1f77b4"),
        ("icl_isolated", "#ff7f0e"),
        ("maml5_seed0", "#d62728"),
        ("maml5_seed154_warmstart_control", "#9467bd"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": (1.25, 1)}
    )
    for key, color in order:
        curve = curves[key]
        mean = np.asarray(curve["mean"], dtype=float) / normalization
        low = np.asarray(curve["low"], dtype=float) / normalization
        high = np.asarray(curve["high"], dtype=float) / normalization
        xs = np.arange(len(mean))
        for axis in axes:
            axis.plot(xs, mean, color=color, linewidth=2.2, label=curve["label"])
            axis.fill_between(xs, low, high, color=color, alpha=0.13)

    for axis in axes:
        axis.axhline(1.0, color="0.45", linestyle="--", linewidth=1)
        axis.axvline(20, color="0.45", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.25)
        axis.set_xlabel("sequence position / preceding examples")
    axes[0].set_xlim(0, 40)
    axes[0].set_ylim(0, 1.35)
    axes[0].set_ylabel("normalized squared error (MSE / 20)")
    axes[0].set_title("full context range")
    axes[0].text(
        0.025, 0.97, "MAML is off-scale at short contexts",
        transform=axes[0].transAxes, va="top", fontsize=8.5, color="0.35",
    )
    axes[1].set_xlim(20, 40)
    tail_ceiling = max(
        np.nanmax(np.asarray(curves[key]["high"], dtype=float)[20:41])
        / normalization
        for key, _ in order
    )
    axes[1].set_ylim(0, max(0.06, 1.08 * tail_ceiling))
    axes[1].set_title("20-support boundary and longer sequences")
    axes[1].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("20-support/20-query ICL versus five-step MAML seeds", fontsize=14)
    fig.tight_layout()
    save(fig, output_dir, "lr20_20x20_context_comparison")


def plot_context_comparison_pending(output_dir):
    """Replace an older PNG rather than leaving stale checkpoint data visible."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.5, 0.5,
            "GPU checkpoint evaluation pending\n"
            "The cached 20/20 comparison no longer matches the current code/checkpoints.",
            ha="center", va="center", transform=ax.transAxes)
    save(fig, output_dir, "lr20_20x20_context_comparison")


def plot_experiment_design_schematic(output_dir):
    """Publication-ready table summarizing what information each design shares."""
    columns = ("experiment", "episode / adaptation schematic", "one outer update uses", "isolates")
    rows = [
        ("supervision q", "S20 | Q1...Qq", "q query losses", "amount of supervision"),
        ("context ICL", "one prompt; losses at {k1,k2,k3}", "mean of 3 losses together", "context-length transfer"),
        ("MAML independent", "theta -> S_k1 / S_k2 / S_k3", "3 separate adaptations + mean Q", "independent support sizes"),
        ("MAML sequential", "theta -> S_k1 -> phi1 -> S_k2 -> phi2 -> S_k3", "Q(phi1), Q(phi2), Q(phi3)", "interacting adaptation stages"),
        ("inter-query causal", "S20 | Q1,y1 -> Q2,y2 -> ...", "all query losses", "reuse of earlier queries"),
        ("inter-query isolated", "S20 -> Qj, independently for each j", "all query losses", "no query-query reuse"),
        ("multiple functions", "interleaved [A],[B],[C] supports | tagged queries", "joint tagged-query loss", "within-episode task binding"),
        ("segmented pretrain", "unmarked fA segment | fB segment | ...", "loss at every sequence point", "boundary/family discovery"),
        ("decision tree 65/20", "65 supports | 20 queries", "matched ICL or MAML objective", "nonlinear function class"),
    ]
    fig, ax = plt.subplots(figsize=(17, 8.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
        colWidths=(0.15, 0.36, 0.27, 0.22), loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("#26364a")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6f9")
        if row > 0 and col == 1:
            cell.get_text().set_fontfamily("monospace")
    ax.set_title(
        "Experimental designs: what is shared, adapted, and supervised",
        fontsize=16, pad=16,
    )
    save(fig, output_dir, "experiment_design_schematic")


def requested_names():
    names = {name for _, group, _, _, _, _ in supervision_specs() for name in group}
    for regime in ("mixed", "under", "over"):
        names.add(f"icl_lr20_context_joint_{regime}_s0_v2")
        names.add(f"maml_lr20_context_{regime}_parallel_i5_s0_v2")
        names.add(f"maml_lr20_context_{regime}_sequential_i5_s0_v2")
    for count in (1, 2, 3):
        names.add(f"icl_lr5_multifunction_m{count}_s0")
        names.add(f"maml_lr5_multifunction_m{count}_i5_s0")
    names.update(
        {
            "segmented_pretraining_s0_v2",
            "maml_lr20_control_pilot_s154",
            "icl_dt_s65_q20_causal_s0",
            "icl_dt_s65_q20_isolated_s0",
            "maml_dt_s65_q20_i5_s0_v2",
        }
    )
    return names


RUNS = {}


def main():
    global RUNS
    args = parse_args()
    if args.suite in ("all", "followups"):
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "plot_followup_experiments_2026_09_21.py"),
            "--entity", args.entity, "--project", args.project,
        ]
        if args.wandb_api_key_file:
            command.extend(["--wandb-api-key-file", str(args.wandb_api_key_file)])
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if args.suite == "followups":
            return
    args.output_dir = args.output_dir.resolve()

    if torch.cuda.is_available():
        context_comparison = refresh_context_comparison(args)
    else:
        from evaluate_context_comparison import cache_matches, expected_metadata
        cache_path = args.output_dir / "context_comparison.json"
        probe = argparse.Namespace(
            num_eval_examples=args.context_num_eval_examples,
            eval_batch_size=args.context_eval_batch_size,
            eval_seed=args.context_eval_seed,
        )
        if cache_matches(cache_path, expected_metadata(probe)):
            context_comparison = json.loads(cache_path.read_text())
        else:
            context_comparison = None
            warnings.warn("20/20 context cache invalid; GPU refresh pending")
    if args.context_comparison_only:
        if context_comparison is not None:
            plot_context_comparison(context_comparison, args.output_dir)
        else:
            plot_context_comparison_pending(args.output_dir)
        return

    # Crossed-mask values cannot be recovered from W&B. Refresh their validated
    # cache first; unchanged checkpoints are reused and any new ones are evaluated.
    if not torch.cuda.is_available() and args.interquery_device == "auto":
        args.interquery_device = "cuda"
    try:
        interquery_records = refresh_interquery_evaluations(args)
    except subprocess.CalledProcessError:
        interquery_records = None
        warnings.warn("crossed-mask inter-query plots await GPU checkpoint refresh")

    configure_wandb_key(args.wandb_api_key_file)
    api = wandb.Api()
    RUNS = fetch_runs(api, args.entity, args.project, requested_names())
    exact_interquery_runs, interquery_maml_run = fetch_exact_interquery_runs(
        api, args.entity, args.project
    )
    # Display names are not unique in this project. Ensure all q=20 plots use
    # the exact runs paired with the crossed-mask checkpoint directories.
    for run in exact_interquery_runs.values():
        RUNS[run.name] = run
    if interquery_maml_run is not None:
        RUNS[interquery_maml_run.name] = interquery_maml_run

    history = History()
    plot_experiment_design_schematic(args.output_dir)
    if context_comparison is not None:
        plot_context_comparison(context_comparison, args.output_dir)
    else:
        plot_context_comparison_pending(args.output_dir)
    plot_supervision(history, args.output_dir, args.batch_size, target_axis=False)
    plot_supervision(history, args.output_dir, args.batch_size, target_axis=True)
    plot_context_generalization(history, args.output_dir)
    plot_multifunction(history, args.output_dir, args.batch_size)
    plot_pretraining(history, args.output_dir, args.batch_size)
    plot_stability(history, args.output_dir, args.batch_size)
    plot_decision_trees(history, args.output_dir, args.batch_size)
    if interquery_records is not None:
        plot_interquery(
            interquery_records,
            exact_interquery_runs,
            interquery_maml_run,
            history,
            args.output_dir,
            args.batch_size,
            args.interquery_thresholds,
        )


if __name__ == "__main__":
    main()
