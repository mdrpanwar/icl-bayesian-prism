#!/usr/bin/env python
"""Create the ICL training-dynamics and context-length plots from W&B runs."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

POINTWISE_METRIC = "linear_regression_eval/pointwise/loss"
TARGET_POSITION = 20


def pointwise_values(row):
    """Extract an integer-indexed pointwise metric from nested or flattened W&B data."""
    values = {}
    nested = row.get(POINTWISE_METRIC)
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            try:
                values[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(nested, (list, tuple)):
        values.update({index: float(value) for index, value in enumerate(nested)})

    for key, value in row.items():
        suffix = None
        for separator in (".", "/"):
            prefix = f"{POINTWISE_METRIC}{separator}"
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                break
        if suffix is None:
            continue
        try:
            values[int(suffix)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def find_runs(entity, project, run_names):
    api = wandb.Api()
    requested = set(run_names)
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$in": sorted(requested)}},
    )
    by_name = {}
    for run in runs:
        if run.name not in requested:
            continue
        previous = by_name.get(run.name)
        if previous is None or str(run.created_at) > str(previous.created_at):
            by_name[run.name] = run

    missing = sorted(requested - set(by_name))
    if missing:
        raise RuntimeError(f"Missing W&B runs: {', '.join(missing)}")
    return by_name


def dynamics(run, position=TARGET_POSITION):
    steps = []
    errors = []
    for row in run.scan_history():
        values = pointwise_values(row)
        if position not in values or row.get("_step") is None:
            continue
        steps.append(int(row["_step"]))
        errors.append(values[position])
    if not steps:
        raise RuntimeError(
            f"Run {run.name!r} has no {POINTWISE_METRIC} values at position {position}"
        )
    return steps, errors


def final_context_curve(run, max_context=40):
    latest_step = -1
    latest_values = None
    for row in run.scan_history():
        values = pointwise_values(row)
        step = row.get("_step")
        if values and step is not None and int(step) >= latest_step:
            latest_step = int(step)
            latest_values = values
    if latest_values is None:
        raise RuntimeError(f"Run {run.name!r} has no pointwise evaluation history")

    missing = [index for index in range(max_context + 1) if index not in latest_values]
    if missing:
        raise RuntimeError(
            f"Run {run.name!r} is missing context positions: {missing}"
        )
    xs = list(range(max_context + 1))
    return xs, [latest_values[index] for index in xs]
def load_maml_curves(specifications):
    """Load normalized MAML context curves from existing local run directories."""
    if not specifications:
        return []
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from plot_meta_paper_figure import get_task_metrics

    curves = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--maml-run must use LABEL=RUN_PATH")
        label, path_text = specification.split("=", 1)
        run_path = Path(path_text)
        if not run_path.is_absolute():
            run_path = REPO_ROOT / run_path
        metrics = get_task_metrics(
            run_path,
            "maml",
            transformer_label=label,
            skip_baselines=True,
        )
        model_metrics = metrics[label]
        curves.append(
            (
                label,
                model_metrics["mean"][:41],
                model_metrics["bootstrap_low"][:41],
                model_metrics["bootstrap_high"][:41],
            )
        )
    return curves




def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics(runs, specifications, output_dir, stem, normalization):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, label in specifications:
        steps, errors = dynamics(runs[run_name])
        ax.plot(steps, [value / normalization for value in errors], label=label)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Squared error at context length 20")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, stem)


def plot_context_curve(run, output_dir, normalization, maml_curves):
    xs, errors = final_context_curve(run, max_context=40)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [value / normalization for value in errors], label="ICL, last 10 terms")
    for label, mean, low, high in maml_curves:
        reference_xs = list(range(len(mean)))
        ax.plot(reference_xs, mean, label=label)
        ax.fill_between(reference_xs, low, high, alpha=0.2)
    ax.set_xlabel("In-context examples")
    ax.set_ylabel("Normalized squared error")
    ax.set_xlim(0, 40)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, "last10_error_vs_context_length")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/icl_experiments"))
    parser.add_argument(
        "--dynamics-normalization",
        type=float,
        default=1.0,
        help="Use 1 for the raw-error dynamics plot.",
    )
    parser.add_argument(
        "--context-normalization",
        type=float,
        default=20.0,
        help="Dimension normalization used by the existing error-context plot.",
    )
    parser.add_argument(
        "--maml-run",
        action="append",
        default=[],
        metavar="LABEL=RUN_PATH",
        help="Overlay an existing local MAML run; may be supplied repeatedly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    fixed_length = [
        (f"icl_l21_all_s{seed}", "Fixed length 21, all terms"),
        (f"icl_l41_all_s{seed}", "Fixed length 41, all terms"),
    ]
    positions = [
        (f"icl_pos_terms01_15_s{seed}", "Terms 1–15"),
        (f"icl_pos_terms06_20_s{seed}", "Terms 6–20"),
        (f"icl_pos_terms21_35_s{seed}", "Terms 21–35"),
        (f"icl_pos_terms26_40_s{seed}", "Terms 26–40"),
    ]
    last10_name = f"icl_l41_last10_s{seed}"
    run_names = [name for name, _ in fixed_length + positions] + [last10_name]
    runs = find_runs(args.entity, args.project, run_names)

    plot_dynamics(
        runs,
        fixed_length,
        args.output_dir,
        "fixed_length_training_dynamics",
        args.dynamics_normalization,
    )
    plot_dynamics(
        runs,
        positions,
        args.output_dir,
        "loss_position_training_dynamics",
        args.dynamics_normalization,
    )
    plot_context_curve(
        runs[last10_name],
        args.output_dir,
        args.context_normalization,
        load_maml_curves(args.maml_run),
    )
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
