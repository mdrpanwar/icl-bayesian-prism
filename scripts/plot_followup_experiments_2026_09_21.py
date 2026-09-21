#!/usr/bin/env python3
"""One CPU-safe entrypoint: refresh live follow-up analysis and write PNGs only.

Usage: .venv/bin/python scripts/plot_followup_experiments_2026_09_21.py
No cached W&B JSON is trusted: the separate analysis script is run first.
Incomplete jobs are labelled, and only actually evaluated support lengths are
shown. Run again at any time to update every figure.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 154645467)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("plots/followups-2026-09-21"))
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument("--wandb-api-key-file", type=Path)
    return parser.parse_args()


def refresh(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "analysis.json"
    command = [sys.executable, str(ROOT / "scripts" /
               "analyze_followup_experiments_2026_09_21.py"),
               "--output", str(path), "--entity", args.entity,
               "--project", args.project]
    if args.wandb_api_key_file:
        command += ["--wandb-api-key-file", str(args.wandb_api_key_file)]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(path.read_text())


def save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_dir / (stem + '.png')}", flush=True)


def select(records, family, method=None, condition=None):
    return sorted((record for record in records.values()
                   if record["family"] == family
                   and (method is None or record["method"] == method)
                   and (condition is None or record["condition"] == condition)),
                  key=lambda record: record["seed"])


def curve(record):
    values = record.get("context_nmse", {})
    xs = np.asarray(sorted(int(index) for index in values), dtype=int)
    return xs, np.asarray([values[str(index)] for index in xs], dtype=float)


def time_series(record):
    key = record.get("target_metric")
    rows = [row for row in record.get("history", []) if key in row]
    return (np.asarray([row["episodes"] for row in rows], dtype=float),
            np.asarray([row[key] for row in rows], dtype=float))


def style(ax, xlabel="support examples at inference", ylabel="NMSE"):
    ax.axhline(1, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def seed_curves(ax, records, label, color, linestyle="-", linewidth=2):
    plotted = False
    for index, record in enumerate(records):
        xs, ys = curve(record)
        if not len(xs):
            continue
        seed_style = linestyle if index == 0 else "--"
        ax.plot(xs, ys, color=color, linestyle=seed_style, linewidth=linewidth,
                alpha=0.95 if len(records) == 1 else 0.72,
                marker="o" if len(xs) < 25 else None,
                markersize=3, label=f"{label}, seed {record['seed']}")
        plotted = True
    return plotted


def seed_dynamics(ax, records, label, color, linestyle="-"):
    plotted = False
    for index, record in enumerate(records):
        xs, ys = time_series(record)
        if not len(xs):
            continue
        ax.plot(xs, ys, color=color, linestyle=linestyle if index == 0 else "--",
                alpha=0.85 if len(records) == 1 else 0.66,
                label=f"{label}, seed {record['seed']}")
        plotted = True
    return plotted


def pending_label(ax, records):
    pending = sum(not record.get("context_nmse") for record in records)
    if pending:
        ax.text(0.98, 0.98, f"{pending}/{len(records)} evaluations pending",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="0.35")


def plot_nine_support(records, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    choices = {"under": "{4,5,6,9,10,11,14,15,16}",
               "mixed": "{9,10,11,14,15,16,19,20,21}",
               "over": "{24,25,26,29,30,31,34,35,36}"}
    for ax, (condition, positions) in zip(axes, choices.items()):
        group = select(records, "nine_support_lr", "ICL", condition)
        seed_curves(ax, group, "joint ICL", "tab:blue")
        for n in (5, 10, 15) if condition == "under" else ((10, 15, 20) if condition == "mixed" else (25, 30, 35)):
            ax.axvline(n, color="0.55", linestyle=":", linewidth=0.7)
        ax.set_title(f"{condition}: nine support lengths")
        ax.text(0.03, 0.97, positions, transform=ax.transAxes,
                va="top", fontsize=8)
        style(ax)
        pending_label(ax, group)
    fig.suptitle("20D linear regression: nine joint ICL losses (solid: seed 0; dashed: seed 154645467)")
    save(fig, output_dir, "lr20_nine_support_context_nmse")


def plot_supervision(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {1: "tab:blue", 5: "tab:orange", 20: "tab:green"}
    for ax, method in zip(axes, ("ICL", "MAML-5")):
        for q in (1, 5, 20):
            group = select(records, "supervision", method, f"q{q}")
            seed_dynamics(ax, group, f"q={q}", colors[q])
        style(ax, "training episodes (batch size accounted for)",
              "first-query NMSE after 20 supports")
        ax.set_title(method)
        ax.legend(fontsize=8)
    fig.suptitle("20D support-20 supervision sweep: historical ICL vs new MAML-5")
    save(fig, output_dir, "lr20_supervision_sweep_first_query")


def plot_multiple_functions(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, method in zip(axes, ("ICL", "MAML-5")):
        for count, color in ((1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")):
            group = select(records, "multiple_functions", method,
                           f"M{count}-literal")
            values = [record.get("target_nmse") for record in group]
            for seed, value in zip(SEEDS, values):
                if value is not None:
                    ax.scatter(count, value, color=color, s=55,
                               marker="o" if seed == 0 else "s")
            valid = [value for value in values if value is not None]
            if valid:
                ax.plot(count, np.mean(valid), "_", color=color, markersize=20,
                        markeredgewidth=2.5)
        group = select(records, "multiple_functions", method, "M1-untagged")
        for record in group:
            value = record.get("target_nmse")
            if value is not None:
                ax.scatter(1.12, value, color="tab:red", marker="x", s=65)
        ax.set_title(method)
        ax.set_xticks((1, 2, 3))
        style(ax, "functions per episode M", "first-query NMSE")
        ax.text(0.03, 0.97, "●/■: seeds; —: mean\n×: M=1 untagged",
                transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("5D interleaved functions: literal tag tokens vs same-size untagged control")
    save(fig, output_dir, "multiple_functions_literal_tags_nmse")


def plot_tree(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for condition, color in (("sparse", "tab:blue"), ("dense", "tab:orange")):
        group = select(records, "tree_variable", "ICL", condition)
        seed_curves(axes[0], group, f"ICL {condition}", color)
        pending_label(axes[0], group)
    for method, condition, color, style_name in (
        ("MAML-5", "parallel-45-65-85", "tab:green", "-"),
        ("ICL", "causal", "tab:blue", "--"),
        ("ICL", "isolated", "tab:orange", "--"),
        ("MAML-5", "parallel", "tab:red", "--"),
    ):
        family = "tree_variable" if condition.startswith("parallel-45") else "tree_fixed"
        seed_curves(axes[1], select(records, family, method, condition),
                    f"{method} {condition}", color, style_name)
    for ax in axes:
        style(ax)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=7)
    axes[0].set_title("variable-support ICL: sparse vs nine-neighbor")
    axes[1].set_title("variable MAML parallel vs fixed-65 controls")
    fig.suptitle("Decision trees: evaluated support lengths only (no invented intermediates)")
    save(fig, output_dir, "decision_tree_variable_support_nmse")


def plot_transfer(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, method in zip(axes, ("ICL", "MAML-5")):
        for condition, color in (("scratch", "tab:blue"),
                                 ("pretrained", "tab:orange")):
            family = "transfer" if method == "ICL" or condition == "pretrained" else "supervision"
            key = condition if family == "transfer" else "q1"
            group = select(records, family, method, key)
            seed_dynamics(ax, group, condition, color)
        style(ax, "training episodes", "first-query NMSE after 20 supports")
        ax.set_title(method)
        ax.legend(fontsize=8)
    fig.suptitle("20D q=1: segmented-pretrained initialization vs scratch")
    save(fig, output_dir, "lr20_q1_pretraining_transfer")


def plot_determinacy(records, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), sharey=True)
    for ax, dimension in zip(axes, (5, 10, 20)):
        for window, color in (("early", "tab:blue"), ("late", "tab:orange")):
            group = select(records, "determinacy", "ICL", f"d{dimension}-{window}")
            if group:
                xs, ys = curve(group[0])
                if len(xs):
                    ax.plot(xs, ys, color=color, label=window)
        style(ax)
        ax.set_title(f"dimension {dimension}")
        if ax.lines and any(line.get_label() in ("early", "late") for line in ax.lines):
            ax.legend(fontsize=8)
    fig.suptitle("Determinacy pilot: early (5–9) vs late (15–19) supervised positions")
    save(fig, output_dir, "linear_determinacy_early_late_nmse")


def plot_stability(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for condition, color in (("guard", "tab:blue"), ("control", "tab:red")):
        group = select(records, "stability", "MAML-5", condition)
        seed_dynamics(axes[0], group, condition, color)
        if not group:
            continue
        rows = [row for row in group[0].get("history", [])
                if "meta_train/outer_grad_norm" in row]
        if rows:
            axes[1].plot([row["episodes"] for row in rows],
                         [row["meta_train/outer_grad_norm"] for row in rows],
                         color=color, label=condition, alpha=0.8)
    style(axes[0], "additional episodes after 150k warm start",
          "first-query NMSE after 20 supports")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("additional episodes after 150k warm start")
    axes[1].set_ylabel("outer gradient norm before clipping")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("MAML seed 154645467: gradient guard pilot vs unguarded control")
    save(fig, output_dir, "maml_stability_guard_diagnostics")


def plot_fixed_k15(records, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    group = select(records, "fixed_k15")
    available = [record for record in group if record.get("target_nmse") is not None]
    if available:
        for record in available:
            ax.scatter(15, record["target_nmse"], s=60,
                       label=f"seed {record['seed']}")
    else:
        ax.text(0.5, 0.55, "15-support evaluation not logged by these runs",
                transform=ax.transAxes, ha="center", fontsize=11)
        ax.text(0.5, 0.43, "GPU checkpoint evaluation required; no interpolation shown",
                transform=ax.transAxes, ha="center", fontsize=9, color="0.4")
    style(ax)
    ax.set_xlim(10, 20)
    if len(available) < len(group):
        ax.text(0.98, 0.98,
                f"{len(group) - len(available)}/{len(group)} checkpoint evaluations pending",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="0.35")
    if available:
        ax.legend(fontsize=8)
    ax.set_title("MAML fixed-15, one-query control")
    save(fig, output_dir, "maml_fixed15_exact_support_nmse")


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    analysis = refresh(args)
    records = analysis["records"]
    plot_nine_support(records, args.output_dir)
    plot_supervision(records, args.output_dir)
    plot_multiple_functions(records, args.output_dir)
    plot_tree(records, args.output_dir)
    plot_transfer(records, args.output_dir)
    plot_determinacy(records, args.output_dir)
    plot_stability(records, args.output_dir)
    plot_fixed_k15(records, args.output_dir)
    print(f"[plot] all PNGs refreshed from {analysis['generated_at_utc']}", flush=True)


if __name__ == "__main__":
    main()
