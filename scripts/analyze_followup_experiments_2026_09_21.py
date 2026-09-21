#!/usr/bin/env python3
"""Fetch live W&B follow-up data and write a reproducible analysis snapshot.

This is deliberately separate from plotting. The plotter invokes this script on
every run, so its JSON is a freshly fetched intermediate, not a stale source of
truth. Missing/queued runs remain explicitly pending.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import warnings

import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 154645467)
PREFIX = "exp20260921-"
LR = "linear_regression_eval/pointwise/loss"
TREE = "decision_tree_eval/pointwise/loss"
MULTI = "multi_function_linear_eval/pointwise/loss"
MAML = "maml_eval/pointwise/loss"


def expected_runs():
    """Explicit experimental design; keys are exact W&B IDs."""
    result = {}

    def add(suffix, family, method, condition, seed, metric, denominator, target):
        run_id = PREFIX + suffix
        result[run_id] = dict(family=family, method=method, condition=condition,
                              seed=seed, metric=metric, denominator=denominator,
                              target=target)

    for seed in SEEDS:
        for condition in ("under", "mixed", "over"):
            add(f"icl-nine-{condition}-s{seed}", "nine_support_lr", "ICL",
                condition, seed, LR, 20, 20)
        for q in (1, 5, 20):
            add(f"maml-lr20-q{q}-s{seed}", "supervision", "MAML-5",
                f"q{q}", seed, MAML, 20, 20)
        for count in (1, 2, 3):
            add(f"icl-mf{count}-literal-s{seed}", "multiple_functions", "ICL",
                f"M{count}-literal", seed, MULTI, 5, 5 * count)
            add(f"maml-mf{count}-literal-s{seed}", "multiple_functions", "MAML-5",
                f"M{count}-literal", seed, MAML, 5, 5 * count)
        for method, metric in (("icl", MULTI), ("maml", MAML)):
            add(f"{method}-mf1-untagged-s{seed}", "multiple_functions",
                "ICL" if method == "icl" else "MAML-5", "M1-untagged",
                seed, metric, 5, 5)
        for condition in ("sparse", "dense"):
            add(f"icl-dt-{condition}-s{seed}", "tree_variable", "ICL",
                condition, seed, TREE, 1, 65)
        add(f"maml-dt-var-bs32-s{seed}", "tree_variable", "MAML-5",
            "parallel-45-65-85", seed, MAML, 1, 65)
        add(f"maml-lr20-k15-q1-s{seed}", "fixed_k15", "MAML-5",
            "k15-q1", seed, MAML, 20, 15)
        add(f"icl-lr20-q1-scratch-s{seed}", "transfer", "ICL", "scratch",
            seed, LR, 20, 20)
        add(f"icl-lr20-q1-pre-s{seed}", "transfer", "ICL", "pretrained",
            seed, LR, 20, 20)
        add(f"maml-lr20-q1-pre-s{seed}", "transfer", "MAML-5", "pretrained",
            seed, MAML, 20, 20)
        for condition in ("sparse", "dense"):
            pass
    for seed in SEEDS:
        add(f"icl-dt-dense-s{seed}", "tree_variable", "ICL", "dense",
            seed, TREE, 1, 65)
        add(f"icl-dt-sparse-s{seed}", "tree_variable", "ICL", "sparse",
            seed, TREE, 1, 65)
    add("icl-dt-fixed-causal-s154", "tree_fixed", "ICL", "causal",
        154645467, TREE, 1, 65)
    add("icl-dt-fixed-isolated-s154", "tree_fixed", "ICL", "isolated",
        154645467, TREE, 1, 65)
    add("maml-dt-fixed-s154", "tree_fixed", "MAML-5", "parallel",
        154645467, MAML, 1, 65)
    for dimension in (5, 10, 20):
        for window in ("early", "late"):
            add(f"icl-det-d{dimension}-{window}-s0", "determinacy", "ICL",
                f"d{dimension}-{window}", 0, LR, dimension,
                5 if window == "early" else 15)
    for condition in ("guard", "control"):
        add(f"maml-{condition}-pilot-s154", "stability", "MAML-5",
            condition, 154645467, MAML, 20, 20)
    historical = {
        "exp20260915-icl-s20-q1-s0": (1, 0),
        "exp20260915-icl-s20-q1-s154645467": (1, 154645467),
        "exp20260915-icl-s20-q5-s0": (5, 0),
        "exp20260915-icl-s20-q5-s154645467": (5, 154645467),
        "0897d9d4-6bc5-4ecf-92da-eec06de6bfdb": (20, 0),
        "31212caa-f704-4531-a3e3-8fe95435f65b": (20, 154645467),
    }
    for run_id, (q, seed) in historical.items():
        result[run_id] = dict(family="supervision", method="ICL",
                              condition=f"q{q}", seed=seed, metric=LR,
                              denominator=20, target=20)
    return result


def configure_key(path):
    if os.environ.get("WANDB_API_KEY"):
        return
    for candidate in (path, os.environ.get("WANDB_API_KEY_FILE_AT"),
                      REPO_ROOT.parent / ".wandb-api-key"):
        if candidate and Path(candidate).is_file():
            os.environ["WANDB_API_KEY"] = Path(candidate).read_text().strip()
            return


def pointwise_summary(summary, prefix, denominator):
    values = {}
    stem = prefix + "."
    for key, value in summary.items():
        if not key.startswith(stem):
            continue
        try:
            index = int(key[len(stem):])
            value = float(value) / denominator
            if math.isfinite(value):
                values[index] = value
        except (TypeError, ValueError):
            continue
    return {str(index): values[index] for index in sorted(values)}


def scalar_history(run, keys, denominator):
    """Fetch unsmoothed training/evaluation history; do not interpolate gaps."""
    by_step = {}
    try:
        for row in run.history(keys=keys, pandas=False, samples=10000):
            step = row.get("_step")
            if step is None:
                continue
            record = by_step.setdefault(int(step), {})
            for key in keys:
                try:
                    value = float(row.get(key))
                    if math.isfinite(value):
                        record[key] = value / denominator if "pointwise/loss" in key else value
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        warnings.warn(f"history unavailable for {run.id}: {exc}")
    return [{"step": step, **by_step[step]} for step in sorted(by_step)]


def parse_run(run, spec):
    summary = dict(run.summary)
    config = dict(run.config)
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    meta = config.get("meta", {})
    if spec["method"] == "MAML-5" and isinstance(meta, dict):
        batch_size = int(meta.get("meta_batch_size", training.get("batch_size", 64)))
    else:
        batch_size = int(training.get("batch_size", 64))
    metric = spec["metric"]
    target_key = f"{metric}.{spec['target']}"
    grad_keys = ("meta_train/outer_grad_norm", "meta_train/outer_update_skipped",
                 "meta_train/outer_updates_skipped_total")
    history_keys = [target_key]
    if spec["family"] == "stability":
        history_keys += list(grad_keys)
    history = scalar_history(run, history_keys, spec["denominator"])
    for row in history:
        row["episodes"] = row["step"] * batch_size
    curve = pointwise_summary(summary, metric, spec["denominator"])
    target = curve.get(str(spec["target"]))
    threshold_crossing = None
    if spec["denominator"] in (5, 10, 20) and spec["family"] not in ("tree_variable", "tree_fixed"):
        for row in history:
            if row["episodes"] >= 10000 and row.get(target_key, math.inf) <= 0.2:
                threshold_crossing = row["episodes"]
                break
    return {
        **spec, "id": run.id, "name": run.name, "state": run.state,
        "url": run.url, "step": summary.get("_step"),
        "batch_size": batch_size, "target_nmse": target,
        "target_metric": target_key, "context_nmse": curve,
        "history": history, "first_episodes_below_0_2": threshold_crossing,
        "evaluation_gap": spec["target"] != 0 and str(spec["target"]) not in curve,
        "guard_skipped_total": summary.get("meta_train/outer_updates_skipped_total"),
    }


def summarize(records):
    groups = {}
    for record in records.values():
        key = (record["family"], record["method"], record["condition"])
        groups.setdefault(key, []).append(record)
    result = []
    for (family, method, condition), group in sorted(groups.items()):
        values = [item["target_nmse"] for item in group if item.get("target_nmse") is not None]
        result.append(dict(family=family, method=method, condition=condition,
                           expected=len(group), observed=len(values),
                           mean_target_nmse=statistics.mean(values) if values else None,
                           min_target_nmse=min(values) if values else None,
                           max_target_nmse=max(values) if values else None,
                           states={str(item["seed"]): item["state"] for item in group}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="mdrpanwar")
    parser.add_argument("--project", default="icl-metal")
    parser.add_argument("--output", type=Path,
                        default=Path("plots/followups-2026-09-21/analysis.json"))
    parser.add_argument("--wandb-api-key-file", type=Path)
    args = parser.parse_args()
    configure_key(args.wandb_api_key_file)
    api = wandb.Api()
    specs = expected_runs()
    # One listing call returns the live September-21 runs, including resumed runs
    # that retain the original ID. Exact-ID matching avoids duplicate-name errors.
    recent = api.runs(f"{args.entity}/{args.project}", filters={
        "created_at": {"$gt": "2026-09-20T00:00:00Z"}}, per_page=100)
    live = {run.id: run for run in recent if run.id in specs}
    for run_id in specs:
        if run_id in live or run_id.startswith(PREFIX):
            continue
        try:
            live[run_id] = api.run(f"{args.entity}/{args.project}/{run_id}")
        except Exception as exc:
            warnings.warn(f"historical control {run_id} unavailable: {exc}")
    records = {}
    for index, (run_id, spec) in enumerate(specs.items(), 1):
        run = live.get(run_id)
        if run is None:
            records[run_id] = {**spec, "id": run_id, "state": "pending",
                               "target_nmse": None, "context_nmse": {}, "history": []}
            continue
        print(f"[analysis] {index}/{len(specs)} {run_id} ({run.state})", flush=True)
        records[run_id] = parse_run(run, spec)
    gpu_eval_path = args.output.parent / "k15_checkpoint_evaluation.json"
    if gpu_eval_path.is_file():
        gpu_eval = json.loads(gpu_eval_path.read_text())
        for seed_text, measured in gpu_eval.get("results", {}).items():
            run_id = PREFIX + f"maml-lr20-k15-q1-s{seed_text}"
            record = records.get(run_id)
            if record is None:
                continue
            signature = measured.get("checkpoint", {})
            checkpoint = REPO_ROOT / signature.get("path", "missing")
            if not checkpoint.is_file():
                continue
            stat = checkpoint.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                signature.get("size"), signature.get("mtime_ns")
            ):
                continue
            run_dir = checkpoint.parent
            retained = list(run_dir.glob("model_*.pt"))
            if retained and checkpoint != max(
                retained, key=lambda path: int(path.stem.split("_")[1])
            ):
                continue
            record["target_nmse"] = measured["support_15_nmse"]
            record["context_nmse"]["15"] = measured["support_15_nmse"]
            record["target_source"] = "GPU checkpoint evaluation (signature validated)"
            record["target_interval"] = [measured["support_15_low"],
                                         measured["support_15_high"]]
            record["evaluation_gap"] = False
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": f"live W&B {args.entity}/{args.project}",
              "normalization": "MSE divided by zero-predictor output variance (20D LR: 20; 5D LR: 5; trees: 1)",
              "metric_note": "All comparisons use first-query NMSE at the specified support length; latest context curves include only actually evaluated positions.",
              "records": records, "groups": summarize(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    report = ["# Follow-up experiment snapshot", "", result["generated_at_utc"], "",
              "First-query NMSE; unfinished runs are provisional. Blank means not yet evaluated.", "",
              "| Experiment | Method | Condition | Evaluable/planned | Mean NMSE |",
              "|---|---|---:|---:|---:|"]
    for group in result["groups"]:
        score = "—" if group["mean_target_nmse"] is None else f"{group['mean_target_nmse']:.3f}"
        report.append(f"| {group['family']} | {group['method']} | {group['condition']} | {group['observed']}/{group['expected']} | {score} |")
    args.output.with_suffix(".md").write_text("\n".join(report) + "\n")
    print(f"[analysis] refreshed {len(live)}/{len(specs)} runs -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
