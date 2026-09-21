#!/usr/bin/env python3
"""Matched dense context evaluation for the selected ICL and MAML models."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import get_run_metrics


CACHE_VERSION = 2
SPECS = (
    {
        "key": "icl_isolated",
        "label": "ICL, isolated queries (seed 0)",
        "kind": "icl",
        "run_id": "exp20260911-icl-iso-causal-s0",
        "path": (
            "models/linear_regression/"
            "icl_lr20_isolated_causal_s0-exp20260911-icl-iso-causal-s0"
        ),
        "checkpoint": "latest",
    },
    {
        "key": "icl_causal",
        "label": "ICL, interacting queries (seed 65765443)",
        "kind": "icl",
        "run_id": "c0a03422-4711-4a13-94f9-ce58245e287a",
        "path": (
            "models/linear_regression/"
            "icl_lr20_causal_c20_l20_s65765443-"
            "c0a03422-4711-4a13-94f9-ce58245e287a"
        ),
        "checkpoint": "latest",
    },
    {
        "key": "maml5_seed0",
        "label": "MAML-5, seed 0 (500k checkpoint)",
        "kind": "maml",
        "run_id": "b92d63fa-31ba-4a24-9860-1728b3e5c32a",
        "path": (
            "models/meta_linear_regression/"
            "maml_lr20_s20_q20_steps5_stable_s0-"
            "b92d63fa-31ba-4a24-9860-1728b3e5c32a"
        ),
        "checkpoint": 500000,
    },
    {
        "key": "maml5_seed154_warmstart_control",
        "label": "MAML-5, seed 154645467 (150k weights + 65k; optimizer reset)",
        "kind": "maml",
        "run_id": "exp20260921-maml-control-pilot-s154",
        "path": (
            "models/meta_linear_regression/"
            "maml_lr20_control_pilot_s154-"
            "exp20260921-maml-control-pilot-s154"
        ),
        "checkpoint": 65000,
    },
)
CODE_PATHS = (
    Path(__file__),
    SRC_DIR / "eval.py",
    SRC_DIR / "meta_eval.py",
    SRC_DIR / "meta_utils.py",
    SRC_DIR / "models.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    # The cluster wrapper supplies these generic training arguments.
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--training.resume_id", dest="resume_id", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/main-experiments-2026-09-15/context_comparison.json"),
    )
    parser.add_argument("--num-eval-examples", type=int, default=1280)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_signature():
    digest = hashlib.sha256()
    for path in CODE_PATHS:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_path(spec):
    run_path = REPO_ROOT / spec["path"]
    checkpoint = spec["checkpoint"]
    return (
        run_path / "state.pt"
        if checkpoint == "latest"
        else run_path / f"model_{checkpoint}.pt"
    )


def checkpoint_signature(spec):
    checkpoint = checkpoint_path(spec)
    stat = checkpoint.stat()
    return {
        "checkpoint": spec["checkpoint"],
        "path": str(checkpoint.relative_to(REPO_ROOT)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def expected_metadata(args):
    return {
        "version": CACHE_VERSION,
        "num_eval_examples": args.num_eval_examples,
        "eval_batch_size": args.eval_batch_size,
        "eval_seed": args.eval_seed,
        "n_points": 41,
        "normalization": 20.0,
        "code_signature": code_signature(),
        "checkpoints": {
            spec["key"]: checkpoint_signature(spec)
            for spec in SPECS
        },
    }


def cache_matches(path, metadata):
    if not path.is_file():
        return False
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if set(cached.get("curves", {})) != {spec["key"] for spec in SPECS}:
        return False
    stored = cached.get("metadata", {})
    if stored == metadata:
        return True
    # RunAI's evaluator and this host can fingerprint different source copies.
    # Accept only a cache newer than every inference source, with identical
    # checkpoints, seed, batch size, task count, normalization, and schema.
    without_code = lambda item: {key: value for key, value in item.items()
                                 if key != "code_signature"}
    if (without_code(stored) == without_code(metadata)
            and path.stat().st_mtime_ns >= max(
                source.stat().st_mtime_ns for source in CODE_PATHS[1:]
            )):
        print("[context-eval] validated GPU cache by checkpoint/protocol signatures and source mtimes", flush=True)
        return True
    return False


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(spec, args):
    reset_rng(args.eval_seed)
    overrides = {
        "n_points": 41,
        "num_eval_examples": args.num_eval_examples,
        "batch_size": args.eval_batch_size,
        "data_sampler_kwargs": {"data_seed": args.eval_seed},
    }
    eval_name = "maml" if spec["kind"] == "maml" else "standard"
    if spec["kind"] == "maml":
        overrides["stride"] = 1
        overrides["progress_desc"] = spec["label"]
        overrides["progress_every_contexts"] = 5
    else:
        overrides["parallel_seed"] = args.eval_seed
    print(f"[context-eval] evaluating {spec['label']}", flush=True)
    started = time.monotonic()
    metrics = get_run_metrics(
        REPO_ROOT / spec["path"],
        step=(-1 if spec["checkpoint"] == "latest" else spec["checkpoint"]),
        cache=False,
        skip_baselines=True,
        eval_names=[eval_name],
        eval_kwargs_overrides=overrides,
    )
    values = next(iter(metrics[eval_name].values()))
    print(
        f"[context-eval] finished {spec['label']} in "
        f"{(time.monotonic() - started) / 60:.1f}m",
        flush=True,
    )
    return {
        "label": spec["label"],
        "kind": spec["kind"],
        "run_id": spec["run_id"],
        "run_path": spec["path"],
        "checkpoint": spec["checkpoint"],
        "mean": values["mean"][:41],
        "low": values["bootstrap_low"][:41],
        "high": values["bootstrap_high"][:41],
    }


def main():
    args = parse_args()
    if args.num_eval_examples <= 0:
        raise ValueError("num-eval-examples must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive")
    if args.num_eval_examples % args.eval_batch_size:
        raise ValueError("num-eval-examples must be divisible by eval-batch-size")
    args.output = (REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    metadata = expected_metadata(args)
    if not args.force and cache_matches(args.output, metadata):
        print(f"[context-eval] validated cache hit: {args.output}", flush=True)
        return
    curves = {spec["key"]: evaluate(spec, args) for spec in SPECS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "curves": curves}, indent=2))
    temporary.replace(args.output)
    print(f"[context-eval] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
