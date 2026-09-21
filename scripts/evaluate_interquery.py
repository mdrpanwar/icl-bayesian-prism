#!/usr/bin/env python3
"""Cross-evaluate a q=20 ICL checkpoint with natural and isolated queries."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from munch import Munch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import eval_model
from meta_utils import load_state_dict_allow_missing_inner_lrs
from models import build_model

CACHE_SCHEMA_VERSION = 3
CACHE_CODE_PATHS = (
    Path(__file__),
    SRC_DIR / "models.py",
    SRC_DIR / "tasks.py",
    SRC_DIR / "samplers.py",
)
CACHE_CODE_SLICES = (
    (SRC_DIR / "eval.py", "def preserve_rng_state", "\ndef build_evals"),
    (
        SRC_DIR / "meta_utils.py",
        "def is_inner_lr_parameter",
        "\ndef trainable_model_params",
    ),
    (SRC_DIR / "meta_utils.py", "def load_state_dict_allow_missing_inner_lrs", None),
)


def parse_args():
    parser = argparse.ArgumentParser()
    # submit_un.sh supplies --config to every workload; it is intentionally
    # ignored here because each evaluated run carries its own resolved config.
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--training.resume_id", dest="resume_id", default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-mode", choices=("causal", "isolated"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support", type=int, default=20)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--num-eval-examples", type=int, default=1280)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=81173)
    parser.add_argument("--test-seed", type=int, default=104729)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def checkpoint_paths(run_dir):
    paths = []
    for path in run_dir.glob("model_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        paths.append((step, path))
    if not paths:
        raise FileNotFoundError(f"no retained model checkpoints in {run_dir}")
    return sorted(paths)


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint():
    digest = hashlib.sha256()
    for path in CACHE_CODE_PATHS:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    for path, start_marker, end_marker in CACHE_CODE_SLICES:
        source = path.read_text()
        start = source.index(start_marker)
        end = source.index(end_marker, start) if end_marker is not None else len(source)
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(source[start:end].encode())
    return digest.hexdigest()


def checkpoint_signature(step, path):
    stat = path.stat()
    return {
        "step": int(step),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def evaluation_settings(args):
    return {
        "support": args.support,
        "queries": args.queries,
        "num_eval_examples": args.num_eval_examples,
        "eval_batch_size": args.eval_batch_size,
        "validation_seed": args.validation_seed,
        "test_seed": args.test_seed,
    }


def cached_records(args, checkpoints, signatures, fingerprint, config_fingerprint):
    if not args.output.is_file():
        return {}
    try:
        payload = json.loads(args.output.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    expected_run_dir = str(args.run_dir.resolve())
    identity_matches = (
        payload.get("run_dir") == expected_run_dir
        and payload.get("training_mode") == args.training_mode
        and payload.get("seed") == args.seed
        and payload.get("support") == args.support
        and payload.get("queries") == args.queries
    )
    if not identity_matches:
        return {}

    records = {
        int(record["step"]): record
        for record in payload.get("checkpoints", [])
        if "step" in record
    }
    cache = payload.get("cache")
    if cache is not None:
        schema_version = cache.get("schema_version")
        if (
            schema_version not in {1, 2, CACHE_SCHEMA_VERSION}
            or cache.get("config_fingerprint") != config_fingerprint
            or cache.get("evaluation_settings") != evaluation_settings(args)
        ):
            return {}
        # Schemas 1-2 are migrated once. Their intervening changes did not affect
        # these frozen ICL predictions. Schema 3 fingerprints only actual
        # inference dependencies; relevant changes invalidate normally.
        if (
            schema_version == CACHE_SCHEMA_VERSION
            and cache.get("code_fingerprint") != fingerprint
        ):
            return {}
        cached_signatures = {
            int(item["step"]): item for item in cache.get("checkpoint_signatures", [])
        }
        return {
            step: records[step]
            for step, _ in checkpoints
            if step in records and cached_signatures.get(step) == signatures[step]
        }

    # One-time migration for JSONs generated before validated caching was added.
    # They were produced by this evaluator with its defaults. Trust them only
    # when they contain every current checkpoint and are newer than the sources.
    expected_steps = [step for step, _ in checkpoints]
    if (
        evaluation_settings(args)
        == {
            "support": 20,
            "queries": 20,
            "num_eval_examples": 1280,
            "eval_batch_size": 64,
            "validation_seed": 81173,
            "test_seed": 104729,
        }
        and sorted(records) == expected_steps
        and args.output.stat().st_mtime_ns
        >= max(item["mtime_ns"] for item in signatures.values())
    ):
        return records
    return {}


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but CUDA is unavailable")
    return torch.device(name)


def build_eval_model(model_conf, isolated, device):
    conf = copy.deepcopy(model_conf)
    conf.attn_implementation = "eager"
    conf.attention_mode = "causal"
    conf.isolate_query_points = bool(isolated)
    conf.prefix_condition_points = 20 if isolated else None
    model = build_model(conf).to(device).eval()
    return model


def evaluate(model, conf, n_points, count, batch_size, seed):
    metrics = eval_model(
        model,
        task_name="linear_regression",
        data_name="gaussian",
        n_dims=conf.model.n_dims,
        n_points=n_points,
        prompting_strategy="standard",
        num_eval_examples=count,
        batch_size=batch_size,
        data_sampler_kwargs={"data_seed": seed},
        task_sampler_kwargs={},
        parallel_seed=seed,
    )
    return [float(value) / conf.model.n_dims for value in metrics["mean"]]


def main():
    args = parse_args()
    config_path = args.run_dir / "config.yaml"
    with config_path.open() as handle:
        conf = Munch.fromDict(yaml.safe_load(handle))

    checkpoints = checkpoint_paths(args.run_dir)
    signatures = {
        step: checkpoint_signature(step, checkpoint)
        for step, checkpoint in checkpoints
    }
    fingerprint = code_fingerprint()
    config_fingerprint = file_fingerprint(config_path)
    reusable = cached_records(
        args, checkpoints, signatures, fingerprint, config_fingerprint
    )
    missing = [(step, path) for step, path in checkpoints if step not in reusable]

    natural_model = None
    isolated_model = None
    device = None
    if missing:
        device = resolve_device(args.device)
        if device.type == "cpu":
            print(
                "[evaluate] CUDA unavailable; uncached checkpoints will be evaluated on CPU",
                flush=True,
            )
        natural_model = build_eval_model(conf.model, isolated=False, device=device)
        isolated_model = build_eval_model(conf.model, isolated=True, device=device)

    n_points = args.support + args.queries
    records = []
    for step, checkpoint in checkpoints:
        if step in reusable:
            record = reusable[step]
        else:
            print(
                f"[evaluate] {args.training_mode} seed={args.seed} step={step}",
                flush=True,
            )
            state_dict = torch.load(checkpoint, map_location=device)
            load_state_dict_allow_missing_inner_lrs(natural_model, state_dict)
            load_state_dict_allow_missing_inner_lrs(isolated_model, state_dict)
            record = {"step": step}
            for split, seed in (
                ("validation", args.validation_seed),
                ("test", args.test_seed),
            ):
                record[split] = {
                    "natural": evaluate(
                        natural_model,
                        conf,
                        n_points,
                        args.num_eval_examples,
                        args.eval_batch_size,
                        seed,
                    ),
                    "isolated": evaluate(
                        isolated_model,
                        conf,
                        n_points,
                        args.num_eval_examples,
                        args.eval_batch_size,
                        seed,
                    ),
                }
        records.append(record)

    final = records[-1]["test"]
    query_slice = slice(args.support, n_points)
    natural = float(np.mean(final["natural"][query_slice]))
    isolated = float(np.mean(final["isolated"][query_slice]))
    payload = {
        "run_dir": str(args.run_dir.resolve()),
        "training_mode": args.training_mode,
        "seed": args.seed,
        "support": args.support,
        "queries": args.queries,
        "selection_metric": "validation isolated-query mean NMSE",
        "final_test_query_nmse": {"natural": natural, "isolated": isolated},
        "final_test_inference_relative_gain": (
            (isolated - natural) / isolated if isolated else None
        ),
        "cache": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_fingerprint": fingerprint,
            "config_fingerprint": config_fingerprint,
            "evaluation_settings": evaluation_settings(args),
            "checkpoint_signatures": [
                signatures[step] for step, _ in checkpoints
            ],
            "reused_checkpoints": len(reusable),
            "evaluated_checkpoints": len(missing),
        },
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        f"[cache] {args.output}: reused={len(reusable)} evaluated={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
