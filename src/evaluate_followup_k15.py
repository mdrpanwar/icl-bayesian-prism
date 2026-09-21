"""GPU-only exact support-15 evaluation for the fixed-k MAML controls.

The ordinary training logs sample only supports 0, 20, 40. This evaluates 15
directly, on identical task seeds for the two MAML seeds. The output records
checkpoint signatures so CPU plotting never treats an obsolete file as live.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch

from eval import get_run_metrics


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    0: ROOT / "models/meta_linear_regression/maml_lr20_k15_q1_guard_s0-exp20260921-maml-lr20-k15-q1-s0",
    154645467: ROOT / "models/meta_linear_regression/maml_lr20_k15_q1_guard_s154645467-exp20260921-maml-lr20-k15-q1-s154645467",
}


def select_checkpoint(run_dir):
    retained = []
    for path in run_dir.glob("model_*.pt"):
        try:
            retained.append((int(path.stem.split("_")[1]), path))
        except (IndexError, ValueError):
            continue
    if retained:
        step, path = max(retained)
    else:
        path = run_dir / "state.pt"
        step = -1
    stat = path.stat()
    return step, {"path": str(path.relative_to(ROOT)), "size": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # The standard RunAI wrapper supplies these flags.
    parser.add_argument("--config")
    parser.add_argument("--run_name")
    parser.add_argument("--training.resume_id")
    parser.add_argument("--output", type=Path,
                        default=Path("plots/followups-2026-09-21/k15_checkpoint_evaluation.json"))
    parser.add_argument("--num-eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=104729)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This checkpoint evaluator must run on a GPU")
    rows = {}
    for seed, run_dir in SPECS.items():
        step, signature = select_checkpoint(run_dir)
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed_all(args.eval_seed)
        print(f"[k15] seed={seed}, checkpoint={signature['path']}", flush=True)
        metrics = get_run_metrics(
            run_dir, step=step, cache=False, skip_baselines=True,
            eval_names=["maml"], eval_kwargs_overrides={
                "n_points": 16, "stride": 5,
                "num_eval_examples": args.num_eval_examples,
                "batch_size": args.eval_batch_size,
                "data_sampler_kwargs": {"data_seed": args.eval_seed},
                "progress_desc": f"fixed k=15 seed={seed}",
                "progress_every_contexts": 1,
            })
        values = next(iter(metrics["maml"].values()))
        rows[str(seed)] = {
            "checkpoint": signature, "step": step,
            "support_15_nmse": float(values["mean"][15]) / 20,
            "support_15_low": float(values["bootstrap_low"][15]) / 20,
            "support_15_high": float(values["bootstrap_high"][15]) / 20,
        }
        print(f"[k15] seed={seed}, NMSE={rows[str(seed)]['support_15_nmse']:.4f}", flush=True)
    relative = (Path(*args.output.parts[1:]) if args.output.parts[0] == ".."
                else args.output)
    output = args.output if args.output.is_absolute() else ROOT / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_eval_examples": args.num_eval_examples,
            "eval_seed": args.eval_seed, "results": rows}
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(output)
    print(f"[k15] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
