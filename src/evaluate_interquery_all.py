"""GPU runner for every crossed-mask cache consumed by the main plotter."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_main_experiments_2026_09_15 import INTERQUERY_SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run_name")
    parser.add_argument("--training.resume_id")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("plots/main-experiments-2026-09-15"))
    parser.add_argument("--num-eval-examples", type=int, default=1280)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=81173)
    parser.add_argument("--test-seed", type=int, default=104729)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    cache_dir = output_dir / "interquery"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for mode, seed, _, relative_run_dir in INTERQUERY_SPECS:
        command = [
            sys.executable, str(ROOT / "scripts" / "evaluate_interquery.py"),
            "--run-dir", str(ROOT / relative_run_dir),
            "--training-mode", mode, "--seed", str(seed),
            "--output", str(cache_dir / f"{mode}_s{seed}.json"),
            "--num-eval-examples", str(args.num_eval_examples),
            "--eval-batch-size", str(args.eval_batch_size),
            "--validation-seed", str(args.validation_seed),
            "--test-seed", str(args.test_seed), "--device", "cuda",
        ]
        print(f"[interquery-all] {mode} seed={seed}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("[interquery-all] all caches validated/refreshed", flush=True)


if __name__ == "__main__":
    main()
