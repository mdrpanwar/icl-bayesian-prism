# Reproducing the September 2026 plots

From the repository root, run:

```bash
.venv/bin/python scripts/plot_main_experiments_2026_09_15.py
```

This is the single plotting entrypoint. It fetches current W&B histories,
reruns `scripts/analyze_followup_experiments_2026_09_21.py`, validates the
checkpoint-evaluation caches, and writes PNGs in
`plots/main-experiments-2026-09-15/` and `plots/followups-2026-09-21/`.
Run it again at any time as training proceeds. For only the newer plots:

```bash
.venv/bin/python scripts/plot_main_experiments_2026_09_15.py --suite followups
```

The refreshed numerical snapshot is
`plots/followups-2026-09-21/analysis.md` (and `analysis.json`). To refresh
that snapshot without plotting, run
`scripts/analyze_followup_experiments_2026_09_21.py` with the same Python.

The W&B histories are live, but exact checkpoint evaluations require a GPU.
The main plotter reuses them only when the checkpoint/protocol signatures
match. If either fixed-15 MAML checkpoint changes, refresh its evaluation
on a GPU before plotting again; from the workspace parent directory:

```bash
env NODE_POOL=default TRAIN_MODULE=evaluate_followup_k15.py CONFIG=conf/experiments/maml_lr20_support20_query1_stable.yaml ./submit_un.sh eval-fu-maml-k15-refresh-$(date +%Y%m%d%H%M%S) --output ../plots/followups-2026-09-21/k15_checkpoint_evaluation.json --num-eval-examples 512 --eval-batch-size 32 --training.resume_id exp20260921-eval-k15
```

If the 20/20 context comparison says its cache is invalid after a code or
checkpoint change, submit this GPU refresh instead:

```bash
env NODE_POOL=default TRAIN_MODULE=../scripts/evaluate_context_comparison.py CONFIG=conf/experiments/maml_lr20_support20_query20_stable.yaml ./submit_un.sh eval-fu-lr20-context-refresh-$(date +%Y%m%d%H%M%S) --output plots/main-experiments-2026-09-15/context_comparison.json --num-eval-examples 1280 --eval-batch-size 32 --eval-seed 0 --training.resume_id exp20260921-eval-lr20-context
```

If crossed-mask inter-query checkpoints change, refresh those GPU caches:

```bash
env NODE_POOL=default TRAIN_MODULE=evaluate_interquery_all.py CONFIG=conf/experiments/maml_lr20_support20_query20_stable.yaml ./submit_un.sh eval-fu-interquery-refresh-$(date +%Y%m%d%H%M%S) --training.resume_id exp20260921-eval-interquery-all
```


Wait for any GPU refresh job to succeed, then rerun the plotting entrypoint.

The second MAML seed in the 20/20 checkpoint comparison uses the completed
unguarded pilot's `model_65000.pt`: 150k-step weights followed by 65k more
updates with a fresh optimizer/RNG, not an uninterrupted 215k-step run.
