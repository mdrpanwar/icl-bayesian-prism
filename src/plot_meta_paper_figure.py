import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf-home")

import matplotlib.pyplot as plt
import seaborn as sns

from eval import baseline_names, get_model_from_run, get_run_metrics

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


TASK_SPECS = {
    "linear_regression": {
        "run_dir": "models/meta_linear_regression",
        "title": "(a) Linear regression",
        "models": [
            "Transformer",
            "Least Squares",
            "3-Nearest Neighbors",
            "Averaging",
        ],
        "xlim": 41,
        "ylim": (-0.1, 1.25),
    },
    "sparse_linear_regression": {
        "run_dir": "models/meta_sparse_linear_regression",
        "title": "(b) Sparse linear functions",
        "models": [
            "Transformer",
            "Least Squares",
            "Averaging",
            "Lasso (alpha=0.01)",
        ],
        "xlim": 41,
        "ylim": (-0.1, 1.25),
    },
    "decision_tree": {
        "run_dir": "models/meta_decision_tree",
        "title": "(c) Decision trees",
        "models": [
            "Transformer",
            "3-Nearest Neighbors",
            "Greedy Tree Learning",
            "XGBoost",
        ],
        "xlim": 101,
        "ylim": (0.0, 1.6),
    },
    "relu_2nn_regression": {
        "run_dir": "models/meta_relu_2nn_regression",
        "title": "(d) 2-layer NN",
        "models": [
            "Transformer",
            "Least Squares",
            "3-Nearest Neighbors",
            "2-layer NN, GD",
        ],
        "xlim": 101,
        "ylim": (-0.1, 1.25),
    },
}


def find_latest_run(run_root):
    run_root = Path(run_root)
    candidates = [
        p for p in run_root.iterdir()
        if p.is_dir() and (p / "config.yaml").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories with config.yaml found in {run_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_run_path(repo_root, task_name, run_path_arg):
    if run_path_arg:
        run_path = Path(run_path_arg)
        if not run_path.is_absolute():
            run_path = Path(repo_root) / run_path
        return run_path
    return find_latest_run(Path(repo_root) / TASK_SPECS[task_name]["run_dir"])


def rename_model(model_name, conf, transformer_label="Transformer"):
    if "gpt2" in model_name:
        return transformer_label
    return baseline_names(model_name)


def get_error_normalization(conf):
    task_name = conf.training.task
    if task_name == "sparse_linear_regression":
        return conf.training.task_kwargs.get("sparsity", conf.model.n_dims)
    if task_name in ["decision_tree", "relu_2nn_regression"]:
        return 1
    return conf.model.n_dims


def get_task_metrics(
    run_path,
    eval_name,
    cache=True,
    transformer_label="Transformer",
    skip_baselines=False,
    baseline_models=None,
    baseline_model_names=None,
    eval_kwargs_overrides=None,
    normalize=True,
):
    metrics = get_run_metrics(
        str(run_path),
        cache=cache,
        skip_baselines=skip_baselines,
        eval_names=[eval_name],
        baseline_models=baseline_models,
        baseline_model_names=baseline_model_names,
        eval_kwargs_overrides=eval_kwargs_overrides,
    )
    if eval_name not in metrics:
        raise KeyError(
            f"Eval '{eval_name}' not found in {run_path / 'metrics.json'}. "
            f"Available evals: {list(metrics.keys())}"
        )
    _, conf = get_model_from_run(str(run_path), only_conf=True)
    normalization = get_error_normalization(conf) if normalize else 1
    requested_baselines = None
    if baseline_models is not None:
        requested_baselines = {
            name
            for baseline in baseline_models
            for name in (baseline.name, rename_model(baseline.name, conf))
        }
    if baseline_model_names is not None:
        requested_baselines = set(baseline_model_names)
    renamed = {}
    for model_name, model_metrics in metrics[eval_name].items():
        renamed_model = rename_model(
            model_name,
            conf,
            transformer_label=transformer_label,
        )
        if (
            requested_baselines is not None
            and renamed_model != transformer_label
            and model_name not in requested_baselines
            and renamed_model not in requested_baselines
        ):
            continue
        renamed[renamed_model] = {
            k: [v_i / normalization for v_i in v]
            for k, v in model_metrics.items()
        }
    return renamed


def plot_task(ax, task_name, task_metrics, model_order=None):
    spec = TASK_SPECS[task_name]
    color_idx = 0
    ax.axhline(1.0, ls="--", color="gray")

    if model_order is None:
        model_order = spec["models"]

    for model_name in model_order:
        if model_name not in task_metrics:
            continue
        model_metrics = task_metrics[model_name]
        mean = model_metrics["mean"][: spec["xlim"]]
        low = model_metrics["bootstrap_low"][: spec["xlim"]]
        high = model_metrics["bootstrap_high"][: spec["xlim"]]
        xs = list(range(len(mean)))
        color = palette[color_idx % len(palette)]
        linestyle = "--" if "sign preproc" in model_name else "-"
        ax.plot(xs, mean, linestyle, label=model_name, color=color, lw=2)
        ax.fill_between(xs, low, high, alpha=0.3, color=color)
        color_idx += 1

    ax.set_title(spec["title"])
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-0.5, spec["xlim"] - 0.5)
    ax.set_ylim(*spec["ylim"])
    ax.legend(loc="upper right", frameon=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a 2x2 paper-style figure for the four meta-learning tasks."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root.",
    )
    parser.add_argument(
        "--eval-name",
        default="maml",
        choices=["maml", "standard"],
        help="Which eval view to plot.",
    )
    parser.add_argument(
        "--linear-run",
        default=None,
        help="Optional run path for meta linear regression.",
    )
    parser.add_argument(
        "--sparse-run",
        default=None,
        help="Optional run path for meta sparse linear regression.",
    )
    parser.add_argument(
        "--relu-run",
        default=None,
        help="Optional run path for meta ReLU 2NN regression.",
    )
    parser.add_argument(
        "--tree-run",
        default=None,
        help="Optional run path for meta decision tree.",
    )
    parser.add_argument(
        "--output",
        default="plots/meta_paper_figure.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--pdf-output",
        default="plots/meta_paper_figure.pdf",
        help="Optional PDF output path.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root)

    run_paths = {
        "linear_regression": resolve_run_path(repo_root, "linear_regression", args.linear_run),
        "sparse_linear_regression": resolve_run_path(repo_root, "sparse_linear_regression", args.sparse_run),
        "decision_tree": resolve_run_path(repo_root, "decision_tree", args.tree_run),
        "relu_2nn_regression": resolve_run_path(repo_root, "relu_2nn_regression", args.relu_run),
    }

    task_metrics = {
        task_name: get_task_metrics(run_path, args.eval_name, cache=True)
        for task_name, run_path in run_paths.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ordered_tasks = [
        "linear_regression",
        "sparse_linear_regression",
        "decision_tree",
        "relu_2nn_regression",
    ]
    for ax, task_name in zip(axes.flat, ordered_tasks):
        plot_task(ax, task_name, task_metrics[task_name])

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if args.pdf_output:
        pdf_path = Path(args.pdf_output)
        if not pdf_path.is_absolute():
            pdf_path = repo_root / pdf_path
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")

    print("Saved figure to:", output_path)
    if args.pdf_output:
        print("Saved figure to:", pdf_path)
    print("Run paths used:")
    for task_name, run_path in run_paths.items():
        print(f"  {task_name}: {run_path}")


if __name__ == "__main__":
    main()
