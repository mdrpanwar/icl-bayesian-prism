import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from meta_eval import eval_model_maml
from meta_utils import (
    INNER_LR_MODULE_NAME,
    configure_inner_lrs,
    get_inner_lrs,
    load_state_dict_allow_missing_inner_lrs,
)
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler, PolynomialsUnbiasedPoints, PolynomialsDegTwoMonomialSelectionUnbiased, NoisyLinearRegressionTaskDiversity, FourierSeriesV2Multitask
import time


def preserve_rng_state(func):
    """Keep deterministic evaluation from perturbing the training RNG stream."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            return func(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return wrapper


def _checkpoint_map_location():
    return None if torch.cuda.is_available() else torch.device("cpu")


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)
    if hasattr(conf, "meta"):
        configure_inner_lrs(model, conf.meta)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location=_checkpoint_map_location())
        load_state_dict_allow_missing_inner_lrs(model, state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path, map_location=_checkpoint_map_location())
        load_state_dict_allow_missing_inner_lrs(model, state_dict)

    return model, conf


def load_into_model_from_run(model, run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as fp:
            conf = Munch.fromDict(yaml.safe_load(fp))
        if hasattr(conf, "meta"):
            configure_inner_lrs(model, conf.meta)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location=_checkpoint_map_location())
        load_state_dict_allow_missing_inner_lrs(model, state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path, map_location=_checkpoint_map_location())
        load_state_dict_allow_missing_inner_lrs(model, state_dict)

    return model

# Functions for evaluation


def eval_batch(model, task_sampler, xs, eval_ood, xs_p=None, data_sampler=None, excess_tensors_eval=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        if isinstance(task, PolynomialsUnbiasedPoints):
            bsize = xs.shape[0]
            n_points = xs.shape[1]
            n_dims = xs.shape[2]
            assert n_dims==1, "n_dims is not 1, please change sampling logic for ys s.t. it is from same distribution as xs but is of shape [batch, n_points]"
            ys, _ = task.rejection_sample_to_form_batch(xs, data_sampler, data_sampler_args={}, bsize=bsize, n_points=n_points, excess_tensors=excess_tensors_eval)
        elif isinstance(task, PolynomialsDegTwoMonomialSelectionUnbiased):
            ys = task.evaluate_ood(xs) if eval_ood else task.evaluate(xs, mode="eval")
        elif isinstance(task, NoisyLinearRegressionTaskDiversity):
            ys = task.evaluate(xs) # for both cases, i.e. eval_ood = True and False
        elif isinstance(task, FourierSeriesV2Multitask):
            ys = task.evaluate_ood(xs) if eval_ood else task.evaluate(xs, mode="eval")
        else:
            ys = task.evaluate_ood(xs) if eval_ood else task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model_batch(model, kwargs, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

    task_name = kwargs["task_name"]
    data_name = kwargs["data_name"]
    n_dims = kwargs["n_dims"]
    n_points = kwargs["n_points"]
    batch_size = kwargs["batch_size"]
    prompting_strategy = kwargs["prompting_strategy"]
    data_sampler_kwargs = dict(kwargs.get("data_sampler_kwargs", {}))
    if seed is not None and data_name == "gaussian":
        # Each evaluation batch gets a deterministic, method-independent x set.
        data_sampler_kwargs["data_seed"] = seed
    task_sampler_kwargs = kwargs.get("task_sampler_kwargs", {})
    excess_tensors_eval = kwargs.get("excess_tensors_eval")
    eval_ood = kwargs.get("eval_ood", False)

    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    if task_name == "noisy_lr_task_diversity":
        if eval_ood:
            task_sampler = lambda **args: NoisyLinearRegressionTaskDiversity(n_dims, batch_size, **args, **task_sampler_kwargs)
        else:
            task_rand_gen = torch.Generator().manual_seed(int(task_sampler_kwargs["task_seed"] + time.time()))
            noise_rand_gen = torch.Generator().manual_seed(int(task_sampler_kwargs["noise_seed"] + time.time()))
            task_sampler = lambda **args: NoisyLinearRegressionTaskDiversity(n_dims, batch_size, task_rand_gen=task_rand_gen,
                        noise_rand_gen=noise_rand_gen, **args, **task_sampler_kwargs)
    else:
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size, **task_sampler_kwargs
        )

    generating_func = globals()[f"gen_{prompting_strategy}"]
    xs, xs_p = generating_func(data_sampler, n_points, batch_size)
    return eval_batch(
        model,
        task_sampler,
        xs,
        eval_ood,
        xs_p,
        data_sampler if task_name == "polynomials_unbiased_points" else None,
        excess_tensors_eval if task_name == "polynomials_unbiased_points" else None,
    )


def eval_model_batch_worker(args):
    model, kwargs, batch_idx, base_seed = args
    return batch_idx, eval_model_batch(model, kwargs, seed=base_seed + batch_idx)



@preserve_rng_state
def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    excess_tensors_eval=None,
    eval_ood=False,
    parallel_batches=1,
    parallel_seed=0,
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    n_batches = num_eval_examples // batch_size
    batch_kwargs = {
        "task_name": task_name,
        "data_name": data_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "prompting_strategy": prompting_strategy,
        "data_sampler_kwargs": data_sampler_kwargs,
        "task_sampler_kwargs": task_sampler_kwargs,
        "excess_tensors_eval": excess_tensors_eval,
        "eval_ood": eval_ood,
    }
    all_metrics = []

    if parallel_batches > 1 and not isinstance(model, torch.nn.Module):
        all_metrics = [None] * n_batches
        max_workers = min(parallel_batches, n_batches)
        worker_args = [
            (model, batch_kwargs, batch_idx, parallel_seed)
            for batch_idx in range(n_batches)
        ]
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(eval_model_batch_worker, arg) for arg in worker_args]
            for future in tqdm(
                as_completed(futures),
                total=n_batches,
                desc=f"{getattr(model, 'name', 'model')} batches",
                leave=False,
            ):
                batch_idx, metrics = future.result()
                all_metrics[batch_idx] = metrics
    else:
        for i in tqdm(
            range(n_batches),
            desc=f"{getattr(model, 'name', 'model')} batches",
            leave=False,
        ):
            metrics = eval_model_batch(model, batch_kwargs, seed=parallel_seed + i)
            all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = getattr(conf.training, "eval_n_points", None)
    if n_points is None:
        n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if "meta" in conf:
        evaluation_kwargs["maml"] = {
            "prompting_strategy": "standard",
            "eval_mode": "maml",
            "inner_lr": conf.meta.inner_lr,
            "inner_lr_mode": getattr(conf.meta, "inner_lr_mode", "fixed"),
            "inner_lr_parameterization": getattr(
                conf.meta, "inner_lr_parameterization", "direct"
            ),
            "inner_lr_bound": getattr(conf.meta, "inner_lr_bound", None),
            "num_inner_steps": conf.meta.num_inner_steps,
            "stride": conf.meta.meta_eval_stride,
        }
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_eval_metrics(model, kwargs):
    kwargs = kwargs.copy()
    eval_mode = kwargs.pop("eval_mode", "standard")

    if eval_mode == "standard":
        return eval_model(model, **kwargs)

    if eval_mode == "maml":
        if isinstance(model, torch.nn.Module):
            kwargs.pop("prompting_strategy")
            if hasattr(model, INNER_LR_MODULE_NAME):
                inner_lr_config = {
                    "inner_lr_mode": kwargs.pop("inner_lr_mode", "learned_per_param"),
                    "inner_lr": kwargs["inner_lr"],
                    "inner_lr_parameterization": kwargs.pop(
                        "inner_lr_parameterization", "direct"
                    ),
                    "inner_lr_bound": kwargs.pop("inner_lr_bound", None),
                }
                kwargs["inner_lrs"] = get_inner_lrs(
                    model,
                    inner_lr_config,
                )
            kwargs.pop("inner_lr_mode", None)
            kwargs.pop("inner_lr_parameterization", None)
            kwargs.pop("inner_lr_bound", None)
            return eval_model_maml(model, **kwargs)
        kwargs.pop("inner_lr")
        kwargs.pop("num_inner_steps")
        kwargs.pop("stride")
        return eval_model(model, **kwargs)

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def compute_evals(
    all_models,
    evaluation_kwargs,
    save_path=None,
    recompute=False,
    progress_desc="evaluations",
):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items(), desc=progress_desc):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            print(f"    computing {eval_name} for {model.name}", flush=True)
            metrics[model.name] = compute_eval_metrics(model, kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path,
    step=-1,
    cache=True,
    skip_model_load=False,
    skip_baselines=False,
    eval_names=None,
    baseline_models=None,
    baseline_model_names=None,
    eval_kwargs_overrides=None,
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        all_models = [model]
        if not skip_baselines:
            if baseline_models is None:
                baselines = models.get_relevant_baselines(conf.training.task)
            else:
                baselines = list(baseline_models)
            if baseline_model_names is not None:
                requested = set(baseline_model_names)
                baselines = [
                    baseline
                    for baseline in baselines
                    if baseline.name in requested or baseline_names(baseline.name) in requested
                ]
            all_models += baselines
    evaluation_kwargs = build_evals(conf)
    if eval_names is not None:
        if isinstance(eval_names, str):
            eval_names = [eval_names]
        missing = [name for name in eval_names if name not in evaluation_kwargs]
        if missing:
            raise KeyError(
                f"Eval(s) {missing} not available for {run_path}. "
                f"Available evals: {list(evaluation_kwargs.keys())}"
            )
        evaluation_kwargs = {name: evaluation_kwargs[name] for name in eval_names}
    if eval_kwargs_overrides:
        for eval_name in evaluation_kwargs:
            evaluation_kwargs[eval_name].update(eval_kwargs_overrides)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(
        all_models,
        evaluation_kwargs,
        save_path,
        recompute,
        progress_desc=os.path.basename(run_path),
    )
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if name == "decision_tree_max_depth=4_preprocess=sign":
        return "Greedy Tree Learning\n(w/ sign preproc.)"
    if name == "xgboost_preprocess=sign":
        return "XGBoost\n(w/ sign preproc.)"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
