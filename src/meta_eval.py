import numpy as np
import torch
from torch.func import functional_call, grad as func_grad, vmap

from samplers import get_data_sampler
from tasks import get_task_sampler
from meta_utils import inner_adapt, trainable_model_params, update_params_with_grads


def aggregate_metrics_with_nans(metrics, bootstrap_trials=1000):
    """Aggregate a (num_eval, n_points) tensor while ignoring NaNs per column."""
    metrics = metrics.float()
    n_points = metrics.shape[1]

    mean = torch.full((n_points,), float("nan"))
    std = torch.full((n_points,), float("nan"))
    bootstrap_low = torch.full((n_points,), float("nan"))
    bootstrap_high = torch.full((n_points,), float("nan"))

    for k in range(n_points):
        vals = metrics[:, k]
        vals = vals[~torch.isnan(vals)]
        if vals.numel() == 0:
            continue

        mean[k] = vals.mean()
        if vals.numel() > 1:
            std[k] = vals.std(unbiased=True)
        else:
            std[k] = 0.0

        bootstrap_indices = torch.randint(
            vals.numel(), size=(bootstrap_trials, vals.numel())
        )
        bootstrap_means = vals[bootstrap_indices].mean(dim=1).sort()[0]
        bootstrap_low[k] = bootstrap_means[int(0.05 * bootstrap_trials)]
        bootstrap_high[k] = bootstrap_means[int(0.95 * bootstrap_trials)]

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "bootstrap_low": bootstrap_low.tolist(),
        "bootstrap_high": bootstrap_high.tolist(),
    }


def collect_maml_metrics_loop(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    inner_lr,
    num_inner_steps,
    inner_lrs=None,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    stride=1,
):
    """Collect per-example MAML losses with train_meta.maml_pointwise_eval semantics.

    The loop order and adaptation logic intentionally mirror the verified
    `maml_pointwise_eval` in `train_meta.py`; the only difference is that this
    function stores one row per eval example so the usual eval pipeline can
    return mean/std/bootstrap summaries.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    base_params = {
        name: p.detach()
        for name, p in trainable_model_params(model).items()
    }
    if inner_lrs is None:
        inner_lrs = inner_lr
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    metrics = torch.full((num_eval_examples, n_points), float("nan"))
    ks = list(range(0, n_points, stride))
    if (n_points - 1) not in ks:
        ks.append(n_points - 1)

    n_batches = (num_eval_examples + batch_size - 1) // batch_size
    example_offset = 0

    for batch_idx in range(n_batches):
        batch_count = min(batch_size, num_eval_examples - batch_idx * batch_size)
        task = task_sampler()
        xs = data_sampler.sample_xs(n_points, batch_size).to(device)
        ys = task.evaluate(xs).to(device)

        for b in range(batch_count):
            row_idx = example_offset + b
            for k in ks:
                init_params = {
                    name: p.clone().requires_grad_(True)
                    for name, p in base_params.items()
                }
                if k == 0:
                    fast_params = init_params
                else:
                    xs_s = xs[b : b + 1, :k, :]
                    ys_s = ys[b : b + 1, :k]
                    fast_params = inner_adapt(
                        model,
                        init_params,
                        xs_s,
                        ys_s,
                        inner_lrs,
                        num_inner_steps,
                        first_order=True,
                        loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
                    )

                xs_q = xs[b : b + 1, k : k + 1, :]
                ys_q = ys[b : b + 1, k : k + 1]
                with torch.no_grad():
                    pred = functional_call(model, fast_params, (xs_q, ys_q))
                metrics[row_idx, k] = ((pred - ys_q) ** 2).mean()

        example_offset += batch_count

    if was_training:
        model.train()

    return metrics


def collect_maml_metrics(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    inner_lr,
    num_inner_steps,
    inner_lrs=None,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    stride=1,
):
    """Collect MAML losses, vectorizing over eval examples for each k.

    This preserves `maml_pointwise_eval` semantics: for every evaluated context
    length k, adapt on each row's prefix as independent length-1 support
    examples and evaluate y_k as a length-1 query. The speedup comes from vmap
    batching the independent per-row adaptation work.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    base_params = {
        name: p.detach()
        for name, p in trainable_model_params(model).items()
    }
    if inner_lrs is None:
        inner_lrs = inner_lr
    elif isinstance(inner_lrs, dict):
        inner_lrs = {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in inner_lrs.items()
        }
    elif torch.is_tensor(inner_lrs):
        inner_lrs = inner_lrs.detach()
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    metrics = torch.full((num_eval_examples, n_points), float("nan"))
    ks = list(range(0, n_points, stride))
    if (n_points - 1) not in ks:
        ks.append(n_points - 1)

    def support_loss(params, xs_support, ys_support):
        preds = functional_call(
            model,
            params,
            (xs_support.unsqueeze(1), ys_support.unsqueeze(1)),
        ).squeeze(1)
        return ((preds - ys_support) ** 2).mean()

    def adapt_then_query(params, xs_support, ys_support, xs_query, ys_query):
        fast_params = params
        for _ in range(num_inner_steps):
            grads = func_grad(support_loss)(fast_params, xs_support, ys_support)
            grads = {
                name: value.detach()
                for name, value in grads.items()
            }
            fast_params = update_params_with_grads(fast_params, grads, inner_lrs)
        pred = functional_call(
            model,
            fast_params,
            (xs_query.unsqueeze(0), ys_query.unsqueeze(0)),
        ).squeeze(0)
        return ((pred - ys_query) ** 2).mean()

    n_batches = (num_eval_examples + batch_size - 1) // batch_size
    example_offset = 0

    for batch_idx in range(n_batches):
        batch_count = min(batch_size, num_eval_examples - batch_idx * batch_size)
        task = task_sampler()
        xs_full = data_sampler.sample_xs(n_points, batch_size).to(device)
        ys_full = task.evaluate(xs_full).to(device)
        xs = xs_full[:batch_count]
        ys = ys_full[:batch_count]

        for k in ks:
            xs_query = xs[:, k : k + 1, :]
            ys_query = ys[:, k : k + 1]
            if k == 0:
                preds = functional_call(model, base_params, (xs_query, ys_query))
                losses = ((preds - ys_query) ** 2).mean(dim=1)
            else:
                losses = vmap(
                    adapt_then_query,
                    in_dims=(None, 0, 0, 0, 0),
                )(
                    base_params,
                    xs[:, :k, :],
                    ys[:, :k],
                    xs_query,
                    ys_query,
                )
            metrics[
                example_offset : example_offset + batch_count,
                k,
            ] = losses.detach().cpu()

        example_offset += batch_count

    if was_training:
        model.train()

    return metrics


def eval_model_maml(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    inner_lr,
    num_inner_steps,
    inner_lrs=None,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    stride=1,
):
    metrics = collect_maml_metrics(
        model=model,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=n_points,
        inner_lr=inner_lr,
        num_inner_steps=num_inner_steps,
        inner_lrs=inner_lrs,
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_sampler_kwargs,
        task_sampler_kwargs=task_sampler_kwargs,
        stride=stride,
    )
    return aggregate_metrics_with_nans(metrics)
