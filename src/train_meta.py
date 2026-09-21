"""
Meta-learning (MAML) trainer for the in-context-learning function classes.

Mirrors `train.py` but the inner loop is a per-task adaptation step on a
support prompt and the outer loss is the prediction error on m fresh query
points (drawn from the same task) after that adaptation.

Key design choices are documented in the README/notes; see the `meta` block of
the config schema (defined in `schema.py`) for the knobs.
"""

import os
import sys
import uuid
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad as func_grad
from quinine import QuinineArgumentParser
from tqdm import tqdm
from munch import Munch
from transformers import get_scheduler
import yaml
import wandb

from schema import meta_training_schema
from models import build_model
from samplers import get_data_sampler, sample_scale
from tasks import get_task_sampler
from curriculum import Curriculum
from eval import (
    eval_model,
    get_run_metrics,
    load_into_model_from_run,
    preserve_rng_state,
)
from meta_utils import (
    configure_inner_lrs,
    get_inner_lrs,
    inner_adapt,
    inner_lr_stats,
    load_state_dict_allow_missing_inner_lrs,
    trainable_model_params,
    update_params_with_grads,
)
from checkpointing import (
    PreemptionHandler,
    atomic_torch_save,
    load_training_state,
    restore_rng_state,
    save_training_state,
)




torch.backends.cudnn.benchmark = True


def sanitize_run_name(run_name):
    if run_name in (None, ""):
        return None
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_." else "-"
        for ch in str(run_name).strip()
    )
    cleaned = cleaned.strip("-_.")
    return cleaned[:80] or None


def apply_run_name(args):
    if getattr(args, "run_name", None) not in (None, ""):
        args.wandb.name = args.run_name


def get_run_dir_name(args, run_id):
    slug = sanitize_run_name(getattr(args, "run_name", None))
    if slug is None:
        return run_id
    return f"{slug}-{run_id}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_meta_attention_backend(args):
    """Reject unsupported attention/backend combinations without mutation."""
    if args.meta.first_order and args.model.attn_implementation == "sdpa":
        raise ValueError(
            "The current vectorized FOMAML implementation uses functorch vmap, "
            "which is incompatible with PyTorch's fused SDPA kernels in this "
            "training path and can fail with `LSE is not correctly aligned "
            "(strideH)`. Set model.attn_implementation: eager for this "
            "implementation, or add an explicit non-vmap SDPA training path "
            "before using sdpa."
        )


# ---------------------------------------------------------------------------
# Support-size sampling
# ---------------------------------------------------------------------------


def get_fixed_support_size(meta_args):
    """Return the configured fixed support size, with no sampling."""
    k = meta_args.fixed_support_size
    if k is None:
        raise ValueError(
            "meta.fixed_support_size must be set when "
            "meta.vary_support_size is 'fixed'."
        )
    if k < 0:
        raise ValueError(f"meta.fixed_support_size must be >= 0, got {k}.")
    return k


def get_support_size_choices(meta_args):
    spec = getattr(meta_args, "support_size_choices", None)
    if spec in (None, ""):
        raise ValueError(
            "meta.support_size_choices must be set when "
            "meta.vary_support_size is a choices mode"
        )
    try:
        choices = [int(value.strip()) for value in str(spec).split(",")]
    except ValueError as exc:
        raise ValueError(
            "meta.support_size_choices must be comma-separated integers"
        ) from exc
    if not choices or any(k < 0 for k in choices):
        raise ValueError("support-size choices must be non-negative")
    if len(set(choices)) != len(choices):
        raise ValueError("support-size choices must not contain duplicates")
    return choices


def sample_support_size(meta_args, curriculum_n_points, curriculum_end):
    mode = meta_args.vary_support_size
    if mode == "match_curriculum":
        # Random k from the currently available curriculum prefix.
        return random.randint(0, curriculum_n_points - 1)
    if mode == "full_range":
        # Random k from the final curriculum range, regardless of current step.
        return random.randint(0, curriculum_end - 1)
    if mode == "fixed":
        # No sampling: always use exactly meta.fixed_support_size.
        return get_fixed_support_size(meta_args)
    if mode == "choices":
        return random.choice(get_support_size_choices(meta_args))
    if mode in {"choices_parallel", "choices_sequential"}:
        raise ValueError(
            f"{mode} returns multiple support sizes; use sample_support_sizes"
        )
    raise ValueError(f"Unknown vary_support_size: {mode}")


def max_support_size_for_step(meta_args, curriculum_n_points, curriculum_end):
    """Largest support set size that can be used at the current step."""
    mode = meta_args.vary_support_size
    if mode == "fixed":
        return get_fixed_support_size(meta_args)
    if mode in {"choices", "choices_parallel", "choices_sequential"}:
        return max(get_support_size_choices(meta_args))
    if mode == "full_range":
        return max(curriculum_end - 1, 0)
    if mode == "match_curriculum":
        return max(curriculum_n_points - 1, 0)
    raise ValueError(f"Unknown vary_support_size: {mode}")


def sample_support_sizes(meta_args, curriculum_n_points, curriculum_end):
    """Return the support sizes to evaluate for one task.

    In the default mode this preserves the old behavior: one randomly sampled
    support size per task. When multi-k support is enabled, each task contributes
    a small set of independent adapt/eval branches: zero-shot, a random small k,
    a random medium k, and the current maximum k. Fixed support size is always
    a single branch, even if multi-k support is enabled in the inherited config.
    """
    if meta_args.vary_support_size == "fixed":
        return [sample_support_size(meta_args, curriculum_n_points, curriculum_end)]

    if meta_args.vary_support_size in {"choices_parallel", "choices_sequential"}:
        return sorted(get_support_size_choices(meta_args))

    if meta_args.vary_support_size == "choices":
        return [sample_support_size(meta_args, curriculum_n_points, curriculum_end)]

    if not getattr(meta_args, "multi_k_support", False):
        return [sample_support_size(meta_args, curriculum_n_points, curriculum_end)]

    max_k = max(curriculum_n_points - 1, 0)
    if max_k == 0:
        return [0]

    small_hi = max(max_k // 3, 1)
    medium_lo = min(small_hi + 1, max_k)
    medium_hi = max((2 * max_k) // 3, medium_lo)

    small_k = random.randint(1, small_hi)
    medium_k = random.randint(medium_lo, medium_hi)
    return [0, small_k, medium_k, max_k]


# ---------------------------------------------------------------------------
# Outer training step
# ---------------------------------------------------------------------------


def apply_outer_update(model, optimizer, loss, meta_args):
    """Backpropagate one outer loss with optional clipping and fail-fast checks."""
    max_norm = getattr(meta_args, "outer_grad_clip_norm", None)
    if max_norm is not None:
        max_norm = float(max_norm)
        if max_norm <= 0:
            raise ValueError("meta.outer_grad_clip_norm must be positive")

    fail_on_nonfinite = bool(getattr(meta_args, "fail_on_nonfinite", False))
    skip_threshold = getattr(meta_args, "max_outer_grad_norm_before_skip", None)
    if skip_threshold is not None:
        skip_threshold = float(skip_threshold)
        if skip_threshold <= 0:
            raise ValueError("meta.max_outer_grad_norm_before_skip must be positive")
    if skip_threshold is not None and not bool(torch.isfinite(loss.detach())):
        optimizer.zero_grad(set_to_none=True)
        return {
            "outer_grad_norm": float("nan"), "outer_grad_was_clipped": 0.0,
            "outer_grad_clip_norm": max_norm, "outer_update_skipped": 1.0,
        }
    if fail_on_nonfinite and not bool(torch.isfinite(loss.detach())):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Non-finite meta loss detected before backward: {loss.detach().item()}"
        )

    loss.backward()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if max_norm is None:
        if parameters:
            grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(parameter.grad.detach().float())
                        for parameter in parameters
                    ]
                )
            )
        else:
            grad_norm = loss.new_zeros(())
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            error_if_nonfinite=fail_on_nonfinite and skip_threshold is None,
        )

    grad_norm_value = float(grad_norm.detach().item())
    if skip_threshold is not None and (
        not np.isfinite(grad_norm_value) or grad_norm_value > skip_threshold
    ):
        optimizer.zero_grad(set_to_none=True)
        return {
            "outer_grad_norm": grad_norm_value,
            "outer_grad_was_clipped": 0.0,
            "outer_grad_clip_norm": max_norm,
            "outer_update_skipped": 1.0,
        }
    if fail_on_nonfinite and not np.isfinite(grad_norm_value):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Non-finite outer gradient norm detected: {grad_norm.detach().item()}"
        )

    was_clipped = bool(
        max_norm is not None and grad_norm.detach().item() > max_norm
    )
    optimizer.step()
    return {
        "outer_grad_norm": float(grad_norm.detach().item()),
        "outer_grad_was_clipped": float(was_clipped),
        "outer_grad_clip_norm": max_norm,
        "outer_update_skipped": 0.0,
    }


def meta_train_step_sequential(
    model, optimizer, xs, ys, meta_args, n_query, task_ids=None,
    return_diagnostics=False,
):
    """Carry one fast-parameter path through every configured support prefix.

    For choices k1 < k2 < ..., adapt on prefix k1, score the common query
    block, continue adapting those fast parameters on prefix k2, score again,
    and so on. The outer objective is the mean of all stage-query losses.
    """
    optimizer.zero_grad()
    batch_size, n_total, n_dims = xs.shape
    support_sizes = sorted(get_support_size_choices(meta_args))
    support_capacity = n_total - n_query
    if support_sizes[-1] > support_capacity:
        raise ValueError(
            "Support size exceeds the sampled support block. "
            f"Got max k={support_sizes[-1]} with "
            f"support_capacity={support_capacity}."
        )

    params = trainable_model_params(model)
    inner_lrs = get_inner_lrs(model, meta_args)
    k_max = support_sizes[-1]
    positions = torch.arange(k_max, device=xs.device)
    xs_support = xs[:, :k_max, :]
    ys_support = ys[:, :k_max]
    xs_query = xs[:, -n_query:, :].reshape(batch_size, n_query, 1, n_dims)
    ys_query = ys[:, -n_query:].reshape(batch_size, n_query, 1)
    support_task_ids = task_ids[:, :k_max] if task_ids is not None else None
    query_task_ids = (
        task_ids[:, -n_query:].reshape(batch_size, n_query, 1)
        if task_ids is not None else None
    )

    def support_loss(fast_params, xs_s, ys_s, mask, ids_s=None):
        kwargs = {"task_ids": ids_s.unsqueeze(1)} if ids_s is not None else None
        preds = functional_call(
            model, fast_params, (xs_s.unsqueeze(1), ys_s.unsqueeze(1)), kwargs
        ).squeeze(1)
        squared_error = (preds - ys_s) ** 2
        return (squared_error * mask).sum() / mask.sum().clamp(min=1.0)

    def adapt_and_score(
        base_params, xs_s, ys_s, xs_q, ys_q, ids_s=None, ids_q=None,
    ):
        fast_params = base_params
        stage_losses = []
        for support_size in support_sizes:
            if support_size > 0:
                mask = (positions < support_size).to(xs.dtype)
                for _ in range(meta_args.num_inner_steps):
                    grads = func_grad(support_loss)(
                        fast_params, xs_s, ys_s, mask, ids_s
                    )
                    if meta_args.first_order:
                        grads = {name: value.detach() for name, value in grads.items()}
                    fast_params = update_params_with_grads(
                        fast_params, grads, inner_lrs
                    )
            kwargs = {"task_ids": ids_q} if ids_q is not None else None
            preds = functional_call(model, fast_params, (xs_q, ys_q), kwargs)
            stage_losses.append(((preds - ys_q) ** 2).mean())
        return torch.stack(stage_losses)

    if task_ids is None:
        stage_losses = vmap(
            adapt_and_score, in_dims=(None, 0, 0, 0, 0)
        )(params, xs_support, ys_support, xs_query, ys_query)
    else:
        stage_losses = vmap(
            adapt_and_score, in_dims=(None, 0, 0, 0, 0, 0, 0)
        )(
            params, xs_support, ys_support, xs_query, ys_query,
            support_task_ids, query_task_ids,
        )

    meta_loss = stage_losses.mean()
    update_stats = apply_outer_update(model, optimizer, meta_loss, meta_args)
    flat_ks = support_sizes * batch_size
    per_task_losses = stage_losses.detach().mean(dim=1).tolist()
    result = (meta_loss.item(), flat_ks, per_task_losses)
    if return_diagnostics:
        return (*result, update_stats)
    return result


def meta_train_step(
    model,
    optimizer,
    task,
    xs,
    ys,
    meta_args,
    curriculum_n_points,
    curriculum_end,
    loss_func,
    n_query,
    task_ids=None,
    return_diagnostics=False,
):
    """Vectorized outer step. Same per-task semantics as `meta_train_step_loop`:
    each task b independently samples one or more support sizes k_b in
    [0, curriculum_n_points-1], adapts on its first k_b examples, and is
    evaluated on its own n_query length-1 query prompts.

    The meta-batch is split into the k_b > 0 group (one vmap'd
    adapt-then-query call with right-padding to k_max + a per-task loss
    mask) and the k_b == 0 group (one bulk forward, no inner step). GPT-2's
    causal attention naturally ignores right-padded positions, so no
    attention mask is needed -- only a loss mask on the support side.

    Support examples are treated as independent length-1 prompts during the
    inner update. Query examples are also independent length-1 prompts.

    `loss_func` is currently unused: support and query both use plain MSE,
    matching the existing tasks (linear regression, etc.). When non-MSE
    meta-tasks are added, generalize the masked support loss below.
    """
    if meta_args.vary_support_size == "choices_sequential":
        return meta_train_step_sequential(
            model, optimizer, xs, ys, meta_args, n_query,
            task_ids=task_ids, return_diagnostics=return_diagnostics,
        )

    optimizer.zero_grad()
    B = xs.shape[0]
    n_total = xs.shape[1]
    n_dims = xs.shape[2]

    # Trainable model params only. Frozen wpe.weight and learned inner-loop LR
    # parameters are excluded; functional_call falls back to stored values for
    # params absent from this dict.
    params = trainable_model_params(model)
    inner_lrs = get_inner_lrs(model, meta_args)

    # Each task's queries become n_query independent length-1 prompts.
    xs_q = xs[:, n_total - n_query : n_total, :].reshape(B, n_query, 1, n_dims)
    ys_q = ys[:, n_total - n_query : n_total].reshape(B, n_query, 1)
    task_ids_q = (
        task_ids[:, n_total - n_query : n_total].reshape(B, n_query, 1)
        if task_ids is not None
        else None
    )

    ks_by_task = [
        sample_support_sizes(meta_args, curriculum_n_points, curriculum_end)
        for _ in range(B)
    ]
    flat_task_ids = []
    flat_ks = []
    flat_weights = []
    for task_id, task_ks in enumerate(ks_by_task):
        branch_weight = 1.0 / len(task_ks)
        for k in task_ks:
            flat_task_ids.append(task_id)
            flat_ks.append(k)
            flat_weights.append(branch_weight)

    task_ids_t = torch.tensor(flat_task_ids, device=xs.device)
    ks_t = torch.tensor(flat_ks, device=xs.device)
    branch_weights_t = torch.tensor(flat_weights, device=xs.device, dtype=xs.dtype)
    has_support = (ks_t > 0).nonzero(as_tuple=True)[0]
    no_support = (ks_t == 0).nonzero(as_tuple=True)[0]
    support_capacity = n_total - n_query
    if flat_ks and max(flat_ks) > support_capacity:
        raise ValueError(
            "Support size exceeds the sampled support block. "
            f"Got max k={max(flat_ks)} with support_capacity={support_capacity}. "
            "Increase the sampled prompt length or lower meta.fixed_support_size."
        )

    per_task_losses_log = torch.zeros(B, device=xs.device)
    total_loss = xs.new_zeros(())

    # ---- k_b > 0 tasks: vmap'd adapt-then-query ----
    if has_support.numel() > 0:
        task_ids_pos = task_ids_t[has_support]
        ks_pos = ks_t[has_support]
        k_max = int(ks_pos.max().item())

        xs_s = xs[task_ids_pos, :k_max, :]
        ys_s = ys[task_ids_pos, :k_max]
        positions = torch.arange(k_max, device=xs.device).unsqueeze(0)
        mask = (positions < ks_pos.unsqueeze(1)).to(xs.dtype)
        xs_q_p = xs_q[task_ids_pos]
        ys_q_p = ys_q[task_ids_pos]
        task_ids_s = (
            task_ids[task_ids_pos, :k_max] if task_ids is not None else None
        )
        task_ids_q_p = (
            task_ids_q[task_ids_pos] if task_ids_q is not None else None
        )
        branch_weights_p = branch_weights_t[has_support]

        def support_loss(p, xs_s_b, ys_s_b, mask_b, task_ids_s_b=None):
            kwargs = (
                {"task_ids": task_ids_s_b.unsqueeze(1)}
                if task_ids_s_b is not None
                else None
            )
            preds = functional_call(
                model,
                p,
                (xs_s_b.unsqueeze(1), ys_s_b.unsqueeze(1)),
                kwargs,
            ).squeeze(1)
            sq = (preds - ys_s_b) ** 2
            return (sq * mask_b).sum() / mask_b.sum().clamp(min=1.0)

        def adapt_then_query(
            p,
            xs_s_b,
            ys_s_b,
            mask_b,
            xs_q_b,
            ys_q_b,
            task_ids_s_b=None,
            task_ids_q_b=None,
        ):
            fp = p
            for _ in range(meta_args.num_inner_steps):
                g = func_grad(support_loss)(
                    fp, xs_s_b, ys_s_b, mask_b, task_ids_s_b
                )
                if meta_args.first_order:
                    g = {n: gv.detach() for n, gv in g.items()}
                fp = update_params_with_grads(fp, g, inner_lrs)
            kwargs = (
                {"task_ids": task_ids_q_b}
                if task_ids_q_b is not None
                else None
            )
            preds_q = functional_call(model, fp, (xs_q_b, ys_q_b), kwargs)
            return ((preds_q - ys_q_b) ** 2).mean()

        if task_ids is None:
            per_task_pos = vmap(
                adapt_then_query, in_dims=(None, 0, 0, 0, 0, 0)
            )(params, xs_s, ys_s, mask, xs_q_p, ys_q_p)
        else:
            per_task_pos = vmap(
                adapt_then_query,
                in_dims=(None, 0, 0, 0, 0, 0, 0, 0),
            )(
                params,
                xs_s,
                ys_s,
                mask,
                xs_q_p,
                ys_q_p,
                task_ids_s,
                task_ids_q_p,
            )
        total_loss = total_loss + (per_task_pos * branch_weights_p).sum()

        with torch.no_grad():
            per_task_losses_log.index_add_(
                0,
                task_ids_pos,
                (per_task_pos.detach() * branch_weights_p).to(per_task_losses_log.dtype),
            )

    # ---- k_b == 0 tasks: no inner step, one batched forward ----
    if no_support.numel() > 0:
        task_ids_zero = task_ids_t[no_support]
        Bz = no_support.numel()
        flat_x = xs_q[task_ids_zero].reshape(Bz * n_query, 1, n_dims)
        flat_y = ys_q[task_ids_zero].reshape(Bz * n_query, 1)
        kwargs = None
        if task_ids_q is not None:
            flat_task_ids = task_ids_q[task_ids_zero].reshape(Bz * n_query, 1)
            kwargs = {"task_ids": flat_task_ids}
        preds_q = functional_call(model, params, (flat_x, flat_y), kwargs)
        per_task_zero = ((preds_q - flat_y) ** 2).reshape(Bz, n_query).mean(dim=1)
        branch_weights_z = branch_weights_t[no_support]
        total_loss = total_loss + (per_task_zero * branch_weights_z).sum()

        with torch.no_grad():
            per_task_losses_log.index_add_(
                0,
                task_ids_zero,
                (per_task_zero.detach() * branch_weights_z).to(per_task_losses_log.dtype),
            )

    meta_loss = total_loss / B
    update_stats = apply_outer_update(model, optimizer, meta_loss, meta_args)

    result = (meta_loss.item(), flat_ks, per_task_losses_log.tolist())
    if return_diagnostics:
        return (*result, update_stats)
    return result


def meta_train_step_loop(
    model,
    optimizer,
    task,
    xs,
    ys,
    meta_args,
    curriculum_n_points,
    curriculum_end,
    loss_func,
    n_query,
):
    """Original per-task Python-loop implementation, preserved for reference and
    A/B comparison. The vmap-vectorized `meta_train_step` is the default; this
    one is called only if you swap the two names.

    Semantics: for each task b, sample k_b in [0, curriculum.n_points-1], use
    xs[b, :k_b], ys[b, :k_b] as support, adapt, then evaluate on xs[b, n_total
    - n_query:], ys[b, n_total - n_query:] reshaped as m parallel length-1
    prompts.
    """
    optimizer.zero_grad()
    meta_batch_size = xs.shape[0]
    n_total = xs.shape[1]
    n_dims = xs.shape[2]

    # Skip frozen params and learned inner-loop LR parameters.
    base_params = trainable_model_params(model)
    inner_lrs = get_inner_lrs(model, meta_args)

    total_query_loss = 0.0
    per_task_query_losses = []
    per_task_k = []

    for b in range(meta_batch_size):
        task_ks = sample_support_sizes(meta_args, curriculum_n_points, curriculum_end)
        per_task_k.extend(task_ks)
        task_query_loss = 0.0
        support_capacity = n_total - n_query
        if max(task_ks) > support_capacity:
            raise ValueError(
                "Support size exceeds the sampled support block. "
                f"Got max k={max(task_ks)} with support_capacity={support_capacity}. "
                "Increase the sampled prompt length or lower meta.fixed_support_size."
            )

        for k_b in task_ks:
            # Support: positions [0, k_b) of task b's prompt. During adaptation,
            # these are evaluated as independent length-1 batch elements.
            xs_s = xs[b : b + 1, :k_b, :]
            ys_s = ys[b : b + 1, :k_b]

            # Inner adaptation
            if meta_args.first_order:
                init_params = {
                    name: p.detach().requires_grad_(True)
                    for name, p in base_params.items()
                }
            else:
                init_params = base_params

            if k_b == 0:
                # No support to adapt on -- skip inner step entirely.
                fast_params = init_params
            else:
                fast_params = inner_adapt(
                    model,
                    init_params,
                    xs_s,
                    ys_s,
                    inner_lrs,
                    meta_args.num_inner_steps,
                    meta_args.first_order,
                    loss_func,
                )

            # Query: m fresh (x, y) pairs from the same task, batched on the
            # batch dim as length-1 prompts so each is a pure k-shot measurement.
            xs_q = xs[b, n_total - n_query : n_total, :].reshape(n_query, 1, n_dims)
            ys_q = ys[b, n_total - n_query : n_total].reshape(n_query, 1)

            loss_weight = 1.0 / len(task_ks)
            preds_q = functional_call(model, fast_params, (xs_q, ys_q))
            loss_q = ((preds_q - ys_q) ** 2).mean() * loss_weight / meta_batch_size

            if meta_args.first_order:
                grads_q = torch.autograd.grad(
                    loss_q, list(fast_params.values()), allow_unused=True
                )
                for (name, p), g in zip(base_params.items(), grads_q):
                    if g is None:
                        continue
                    if p.grad is None:
                        p.grad = g.detach().clone()
                    else:
                        p.grad = p.grad + g.detach()
            else:
                loss_q.backward()

            task_query_loss += loss_q.detach().item() * meta_batch_size

        total_query_loss += task_query_loss
        per_task_query_losses.append(task_query_loss)

    optimizer.step()
    mean_query_loss = total_query_loss / meta_batch_size
    return mean_query_loss, per_task_k, per_task_query_losses


# ---------------------------------------------------------------------------
# MAML-style pointwise eval
# ---------------------------------------------------------------------------



@preserve_rng_state
def maml_pointwise_eval(
    model,
    data_sampler,
    task_sampler,
    n_dims,
    n_points,
    inner_lr,
    num_inner_steps,
    stride,
    num_tasks,
    eval_batch_size,
    inner_lrs=None,
    seed=0,
):
    """Pointwise MAML eval on fixed task/prompt samples.

    For each sampled task/prompt row, evaluate every requested context length k by
    adapting on that row's prefix of length k, then predicting y_k from x_k.
    This matches ICL pointwise eval semantics, except that the model parameters
    are updated before each prediction.

    Eval adapts parameters with inner-loop gradient descent on each support set,
    but does not build higher-order graphs because no meta-gradient is computed.

    Returns a 1D tensor `losses` of length n_points with NaN at non-evaluated k's
    if stride > 1.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    base_params = {
        name: p.detach()
        for name, p in trainable_model_params(model).items()
    }
    if inner_lrs is None:
        inner_lrs = inner_lr

    losses_per_k = torch.full((n_points,), float("nan"))

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    ks = list(range(0, n_points, stride))
    if (n_points - 1) not in ks:
        ks.append(n_points - 1)

    n_batches = (num_tasks + eval_batch_size - 1) // eval_batch_size
    sq_errs_by_k = {k: [] for k in ks}

    for batch_idx in range(n_batches):
        batch_count = min(eval_batch_size, num_tasks - batch_idx * eval_batch_size)
        batch_seed = seed + batch_idx
        random.seed(batch_seed)
        np.random.seed(batch_seed % (2**32 - 1))
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        # Match eval_model: reset the private Gaussian generator per batch,
        # then sample x before the task. ICL and MAML consequently see the
        # same fixed rows for the same seed and batch size.
        if hasattr(data_sampler, "data_rand_gen"):
            data_sampler.data_rand_gen.manual_seed(batch_seed)
        xs = data_sampler.sample_xs(n_points, eval_batch_size).to(device)
        task = task_sampler()
        ys = task.evaluate(xs).to(device)
        episode_task_ids = getattr(task, "task_ids", None)
        if episode_task_ids is not None:
            episode_task_ids = episode_task_ids.to(device)

        for b in range(batch_count):
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
                        loss_func=lambda pred, target: (
                            (pred - target) ** 2
                        ).mean(),
                        task_ids_support=(
                            episode_task_ids[b : b + 1, :k]
                            if episode_task_ids is not None else None
                        ),
                    )

                xs_q = xs[b : b + 1, k : k + 1, :]
                ys_q = ys[b : b + 1, k : k + 1]
                query_kwargs = None
                if episode_task_ids is not None:
                    query_kwargs = {
                        "task_ids": episode_task_ids[b : b + 1, k : k + 1]
                    }
                with torch.no_grad():
                    pred = functional_call(
                        model, fast_params, (xs_q, ys_q), query_kwargs
                    )
                sq_errs_by_k[k].append(((pred - ys_q) ** 2).mean().item())

    for k, sq_errs in sq_errs_by_k.items():
        losses_per_k[k] = float(np.mean(sq_errs))

    if was_training:
        model.train()
    return losses_per_k


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def wandb_log_pointwise(prefix, losses, baseline_loss, step):
    valid_ks = [int(k) for k, v in enumerate(losses.tolist()) if not np.isnan(v)]
    valid_vals = [float(losses[k].item()) for k in valid_ks]
    overall = float(np.mean(valid_vals)) if valid_vals else float("nan")
    wandb.log(
        {
            f"{prefix}/overall_loss": overall,
            f"{prefix}/excess_loss": overall / baseline_loss if baseline_loss > 0 else float("nan"),
            f"{prefix}/pointwise/loss": dict(zip(valid_ks, valid_vals)),
        },
        step=step,
    )


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------


def get_outer_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = None
    if args.training.schedule is not None:
        assert args.training.schedule == "triangle", "Only triangle schedule supported."
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=args.training.train_steps,
        )
    return optimizer, lr_scheduler


def meta_train(model, args):
    configure_inner_lrs(model, args.meta)
    optimizer, lr_scheduler = get_outer_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    state_path = os.path.join(args.out_dir, "state.pt")
    starting_step, resume_state = load_training_state(
        state_path, model, optimizer, lr_scheduler=lr_scheduler,
        curriculum=curriculum,
    )
    warm_start_path = args.training.warm_start_model_checkpoint
    if warm_start_path is not None and resume_state is None:
        load_state_dict_allow_missing_inner_lrs(model, torch.load(
            warm_start_path, map_location=next(model.parameters()).device,
            weights_only=True,
        ))
        print(
            f"[warm-start] weights from {warm_start_path}; "
            "optimizer, RNG, curriculum and step counter start fresh",
            flush=True,
        )

    n_dims = model.n_dims
    meta_args = args.meta
    meta_bsize = meta_args.meta_batch_size
    n_query = meta_args.num_query_points

    data_kwargs = dict(args.training.data_kwargs or {})
    if args.training.data == "gaussian":
        data_kwargs.setdefault("data_seed", args.training.seed)
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_kwargs)

    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        meta_bsize,
        num_tasks=args.training.num_tasks,
        out_dir=args.out_dir,
        is_save_task_pool=args.is_save_task_pool,
        **args.training.task_kwargs,
    )
    if resume_state is not None:
        restored_rng = restore_rng_state(resume_state, data_sampler)
        print(
            f"[checkpoint] resuming at step {starting_step}; "
            f"rng_restored={restored_rng}",
            flush=True,
        )


    pbar = tqdm(range(starting_step, args.training.train_steps))

    profile_meta = os.environ.get("PROFILE_META")
    profile_warmup, profile_active = 3, 3
    profile_total = profile_warmup + profile_active
    prof = None  # opened lazily after warmup steps below
    preemption = PreemptionHandler().install()
    compute_profile_active = int(args.training.compute_profile_steps)
    compute_profile_warmup = int(args.training.compute_profile_warmup_steps)
    if compute_profile_active < 0 or compute_profile_warmup < 0:
        raise ValueError("compute profiling step counts must be non-negative")
    compute_profile_start = starting_step + compute_profile_warmup
    compute_profile_end = compute_profile_start + compute_profile_active
    compute_profile_started_at = None
    profile_device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    skipped_updates = 0


    for i in pbar:
        if compute_profile_active and i == compute_profile_start:
            torch.cuda.synchronize()
            compute_profile_started_at = time.perf_counter()

        if profile_meta and i == profile_warmup and prof is None:
            torch.cuda.synchronize()
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            prof.__enter__()

        # Per-step total examples: support comes before a fresh query block.
        # In fixed-k mode, allocate enough support examples from the first step
        # instead of waiting for the curriculum's point count to catch up.
        support_capacity = max_support_size_for_step(
            meta_args,
            curriculum.n_points,
            curriculum.n_points_schedule.end,
        )
        n_total = max(curriculum.n_points, support_capacity) + n_query

        task_sampler_args = {}
        if "sparse_linear_regression" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated

        with torch.profiler.record_function("sample_data"):
            task = task_sampler(**task_sampler_args)
            xs = data_sampler.sample_xs(
                n_total,
                meta_bsize,
                curriculum.n_dims_truncated,
            )
            ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        with torch.profiler.record_function("to_cuda"):
            xs_cuda = xs.cuda()
            ys_cuda = ys.cuda()
            task_ids = getattr(task, "task_ids", None)
            task_ids_cuda = task_ids.cuda() if task_ids is not None else None

        with torch.profiler.record_function("meta_train_step"):
            (
                loss,
                per_task_k,
                per_task_q_loss,
                update_stats,
            ) = meta_train_step(
                model,
                optimizer,
                task,
                xs_cuda,
                ys_cuda,
                meta_args,
                curriculum.n_points,
                curriculum.n_points_schedule.end,
                loss_func,
                n_query,
                task_ids=task_ids_cuda,
                return_diagnostics=True,
            )

        if profile_meta and prof is not None and i + 1 == profile_total:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            print(
                prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=25
                )
            )
            trace_path = os.path.join(args.out_dir, "trace.json")
            prof.export_chrome_trace(trace_path)
            print(f"[profiler] chrome trace written to {trace_path}")
            sys.exit(0)
        skipped_updates += int(update_stats["outer_update_skipped"])
        if update_stats["outer_update_skipped"]:
            print(
                f"[outer-guard] skipped step {i + 1}: "
                f"preclip_norm={update_stats['outer_grad_norm']}, "
                f"total_skipped={skipped_updates}",
                flush=True,
            )
        if lr_scheduler is not None and not update_stats["outer_update_skipped"]:
            lr_scheduler.step()
        if compute_profile_active:
            curriculum.update()
            if preemption.requested:
                if not args.test_run:
                    save_training_state(
                        state_path, model, optimizer, i,
                        lr_scheduler=lr_scheduler, curriculum=curriculum,
                        data_sampler=data_sampler,
                    )
                preemption.exit_after_checkpoint()
            if i + 1 >= compute_profile_end:
                if compute_profile_started_at is None:
                    raise RuntimeError(
                        "compute profile ended without starting its timer"
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - compute_profile_started_at
                seconds_per_step = elapsed / compute_profile_active
                profile_payload = {
                    "compute_profile/seconds_per_step": seconds_per_step,
                    "compute_profile/warmup_steps": compute_profile_warmup,
                    "compute_profile/measured_steps": compute_profile_active,
                    "compute_profile/device": profile_device,
                }
                if not args.test_run:
                    wandb.log(profile_payload, step=i + 1)
                print(
                    f"[compute-profile] {seconds_per_step:.6f} seconds/step "
                    f"on {profile_device}",
                    flush=True,
                )
                preemption.restore()
                return

        baseline_loss = (
            sum(max(curriculum.n_dims_truncated - ii, 0) for ii in range(curriculum.n_points))
            / curriculum.n_points
        )

        # Log scalar training metrics
        if (
            (i == 0
             or (i > 0 and (i + 1) % args.wandb.log_every_steps == 0)
             or update_stats["outer_update_skipped"]
             or i + 1 == args.training.train_steps)
            and not args.test_run
        ):
            log_payload = {
                "meta_train/query_loss": loss,
                "meta_train/excess_loss": loss / baseline_loss if baseline_loss > 0 else float("nan"),
                "meta_train/mean_k": float(np.mean(per_task_k)),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
                "meta_train/outer_updates_skipped_total": skipped_updates,
                "tasks_seen": (i + 1) * meta_bsize,
            }
            log_payload.update(
                {
                    f"meta_train/{key}": value
                    for key, value in update_stats.items()
                    if value is not None
                }
            )

            log_payload.update(
                inner_lr_stats(
                    model,
                    meta_args,
                    include_layer=(i + 1) % 1000 == 0 or i + 1 == args.training.train_steps,
                    include_layer_module=(i + 1) % 5000 == 0 or i + 1 == args.training.train_steps,
                )
            )
            wandb.log(
                log_payload,
                step=i + 1,
            )

        # Periodic eval
        if (
            (i == 0
             or (i > 0 and (i + 1) % args.training.eval_every_steps == 0)
             or i + 1 == args.training.train_steps)
            and not args.test_run
        ):
            if meta_args.run_icl_style_eval:
                metrics = eval_model(
                    model,
                    task_name=args.training.task,
                    data_name=args.training.data,
                    n_dims=n_dims,
                    n_points=(args.training.eval_n_points or curriculum.n_points_schedule.end),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=args.training.task_kwargs,
                )
                point_wise_tags = list(range(len(metrics["mean"])))
                wandb.log(
                    {
                        "icl_eval/overall_loss": float(np.mean(metrics["mean"])),
                        "icl_eval/excess_loss": float(np.mean(metrics["mean"])) / baseline_loss if baseline_loss > 0 else float("nan"),
                        "icl_eval/pointwise/loss": dict(zip(point_wise_tags, list(metrics["mean"]))),
                    },
                    step=i + 1,
                )

        if (
            meta_args.run_maml_style_eval
            and (
                i == 0 or
                (i + 1) % meta_args.meta_eval_every_steps == 0
                or i + 1 == args.training.train_steps
            )
            and not args.test_run
        ):
            eval_batch_size = min(64, meta_args.meta_eval_num_tasks)
            eval_task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                eval_batch_size,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )
            eval_data_kwargs = dict(data_kwargs)
            if args.training.data == "gaussian":
                eval_data_kwargs["data_seed"] = meta_args.meta_eval_seed
            eval_data_sampler = get_data_sampler(
                args.training.data, n_dims=n_dims, **eval_data_kwargs
            )
            losses = maml_pointwise_eval(
                model,
                data_sampler=eval_data_sampler,
                task_sampler=eval_task_sampler,
                n_dims=n_dims,
                n_points=(args.training.eval_n_points or curriculum.n_points_schedule.end),
                inner_lr=meta_args.inner_lr,
                num_inner_steps=meta_args.num_inner_steps,
                stride=meta_args.meta_eval_stride,
                num_tasks=meta_args.meta_eval_num_tasks,
                eval_batch_size=eval_batch_size,
                inner_lrs=get_inner_lrs(model, meta_args),
                seed=meta_args.meta_eval_seed,
            )
            wandb_log_pointwise("maml_eval", losses, baseline_loss, step=i + 1)

        curriculum.update()
        pbar.set_description(f"meta_loss {loss:.4f}")

        one_indexed_steps = i + 1
        if (
            one_indexed_steps % args.training.save_every_steps == 0
            or one_indexed_steps == args.training.train_steps
        ) and not args.test_run:
            save_training_state(
                state_path, model, optimizer, i,
                lr_scheduler=lr_scheduler, curriculum=curriculum,
                data_sampler=data_sampler,
            )


        if (
            args.training.keep_every_steps > 0
            and (
                one_indexed_steps % args.training.keep_every_steps == 0
                or one_indexed_steps == args.training.train_steps
            )
            and not args.test_run
            and one_indexed_steps > 0
        ):
            atomic_torch_save(
                model.state_dict(),
                os.path.join(args.out_dir, f"model_{one_indexed_steps}.pt"),
            )
            save_training_state(
                os.path.join(args.out_dir, f"state_{one_indexed_steps}.pt"),
                model, optimizer, i,
                lr_scheduler=lr_scheduler, curriculum=curriculum,
                data_sampler=data_sampler,
            )
        if preemption.requested:
            if not args.test_run:
                save_training_state(
                    state_path, model, optimizer, i,
                    lr_scheduler=lr_scheduler, curriculum=curriculum,
                    data_sampler=data_sampler,
                )
            print(f"[checkpoint] saved step {one_indexed_steps}", flush=True)
            preemption.exit_after_checkpoint()
    preemption.restore()


def main(args):
    seed_everything(args.training.seed)
    if not args.meta.first_order and args.model.attn_implementation != "eager":
        raise ValueError(
            "Second-order MAML (meta.first_order: false) requires "
            "model.attn_implementation: eager. SDPA's flash/efficient backends "
            "do not implement double-backward. "
            f"Got first_order={args.meta.first_order}, "
            f"attn_implementation={args.model.attn_implementation!r}."
        )
    validate_meta_attention_backend(args)

    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 50
    else:
        wandb_entity = args.wandb.entity
        if wandb_entity in (None, ""):
            wandb_entity = None

        wandb_init_kwargs = dict(
            dir=args.out_dir,
            project=args.wandb.project,
            id=args.training.resume_id,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume="allow",
        )
        if wandb_entity is not None:
            wandb_init_kwargs["entity"] = wandb_entity

        wandb.init(**wandb_init_kwargs)

    model = build_model(args.model)
    configure_inner_lrs(model, args.meta)
    if args.model.load_model_path is not None:
        load_into_model_from_run(model, args.model.load_model_path)
    model.cuda()
    model.train()

    meta_train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=meta_training_schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2"], "Meta-learning only implemented for gpt2 family."
    print(f"Running meta-learning with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        args.training.resume_id = run_id
        apply_run_name(args)
        out_dir = os.path.join(args.out_dir, get_run_dir_name(args, run_id))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
