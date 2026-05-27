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
from eval import eval_model, load_into_model_from_run, get_run_metrics


torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


def inner_adapt(
    model,
    init_params,
    xs_support,
    ys_support,
    inner_lr,
    num_inner_steps,
    first_order,
    loss_func,
):
    """Run num_inner_steps SGD steps on the support loss, returning fast weights.

    init_params: dict of {name: tensor}, possibly the live nn.Module params
        (when full MAML) or detached copies (when first-order).
    """
    fast_params = init_params
    for _ in range(num_inner_steps):
        preds = functional_call(model, fast_params, (xs_support, ys_support))
        loss_s = loss_func(preds, ys_support)
        grads = torch.autograd.grad(
            loss_s,
            list(fast_params.values()),
            create_graph=not first_order,
            allow_unused=True,
        )
        # unused = [n for (n, _), g in zip(fast_params.items(), grads) if g is None]
        # print(f"[inner_adapt] params with grad=None: {unused}")
        # breakpoint()
        
        # Some GPT2 params (wte, wpe under pos_encode=False) never enter the
        # graph; their grad is None and they should be left untouched.
        fast_params = {
            name: p if g is None else p - inner_lr * g
            for (name, p), g in zip(fast_params.items(), grads)
        }
    return fast_params


# ---------------------------------------------------------------------------
# Support-size sampling
# ---------------------------------------------------------------------------


def sample_support_size(meta_args, curriculum_n_points, curriculum_end):
    mode = meta_args.vary_support_size
    if mode == "match_curriculum":
        return random.randint(0, curriculum_n_points - 1)
    if mode == "full_range":
        return random.randint(0, curriculum_end - 1)
    if mode == "fixed":
        k = meta_args.fixed_support_size
        if k is None:
            k = curriculum_end - 1
        return k
    raise ValueError(f"Unknown vary_support_size: {mode}")


# ---------------------------------------------------------------------------
# Outer training step
# ---------------------------------------------------------------------------


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
):
    """Vectorized outer step. Same per-task semantics as `meta_train_step_loop`:
    each task b independently samples k_b in [0, curriculum_n_points-1],
    adapts on its first k_b examples, and is evaluated on its own n_query
    length-1 query prompts.

    The meta-batch is split into the k_b > 0 group (one vmap'd
    adapt-then-query call with right-padding to k_max + a per-task loss
    mask) and the k_b == 0 group (one bulk forward, no inner step). GPT-2's
    causal attention naturally ignores right-padded positions, so no
    attention mask is needed -- only a loss mask on the support side.

    `loss_func` is currently unused: support and query both use plain MSE,
    matching the existing tasks (linear regression, etc.). When non-MSE
    meta-tasks are added, generalize the masked support loss below.
    """
    optimizer.zero_grad()
    B = xs.shape[0]
    n_total = xs.shape[1]
    n_dims = xs.shape[2]

    # Trainable params only. Frozen wpe.weight is excluded; functional_call
    # falls back to the model's stored value (zero) for any name not in dict.
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # Each task's queries become n_query independent length-1 prompts.
    xs_q = xs[:, n_total - n_query : n_total, :].reshape(B, n_query, 1, n_dims)
    ys_q = ys[:, n_total - n_query : n_total].reshape(B, n_query, 1)

    ks = [
        sample_support_size(meta_args, curriculum_n_points, curriculum_end)
        for _ in range(B)
    ]
    ks_t = torch.tensor(ks, device=xs.device)
    has_support = (ks_t > 0).nonzero(as_tuple=True)[0]
    no_support = (ks_t == 0).nonzero(as_tuple=True)[0]

    per_task_losses_log = torch.zeros(B, device=xs.device)
    total_loss = xs.new_zeros(())

    # ---- k_b > 0 tasks: vmap'd adapt-then-query ----
    if has_support.numel() > 0:
        ks_pos = ks_t[has_support]
        k_max = int(ks_pos.max().item())

        xs_s = xs[has_support, :k_max, :]
        ys_s = ys[has_support, :k_max]
        positions = torch.arange(k_max, device=xs.device).unsqueeze(0)
        mask = (positions < ks_pos.unsqueeze(1)).to(xs.dtype)
        xs_q_p = xs_q[has_support]
        ys_q_p = ys_q[has_support]

        def support_loss(p, xs_s_b, ys_s_b, mask_b):
            preds = functional_call(
                model, p, (xs_s_b.unsqueeze(0), ys_s_b.unsqueeze(0))
            ).squeeze(0)
            sq = (preds - ys_s_b) ** 2
            return (sq * mask_b).sum() / mask_b.sum().clamp(min=1.0)

        def adapt_then_query(p, xs_s_b, ys_s_b, mask_b, xs_q_b, ys_q_b):
            fp = p
            for _ in range(meta_args.num_inner_steps):
                g = func_grad(support_loss)(fp, xs_s_b, ys_s_b, mask_b)
                if meta_args.first_order:
                    g = {n: gv.detach() for n, gv in g.items()}
                fp = {n: pv - meta_args.inner_lr * g[n] for n, pv in fp.items()}
            preds_q = functional_call(model, fp, (xs_q_b, ys_q_b))
            return ((preds_q - ys_q_b) ** 2).mean()

        per_task_pos = vmap(
            adapt_then_query, in_dims=(None, 0, 0, 0, 0, 0)
        )(params, xs_s, ys_s, mask, xs_q_p, ys_q_p)
        total_loss = total_loss + per_task_pos.sum()

        with torch.no_grad():
            per_task_losses_log[has_support] = per_task_pos.detach()

    # ---- k_b == 0 tasks: no inner step, one batched forward ----
    if no_support.numel() > 0:
        Bz = no_support.numel()
        flat_x = xs_q[no_support].reshape(Bz * n_query, 1, n_dims)
        flat_y = ys_q[no_support].reshape(Bz * n_query, 1)
        preds_q = functional_call(model, params, (flat_x, flat_y))
        per_task_zero = ((preds_q - flat_y) ** 2).reshape(Bz, n_query).mean(dim=1)
        total_loss = total_loss + per_task_zero.sum()

        with torch.no_grad():
            per_task_losses_log[no_support] = per_task_zero.detach()

    meta_loss = total_loss / B
    meta_loss.backward()
    optimizer.step()

    return meta_loss.item(), ks, per_task_losses_log.tolist()


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

    # Skip frozen params (wpe.weight when pos_encode=False); functional_call
    # falls back to the model's stored values for any name absent from the dict.
    base_params = {
        name: p for name, p in model.named_parameters() if p.requires_grad
    }

    total_query_loss = 0.0
    per_task_query_losses = []
    per_task_k = []

    for b in range(meta_batch_size):
        k_b = sample_support_size(meta_args, curriculum_n_points, curriculum_end)
        per_task_k.append(k_b)

        # Support: positions [0, k_b) of task b's prompt
        xs_s = xs[b : b + 1, :k_b, :]
        ys_s = ys[b : b + 1, :k_b]

        # Inner adaptation
        if meta_args.first_order:
            init_params = {name: p.detach().requires_grad_(True) for name, p in base_params.items()}
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
                meta_args.inner_lr,
                meta_args.num_inner_steps,
                meta_args.first_order,
                loss_func,
            )

        # Query: m fresh (x, y) pairs from the same task, batched on the batch
        # dim as length-1 prompts so each is a pure k-shot measurement.
        xs_q = xs[b, n_total - n_query : n_total, :].reshape(n_query, 1, n_dims)
        ys_q = ys[b, n_total - n_query : n_total].reshape(n_query, 1)

        preds_q = functional_call(model, fast_params, (xs_q, ys_q))
        loss_q = ((preds_q - ys_q) ** 2).mean() / meta_batch_size

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

        total_query_loss += loss_q.detach().item() * meta_batch_size
        per_task_query_losses.append(loss_q.detach().item() * meta_batch_size)

    optimizer.step()
    mean_query_loss = total_query_loss / meta_batch_size
    return mean_query_loss, per_task_k, per_task_query_losses


# ---------------------------------------------------------------------------
# MAML-style pointwise eval
# ---------------------------------------------------------------------------


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
        for name, p in model.named_parameters()
        if p.requires_grad
    }

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
        task = task_sampler()
        xs = data_sampler.sample_xs(n_points, eval_batch_size).to(device)
        ys = task.evaluate(xs).to(device)

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
                        inner_lr,
                        num_inner_steps,
                        first_order=True,
                        loss_func=lambda pred, target: (
                            (pred - target) ** 2
                        ).mean(),
                    )

                xs_q = xs[b : b + 1, k : k + 1, :]
                ys_q = ys[b : b + 1, k : k + 1]
                with torch.no_grad():
                    pred = functional_call(model, fast_params, (xs_q, ys_q))
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
    optimizer, lr_scheduler = get_outer_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"] + 1
        for _ in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    meta_args = args.meta
    meta_bsize = meta_args.meta_batch_size
    n_query = meta_args.num_query_points

    data_kwargs = args.training.data_kwargs or {}
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

    pbar = tqdm(range(starting_step, args.training.train_steps))

    profile_meta = os.environ.get("PROFILE_META")
    profile_warmup, profile_active = 3, 3
    profile_total = profile_warmup + profile_active
    prof = None  # opened lazily after warmup steps below

    for i in pbar:
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

        # Per-step total examples: support uses up to curriculum.n_points-1 of
        # the first n_points positions; queries are appended after that.
        n_total = curriculum.n_points + n_query

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

        with torch.profiler.record_function("meta_train_step"):
            loss, per_task_k, per_task_q_loss = meta_train_step(
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
        if lr_scheduler is not None:
            lr_scheduler.step()

        baseline_loss = (
            sum(max(curriculum.n_dims_truncated - ii, 0) for ii in range(curriculum.n_points))
            / curriculum.n_points
        )

        # Log scalar training metrics
        if (
            (i == 0
             or (i > 0 and (i + 1) % args.wandb.log_every_steps == 0)
             or i + 1 == args.training.train_steps)
            and not args.test_run
        ):
            wandb.log(
                {
                    "meta_train/query_loss": loss,
                    "meta_train/excess_loss": loss / baseline_loss if baseline_loss > 0 else float("nan"),
                    "meta_train/mean_k": float(np.mean(per_task_k)),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
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
                    n_points=curriculum.n_points_schedule.end,
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
            losses = maml_pointwise_eval(
                model,
                data_sampler=data_sampler,
                task_sampler=eval_task_sampler,
                n_dims=n_dims,
                n_points=curriculum.n_points_schedule.end,
                inner_lr=meta_args.inner_lr,
                num_inner_steps=meta_args.num_inner_steps,
                stride=meta_args.meta_eval_stride,
                num_tasks=meta_args.meta_eval_num_tasks,
                eval_batch_size=eval_batch_size,
            )
            wandb_log_pointwise("maml_eval", losses, baseline_loss, step=i + 1)

        curriculum.update()
        pbar.set_description(f"meta_loss {loss:.4f}")

        one_indexed_steps = i + 1
        if (
            one_indexed_steps % args.training.save_every_steps == 0
            or one_indexed_steps == args.training.train_steps
        ) and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and (
                one_indexed_steps % args.training.keep_every_steps == 0
                or one_indexed_steps == args.training.train_steps
            )
            and not args.test_run
            and one_indexed_steps > 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, f"model_{one_indexed_steps}.pt"),
            )


def main(args):
    if not args.meta.first_order and args.model.attn_implementation != "eager":
        raise ValueError(
            "Second-order MAML (meta.first_order: false) requires "
            "model.attn_implementation: eager. SDPA's flash/efficient backends "
            "do not implement double-backward. "
            f"Got first_order={args.meta.first_order}, "
            f"attn_implementation={args.model.attn_implementation!r}."
        )

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
        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
