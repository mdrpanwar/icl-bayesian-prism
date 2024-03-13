import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler, PolynomialsUnbiasedPoints
from samplers import get_data_sampler, sample_scale
from curriculum import Curriculum
from schema import schema
from models import build_model
from eval import eval_model, load_into_model_from_run
from train import (
    get_training_optimizer,
    get_n_points_eval,
    wandb_log_task,
    sample_seeds,
)
import wandb
import pdb

torch.backends.cudnn.benchmark = True


def train_step(
    model, xs, ys, task_labels, optimizer, loss_func, k_steps_for_loss="all"
):
    optimizer.zero_grad()
    reg_prefix = torch.zeros(xs.shape[0]).to(xs.device).long()
    cls_prefix = torch.ones(xs.shape[0]).to(xs.device).long()

    reg_preds = model(xs, ys, reg_prefix)
    cls_preds = model(xs, ys, cls_prefix)
    if k_steps_for_loss == "all":
        reg_loss = loss_func(reg_preds, ys)
        # cls_loss = loss_func(cls_preds, task_labels)
    else:
        reg_loss = loss_func(
            reg_preds[:, -int(k_steps_for_loss) :], ys[:, -int(k_steps_for_loss) :]
        )
    cls_loss = loss_func(cls_preds, task_labels.unsqueeze(1))
    loss = reg_loss + cls_loss

    loss.backward()
    optimizer.step()
    return (
        loss.detach().item(),
        reg_preds.detach(),
        cls_preds.detach(),
        reg_loss.detach().item(),
        cls_loss.detach().item(),
    )


def train(model, args):
    optimizer = get_training_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")

    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    if args.training.data_transformation_args is not None:
        scale = sample_scale(
            method=args.training.data_transformation_args.get("method", None),
            n_dims=n_dims,
            normalize=args.training.data_transformation_args.get("normalize", False),
            seed=args.training.data_transformation_args.get("seed", None),
        )
    else:
        scale = None

    data_kwargs = args.training.data_kwargs
    if args.training.data == "gaussian":
        if data_kwargs is None:
            data_kwargs = {}
        data_kwargs.update({"scale": scale})

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_kwargs)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        # pdb.set_trace()
        task = task_sampler(**task_sampler_args)
        curriculum.n_points = (
            task.get_bound() + 1 if "_cs" in args.training.task else curriculum.n_points
        )
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        ys = task.evaluate(xs)
        task_labels = task.get_task_label(xs)
        loss_func = task.get_training_metric()
        loss, reg_output, cls_output, reg_loss, cls_loss = train_step(
            model,
            xs.cuda(),
            ys.cuda(),
            task_labels.cuda(),
            optimizer,
            loss_func,
            k_steps_for_loss=args.training.k_steps_for_loss,
        )

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(reg_output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "reg_loss": reg_loss,
                    "excess_loss": reg_loss / baseline_loss,
                    "cls_loss": cls_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )
        if i % args.training.eval_every_steps == 0 and not args.test_run:
            num_tasks = 3 if args.training.task == "three_task_mixer" else 2
            task_names = [f"task{i+1}" for i in range(num_tasks)]
            tasks = [
                args.training.task_kwargs.get(task_name) for task_name in task_names
            ]
            task_spec_params_dict = args.training.task_kwargs.get(
                "task_spec_params_dict", {}
            )
            tasks_kwargs = [task_spec_params_dict.get(task_i, {}) for task_i in tasks]

            for task_name, task_kwargs in zip(tasks, tasks_kwargs):
                metrics = eval_model(
                    model,
                    task_name=task_name,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(
                        task_name, args.model.n_dims, task_kwargs, curriculum
                    ),
                    prompting_strategy="standard",
                    batch_size=args.training.batch_size,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task_kwargs,
                )
                wandb_log_task(
                    task_name, metrics, baseline_loss, point_wise_tags, step=i
                )
        curriculum.update()
        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    if args.model.load_model_path is not None:
        run_path = os.path.join(
            "../models", args.training.task, args.model.load_model_path
        )
        load_into_model_from_run(model, run_path)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "gpt2_task_prefix"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
