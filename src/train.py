import os
import sys
from munch import Munch
from random import randint
import uuid
import random
import itertools

from quinine import QuinineArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler, PolynomialsUnbiasedPoints, HaarWavelets
from samplers import get_data_sampler, sample_scale
from curriculum import Curriculum
from schema import schema
from models import build_model
from eval import eval_model, load_into_model_from_run
import pickle
from KS_monomial_sets import monomial_terms
from torch.distributions.multivariate_normal import MultivariateNormal
from transformers import get_scheduler
import copy
import wandb
import pdb
import time
from training_utils import compute_and_log_model_norm, filter_hold_out_freq, listify, equal_ignore_order

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, batch_idx, max_train_steps, k_steps_for_loss="all", num_accum_steps=1, lr_scheduler=None):
    # optimizer.zero_grad()
    output = model(xs, ys)
    if k_steps_for_loss == "all":
        loss = loss_func(output, ys)
    else:
        loss = loss_func(
            output[:, -int(k_steps_for_loss) :], ys[:, -int(k_steps_for_loss) :]
        )

    # normalize loss to account for batch accumulation
    loss = loss / num_accum_steps

    loss.backward()
    # optimizer.step()

    if ((batch_idx + 1) % num_accum_steps == 0) or (batch_idx + 1 == max_train_steps):
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def wandb_log_task(task, metrics_task, baseline_loss, point_wise_tags, step, suffix="", loss_scaling_factor=1.0):
    wandb.log(
        {
            f"{task}_eval{suffix}/overall_loss": np.mean(metrics_task["mean"]) * loss_scaling_factor,
            f"{task}_eval{suffix}/excess_loss": np.mean(metrics_task["mean"]) * loss_scaling_factor / baseline_loss,
            f"{task}_eval{suffix}/pointwise/loss": dict(
                zip(point_wise_tags, np.array(metrics_task["mean"]) * loss_scaling_factor)
            ),
        },
        step=step,
    )

def get_n_points_eval(task, n_dims, task_kwargs, curriculum):
    return curriculum.n_points_schedule.end
    # n_points_eval = 0
    # if 'polynomials_deg2_monomials_selection' in task:
    #     n_points_eval = curriculum.n_points_schedule.end
    # elif 'polynomials' == task or 'polynomials_unbiased_points' == task:
    #     # n_points_eval = 2 * task_kwargs.max_degree + 1
    #     n_points_eval = curriculum.n_points_schedule.end
    # elif "_cs" not in task:
    #     # n_points_eval = 2 * n_dims + 1
    #     # TODO: if we choose to log the inf-norm-optimization performance while training then this will break when n_points_eval > current value of input dimensions + 1
    #     # This is because inf-norm-optimization LPP is feasible only when n_points_eval <= current value of input dimension + 1
    #     # Remedy for this is to use a different n_points_eval value for each solver such that it is always feasible
    #     n_points_eval = curriculum.n_points_schedule.end
    # else:
    #     n_points_eval = curriculum.n_points_schedule.end
    # return n_points_eval

def get_training_optimizer(model, args):
    optimizer = None
    lr_scheduler = None
    if args.model.train_only_emb:
        # set requires_grad=False for all params
        for param in model.parameters():
            param.requires_grad = False

        # set requires_grad=True for model._read_in i.e. embedding layer
        for param in model._read_in.parameters():
            param.requires_grad = True

        # pass only the params with requires_grad=True to the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.training.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    if args.training.schedule is not None:
        assert args.training.schedule == "triangle", "Only triangle learning rate schedule is implemented."
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=args.training.train_steps,
        )
    return optimizer, lr_scheduler

def get_all_deg2_term_indices(n_dims):
    all_deg2_terms = []
    for i in range(n_dims):
        all_deg2_terms.append([i, i])
        for j in range(i+1,n_dims):
            all_deg2_terms.append([i, j])
    return all_deg2_terms

def validateTaskKwargs(args):
    taskName = args.training.task
    task_kwargs = args.training.task_kwargs
    if taskName == "polynomials_deg2_monomials_selection_unbiased":
        variant = task_kwargs["variant"]
        assert variant in ["fixedS", "fixedK", "randomS"], "invalid variant provided"
        if variant == "fixedS":
            assert len(task_kwargs["fixedS"]) == task_kwargs["numDeg2Select"], "Length of fixed S is different from number of monomial degree 2 terms to be selected"
            fixedS_array = np.array(task_kwargs["fixedS"])
            assert np.all((0 <= fixedS_array) & (fixedS_array <= args.model.n_dims-1)), "Some index in fixedS is out of bounds [0, n_dims-1]"
        elif variant == "fixedK":
            fixedK_array = np.array(task_kwargs["fixedK"])
            assert fixedK_array.shape == (task_kwargs["sizeOfK"], task_kwargs["numDeg2Select"], 2), "Shape of fixed K is different from (|K|, |S|, 2) as per config"
            assert np.all((0 <= fixedK_array) & (fixedK_array <= args.model.n_dims-1)), "For some S in fixedK, some index is out of bounds [0, n_dims-1]"
        elif variant == "randomS":
            assert task_kwargs["numDeg2Select"] < len(task_kwargs["all_deg2_terms"]), "|S| must be less than the number of degree 2 terms"

def train(model, args):
    optimizer, lr_scheduler = get_training_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"] + 1
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

    if args.training.granular_ckpt_till is not None:
        assert args.training.granular_ckpt_every is not None, "granular_ckpt_every cannot be None when granular_ckpt_till is not None."

    if args.training.schedule is not None:
        assert args.training.warmup_steps is not None, "warmup_steps cannot be None when learning rate schedule is not None."

    data_kwargs = args.training.data_kwargs
    if args.training.data == "gaussian":
        if data_kwargs is None:
            data_kwargs = {}
        data_kwargs.update({"scale": scale})

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_kwargs)

    excess_tensors = {}
    excess_tensors_eval = {}
    loss_scaling_factor = 1.0
    if args.training.task == "polynomials_unbiased_points":
        excess_tensors["tens"] = torch.empty(0, curriculum.n_points)
        excess_tensors_eval["tens"] = torch.empty(0, 2*args.training.task_kwargs.max_degree+1)
        excess_tensors["coefs"] = torch.empty(0, args.training.task_kwargs.max_degree+1)
        excess_tensors_eval["coefs"] = torch.empty(0, args.training.task_kwargs.max_degree+1)
    elif args.training.task == "polynomials_deg2_monomials_selection_unbiased":
        # make a list of all the deg 2 monomial term indices
        # for n_dims = 20, there are 20 + 190 terms: [[0, 0], [1, 1],...,[19, 19], [0, 1], [0, 2], ....[18, 19]]
        all_deg2_terms = get_all_deg2_term_indices(n_dims)
        args.training.task_kwargs["all_deg2_terms"] = all_deg2_terms
        variant = args.training.task_kwargs["variant"]
        if variant == "fixedK":
            numDeg2Select = args.training.task_kwargs["numDeg2Select"]
            sizeOfK = args.training.task_kwargs["sizeOfK"]
            args.training.task_kwargs["fixedK"] = torch.tensor(monomial_terms[f"{n_dims}-{sizeOfK}-{numDeg2Select}"], dtype=torch.int64)
    elif args.training.task == "gaussian_mixture_linear_regression":
        mean = torch.zeros(size=(n_dims,))
        mean[0] = args.training.task_kwargs["gaussian_centre_abs"]
        cov = torch.eye(n_dims)
        cov[0,0] = 1e-8
        distrib1 = MultivariateNormal(loc=mean, covariance_matrix=cov)
        distrib2 = MultivariateNormal(loc=-mean, covariance_matrix=cov)
        args.training.task_kwargs["distrib1"] = distrib1
        args.training.task_kwargs["distrib2"] = distrib2
    elif args.training.task == "haar_wavelets":
        # create a dummy object to access haar methods
        hw = HaarWavelets(n_dims=1, batch_size=4)
        max_level = args.training.task_kwargs["max_level"]
        # create the vectorized basis based on max_level of haar wavelets
        vectorized_basis = [np.vectorize(f) for f in hw.haar_basis(max_level=max_level)]
        args.training.task_kwargs["vectorized_basis"] = vectorized_basis
    elif args.training.task == "noisy_lr_task_diversity":
        loss_scaling_factor = (1/n_dims)
    elif args.training.task == "fourier_series_multitask":
        # make a list of all the subsets of fourier basis with freq in [1, max_frequency]
        # for max_frequency = 20 and sizeOfS=3, there are 20C3 subsets: [[1, 2, 3], [1, 2, 4], ...]
        sizeOfS = args.training.task_kwargs["sizeOfS"]
        max_frequency = args.training.task_kwargs["max_frequency"]
        training_S_list = list(map(set, itertools.combinations(list(range(1, max_frequency+1)), sizeOfS)))
        all_S_list = training_S_list
        variant = args.training.task_kwargs["variant"]

        if "hold_out_freq" in args.training.task_kwargs:
            # filter training_S_list to not contain any hold out freq
            hold_out_freq = args.training.task_kwargs["hold_out_freq"]
            training_S_list = filter_hold_out_freq(hold_out_freq, training_S_list)

        if variant == "fixedK":
            sizeOfK = args.training.task_kwargs["sizeOfK"]
            K_for_training = random.sample(training_S_list, sizeOfK)
            training_S_list = K_for_training

        OOD_S_list = [S for S in all_S_list if S not in training_S_list]
        assert equal_ignore_order(OOD_S_list + training_S_list, all_S_list), "OOD_S_list must be the difference of all_S_list and training_S_list"

        args.training.task_kwargs["training_S_list"] = listify(training_S_list)
        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

        args.training.task_kwargs["training_S_list"] = torch.tensor(args.training.task_kwargs["training_S_list"], dtype=torch.int64)
        args.training.task_kwargs["OOD_S_list"] = torch.tensor(listify(OOD_S_list), dtype=torch.int64)
        args.training.task_kwargs["all_S_list"] = torch.tensor(listify(all_S_list), dtype=torch.int64)
        if ("hold_out_freq" in args.training.task_kwargs) or variant == "fixedK":
            # log the S's used for training. When variant == "randomS" and we do not hold out any freq, then there is no point logging as every possible subset S is equally likely during training.
            wandb.config["fourier_multitask_training_subsets_S"] = listify(training_S_list)

    validateTaskKwargs(args)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        out_dir=args.out_dir,
        is_save_task_pool=args.is_save_task_pool,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))
    # log also when i+1 == args.training.train_steps

    num_training_examples = args.training.num_training_examples
    num_accum_steps = args.training.num_accum_steps
    optimizer.zero_grad()
    log_loss = 0.
    log_point_wise_loss = 0.
    # outputs_list = []
    for i in pbar:
        if (i % num_accum_steps == 0):
            log_loss = 0.
            log_point_wise_loss = torch.zeros(size=(curriculum.n_points,), dtype=torch.float32).cuda()

        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse_linear_regression" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        if "fourier_series" in args.training.task:
            args.training.task_kwargs["max_frequency"] = curriculum.max_freq
            task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )
        elif args.training.task == "random_fourier_features":
            args.training.task_kwargs["rff_dim"] = curriculum.rff_dim
            task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )
        # pdb.set_trace()
        task = task_sampler(**task_sampler_args)
        # curriculum.n_points = (
        #     task.get_bound() + 1 if "_cs" in args.training.task else curriculum.n_points
        # )
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        # import time
        # start = time.time()
        if isinstance(task, PolynomialsUnbiasedPoints):
            assert n_dims==1, "n_dims is not 1, please change sampling logic for ys s.t. it is from same distribution as xs but is of shape [batch, n_points]"
            # form the batch of ys w/ or w/o rejection sampling as needed            
            ys, _ = task.rejection_sample_to_form_batch(xs, data_sampler, data_sampler_args, bsize, curriculum.n_points, excess_tensors)
        else:
            ys = task.evaluate(xs)

        # end = time.time()
        # print("time",end - start)
        # outputs_list.append(ys)
        # fname='save_op.pkl'
        # if i % 1000 == 0:
        #     with open(fname, 'wb') as handle:
        #         pickle.dump(outputs_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        loss_func = task.get_training_metric()
        loss, output = train_step(
            model,
            xs.cuda(),
            ys.cuda(),
            optimizer,
            loss_func,
            batch_idx=i,
            max_train_steps=args.training.train_steps,
            k_steps_for_loss=args.training.k_steps_for_loss,
            num_accum_steps=num_accum_steps,
            lr_scheduler=lr_scheduler,
        )

        log_loss += loss
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)
        point_wise_loss = point_wise_loss/num_accum_steps
        log_point_wise_loss += point_wise_loss

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if (
            (i+1 == num_accum_steps or # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
            (i > 0 and (i+1) % args.wandb.log_every_steps == 0) or # log during training whenever we pass the logging interval
            i+1 == args.training.train_steps) # log at the last train step
            and not args.test_run
        ):
            wandb.log(
                {
                    "overall_loss": log_loss * loss_scaling_factor,
                    "excess_loss": log_loss * loss_scaling_factor / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, log_point_wise_loss.cpu().numpy() * loss_scaling_factor)
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "max_freq": curriculum.max_freq,
                    "rff_dim": curriculum.rff_dim,
                },
                step=(i+1)//num_accum_steps,
            )

        if (
            (i+1 == num_accum_steps or # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
            (i > 0 and (i+1) % args.training.eval_every_steps == 0) or # log during training whenever we pass the logging interval
            i+1 == args.training.train_steps) # log at the last train step
            and not args.test_run
        ):
            if args.training.task == "two_task_mixer":
                task1 = args.training.task_kwargs.get("task1", "linear_regression")
                task2 = args.training.task_kwargs.get("task2", "decision_tree")
                task_spec_params_dict = args.training.task_kwargs.get(
                    "task_spec_params_dict", {}
                )
                task1_kwargs = task_spec_params_dict.get(task1, {})
                task2_kwargs = task_spec_params_dict.get(task2, {})

                metrics_task1 = eval_model(
                    model,
                    task_name=task1,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(task1, args.model.n_dims, task1_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task1_kwargs,
                )

                metrics_task2 = eval_model(
                    model,
                    task_name=task2,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(task2, args.model.n_dims, task2_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task2_kwargs,
                )
                wandb_log_task(task1, metrics_task1, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps)
                wandb_log_task(task2, metrics_task2, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps)

            elif args.training.task == "three_task_mixer":
                task1 = args.training.task_kwargs.get("task1", "linear_regression")
                task2 = args.training.task_kwargs.get("task2", "decision_tree")
                task3 = args.training.task_kwargs.get("task3", "relu_2nn_regression")

                task_spec_params_dict = args.training.task_kwargs.get(
                    "task_spec_params_dict", {}
                )
                task1_kwargs = task_spec_params_dict.get(task1, {})
                task2_kwargs = task_spec_params_dict.get(task2, {})
                task3_kwargs = task_spec_params_dict.get(task3, {})

                metrics_task1 = eval_model(
                    model,
                    task_name=task1,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(task1, args.model.n_dims, task1_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task1_kwargs,
                )

                metrics_task2 = eval_model(
                    model,
                    task_name=task2,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(task2, args.model.n_dims, task2_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task2_kwargs,
                )

                metrics_task3 = eval_model(
                    model,
                    task_name=task3,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(task3, args.model.n_dims, task3_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task3_kwargs,
                )

                wandb_log_task(task1, metrics_task1, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps)
                wandb_log_task(task2, metrics_task2, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps)
                wandb_log_task(task3, metrics_task3, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps)

            else:
                n_dims = args.model.n_dims
                eval_task_sampler_kwargs = args.training.task_kwargs
                eval_data_kwargs = data_kwargs
                if args.training.task == "noisy_lr_task_diversity":
                    eval_task_sampler_kwargs = copy.deepcopy(args.training.task_kwargs)
                    eval_task_sampler_kwargs["pool_dict"] = task.pool_dict
                    eval_data_kwargs = copy.deepcopy(data_kwargs)
                    eval_data_kwargs["data_seed"] = int(eval_data_kwargs["data_seed"] + time.time())
                metrics = eval_model(
                    model,
                    task_name=args.training.task,
                    data_name=args.training.data,
                    n_dims=args.model.n_dims,
                    n_points=get_n_points_eval(args.training.task, args.model.n_dims, args.training.task_kwargs, curriculum),
                    prompting_strategy="standard",
                    batch_size=64,
                    data_sampler_kwargs=eval_data_kwargs,
                    task_sampler_kwargs=eval_task_sampler_kwargs,
                    excess_tensors_eval=excess_tensors_eval
                )

                wandb_log_task(args.training.task, metrics, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps, loss_scaling_factor=loss_scaling_factor)

                if args.training.eval_ood:
                    assert args.training.task in ["polynomials_deg2_monomials_selection_unbiased", "noisy_lr_task_diversity", "fourier_series_multitask"], "task is not in the list of tasks for OOD evaluation"
                    eval_task_sampler_kwargs = args.training.task_kwargs
                    eval_data_kwargs = data_kwargs
                    if args.training.task == "noisy_lr_task_diversity":
                        # eval_task_sampler_kwargs = copy.deepcopy(args.training.task_kwargs)
                        # eval_task_sampler_kwargs["pool_dict"] = task.pool_dict
                        eval_data_kwargs = copy.deepcopy(data_kwargs)
                        eval_data_kwargs["data_seed"] = int(eval_data_kwargs["data_seed"] + time.time())
                    metrics_ood = eval_model(
                        model,
                        task_name=args.training.task,
                        data_name=args.training.data,
                        n_dims=args.model.n_dims,
                        n_points=get_n_points_eval(args.training.task, args.model.n_dims, args.training.task_kwargs, curriculum),
                        prompting_strategy="standard",
                        batch_size=64,
                        data_sampler_kwargs=eval_data_kwargs,
                        task_sampler_kwargs=eval_task_sampler_kwargs,
                        excess_tensors_eval=excess_tensors_eval,
                        eval_ood=True
                    )
                    wandb_log_task(args.training.task, metrics_ood, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps, suffix="_ood", loss_scaling_factor=loss_scaling_factor)

                if args.training.log_model_norm:
                    compute_and_log_model_norm(model, step=(i+1)//num_accum_steps)

        curriculum.update()

        one_indexed_steps = i + 1
        pbar.set_description(f"loss {loss * loss_scaling_factor}")
        if (one_indexed_steps % args.training.save_every_steps == 0  or one_indexed_steps == args.training.train_steps) and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i, # 0-indexed because this is used while resuming the training
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and ((one_indexed_steps % args.training.keep_every_steps == 0 or one_indexed_steps == args.training.train_steps)
            or (args.training.granular_ckpt_till is not None and one_indexed_steps <= args.training.granular_ckpt_till and one_indexed_steps % args.training.granular_ckpt_every == 0))
            and not args.test_run
            and one_indexed_steps > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{one_indexed_steps}.pt"))


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
        run_path = os.path.join(args.model.load_model_path)
        load_into_model_from_run(model, run_path)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval

def updateStepsByGradAccumSteps(args):
    num_accum_steps = args.training.num_accum_steps
    args.training.train_steps *= num_accum_steps
    args.training.eval_every_steps *= num_accum_steps
    args.training.save_every_steps *= num_accum_steps
    args.training.keep_every_steps *= num_accum_steps

    args.training.curriculum.dims.interval *= num_accum_steps
    args.training.curriculum.points.interval *= num_accum_steps

    args.wandb.log_every_steps *= num_accum_steps

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    updateStepsByGradAccumSteps(args)
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
