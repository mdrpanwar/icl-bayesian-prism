import os
from collections import OrderedDict
import glob
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run, eval_model
from samplers import get_data_sampler
from tasks import get_task_sampler

run_dir = "models"


def eval_run(run_path, step, task):
    model, conf = get_model_from_run(run_path, step=step)
    model.to("cuda")
    metrics = eval_model(
        model,
        task,
        data_name=conf.training.data,
        n_dims=conf.model.n_dims,
        n_points=2 * conf.model.n_dims,
        prompting_strategy="standard",
        batch_size=conf.training.batch_size,
        task_sampler_kwargs=conf.training.task_kwargs,
    )
    return metrics


def eval_runs(run_id, task):
    run_path = os.path.join(run_dir, task, run_id)
    run_steps = sorted(
        [
            int(filename.split("/")[-1].split("_")[-1].split(".")[0])
            for filename in glob.glob(f"{run_path}/model_*.pt")
        ]
    )
    print(run_steps)
    for step in run_steps:
        print(f"Running for {step}th step")
        metrics = eval_run(run_path, step, task)["mean"]
        plt.plot(metrics, lw=2, label=f"step={step}")
        # print(metrics[21])
        # print("*" * 25)
        # print()
    plt.yscale("log")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig(f"{run_path}/stepwise_error_trends.pdf")
    plt.savefig(f"{run_path}/stepwise_error_trends.png")


def main():
    task = "linear_regression"
    run_id = "4b90cf8a-1f0c-4246-b156-6f50efa4441e"
    eval_runs(run_id, task)


if __name__ == "__main__":
    main()
