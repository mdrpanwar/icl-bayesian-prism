import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from models import NNModel, LeastSquaresModel
from eval import read_run_dir, get_model_from_run
from tasks import Polynomials, PolynomialsFactorForm, FourierSeriesV2
from samplers import get_data_sampler
from tasks import get_task_sampler

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")
mpl.rcParams["figure.dpi"] = 300

run_dir = "../models"


class FourierFeatures:
    def __init__(self, max_freq, L):
        self.max_freq = max_freq
        self.L = L

    def transform(self, x):
        # Currently on supports 1-d features
        assert x.shape[-1] == 1
        batch_size = x.shape[0]
        sine_features = torch.sin(
            (math.pi / (self.L))
            * torch.arange(1, self.max_freq + 1).unsqueeze(0)
            * x.view(-1, 1)
        ).view(batch_size, -1, self.max_freq)
        cosine_features = torch.cos(
            (math.pi / (self.L))
            * torch.arange(self.max_freq + 1).unsqueeze(0)
            * x.view(-1, 1)
        ).view(batch_size, -1, self.max_freq + 1)

        return torch.cat([sine_features, cosine_features], dim=-1)


def main():
    task = "fourier_series"
    run_id = int(sys.argv[1])
    max_freq = int(sys.argv[2])
    mode = str(sys.argv[3])
    model_type = str(sys.argv[4])
    curriculum = (sys.argv[5]) == "True"
    method = "DFT"  # str(sys.argv[6])
    pure_freq = False  # (sys.argv[7]) == "True"
    min_freq = max_freq if pure_freq else 1

    if model_type == "Transformer":
        if curriculum:
            model_type = f"{model_type} w Curriculum"

    save_path = f"fourier_animations/function_fits/v2/{mode}/{model_type}"
    if pure_freq:
        save_path = f"{save_path}/PureFrequency"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    run_path = os.path.join(run_dir, task, run_id)
    model, conf = get_model_from_run(run_path)

    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    data_sampler = get_data_sampler(
        conf.training.data, n_dims, **conf.training.data_kwargs
    )
    task_sampler = get_task_sampler(
        conf.training.task, n_dims, batch_size, **conf.training.task_kwargs
    )
    task = task_sampler()
    xs = data_sampler.sample_xs(
        b_size=batch_size, n_points=conf.training.curriculum.points.end
    )
    ys = task.evaluate(xs)

    torch.manual_seed(seed)
    L = 5
    prompt_len = 2
    n = 40
    task = FourierSeriesV2(
        1, 1, max_frequency=max_freq, min_frequency=min_freq, standardize=True, L=L
    )
    xs = data_sampler.sample_xs(b_size=1, n_points=n)
    ys = task.evaluate(xs)

    y_preds = []
    x_cont = torch.linspace(-L, L, n).unsqueeze(0).unsqueeze(-1)
    y_cont = task.evaluate(x_cont)

    y_preds_prefixes = []
    print("Getting Prefix Resuts")
    fig, axs = plt.subplots(1, 21, figsize=(100, 5))

    for pl_idx, pl in enumerate(tqdm([0.5] + list(range(1, 21)))):
        if mode == "interpolate":
            x_prefix = xs[:, : int(2 * pl)]
            y_prefix = ys[:, : int(2 * pl)]
        else:
            x_prefix = x_cont[:, : int(2 * pl)]
            y_prefix = y_cont[:, : int(2 * pl)]

        x_prompt = torch.cat(
            [x_prefix.repeat(x_cont.size(1), 1, 1), x_cont.transpose(1, 0)], axis=1
        )
        y_prompt = torch.cat(
            [y_prefix.repeat(y_cont.size(1), 1), y_cont.transpose(1, 0)], axis=1
        )

        if "Transformer" in model_type:
            with torch.no_grad():
                if pl != 0:
                    y_preds = model(x_prompt, y_prompt).squeeze()[:, -1]
                else:
                    y_preds = model(x_prompt, y_prompt).squeeze()

            y_preds = y_preds.detach().cpu().numpy()

        elif model_type == "Fourier LSQ":
            lsq_model = LinearRegression(fit_intercept=False)
            x_fourier_prefix = (
                FourierFeatures(
                    max_freq=conf.training.task_kwargs["max_frequency"], L=L
                )
                .transform(x_prefix)
                .squeeze(0)
                .numpy()
            )
            x_fourier_cont = (
                FourierFeatures(
                    max_freq=conf.training.task_kwargs["max_frequency"], L=L
                )
                .transform(x_cont)
                .squeeze(0)
                .numpy()
            )
            lsq_model.fit(x_fourier_prefix, y_prefix.numpy().squeeze(0))
            y_preds = lsq_model.predict(x_fourier_cont)

        else:
            raise NotImplementedError()

        sns.lineplot(
            x_cont.squeeze(), y_cont.squeeze(), label="Ground Truth", ax=axs[pl_idx]
        )
        sns.lineplot(
            x_cont.squeeze(), y_preds, label="Predicted Function", ax=axs[pl_idx]
        )
        if pl != 0:
            sns.scatterplot(
                x_prefix.squeeze(0).squeeze(-1),
                y_prefix.squeeze(0),
                label="Prompt",
                ax=axs[pl_idx],
            )
        axs[pl_idx].set_title(f"Prompt Length: {int(2*pl)}")

    plt.suptitle(f"Gold Frequency: {max_freq}\nModel: {model_type}")
    fig.tight_layout()
    plt.savefig(
        f"{save_path}/{model_type}_gold_freq_{max_freq}_seed{seed}.png", dpi=300
    )


if __name__ == "__main__":
    main()
