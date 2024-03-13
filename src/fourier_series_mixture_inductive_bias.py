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
from IPython.display import HTML
from sklearn.linear_model import LinearRegression, LassoCV
from models import NNModel, LeastSquaresModel
from eval import read_run_dir, get_model_from_run
from tasks import Polynomials, PolynomialsFactorForm, FourierSeries
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
            * torch.arange(1, self.max_freq + 1).unsqueeze(0)
            * x.view(-1, 1)
        ).view(batch_size, -1, self.max_freq)

        return torch.cat([sine_features, cosine_features], dim=-1)


def main():
    task = "fourier_series"
    run_id = int(sys.argv[1])
    max_freq = int(sys.argv[2])
    mode = str(sys.argv[3])
    model_type = str(sys.argv[4])
    method = "DFT"  # str(sys.argv[6])
    pure_freq = False  # (sys.argv[7]) == "True"
    min_freq = max_freq if pure_freq else 1
    save_path = (
        f"fourier_animations/mixture/inductive_bias/v2/{mode}/{method}/{model_type}"
    )

    if pure_freq:
        save_path = f"{save_path}/PureFrequency"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_path = os.path.join(run_dir, task, run_id)

    model, conf = get_model_from_run(run_path)

    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size
    L = conf.training.task_kwargs["L"]
    n = 40

    data_sampler = get_data_sampler(
        conf.training.data, n_dims, **conf.training.data_kwargs
    )
    y_preds = []
    x_cont = torch.linspace(-L, L, n).unsqueeze(0).unsqueeze(-1)
    y_preds_prefixes = []
    fig, axs = plt.subplots(1, 21, figsize=(100, 5))

    for pl_idx, pl in enumerate(tqdm([0.5] + list(range(1, 21)))):
        freq_dist = []
        seed_ls = []
        freqs_ls = []
        for seed in tqdm([1, 11, 22, 33, 44, 55, 66, 77, 88, 99]):
            torch.manual_seed(seed)
            task = FourierSeries(
                1,
                1,
                max_frequency=max_freq,
                min_frequency=min_freq,
                standardize=True,
                L=L,
            )
            xs = data_sampler.sample_xs(b_size=1, n_points=n)
            ys = task.evaluate(xs)
            y_cont = task.evaluate(x_cont)

            if mode == "interpolate":
                x_prefix = xs[:, : int(2 * pl)]
                y_prefix = ys[:, : int(2 * pl)]
            else:
                x_prefix = x_cont[:, : int(2 * pl)]
                y_prefix = y_cont[:, : int(2 * pl)]
            y_preds = []

            x_prompt = torch.cat(
                [x_prefix.repeat(x_cont.size(1), 1, 1), x_cont.transpose(1, 0)], axis=1
            )
            y_prompt = torch.cat(
                [y_prefix.repeat(y_cont.size(1), 1), y_cont.transpose(1, 0)], axis=1
            )

            if "Transformer" in model_type:
                with torch.no_grad():
                    breakpoint()
                    if pl != 0:
                        y_preds = model(x_prompt, y_prompt).squeeze()[:, -1]
                    else:
                        y_preds = model(x_prompt, y_prompt).squeeze()

                y_preds = y_preds.detach().cpu().numpy()

            elif model_type == "Fourier LSQ-Max":
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

            elif model_type == "Fourier LSQ-Gold":
                lsq_model = LinearRegression(fit_intercept=False)
                x_fourier_prefix = (
                    FourierFeatures(max_freq=max_freq, L=L)
                    .transform(x_prefix)
                    .squeeze(0)
                    .numpy()
                )
                x_fourier_cont = (
                    FourierFeatures(max_freq=max_freq, L=L)
                    .transform(x_cont)
                    .squeeze(0)
                    .numpy()
                )
                lsq_model.fit(x_fourier_prefix, y_prefix.numpy().squeeze(0))
                y_preds = lsq_model.predict(x_fourier_cont)

            else:
                raise NotImplementedError()

            if method == "FourierFit":
                fourier_fit_model = LinearRegression()
                x_fourier = torch.Tensor(
                    FourierFeatures(max_freq=n // 2 - 1, L=L).transform(x_cont)
                ).squeeze()
                y_fourier = y_preds.squeeze()
                fourier_fit_model = fourier_fit_model.fit(x_fourier, y_fourier)
                fit_score = fourier_fit_model.score(x_fourier, y_fourier)

                weights = fourier_fit_model.coef_
                coefs = weights.squeeze()
                a_coefs, b_coefs = coefs[: len(coefs) // 2], coefs[len(coefs) // 2 :]

            else:
                y_fourier = y_preds.squeeze()
                ft = np.fft.fft(y_fourier)
                ft_shifted = np.fft.fftshift(ft)
                ft_normalized = ft_shifted / len(y_fourier)
                ft_positive = ft_normalized[1 + len(y_fourier) // 2 :]

                a_coefs = ft_positive.real
                b_coefs = ft_positive.imag

                fit_score = ""

            coefs_normalized = (a_coefs**2 + b_coefs**2) / (
                (a_coefs**2).sum() + (b_coefs**2).sum()
            )

            def max_index(lst, item):
                try:
                    return max([i for i, x in enumerate(lst) if x == item])
                except ValueError:
                    return -1

            freq_coefs = coefs_normalized  # np.maximum(a_coefs, b_coefs)
            freq_dist += freq_coefs.tolist()
            seed_ls += [seed for _ in range(len(freq_coefs))]
            freqs_ls += list(range(1, len(freq_coefs) + 1))

        freq_dist_df = pd.DataFrame(
            {"Seed": seed_ls, "Frequency": freqs_ls, "Coefficient": freq_dist}
        )
        freq_dist_df = freq_dist_df.dropna()
        sns.boxplot(data=freq_dist_df, x="Frequency", y="Coefficient", ax=axs[pl_idx])
        # sns.countplot(freq_dist)
        axs[pl_idx].set_title(f"Prompt Length: {2*pl}")

    plt.suptitle(f"Frequency: {max_freq}\nModel: {model_type} Method: {method}")
    fig.tight_layout()
    plt.savefig(f"{save_path}/{model_type}_freq_dist_gold_freq_{max_freq}.png", dpi=300)


if __name__ == "__main__":
    main()
