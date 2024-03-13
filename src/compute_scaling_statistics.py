import os
import argparse
import math
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
from sklearn.preprocessing import PolynomialFeatures

# from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from samplers import UniformSampler, GaussianSampler
from models import NNModel, LeastSquaresModel
from eval import read_run_dir, get_model_from_run
from tasks import PolynomialsFactorForm, Task
from samplers import get_data_sampler
from tasks import get_task_sampler


def main():
    degrees = list(range(1, 8))
    x_dist_a, x_dist_b = -5, 5
    root_dist_a, root_dist_b = -5, 5
    batch_size = 100000
    n_points = 200
    sample_n_times = 10
    data_sampler = UniformSampler(1, x_dist_a, x_dist_b)
    degree2stats_ls = {}
    degree2stats_ls["mu"] = defaultdict(list)
    degree2stats_ls["std"] = defaultdict(list)

    for degree in tqdm(degrees):
        for _ in range(sample_n_times):
            poly = PolynomialsFactorForm(
                1,
                batch_size=batch_size,
                max_degree=degree,
                min_degree=degree,
                standardize=False,
                root_dist="discrete",
                root_dist_kwargs={"a": root_dist_a, "b": root_dist_b},
            )
            xs = data_sampler.sample_xs(n_points, batch_size)
            ys = poly.evaluate(xs)

            mu = ys.mean().item()
            sigma = ys.std().item()
            degree2stats_ls["mu"][degree].append(mu)
            degree2stats_ls["std"][degree].append(sigma)

    degree2stats = {}
    degree2stats["mu"] = {}
    degree2stats["std"] = {}

    for degree in degrees:
        degree2stats["mu"][degree] = {
            "mean": np.mean(degree2stats_ls["mu"][degree]),
            "std": np.std(degree2stats_ls["mu"][degree]),
        }

        degree2stats["std"][degree] = {
            "mean": np.mean(degree2stats_ls["std"][degree]),
            "std": np.std(degree2stats_ls["std"][degree]),
        }

    with open(
        f"stats/polynomials_scaling_stats_xdist_{x_dist_a}{x_dist_b}_rootdist{root_dist_a}{root_dist_b}.json",
        "w",
    ) as f:
        json.dump(obj=degree2stats, fp=f, indent=4)


if __name__ == "__main__":
    main()
