import os
import json
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eval import read_run_dir, get_model_from_run
from tasks import Polynomials
from samplers import get_data_sampler
from tasks import get_task_sampler


def interpolation_eval_step(model, degree, n_points, seed=None):
    task = Polynomials(1, 1, seeds=[seed], min_degree=degree, max_degree=degree)
    x_cont = (
        torch.linspace(-2, 2, n_points).unsqueeze(0).unsqueeze(-1)
    )  # 20  equally spaced points in (-2, 2) # 1, 20, 1
    y_cont = task.evaluate(x_cont)
    polyfit_errors = []
    transformer_errors = []
    for prompt_size in range(2, degree + 2):
        chosen_idxs = (
            torch.randint(n_points, (prompt_size,)).sort().values.numpy().tolist()
        )  # choose prompt_size many integers from [0, 20)
        x_prompt = x_cont.squeeze()[chosen_idxs].unsqueeze(0).unsqueeze(-1)
        y_prompt = y_cont.squeeze()[chosen_idxs].unsqueeze(0)

        polyfit_featurizer = PolynomialFeatures(degree=prompt_size - 1)
        x_prompt_poly_features = polyfit_featurizer.fit_transform(
            x_prompt.squeeze(0).numpy()
        )
        poly_fit_model = LinearRegression().fit(
            x_prompt_poly_features, y_prompt.squeeze(0).squeeze(-1).numpy()
        )

        x_pred = (
            torch.tensor(
                [
                    x.item()
                    for idx, x in enumerate(x_cont.squeeze())
                    if idx not in chosen_idxs
                ]
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        y_pred = torch.tensor(
            [
                y.item()
                for idx, y in enumerate(y_cont.squeeze())
                if idx not in chosen_idxs
            ]
        ).unsqueeze(0)

        y_polyfit_preds = poly_fit_model.predict(
            polyfit_featurizer.transform(x_pred.squeeze(0).numpy())
        )
        y_transformer_preds = []

        for i in range(x_pred.shape[1]):
            x_prompt_with_ex = torch.cat([x_prompt, x_pred[:, i : i + 1]], axis=1)
            y_prompt_with_ex = torch.cat([y_prompt, y_pred[:, i : i + 1]], axis=1)
            with torch.no_grad():
                y_inter_pred = model(x_prompt_with_ex, y_prompt_with_ex).squeeze()[-1]
                y_transformer_preds.append(y_inter_pred.item())

        y_transformer_preds = np.array(y_transformer_preds)
        polyfit_errors.append(np.mean((y_polyfit_preds - y_pred[0].numpy()) ** 2))
        transformer_errors.append(
            np.mean((y_transformer_preds - y_pred[0].numpy()) ** 2)
        )

    return np.array(transformer_errors), np.array(polyfit_errors)


def interpolation_eval(model, degree, n_points, num_evals=100):
    transformer_errors = np.zeros(degree)
    polyfit_errors = np.zeros(degree)
    for eval_step in tqdm(range(num_evals)):
        transformer_err, polyfit_err = interpolation_eval_step(
            model, degree, n_points, seed=eval_step
        )
        transformer_errors += transformer_err
        polyfit_errors += polyfit_err

    transformer_errors /= num_evals
    polyfit_errors /= num_evals

    return {
        i
        + 2: {"Transformer": transformer_errors[i], "Polynomial Fit": polyfit_errors[i]}
        for i in range(degree)
    }


def extrapolation_eval_step(model, degree, n_points, seed=None):
    task = Polynomials(1, 1, seeds=[seed], min_degree=degree, max_degree=degree)
    x_cont = torch.linspace(-2, 2, n_points).unsqueeze(0).unsqueeze(-1)
    y_cont = task.evaluate(x_cont)
    polyfit_errors = []
    transformer_errors = []
    for prompt_size in range(2, degree + 2):
        chosen_idxs = list(range(prompt_size))
        x_prompt = x_cont.squeeze()[chosen_idxs].unsqueeze(0).unsqueeze(-1)
        y_prompt = y_cont.squeeze()[chosen_idxs].unsqueeze(0)

        polyfit_featurizer = PolynomialFeatures(degree=prompt_size - 1)
        x_prompt_poly_features = polyfit_featurizer.fit_transform(
            x_prompt.squeeze(0).numpy()
        )
        poly_fit_model = LinearRegression().fit(
            x_prompt_poly_features, y_prompt.squeeze(0).squeeze(-1).numpy()
        )

        x_pred = (
            torch.tensor(
                [
                    x.item()
                    for idx, x in enumerate(x_cont.squeeze())
                    if idx not in chosen_idxs
                ]
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        y_pred = torch.tensor(
            [
                y.item()
                for idx, y in enumerate(y_cont.squeeze())
                if idx not in chosen_idxs
            ]
        ).unsqueeze(0)

        y_polyfit_preds = poly_fit_model.predict(
            polyfit_featurizer.transform(x_pred.squeeze(0).numpy())
        )
        y_transformer_preds = []

        for i in range(x_pred.shape[1]):
            x_prompt_with_ex = torch.cat([x_prompt, x_pred[:, i : i + 1]], axis=1)
            y_prompt_with_ex = torch.cat([y_prompt, y_pred[:, i : i + 1]], axis=1)
            with torch.no_grad():
                y_inter_pred = model(x_prompt_with_ex, y_prompt_with_ex).squeeze()[-1]
                y_transformer_preds.append(y_inter_pred.item())

        y_transformer_preds = np.array(y_transformer_preds)
        polyfit_errors.append(np.mean((y_polyfit_preds - y_pred[0].numpy()) ** 2))
        transformer_errors.append(
            np.mean((y_transformer_preds - y_pred[0].numpy()) ** 2)
        )

    return np.array(transformer_errors), np.array(polyfit_errors)


def extrapolation_eval(model, degree, n_points, num_evals=100):
    transformer_errors = np.zeros(degree)
    polyfit_errors = np.zeros(degree)
    for eval_step in tqdm(range(num_evals)):
        transformer_err, polyfit_err = extrapolation_eval_step(
            model, degree, n_points, seed=eval_step
        )
        transformer_errors += transformer_err
        polyfit_errors += polyfit_err

    transformer_errors /= num_evals
    polyfit_errors /= num_evals

    return {
        i
        + 2: {"Transformer": transformer_errors[i], "Polynomial Fit": polyfit_errors[i]}
        for i in range(degree)
    }

    # return transformer_errors, polyfit_errors


def main():
    run_dir = "../models"
    task = "polynomials"
    run_id = "ee1ebdba-ed17-4ba7-8d6b-ba9b72fb2bef"  # if you train more models, replace with the run_id from the table above
    run_path = os.path.join(run_dir, task, run_id)
    model, conf = get_model_from_run(run_path)
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    int_errors_dict = {}
    ext_errors_dict = {}
    for degree in tqdm(range(1, 8)):
        int_errors_dict[f"Degree-{degree}"] = interpolation_eval(
            model, degree=degree, n_points=20, num_evals=100
        )
        ext_errors_dict[f"Degree-{degree}"] = extrapolation_eval(
            model, degree=degree, n_points=20, num_evals=100
        )

    with open(f"{run_dir}/interpolate.json", "w") as f:
        json.dump(int_errors_dict, f)

    with open(f"{run_dir}/extrapolate.json", "w") as f:
        json.dump(ext_errors_dict, f)


if __name__ == "__main__":
    main()
