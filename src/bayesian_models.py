import numpy as np
import arviz as avz
import pymc as pm
from tqdm import tqdm
import torch
from scipy.stats import norm, laplace
from tasks import SignVectorCS, SparseLinearRegression, LinearRegression
from samplers import GaussianSampler


class PMESparseRegressionModel:
    def __init__(self, b=1):
        self.name = "BayesianSparseRegression"
        self.b = b

    def __call__(self, xs, ys, return_weights=False):
        xs, ys = xs.cpu(), ys.cpu()
        inds = range(ys.shape[1])
        preds = []
        all_ws = []

        for i in tqdm(inds):
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue

            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            pms = self.bayesian_fit_batch(train_xs.numpy(), train_ys.numpy())
            all_ws.append(pms)

            pred = test_x @ pms.unsqueeze(-1)
            preds.append(pred[:, 0, 0])

        if not return_weights:
            return torch.stack(preds, dim=1)

        else:
            return torch.stack(preds, dim=1), all_ws

    def bayesian_fit_batch(self, xs, ys):
        pms = []
        for batch in tqdm(range(xs.shape[0])):
            xs_batch, ys_batch = xs[batch], ys[batch]
            pm = self.bayesian_fit(xs_batch, ys_batch)
            pms.append(pm.unsqueeze(0))

        return torch.cat(pms, dim=0)

    def bayesian_fit(self, xs, ys):
        n_dims = xs.shape[-1]
        bayesian_model = pm.Model()
        with bayesian_model:
            # Defining priors
            w = pm.Laplace("w", mu=0, b=self.b, shape=(n_dims, 1))
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Define likelihood

            like_mu = xs @ w
            likelihood = pm.Normal("y", mu=like_mu, sigma=sigma, observed=ys[:, None])

            # Fit the model
            trace = pm.sample()
            summary = avz.summary(trace)
            posterior_mean = summary[summary.index.str.contains("w")]["mean"].values

            return torch.tensor(posterior_mean).float()


class PMESignVectorRegressionModel(PMESparseRegressionModel):
    def __init__(self):
        self.name = "BayesianSignVectorRegression"

    def bayesian_fit(self, xs, ys):
        def logp(value):
            return pm.math.switch(
                value == 1,
                pm.math.log(0.5),
                pm.math.switch(value == -1, pm.math.log(0.5), -np.inf),
            )

        n_dims = xs.shape[-1]
        bayesian_model = pm.Model()
        with bayesian_model:
            # Defining priors
            # w = pm.Bernoulli("w", p = 0.5, shape= (n_dims, 1))
            w = pm.DensityDist("w", logp=logp, shape=(n_dims, 1))
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Define likelihood

            like_mu = xs @ (2 * w - 1)
            likelihood = pm.Normal("y", mu=like_mu, sigma=sigma, observed=ys[:, None])

            # Fit the model
            trace = pm.sample()
            summary = avz.summary(trace)
            posterior_mean = summary[summary.index.str.contains("w")]["mean"].values

            return torch.tensor(posterior_mean).float()
