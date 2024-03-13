import math

import torch
import numpy as np
from samplers import sample_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from joblib import Parallel, delayed
from indexing_utils import batched_index_select
import pdb
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.laplace import Laplace
from functools import partial
import os
import time

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, out_dir=None, is_save_task_pool=True, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "sparse_reg_laplacian_prior": SparseRegressionLaplacianPrior,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "noisy_lr_task_diversity": NoisyLinearRegressionTaskDiversity,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "relu_2nn_regression_with_bias": Relu2nnRegressionWithBias,
        "relu_3nn_regression": Relu3nnRegression,
        "decision_tree": DecisionTree,
        "sparse_linear_mixer": SparseLinearMixer,
        "low_rank_cs": LowRankCS,
        "sign_vec_cs": SignVectorCS,
        "two_task_mixer": TwoTaskMixer,
        "three_task_mixer": ThreeTaskMixer,
        "polynomials": Polynomials,
        "polynomials_factor_form": PolynomialsFactorForm,
        "polynomials_unbiased_points": PolynomialsUnbiasedPoints,
        "fourier_series_mixture": FourierSeries,
        "fourier_series": FourierSeriesV2,
        "fourier_series_complexity_bias": FourierSeriesWHighFreqBias,
        "random_fourier_features": FourierRandomFeatures,
        "polynomials_deg2_monomials_selection_biased": PolynomialsDegTwoMonomialSelectionBiased,
        "polynomials_deg2_monomials_selection_unbiased": PolynomialsDegTwoMonomialSelectionUnbiased,
        "gaussian_mixture_linear_regression": GMMLinearRegression,
        "three_gaussian_mixture_linear_regression": GMMLinearRegressionThreeMixture,
        "uniform_mixture_linear_regression": UniformPriorMixLinearRegression,
        "haar_wavelets": HaarWavelets,
        "fourier_series_multitask": FourierSeriesV2Multitask,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            if task_name == "noisy_lr_task_diversity":
                task_rand_gen = torch.Generator().manual_seed(kwargs["task_seed"])
                noise_rand_gen = torch.Generator().manual_seed(kwargs["noise_seed"])
                pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, task_rand_gen, out_dir, is_save_task_pool, **kwargs)
                return lambda **args: task_cls(n_dims, batch_size, pool_dict, task_rand_gen=task_rand_gen,
                    noise_rand_gen=noise_rand_gen, **args, **kwargs)
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        transformation_args=None,
        skew=False,
        skew_seed=42,
        normalize_outputs=False,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.normalize_outputs = normalize_outputs

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

        if transformation_args is not None:
            self.transformation = sample_scale(
                method=transformation_args.get("method", None),
                n_dims=n_dims,
                normalize=transformation_args.get("normalize", False),
                seed=transformation_args.get("seed", None),
            )
        else:
            self.transformation = None
        if self.transformation is not None:
            self.w_b[:, :, 0] = self.w_b[:, :, 0] @ self.transformation

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseRegressionLaplacianPrior(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        transformation_args=None,
        skew=False,
        skew_seed=42,
        normalize_outputs=False,
        b=1,
    ):
        super(SparseRegressionLaplacianPrior, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )

        self.scale = scale
        self.b = b
        self.normalize_outputs = normalize_outputs
        self.dist = Laplace(
            torch.zeros(torch.tensor(n_dims)),
            b * torch.ones(torch.tensor(n_dims)),
        )
        if pool_dict is None and seeds is None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            self.w_b[:, :, 0] = self.dist.sample_n(batch_size)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i, :, 0] = self.dist.sample()

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]

        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(2 * (self.b**2) * self.n_dims)


class GMMLinearRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        transformation_args=None,
        skew=False,
        skew_seed=42,
        mixing_ratio=0.5,
        normalize_outputs=False,
        distrib1=None,
        distrib2=None,
        gaussian_centre_abs=1,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(GMMLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.normalize_outputs = normalize_outputs

        # Let \Sigma1 be the identity matrix with the top left entry set to 0. 
        # And let e1 be the vector with the first coordinate 1 and the rest 0. 
        # Then to generate training data we sample w according to the following mixture:

        #     1/2 N(-e1, \Sigma1) + 1/2 N(e1, \Sigma1)

        if pool_dict is None and seeds is None:
            # mean = torch.zeros(size=(self.n_dims,))
            # mean[0] = 1
            # cov = torch.eye(self.n_dims)
            # cov[0,0] = 1e-8
            # distrib1 = MultivariateNormal(loc=mean, covariance_matrix=cov)
            # distrib2 = MultivariateNormal(loc=-mean, covariance_matrix=cov)
            selected_distrib = distrib1 if np.random.rand() < mixing_ratio else distrib2

            self.w_b = selected_distrib.sample(sample_shape=(self.b_size,)).unsqueeze(dim=-1)

        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class GMMLinearRegressionThreeMixture(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        transformation_args=None,
        skew=False,
        skew_seed=42,
        mixing_ratio={"prior1": 0.3333, "prior2": 0.3333, "prior3": 0.3334},
        normalize_outputs=False,
        priors=None, # [distrib1, distrib2, distrib3]
        gaussian_centre_abs=None, # {"prior1": -3, "prior2": 1, "prior3": 3},
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(GMMLinearRegressionThreeMixture, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.normalize_outputs = normalize_outputs

        assert len(mixing_ratio) == 3, "mixing_ratio must have 3 keys, one per prior"
        assert sum(mixing_ratio.values()) == 1.0, "mixing_ratio values must sum to 1.0"
        assert len(priors) == 3, "priors should be a list of 3 Gaussian distributions"
        max_prob_prior1 = mixing_ratio["prior1"]
        max_prob_prior2 = mixing_ratio["prior2"] + max_prob_prior1

        prior_selector_rand = np.random.rand()
        if prior_selector_rand < max_prob_prior1:
            self.selected_prior = priors[0]
        elif max_prob_prior1 < prior_selector_rand < max_prob_prior2:
            self.selected_prior = priors[1]
        else:
            self.selected_prior = priors[2]

        if pool_dict is None and seeds is None:
            self.w_b = self.selected_prior.sample(sample_shape=(self.b_size,)).unsqueeze(dim=-1)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class UniformPriorMixLinearRegression(Task):
    def sample_uniform(self, a, b, b_size, n_dims):
        w_b = torch.rand(b_size, n_dims, 1)
        w_b = (b - a) * w_b + a
        return w_b

    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        transformation_args=None,
        skew=False,
        skew_seed=42,
        mixing_ratio=0.5,
        normalize_outputs=False,
        w_a1 = 0,
        w_b1 = 1,
        w_a2 = 0,
        w_b2 = 1,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(UniformPriorMixLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.normalize_outputs = normalize_outputs
        self.mixing_ratio = mixing_ratio
        # first distribution is uniform over square [w_a1, w_b1]^2
        # second distribution is uniform over square [w_a2, w_b2]^2
        self.w_a1 = w_a1
        self.w_b1 = w_b1
        self.w_a2 = w_a2
        self.w_b2 = w_b2

        if pool_dict is None and seeds is None:
            self.w_b = self.sample_uniform(self.w_a1, self.w_b1, self.b_size, self.n_dims) if np.random.rand() < self.mixing_ratio else \
                        self.sample_uniform(self.w_a2, self.w_b2, self.b_size, self.n_dims)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
        normalize_outputs=False,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, normalize_outputs=normalize_outputs
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.sparsity)
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearMixer(SparseLinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
        mixing_ratio=0.5,
    ):
        super(SparseLinearMixer, self).__init__(
            n_dims,
            batch_size,
            pool_dict,
            seeds,
            scale,
            sparsity=sparsity if np.random.rand() > mixing_ratio else n_dims,
            valid_coords=valid_coords,
        )


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
        **task_kwargs,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, **task_kwargs,
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class NoisyLinearRegressionTaskDiversity(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        task_rand_gen=None,
        noise_rand_gen=None,
        task_scale=1.0,
        noise_scale=1.0,
        task_seed=None,
        noise_seed=None,
        # transformation_args=None,
        # skew=False,
        # skew_seed=42,
        # normalize_outputs=True
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(NoisyLinearRegressionTaskDiversity, self).__init__(n_dims, batch_size, pool_dict)
        if self.pool_dict is not None:
            self.num_tasks = self.pool_dict["w"].shape[0]
        # self.data_seed = data_seed
        # self.task_seed = task_seed
        # self.noise_seed = noise_seed
        # self.data_scale = data_scale
        self.task_scale = task_scale
        self.noise_scale = noise_scale
        # during evaluation, just create generators with random seeds every time
        if task_rand_gen is None:
            task_rand_gen = torch.Generator().manual_seed(int(task_seed + time.time()))
        if noise_rand_gen is None:
            noise_rand_gen = torch.Generator().manual_seed(int(noise_seed + time.time()))
        self.task_rand_gen = task_rand_gen
        self.noise_rand_gen = noise_rand_gen
        # self.normalize_outputs = normalize_outputs

        if pool_dict is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1, generator=self.task_rand_gen) * self.task_scale
        else:
            assert "w" in pool_dict
            indices = torch.randint(len(pool_dict["w"]), (self.b_size,), generator=self.task_rand_gen)
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b)[:, :, 0]
        noise = torch.randn(ys_b.shape, generator=self.noise_rand_gen) * self.noise_scale
        # if self.normalize_outputs:
        #     ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b + noise

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, task_rand_gen, out_dir, is_save_task_pool, task_scale, **kwargs):  # ignore extra args
        pool_dict = {"w": torch.randn(num_tasks, n_dims, 1, generator=task_rand_gen) * task_scale}
        if is_save_task_pool:
            torch.save(pool_dict["w"], os.path.join(out_dir, "task_pool.pt"))
        # Load as task_pool = torch.load(os.path.join(out_dir, "task_pool.pt"))
        return pool_dict

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def evaluate_oracle(xs_b, w_b):
        ys_b = (xs_b @ w_b)[:, :, 0]
        return ys_b

class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b ** 2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegressionWithBias(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=10,
        normalize_outputs=True,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegressionWithBias, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.normalize_outputs = normalize_outputs

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size) # b, 1, 10
            self.bias = torch.randn(self.b_size, 1, hidden_layer_size) # b, 1, 10
            # self.W2 = torch.randn(self.b_size, hidden_layer_size, 1) # b, 10, 1

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        bias = self.bias.to(xs_b.device)
        # W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        # xs_b -- b, p, 1
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1 + bias)).sum(dim=-1)
        # Var(ys_old) = ((n_dims+1)/2)*h
        # So, to make var=n_dims as LR, we do ys_new = ys_old * sqrt(2/h)
        # Then, Var(ys_new) = Var(ys_old) * 2/h = n_dims
        # ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        # ys_b_nn = self.scale * ys_b_nn
        if self.normalize_outputs:
            ys_b_nn = ys_b_nn / math.sqrt(self.hidden_layer_size*(self.n_dims+1)/2)
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    # @staticmethod
    # def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
    #     return {
    #         "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
    #         "W2": torch.randn(num_tasks, hidden_layer_size, 1),
    #     }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=4,
        normalize_outputs=True,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.normalize_outputs = normalize_outputs

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        # Var(ys_old) = (n_dims/2)*h
        # So, to make var=n_dims as LR, we do ys_new = ys_old * sqrt(2/h)
        # Then, Var(ys_new) = Var(ys_old) * 2/h = n_dims
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        if self.normalize_outputs:
            ys_b_nn = ys_b_nn / math.sqrt(self.n_dims)
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class Relu3nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=4,
        normalize_outputs=True,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu3nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.normalize_outputs = normalize_outputs

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, hidden_layer_size)
            self.W3 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, hidden_layer_size)
            self.W3 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, hidden_layer_size, generator=generator)
                self.W3[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict and "W3" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"]) == len(pool_dict["W3"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]
            self.W3 = pool_dict["W3"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        W3 = self.W3.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        layer_1_activations = torch.nn.functional.relu(xs_b @ W1)
        layer_2_activations = torch.nn.functional.relu(layer_1_activations @ W2)
        ys_b_nn = (layer_2_activations @ W3)[:, :, 0]
        # Var(ys_old) = (n_dims/4)*h^2
        # So, to make var=n_dims as LR, we do ys_new = ys_old * sqrt(4/h^2) i.e. ys_old * 2/h
        # Then, Var(ys_new) = Var(ys_old) * 4/h^2 = n_dims
        ys_b_nn = ys_b_nn * (2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        if self.normalize_outputs:
            ys_b_nn = ys_b_nn / math.sqrt(self.n_dims)
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, hidden_layer_size),
            "W3": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        depth=4,
        scale_leaf_by_ndims=False, # This must be False for mean and variance of targets to be 0 and 1
    ):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth
        leaf_node_var = n_dims if scale_leaf_by_ndims else 1
        # Note: depth-1 means 1 non-leaf node and 2 leaf nodes; depth-k DT means 2^k leaf nodes and 2^k-1 non-leaf nodes.
        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = np.sqrt(leaf_node_var) * torch.randn(
                self.dt_tensor.shape
            )
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = np.sqrt(leaf_node_var) * torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            # In the following loop, after processing level k of DT (i.e. after running the loop for j=k-1), cur_nodes indexes in next level's leaves (2^k of them)
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class CompressedSensing(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(CompressedSensing, self).__init__(
            n_dims=n_dims, batch_size=batch_size, pool_dict=pool_dict, seeds=seeds
        )
        self.scale = scale

    def evaluate(self, xs):
        pass

    def get_bound(self):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LowRankCS(CompressedSensing):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rank=3):
        super(LowRankCS, self).__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            scale=scale,
        )
        self.m = math.isqrt(n_dims)
        self.rank = rank
        # n_dims should be a perfect square
        assert self.m ** 2 == n_dims
        if pool_dict is None and seeds is None:
            self.w_b = self.get_vec_low_rank_matrix(self.b_size, self.m, self.rank)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = self.get_vec_low_rank_matrix(
                    1, self.m, self.rank, generator=generator
                )[0]
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    def get_bound(self):
        # 3r(2m − r)
        return 3 * self.rank * (2 * self.m - self.rank)

    @staticmethod
    def get_vec_low_rank_matrix(b_size, mat_dim, rank, generator=None):
        base_matrix = torch.randn(b_size, rank, mat_dim, generator=generator)
        w_b = base_matrix.transpose(-1, -2) @ base_matrix
        w_b = w_b.view(b_size, -1, 1)
        return w_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rank, **kwargs):  # ignore extra args
        m = math.isqrt(n_dims)
        assert m ** 2 == n_dims
        return {"w": LowRankCS.get_vec_low_rank_matrix(num_tasks, m, rank)}


class SignVectorCS(CompressedSensing):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, normalize_outputs=True):
        super(SignVectorCS, self).__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            scale=scale,
        )
        self.normalize_outputs = normalize_outputs
        # Make sure that the dimensionality is even (Though Probably not needed)
        assert self.n_dims % 2 == 0

        if pool_dict is None and seeds is None:
            self.w_b = self.get_sign_vec(self.b_size, self.n_dims)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = self.get_sign_vec(1, self.n_dims, generator=generator)[0]
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    def get_bound(self):
        return self.n_dims // 2

    @staticmethod
    def get_sign_vec(b_size, n_dims, generator=None):
        # w_b = torch.tensor(np.random.choice([1, -1], (b_size, n_dims, 1)))
        w_b = torch.randint(
            low=-1, high=1, size=(b_size, n_dims, 1), generator=generator
        ).float()
        w_b = w_b * 2 + 1
        return w_b


class TwoTaskMixer(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        task1="linear_regression",
        task2="decision_tree",
        mixing_ratio=0.5,
        task_spec_params_dict={},
    ):
        self.task1 = task1
        self.task2 = task2
        self.selected_task = task1 if np.random.rand() < mixing_ratio else task2
        self.task = get_task_sampler(
            self.selected_task,
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            **task_spec_params_dict.get(self.selected_task, {})
        )(seeds=seeds)

    def evaluate(self, xs):
        return self.task.evaluate(xs)

    def get_task_label(self, xs):
        if self.selected_task == self.task1:
            return torch.zeros(xs.shape[0]).to(xs.device).long()
        else:
            return torch.ones(xs.shape[0]).to(xs.device).long()

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class ThreeTaskMixer(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        task1="linear_regression",
        task2="decision_tree",
        task3="relu_2nn_regression",
        mixing_ratio={"task1": 0.3333, "task2": 0.3333, "task3": 0.3334},
        task_spec_params_dict={},
    ):
        # import pdb
        # pdb.set_trace()
        self.tasks = [task1, task2, task3]

        assert len(mixing_ratio) == 3, "mixing_ratio must have 3 keys, one per task"
        assert sum(mixing_ratio.values()) == 1.0, "mixing_ratio values must sum to 1.0"
        max_prob_task1 = mixing_ratio["task1"]
        max_prob_task2 = mixing_ratio["task2"] + max_prob_task1
        # max_prob_task3 = mixing_ratio[2] + max_prob_task2

        task_selector_rand = np.random.rand()
        if task_selector_rand < max_prob_task1:
            self.selected_task = task1
        elif max_prob_task1 < task_selector_rand < max_prob_task2:
            self.selected_task = task2
        else:
            self.selected_task = task3

        # self.selected_task = task1 if np.random.rand() < mixing_ratio else task2
        self.task = get_task_sampler(
            self.selected_task,
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            **task_spec_params_dict.get(self.selected_task, {})
        )(seeds=seeds)

    def evaluate(self, xs):
        return self.task.evaluate(xs)

    def get_task_label(self, xs):
        task_labels = torch.zeros(xs.shape[0], len(self.tasks)).to(xs.device)
        task_labels[:, self.tasks.index(self.selected_task)] = 1
        return task_labels

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class Polynomials(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_degree=5,
        min_degree=1,
    ):
        super(Polynomials, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1
        self.scale = scale
        self.max_degree = max_degree
        if pool_dict is None and seeds is None:
            self.degree = torch.randint(min_degree, max_degree + 1, size=(self.b_size,))
            degree_mask = (
                torch.arange(max_degree + 1)[None, :] < (1 + self.degree)[:, None]
            ).float()
            self.coefs = torch.randn(self.b_size, max_degree + 1)
            self.coefs = self.coefs * degree_mask

        elif seeds is not None:
            self.degree = []
            self.coefs = torch.zeros(self.b_size, max_degree + 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.degree.append(
                    torch.randint(
                        min_degree, max_degree + 1, size=(1,), generator=generator
                    )
                )
                self.coefs[i] = torch.randn(max_degree + 1, generator=generator)
            self.degree = torch.cat(self.degree, dim=0)
            degree_mask = (
                torch.arange(max_degree + 1)[None, :] < (1 + self.degree)[:, None]
            ).float()
            self.coefs = self.coefs * degree_mask

        else:
            raise NotImplementedError()

    def evaluate(self, xs_b):
        coefs = self.coefs.to(xs_b.device)
        poly_xs_b = PolynomialFeatures(self.max_degree).fit_transform(xs_b.view(-1, 1))
        poly_xs_b = torch.tensor(poly_xs_b).to(xs_b.device)
        poly_xs_b = poly_xs_b[:, None, :].view(self.b_size, -1, self.max_degree + 1)
        ys_b = self.scale * ((coefs[:, None, :] * poly_xs_b).sum(dim=-1))
        return ys_b.float()

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class PolynomialsFactorForm(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_degree=5,
        min_degree=1,
        root_var=1,
        standardize=False,
        root_dist="gaussian",
        x_dist="uniform",
        root_dist_kwargs={},
        dist_kwargs={},
    ):
        super(PolynomialsFactorForm, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )
        assert n_dims == 1

        self.scale = scale
        self.max_degree = max_degree
        self.standardize = standardize
        self.x_dist = "uniform"
        self.root_dist = root_dist
        self.dist_kwargs = dist_kwargs
        self.root_dist_kwargs = root_dist_kwargs
        self.scaling_statistics = {}
        if self.x_dist == "uniform" and self.root_dist == "disjoint":
            xdist_a, xdist_b = self.dist_kwargs.get("a", -5), self.dist_kwargs.get(
                "b", 5
            )
            rootdist_a, rootdist_b = self.root_dist_kwargs.get(
                "a", -5
            ), self.root_dist_kwargs.get("b", 5)
            if xdist_a == -5 and xdist_b == 5 and rootdist_a == -5 and rootdist_b == 5:
                with open(
                    "../stats/polynomials_scaling_stats_xdist_-55_rootdist-55.json", "r"
                ) as f:
                    self.scaling_statistics = json.load(f)
        self.degree = torch.randint(min_degree, max_degree + 1, size=(self.b_size,))
        self.roots = []

        for degree in self.degree:
            if root_dist == "gaussian":
                self.roots.append(
                    math.sqrt(root_dist_kwargs.get("var", 1)) * torch.randn(1, degree)
                    + root_dist_kwargs.get("mu", 0)
                )
            elif root_dist == "uniform":
                roots = torch.rand(1, degree)
                a, b = (
                    root_dist_kwargs.get("a", -2),
                    root_dist_kwargs.get("b", 2),
                )
                roots = (b - a) * roots + a
                self.roots.append(roots)
            elif root_dist == "uniform-dynamic":
                roots = torch.rand(1, degree)
                a, b = (
                    root_dist_kwargs.get("a", -2),
                    root_dist_kwargs.get("b", 2),
                )
                if a < 0:
                    a = -1 * (abs(a)) ** (degree - 1)
                else:
                    a = a ** (degree - 1)

                if b < 0:
                    b = -1 * (abs(b)) ** (degree - 1)
                else:
                    b = b ** (degree)
                roots = (b - a) * roots + a
                self.roots.append(roots)
            elif root_dist == "disjoint":
                a, b = root_dist_kwargs.get("a", -2), root_dist_kwargs.get("b", 2)
                interval_length = 2 * (b - a) / (3 * degree)
                roots = []
                interval_start = a
                for i in range(degree):
                    interval_end = interval_start + interval_length
                    root = torch.rand(1, 1)
                    root = (interval_end - interval_start) * root + interval_start
                    roots.append(root)
                    interval_start = interval_end + interval_length / 2
                roots = torch.cat(roots, axis=1)
                self.roots.append(roots)

    def evaluate(self, xs_b):
        preds = []
        for batch in range(xs_b.size(0)):
            pred = torch.prod(
                xs_b[batch].squeeze(-1).unsqueeze(1) - self.roots[batch], axis=1
            ).unsqueeze(0)
            if self.standardize:
                if self.x_dist == "uniform" and self.root_dist == "gaussian":
                    a, b = self.dist_kwargs.get("a", -2), self.dist_kwargs.get("b", 2)
                    mu = self.exp_poly(self.degree[batch], a, b)
                    var = self.var_poly(
                        self.degree[batch],
                        a,
                        b,
                        root_var=self.root_dist_kwargs.get("var", 1),
                    )
                    pred = (pred - mu) / math.sqrt(var)
                elif self.x_dist == "uniform" and self.root_dist == "disjoint":
                    if self.scaling_statistics == {}:
                        raise NotImplementedError()
                    else:
                        mu = self.scaling_statistics["mu"][
                            str(self.degree[batch].item())
                        ]["mean"]
                        std = self.scaling_statistics["std"][
                            str(self.degree[batch].item())
                        ]["mean"]
                        pred = (pred - mu) / std
                else:
                    raise NotImplementedError()
            preds.append(pred)
        return torch.cat(preds, axis=0)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    def exp_xn_uniform(a, b, n):
        return (1 / ((b - a) * (n + 1))) * (b ** (n + 1) - a ** (n + 1))

    @staticmethod
    def exp_poly(degree, a, b):
        return PolynomialsFactorForm.exp_xn_uniform(a, b, degree)

    @staticmethod
    def var_poly(degree, a, b, root_var=1):
        var = (
            PolynomialsFactorForm.exp_xn_uniform(a, b, degree * 2)
            - (PolynomialsFactorForm.exp_xn_uniform(a, b, degree)) ** 2
        )
        for i in range(degree):
            coef = math.comb(degree, degree - i) * (root_var ** (degree - i))
            var += coef * (PolynomialsFactorForm.exp_xn_uniform(a, b, 2 * i))
        return var


class PolynomialsUnbiasedPoints(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_degree=5,
        min_degree=1,
        rejection_sampling_coeff_L1_norm_bound=None,
    ):
        super(PolynomialsUnbiasedPoints, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )
        assert n_dims == 1
        self.scale = scale
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.rejection_sampling_coeff_L1_norm_bound = (
            rejection_sampling_coeff_L1_norm_bound
        )
        self.n_dims = n_dims

        self.pool_dict = pool_dict
        self.seeds = seeds

    def make_poly_from_points(self, x, y, d):
        # fit a d degree poly for points (x1, y1), (x2, y2), ... (x_{d+1}, y_{d+1})
        # return the polynomial as coefficients for terms [1, x, x^2, ...x^M]; M=self.max_degree
        poly = PolynomialFeatures(degree=d)
        poly_features = poly.fit_transform(x.reshape(-1, 1))

        poly_reg_model = linear_model.LinearRegression(fit_intercept=False)
        poly_reg_model.fit(poly_features, y)

        return torch.from_numpy(
            np.pad(
                poly_reg_model.coef_,
                (0, self.max_degree - d),
                "constant",
                constant_values=(0.0,),
            )
        )

    def make_poly_from_points_batch(self, xs_b, ys_b_poly):
        if self.pool_dict is None and self.seeds is None:
            self.degree = torch.randint(
                self.min_degree, self.max_degree + 1, size=(self.b_size,)
            ).tolist()
            # print("self.degree",self.degree)

            # self.coefs found from polynomial regression

            # loop version
            coefs = []
            for b, deg in enumerate(self.degree):
                coefs.append(
                    self.make_poly_from_points(
                        xs_b[b, : (deg + 1), :], ys_b_poly[b, : (deg + 1)], deg
                    )
                )

            # parallelized version
            # coefs_parallel = Parallel(n_jobs=32)(delayed(self.make_poly_from_points)(xs_b[b,:(deg+1),:],
            #             ys_b_poly[b,:(deg+1)],
            #             deg) for b, deg in enumerate(self.degree))

            self.coefs = torch.stack(coefs, dim=0)
            # self.coefs = torch.stack(coefs_parallel, dim=0)

            # print("eqqq", torch.equal(self.coefs, self.coefs_parallel))

        else:
            raise NotImplementedError()

    def evaluate(self, xs_b, ys_b_poly):
        # make poly
        self.make_poly_from_points_batch(xs_b, ys_b_poly)

        coefs = self.coefs.to(xs_b.device)
        # print("coefs",coefs)
        poly_xs_b = PolynomialFeatures(self.max_degree).fit_transform(xs_b.view(-1, 1))
        poly_xs_b = torch.tensor(poly_xs_b).to(xs_b.device)
        poly_xs_b = poly_xs_b[:, None, :].view(self.b_size, -1, self.max_degree + 1)
        ys_b = self.scale * ((coefs[:, None, :] * poly_xs_b).sum(dim=-1))
        return ys_b.float(), self.coefs

    def sample_batch_ys(self, xs, data_sampler, data_sampler_args, bsize, n_points):
        ys_poly_construction = data_sampler.sample_xs(
            n_points,
            bsize,
            **data_sampler_args,
        ).reshape(bsize, n_points)
        ys, coef = self.evaluate(
            xs,
            ys_poly_construction  # we need a ys_poly_construction of shape [batch, n_points] but the sample_xs gives [batch, n_points, n_dims].
            # Since n_dims=1, we reshape to get the desired shape. If n_dims is not 1 then we can't use sample_xs and reshape.
        )
        return ys, coef

    # Version 1
    def rejection_sample_to_form_batch(
        self, xs, data_sampler, data_sampler_args, bsize, n_points, excess_tensors
    ):
        ys = None
        # pdb.set_trace()
        if self.rejection_sampling_coeff_L1_norm_bound is None:
            ys, _ = self.sample_batch_ys(
                xs, data_sampler, data_sampler_args, bsize, n_points
            )
        else:
            # loop until len(self.excess_elems_satisfying_norm_bound) < batch_size:
            while len(excess_tensors["tens"]) < self.b_size:
                batches_ys = []
                batches_coefs = []
                # sample 50 batches as usual
                for i in range(50):
                    ys_b, coef_b = self.sample_batch_ys(
                        xs, data_sampler, data_sampler_args, bsize, n_points
                    )
                    batches_ys.append(ys_b)
                    batches_coefs.append(coef_b)
                concat_batches_poly = torch.cat(
                    batches_ys, dim=0
                )  # 50*batch_size, n_points
                concat_batches_coefs = torch.cat(
                    batches_coefs, dim=0
                )  # 50*batch_size, max_degree + 1
                # filter the norm bound solutions and save in self.excess_elems_satisfying_norm_bound
                norms = torch.norm(concat_batches_coefs, p=1, dim=1)  # 50*batch_size
                filtered_poly = concat_batches_poly[
                    norms <= self.rejection_sampling_coeff_L1_norm_bound, :
                ]
                filtered_coefs = concat_batches_coefs[
                    norms <= self.rejection_sampling_coeff_L1_norm_bound, :
                ]
                excess_tensors["tens"] = torch.cat(
                    [excess_tensors["tens"], filtered_poly], dim=0
                )
                excess_tensors["coefs"] = torch.cat(
                    [excess_tensors["coefs"], filtered_coefs], dim=0
                )
            # use whatever needed to construct the batch and leave remaining in self.excess_elems_satisfying_norm_bound
            ys = excess_tensors["tens"][: self.b_size, :]
            coefs_to_return = excess_tensors["coefs"][: self.b_size, :]
            excess_tensors["tens"] = excess_tensors["tens"][self.b_size :, :]
            excess_tensors["coefs"] = excess_tensors["coefs"][self.b_size :, :]
        return ys, coefs_to_return

    # Version 2
    # def rejection_sample_to_form_batch(self, xs, data_sampler, data_sampler_args, bsize, n_points, excess_tensors):
    #     ys = None
    #     # pdb.set_trace()
    #     if self.rejection_sampling_coeff_L1_norm_bound is None:
    #         ys, _ = self.sample_batch_ys(xs, data_sampler, data_sampler_args, bsize, n_points)
    #     else:
    #         # loop until len(excess_elems_satisfying_norm_bound) < batch_size:
    #         len_excess = len(excess_tensors["tens"])
    #         while len_excess < self.b_size:
    #             # sample 50 batches as usual
    #             for i in range(50):
    #                 ys_b, coef_b = self.sample_batch_ys(xs, data_sampler, data_sampler_args, bsize, n_points)
    #                 norms = torch.norm(coef_b, p=1, dim=1) # batch_size
    #                 filtered_poly = ys_b[norms<=self.rejection_sampling_coeff_L1_norm_bound,:]
    #                 excess_tensors["tens"] = torch.cat([excess_tensors["tens"], filtered_poly], dim=0)
    #             len_excess = len(excess_tensors["tens"])
    #         # use whatever needed to construct the batch and leave remaining in excess_elems_satisfying_norm_bound
    #         ys = excess_tensors["tens"][:self.b_size, :]
    #         excess_tensors["tens"] = excess_tensors["tens"][self.b_size:,:]

    #     return ys

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class PolynomialsDegTwoMonomialSelectionBiased(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        normalize_outputs=True,  # scale target ys appropriately such that variance is 1.
        # We ideally want mean=0 and var=1 but we can't change the mean as that involves subtraction which changes the task.
        # We can change the variance without affecting the regression task as the scale factor can be thought of as getting absorbed in the weight matrix.
        # Also, since in these regression-like tasks mean is already 0, we only need to scale the variance.
        # max_degree=5,
        # min_degree=1,
    ):
        super(PolynomialsDegTwoMonomialSelectionBiased, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # assert n_dims == 1
        self.scale = scale
        # self.max_degree = max_degree
        # self.min_degree = min_degree
        # self.n_dims = n_dims
        # self.b_size = batch_size

        # self.pool_dict = pool_dict
        # self.seeds = seeds
        self.normalize_outputs = normalize_outputs

        if pool_dict is None and seeds is None:
            # select batch size many monomials and weights
            ## first index for monomial term (B, P)
            self.monomial_idx_1 = torch.stack([torch.randperm(self.n_dims) for b in range (self.b_size)],dim=0)
            # print("mon. idx 1", self.monomial_idx_1 )

            ## second index for monomial term (B, P)
            self.monomial_idx_2 = torch.stack([torch.randperm(self.n_dims) for b in range (self.b_size)],dim=0)
            # print("mon. idx 2", self.monomial_idx_2)

            # weight
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
            # print("w_b", self.w_b)
            # self.w_b = torch.ones(self.b_size, self.n_dims, 1)
        else:
            raise NotImplementedError()

    # evaluate
    # process: batch x points x dims input comes --> for each point use the monomial selection matrix (2 matrices of shape batch x 20) above to mix-multiply, getting batch x points x dims where the last dimension contains 20 monomial terms now --> bmm with batch x dims x 1 weight matrix to get ys of shape batch x points x 1.
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        monomial_var_1 = batched_index_select(xs_b, -1, self.monomial_idx_1.to(xs_b.device)) # B, P, D
        monomial_var_2 = batched_index_select(xs_b, -1, self.monomial_idx_2.to(xs_b.device)) # B, P, D
        # print("mon. var 1", monomial_var_1)
        # print("mon. var 2", monomial_var_2)

        monomial_terms = (monomial_var_1 * monomial_var_2).to(xs_b.device) # B, P, D
        # print("mon. terms", monomial_terms)
        ys_b = self.scale * (monomial_terms @ w_b)[:, :, 0] # B, P
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.n_dims)
        return ys_b

    def evaluate_ood(self, xs_b):
        # select batch size many monomials

        # make a list of all the 20 + 190 deg 2 monomial term indices i.e. {[0, 0], [1, 1],...,[19, 19], [0, 1], [0, 2], ....[18, 19]}
        all_deg2_terms = []
        for i in range(self.n_dims):
            all_deg2_terms.append([i, i])
            for j in range(i+1,self.n_dims):
                all_deg2_terms.append([i, j])
        # print(len(all_deg2_terms))
        # print(all_deg2_terms)
        # for b in batch_size:
            # select 20 elements from the list
        # stack this to get a tensor of shape batch x 20 x 2
        selected_monomial_indices = torch.stack([torch.tensor(random.sample(all_deg2_terms, self.n_dims),dtype=torch.int64)
                                                  for b in range (self.b_size)],dim=0)
        # put the first and second indices of all elements in separate tensors using indexing
        self.monomial_idx_1 = selected_monomial_indices[:,:,0]
        self.monomial_idx_2 = selected_monomial_indices[:,:,1]
        return self.evaluate(xs_b)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class PolynomialsDegTwoMonomialSelectionUnbiased(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        all_deg2_terms=None,
        variant="",
        numDeg2Select=0, # number of degree 2 monomial terms to be selected # relevant for non-fixedS variants
        fixedS=None,
        sizeOfK=0, # number of different S's used for the fixedK variant
        fixedK=None,
        normalize_outputs=True, # scale target ys appropriately such that variance is 1.
                        # We ideally want mean=0 and var=1 but we can't change the mean as that involves subtraction which changes the task.
                        # We can change the variance without affecting the regression task as the scale factor can be thought of as getting absorbed in the weight matrix.
                        # Also, since in these regression-like tasks mean is already 0, we only need to scale the variance.
        # max_degree=5,
        # min_degree=1,
        each_function_different_S=False, # whether to use a different S per function in batch or form each function in a batch using the same S
    ):
        super(PolynomialsDegTwoMonomialSelectionUnbiased, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # assert n_dims == 1
        self.scale = scale
        self.all_deg2_terms = all_deg2_terms
        self.variant = variant
        self.numDeg2Select = numDeg2Select
        self.sizeOfK = sizeOfK
        # self.max_degree = max_degree
        # self.min_degree = min_degree
        # self.n_dims = n_dims
        # self.b_size = batch_size

        # self.pool_dict = pool_dict
        # self.seeds = seeds
        self.fixedS = fixedS # list of lists, shape (numDeg2Select, 2)
        self.fixedK = fixedK # torch.int64 tensor of shape (sizeOfK, numDeg2Select, 2)
        self.normalize_outputs = normalize_outputs
        self.each_function_different_S = each_function_different_S

        if pool_dict is None and seeds is None:
            # select batch size many monomials and weights

            # selecting monomials
            if self.variant == "fixedS":
                self.selected_monomial_indices = torch.stack([torch.tensor(self.fixedS, dtype=torch.int64) for b in range (self.b_size)],dim=0) # (B, numDeg2Select, 2)
            elif self.variant == "fixedK":
                if self.each_function_different_S:
                    chosenSindices = torch.randint(0, self.sizeOfK, size=(self.b_size,))
                    self.selected_monomial_indices = self.fixedK[chosenSindices, :, :] # (B, numDeg2Select, 2)
                else:
                    chosenSindex = random.randint(0, self.sizeOfK-1)
                    # self.selected_monomial_indices = torch.stack([self.fixedK[chosenSindex] for b in range (self.b_size)],dim=0) # (B, numDeg2Select, 2)
                    self.selected_monomial_indices = self.fixedK[chosenSindex].repeat(self.b_size, 1, 1) # (B, numDeg2Select, 2)
            elif self.variant == "randomS":
                # each f in every batch is formed by a completely random set S; There is no notion of K here. Just that we sample batch size-many different S's and create f's (one per S).
                # bounds and sparsity: d=10 => no. of deg 2 terms = 55; For sparsity = |S| = 5, bound=30.23 << 55
                # model capacity: The standard transformer (emb=256, head=8, layer=12, params=22.4M) model is sufficient to learn upto 50-dim problems.
                # Hence, for this experiment, we should not have issues with the standard model.
                self.selected_monomial_indices = torch.stack([torch.tensor(random.sample(self.all_deg2_terms, self.numDeg2Select),dtype=torch.int64) for b in range (self.b_size)],dim=0) # (B, numDeg2Select, 2)
            else:
                raise NotImplementedError()

            # selecting weights
            self.w_b = torch.randn(self.b_size, self.numDeg2Select, 1) # (B, numDeg2Select)
            # print("w_b", self.w_b)
            # self.w_b = torch.ones(self.b_size, self.n_dims, 1)
        else:
            raise NotImplementedError()

    def evaluate_assist(self, xs_b): # B, P, n_dims
        w_b = self.w_b.to(xs_b.device)
        monomial_var_1 = batched_index_select(xs_b, -1, self.monomial_idx_1.to(xs_b.device)) # B, P, numDeg2Select
        monomial_var_2 = batched_index_select(xs_b, -1, self.monomial_idx_2.to(xs_b.device)) # B, P, numDeg2Select
        # print("mon. var 1", monomial_var_1)
        # print("mon. var 2", monomial_var_2)

        monomial_terms = (monomial_var_1 * monomial_var_2).to(xs_b.device) # B, P, numDeg2Select
        # print("mon. terms", monomial_terms)
        ys_b = self.scale * (monomial_terms @ w_b)[:, :, 0] # B, P
        if self.normalize_outputs:
            ys_b = ys_b / math.sqrt(self.numDeg2Select)
        return ys_b

    # evaluate
    # process: batch x points x dims input comes --> for each point use the monomial selection matrix (2 matrices of shape batch x 20) above to mix-multiply, getting batch x points x dims where the last dimension contains 20 monomial terms now --> bmm with batch x dims x 1 weight matrix to get ys of shape batch x points x 1.
    def evaluate(self, xs_b, mode="train", ind=None, chosenSindices=None):
        if self.variant in ["fixedS", "randomS"]: # train as well as evaluate on a batch with polynomials formed using just one S, i.e. the fixed S; or on ones formed using random S
            pass
        elif self.variant == "fixedK":
            if mode == "train": # create an S-pure batch (i.e. choose an S from K and form the batch)
                pass
                # chosenSindex = ind
                # self.selected_monomial_indices = torch.stack([self.fixedK[chosenSindex] for b in range (self.b_size)],dim=0) # (B, numDeg2Select, 2)
                # self.selected_monomial_indices = self.fixedK[chosenSindex].repeat(self.b_size, 1, 1) # (B, numDeg2Select, 2)
            elif mode == "eval": # create a batch with equal number of polynomials formed using each S in K
                # choose batch size many indices in range [0, sizeOfK] with replacement
                if chosenSindices is None:
                    chosenSindices = torch.randint(0, self.sizeOfK, (self.b_size,))
                self.selected_monomial_indices = self.fixedK[chosenSindices, :, :] # (B, numDeg2Select, 2)
            else:
                raise NotImplementedError

        # put the first and second indices of all elements in separate tensors using indexing
        self.monomial_idx_1 = self.selected_monomial_indices[:,:,0] # (B, numDeg2Select)
        self.monomial_idx_2 = self.selected_monomial_indices[:,:,1] # (B, numDeg2Select)
        return self.evaluate_assist(xs_b)

    def evaluate_ood(self, xs_b):
        # select batch size many monomials
        # stack this to get a tensor of shape batch x 20 x 2
        self.selected_monomial_indices = torch.stack([torch.tensor(random.sample(self.all_deg2_terms, self.numDeg2Select),dtype=torch.int64) for b in range (self.b_size)],dim=0)
        # self.selected_monomial_indices = torch.tensor(random.sample(self.all_deg2_terms, self.numDeg2Select),dtype=torch.int64).repeat(self.b_size, 1, 1)
        # put the first and second indices of all elements in separate tensors using indexing
        self.monomial_idx_1 = self.selected_monomial_indices[:,:,0]
        self.monomial_idx_2 = self.selected_monomial_indices[:,:,1]
        return self.evaluate_assist(xs_b)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierSeries(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        min_frequency=1,
        max_frequency=10,
        L=5,
        standardize=False,
    ):
        super(FourierSeries, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1
        self.scale = scale
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.L = L
        self.standardize = standardize

        if pool_dict is None and seeds is None:
            self.frequencies = torch.randint(
                min_frequency, max_frequency + 1, size=(batch_size,)
            )

            self.a_coefs = torch.randn(self.b_size, max_frequency)
            self.b_coefs = torch.randn(self.b_size, max_frequency)

        elif seeds is not None:
            self.frequencies = []
            self.a_coefs = torch.zeros(self.b_size, max_frequency)
            self.b_coefs = torch.zeros(self.b_size, max_frequency)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.frequencies.append(
                    torch.randint(
                        min_frequency, max_frequency + 1, size=(1,), generator=generator
                    )
                )
                self.a_coefs = torch.randn(max_frequency, generator=generator)
                self.b_coefs = torch.randn(max_frequency, generator=generator)
            self.frequencies = torch.tensor(self.frequencies)
        else:
            raise NotImplementedError()

        a_freq_mask = (
            torch.arange(1, max_frequency + 1)[None, :]
            < (1 + self.frequencies)[:, None]
        ).float()
        self.a_coefs = self.a_coefs * a_freq_mask

        b_freq_mask = (
            torch.arange(1, max_frequency + 1)[None, :]
            < (1 + self.frequencies)[:, None]
        ).float()
        self.b_coefs = self.b_coefs * b_freq_mask

    def evaluate(self, xs_b):
        device = xs_b.device

        a_coefs = self.a_coefs.to(device)
        b_coefs = self.b_coefs.to(device)

        cosine_terms = self.a_coefs.unsqueeze(1) * (
            torch.cos(
                (math.pi / (self.L))
                * torch.arange(1, self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency)
        )

        sine_terms = self.b_coefs.unsqueeze(1) * (
            torch.sin(
                (math.pi / (self.L))
                * torch.arange(1, self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency)
        )

        if self.standardize:
            return (
                self.scale
                * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))
                / ((self.frequencies - self.min_frequency + 1).sqrt().unsqueeze(dim=-1))
            )
        return self.scale * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierSeriesV2(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_frequency=10,
        min_frequency=1,
        L=5,
        standardize=False,
        intercept=False,
    ):
        super(FourierSeriesV2, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1

        self.scale = scale
        self.max_frequency = max_frequency
        self.L = L
        self.standardize = standardize

        if pool_dict is None and seeds is None:
            coefs = torch.randn(self.b_size, 2 * max_frequency + 1)
            self.a_coefs = coefs[:, : max_frequency + 1]
            self.b_coefs = coefs[:, max_frequency + 1 :]

        elif seeds is not None:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        a_freq_mask = torch.zeros(1, max_frequency + 1)
        a_freq_mask[:, 0] = 1
        a_freq_mask[:, min_frequency:] = 1
        b_freq_mask = torch.zeros(1, max_frequency)
        b_freq_mask[:, min_frequency - 1 :] = 1

        self.a_coefs = self.a_coefs * a_freq_mask
        self.b_coefs = self.b_coefs * b_freq_mask

    def evaluate(self, xs_b):
        device = xs_b.device

        a_coefs = self.a_coefs.to(device)
        b_coefs = self.b_coefs.to(device)

        cosine_terms = self.a_coefs.unsqueeze(1) * (
            torch.cos(
                (math.pi / (self.L))
                * torch.arange(self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency + 1)
        )

        sine_terms = self.b_coefs.unsqueeze(1) * (
            torch.sin(
                (math.pi / (self.L))
                * torch.arange(1, self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency)
        )

        if self.standardize:
            return (
                self.scale
                * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))
                / (math.sqrt(self.max_frequency))
            )
        return self.scale * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierSeriesWHighFreqBias(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        min_frequency=1,
        max_frequency=10,
        L=5,
        standardize=False,
    ):
        super(FourierSeriesWHighFreqBias, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )
        assert n_dims == 1
        self.scale = scale
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.L = L
        self.standardize = standardize

        if pool_dict is None and seeds is None:
            self.min_frequencies = torch.randint(
                min_frequency, max_frequency + 1, size=(batch_size,)
            )

            self.a_coefs = torch.randn(self.b_size, max_frequency)
            self.b_coefs = torch.randn(self.b_size, max_frequency)

        else:
            raise NotImplementedError()

        a_freq_mask = (
            torch.arange(1, max_frequency + 1)[None, :]
            >= (self.min_frequencies)[:, None]
        ).float()
        b_freq_mask = (
            torch.arange(1, max_frequency + 1)[None, :]
            >= (self.min_frequencies)[:, None]
        ).float()

        self.a_coefs = self.a_coefs * a_freq_mask
        self.b_coefs = self.b_coefs * b_freq_mask

    def evaluate(self, xs_b):
        device = xs_b.device

        a_coefs = self.a_coefs.to(device)
        b_coefs = self.b_coefs.to(device)

        cosine_terms = self.a_coefs.unsqueeze(1) * (
            torch.cos(
                (math.pi / (self.L))
                * torch.arange(1, self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency)
        )

        sine_terms = self.b_coefs.unsqueeze(1) * (
            torch.sin(
                (math.pi / (self.L))
                * torch.arange(1, self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b.view(-1, 1)
            ).view(xs_b.size(0), -1, self.max_frequency)
        )

        if self.standardize:
            return (
                self.scale
                * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))
                / ((self.max_frequency - self.min_frequencies + 1).sqrt().unsqueeze(dim=-1))
            )
        return self.scale * (sine_terms.sum(axis=-1) + cosine_terms.sum(axis=-1))

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierRandomFeatures(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        rff_dim=20,
        fixed_vectors=True,
        standardize=False,
        sigma=1,
    ):
        super(FourierRandomFeatures, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )

        self.n_dims = n_dims
        self.rff_dim = rff_dim
        self.standardize = standardize
        self.scale = scale
        self.fixed_vectors = fixed_vectors
        self.sigma = sigma

        if pool_dict is None and seeds is None:
            if self.fixed_vectors:
                np.random.seed(42)
                self.W_, self.b_ = self._init_rff_vars()
            else:
                self.W_, self.b_ = self._init_rff_vars()
            self.coefs = torch.randn(
                self.b_size, self.rff_dim, 1
            )  # / math.sqrt(rff_dim)

        elif seeds is not None:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def evaluate(self, xs_b):
        device = xs_b.device
        coefs = self.coefs.to(device)
        # vectors = self.vectors.to(device)

        xs_b_rff_ready = xs_b.view(-1, xs_b.shape[-1]).cpu().numpy()
        features_rff_out = self._get_rffs(xs_b_rff_ready)
        features = (
            torch.tensor(features_rff_out)
            .float()
            .view(xs_b.shape[0], xs_b.shape[1], -1)
            .to(device)
        )

        # features = xs_b @ vectors.transpose(-1, -2)
        # features = torch.cat([torch.cos(features), torch.sin(features)], axis = -1) / math.sqrt(self.D)

        ys_b = self.scale * (features @ coefs)[:, :, 0]

        # if self.standardize:
        #     ys_b /= math.sqrt(self.rff_dim)

        return ys_b

    def extract_features(self, xs_b):
        # device = xs_b.device
        # coefs = self.coefs.to(device)

        # if xs_b.size(0) == 1:
        #     vectors = self.vectors.to(device)[:1]
        # else:
        #     vectors = self.vectors.to(device)

        # features = xs_b @ vectors.transpose(-1, -2)
        # features = torch.cat([torch.cos(features), torch.sin(features)], axis = -1) / math.sqrt(self.D)

        device = xs_b.device
        coefs = self.coefs.to(device)
        # vectors = self.vectors.to(device)

        xs_b_rff_ready = xs_b.view(-1, xs_b.shape[-1]).cpu().numpy()
        features_rff_out = self._get_rffs(xs_b_rff_ready)
        features = (
            torch.tensor(features_rff_out)
            .float()
            .view(xs_b.shape[0], xs_b.shape[1], -1)
            .to(device)
        )

        return features

    def _init_rff_vars(self):
        W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, self.n_dims))
        b = np.random.uniform(0, 2 * np.pi, size=self.rff_dim)

        return W, b

    def _get_rffs(self, X, return_vars=False):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            W, b = self._init_rff_vars()

        B = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1.0 / np.sqrt(self.rff_dim)
        Z = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)

        if return_vars:
            return Z.T, W, b
        return Z.T

    @staticmethod
    def get_random_vectors(n_vectors, n_dims, batch_size, fixed):
        if not fixed:
            return torch.randn(batch_size, n_vectors, n_dims)
        else:
            vectors = []
            for i in range(n_vectors):
                torch.manual_seed(i)
                vectors.append(torch.randn(1, n_dims))
            vectors = torch.cat(vectors, axis=0)
            return vectors.unsqueeze(0).repeat(batch_size, 1, 1)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FourierSeriesV2Multitask(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hold_out_freq=None,
        all_S_list=None,
        training_S_list=None, # list of S's to train on
        OOD_S_list=None,
        variant="",
        sizeOfS=0,
        sizeOfK=0,
        max_frequency=20,
        min_frequency=1,
        L=5,
        standardize=False,
        each_function_different_S=False, # whether to use a different S per function in batch or form each function in a batch using the same S
    ):
        super(FourierSeriesV2Multitask, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1

        self.scale = scale
        self.max_frequency = max_frequency
        self.L = L
        self.standardize = standardize
        self.training_S_list = training_S_list
        self.OOD_S_list = OOD_S_list
        self.all_S_list = all_S_list
        self.variant = variant
        self.sizeOfS = sizeOfS
        self.sizeOfK = sizeOfK
        self.each_function_different_S = each_function_different_S

        if pool_dict is None and seeds is None:
            if self.variant == "fixedK":
                if self.each_function_different_S:
                    chosenSindices = torch.randint(0, self.sizeOfK, size=(self.b_size,))
                    self.selected_fourier_freqs = self.training_S_list[chosenSindices] # (B, |S|)
                else:
                    # Each batch is task pure, i.e. each batch has functions formed using a single S
                    chosenSindex = random.randint(0, self.sizeOfK-1)
                    self.selected_fourier_freqs = self.training_S_list[chosenSindex].repeat(self.b_size, 1) # (B, |S|)
            elif self.variant == "randomS":
                # each f in every batch is formed by a completely random set S (with replacement); There is no notion of K here. Just that we sample batch size-many different S's and create f's (one per S).
                # Refrence for numpy.random.choice equivalent in pytorch: https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
                self.selected_fourier_freqs = self.training_S_list[torch.randint(len(self.training_S_list), (self.b_size,))] # B, |S|
            else:
                raise NotImplementedError()

            coefs = torch.randn(self.b_size, 2 * self.sizeOfS + 1) # (B, 2*|S|+1)
            self.icpt_coefs = coefs[:, :1] # (B, 1) -- intercept term's weight
            self.a_coefs = coefs[:, 1:self.sizeOfS + 1] # (B, |S|) -- intercept and cosine terms' weights
            self.b_coefs = coefs[:, self.sizeOfS + 1 :] # (B, |S|) -- sine terms' weights
        else:
            raise NotImplementedError()

    def evaluate_assist(self, xs_b):
        # xs_b -- B, P, 1
        device = xs_b.device

        icpt_coefs = self.icpt_coefs.to(device) # (B, 1)
        a_coefs = self.a_coefs.to(device) # (B, |S|)
        b_coefs = self.b_coefs.to(device) # (B, |S|)

        # cosine terms
        cosine_terms = a_coefs.unsqueeze(1) * ( # B, 1, |S|
            torch.cos(
                (math.pi / (self.L))
                * self.selected_fourier_freqs.unsqueeze(1).to(device) # B, 1, |S|
                # * torch.arange(self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b # B, P, 1
            ) # B, P, |S|
        ) # B, P, |S|

        # sine terms
        sine_terms = b_coefs.unsqueeze(1) * ( # B, 1, |S|
            torch.sin(
                (math.pi / (self.L))
                * self.selected_fourier_freqs.unsqueeze(1).to(device) # B, 1, |S|
                # * torch.arange(self.max_frequency + 1).unsqueeze(0).to(device)
                * xs_b # B, P, 1
            ) # B, P, |S|
        ) # B, P, |S|

        all_terms_sum = (icpt_coefs # (B, 1)
                        + sine_terms.sum(axis=-1)  # (B, P)
                        + cosine_terms.sum(axis=-1)) # (B, P)

        if self.standardize:
            return (
                self.scale * all_terms_sum / math.sqrt(self.sizeOfS)
            )
        return self.scale * all_terms_sum

    def evaluate(self, xs_b, mode="train", chosenSindices=None):
        if self.variant in ["randomS"]: # train as well as evaluate on a batch with polynomials formed using random S
            pass
        elif self.variant == "fixedK":
            if mode == "train": # create an S-pure batch (i.e. choose an S from K and form the batch)
                pass
            elif mode == "eval": # create a batch with S's at indices=chosenSindices or with equal number of polynomials formed using each S in K
                if chosenSindices is None:
                    # choose batch size many indices in range [0, sizeOfK-1] with replacement
                    chosenSindices = torch.randint(0, self.sizeOfK, (self.b_size,))
                self.selected_fourier_freqs = self.training_S_list[chosenSindices] # (B, |S|)
            else:
                raise NotImplementedError

        return self.evaluate_assist(xs_b)

    def evaluate_ood(self, xs_b, mode="strict", provided_S_list=None):
        # Modes:
            # random - eval on a batch of functions formed from random S's (in-distribution and OOD both)
            # strict - eval on a batch of functions formed from strictly OOD S's
            # use_provided - eval on a batch of functions formed from the provided set of S's
        if len(self.OOD_S_list) == 0:
            mode = "random"

        if mode == "use_provided":
            # sample S's from the provided list of S's and use it to construct the batch
            assert provided_S_list is not None, "When mode is 'use_provided', provided_S_list should not be None."
            chosenSindices = torch.randint(0, len(provided_S_list), (self.b_size,))
            self.selected_fourier_freqs = provided_S_list[chosenSindices] # (B, |S|)

        elif mode == "strict" and self.variant == "fixedK":
            chosenSindices = torch.randint(0, len(self.OOD_S_list), (self.b_size,))
            self.selected_fourier_freqs = self.OOD_S_list[chosenSindices] # (B, |S|)

        elif mode == "random":
            chosenSindices = torch.randint(0, len(self.all_S_list), (self.b_size,))
            self.selected_fourier_freqs = self.all_S_list[chosenSindices] # (B, |S|)

        return self.evaluate_assist(xs_b)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class HaarWavelets(Task):
    def mother_wavelet(self, t):
        if 0<=t<0.5:
            return 1
        elif 0.5<=t<1:
            return -1
        else:
            return 0

    def psi_nk(self, n,k,t):
        return 2**(n/2)*self.mother_wavelet((2**n)*t-k)

    def id(self, t):
        return t

    def haar_basis(self, max_level=3):
        """
            max_level is the max value of n
            Defining a haar basis on [0, 1], conditioned on max value of n, as given here: https://en.wikipedia.org/wiki/Haar_wavelet#Haar_system_on_the_unit_interval_and_related_systems
        """
        basis = [] # basis containing the constant function 1
        for n in range(max_level+1):
            for k in range(2**n):
                basis.append(partial(self.psi_nk,n,k))
        
        basis.append(self.id)
        assert len(basis) == 2**(max_level+1)
        return basis

    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_level=3,
        vectorized_basis=None,
        normalize_outputs=True
    ):
        
        super(HaarWavelets, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1
        
        self.scale = scale
        self.max_level = max_level
        self.vectorized_basis = vectorized_basis
        self.normalize_outputs = normalize_outputs
        
        if pool_dict is None and seeds is None:  
            self.coefs = torch.randn(self.b_size, 2 ** (self.max_level + 1))
        
        elif seeds is not None: 
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def evaluate(self, xs_b):
        device = xs_b.device
        # print(xs_b.device)
        # print(xs_b.dtype)

        # batch_size = xs_b.shape[0]
        # n_points=xs_b.shape[1]
        # n_dims = xs_b.shape[2]
        # assert n_dims == 1
        # coefs_np = coefs.cpu().detach().numpy()
        coefs = self.coefs.to(device)
        xs_b_np = xs_b.cpu().detach().numpy()

        ys_b = []
        for b in range(self.b_size):
            # evaluate the basis at points to get n_points values for each basis element
            eval_basis_pts = torch.from_numpy(np.array([self.vectorized_basis[i](xs_b_np[b].squeeze(axis=-1)) for i in range(len(self.vectorized_basis))], dtype=np.float32)).to(device) # len(vectorized_basis) x n_points
            # print("eval basis pts", eval_basis_pts.shape)
            # scale the basis
            eval_basis_pts_coefs = coefs[b].unsqueeze(dim=-1) * eval_basis_pts # len(vectorized_basis) x n_points
            # print("eval basis pts coefs", eval_basis_pts_coefs.shape)
            eval_pts = eval_basis_pts_coefs.sum(dim=0) # n_points
            # print("eval pts", eval_pts.shape)


            ys_b.append(eval_pts) # n_points sized vector

        # print(len(ys_b))
        # print(ys_b)
        ys_b = torch.stack(ys_b, dim=0).to(device)
        # print(ys_b.device)
        # print(ys_b.dtype)
        # print(ys_b.shape)
        if self.normalize_outputs:
            ys_b = ys_b / 2**(self.max_level + 1)
        return ys_b


    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error    

