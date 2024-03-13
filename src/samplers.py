import math
import time

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform": UniformSampler
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False, seed = None):
    n_dims = len(eigenvalues)
    if seed is None:
        random_matrix = torch.randn(n_dims, n_dims)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        random_matrix = torch.randn(n_dims, n_dims, generator=generator)
    U, _, _ = torch.linalg.svd(random_matrix)
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

def sample_scale(method, n_dims, normalize=False, seed = None):
    if method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True, seed = seed)
        return scale
    return None


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, positive_orthant=False, data_scale=None, data_seed=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.data_scale = data_scale
        self.positive_orthant = positive_orthant
        if data_seed is None:
            data_seed = int(123456789123 + time.time())
        self.data_seed = data_seed
        self.data_rand_gen = torch.Generator().manual_seed(self.data_seed)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims, generator=self.data_rand_gen)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)

        if self.positive_orthant:
            # make all the coordinates positive
            xs_b = torch.abs(xs_b)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.data_scale is not None:
            xs_b *= self.data_scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class UniformSampler(DataSampler):
    
    def __init__(self, n_dims, a = 0, b = 1, bias=None, scale=None, positive_orthant=False):
        super().__init__(n_dims)
        self.a = a
        self.b = b
        self.bias = bias
        self.scale = scale
        self.positive_orthant = positive_orthant
        
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        
        if seeds is None:
            xs_b = torch.rand(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if self.positive_orthant:
            # make all the coordinates positive
            xs_b = torch.abs(xs_b)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        xs_b = (self.b - self.a) * xs_b + self.a
        return xs_b