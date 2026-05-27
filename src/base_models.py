import torch
from torch import nn
from torch.nn import init


class TwoLayerNeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(TwoLayerNeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ThreeLayerNeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(ThreeLayerNeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):
        super(ParallelNetworks, self).__init__()
        self.num_models = num_models
        self.model_class = model_class
        self.model_class_init_args = model_class_init_args
        self._vectorized = model_class in (TwoLayerNeuralNetwork, ThreeLayerNeuralNetwork)

        if self._vectorized:
            in_size = model_class_init_args.get("in_size", 50)
            hidden_size = model_class_init_args.get("hidden_size", 1000)
            out_size = model_class_init_args.get("out_size", 1)

            self.w1 = nn.Parameter(torch.empty(num_models, in_size, hidden_size))
            self.b1 = nn.Parameter(torch.empty(num_models, hidden_size))
            self.w2 = nn.Parameter(torch.empty(num_models, hidden_size, hidden_size if model_class is ThreeLayerNeuralNetwork else out_size))
            self.b2 = nn.Parameter(torch.empty(num_models, hidden_size if model_class is ThreeLayerNeuralNetwork else out_size))
            if model_class is ThreeLayerNeuralNetwork:
                self.w3 = nn.Parameter(torch.empty(num_models, hidden_size, out_size))
                self.b3 = nn.Parameter(torch.empty(num_models, out_size))
            self.reset_parameters()
        else:
            self.nets = nn.ModuleList(
                [model_class(**model_class_init_args) for i in range(num_models)]
            )

    def _reset_linear_parameters(self, weight, bias):
        for i in range(weight.shape[0]):
            weight_i = weight[i].T
            init.kaiming_uniform_(weight_i, a=5**0.5)
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_i)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            init.uniform_(bias[i], -bound, bound)

    def reset_parameters(self):
        self._reset_linear_parameters(self.w1, self.b1)
        self._reset_linear_parameters(self.w2, self.b2)
        if self.model_class is ThreeLayerNeuralNetwork:
            self._reset_linear_parameters(self.w3, self.b3)

    def forward(self, xs):
        if self._vectorized:
            assert xs.shape[0] == self.num_models
            out = torch.bmm(xs, self.w1) + self.b1[:, None, :]
            out = torch.relu(out)
            out = torch.bmm(out, self.w2) + self.b2[:, None, :]
            if self.model_class is ThreeLayerNeuralNetwork:
                out = torch.relu(out)
                out = torch.bmm(out, self.w3) + self.b3[:, None, :]
            return out

        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs

