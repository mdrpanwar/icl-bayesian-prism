import os
import sys
import unittest
from unittest.mock import patch

from munch import Munch
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import eval as eval_module
from meta_utils import configure_inner_lrs


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, xs, ys):
        return xs[..., 0] * self.weight


class EvalMetaLearningRateTests(unittest.TestCase):
    def test_bounded_inner_rates_are_reconstructed_for_offline_eval(self):
        model = ToyModel()
        config = Munch(
            inner_lr=0.002,
            inner_lr_mode="learned_per_param",
            inner_lr_parameterization="bounded_signed",
            inner_lr_bound=0.05,
        )
        configure_inner_lrs(model, config)
        with torch.no_grad():
            model.maml_inner_lrs["weight"].fill_(100.0)

        kwargs = {
            "eval_mode": "maml",
            "prompting_strategy": "standard",
            "inner_lr": 0.002,
            "inner_lr_mode": "learned_per_param",
            "inner_lr_parameterization": "bounded_signed",
            "inner_lr_bound": 0.05,
            "num_inner_steps": 5,
            "stride": 1,
        }
        sentinel = object()
        with patch.object(eval_module, "eval_model_maml", return_value=sentinel) as mocked:
            result = eval_module.compute_eval_metrics(model, kwargs)

        self.assertIs(result, sentinel)
        passed = mocked.call_args.kwargs
        self.assertNotIn("inner_lr_parameterization", passed)
        effective = passed["inner_lrs"]["weight"]
        self.assertLessEqual(float(effective.abs().max()), 0.0500001)
        self.assertGreater(float(effective.item()), 0.049)

    def test_build_evals_preserves_meta_learning_rate_parameterization(self):
        conf = Munch.fromDict(
            {
                "model": {"n_dims": 2},
                "training": {
                    "eval_n_points": 3,
                    "batch_size": 4,
                    "task": "linear_regression",
                    "data": "gaussian",
                    "curriculum": {"points": {"end": 3}},
                },
                "meta": {
                    "inner_lr": 0.002,
                    "inner_lr_mode": "learned_per_param",
                    "inner_lr_parameterization": "bounded_signed",
                    "inner_lr_bound": 0.05,
                    "num_inner_steps": 5,
                    "meta_eval_stride": 1,
                },
            }
        )
        maml = eval_module.build_evals(conf)["maml"]
        self.assertEqual(maml["inner_lr_mode"], "learned_per_param")
        self.assertEqual(maml["inner_lr_parameterization"], "bounded_signed")
        self.assertEqual(maml["inner_lr_bound"], 0.05)


if __name__ == "__main__":
    unittest.main()
