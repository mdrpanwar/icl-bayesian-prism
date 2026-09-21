import os
import random
import sys
import unittest

import torch
from munch import Munch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from meta_utils import (
    configure_inner_lrs,
    get_inner_lrs,
    inner_adapt,
    inner_lr_stats,
    trainable_model_params,
)
import meta_eval
import train_meta
from train import train_step
from train_meta import (
    get_support_size_choices,
    apply_outer_update,
    max_support_size_for_step,
    meta_train_step,
    sample_support_sizes,
    validate_meta_attention_backend,
)


class ToySequenceRegressor(torch.nn.Module):
    def __init__(self, weight=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[weight]], dtype=torch.float32))

    def forward(self, xs, ys, inds=None):
        del ys
        preds = (xs @ self.weight).squeeze(-1)
        if inds is not None:
            return preds[:, inds]
        return preds


class LengthSensitiveRegressor(torch.nn.Module):
    def __init__(self, weight=0.0, seq_offset=10.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[weight]], dtype=torch.float32))
        self.seq_offset = seq_offset

    def forward(self, xs, ys, inds=None):
        del ys
        seq_len = xs.shape[1]
        preds = (xs @ self.weight).squeeze(-1) + self.seq_offset * (seq_len - 1)
        if inds is not None:
            return preds[:, inds]
        return preds


class ToyLayeredRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = torch.nn.Module()
        self._backbone.h = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.Linear(1, 1, bias=False),
                        "mlp": torch.nn.Linear(1, 1, bias=False),
                        "ln_1": torch.nn.LayerNorm(1),
                    }
                )
            ]
        )
        self._read_out = torch.nn.Linear(1, 1, bias=False)

    def forward(self, xs, ys, inds=None):
        del ys
        h = self._backbone.h[0].attn(xs)
        h = self._backbone.h[0].mlp(h)
        h = self._backbone.h[0].ln_1(h)
        preds = self._read_out(h).squeeze(-1)
        if inds is not None:
            return preds[:, inds]
        return preds


class ConstantDataSampler:
    def sample_xs(self, n_points, batch_size):
        return torch.ones(batch_size, n_points, 1)


class ConstantTask:
    def evaluate(self, xs):
        return torch.full(xs.shape[:2], 2.0)


def constant_task_sampler():
    return ConstantTask()


def meta_args(**overrides):
    args = Munch(
        inner_lr=0.1,
        inner_lr_mode="fixed",
        num_inner_steps=1,
        first_order=False,
        meta_batch_size=2,
        num_query_points=1,
        vary_support_size="match_curriculum",
        fixed_support_size=None,
        support_size_choices=None,
        multi_k_support=False,
    )
    args.update(overrides)
    return args


class TrainMetaTests(unittest.TestCase):
    def test_discrete_support_choices(self):
        args = meta_args(
            vary_support_size="choices",
            support_size_choices="5,10,15",
        )
        random.seed(7)
        samples = [
            sample_support_sizes(args, curriculum_n_points=3, curriculum_end=41)
            for _ in range(100)
        ]
        observed = {sample[0] for sample in samples}
        self.assertEqual(get_support_size_choices(args), [5, 10, 15])
        self.assertEqual(observed, {5, 10, 15})
        self.assertEqual(max_support_size_for_step(args, 3, 41), 15)

    def test_joint_icl_position_set_averages_all_positions_per_sequence(self):
        model = ToySequenceRegressor(weight=0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        xs = torch.ones(1, 3, 1)
        ys = torch.tensor([[1.0, 100.0, 3.0]])

        loss, _ = train_step(
            model, xs, ys, optimizer,
            lambda pred, target: ((pred - target) ** 2).mean(),
            batch_idx=0, max_train_steps=1, loss_position_set="0,2",
        )

        self.assertAlmostEqual(loss, 5.0)

    def test_all_choices_parallel_returns_every_configured_support_size(self):
        args = meta_args(
            vary_support_size="choices_parallel",
            support_size_choices="15,5,10",
        )
        self.assertEqual(
            sample_support_sizes(args, curriculum_n_points=3, curriculum_end=41),
            [5, 10, 15],
        )
        self.assertEqual(max_support_size_for_step(args, 3, 41), 15)

    def test_all_choices_sequential_carries_fast_parameters_between_stages(self):
        model = ToySequenceRegressor(weight=0.0)
        args = meta_args(
            vary_support_size="choices_sequential",
            support_size_choices="1,3",
            inner_lr=0.1,
            num_inner_steps=1,
            num_query_points=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        xs = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        ys = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])

        loss, ks, per_task_losses = meta_train_step(
            model, optimizer, None, xs, ys, args,
            curriculum_n_points=3, curriculum_end=3,
            loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
            n_query=2,
        )

        query_x = torch.tensor([4.0, 5.0])
        query_y = torch.tensor([8.0, 10.0])
        weight_after_1 = 0.4
        weight_after_3 = 142.0 / 75.0
        expected = torch.stack([
            ((weight_after_1 * query_x - query_y) ** 2).mean(),
            ((weight_after_3 * query_x - query_y) ** 2).mean(),
        ]).mean()
        self.assertEqual(ks, [1, 3])
        self.assertTrue(torch.allclose(torch.tensor(loss), expected))
        self.assertTrue(torch.allclose(torch.tensor(per_task_losses[0]), expected))

    def test_sdpa_first_order_validation_is_explicit_and_non_mutating(self):
        args = Munch(
            meta=Munch(first_order=True),
            model=Munch(attn_implementation="sdpa"),
        )

        with self.assertRaisesRegex(ValueError, "functorch vmap"):
            validate_meta_attention_backend(args)
        self.assertEqual(args.model.attn_implementation, "sdpa")

        eager_args = Munch(
            meta=Munch(first_order=True),
            model=Munch(attn_implementation="eager"),
        )
        validate_meta_attention_backend(eager_args)
        self.assertEqual(eager_args.model.attn_implementation, "eager")

    def test_multi_k_support_samples_zero_random_ranges_and_current_max(self):
        args = meta_args(multi_k_support=True)
        random.seed(0)

        samples = [
            sample_support_sizes(args, curriculum_n_points=13, curriculum_end=41)
            for _ in range(20)
        ]

        self.assertTrue(all(len(ks) == 4 for ks in samples))
        self.assertTrue(all(ks[0] == 0 for ks in samples))
        self.assertTrue(all(ks[-1] == 12 for ks in samples))
        self.assertGreater(len({ks[1] for ks in samples}), 1)
        self.assertGreater(len({ks[2] for ks in samples}), 1)

    def test_single_k_support_preserves_existing_sampling_mode(self):
        args = meta_args(multi_k_support=False)
        random.seed(1)

        samples = [
            sample_support_sizes(args, curriculum_n_points=8, curriculum_end=41)
            for _ in range(20)
        ]

        self.assertTrue(all(len(ks) == 1 for ks in samples))
        self.assertTrue(all(0 <= ks[0] <= 7 for ks in samples))
        self.assertGreater(len({ks[0] for ks in samples}), 1)

    def test_fixed_support_size_overrides_multi_k_support(self):
        args = meta_args(
            vary_support_size="fixed",
            fixed_support_size=25,
            multi_k_support=True,
        )

        samples = [
            sample_support_sizes(args, curriculum_n_points=8, curriculum_end=41)
            for _ in range(20)
        ]

        self.assertEqual(samples, [[25]] * 20)
        self.assertEqual(
            max_support_size_for_step(args, curriculum_n_points=8, curriculum_end=41),
            25,
        )

    def test_fixed_support_size_requires_explicit_k(self):
        args = meta_args(vary_support_size="fixed", fixed_support_size=None)

        with self.assertRaisesRegex(ValueError, "fixed_support_size must be set"):
            sample_support_sizes(args, curriculum_n_points=8, curriculum_end=41)

    def test_fixed_support_size_can_exceed_current_curriculum_points(self):
        model = ToySequenceRegressor(weight=0.0)
        args = meta_args(
            vary_support_size="fixed",
            fixed_support_size=3,
            multi_k_support=True,
            num_inner_steps=1,
            num_query_points=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        xs = torch.ones(1, 5, 1)
        ys = torch.full((1, 5), 2.0)

        loss, ks, per_task_losses = meta_train_step(
            model=model,
            optimizer=optimizer,
            task=None,
            xs=xs,
            ys=ys,
            meta_args=args,
            curriculum_n_points=2,
            curriculum_end=5,
            loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
            n_query=2,
        )

        self.assertTrue(loss > 0)
        self.assertEqual(ks, [3])
        self.assertEqual(len(per_task_losses), 1)

    def test_learned_per_param_inner_lr_changes_inner_update_and_gets_gradient(self):
        model = ToySequenceRegressor(weight=1.0)
        args = meta_args(inner_lr=0.5, inner_lr_mode="learned_per_param")
        configure_inner_lrs(model, args)

        params = trainable_model_params(model)
        inner_lrs = get_inner_lrs(model, args)

        xs_support = torch.tensor([[[1.0]]])
        ys_support = torch.tensor([[3.0]])
        fast_params = inner_adapt(
            model,
            params,
            xs_support,
            ys_support,
            inner_lrs,
            num_inner_steps=1,
            first_order=False,
            loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
        )

        self.assertTrue(torch.allclose(fast_params["weight"], torch.tensor([[3.0]])))

        xs_query = torch.tensor([[[1.0]]])
        ys_query = torch.tensor([[4.0]])
        pred = torch.func.functional_call(model, fast_params, (xs_query, ys_query))
        loss = ((pred - ys_query) ** 2).mean()
        loss.backward()

        self.assertIsNotNone(model.maml_inner_lrs["weight"].grad)
        self.assertNotEqual(float(model.maml_inner_lrs["weight"].grad.item()), 0.0)

    def test_bounded_signed_inner_lrs_start_at_requested_value_and_stay_bounded(self):
        model = ToySequenceRegressor(weight=1.0)
        args = meta_args(
            inner_lr=0.002,
            inner_lr_mode="learned_per_param",
            inner_lr_parameterization="bounded_signed",
            inner_lr_bound=0.05,
        )
        configure_inner_lrs(model, args)

        effective = get_inner_lrs(model, args)["weight"]
        self.assertTrue(torch.allclose(effective, torch.full_like(effective, 0.002)))

        effective.sum().backward()
        raw = model.maml_inner_lrs["weight"]
        self.assertIsNotNone(raw.grad)
        self.assertNotEqual(float(raw.grad.item()), 0.0)

        with torch.no_grad():
            raw.fill_(100.0)
        bounded = get_inner_lrs(model, args)["weight"]
        self.assertLessEqual(float(bounded.abs().max()), 0.0500001)

        stats = inner_lr_stats(model, args)
        self.assertIn("inner_lr/raw_global/abs_max", stats)
        self.assertLessEqual(stats["inner_lr/global/abs_max"], 0.0500001)

    def test_bounded_signed_inner_lrs_validate_initial_value(self):
        model = ToySequenceRegressor(weight=1.0)
        args = meta_args(
            inner_lr=0.05,
            inner_lr_mode="learned_per_param",
            inner_lr_parameterization="bounded_signed",
            inner_lr_bound=0.05,
        )
        with self.assertRaisesRegex(ValueError, "abs.inner_lr"):
            configure_inner_lrs(model, args)

    def test_outer_update_clips_gradients_and_reports_preclip_norm(self):
        model = ToySequenceRegressor(weight=10.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        args = meta_args(outer_grad_clip_norm=1.0, fail_on_nonfinite=True)
        loss = (100.0 * model.weight).square().sum()

        stats = apply_outer_update(model, optimizer, loss, args)

        self.assertGreater(stats["outer_grad_norm"], 1.0)
        self.assertEqual(stats["outer_grad_was_clipped"], 1.0)
        self.assertLessEqual(float(model.weight.grad.norm()), 1.000001)

    def test_outer_update_rejects_nonfinite_loss_before_optimizer_step(self):
        model = ToySequenceRegressor(weight=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        args = meta_args(outer_grad_clip_norm=1.0, fail_on_nonfinite=True)
        before = model.weight.detach().clone()
        loss = model.weight.sum() * torch.tensor(float("nan"))

        with self.assertRaisesRegex(FloatingPointError, "Non-finite meta loss"):
            apply_outer_update(model, optimizer, loss, args)

        self.assertTrue(torch.equal(model.weight.detach(), before))

    def test_outer_guard_skips_catastrophic_preclip_norm(self):
        model = ToySequenceRegressor(weight=10.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        args = meta_args(
            outer_grad_clip_norm=1.0,
            max_outer_grad_norm_before_skip=100.0,
            fail_on_nonfinite=True,
        )
        before = model.weight.detach().clone()
        loss = (100.0 * model.weight).square().sum()
        stats = apply_outer_update(model, optimizer, loss, args)
        self.assertEqual(stats["outer_update_skipped"], 1.0)
        self.assertGreater(stats["outer_grad_norm"], 100.0)
        self.assertTrue(torch.equal(model.weight.detach(), before))

    def test_outer_guard_skips_nonfinite_loss(self):
        model = ToySequenceRegressor(weight=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        args = meta_args(
            outer_grad_clip_norm=1.0,
            max_outer_grad_norm_before_skip=100.0,
            fail_on_nonfinite=True,
        )
        loss = model.weight.sum() * torch.tensor(float("nan"))
        stats = apply_outer_update(model, optimizer, loss, args)
        self.assertEqual(stats["outer_update_skipped"], 1.0)
        self.assertEqual(len(optimizer.state), 0)

    def test_inner_lr_stats_include_global_layer_and_layer_module_keys(self):
        model = ToyLayeredRegressor()
        args = meta_args(inner_lr=0.1, inner_lr_mode="learned_per_param")
        configure_inner_lrs(model, args)

        stats = inner_lr_stats(
            model,
            args,
            include_layer=True,
            include_layer_module=True,
        )

        self.assertIn("inner_lr/global/mean", stats)
        self.assertIn("inner_lr/global/std", stats)
        self.assertIn("inner_lr/global/neg_frac", stats)
        self.assertIn("inner_lr/layers/layer_00/mean", stats)
        self.assertIn("inner_lr/layers/read_out/mean", stats)
        self.assertIn("inner_lr/layer_modules/layer_00/attn/mean", stats)
        self.assertIn("inner_lr/layer_modules/layer_00/mlp/mean", stats)
        self.assertIn("inner_lr/layer_modules/layer_00/ln/mean", stats)
        self.assertIn("inner_lr/layer_modules/read_out/read_out/mean", stats)
        self.assertAlmostEqual(stats["inner_lr/global/mean"], 0.1, places=6)

    def test_meta_train_step_updates_learned_inner_lr(self):
        torch.manual_seed(0)
        random.seed(0)
        model = ToySequenceRegressor(weight=0.0)
        args = meta_args(
            inner_lr=0.1,
            inner_lr_mode="learned_per_param",
            multi_k_support=True,
            num_inner_steps=1,
            num_query_points=1,
        )
        configure_inner_lrs(model, args)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        xs = torch.ones(2, 5, 1)
        ys = torch.full((2, 5), 2.0)
        before = model.maml_inner_lrs["weight"].detach().clone()

        loss, ks, per_task_losses = meta_train_step(
            model=model,
            optimizer=optimizer,
            task=None,
            xs=xs,
            ys=ys,
            meta_args=args,
            curriculum_n_points=4,
            curriculum_end=4,
            loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
            n_query=1,
        )

        self.assertTrue(loss > 0)
        self.assertEqual(len(ks), 8)
        self.assertEqual(len(per_task_losses), 2)
        self.assertFalse(torch.allclose(model.maml_inner_lrs["weight"], before))

    def test_multi_k_branches_share_task_examples_and_queries_but_adapt_independently(self):
        model = ToySequenceRegressor(weight=0.0)
        args = meta_args(
            inner_lr=0.1,
            inner_lr_mode="fixed",
            multi_k_support=True,
            num_inner_steps=1,
            num_query_points=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        xs = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        ys = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])

        original_sampler = train_meta.sample_support_sizes
        train_meta.sample_support_sizes = lambda *args, **kwargs: [0, 1, 3]
        try:
            loss, ks, per_task_losses = meta_train_step(
                model=model,
                optimizer=optimizer,
                task=None,
                xs=xs,
                ys=ys,
                meta_args=args,
                curriculum_n_points=3,
                curriculum_end=3,
                loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
                n_query=2,
            )
        finally:
            train_meta.sample_support_sizes = original_sampler

        # All branches share the same task row and query block x=[4, 5],
        # y=[8, 10], but their fast weights are independent:
        # k=0: w'=0
        # k=1: grad on (1,2) is -4, so w'=0.4
        # k=3: grad mean over (1,2),(2,4),(3,6) is -56/3, so w'=28/15.
        query_x = torch.tensor([4.0, 5.0])
        query_y = torch.tensor([8.0, 10.0])
        expected_branch_losses = torch.tensor(
            [
                ((0.0 * query_x - query_y) ** 2).mean(),
                ((0.4 * query_x - query_y) ** 2).mean(),
                (((28.0 / 15.0) * query_x - query_y) ** 2).mean(),
            ]
        )
        expected = expected_branch_losses.mean()

        self.assertEqual(ks, [0, 1, 3])
        self.assertTrue(torch.allclose(torch.tensor(loss), expected))
        self.assertTrue(torch.allclose(torch.tensor(per_task_losses[0]), expected))

    def test_eval_adapts_on_support_examples_as_independent_batch_elements(self):
        model = LengthSensitiveRegressor(weight=0.0, seq_offset=10.0)
        original_data_sampler = meta_eval.get_data_sampler
        original_task_sampler = meta_eval.get_task_sampler
        meta_eval.get_data_sampler = lambda *args, **kwargs: ConstantDataSampler()
        meta_eval.get_task_sampler = lambda *args, **kwargs: constant_task_sampler

        try:
            metrics_loop = meta_eval.collect_maml_metrics_loop(
                model=model,
                task_name="toy",
                data_name="toy",
                n_dims=1,
                n_points=3,
                inner_lr=0.1,
                num_inner_steps=1,
                num_eval_examples=1,
                batch_size=1,
                stride=1,
            )
            metrics_vmap = meta_eval.collect_maml_metrics(
                model=model,
                task_name="toy",
                data_name="toy",
                n_dims=1,
                n_points=3,
                inner_lr=0.1,
                num_inner_steps=1,
                num_eval_examples=1,
                batch_size=1,
                stride=1,
            )
        finally:
            meta_eval.get_data_sampler = original_data_sampler
            meta_eval.get_task_sampler = original_task_sampler

        # Independent support examples have seq_len=1 in the support loss:
        #   w' = 0 - 0.1 * 2 * (0 - 2) = 0.4
        #   query loss = (0.4 - 2)^2 = 2.56
        # If support were fed as one sequence of length 2, the model's
        # seq_offset would produce a very different loss.
        expected_loss = torch.tensor(2.56)
        self.assertTrue(torch.allclose(metrics_loop[0, 2], expected_loss))
        self.assertTrue(torch.allclose(metrics_vmap[0, 2], expected_loss))
        self.assertFalse(torch.isnan(metrics_vmap[0]).any())


if __name__ == "__main__":
    unittest.main()
