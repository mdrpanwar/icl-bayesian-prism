import math
import os
import sys
import unittest

import torch
from munch import Munch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import TransformerModel
from tasks import MultiFunctionLinearRegression, SegmentedPretraining
from train_meta import meta_train_step


class TaggedToyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.1]]))
        self.slot_bias = torch.nn.Parameter(torch.zeros(3))

    def forward(self, xs, ys, inds=None, task_ids=None):
        del ys
        output = (xs @ self.weight).squeeze(-1) + self.slot_bias[task_ids]
        return output if inds is None else output[:, inds]


class EpisodeTaskTests(unittest.TestCase):
    def test_vectorized_maml_accepts_per_example_task_ids(self):
        model = TaggedToyRegressor()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        meta_args = Munch(
            inner_lr=0.01,
            inner_lr_mode="fixed",
            inner_lr_parameterization="direct",
            inner_lr_bound=None,
            outer_grad_clip_norm=None,
            fail_on_nonfinite=True,
            num_inner_steps=1,
            first_order=False,
            meta_batch_size=2,
            num_query_points=2,
            vary_support_size="fixed",
            fixed_support_size=2,
            support_size_choices=None,
            multi_k_support=False,
        )
        xs = torch.randn(2, 4, 1)
        ys = torch.randn(2, 4)
        task_ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        loss, ks, per_task = meta_train_step(
            model,
            optimizer,
            None,
            xs,
            ys,
            meta_args,
            curriculum_n_points=2,
            curriculum_end=2,
            loss_func=lambda pred, target: ((pred - target) ** 2).mean(),
            n_query=2,
            task_ids=task_ids,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(ks, [2, 2])
        self.assertEqual(len(per_task), 2)

    def test_multi_function_layout_and_targets(self):
        torch.manual_seed(3)
        task = MultiFunctionLinearRegression(
            n_dims=2,
            batch_size=4,
            num_functions=3,
            support_per_function=2,
        )
        xs = torch.randn(4, 9, 2)
        ys = task.evaluate(xs)

        self.assertEqual(task.task_ids.shape, (4, 9))
        for row in range(4):
            support_counts = torch.bincount(task.task_ids[row, :6], minlength=3)
            query_counts = torch.bincount(task.task_ids[row, 6:], minlength=3)
            self.assertTrue(torch.equal(support_counts, torch.tensor([2, 2, 2])))
            self.assertTrue(torch.equal(query_counts, torch.tensor([1, 1, 1])))
            selected = task.weights[row, task.task_ids[row]]
            self.assertTrue(torch.allclose(ys[row], (xs[row] * selected).sum(-1)))

    def test_task_slot_model_requires_and_accepts_ids(self):
        model = TransformerModel(
            n_dims=2,
            n_positions=4,
            n_embd=8,
            n_layer=1,
            n_head=2,
            pos_encode=False,
            num_task_slots=3,
        )
        xs = torch.randn(2, 4, 2)
        ys = torch.randn(2, 4)
        with self.assertRaisesRegex(ValueError, "task_ids are required"):
            model(xs, ys)
        output = model(xs, ys, task_ids=torch.zeros(2, 4, dtype=torch.long))
        self.assertEqual(output.shape, (2, 4))

    def test_literal_tag_tokens_have_correct_layout_and_isolation(self):
        model = TransformerModel(
            n_dims=2, n_positions=4, n_embd=8, n_layer=1, n_head=2,
            pos_encode=False, num_task_slots=2, task_tag_mode="prefix_token",
            prefix_condition_points=2, isolate_query_points=True,
        )
        xs = torch.randn(2, 4, 2)
        ys = torch.randn(2, 4)
        ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        output = model(xs, ys, task_ids=ids)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(model._backbone.config.n_positions, 12)
        mask = model._backbone.h[0].attn.bias[0, 0]
        self.assertTrue(bool(mask[7, 6]))  # query x sees its tag
        self.assertFalse(bool(mask[7, 8]))  # query x cannot see its y
        self.assertFalse(bool(mask[10, 6]))  # next query cannot see prior query
        output.sum().backward()
        self.assertIsNotNone(model._task_slot_embeddings.weight.grad)

    def test_literal_tag_tokens_support_vectorized_maml(self):
        model = TransformerModel(
            n_dims=2, n_positions=3, n_embd=8, n_layer=1, n_head=2,
            pos_encode=False, num_task_slots=2, task_tag_mode="prefix_token",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        meta_args = Munch(
            inner_lr=0.01, inner_lr_mode="fixed", num_inner_steps=1,
            first_order=False, meta_batch_size=2, num_query_points=1,
            vary_support_size="fixed", fixed_support_size=2,
            outer_grad_clip_norm=1.0, fail_on_nonfinite=True,
        )
        xs = torch.randn(2, 3, 2)
        ys = torch.randn(2, 3)
        ids = torch.tensor([[0, 1, 0], [1, 0, 1]])
        loss, ks, values = meta_train_step(
            model, optimizer, None, xs, ys, meta_args, 2, 2,
            lambda pred, target: ((pred - target) ** 2).mean(), 1,
            task_ids=ids,
        )
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(ks, [2, 2])
        self.assertEqual(len(values), 2)

    def test_untagged_control_matches_tagged_trainable_parameter_count(self):
        common = dict(
            n_dims=2, n_positions=4, n_embd=8, n_layer=1, n_head=2,
            pos_encode=False, num_task_slots=1,
        )
        tagged = TransformerModel(**common, task_tag_mode="prefix_token")
        untagged = TransformerModel(**common, task_tag_mode="none")
        count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(count(tagged), count(untagged))
        xs = torch.randn(2, 4, 2)
        ys = torch.randn(2, 4)
        ids = torch.zeros(2, 4, dtype=torch.long)
        self.assertEqual(untagged(xs, ys, task_ids=ids).shape, (2, 4))

    def test_segmented_pretraining_metadata_and_finite_targets(self):
        torch.manual_seed(5)
        task = SegmentedPretraining(n_dims=4, batch_size=8)
        xs = torch.randn(8, 17, 4)
        ys = task.evaluate(xs)
        self.assertEqual(ys.shape, (8, 17))
        self.assertTrue(torch.isfinite(ys).all())
        self.assertGreaterEqual(int(task.segment_offsets.min()), 1)
        self.assertLessEqual(int(task.segment_offsets.max()), 5)
        self.assertGreaterEqual(int(task.family_ids.min()), 0)
        self.assertLess(int(task.family_ids.max()), len(task.families))

    def test_segmented_coordinate_offset_is_add_k(self):
        task = SegmentedPretraining(n_dims=4, batch_size=1)
        xs = torch.randn(7, 4)
        torch.manual_seed(19)
        coordinate = int(torch.randint(4, (1,)).item())
        offset = torch.randn(())
        torch.manual_seed(19)
        actual = task._sample_function("coordinate_offset", xs)
        expected = (xs[:, coordinate] + offset) / math.sqrt(2)
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
