import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import TransformerModel


class AttentionModeTests(unittest.TestCase):
    def test_prefix_is_bidirectional_and_suffix_remains_causal(self):
        model = TransformerModel(
            n_dims=3,
            n_positions=5,
            n_embd=12,
            n_layer=1,
            n_head=3,
            pos_encode=False,
            attn_implementation="eager",
            attention_mode="prefix_bidirectional",
            prefix_condition_points=2,
        ).eval()
        mask = model._backbone.h[0].attn.bias[0, 0]

        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[:4, 4:].any())
        self.assertTrue(mask[4, :5].all())
        self.assertFalse(mask[4, 5:].any())
        self.assertTrue(
            torch.equal(mask[5:], torch.tril(torch.ones_like(mask))[5:])
        )

        xs = torch.randn(2, 5, 3)
        ys = torch.randn(2, 5)
        with torch.no_grad():
            prediction = model(xs, ys)[:, 2].clone()
            ys[:, 2] += 1000
            changed_label_prediction = model(xs, ys)[:, 2]
        self.assertTrue(torch.equal(prediction, changed_label_prediction))

    def test_query_isolation_blocks_inter_query_attention(self):
        model = TransformerModel(
            n_dims=3,
            n_positions=5,
            n_embd=12,
            n_layer=1,
            n_head=3,
            pos_encode=False,
            attn_implementation="eager",
            attention_mode="causal",
            prefix_condition_points=2,
            isolate_query_points=True,
        ).eval()
        mask = model._backbone.h[0].attn.bias[0, 0]

        causal = torch.tril(torch.ones_like(mask))
        self.assertTrue(torch.equal(mask[:4], causal[:4]))
        for point in range(2, 5):
            x_token = 2 * point
            y_token = x_token + 1
            expected_x = torch.zeros(10, dtype=torch.bool)
            expected_x[:4] = True
            expected_x[x_token] = True
            expected_y = expected_x.clone()
            expected_y[y_token] = True
            self.assertTrue(torch.equal(mask[x_token], expected_x))
            self.assertTrue(torch.equal(mask[y_token], expected_y))

        xs = torch.randn(2, 5, 3)
        ys = torch.randn(2, 5)
        with torch.no_grad():
            later_prediction = model(xs, ys)[:, 4].clone()
            ys[:, 2:4] += 1000
            perturbed_prediction = model(xs, ys)[:, 4]
        self.assertTrue(torch.allclose(later_prediction, perturbed_prediction))

    def test_isolated_queries_allow_bidirectional_support_only(self):
        model = TransformerModel(
            n_dims=3,
            n_positions=5,
            n_embd=12,
            n_layer=1,
            n_head=3,
            pos_encode=False,
            attn_implementation="eager",
            attention_mode="prefix_bidirectional",
            prefix_condition_points=2,
            isolate_query_points=True,
        )
        mask = model._backbone.h[0].attn.bias[0, 0]
        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[:4, 4:].any())
        self.assertTrue(mask[4, :4].all())
        self.assertFalse(mask[4, 5:].any())

    def test_query_isolation_requires_eager_attention(self):
        with self.assertRaisesRegex(ValueError, "requires.*eager"):
            TransformerModel(
                n_dims=3,
                n_positions=5,
                n_embd=12,
                n_layer=1,
                n_head=3,
                attn_implementation="sdpa",
                attention_mode="causal",
                prefix_condition_points=2,
                isolate_query_points=True,
            )


    def test_prefix_mode_requires_eager_attention(self):
        with self.assertRaisesRegex(ValueError, "requires.*eager"):
            TransformerModel(
                n_dims=3,
                n_positions=5,
                n_embd=12,
                n_layer=1,
                n_head=3,
                attn_implementation="sdpa",
                attention_mode="prefix_bidirectional",
                prefix_condition_points=2,
            )


if __name__ == "__main__":
    unittest.main()
