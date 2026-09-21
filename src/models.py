import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import numpy as np
import os
import torch.distributions.normal as tdn
from samplers import sample_scale

from base_models import TwoLayerNeuralNetwork, ThreeLayerNeuralNetwork, ParallelNetworks

def build_model(conf):
    if conf.family == "gpt2":
        try:
            model = TransformerModel(
                n_dims=conf.n_dims,
                n_positions=conf.n_positions,
                n_embd=conf.n_embd,
                n_layer=conf.n_layer,
                n_head=conf.n_head,
                pos_encode=conf.pos_encode,
                attn_implementation=conf.attn_implementation,
                attention_mode=getattr(conf, "attention_mode", "causal"),
                prefix_condition_points=getattr(
                    conf, "prefix_condition_points", None
                ),
                isolate_query_points=getattr(
                    conf, "isolate_query_points", False
                ),
                num_task_slots=getattr(conf, "num_task_slots", 0),
                task_tag_mode=getattr(conf, "task_tag_mode", "additive"),
                # resid_pdrop=conf.resid_pdrop,
                # embd_pdrop=conf.embd_pdrop,
                # attn_pdrop=conf.attn_pdrop,
                # use_cache=conf.use_cache
            )
        except AttributeError:
            model = TransformerModel(
                n_dims=conf.n_dims,
                n_positions=conf.n_positions,
                n_embd=conf.n_embd,
                n_layer=conf.n_layer,
                n_head=conf.n_head,
            )
    elif conf.family == "gpt2_task_prefix":
        model = TaskPrefixTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            pos_encode=conf.pos_encode,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "gaussian_mixture_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [(NNModel, {"n_neighbors": 3}), (AveragingModel, {}),],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": TwoLayerNeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    # Composite episode generators need not have a meaningful classical
    # baseline. Returning none keeps final model evaluation from turning a
    # successfully completed training job into a failed job.
    models = [
        model_cls(**kwargs)
        for model_cls, kwargs in task_to_baselines.get(task_name, [])
    ]
    return models


class TransformerModel(nn.Module):
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True
        ,resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
        attn_implementation="eager",
        attention_mode="causal",
        prefix_condition_points=None,
        isolate_query_points=False,
        num_task_slots=0,
        task_tag_mode="additive",
    ):
        super(TransformerModel, self).__init__()
        if task_tag_mode not in {"additive", "prefix_token", "none"}:
            raise ValueError(f"Unknown task_tag_mode: {task_tag_mode}")
        tokens_per_point = 3 if task_tag_mode == "prefix_token" else 2
        configuration = GPT2Config(
            n_positions=tokens_per_point * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
        )
        self.name = (
            f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_pos_encode{pos_encode}"
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._pos_encode = pos_encode
        self._attention_mode = attention_mode
        self._prefix_condition_points = prefix_condition_points
        self._isolate_query_points = isolate_query_points
        self._task_tag_mode = task_tag_mode
        self._tokens_per_point = tokens_per_point
        self._num_task_slots = int(num_task_slots)
        if self._num_task_slots < 0:
            raise ValueError("num_task_slots must be non-negative")
        if task_tag_mode == "prefix_token" and self._num_task_slots == 0:
            raise ValueError("prefix_token requires num_task_slots > 0")
        self._read_in = nn.Linear(n_dims, n_embd)
        self._task_slot_embeddings = (
            nn.Embedding(self._num_task_slots, n_embd)
            if self._num_task_slots
            else None
        )
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        self._apply_pos_encode_freeze()
        self._apply_attention_mask()

    def _apply_pos_encode_freeze(self):
        # When pos_encode=False, we keep the wpe parameter (so the standard
        # GPT2Model.forward path runs unchanged) but pin it to zero and freeze
        # it, which makes `inputs_embeds + wpe(pos)` equivalent to the old
        # GPT2ModelWOPosEncodings behaviour without an overridden forward.
        if not self._pos_encode:
            with torch.no_grad():
                self._backbone.wpe.weight.zero_()
            self._backbone.wpe.weight.requires_grad_(False)

    def _apply_attention_mask(self):
        """Install the configured prefix/query visibility mask."""
        if self._attention_mode not in {"causal", "prefix_bidirectional"}:
            raise ValueError(f"Unknown attention mode: {self._attention_mode!r}")

        max_tokens = self._tokens_per_point * self.n_positions
        mask_device = self._backbone.h[0].attn.bias.device
        mask = torch.tril(
            torch.ones(
                max_tokens, max_tokens, dtype=torch.bool, device=mask_device
            )
        )
        prefix_points = self._prefix_condition_points
        needs_custom_mask = (
            self._attention_mode == "prefix_bidirectional"
            or self._isolate_query_points
        )
        if needs_custom_mask:
            if self._backbone.config._attn_implementation != "eager":
                raise ValueError(
                    "Custom prefix/query attention requires "
                    "model.attn_implementation=eager"
                )
            if prefix_points is None or not 0 < prefix_points < self.n_positions:
                raise ValueError(
                    "prefix_condition_points must be between 1 and "
                    f"n_positions-1, got {prefix_points!r}"
                )

        if self._attention_mode == "prefix_bidirectional":
            prefix_tokens = self._tokens_per_point * prefix_points
            mask[:prefix_tokens, :prefix_tokens] = True

        if self._isolate_query_points:
            prefix_tokens = self._tokens_per_point * prefix_points
            mask[prefix_tokens:, :] = False
            for point in range(prefix_points, self.n_positions):
                first_token = self._tokens_per_point * point
                for token in range(first_token, first_token + self._tokens_per_point):
                    mask[token, :prefix_tokens] = True
                    mask[token, first_token : token + 1] = True


        mask = mask.view(1, 1, max_tokens, max_tokens)
        for block in self._backbone.h:
            # GPT2Attention already registers `bias` as a non-persistent
            # buffer, so assignment keeps the mask device-aware and out of
            # checkpoints.
            block.attn.bias = mask.clone()

    def load_state_dict(self, state_dict, strict=True):
        # Older transformers (<4.21) saved an `attn.masked_bias` buffer per
        # layer that no longer exists; strip it so old checkpoints load cleanly.
        state_dict = {
            k: v for k, v in state_dict.items() if not k.endswith(".attn.masked_bias")
        }
        result = super().load_state_dict(state_dict, strict=strict)
        self._apply_pos_encode_freeze()
        self._apply_attention_mask()
        return result

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(
        self,
        xs,
        ys,
        inds=None,
        output_hidden_states=False,
        task_ids=None,
    ):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys) # interleaved x and y in n_points dim i.e. x1, y1, x2, y2, ...
        embeds = self._read_in(zs)
        if self._task_tag_mode != "none" and self._task_slot_embeddings is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids are required when model.num_task_slots is positive"
                )
            if task_ids.shape != ys.shape:
                raise ValueError(
                    f"task_ids shape {tuple(task_ids.shape)} must match "
                    f"ys shape {tuple(ys.shape)}"
                )
            slot_embeds = self._task_slot_embeddings(task_ids.long())
            if self._task_tag_mode == "prefix_token":
                embeds = torch.stack(
                    (slot_embeds, embeds[:, 0::2], embeds[:, 1::2]), dim=2
                ).reshape(ys.shape[0], 3 * ys.shape[1], -1)
            else:
                embeds = embeds + slot_embeds.repeat_interleave(2, dim=1)
        elif task_ids is not None and self._task_tag_mode != "none":
            raise ValueError(
                "task_ids were provided but model.num_task_slots is zero"
            )
        backbone_output = self._backbone(inputs_embeds=embeds, output_hidden_states=output_hidden_states, return_dict=True)
        # if output_hidden_states=True, backbone_output.hidden_states = list of len n_layers with each element of shape (batch_size, n_points * 2, embed_dim). Each element corresponds to the output from the application of i-th layer (0th element being the input and 12th element being the output of last layer)

        # input_shape = embeds.size()[:-1]
        # temp = torch.arange(0, input_shape[-1], dtype=torch.long, device=embeds.device)
        # post_encods = self._backbone.wpe(temp).unsqueeze(0).view(-1, 256)
        # print(post_encods.shape)
        # pos_embeds = embeds + post_encods
        # test_eq = pos_embeds == backbone_output.hidden_states[0]
        # a=torch.all(test_eq)
        # import pdb
        # pdb.set_trace()
        output = backbone_output.last_hidden_state
        prediction = self._read_out(output)
        prediction = prediction[:, 1::3, 0] if self._task_tag_mode == "prefix_token" else prediction[:, ::2, 0]
        if not output_hidden_states:
            return prediction[:, inds]  # predict only on xs
        else:
            return prediction[:, inds], backbone_output.hidden_states


class TaskPrefixTransformerModel(TransformerModel):
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True
    ):
        super(TaskPrefixTransformerModel, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pos_encode
        )
        self.task_prefix_embeddings = nn.Embedding(2, n_embd)

    def forward(self, xs, ys, prefix=None, inds=None, output_hidden_states=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        if prefix is None:
            prefix = torch.zeros(xs.shape[0]).to(xs.device).long()
        zs = self._combine(
            xs, ys
        )  # interleaved x and y in n_points dim i.e. x1, y1, x2, y2, ...
        embeds = self._read_in(zs)
        prefix_embeds = self.task_prefix_embeddings(prefix).unsqueeze(1)
        embeds = torch.cat([prefix_embeds, embeds], axis=1)
        backbone_output = self._backbone(
            inputs_embeds=embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        output = backbone_output.last_hidden_state
        prediction = self._read_out(output)
        if not output_hidden_states:
            return prediction[:, 1::2, 0][:, inds]  # predict only on xs
        else:
            return prediction[:, 1::2, 0][:, inds], backbone_output.hidden_states

########################################################################################################################
# Ridge (ICL Task Diversity)                                                                                                               #
########################################################################################################################

class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, data, targets):
        """
        Args:
            xs: batch_size x n_points x n_dims (float)
            ys: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        batch_size, n_points, _ = data.shape
        targets = np.expand_dims(targets, -1)  # batch_size x n_points x 1
        preds = [np.zeros(batch_size, dtype=data.dtype)]
        weights = []
        # preds.extend(
        #     [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam) for _i in range(1, n_points)]
        # )
        for _i in tqdm(range(1, n_points)):
            curr_pred, curr_weight = self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam)
            preds.append(curr_pred)
            weights.append(curr_weight.squeeze()) # (n_points - 1, batch x n_dims)
        preds = np.stack(preds, axis=1)
        weights = np.stack(weights, axis=0) # (n_points - 1, batch x n_dims)
        return preds, weights

    def predict(self, X, Y, test_x, lam: float):
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            lam: (float)
        Return:
            batch_size (float)
        """
        _, _, n_dims = X.shape
        XT = X.transpose((0, 2, 1))  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = np.matmul(XT, X) + lam * np.eye(n_dims, dtype=X.dtype)  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = np.linalg.solve(ridge_matrix.astype(np.float32), XT_Y.astype(np.float32)).astype(X.dtype) # batch x n_dims x 1
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0], ws


########################################################################################################################
# MMSE (ICL Task Diversity)                                                                                                 #
########################################################################################################################


class DiscreteMMSE:
    def __init__(self, scale, task_pool):
        self.scale = scale
        self.task_pool = task_pool # num_tasks x n_dims x 1

    def __call__(self, data, targets):
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        _, n_points, _ = data.shape
        targets = np.expand_dims(targets, -1)  # batch_size x n_points x 1
        W = self.task_pool.squeeze().T  # n_dims x n_tasks  (maybe do squeeze and transpose in setup?)
        preds = [data[:, 0] @ W.mean(axis=1)]  # batch_size
        weights = []
        # preds.extend(
        #     [
        #         self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
        #         for _i in range(1, n_points)
        #     ]
        # )
        for _i in tqdm(range(1, n_points)):
            curr_pred, curr_weight = self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
            preds.append(curr_pred)
            weights.append(curr_weight.squeeze()) # curr_weight = (batch x n_dims)
        preds = np.stack(preds, axis=1)  # batch_size x n_points
        weights = np.stack(weights, axis=0) # (n_points - 1, batch x n_dims)
        return preds, weights

    def predict(self, X, Y, test_x, W, scale: float):
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        W = torch.from_numpy(W)
        # alpha = tfd.Normal(0, scale).log_prob(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST)).astype(self.dtype).sum(axis=1)
        alpha = tdn.Normal(0, scale).log_prob(Y - torch.matmul(X, W)).sum(dim=1) # batch x n_tasks
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        # w_mmse = jnp.expand_dims(jnp.matmul(jax.nn.softmax(alpha, axis=1), W.T, precision=jax.lax.Precision.HIGHEST), -1) # n_dims x 1
        w_mmse = torch.unsqueeze(torch.matmul(F.softmax(alpha, dim=1), W.t()), dim=-1) # batch_size x n_dims x 1
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
        pred = test_x @ w_mmse.detach().cpu().numpy()
        return pred[:, 0, 0], w_mmse.detach().cpu().numpy()

class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None, return_weights=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = []
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )
            all_ws.append(ws)
            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        if not return_weights:
            return torch.stack(preds, dim=1)
        else:
            return torch.stack(preds, dim=1), all_ws


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):
                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None, preprocess=None):
        self.max_depth = max_depth
        self.preprocess = preprocess
        preprocess_name = "" if preprocess is None else f"_preprocess={preprocess}"
        self.name = f"decision_tree_max_depth={max_depth}{preprocess_name}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if self.preprocess == "sign":
            xs = xs.sign()
        elif self.preprocess is not None:
            raise ValueError(f"Unknown preprocess: {self.preprocess}")

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self, n_jobs=None, preprocess=None):
        self.n_jobs = n_jobs
        self.preprocess = preprocess
        preprocess_name = "" if preprocess is None else f"_preprocess={preprocess}"
        self.name = f"xgboost{preprocess_name}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if self.preprocess == "sign":
            xs = xs.sign()
        elif self.preprocess is not None:
            raise ValueError(f"Unknown preprocess: {self.preprocess}")

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor(n_jobs=self.n_jobs)

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
