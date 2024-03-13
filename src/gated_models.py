import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GatedGPT2Model
from models import TransformerModel


def build_model(conf):
    if conf.family == "gpt2":
        model = GatedTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            pos_encode=conf.pos_encode,
        )
    elif conf.family == "dsp_gpt2":
        model = DSPGatedTransformerModel(
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


class GatedTransformerModel(TransformerModel):
    def __init__(
        self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pos_encode=True
    ):
        super(GatedTransformerModel, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pos_encode
        )
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.config = configuration
        self.name = self.name.replace("gpt2", "gated_gpt2")
        self._backbone = GatedGPT2Model(configuration, pos_encode=pos_encode)

    def get_gate_values(self):
        self._backbone.get_gate_values()

    def apply_gates(self, l0_penalty):
        self._backbone.apply_gates(l0_penalty)

    def remove_gates(self):
        self._backbone.remove_gates()

    def apply_masks(self, head_mask):
        self._backbone.apply_masks(head_mask)

    def get_masks(self):
        self._backbone.get_masks()

    def apply_dsp(self, num_of_heads, temperature=None, use_ste=False):
        self._backbone.apply_dsp(num_of_heads, temperature, use_ste)


class DSPGatedTransformerModel(GatedTransformerModel):
    def forward(self, xs, ys, inds=None, output_hidden_states=False, return_dict=False):
        preds = super().forward(xs, ys, inds, output_hidden_states)
        loss = (ys - preds).square().mean()

        return (loss, preds)
