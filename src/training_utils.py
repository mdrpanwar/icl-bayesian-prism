import wandb
import torch
from math import sqrt

"""
    norm computation functions are adapted from https://github.com/viking-sudo-rm/norm-growth/blob/master/t5_main.py
"""

def _fan_in(shape) -> int:
    # This is from some TensorFlow code or something.
    return float(shape[-2]) if len(shape) > 1 else float(shape[-1])

def get_param_norm(params, normalize: bool = False, is_min: bool = False):
    # There are weird scalars in here, which we filter out.
    # values = [v for v in params if len(v.shape) > 0] 
    values = params   
    if is_min:
        # Take the linear transformation in the network with the least norm.
        values = [v / sqrt(v.numel()) for v in values if len(v.shape) == 2]
        norms = [torch.linalg.norm(v).item() for v in values]
        return min(norms)
    else:
        # This is the 2-norm.
        if normalize:
            values = [value / sqrt(_fan_in(value.shape)) for value in values]
        flat = torch.cat([value.flatten() for value in values])
        norm = torch.linalg.vector_norm(flat, ord=2)
        return norm.item()

def get_list_of_param_values(model):
    # return [param for name, param in model.named_parameters()]
    return [param.detach() for param in model.parameters()]

def compute_model_norm(model):
    values = get_list_of_param_values(model)
    norm = get_param_norm(values, normalize=False, is_min=False)
    return norm

def compute_and_log_model_norm(model, step):
    model_norm = compute_model_norm(model)
    wandb.log(
        {
            "model_weight_norm": model_norm,
        },
        step=step,
    )

def filter_hold_out_freq(hold_out_freq, training_S_list):
    filter_freq = hold_out_freq["freq"]
    if hold_out_freq["type"] == "single":
        # "hold_out_freq": {"type": "single", "freq": [4, 10, 18]}
        filter_freq_set = set(filter_freq)
        training_S_list_return = list(filter(lambda x: not bool(filter_freq_set & x), training_S_list))
    elif hold_out_freq["type"] == "pair":
        # "hold_out_freq": {"type": "pair", "freq": [[4, 9], [3, 17], [1, 2], ....]}
        filter_freq_set = [set(x) for x in filter_freq]
        training_S_list_return = []
        want_set = False
        for S_set in training_S_list:
            want_set = True
            for filter_freq_pair in filter_freq_set:
                if len(filter_freq_pair & S_set) == 2:
                    want_set = False
                    break
            if want_set:
                training_S_list_return.append(S_set)
    return training_S_list_return

def listify(some_list_of_sets):
    return [list(x) for x in some_list_of_sets]

def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched