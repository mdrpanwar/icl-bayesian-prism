import math

import torch
from torch import nn
from torch.func import functional_call


INNER_LR_MODULE_NAME = "maml_inner_lrs"
INNER_LR_NAME_MAP_ATTR = "_maml_inner_lr_name_map"


def _get_meta_attr(meta_args, name, default=None):
    if isinstance(meta_args, dict):
        return meta_args.get(name, default)
    return getattr(meta_args, name, default)


def is_inner_lr_parameter(name):
    return name.startswith(f"{INNER_LR_MODULE_NAME}.")


def trainable_model_params(model):
    return {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad and not is_inner_lr_parameter(name)
    }


def safe_inner_lr_key(param_name):
    return param_name.replace("_", "__us__").replace(".", "__dot__")


def _inner_lr_parameterization(meta_args):
    parameterization = _get_meta_attr(
        meta_args, "inner_lr_parameterization", "direct"
    )
    if parameterization not in {"direct", "bounded_signed"}:
        raise ValueError(
            "Unknown inner_lr_parameterization: "
            f"{parameterization!r}"
        )
    return parameterization


def _effective_inner_lr(raw_inner_lr, meta_args):
    if _inner_lr_parameterization(meta_args) == "direct":
        return raw_inner_lr
    bound = _get_meta_attr(meta_args, "inner_lr_bound", None)
    if bound is None or float(bound) <= 0:
        raise ValueError("bounded_signed inner LRs require inner_lr_bound > 0")
    return float(bound) * torch.tanh(raw_inner_lr)


def configure_inner_lrs(model, meta_args):
    mode = _get_meta_attr(meta_args, "inner_lr_mode", "fixed")
    if mode == "fixed":
        return None
    if mode != "learned_per_param":
        raise ValueError(f"Unknown inner_lr_mode: {mode}")

    parameterization = _inner_lr_parameterization(meta_args)
    init = float(_get_meta_attr(meta_args, "inner_lr", 0.01))
    parameter_init = init
    if parameterization == "bounded_signed":
        bound = _get_meta_attr(meta_args, "inner_lr_bound", None)
        if bound is None or float(bound) <= 0:
            raise ValueError("bounded_signed inner LRs require inner_lr_bound > 0")
        bound = float(bound)
        if abs(init) >= bound:
            raise ValueError(
                "bounded_signed requires abs(inner_lr) < inner_lr_bound"
            )
        parameter_init = math.atanh(init / bound)

    params = trainable_model_params(model)
    name_map = {name: safe_inner_lr_key(name) for name in params}

    existing = getattr(model, INNER_LR_MODULE_NAME, None)
    if existing is not None:
        setattr(model, INNER_LR_NAME_MAP_ATTR, name_map)
        return existing

    inner_lrs = nn.ParameterDict()
    for name, p in params.items():
        inner_lrs[name_map[name]] = nn.Parameter(torch.full_like(p, parameter_init))

    setattr(model, INNER_LR_MODULE_NAME, inner_lrs)
    setattr(model, INNER_LR_NAME_MAP_ATTR, name_map)
    return inner_lrs


def get_inner_lrs(model, meta_args):
    mode = _get_meta_attr(meta_args, "inner_lr_mode", "fixed")
    if mode == "fixed":
        return float(_get_meta_attr(meta_args, "inner_lr", 0.01))

    configure_inner_lrs(model, meta_args)
    inner_lrs = getattr(model, INNER_LR_MODULE_NAME)
    name_map = getattr(model, INNER_LR_NAME_MAP_ATTR)
    return {
        name: _effective_inner_lr(inner_lrs[key], meta_args)
        for name, key in name_map.items()
    }


def _tensor_stats(tensors):
    if not tensors:
        return {}

    flat = torch.cat([t.detach().float().reshape(-1) for t in tensors])
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "abs_mean": flat.abs().mean().item(),
        "abs_max": flat.abs().max().item(),
        "neg_frac": (flat < 0).float().mean().item(),
    }


def _inner_lr_layer_name(param_name):
    parts = param_name.split(".")
    if "_backbone" in parts and "h" in parts:
        h_idx = parts.index("h")
        if h_idx + 1 < len(parts) and parts[h_idx + 1].isdigit():
            return f"layer_{int(parts[h_idx + 1]):02d}"
    if param_name.startswith("_read_in."):
        return "read_in"
    if param_name.startswith("_read_out."):
        return "read_out"
    if "wte" in parts:
        return "embedding"
    if "ln" in param_name:
        return "ln"
    return "other"


def _inner_lr_module_name(param_name):
    parts = param_name.split(".")
    if "attn" in parts:
        return "attn"
    if "mlp" in parts:
        return "mlp"
    if any(part.startswith("ln") for part in parts):
        return "ln"
    if param_name.startswith("_read_in."):
        return "read_in"
    if param_name.startswith("_read_out."):
        return "read_out"
    if "wte" in parts:
        return "embedding"
    return "other"


def inner_lr_stats(model, meta_args, include_layer=False, include_layer_module=False):
    """Return scalar logging stats for learned per-parameter inner LRs."""
    if _get_meta_attr(meta_args, "inner_lr_mode", "fixed") != "learned_per_param":
        return {}

    inner_lrs = get_inner_lrs(model, meta_args)
    stats = {
        f"inner_lr/global/{key}": value
        for key, value in _tensor_stats(list(inner_lrs.values())).items()
    }
    if _inner_lr_parameterization(meta_args) == "bounded_signed":
        raw_inner_lrs = getattr(model, INNER_LR_MODULE_NAME)
        stats.update(
            {
                f"inner_lr/raw_global/{key}": value
                for key, value in _tensor_stats(list(raw_inner_lrs.values())).items()
            }
        )

    if include_layer:
        layer_tensors = {}
        for name, tensor in inner_lrs.items():
            layer_tensors.setdefault(_inner_lr_layer_name(name), []).append(tensor)
        for layer, tensors in layer_tensors.items():
            for key, value in _tensor_stats(tensors).items():
                stats[f"inner_lr/layers/{layer}/{key}"] = value

    if include_layer_module:
        layer_module_tensors = {}
        for name, tensor in inner_lrs.items():
            group = (_inner_lr_layer_name(name), _inner_lr_module_name(name))
            layer_module_tensors.setdefault(group, []).append(tensor)
        for (layer, module), tensors in layer_module_tensors.items():
            for key, value in _tensor_stats(tensors).items():
                stats[f"inner_lr/layer_modules/{layer}/{module}/{key}"] = value

    return stats


def inner_lr_for_param(inner_lrs, name):
    if isinstance(inner_lrs, dict):
        return inner_lrs[name]
    return inner_lrs


def update_params_with_grads(params, grads, inner_lrs):
    return {
        name: (
            p
            if grads[name] is None
            else p - inner_lr_for_param(inner_lrs, name) * grads[name]
        )
        for name, p in params.items()
    }


def inner_adapt(
    model,
    init_params,
    xs_support,
    ys_support,
    inner_lrs,
    num_inner_steps,
    first_order,
    loss_func,
    task_ids_support=None,
):
    """Run inner-loop gradient updates on independent support examples."""
    fast_params = init_params
    for _ in range(num_inner_steps):
        xs_batch = xs_support.reshape(-1, 1, xs_support.shape[-1])
        ys_batch = ys_support.reshape(-1, 1)
        kwargs = None
        if task_ids_support is not None:
            kwargs = {"task_ids": task_ids_support.reshape(-1, 1)}
        preds = functional_call(model, fast_params, (xs_batch, ys_batch), kwargs)
        loss_s = loss_func(preds, ys_batch)
        grads = torch.autograd.grad(
            loss_s,
            list(fast_params.values()),
            create_graph=not first_order,
            allow_unused=True,
        )
        grads = {name: g for (name, _), g in zip(fast_params.items(), grads)}
        fast_params = update_params_with_grads(fast_params, grads, inner_lrs)
    return fast_params


def load_state_dict_allow_missing_inner_lrs(model, state_dict):
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not is_inner_lr_parameter(key)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not is_inner_lr_parameter(key)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Error(s) in loading state_dict for "
            f"{model.__class__.__name__}: missing={missing}, unexpected={unexpected}"
        )
