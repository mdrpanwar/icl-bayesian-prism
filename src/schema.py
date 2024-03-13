from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge
# from quinine.common.cerberus import tlistorstring


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "pos_encode": merge(tboolean, required),
    "load_model_path": merge(tstring, nullable, default(None)),
    "train_only_emb": merge(tboolean, default(False)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_opt_schema = {
    "start": merge(tinteger, nullable, default(None)),  # initial parameter
    "end": merge(tinteger, nullable, default(None)),  # limit of final value
    "inc": merge(tinteger, nullable, default(None)),  # how much to increment each time
    "interval": merge(
        tinteger, nullable, default(None)
    ),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "max_freq": stdict(curriculum_opt_schema),
    "rff_dim": stdict(curriculum_opt_schema),
}

TASK_LIST = [
    "linear_regression",
    "noisy_linear_regression",
    "sparse_linear_regression",
    "sparse_reg_laplacian_prior",
    "linear_classification",
    "relu_2nn_regression",
    "relu_2nn_regression_with_bias",
    "sigmoid_2nn_regression_with_bias",
    "relu_3nn_regression",
    "decision_tree",
    "sparse_linear_mixer",
    "low_rank_cs",
    "sign_vec_cs",
    "two_task_mixer",
    "three_task_mixer",
    "polynomials",
    "polynomials_factor_form",
    "polynomials_unbiased_points",
    "fourier_series_mixture",
    "fourier_series",
    "fourier_series_complexity_bias",
    "random_fourier_features",
    "polynomials_deg2_monomials_selection_biased",
    "polynomials_deg2_monomials_selection_unbiased",
    "gaussian_mixture_linear_regression",
    "three_gaussian_mixture_linear_regression",
    "uniform_mixture_linear_regression",
    "haar_wavelets",
    "noisy_lr_task_diversity",
    "fourier_series_multitask",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    # For two_task_mixer, "task_kwargs": {"task1": str, "task2": str, "mixing_ratio": float, "task_spec_params_dict":{"<task1>" : {...}, "<task2>" : {...}}}
    # For three_task_mixer, "task_kwargs": {"task1": "<task1>", "task2": "<task2>", "task3": "<task3>", "mixing_ratio": {"task1": proportion of task1, "task2": proportion of task2, "task3": proportion of task3}, "task_spec_params_dict":{"<task1>" : {...}, "<task2>" : {...}, "<task3>" : {...}}}
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "uniform"])),
    "data_kwargs": merge(tdict, nullable, default(None)),
    "data_transformation_args": merge(tdict, nullable, default(None)),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "eval_every_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "eval_ood": merge(tboolean, default(False)),
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "k_steps_for_loss": merge(tstring, default("all")),
    "num_accum_steps": merge(tinteger, default(1)), # number of gradient accumulation steps; we take an optimizer step every num_accum_steps steps
    "schedule": merge(tstring, nullable, default(None)),
    "warmup_steps": merge(tinteger, nullable, default(None)),
    "log_model_norm": merge(tboolean, default(False)),
    "granular_ckpt_till": merge(tinteger, nullable, default(None)), # checkpoint every `granular_ckpt_every` steps until `granular_ckpt_till` steps
    "granular_ckpt_every": merge(tinteger, nullable, default(None)),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
    "is_save_task_pool": merge(tboolean, default(True)), # only applicable for task diversity task
}
