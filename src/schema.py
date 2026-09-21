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
    # SDPA's fused backends don't implement double-backward, so second-order MAML
    # needs eager. FOMAML / ICL can use sdpa to dispatch FlashAttention.
    "attn_implementation": merge(
        tstring, allowed(["eager", "sdpa"]), default("eager")
    ),
    # prefix_bidirectional makes only the conditioning prefix fully visible;
    # tokens after the prefix retain the usual causal mask.
    # Keep each prediction/query point independent: it can attend to the
    # conditioning prefix and its own x/y tokens, but not to other queries.
    "isolate_query_points": merge(tboolean, default(False)),
    # Optional per-example task/function identities. When positive, the model
    # adds a learned task-slot embedding to both tokens of every (x, y) pair.
    "num_task_slots": merge(tinteger, default(0)),
    # additive is the earlier pilot; prefix_token makes [tag,x,y] triplets.
    # none ignores IDs and retains the slot table for a parameter-matched
    # untagged control.
    "task_tag_mode": merge(
        tstring, allowed(["additive", "prefix_token", "none"]), default("additive")
    ),

    "attention_mode": merge(
        tstring,
        allowed(["causal", "prefix_bidirectional"]),
        default("causal"),
    ),
    "prefix_condition_points": merge(tinteger, nullable, default(None)),
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
    "multi_function_linear",
    "segmented_pretraining",
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
    "warm_start_model_checkpoint": merge(tstring, nullable, default(None)),
    "curriculum": stdict(curriculum_schema),
    "k_steps_for_loss": merge(tstring, default("all")),
    # Optional zero-based selector: all, last:N, or range:START:STOP (STOP exclusive).
    # When set, k_steps_for_loss must remain "all".
    "loss_positions": merge(tstring, nullable, default(None)),
    # Optional comma-separated zero-based query positions. One position is
    # sampled independently for every row of a training batch.
    "loss_position_choices": merge(tstring, nullable, default(None)),
    # Optional comma-separated zero-based query positions. Every position is
    # supervised in every sequence and their losses are averaged together.
    "loss_position_set": merge(tstring, nullable, default(None)),
    "eval_n_points": merge(tinteger, nullable, default(None)),
    "seed": merge(tinteger, default(0)),
    "num_accum_steps": merge(tinteger, default(1)), # number of gradient accumulation steps; we take an optimizer step every num_accum_steps steps
    "schedule": merge(tstring, nullable, default(None)),
    "warmup_steps": merge(tinteger, nullable, default(None)),
    # When positive, skip evaluation/checkpointing and time this many optimizer
    # steps after the requested warmup. Used for same-GPU compute calibration.
    "compute_profile_warmup_steps": merge(tinteger, default(20)),
    "compute_profile_steps": merge(tinteger, default(0)),
    "log_model_norm": merge(tboolean, default(False)),
    "granular_ckpt_till": merge(tinteger, nullable, default(None)), # checkpoint every `granular_ckpt_every` steps until `granular_ckpt_till` steps
    "granular_ckpt_every": merge(tinteger, nullable, default(None)),
}

wandb_schema = {
    "project": merge(tstring, default("icl-metal")),
    "entity": merge(tstring, nullable, default("mdrpanwar")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "run_name": merge(tstring, nullable, default(None)),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
    "is_save_task_pool": merge(tboolean, default(True)), # only applicable for task diversity task
}

# Meta-learning (MAML) schema. Used by train_meta.py.
meta_schema = {
    "inner_lr": merge(tfloat, default(0.01)),
    "inner_lr_mode": merge(
        tstring,
        allowed(["fixed", "learned_per_param"]),
        default("fixed"),
    ),
    "inner_lr_parameterization": merge(
        tstring,
        allowed(["direct", "bounded_signed"]),
        default("direct"),
    ),
    "inner_lr_bound": merge(tfloat, nullable, default(None)),
    "outer_grad_clip_norm": merge(tfloat, nullable, default(None)),
    "max_outer_grad_norm_before_skip": merge(tfloat, nullable, default(None)),
    "fail_on_nonfinite": merge(tboolean, default(False)),
    "num_inner_steps": merge(tinteger, default(1)),
    "first_order": merge(tboolean, default(False)),
    "meta_batch_size": merge(tinteger, default(8)),
    "num_query_points": merge(tinteger, default(10)),
    "vary_support_size": merge(
        tstring,
        allowed([
            "match_curriculum", "full_range", "fixed", "choices",
            "choices_parallel", "choices_sequential",
        ]),
        default("match_curriculum"),
    ),
    "fixed_support_size": merge(tinteger, nullable, default(None)),
    # Comma-separated support sizes used by all three choices modes. `choices`
    # samples one k; `choices_parallel` adapts independently at every k;
    # `choices_sequential` carries fast parameters through the sorted k values.
    "support_size_choices": merge(tstring, nullable, default(None)),
    "multi_k_support": merge(tboolean, default(False)),
    "meta_eval_every_steps": merge(tinteger, default(5000)),
    "meta_eval_stride": merge(tinteger, default(1)),
    "meta_eval_num_tasks": merge(tinteger, default(256)),
    "meta_eval_seed": merge(tinteger, default(0)),
    "run_icl_style_eval": merge(tboolean, default(True)),
    "run_maml_style_eval": merge(tboolean, default(True)),
}

meta_training_schema = dict(schema)
meta_training_schema["meta"] = stdict(meta_schema)
