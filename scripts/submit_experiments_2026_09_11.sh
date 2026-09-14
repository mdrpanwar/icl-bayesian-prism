#!/usr/bin/env bash
set -euo pipefail

# Experiments agreed on 2026-09-11.
# Preview:
#   DRY_RUN=1 ./scripts/submit_experiments_2026_09_11.sh
DRY_RUN=${DRY_RUN:-0}
NODE_POOL=${NODE_POOL:-h100}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
SUBMIT_SCRIPT="$WORKSPACE_ROOT/submit_un.sh"
cd "$WORKSPACE_ROOT"

[[ -x "$SUBMIT_SCRIPT" ]] || {
    echo "Missing executable launcher: $SUBMIT_SCRIPT" >&2
    exit 1
}

submit_job() {
    local module=$1
    local config=$2
    local job_name=$3
    local run_name=$4
    local run_id=$5
    shift 5

    local command=(
        env
        "TRAIN_MODULE=$module"
        "CONFIG=$config"
        "NODE_POOL=$NODE_POOL"
        "RUN_ID=$run_id"
        "$SUBMIT_SCRIPT"
        "$job_name"
        --run_name "$run_name"
        "$@"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

# 1) Continue the longest stable 5-step Meta-SGD run from its local state.pt.
# Set SKIP_RESUME=1 when that independently submitted job is already active.
# if [[ "${SKIP_RESUME:-0}" != "1" ]]; then
#     submit_job train_meta.py \
#         conf/experiments/maml_lr20_support20_query20_stable.yaml \
#         maml-s20-q20-i5-resume-b92 \
#         maml_lr20_s20_q20_steps5_stable_s0 \
#         b92d63fa-31ba-4a24-9860-1728b3e5c32a \
#         --training.seed 0 \
#         --training.train_steps 500000 \
#         --meta.num_inner_steps 5
# fi

# 2) Correct 20-support/20-query ICL controls. Every query sees the support
# prefix and itself, but never another query.
# for mode in causal prefix_bidirectional; do
#     if [[ "$mode" == causal ]]; then
#         short_mode=causal
#     else
#         short_mode=bidir
#     fi
#     for seed in 0 154645467 65765443; do
#         submit_job train.py \
#             conf/experiments/icl_lr20_isolated_queries.yaml \
#             "icl-iso-${short_mode}-s${seed}" \
#             "icl_lr20_isolated_${short_mode}_s${seed}" \
#             "exp20260911-icl-iso-${short_mode}-s${seed}" \
#             --training.seed "$seed" \
#             --model.attention_mode "$mode"
#     done
# done

# 3) Five-dimensional objective-budget and positional controls.
# submit_job train.py conf/experiments/icl_lr5_loss_budget.yaml \
#     icl-lr5-last4-250k-s0 \
#     icl_lr5_l11_last4_250k_s0 \
#     exp20260911-icl-lr5-last4-s0 \
#     --training.seed 0 \
#     --training.train_steps 250000 \
#     --training.k_steps_for_loss 4

# submit_job train.py conf/experiments/icl_lr5_last1_positional.yaml \
#     icl-lr5-last1-pe-1000k-s0 \
#     icl_lr5_l11_last1_pos_1000k_s0 \
#     exp20260911-icl-lr5-last1-pe-s0 \
#     --training.seed 0 \
#     --training.train_steps 1000000 \
#     --training.k_steps_for_loss 1

# 4) Short same-H100 calibration runs for the primary GPU-hours plot.
submit_job train.py conf/experiments/icl_lr20_isolated_queries.yaml \
    profile-icl-iso-causal \
    profile_icl_isolated_causal_h100 \
    exp20260911-profile-icl-causal \
    --training.seed 0 \
    --training.train_steps 220 \
    --training.compute_profile_warmup_steps 20 \
    --training.compute_profile_steps 200 \
    --model.attention_mode causal

submit_job train.py conf/experiments/icl_lr20_isolated_queries.yaml \
    profile-icl-iso-bidir \
    profile_icl_isolated_bidir_h100 \
    exp20260911-profile-icl-bidir \
    --training.seed 0 \
    --training.train_steps 220 \
    --training.compute_profile_warmup_steps 20 \
    --training.compute_profile_steps 200 \
    --model.attention_mode prefix_bidirectional

for inner_steps in 2 5; do
    submit_job train_meta.py \
        conf/experiments/maml_lr20_support20_query20_stable.yaml \
        "profile-maml-i${inner_steps}" \
        "profile_maml_steps${inner_steps}_h100" \
        "exp20260911-profile-maml-i${inner_steps}" \
        --training.seed 0 \
        --training.train_steps 220 \
        --training.compute_profile_warmup_steps 20 \
        --training.compute_profile_steps 200 \
        --meta.num_inner_steps "$inner_steps"
done

# 5) Lower-priority position-window control: human terms 5-19 are exactly
# zero-based positions [4, 19), i.e. 15 terms.
# for seed in 0 154645467 65765443; do
#     submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
#         "icl-pos05-19-s${seed}" \
#         "icl_pos_terms05_19_s${seed}" \
#         "exp20260911-icl-pos05-19-s${seed}" \
#         --training.seed "$seed" \
#         --training.loss_positions range:4:19
# done
