#!/usr/bin/env bash
set -euo pipefail

# Preview every submission without launching it:
#   DRY_RUN=1 ./scripts/submit_next_experiments.sh
DRY_RUN=${DRY_RUN:-0}
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
    shift 4

    local command=(
        env "TRAIN_MODULE=$module" "CONFIG=$config"
        "$SUBMIT_SCRIPT" "$job_name"
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

# 1) 5D loss-budget control. Both are evaluated at context length 5.
submit_job train.py conf/experiments/icl_lr5_loss_budget.yaml \
    icl-lr5-last5-200k-s0 icl_lr5_l11_last5_200k_s0 \
    --training.seed 0 \
    --training.train_steps 200000 \
    --training.k_steps_for_loss 5

submit_job train.py conf/experiments/icl_lr5_loss_budget.yaml \
    icl-lr5-last1-1000k-s0 icl_lr5_l11_last1_1000k_s0 \
    --training.seed 0 \
    --training.train_steps 1000000 \
    --training.k_steps_for_loss 1

# 2) Matched ICL/MAML: 20 conditioning/support and 20 loss/query examples.
submit_job train.py conf/experiments/icl_lr20_condition20_loss20.yaml \
    icl-c20-l20-causal-s0 icl_lr20_causal_c20_l20_s0 \
    --training.seed 0 \
    --model.attn_implementation sdpa \
    --model.attention_mode causal

submit_job train.py conf/experiments/icl_lr20_condition20_loss20.yaml \
    icl-c20-l20-bidir-s0 icl_lr20_prefixbidir_c20_l20_s0 \
    --training.seed 0 \
    --model.attn_implementation eager \
    --model.attention_mode prefix_bidirectional \
    --model.prefix_condition_points 20

for inner_steps in 2 5; do
    submit_job train_meta.py conf/experiments/maml_lr20_support20_query20.yaml \
        "maml-s20-q20-i${inner_steps}-s0" \
        "maml_lr20_s20_q20_steps${inner_steps}_s0" \
        --training.seed 0 \
        --meta.num_inner_steps "$inner_steps"
done

# 3) Positional embeddings for last-one training. The existing no-position
# length-21 run is reused; only the missing three runs are launched.
submit_job train.py conf/experiments/icl_lr20_last1_positional.yaml \
    icl-l21-last1-pos-s0 icl_lr20_l21_last1_pos_s0 \
    --training.seed 0 \
    --model.pos_encode true

submit_job train.py conf/experiments/icl_lr20_last1_positional.yaml \
    icl-l41-last1-nopos-s0 icl_lr20_l41_last1_nopos_s0 \
    --training.seed 0 \
    --model.pos_encode false \
    --training.curriculum.points.start 41 \
    --training.curriculum.points.end 41

submit_job train.py conf/experiments/icl_lr20_last1_positional.yaml \
    icl-l41-last1-pos-s0 icl_lr20_l41_last1_pos_s0 \
    --training.seed 0 \
    --model.pos_encode true \
    --training.curriculum.points.start 41 \
    --training.curriculum.points.end 41

# 4) Lower priority: add seeds 1 and 2 to the existing seed-0 position-window
# runs. This block is intentionally last in the submission order.
for seed in 154645467 65765443; do
    submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
        "icl-pos01-15-s${seed}" "icl_pos_terms01_15_s${seed}" \
        --training.seed "$seed" \
        --training.loss_positions range:0:15
    submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
        "icl-pos06-20-s${seed}" "icl_pos_terms06_20_s${seed}" \
        --training.seed "$seed" \
        --training.loss_positions range:5:20
    submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
        "icl-pos21-35-s${seed}" "icl_pos_terms21_35_s${seed}" \
        --training.seed "$seed" \
        --training.loss_positions range:20:35
    submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
        "icl-pos26-40-s${seed}" "icl_pos_terms26_40_s${seed}" \
        --training.seed "$seed" \
        --training.loss_positions range:25:40
done
