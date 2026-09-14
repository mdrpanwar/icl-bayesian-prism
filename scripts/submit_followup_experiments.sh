#!/usr/bin/env bash
set -euo pipefail

# Launch the stabilized MAML and follow-up ICL experiments.
# Preview without submitting:
#   DRY_RUN=1 ./scripts/submit_followup_experiments.sh
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

# 1) Stable 20-support/20-query Meta-SGD controls.
for inner_steps in 2 5; do
    submit_job train_meta.py \
        conf/experiments/maml_lr20_support20_query20_stable.yaml \
        "maml-s20-q20-i${inner_steps}-stable-s0" \
        "maml_lr20_s20_q20_steps${inner_steps}_stable_s0" \
        --training.seed 0 \
        --meta.num_inner_steps "$inner_steps"
done

# 2) Two additional exact 20-conditioning/20-loss causal ICL seeds.
for seed in 154645467 65765443; do
    submit_job train.py conf/experiments/icl_lr20_condition20_loss20.yaml \
        "icl-c20-l20-causal-s${seed}" \
        "icl_lr20_causal_c20_l20_s${seed}" \
        --training.seed "$seed" \
        --model.attn_implementation sdpa \
        --model.attention_mode causal
done

# 3) Equal-supervision-budget 5D control: 2 losses x 500k steps = 1M.
submit_job train.py conf/experiments/icl_lr5_loss_budget.yaml \
    icl-lr5-last2-500k-s0 icl_lr5_l11_last2_500k_s0 \
    --training.seed 0 \
    --training.train_steps 500000 \
    --training.k_steps_for_loss 2

# 4) Position-embedding counterparts for the matched terms-21--35 seeds.
for seed in 154645467 65765443; do
    submit_job train.py conf/experiments/icl_lr20_position_windows.yaml \
        "icl-pos21-35-pe-s${seed}" \
        "icl_pos_terms21_35_pos_s${seed}" \
        --training.seed "$seed" \
        --training.loss_positions range:20:35 \
        --model.pos_encode true
done
