#!/usr/bin/env bash
set -euo pipefail

# Launches fixed-length dynamics, objective-position, and last-10 ICL runs.
# For a command preview: DRY_RUN=1 ./scripts/submit_icl_experiments.sh
SEED=${SEED:-0}
DRY_RUN=${DRY_RUN:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SUBMIT_SCRIPT="$WORKSPACE_DIR/submit_un.sh"
cd "$WORKSPACE_DIR"

[[ -x "$SUBMIT_SCRIPT" ]] || { echo "Missing executable: $SUBMIT_SCRIPT" >&2; exit 1; }

submit_run() {
    local job_name=$1
    local run_name=$2
    local sequence_length=$3
    local eval_length=$4
    local seed=$5
    shift 5

    local command=(
        env TRAIN_MODULE=train.py CONFIG=conf/linear_regression.yaml
        "$SUBMIT_SCRIPT" "$job_name"
        --run_name "$run_name"
        --training.seed "$seed"
        --training.eval_n_points "$eval_length"
        --training.curriculum.points.start "$sequence_length"
        --training.curriculum.points.end "$sequence_length"
        --training.curriculum.points.inc 1
        --training.curriculum.dims.start 20
        --training.curriculum.dims.end 20
        --training.curriculum.dims.inc 1
        "$@"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

# Experiment 1: fixed length 21 versus 41.
# submit_run "icl-l21-all-s${SEED}" "icl_l21_all_s${SEED}" 21 41 "$SEED" \
#     --training.k_steps_for_loss all
# submit_run "icl-l41-all-s${SEED}" "icl_l41_all_s${SEED}" 41 41 "$SEED" \
#     --training.k_steps_for_loss all

# Experiment 2: four equal-width windows of 15 one-indexed objective terms.
# Configuration ranges are zero-based and half-open.
submit_run "icl-pos01-15-s${SEED}" "icl_pos_terms01_15_s${SEED}" 41 41 "$SEED" \
    --training.loss_positions range:0:15
submit_run "icl-pos06-20-s${SEED}" "icl_pos_terms06_20_s${SEED}" 41 41 "$SEED" \
    --training.loss_positions range:5:20
submit_run "icl-pos21-35-s${SEED}" "icl_pos_terms21_35_s${SEED}" 41 41 "$SEED" \
    --training.loss_positions range:20:35
submit_run "icl-pos26-40-s${SEED}" "icl_pos_terms26_40_s${SEED}" 41 41 "$SEED" \
    --training.loss_positions range:25:40

# Experiment 3: last ten loss terms, evaluated at context lengths 0 through 40.
submit_run "icl-l41-k10-s${SEED}" "icl_l41_last10_s${SEED}" 41 41 "$SEED" \
    --training.k_steps_for_loss 10
