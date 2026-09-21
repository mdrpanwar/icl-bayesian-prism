#!/usr/bin/env bash
set -euo pipefail

# Confirmed breadth-first experiment suite. Every run ID is stable across
# node-pool resubmission, so state.pt and W&B resume exactly.
#
# Usage:
#   DRY_RUN=1 STAGE=priority1 NODE_POOL=b300 ./scripts/submit_main_experiments_2026_09_15.sh
#   STAGE=supervision NODE_POOL=h200 ./scripts/submit_main_experiments_2026_09_15.sh

DRY_RUN=${DRY_RUN:-0}
STAGE=${STAGE:-all}
NODE_POOL=${NODE_POOL:-b300}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
SUBMIT_SCRIPT="$WORKSPACE_ROOT/submit_un.sh"
cd "$WORKSPACE_ROOT"

submit_job() {
    local module=$1 config=$2 job_name="${3}${JOB_SUFFIX:-}" run_name=$4 run_id=$5
    shift 5
    local command=(
        env "TRAIN_MODULE=$module" "CONFIG=$config" "NODE_POOL=$NODE_POOL"
        "RUN_ID=$run_id" "$SUBMIT_SCRIPT" "$job_name"
        --run_name "$run_name" "$@"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

submit_eval() {
    local job_name=$1 training_mode=$2 seed=$3 run_dir=$4
    local output="../plots/main-experiments-2026-09-15/interquery/${training_mode}_s${seed}.json"
    submit_job ../scripts/evaluate_interquery.py conf/base.yaml \
        "$job_name" "$job_name" "exp20260915-${job_name}" \
        --run-dir "$run_dir" --training-mode "$training_mode" \
        --seed "$seed" --output "$output"
}

priority1() {
    submit_eval iq-eval-causal-s0 causal 0 \
        ../models/linear_regression/icl_lr20_causal_c20_l20_s0-0897d9d4-6bc5-4ecf-92da-eec06de6bfdb
    submit_eval iq-eval-causal-s154 causal 154645467 \
        ../models/linear_regression/icl_lr20_causal_c20_l20_s154645467-31212caa-f704-4531-a3e3-8fe95435f65b
    submit_eval iq-eval-causal-s657 causal 65765443 \
        ../models/linear_regression/icl_lr20_causal_c20_l20_s65765443-c0a03422-4711-4a13-94f9-ce58245e287a
    submit_eval iq-eval-isolated-s0 isolated 0 \
        ../models/linear_regression/icl_lr20_isolated_causal_s0-exp20260911-icl-iso-causal-s0
    submit_eval iq-eval-isolated-s154 isolated 154645467 \
        ../models/linear_regression/icl_lr20_isolated_causal_s154645467-exp20260911-icl-iso-causal-s154645467
    submit_eval iq-eval-isolated-s657 isolated 65765443 \
        ../models/linear_regression/icl_lr20_isolated_causal_s65765443-exp20260911-icl-iso-causal-s65765443
}

supervision() {
    for seed in 0 154645467; do
        submit_job train.py conf/experiments/icl_lr20_support20_query1.yaml \
            "icl-s20-q1-s${seed}" "icl_lr20_s20_q1_s${seed}" \
            "exp20260915-icl-s20-q1-s${seed}" --training.seed "$seed"
        submit_job train.py conf/experiments/icl_lr20_support20_query5.yaml \
            "icl-s20-q5-s${seed}" "icl_lr20_s20_q5_s${seed}" \
            "exp20260915-icl-s20-q5-s${seed}" --training.seed "$seed"
        submit_job train.py conf/experiments/icl_lr20_support20_query5_isolated.yaml \
            "icl-s20-q5-iso-s${seed}" "icl_lr20_s20_q5_isolated_s${seed}" \
            "exp20260915-icl-s20-q5-iso-s${seed}" --training.seed "$seed"
        submit_job train_meta.py conf/experiments/maml_lr20_support20_query1_stable.yaml \
            "maml-s20-q1-i5-s${seed}" "maml_lr20_s20_q1_i5_s${seed}" \
            "exp20260915-maml-s20-q1-i5-s${seed}" --training.seed "$seed"
        submit_job train_meta.py conf/experiments/maml_lr20_support20_query5_stable.yaml \
            "maml-s20-q5-i5-s${seed}" "maml_lr20_s20_q5_i5_s${seed}" \
            "exp20260915-maml-s20-q5-i5-s${seed}" --training.seed "$seed"
    done
}

supervision_remaining() {
    # q=1 ICL is launched with priority1; this stage is the second group of
    # eight and deliberately excludes those already-running jobs.
    for seed in 0 154645467; do
        submit_job train.py conf/experiments/icl_lr20_support20_query5.yaml \
            "icl-s20-q5-s${seed}" "icl_lr20_s20_q5_s${seed}" \
            "exp20260915-icl-s20-q5-s${seed}" --training.seed "$seed"
        submit_job train.py conf/experiments/icl_lr20_support20_query5_isolated.yaml \
            "icl-s20-q5-iso-s${seed}" "icl_lr20_s20_q5_isolated_s${seed}" \
            "exp20260915-icl-s20-q5-iso-s${seed}" --training.seed "$seed"
        submit_job train_meta.py conf/experiments/maml_lr20_support20_query1_stable.yaml \
            "maml-s20-q1-i5-s${seed}" "maml_lr20_s20_q1_i5_s${seed}" \
            "exp20260915-maml-s20-q1-i5-s${seed}" --training.seed "$seed"
        submit_job train_meta.py conf/experiments/maml_lr20_support20_query5_stable.yaml \
            "maml-s20-q5-i5-s${seed}" "maml_lr20_s20_q5_i5_s${seed}" \
            "exp20260915-maml-s20-q5-i5-s${seed}" --training.seed "$seed"
    done
}

context_lengths() {
    context_icl
    context_maml
}

context_icl() {
    for regime in mixed under over; do
        submit_job train.py "conf/experiments/icl_lr20_context_choices_${regime}.yaml" \
            "icl-context-joint-${regime}-s0-v2" \
            "icl_lr20_context_joint_${regime}_s0_v2" \
            "exp20260915-icl-context-joint-${regime}-s0-v2" --training.seed 0
    done
}

icl_context_under() {
    submit_job train.py conf/experiments/icl_lr20_context_choices_under.yaml \
        icl-context-joint-under-s0-v2 icl_lr20_context_joint_under_s0_v2 \
        exp20260915-icl-context-joint-under-s0-v2 --training.seed 0
}

context_maml() {
    for regime in mixed under over; do
        submit_job train_meta.py "conf/experiments/maml_lr20_context_choices_${regime}.yaml" \
            "maml-context-${regime}-par-i5-s0-v2" \
            "maml_lr20_context_${regime}_parallel_i5_s0_v2" \
            "exp20260915-maml-context-${regime}-parallel-i5-s0-v2" --training.seed 0
        submit_job train_meta.py "conf/experiments/maml_lr20_context_sequential_${regime}.yaml" \
            "maml-context-${regime}-seq-i5-s0-v2" \
            "maml_lr20_context_${regime}_sequential_i5_s0_v2" \
            "exp20260915-maml-context-${regime}-sequential-i5-s0-v2" --training.seed 0
    done
}

multifunction() {
    for count in 1 2 3; do
        submit_job train.py "conf/experiments/icl_lr5_multifunction_m${count}.yaml" \
            "icl-lr5-mf${count}-s0" "icl_lr5_multifunction_m${count}_s0" \
            "exp20260915-icl-lr5-mf${count}-s0" --training.seed 0
        submit_job train_meta.py "conf/experiments/maml_lr5_multifunction_m${count}.yaml" \
            "maml-lr5-mf${count}-i5-s0" "maml_lr5_multifunction_m${count}_i5_s0" \
            "exp20260915-maml-lr5-mf${count}-i5-s0" --training.seed 0
    done
}

breadth_group3() {
    # Eight jobs: the three higher-priority ICL context pilots, all three ICL
    # multi-function pilots, and the two cheaper MAML multi-function pilots.
    context_icl
    for count in 1 2 3; do
        submit_job train.py "conf/experiments/icl_lr5_multifunction_m${count}.yaml" \
            "icl-lr5-mf${count}-s0" "icl_lr5_multifunction_m${count}_s0" \
            "exp20260915-icl-lr5-mf${count}-s0" --training.seed 0
    done
    for count in 1 2; do
        submit_job train_meta.py "conf/experiments/maml_lr5_multifunction_m${count}.yaml" \
            "maml-lr5-mf${count}-i5-s0" "maml_lr5_multifunction_m${count}_i5_s0" \
            "exp20260915-maml-lr5-mf${count}-i5-s0" --training.seed 0
    done
}

breadth_group3_overflow() {
    # H200 accepted the mixed/under context pilots; these are the six
    # zero-allocation jobs that continue down the fallback order.
    submit_job train.py conf/experiments/icl_lr20_context_choices_over.yaml \
        icl-context-joint-over-s0-v2 icl_lr20_context_joint_over_s0_v2 \
        exp20260915-icl-context-joint-over-s0-v2 --training.seed 0
    for count in 1 2 3; do
        submit_job train.py "conf/experiments/icl_lr5_multifunction_m${count}.yaml" \
            "icl-lr5-mf${count}-s0" "icl_lr5_multifunction_m${count}_s0" \
            "exp20260915-icl-lr5-mf${count}-s0" --training.seed 0
    done
    for count in 1 2; do
        submit_job train_meta.py "conf/experiments/maml_lr5_multifunction_m${count}.yaml" \
            "maml-lr5-mf${count}-i5-s0" "maml_lr5_multifunction_m${count}_i5_s0" \
            "exp20260915-maml-lr5-mf${count}-i5-s0" --training.seed 0
    done
}

multifunction_remaining() {
    submit_job train_meta.py conf/experiments/maml_lr5_multifunction_m3.yaml \
        maml-lr5-mf3-i5-s0 maml_lr5_multifunction_m3_i5_s0 \
        exp20260915-maml-lr5-mf3-i5-s0 --training.seed 0
}

finalize_decision_trees() {
    submit_job train.py conf/experiments/icl_dt_support65_query20.yaml \
        finalize-icl-dt-s65-q20-s0 icl_dt_s65_q20_causal_s0 \
        exp20260915-icl-dt-s65-q20-s0 --training.seed 0
    submit_job train.py conf/experiments/icl_dt_support65_query20_isolated.yaml \
        finalize-icl-dt-s65-q20-iso-s0 icl_dt_s65_q20_isolated_s0 \
        exp20260915-icl-dt-s65-q20-iso-s0 --training.seed 0
}

finalize_multifunction_remaining() {
    for count in 2 3; do
        submit_job train.py "conf/experiments/icl_lr5_multifunction_m${count}.yaml" \
            "finalize-icl-lr5-mf${count}-s0" "icl_lr5_multifunction_m${count}_s0" \
            "exp20260915-icl-lr5-mf${count}-s0" --training.seed 0
    done
}

finalize_completed() {
    for count in 1 2 3; do
        submit_job train.py "conf/experiments/icl_lr5_multifunction_m${count}.yaml" \
            "finalize-icl-lr5-mf${count}-s0" "icl_lr5_multifunction_m${count}_s0" \
            "exp20260915-icl-lr5-mf${count}-s0" --training.seed 0
    done
    submit_job train.py conf/experiments/icl_segmented_pretraining.yaml \
        finalize-segmented-pretrain-s0 segmented_pretraining_s0_v2 \
        exp20260915-segmented-pretrain-s0-v2 --training.seed 0
}

pretraining() {
    submit_job train.py conf/experiments/icl_segmented_pretraining.yaml \
        segmented-pretrain-s0 segmented_pretraining_s0_v2 \
        exp20260915-segmented-pretrain-s0-v2 --training.seed 0
}

maml_stability() {
    submit_job train_meta.py conf/experiments/maml_lr20_support20_query20_stable.yaml \
        maml-s20-q20-i5-resume-s0 maml_lr20_s20_q20_steps5_stable_s0 \
        b92d63fa-31ba-4a24-9860-1728b3e5c32a \
        --training.seed 0 --training.train_steps 500000 --meta.num_inner_steps 5
    submit_job train_meta.py conf/experiments/maml_lr20_support20_query20_stable.yaml \
        maml-s20-q20-i5-s154 maml_lr20_s20_q20_steps5_stable_s154645467 \
        exp20260915-maml-s20-q20-i5-s154 \
        --training.seed 154645467 --training.train_steps 500000 --meta.num_inner_steps 5
}

decision_trees() {
    submit_job train.py conf/experiments/icl_dt_support65_query20.yaml \
        icl-dt-s65-q20-s0 icl_dt_s65_q20_causal_s0 \
        exp20260915-icl-dt-s65-q20-s0 --training.seed 0
    submit_job train.py conf/experiments/icl_dt_support65_query20_isolated.yaml \
        icl-dt-s65-q20-iso-s0 icl_dt_s65_q20_isolated_s0 \
        exp20260915-icl-dt-s65-q20-iso-s0 --training.seed 0
    submit_job train_meta.py conf/experiments/maml_dt_support65_query20_stable.yaml \
        maml-dt-s65-q20-i5-s0 maml_dt_s65_q20_i5_s0_v2 \
        exp20260915-maml-dt-s65-q20-i5-s0-v2 --training.seed 0
}

maml_decision_tree() {
    submit_job train_meta.py conf/experiments/maml_dt_support65_query20_stable.yaml \
        maml-dt-s65-q20-i5-s0 maml_dt_s65_q20_i5_s0_v2 \
        exp20260915-maml-dt-s65-q20-i5-s0-v2 --training.seed 0
}

breadth_group4() {
    # Seven remaining ungated pilots.  The three MAML context-generalization
    # runs are not included; they remain gated on the new q=20 stability seed.
    multifunction_remaining
    pretraining
    maml_stability
    decision_trees
}

case "$STAGE" in
    priority1) priority1 ;;
    supervision) supervision ;;
    supervision_remaining) supervision_remaining ;;
    context_lengths) context_lengths ;;
    context_icl) context_icl ;;
    icl_context_under) icl_context_under ;;
    context_maml) context_maml ;;
    multifunction) multifunction ;;
    breadth_group3) breadth_group3 ;;
    breadth_group3_overflow) breadth_group3_overflow ;;
    multifunction_remaining) multifunction_remaining ;;
    finalize_multifunction_remaining) finalize_multifunction_remaining ;;
    finalize_decision_trees) finalize_decision_trees ;;
    finalize_completed) finalize_completed ;;
    pretraining) pretraining ;;
    maml_stability) maml_stability ;;
    decision_trees) decision_trees ;;
    maml_decision_tree) maml_decision_tree ;;
    breadth_group4) breadth_group4 ;;
    all)
        priority1
        supervision
        context_lengths
        multifunction
        pretraining
        maml_stability
        decision_trees
        ;;
    *)
        echo "unknown STAGE=$STAGE" >&2
        exit 2
        ;;
esac
