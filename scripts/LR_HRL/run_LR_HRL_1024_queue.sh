#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/gpufree-data/quadruped_arm_LR_HRL}"
PYTHON="${PYTHON:-/root/gpufree-data/envs/isaaclab/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/logs/LR_HRL_queue_${STAMP}"
mkdir -p "${LOG_DIR}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT_DIR}/source/quadruped_arm:${PYTHONPATH:-}"

run_one() {
    local task_id="$1"
    local device="$2"
    local run_name="$3"
    local log_file="${LOG_DIR}/${run_name}.log"
    echo "[START] ${task_id} ${run_name} on ${device} -> ${log_file}"
    (
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/rsl_rl/train.py \
            --task "${task_id}" \
            --num_envs 1024 \
            --max_iterations 1024 \
            --headless \
            --device "${device}" \
            --run_name "${run_name}"
    ) >"${log_file}" 2>&1
    echo "[DONE] ${task_id} ${run_name}"
}

run_pair() {
    local family="$1"
    local hrl_task="$2"
    local baseline_task="$3"
    local hrl_run="LR_HRL_${family}_1024_${STAMP}"
    local base_run="LR_Baseline_${family}_1024_${STAMP}"

    run_one "${hrl_task}" "cuda:0" "${hrl_run}" &
    local pid_hrl=$!
    run_one "${baseline_task}" "cuda:1" "${base_run}" &
    local pid_base=$!

    wait "${pid_hrl}"
    wait "${pid_base}"
    echo "[PAIR_DONE] ${family}"
}

run_pair "route" "LR-HRL-Route-v0" "LR-Baseline-Route-v0"
run_pair "slalom" "LR-HRL-Slalom-v0" "LR-Baseline-Slalom-v0"
run_pair "narrow" "LR-HRL-Narrow-v0" "LR-Baseline-Narrow-v0"
run_pair "manip" "LR-HRL-Manip-v0" "LR-Baseline-Manip-v0"
run_pair "grasp" "LR-HRL-Grasp-v0" "LR-Baseline-Grasp-v0"
run_pair "recovery" "LR-HRL-Recovery-v0" "LR-Baseline-Recovery-v0"

echo "[ALL_DONE] LR_HRL 1024 queue finished at $(date -Is)"
