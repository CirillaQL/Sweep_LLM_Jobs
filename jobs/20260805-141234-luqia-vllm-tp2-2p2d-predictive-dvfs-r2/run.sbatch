#!/usr/bin/env bash
#SBATCH --job-name=vllm-tp2-2p2d-predictive-dvfs-r2
#SBATCH --partition=long
#SBATCH --time=00:50:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260805-141234-luqia-vllm-tp2-2p2d-predictive-dvfs-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260805-141234-luqia-vllm-tp2-2p2d-predictive-dvfs-r2/slurm-%j.err

set -uo pipefail

JOB_NAME="20260805-141234-luqia-vllm-tp2-2p2d-predictive-dvfs-r2"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
CHECKER_JOB="${BROKER_ROOT}/jobs/20260805-111027-luqia-vllm-tp2-2p2d-predictive-dvfs"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE=tp2_2p2d_predictive_dvfs_results
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=8
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:50:00

export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_only
export POLICY_LABEL_OVERRIDE=tp2_2p2d_predictive_dvfs
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export SATURATION_ACTION_OVERRIDE=frequency
export SCALE_IN_ENABLED_OVERRIDE=false

export INITIAL_PREFILL_INSTANCES_OVERRIDE=2
export INITIAL_DECODE_INSTANCES_OVERRIDE=2
export MAX_PREFILL_INSTANCES_OVERRIDE=2
export MAX_DECODE_INSTANCES_OVERRIDE=2
export PREFILL_TP_SIZE_OVERRIDE=2
export DECODE_TP_SIZE_OVERRIDE=2
export PROXY_DEFAULT_TP_SIZE=2

export GPU_CLOCK_CONTROL_MODE_OVERRIDE=manual
export CLOCK_ACK_MODE_OVERRIDE=monitor
export CLOCK_ACK_TRANSPORT_OVERRIDE=http_with_file_fallback
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=abort

export POWER_MONITOR_NEPTUNE_GPUS_OVERRIDE=4
export POWER_MONITOR_GANYMEDE_GPUS_OVERRIDE=4
export VLLM_STEP_MEM_OVERRIDE=64G
export MAX_MODEL_LEN_OVERRIDE=4096
export MAX_NUM_BATCHED_TOKENS_OVERRIDE=4096
export MAX_NUM_SEQS_OVERRIDE=32

export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0

export WINDOW_WARMUP_SECONDS_OVERRIDE=5
export BENCH_NUM_WARMUPS_OVERRIDE=0
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export MIN_PROMPTS_OVERRIDE=16
export MAX_PROMPTS_OVERRIDE=96
export MAX_CONCURRENCY_OVERRIDE=32
export BENCH_SAVE_DETAILED_OVERRIDE=true
export RUN_REFERENCE_COMPARISON_OVERRIDE=false

BASE_RC=0
bash "${SOURCE_JOB}/run.sbatch" || BASE_RC=$?

CHECK_RC=0
if [ "$BASE_RC" -eq 0 ]; then
  PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
  OUT_DIR="${JOB_DIR_OVERRIDE}/${OUT_DIR_NAME_OVERRIDE}"
  "$PYTHON_BIN" "${CHECKER_JOB}/check_prediction_error.py" \
    --out-dir "$OUT_DIR" \
    --workloads "$WORKLOAD_FILE_OVERRIDE" \
    --expected-workloads 8 \
    --expected-prefill-instances 2 \
    --expected-decode-instances 2 \
    --expected-tp 2 \
    --expected-gpus-per-role 4 \
    --slo-ttft-ms 500 \
    --slo-tpot-ms 200 \
    --clock-tolerance-mhz 90 || CHECK_RC=$?
else
  echo "prediction_error_check_skipped=true base_rc=${BASE_RC}"
fi

echo "base_job_rc=${BASE_RC} prediction_error_check_rc=${CHECK_RC}"
if [ "$BASE_RC" -ne 0 ]; then
  exit "$BASE_RC"
fi
exit "$CHECK_RC"
