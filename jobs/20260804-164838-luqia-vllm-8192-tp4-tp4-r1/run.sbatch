#!/usr/bin/env bash
#SBATCH --job-name=vllm-8192-tp4-tp4-r1
#SBATCH --partition=long
#SBATCH --time=00:35:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260804-164838-luqia-vllm-8192-tp4-tp4-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260804-164838-luqia-vllm-8192-tp4-tp4-r1/slurm-%j.err

set -uo pipefail

JOB_NAME="20260804-164838-luqia-vllm-8192-tp4-tp4-r1"
ASSET_JOB_NAME="20260804-151704-luqia-vllm-8192-tp4-tp4"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
ASSET_JOB="${BROKER_ROOT}/jobs/${ASSET_JOB_NAME}"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE="fixed_tp4_tp4_8192_results"
export WORKLOAD_FILE_OVERRIDE="${ASSET_JOB}/request_trace.csv"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=4
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:35:00

export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=false
export FIXED_PLACEMENT_FILE_OVERRIDE="${ASSET_JOB}/fixed_placement.json"
export POLICY_LABEL_OVERRIDE=fixed_tp4_tp4_8192
export GPU_CLOCK_CONTROL_MODE_OVERRIDE=auto
export SATURATION_ACTION_OVERRIDE=frequency
export SCALE_IN_ENABLED_OVERRIDE=false
export INITIAL_PREFILL_INSTANCES_OVERRIDE=1
export INITIAL_DECODE_INSTANCES_OVERRIDE=1
export MAX_PREFILL_INSTANCES_OVERRIDE=1
export MAX_DECODE_INSTANCES_OVERRIDE=1
export PREFILL_TP_SIZE_OVERRIDE=4
export DECODE_TP_SIZE_OVERRIDE=4

export POWER_MONITOR_NEPTUNE_GPUS_OVERRIDE=4
export POWER_MONITOR_GANYMEDE_GPUS_OVERRIDE=4
export VLLM_STEP_MEM_OVERRIDE=128G

export MAX_MODEL_LEN_OVERRIDE=16384
export MAX_NUM_BATCHED_TOKENS_OVERRIDE=16384
export MAX_NUM_SEQS_OVERRIDE=16

export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0

export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export WINDOW_WARMUP_SECONDS_OVERRIDE=20
export BENCH_NUM_WARMUPS_OVERRIDE=0
export MIN_PROMPTS_OVERRIDE=12
export MAX_PROMPTS_OVERRIDE=64
export MAX_CONCURRENCY_OVERRIDE=64

BASE_RC=0
bash "${SOURCE_JOB}/run.sbatch" || BASE_RC=$?

METRICS_RC=0
if [ "$BASE_RC" -eq 0 ]; then
  PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
  OUT_DIR="${JOB_DIR_OVERRIDE}/${OUT_DIR_NAME_OVERRIDE}"
  "$PYTHON_BIN" "${ASSET_JOB}/check_latency_metrics.py" \
    --out-dir "$OUT_DIR" \
    --workloads "$WORKLOAD_FILE_OVERRIDE" \
    --expected-workloads 4 \
    --expected-input-len 8192 \
    --expected-prefill-instances 1 \
    --expected-decode-instances 1 \
    --expected-prefill-tp 4 \
    --expected-decode-tp 4 \
    --slo-ttft-ms 500 \
    --slo-tpot-ms 200 || METRICS_RC=$?
else
  echo "latency_metrics_check_skipped=true base_rc=${BASE_RC}"
fi

echo "base_job_rc=${BASE_RC} latency_metrics_check_rc=${METRICS_RC}"
if [ "$BASE_RC" -ne 0 ]; then
  exit "$BASE_RC"
fi
exit "$METRICS_RC"
