#!/usr/bin/env bash
#SBATCH --job-name=vllm-uranus-ganymede-tp2-auto
#SBATCH --partition=long
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-105636-luqia-vllm-uranus-ganymede-tp2-auto/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-105636-luqia-vllm-uranus-ganymede-tp2-auto/slurm-%j.err

set -uo pipefail

JOB_NAME="20260806-105636-luqia-vllm-uranus-ganymede-tp2-auto"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE=tp2_auto_results
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=4
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:40:00

export PREFILL_NODE_OVERRIDE=uranus
export PREFILL_IP_OVERRIDE=10.1.0.5
export DECODE_NODE_OVERRIDE=ganymede
export DECODE_IP_OVERRIDE=10.1.0.3
export PROXY_IP_OVERRIDE=10.1.0.3

export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=false
export FIXED_PLACEMENT_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/fixed_placement.json"
export POLICY_LABEL_OVERRIDE=fixed_uranus_ganymede_tp2_auto
export GPU_CLOCK_CONTROL_MODE_OVERRIDE=auto
export DEFER_INSTANCE_TELEMETRY_UNTIL_READY_OVERRIDE=true
export SATURATION_ACTION_OVERRIDE=frequency
export SCALE_IN_ENABLED_OVERRIDE=false

export INITIAL_PREFILL_INSTANCES_OVERRIDE=1
export INITIAL_DECODE_INSTANCES_OVERRIDE=1
export MAX_PREFILL_INSTANCES_OVERRIDE=1
export MAX_DECODE_INSTANCES_OVERRIDE=1
export PREFILL_TP_SIZE_OVERRIDE=2
export DECODE_TP_SIZE_OVERRIDE=2
export PROXY_DEFAULT_TP_SIZE=2

export POWER_MONITOR_PREFILL_GPUS_OVERRIDE=2
export POWER_MONITOR_DECODE_GPUS_OVERRIDE=2
export VLLM_STEP_MEM_OVERRIDE=64G
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
export WINDOW_WARMUP_SECONDS_OVERRIDE=10
export BENCH_NUM_WARMUPS_OVERRIDE=0
export MIN_PROMPTS_OVERRIDE=12
export MAX_PROMPTS_OVERRIDE=48
export MAX_CONCURRENCY_OVERRIDE=32
export BENCH_SAVE_DETAILED_OVERRIDE=true

BASE_RC=0
bash "${SOURCE_JOB}/run.sbatch" || BASE_RC=$?

CHECK_RC=0
if [ "$BASE_RC" -eq 0 ]; then
  PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
  "$PYTHON_BIN" "${JOB_DIR_OVERRIDE}/check_results.py" \
    --out-dir "${JOB_DIR_OVERRIDE}/${OUT_DIR_NAME_OVERRIDE}" \
    --workloads "$WORKLOAD_FILE_OVERRIDE" \
    --expected-workloads 4 \
    --slo-ttft-ms 500 \
    --slo-tpot-ms 200 || CHECK_RC=$?
else
  echo "result_check_skipped=true base_rc=${BASE_RC}"
fi

echo "base_job_rc=${BASE_RC} result_check_rc=${CHECK_RC}"
if [ "$BASE_RC" -ne 0 ]; then
  exit "$BASE_RC"
fi
exit "$CHECK_RC"
