#!/usr/bin/env bash
#SBATCH --job-name=vllm-uranus-ganymede-tp2-predictive-dvfs
#SBATCH --partition=long
#SBATCH --time=00:50:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-215045-luqia-vllm-uranus-ganymede-tp2-predictive-dvfs/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-215045-luqia-vllm-uranus-ganymede-tp2-predictive-dvfs/slurm-%j.err

set -uo pipefail

JOB_NAME="20260806-215045-luqia-vllm-uranus-ganymede-tp2-predictive-dvfs"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
CHECKER_JOB="${BROKER_ROOT}/jobs/20260805-111027-luqia-vllm-tp2-2p2d-predictive-dvfs"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE=tp2_predictive_dvfs_results
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=5
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:50:00

# One TP=2 prefill replica on two Uranus L40S GPUs and one TP=2 decode
# replica on two Ganymede L4 GPUs.
export PREFILL_NODE_OVERRIDE=uranus
export PREFILL_IP_OVERRIDE=10.1.0.5
export DECODE_NODE_OVERRIDE=ganymede
export DECODE_IP_OVERRIDE=10.1.0.3
export PROXY_IP_OVERRIDE=10.1.0.3
export INITIAL_PREFILL_INSTANCES_OVERRIDE=1
export INITIAL_DECODE_INSTANCES_OVERRIDE=1
export MAX_PREFILL_INSTANCES_OVERRIDE=1
export MAX_DECODE_INSTANCES_OVERRIDE=1
export PREFILL_TP_SIZE_OVERRIDE=2
export DECODE_TP_SIZE_OVERRIDE=2
export PROXY_DEFAULT_TP_SIZE=2

# For each request window, predict P99 TTFT/TPOT and the minimum-power clock
# pair, then apply that pair to every GPU in the two TP groups with -lgc.
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_only
export POLICY_LABEL_OVERRIDE=uranus_ganymede_tp2_predictive_dvfs
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export SATURATION_ACTION_OVERRIDE=frequency
export SCALE_IN_ENABLED_OVERRIDE=false
export GPU_CLOCK_CONTROL_MODE_OVERRIDE=manual
export GPU_STARTUP_CLOCK_MODE_OVERRIDE=max
export CLOCK_ACK_MODE_OVERRIDE=monitor
export CLOCK_ACK_TRANSPORT_OVERRIDE=http_with_file_fallback
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=abort
export DEFER_INSTANCE_TELEMETRY_UNTIL_READY_OVERRIDE=true

export POWER_MONITOR_PREFILL_GPUS_OVERRIDE=2
export POWER_MONITOR_DECODE_GPUS_OVERRIDE=2
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
export MIN_PROMPTS_OVERRIDE=12
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
    --expected-workloads 5 \
    --expected-prefill-instances 1 \
    --expected-decode-instances 1 \
    --expected-tp 2 \
    --expected-gpus-per-role 2 \
    --expected-prefill-node uranus \
    --expected-decode-node ganymede \
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
