#!/usr/bin/env bash
#SBATCH --job-name=vllm-queue-mns32
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260729-152910-luqia-vllm-nogate-short-high-rps-maxseqs-32/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260729-152910-luqia-vllm-nogate-short-high-rps-maxseqs-32/slurm-%j.err

set -uo pipefail

JOB_NAME="20260729-152910-luqia-vllm-nogate-short-high-rps-maxseqs-32"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
OUT_DIR="${JOB_DIR}/queue_backlog_results"
BASELINE_JOB="${BROKER_ROOT}/jobs/20260729-120145-luqia-vllm-nogate-short-high-rps-queue"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"

export JOB_DIR_OVERRIDE="$JOB_DIR"
export OUT_DIR_NAME_OVERRIDE=queue_backlog_results
export WORKLOAD_FILE_OVERRIDE="${BASELINE_JOB}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=3
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_only
export POLICY_LABEL_OVERRIDE=gate_off_short_high_rps_queue_maxseqs_32
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
unset MAX_L4_FREQ_OVERRIDE
export MAX_L40S_FREQ_OVERRIDE=2520
export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0
export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=900
export MIN_PROMPTS_OVERRIDE=12
export MAX_PROMPTS_OVERRIDE=1920
export MAX_CONCURRENCY_OVERRIDE=512
export MAX_NUM_SEQS_OVERRIDE=32
export BENCH_SAVE_DETAILED_OVERRIDE=true
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=continue

mkdir -p "$OUT_DIR"
PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
PREFILL_HTTP_PORT=$((32000 + PORT_OFFSET))
DECODE_HTTP_PORT=$((33000 + PORT_OFFSET))
METRICS_PID=""

stop_metrics() {
  if [ -n "$METRICS_PID" ] && kill -0 "$METRICS_PID" 2>/dev/null; then
    kill "$METRICS_PID" 2>/dev/null || true
    wait "$METRICS_PID" 2>/dev/null || true
  fi
  METRICS_PID=""
}

trap stop_metrics EXIT
"$PYTHON_BIN" -u "${BASELINE_JOB}/metrics_monitor.py" \
  --output "${OUT_DIR}/vllm_queue_metrics.csv" \
  --interval 1 \
  --prefill-url "http://10.1.0.6:${PREFILL_HTTP_PORT}/metrics" \
  --decode-url "http://10.1.0.3:${DECODE_HTTP_PORT}/metrics" &
METRICS_PID=$!

bash "${SOURCE_JOB}/run.sbatch"
PARENT_RC=$?
stop_metrics

"$PYTHON_BIN" "${BASELINE_JOB}/analyze_queue_results.py" \
  --out-dir "$OUT_DIR" \
  --workload-id short_high_rps_queue \
  --rate 32 \
  --bin-seconds 5 || true

exit "$PARENT_RC"
