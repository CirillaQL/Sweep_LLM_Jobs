#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-observability
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260729-231825-luqia-vllm-pd-observability-request-traces/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260729-231825-luqia-vllm-pd-observability-request-traces/slurm-%j.err

set -uo pipefail

JOB_NAME="20260729-231825-luqia-vllm-pd-observability-request-traces"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
OUT_DIR="${JOB_DIR}/observability_results"
BASE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
MODEL="mistralai/Mistral-7B-v0.1"

NODE_SCRIPT="${JOB_DIR}/start_pd_node_observability.sh"
PROXY_SCRIPT="${JOB_DIR}/pd_proxy_observability.py"
OTEL_COLLECTOR="${JOB_DIR}/otel_collector.py"
METRICS_COLLECTOR="${JOB_DIR}/metrics_collector.py"
BENCH_SCRIPT="${JOB_DIR}/stream_bench.py"
DRAIN_SCRIPT="${JOB_DIR}/drain_observer.py"
LOG_TO_CSV_SCRIPT="${JOB_DIR}/vllm_log_to_csv.py"
QUEUE_JOIN_SCRIPT="${JOB_DIR}/join_queue_state.py"
WORKLOAD_FILE="${JOB_DIR}/workload_windows.csv"
PLACEMENT_FILE="${JOB_DIR}/fixed_placement.json"
CLOCK_PROBE="${BASE_JOB}/clock_probe.py"
CLOCK_CONTROL_DIR="${OUT_DIR}/clock_control"

PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
PREFILL_HTTP_PORT=$((32000 + PORT_OFFSET))
DECODE_HTTP_PORT=$((33000 + PORT_OFFSET))
KV_PORT=$((34000 + PORT_OFFSET))
OTLP_PORT=$((35000 + PORT_OFFSET))
PROXY_IP=10.1.0.3
PREFILL_IP=10.1.0.6
DECODE_IP=10.1.0.3
OTLP_TRACES_ENDPOINT="http://${PROXY_IP}:${OTLP_PORT}/v1/traces"

export HOME="/data/users/chjing"
export USER=chjing
export LOGNAME=chjing
export HF_HOME="${HOME}/.cache/huggingface"
export XDG_CACHE_HOME="${HOME}/.cache"
export PATH="${ENV_DIR}/bin:${PATH}"
export JOB_DIR OUT_DIR ENV_DIR PYTHON_BIN VLLM_BIN MODEL PLACEMENT_FILE CLOCK_PROBE
export CLOCK_CONTROL_DIR PROXY_IP PROXY_HTTP_PORT PROXY_REGISTER_PORT
export PREFILL_HTTP_PORT DECODE_HTTP_PORT PREFILL_IP DECODE_IP KV_PORT
export OTLP_TRACES_ENDPOINT
export GPU_CLOCK_CONTROL_MODE_OVERRIDE=auto
export CLOCK_ACK_MODE_OVERRIDE=monitor
export MAX_NUM_SEQS_OVERRIDE=64
export PROXY_EVENTS_CSV="${OUT_DIR}/proxy_events.csv"

# Avoid an SMT-adjusted inherited value conflicting with nested Slurm steps.
unset SLURM_CPUS_PER_TASK

mkdir -p "$OUT_DIR" "$CLOCK_CONTROL_DIR"
ulimit -n 65536
echo "0 0" > "${CLOCK_CONTROL_DIR}/neptune.ack"
echo "0 0" > "${CLOCK_CONTROL_DIR}/ganymede.ack"

OTEL_PID=""
PROXY_PID=""
SERVER_STEP_PID=""
METRICS_PID=""
BENCH_PID=""

stop_pid() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_pid "$BENCH_PID"
  stop_pid "$METRICS_PID"
  stop_pid "$SERVER_STEP_PID"
  # The OTLP exporter flushes during vLLM shutdown; keep the receiver alive.
  sleep 5
  stop_pid "$PROXY_PID"
  stop_pid "$OTEL_PID"
  echo "job_cleanup_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_url() {
  local label="$1"
  local url="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --connect-timeout 2 --max-time 3 "$url" >/dev/null 2>&1; then
      echo "ready label=${label} url=${url}"
      return 0
    fi
    sleep 2
  done
  echo "readiness_timeout label=${label} url=${url}"
  return 1
}

wait_for_registry() {
  local deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --connect-timeout 2 --max-time 3 \
      "http://127.0.0.1:${PROXY_HTTP_PORT}/registry" \
      | "$PYTHON_BIN" -c \
        'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d.get("prefill") and d.get("decode") else 1)' \
      >/dev/null 2>&1; then
      echo "pd_registry_ready=true"
      return 0
    fi
    sleep 2
  done
  echo "pd_registry_ready=false"
  return 1
}

echo "job_name=${JOB_NAME}"
echo "batch_host=$(hostname -s)"
echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "slurm_nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "model=${MODEL}"
echo "max_num_seqs=${MAX_NUM_SEQS_OVERRIDE}"
echo "otlp_traces_endpoint=${OTLP_TRACES_ENDPOINT}"
echo "observability_flags=kv_cache_metrics,enable_logging_iteration_details,collect_detailed_traces"
echo "request_id_transport=X-Request-Id"
echo "unix_ts,slurm_job_id,job_name,model,prefill_node,decode_node,max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,kv_connector,kv_cache_metrics,kv_cache_metrics_sample,enable_logging_iteration_details,collect_detailed_traces,enable_prefix_caching,otlp_endpoint" \
  > "${OUT_DIR}/job_metadata.csv"
echo "$(date +%s.%N),${SLURM_JOB_ID:-},${JOB_NAME},${MODEL},neptune,ganymede,64,4096,0.82,P2pNcclConnector,true,1.0,true,all,true,${OTLP_TRACES_ENDPOINT}" \
  >> "${OUT_DIR}/job_metadata.csv"

if [ "$(hostname -s)" != ganymede ]; then
  echo "expected_batch_host_ganymede=true actual=$(hostname -s)"
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true"
  exit 3
fi
for required in "$NODE_SCRIPT" "$PROXY_SCRIPT" "$OTEL_COLLECTOR" \
  "$METRICS_COLLECTOR" "$BENCH_SCRIPT" "$DRAIN_SCRIPT" "$WORKLOAD_FILE" \
  "$LOG_TO_CSV_SCRIPT" "$QUEUE_JOIN_SCRIPT" "$PLACEMENT_FILE" "$CLOCK_PROBE"; do
  if [ ! -s "$required" ]; then
    echo "required_artifact_missing=${required}"
    exit 4
  fi
done

echo "phase=preflight"
if ! srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" preflight > "${OUT_DIR}/preflight.log" 2>&1; then
  cat "${OUT_DIR}/preflight.log"
  exit 5
fi
cat "${OUT_DIR}/preflight.log"

echo "phase=start_otlp_collector"
"$PYTHON_BIN" -u "$OTEL_COLLECTOR" \
  --host 0.0.0.0 \
  --port "$OTLP_PORT" \
  --output-dir "$OUT_DIR" \
  > "${OUT_DIR}/otel_collector.log" 2>&1 &
OTEL_PID=$!
wait_for_url otlp "http://127.0.0.1:${OTLP_PORT}/health" 60 || exit 6

echo "phase=start_proxy"
export PROXY_REGISTER_HOST=0.0.0.0
export PROXY_HTTP_HOST=0.0.0.0
"$PYTHON_BIN" -u "$PROXY_SCRIPT" > "${OUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_url proxy "http://127.0.0.1:${PROXY_HTTP_PORT}/health" 60 || exit 7

echo "phase=start_pd_servers"
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" serve > "${OUT_DIR}/server_step.log" 2>&1 &
SERVER_STEP_PID=$!
wait_for_url prefill "http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/v1/models" 600 || exit 8
wait_for_url decode "http://${DECODE_IP}:${DECODE_HTTP_PORT}/v1/models" 600 || exit 9
wait_for_registry || exit 10

echo "phase=start_metrics_collector"
"$PYTHON_BIN" -u "$METRICS_COLLECTOR" \
  --endpoint "prefill=http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/metrics" \
  --endpoint "decode=http://${DECODE_IP}:${DECODE_HTTP_PORT}/metrics" \
  --output-dir "$OUT_DIR" \
  --interval 0.5 \
  > "${OUT_DIR}/metrics_collector.log" 2>&1 &
METRICS_PID=$!

echo "phase=streaming_benchmark"
"$PYTHON_BIN" -u "$BENCH_SCRIPT" \
  --endpoint "http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1/completions" \
  --model "$MODEL" \
  --workloads "$WORKLOAD_FILE" \
  --output-dir "$OUT_DIR" \
  > "${OUT_DIR}/stream_bench.log" 2>&1 &
BENCH_PID=$!
BENCH_RC=0
wait "$BENCH_PID" || BENCH_RC=$?
BENCH_PID=""
cat "${OUT_DIR}/stream_bench.log"
echo "stream_bench_rc=${BENCH_RC}"

echo "phase=drain_observation"
DRAIN_RC=0
"$PYTHON_BIN" -u "$DRAIN_SCRIPT" \
  --endpoint "prefill=http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/metrics" \
  --endpoint "decode=http://${DECODE_IP}:${DECODE_HTTP_PORT}/metrics" \
  --output-dir "$OUT_DIR" \
  --interval 0.5 \
  --max-seconds 180 \
  --zero-samples 3 \
  > "${OUT_DIR}/drain_observer.log" 2>&1 || DRAIN_RC=$?
cat "${OUT_DIR}/drain_observer.log"
echo "drain_observer_rc=${DRAIN_RC}"

stop_pid "$METRICS_PID"
METRICS_PID=""
sleep 10

echo "phase=queue_state_join"
"$PYTHON_BIN" "$QUEUE_JOIN_SCRIPT" \
  --requests "${OUT_DIR}/requests.csv" \
  --proxy-events "${OUT_DIR}/proxy_events.csv" \
  --metrics "${OUT_DIR}/vllm_metrics_snapshots.csv" \
  --output "${OUT_DIR}/queue_state_at_arrival.csv"

echo "phase=vllm_log_csv_export"
"$PYTHON_BIN" "$LOG_TO_CSV_SCRIPT" \
  --log "prefill=${OUT_DIR}/prefill_server.log" \
  --log "decode=${OUT_DIR}/decode_server.log" \
  --output "${OUT_DIR}/vllm_observability_log_events.csv"

echo "phase=artifact_check"
for artifact in requests.csv token_timestamps.csv client_events.csv \
  window_summary.csv workload_events.csv vllm_metrics_long.csv \
  vllm_metrics_snapshots.csv drain_samples.csv drain_summary.csv \
  proxy_events.csv vllm_observability_log_events.csv otel_batches.csv \
  otel_spans.csv job_metadata.csv queue_state_at_arrival.csv; do
  if [ -s "${OUT_DIR}/${artifact}" ]; then
    echo "artifact=${artifact} bytes=$(wc -c < "${OUT_DIR}/${artifact}")"
  else
    echo "artifact_missing=${artifact}"
  fi
done

if [ "$BENCH_RC" -ne 0 ]; then
  exit "$BENCH_RC"
fi
echo "job_exit_rc=0"
exit 0
