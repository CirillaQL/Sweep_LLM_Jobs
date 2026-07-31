#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-observability-fixes-r1
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-105244-luqia-vllm-pd-observability-fixes-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-105244-luqia-vllm-pd-observability-fixes-r1/slurm-%j.err

set -uo pipefail

JOB_NAME="20260731-105244-luqia-vllm-pd-observability-fixes-r1"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
OUT_DIR="${JOB_DIR}/observability_results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
MODEL="mistralai/Mistral-7B-v0.1"
WORK_ROOT_BASE="/data/users/chjing/vllm_job_work"
JOB_WORK_DIR="${WORK_ROOT_BASE}/${SLURM_JOB_ID:-manual}"
BATCH_WORK_DIR="${JOB_WORK_DIR}/batch-ganymede"

NODE_SCRIPT="${JOB_DIR}/start_pd_node_chjing_workdir.sh"
PROXY_SCRIPT="${JOB_DIR}/pd_proxy_observability.py"
OTEL_COLLECTOR="${JOB_DIR}/otel_collector.py"
METRICS_COLLECTOR="${JOB_DIR}/metrics_collector.py"
BENCH_SCRIPT="${JOB_DIR}/stream_bench.py"
DRAIN_SCRIPT="${JOB_DIR}/drain_observer.py"
LOG_TO_CSV_SCRIPT="${JOB_DIR}/vllm_log_to_csv.py"
QUEUE_JOIN_SCRIPT="${JOB_DIR}/join_queue_state.py"
KV_TRANSFER_EXTRACTOR="${JOB_DIR}/kv_transfer_extractor.py"
WORKLOAD_FILE="${JOB_DIR}/workload_windows.csv"
CLICKHOUSE_UPLOADER="${JOB_DIR}/clickhouse_batch_uploader.py"
CLICKHOUSE_CREDENTIALS="${JOB_DIR}/clickhouse_credentials.env"
CLICKHOUSE_STATE="${OUT_DIR}/clickhouse_upload_state.json"
JOB_STARTED_UNIX_NS="$(date +%s%N)"

set -a
# This is a temporary database credential authorized for this disposable test.
source "$CLICKHOUSE_CREDENTIALS"
set +a

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

OTEL_BUNDLE="${JOB_DIR}/vendor/otel_bundle.zip"
if [ ! -s "$OTEL_BUNDLE" ]; then
  echo "vendored_otel_bundle_missing=true path=${OTEL_BUNDLE}"
  exit 3
fi

mkdir -p "${BATCH_WORK_DIR}/xdg-cache" \
  "${BATCH_WORK_DIR}/xdg-config" "${BATCH_WORK_DIR}/tmp"
cd "$BATCH_WORK_DIR" || exit 3

export HF_HOME="/data/users/chjing/.cache/huggingface"
export XDG_CACHE_HOME="${BATCH_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${BATCH_WORK_DIR}/xdg-config"
export TMPDIR="${BATCH_WORK_DIR}/tmp"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${OTEL_BUNDLE}${PYTHONPATH:+:${PYTHONPATH}}"
export JOB_DIR OUT_DIR ENV_DIR PYTHON_BIN VLLM_BIN MODEL WORK_ROOT_BASE
export PROXY_IP PROXY_HTTP_PORT PROXY_REGISTER_PORT
export PREFILL_HTTP_PORT DECODE_HTTP_PORT PREFILL_IP DECODE_IP KV_PORT
export OTLP_TRACES_ENDPOINT
export MAX_NUM_SEQS_OVERRIDE=64
export PROXY_EVENTS_CSV="${OUT_DIR}/proxy_events.csv"

# Avoid an SMT-adjusted inherited value conflicting with nested Slurm steps.
unset SLURM_CPUS_PER_TASK

mkdir -p "$OUT_DIR"
ulimit -n 65536

OTEL_PID=""
PROXY_PID=""
SERVER_STEP_PID=""
METRICS_PID=""
BENCH_PID=""
CLICKHOUSE_PID=""

stop_pid() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup_job_work_dir() {
  local job_id="${SLURM_JOB_ID:-}"
  local expected_job_work_dir=""

  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "job_work_cleanup_skipped=true reason=invalid_slurm_job_id value=${job_id:-unset}"
    return 1
  fi

  expected_job_work_dir="${WORK_ROOT_BASE}/${job_id}"
  if [ "$JOB_WORK_DIR" != "$expected_job_work_dir" ] \
    || [ "$JOB_WORK_DIR" = "$WORK_ROOT_BASE" ]; then
    echo "job_work_cleanup_skipped=true reason=unsafe_path path=${JOB_WORK_DIR}"
    return 1
  fi

  if [ ! -d "$JOB_WORK_DIR" ]; then
    echo "job_work_cleanup_success=true already_absent=true path=${JOB_WORK_DIR}"
    return 0
  fi

  # Leave the directory before removing it and delete only this numeric Job ID.
  if ! cd "$JOB_DIR"; then
    echo "job_work_cleanup_skipped=true reason=cannot_leave_work_dir path=${JOB_WORK_DIR}"
    return 1
  fi
  if ! rm -rf -- "$JOB_WORK_DIR"; then
    echo "job_work_cleanup_success=false reason=remove_failed path=${JOB_WORK_DIR}"
    return 1
  fi
  if [ -e "$JOB_WORK_DIR" ]; then
    echo "job_work_cleanup_success=false path=${JOB_WORK_DIR}"
    return 1
  fi

  echo "job_work_cleanup_success=true path=${JOB_WORK_DIR}"
}

cleanup() {
  local rc=$?
  local work_cleanup_rc=0
  trap - EXIT INT TERM
  stop_pid "$BENCH_PID"
  stop_pid "$METRICS_PID"
  stop_pid "$CLICKHOUSE_PID"
  stop_pid "$SERVER_STEP_PID"
  # Keep the OTLP receiver alive while vLLM flushes pending spans.
  sleep 5
  stop_pid "$PROXY_PID"
  stop_pid "$OTEL_PID"
  cleanup_job_work_dir || work_cleanup_rc=$?
  if [ "$rc" -eq 0 ] && [ "$work_cleanup_rc" -ne 0 ]; then
    rc="$work_cleanup_rc"
  fi
  echo "job_work_cleanup_rc=${work_cleanup_rc}"
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

wait_for_server_url() {
  local label="$1"
  local url="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --connect-timeout 2 --max-time 3 "$url" >/dev/null 2>&1; then
      echo "ready label=${label} url=${url}"
      return 0
    fi
    if [ -n "$SERVER_STEP_PID" ] && ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
      echo "server_step_exited_before_ready=true label=${label}"
      echo "server_step_log_tail_begin"
      tail -n 160 "${OUT_DIR}/server_step.log" 2>/dev/null || true
      echo "decode_server_log_tail_begin"
      tail -n 160 "${OUT_DIR}/decode_server.log" 2>/dev/null || true
      echo "prefill_server_log_tail_begin"
      tail -n 160 "${OUT_DIR}/prefill_server.log" 2>/dev/null || true
      return 1
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
echo "gpu_dvfs=automatic_hardware manual_frequency_control=false scheduler_prediction=false"
echo "attention_backend=FLASH_ATTN"
echo "work_root_base=${WORK_ROOT_BASE}"
echo "batch_work_dir=${BATCH_WORK_DIR} batch_cwd=${PWD}"
echo "vendored_otel_bundle=${OTEL_BUNDLE}"
echo "unix_ts,slurm_job_id,job_name,model,prefill_node,decode_node,max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,kv_connector,kv_cache_metrics,kv_cache_metrics_sample,enable_logging_iteration_details,collect_detailed_traces,enable_prefix_caching,otlp_endpoint,dvfs_mode,manual_frequency_control,scheduler_prediction,attention_backend,work_root_base" \
  > "${OUT_DIR}/job_metadata.csv"
echo "$(date +%s.%N),${SLURM_JOB_ID:-},${JOB_NAME},${MODEL},neptune,ganymede,64,4096,0.82,P2pNcclConnector,true,1.0,true,all,true,${OTLP_TRACES_ENDPOINT},automatic_hardware,false,false,FLASH_ATTN,${WORK_ROOT_BASE}" \
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
  "$LOG_TO_CSV_SCRIPT" "$QUEUE_JOIN_SCRIPT" "$KV_TRANSFER_EXTRACTOR" \
  "$CLICKHOUSE_UPLOADER" \
  "$CLICKHOUSE_CREDENTIALS"; do
  if [ ! -s "$required" ]; then
    echo "required_artifact_missing=${required}"
    exit 4
  fi
done

echo "phase=local_otel_preflight"
if ! "$PYTHON_BIN" - <<'PY'
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider

print("local_otel_preflight_ok=true")
PY
then
  echo "local_otel_preflight_ok=false"
  exit 5
fi

echo "phase=node_preflight"
if ! srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" preflight > "${OUT_DIR}/preflight.log" 2>&1; then
  cat "${OUT_DIR}/preflight.log"
  exit 6
fi
cat "${OUT_DIR}/preflight.log"

echo "phase=start_otlp_collector"
"$PYTHON_BIN" -u "$OTEL_COLLECTOR" \
  --host 0.0.0.0 \
  --port "$OTLP_PORT" \
  --output-dir "$OUT_DIR" \
  > "${OUT_DIR}/otel_collector.log" 2>&1 &
OTEL_PID=$!
wait_for_url otlp "http://127.0.0.1:${OTLP_PORT}/health" 60 || exit 7

echo "phase=start_proxy"
export PROXY_REGISTER_HOST=0.0.0.0
export PROXY_HTTP_HOST=0.0.0.0
"$PYTHON_BIN" -u "$PROXY_SCRIPT" > "${OUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_url proxy "http://127.0.0.1:${PROXY_HTTP_PORT}/health" 60 || exit 8

echo "phase=start_pd_servers"
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" serve > "${OUT_DIR}/server_step.log" 2>&1 &
SERVER_STEP_PID=$!
wait_for_server_url prefill "http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/v1/models" 600 || exit 9
wait_for_server_url decode "http://${DECODE_IP}:${DECODE_HTTP_PORT}/v1/models" 600 || exit 10
wait_for_registry || exit 11

echo "phase=start_metrics_collector"
"$PYTHON_BIN" -u "$METRICS_COLLECTOR" \
  --endpoint "prefill=http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/metrics" \
  --endpoint "decode=http://${DECODE_IP}:${DECODE_HTTP_PORT}/metrics" \
  --output-dir "$OUT_DIR" \
  --interval 0.5 \
  > "${OUT_DIR}/metrics_collector.log" 2>&1 &
METRICS_PID=$!

echo "phase=start_clickhouse_live_uploader"
"$PYTHON_BIN" -u "$CLICKHOUSE_UPLOADER" \
  --mode live \
  --output-dir "$OUT_DIR" \
  --workloads "$WORKLOAD_FILE" \
  --job-id "${SLURM_JOB_ID:-manual}" \
  --job-name "$JOB_NAME" \
  --started-unix-ns "$JOB_STARTED_UNIX_NS" \
  --interval 60 \
  --state-file "$CLICKHOUSE_STATE" \
  > "${OUT_DIR}/clickhouse_live_uploader.log" 2>&1 &
CLICKHOUSE_PID=$!
sleep 3
if ! kill -0 "$CLICKHOUSE_PID" 2>/dev/null; then
  echo "clickhouse_live_uploader_started=false"
  cat "${OUT_DIR}/clickhouse_live_uploader.log" 2>/dev/null || true
  exit 12
fi
echo "clickhouse_live_uploader_started=true interval_s=60"

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

echo "phase=freeze_all_observability_sources"
stop_pid "$CLICKHOUSE_PID"
CLICKHOUSE_PID=""
stop_pid "$PROXY_PID"
PROXY_PID=""
stop_pid "$SERVER_STEP_PID"
SERVER_STEP_PID=""
# Keep the receiver alive while vLLM shutdown flushes pending OTLP spans.
sleep 5
stop_pid "$OTEL_PID"
OTEL_PID=""
echo "observability_sources_frozen=true"

echo "phase=queue_state_join"
"$PYTHON_BIN" "$QUEUE_JOIN_SCRIPT" \
  --requests "${OUT_DIR}/requests.csv" \
  --proxy-events "${OUT_DIR}/proxy_events.csv" \
  --metrics "${OUT_DIR}/vllm_metrics_snapshots.csv" \
  --output "${OUT_DIR}/queue_state_at_arrival.csv"

echo "phase=kv_transfer_extract"
"$PYTHON_BIN" "$KV_TRANSFER_EXTRACTOR" \
  --requests "${OUT_DIR}/queue_state_at_arrival.csv" \
  --proxy-events "${OUT_DIR}/proxy_events.csv" \
  --output "${OUT_DIR}/kv_transfer_events.csv"

echo "phase=vllm_log_csv_export"
"$PYTHON_BIN" "$LOG_TO_CSV_SCRIPT" \
  --log "prefill=${OUT_DIR}/prefill_server.log" \
  --log "decode=${OUT_DIR}/decode_server.log" \
  --output "${OUT_DIR}/vllm_observability_log_events.csv"

cat "${OUT_DIR}/clickhouse_live_uploader.log" 2>/dev/null || true

echo "phase=clickhouse_final_upload"
CLICKHOUSE_FINAL_RC=0
"$PYTHON_BIN" -u "$CLICKHOUSE_UPLOADER" \
  --mode final \
  --output-dir "$OUT_DIR" \
  --workloads "$WORKLOAD_FILE" \
  --job-id "${SLURM_JOB_ID:-manual}" \
  --job-name "$JOB_NAME" \
  --started-unix-ns "$JOB_STARTED_UNIX_NS" \
  --state-file "$CLICKHOUSE_STATE" \
  > "${OUT_DIR}/clickhouse_final_upload.log" 2>&1 \
  || CLICKHOUSE_FINAL_RC=$?
cat "${OUT_DIR}/clickhouse_final_upload.log" 2>/dev/null || true
echo "clickhouse_final_upload_rc=${CLICKHOUSE_FINAL_RC}"

echo "phase=artifact_check"
for artifact in requests.csv token_timestamps.csv stream_chunk_events.csv \
  client_events.csv \
  window_summary.csv workload_events.csv vllm_metrics_snapshots.csv \
  drain_samples.csv drain_summary.csv \
  proxy_events.csv kv_transfer_events.csv \
  vllm_observability_log_events.csv otel_batches.csv \
  otel_spans.csv job_metadata.csv queue_state_at_arrival.csv \
  clickhouse_upload_state.json clickhouse_live_uploader.log \
  clickhouse_final_upload.log; do
  if [ -s "${OUT_DIR}/${artifact}" ]; then
    echo "artifact=${artifact} bytes=$(wc -c < "${OUT_DIR}/${artifact}")"
  else
    echo "artifact_missing=${artifact}"
  fi
done

if [ "$BENCH_RC" -ne 0 ]; then
  exit "$BENCH_RC"
fi
if [ "$CLICKHOUSE_FINAL_RC" -ne 0 ]; then
  exit "$CLICKHOUSE_FINAL_RC"
fi
echo "job_exit_rc=0"
exit 0
