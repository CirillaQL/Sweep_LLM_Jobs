#!/usr/bin/env bash
#SBATCH --job-name=vllm-saturation-gate-energy
#SBATCH --partition=long
#SBATCH --time=00:28:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate/slurm-%j.err

set -uo pipefail

SOURCE_JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
JOB_DIR="${JOB_DIR_OVERRIDE:-${SOURCE_JOB_DIR}}"
OUT_DIR="${JOB_DIR}/${OUT_DIR_NAME_OVERRIDE:-saturation_gate_results}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
NODE_SCRIPT="${SOURCE_JOB_DIR}/start_pd_node.sh"
PROXY_SCRIPT="${SOURCE_JOB_DIR}/pd_proxy.py"
CLOCK_PROBE="${SOURCE_JOB_DIR}/clock_probe.py"
SCHEDULER_SCRIPT="${SOURCE_JOB_DIR}/scheduler.py"
MODEL_BUNDLE="${SOURCE_JOB_DIR}/model_bundle.json"
SATURATION_BUNDLE="${SOURCE_JOB_DIR}/saturation_bundle.json"
WORKLOAD_FILE="${WORKLOAD_FILE_OVERRIDE:-${SOURCE_JOB_DIR}/workloads.csv}"
ENERGY_SUMMARY_SCRIPT="${SOURCE_JOB_DIR}/energy_summary.py"
POWER_MONITOR_SCRIPT="${SOURCE_JOB_DIR}/monitor_gpu_power.py"
LIVE_SUMMARY_SCRIPT="${SOURCE_JOB_DIR}/summarize_live_results.py"
COMPARE_SCRIPT="${SOURCE_JOB_DIR}/compare_gate_results.py"
REFERENCE_COMPARISON="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-122040-luqia-vllm-pd-baseline-energy-r1/baseline_results/comparison.json"
PLACEMENT_FILE="${OUT_DIR}/active_placement.json"
CLOCK_CONTROL_DIR="${OUT_DIR}/clock_control"
MODEL="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
JOB_TIME_LIMIT_LABEL="${JOB_TIME_LIMIT_LABEL_OVERRIDE:-00:28:00}"

SLO_TTFT_MS=500
SLO_TPOT_MS=200
BENCH_WINDOW_SECONDS="${BENCH_WINDOW_SECONDS_OVERRIDE:-10}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS_OVERRIDE:-180}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS_OVERRIDE:-1}"
WINDOW_WARMUP_SECONDS="${WINDOW_WARMUP_SECONDS_OVERRIDE:-0}"
MIN_PROMPTS="${MIN_PROMPTS_OVERRIDE:-12}"
MAX_PROMPTS="${MAX_PROMPTS_OVERRIDE:-120}"
MAX_CONCURRENCY="${MAX_CONCURRENCY_OVERRIDE:-16}"
BENCH_SAVE_DETAILED="${BENCH_SAVE_DETAILED_OVERRIDE:-false}"
EXPECTED_WORKLOAD_COUNT="${EXPECTED_WORKLOAD_COUNT_OVERRIDE:-3}"
REQUIRE_ALL_WORKLOADS_EXECUTED="${REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE:-false}"
ALLOW_ZERO_ADMIT="${ALLOW_ZERO_ADMIT_OVERRIDE:-false}"
SCHEDULER_PLACEMENT="${SCHEDULER_PLACEMENT_OVERRIDE:-l40s-prefill-l4-decode}"
OVERLOAD_ACTION="${OVERLOAD_ACTION_OVERRIDE:-min-slo-violation}"
RUN_REFERENCE_COMPARISON="${RUN_REFERENCE_COMPARISON_OVERRIDE:-true}"
ONLINE_SCHEDULING="${ONLINE_SCHEDULING_OVERRIDE:-false}"
MAX_L4_FREQ="${MAX_L4_FREQ_OVERRIDE:-}"
MAX_L40S_FREQ="${MAX_L40S_FREQ_OVERRIDE:-}"
CLOCK_MISMATCH_ACTION="${CLOCK_MISMATCH_ACTION_OVERRIDE:-abort}"
SCHEDULER_ENABLED="${SCHEDULER_ENABLED_OVERRIDE:-true}"
FIXED_PLACEMENT_FILE="${FIXED_PLACEMENT_FILE_OVERRIDE:-}"
GPU_CLOCK_CONTROL_MODE="${GPU_CLOCK_CONTROL_MODE_OVERRIDE:-manual}"
CLOCK_ACK_MODE="${CLOCK_ACK_MODE_OVERRIDE:-monitor}"
CLOCK_ACK_TRANSPORT="${CLOCK_ACK_TRANSPORT_OVERRIDE:-http_with_file_fallback}"
POLICY_LABEL="${POLICY_LABEL_OVERRIDE:-latency_plus_saturation}"
ACTIVE_SCHEDULER_POLICY="${ACTIVE_SCHEDULER_POLICY_OVERRIDE:-latency_plus_saturation}"
KV_EFFECTIVE_BANDWIDTH_GBPS="${KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE:-}"
KV_NUM_LAYERS="${KV_NUM_LAYERS_OVERRIDE:-32}"
KV_NUM_HEADS="${KV_NUM_HEADS_OVERRIDE:-8}"
KV_HEAD_DIM="${KV_HEAD_DIM_OVERRIDE:-128}"
KV_BYTES_PER_ELEMENT="${KV_BYTES_PER_ELEMENT_OVERRIDE:-2}"
DISPATCH_MS="${DISPATCH_MS_OVERRIDE:-0}"
SATURATION_ACTION="${AUTOSCALE_ACTION_OVERRIDE:-${SATURATION_ACTION_OVERRIDE:-frequency}}"
SCALE_IN_ENABLED="${SCALE_IN_ENABLED_OVERRIDE:-true}"
MAX_PREFILL_INSTANCES="${MAX_PREFILL_INSTANCES_OVERRIDE:-2}"
MAX_DECODE_INSTANCES="${MAX_DECODE_INSTANCES_OVERRIDE:-2}"
INITIAL_PREFILL_INSTANCES="${INITIAL_PREFILL_INSTANCES_OVERRIDE:-1}"
INITIAL_DECODE_INSTANCES="${INITIAL_DECODE_INSTANCES_OVERRIDE:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE_OVERRIDE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE_OVERRIDE:-1}"
POWER_MONITOR_GANYMEDE_GPUS="${POWER_MONITOR_GANYMEDE_GPUS_OVERRIDE:-${POWER_MONITOR_GPUS_PER_NODE_OVERRIDE:-${SLURM_GPUS_ON_NODE:-1}}}"
POWER_MONITOR_NEPTUNE_GPUS="${POWER_MONITOR_NEPTUNE_GPUS_OVERRIDE:-${POWER_MONITOR_GPUS_PER_NODE_OVERRIDE:-${SLURM_GPUS_ON_NODE:-1}}}"
POWER_MONITOR_TOTAL_GPUS=$((POWER_MONITOR_GANYMEDE_GPUS + POWER_MONITOR_NEPTUNE_GPUS))
VLLM_STEP_MEM="${VLLM_STEP_MEM_OVERRIDE:-16G}"
CURRENT_PREFILL_INSTANCES="$INITIAL_PREFILL_INSTANCES"
CURRENT_DECODE_INSTANCES="$INITIAL_DECODE_INSTANCES"

case "$CLOCK_ACK_TRANSPORT" in
  http|file|http_with_file_fallback) ;;
  *)
    echo "unsupported_clock_ack_transport=${CLOCK_ACK_TRANSPORT}"
    exit 26
    ;;
esac
case "$SATURATION_ACTION" in
  frequency|scale_out) ;;
  *)
    echo "unsupported_saturation_action=${SATURATION_ACTION}"
    exit 27
    ;;
esac
case "$SCALE_IN_ENABLED" in
  true|false) ;;
  *)
    echo "invalid_scale_in_enabled=${SCALE_IN_ENABLED}"
    exit 27
    ;;
esac
case "$REQUIRE_ALL_WORKLOADS_EXECUTED" in
  true|false) ;;
  *)
    echo "invalid_require_all_workloads_executed=${REQUIRE_ALL_WORKLOADS_EXECUTED}"
    exit 27
    ;;
esac
for count in "$MAX_PREFILL_INSTANCES" "$MAX_DECODE_INSTANCES" \
  "$INITIAL_PREFILL_INSTANCES" "$INITIAL_DECODE_INSTANCES" \
  "$PREFILL_TP_SIZE" "$DECODE_TP_SIZE" \
  "$POWER_MONITOR_GANYMEDE_GPUS" "$POWER_MONITOR_NEPTUNE_GPUS"; do
  case "$count" in
    ''|*[!0-9]*|0)
      echo "invalid_max_instance_count=${count:-unset}"
      exit 27
      ;;
  esac
done
if [ "$INITIAL_PREFILL_INSTANCES" -gt "$MAX_PREFILL_INSTANCES" ] || \
  [ "$INITIAL_DECODE_INSTANCES" -gt "$MAX_DECODE_INSTANCES" ]; then
  echo "initial_instance_count_exceeds_max initial_prefill=${INITIAL_PREFILL_INSTANCES} max_prefill=${MAX_PREFILL_INSTANCES} initial_decode=${INITIAL_DECODE_INSTANCES} max_decode=${MAX_DECODE_INSTANCES}"
  exit 27
fi
if [ "$POWER_MONITOR_NEPTUNE_GPUS" -lt "$((INITIAL_PREFILL_INSTANCES * PREFILL_TP_SIZE))" ] || \
  [ "$POWER_MONITOR_GANYMEDE_GPUS" -lt "$((INITIAL_DECODE_INSTANCES * DECODE_TP_SIZE))" ]; then
  echo "power_monitor_gpu_count_too_small_for_initial_instances ganymede_gpus=${POWER_MONITOR_GANYMEDE_GPUS} neptune_gpus=${POWER_MONITOR_NEPTUNE_GPUS} initial_prefill_instances=${INITIAL_PREFILL_INSTANCES} prefill_tp=${PREFILL_TP_SIZE} initial_decode_instances=${INITIAL_DECODE_INSTANCES} decode_tp=${DECODE_TP_SIZE}"
  exit 27
fi
if { [ "$PREFILL_TP_SIZE" -gt 1 ] || [ "$DECODE_TP_SIZE" -gt 1 ]; } && \
  { [ "$MAX_PREFILL_INSTANCES" -ne 1 ] || [ "$MAX_DECODE_INSTANCES" -ne 1 ] || \
    [ "$INITIAL_PREFILL_INSTANCES" -ne 1 ] || [ "$INITIAL_DECODE_INSTANCES" -ne 1 ]; }; then
  echo "multi_instance_tensor_parallel_unsupported=true prefill_tp=${PREFILL_TP_SIZE} decode_tp=${DECODE_TP_SIZE}"
  exit 27
fi
if { [ "$PREFILL_TP_SIZE" -gt 1 ] || [ "$DECODE_TP_SIZE" -gt 1 ]; } && \
  [ "$SCHEDULER_ENABLED" = true ]; then
  echo "scheduled_tensor_parallel_unsupported=true action=use_fixed_placement"
  exit 27
fi
if { [ "$PREFILL_TP_SIZE" -gt 1 ] || [ "$DECODE_TP_SIZE" -gt 1 ]; } && \
  [ "$GPU_CLOCK_CONTROL_MODE" = manual ]; then
  echo "manual_clock_control_tensor_parallel_unsupported=true action=use_auto_dvfs"
  exit 27
fi
case "$WINDOW_WARMUP_SECONDS" in
  ''|*[!0-9]*)
    echo "invalid_window_warmup_seconds=${WINDOW_WARMUP_SECONDS:-unset}"
    exit 27
    ;;
esac
case "$BENCH_NUM_WARMUPS" in
  ''|*[!0-9]*)
    echo "invalid_bench_num_warmups=${BENCH_NUM_WARMUPS:-unset}"
    exit 27
    ;;
esac
if [ "$SATURATION_ACTION" = scale_out ] && \
  { [ "$POWER_MONITOR_NEPTUNE_GPUS" -lt "$MAX_PREFILL_INSTANCES" ] || \
    [ "$POWER_MONITOR_GANYMEDE_GPUS" -lt "$MAX_DECODE_INSTANCES" ]; }; then
  echo "power_monitor_gpu_count_too_small ganymede_gpus=${POWER_MONITOR_GANYMEDE_GPUS} neptune_gpus=${POWER_MONITOR_NEPTUNE_GPUS} max_prefill_instances=${MAX_PREFILL_INSTANCES} max_decode_instances=${MAX_DECODE_INSTANCES}"
  exit 27
fi

if [ -f "${JOB_DIR}/CANCEL_BEFORE_RUN" ]; then
  echo "cancel_before_run=true job_dir=${JOB_DIR}"
  echo "job_exit_rc=0"
  exit 0
fi

PROXY_IP=10.1.0.3
PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
PREFILL_HTTP_PORT=$((32000 + PORT_OFFSET))
DECODE_HTTP_PORT=$((33000 + PORT_OFFSET))
KV_PORT=$((34000 + PORT_OFFSET))
PREFILL_IP=10.1.0.6
DECODE_IP=10.1.0.3

export HOME="/data/users/chjing"
export USER=chjing
export LOGNAME=chjing
export HF_HOME="${HOME}/.cache/huggingface"
export XDG_CACHE_HOME="${HOME}/.cache"
export PATH="${ENV_DIR}/bin:${PATH}"

export JOB_DIR OUT_DIR ENV_DIR PYTHON_BIN VLLM_BIN MODEL PLACEMENT_FILE CLOCK_PROBE
export CLOCK_CONTROL_DIR PROXY_IP PROXY_HTTP_PORT PROXY_REGISTER_PORT
export PREFILL_HTTP_PORT DECODE_HTTP_PORT PREFILL_IP DECODE_IP KV_PORT
export GPU_CLOCK_CONTROL_MODE_OVERRIDE
export CLOCK_ACK_MODE_OVERRIDE="$CLOCK_ACK_MODE"
export CLOCK_ACK_TRANSPORT_OVERRIDE="$CLOCK_ACK_TRANSPORT"

# Batch-level defaults must not leak into nested steps. In particular, an
# inherited --gpus-per-node allocation made each one-GPU vLLM step reserve all
# GPUs on its node, leaving no GRES for scale-out.
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK
unset SLURM_GPUS
unset SLURM_GPUS_PER_NODE
unset SLURM_GPUS_PER_SOCKET
unset SLURM_GPUS_PER_TASK
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

mkdir -p "$OUT_DIR" "$CLOCK_CONTROL_DIR"
for lifecycle_path in \
  "${OUT_DIR}"/*_telemetry.csv \
  "${OUT_DIR}"/allocation_*_power.csv \
  "${OUT_DIR}"/*_server.log \
  "${OUT_DIR}"/scale_*_step.log; do
  if [ -e "$lifecycle_path" ]; then
    : > "$lifecycle_path"
  fi
done
: > "${CLOCK_CONTROL_DIR}/neptune.ack"
: > "${CLOCK_CONTROL_DIR}/ganymede.ack"
echo "unix_ts,event,seq,workload_id,policy" > "${OUT_DIR}/events.csv"
ulimit -n 65536

PROXY_PID=""
SERVER_STEP_PIDS=()
POWER_MONITOR_PIDS=()
POWER_MONITOR_LOGS=()
SCALE_STEP_PIDS=()
FINAL_RC=0
RESET_REQUIRED=false
RESET_DONE=false
BENCH_FAILURES=0
BENCHMARKS_COMPLETED=0
GATE_ADMITTED=0
GATE_REJECTED=0
OVERLOAD_FALLBACKS=0
SCALE_OUT_EVENTS=0
SCALE_IN_EVENTS=0

section() {
  echo
  echo "===== $1 ====="
}

stop_scale_instances() {
  local pid
  for pid in "${SCALE_STEP_PIDS[@]}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "stopping_scale_step_pid=${pid}"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  SCALE_STEP_PIDS=()
}

power_monitors_alive() {
  local pid
  if [ "${#POWER_MONITOR_PIDS[@]}" -ne 2 ]; then
    return 1
  fi
  for pid in "${POWER_MONITOR_PIDS[@]}"; do
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

server_steps_alive() {
  local pid
  if [ "${#SERVER_STEP_PIDS[@]}" -ne 2 ]; then
    return 1
  fi
  for pid in "${SERVER_STEP_PIDS[@]}"; do
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

scale_instances_alive() {
  local pid
  for pid in "${SCALE_STEP_PIDS[@]}"; do
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

dump_power_monitor_logs() {
  local log_path
  for log_path in "${POWER_MONITOR_LOGS[@]}"; do
    echo "===== power_monitor_log path=${log_path} ====="
    cat "$log_path" 2>/dev/null || true
  done
}

stop_components() {
  stop_scale_instances
  local server_pid
  for server_pid in "${SERVER_STEP_PIDS[@]}"; do
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
      echo "stopping_server_step_pid=${server_pid}"
      kill "$server_pid" 2>/dev/null || true
    fi
    if [ -n "$server_pid" ]; then
      wait "$server_pid" 2>/dev/null || true
    fi
  done
  SERVER_STEP_PIDS=()
  local power_pid
  for power_pid in "${POWER_MONITOR_PIDS[@]}"; do
    if [ -n "$power_pid" ] && kill -0 "$power_pid" 2>/dev/null; then
      echo "stopping_power_monitor_pid=${power_pid}"
      kill "$power_pid" 2>/dev/null || true
    fi
    if [ -n "$power_pid" ]; then
      wait "$power_pid" 2>/dev/null || true
    fi
  done
  POWER_MONITOR_PIDS=()
  POWER_MONITOR_LOGS=()
  if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "stopping_proxy_pid=${PROXY_PID}"
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  PROXY_PID=""
}

reset_gpu_clocks() {
  if [ "$RESET_REQUIRED" != true ] || [ "$RESET_DONE" = true ]; then
    return 0
  fi
  section "parent_gpu_clock_reset"
  echo "reset_strategy=fresh_srun_after_server_step"
  : > "${OUT_DIR}/reset_step.log"
  local reset_rc=0
  srun --kill-on-bad-exit=1 --exact --nodes=1 \
    --nodelist=neptune --ntasks="$CURRENT_PREFILL_INSTANCES" \
    --ntasks-per-node="$CURRENT_PREFILL_INSTANCES" --gpus-per-task=1 \
    --cpus-per-task=8 --mem=4G "$NODE_SCRIPT" reset \
    >> "${OUT_DIR}/reset_step.log" 2>&1 || reset_rc=$?
  srun --kill-on-bad-exit=1 --exact --nodes=1 \
    --nodelist=ganymede --ntasks="$CURRENT_DECODE_INSTANCES" \
    --ntasks-per-node="$CURRENT_DECODE_INSTANCES" --gpus-per-task=1 \
    --cpus-per-task=8 --mem=4G "$NODE_SCRIPT" reset \
    >> "${OUT_DIR}/reset_step.log" 2>&1 || reset_rc=$?
  cat "${OUT_DIR}/reset_step.log" 2>/dev/null || true
  echo "parent_reset_rc=${reset_rc}"
  if [ "$reset_rc" -ne 0 ]; then
    echo "parent_reset_verified=false"
    return "$reset_rc"
  fi
  RESET_DONE=true
  echo "parent_reset_verified=true"
  return 0
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_components
  if ! reset_gpu_clocks; then
    echo "cleanup_reset_failed=true"
    if [ "$rc" -eq 0 ]; then
      rc=30
    fi
  fi
  echo "job_cleanup_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_url() {
  local name="$1"
  local url="$2"
  local max_seconds="$3"
  local elapsed=0
  while ! curl -fsS "$url" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo "waiting name=${name} elapsed_s=${elapsed} url=${url}"
    if [ "${#SERVER_STEP_PIDS[@]}" -gt 0 ] && ! server_steps_alive; then
      echo "server_step_died_while_waiting=true"
      return 1
    fi
    if [ "$elapsed" -ge "$max_seconds" ]; then
      echo "wait_timeout name=${name} url=${url}"
      return 1
    fi
  done
  echo "ready name=${name} elapsed_s=${elapsed} url=${url}"
}

placement_values() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") not in {"OK", "OVERLOAD_FALLBACK"}:
    raise SystemExit(f"unusable scheduler result: {result.get('status')}")
recommended = result["recommended"]
prefill = recommended["prefill"]
decode = recommended["decode"]
print(
    prefill["node_group"], int(prefill["rec_freq_mhz"]),
    decode["node_group"], int(decode["rec_freq_mhz"]),
    recommended["predicted_cluster_power_w"],
)
PY
}

decision_status() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
recommended = result.get("recommended", {})
print(
    result.get("status", "UNKNOWN"),
    recommended.get("predicted_overload_violation_probability", "NA"),
)
PY
}

schedule_one() {
  local seq="$1"
  local workload_id="$2"
  local il="$3"
  local ol="$4"
  local rate="$5"
  local track_counts="${6:-true}"
  local latency_decision="${OUT_DIR}/decision_latency_${seq}_${workload_id}.json"
  local gate_decision="${OUT_DIR}/decision_gate_${seq}_${workload_id}.json"
  local active_decision="${OUT_DIR}/decision_active_${seq}_${workload_id}.json"
  local limit_args=()
  local kv_args=(
    --kv-num-layers "$KV_NUM_LAYERS"
    --kv-num-heads "$KV_NUM_HEADS"
    --kv-head-dim "$KV_HEAD_DIM"
    --kv-bytes-per-element "$KV_BYTES_PER_ELEMENT"
    --dispatch-ms "$DISPATCH_MS"
    --prefill-instances "$CURRENT_PREFILL_INSTANCES"
    --decode-instances "$CURRENT_DECODE_INSTANCES"
  )
  if [ -n "$KV_EFFECTIVE_BANDWIDTH_GBPS" ]; then
    kv_args+=(--kv-effective-bandwidth-gbps "$KV_EFFECTIVE_BANDWIDTH_GBPS")
  fi
  if [ -n "$MAX_L4_FREQ" ]; then
    limit_args+=(--max-l4-freq "$MAX_L4_FREQ")
  fi
  if [ -n "$MAX_L40S_FREQ" ]; then
    limit_args+=(--max-l40s-freq "$MAX_L40S_FREQ")
  fi

  echo "scheduler_workload seq=${seq} id=${workload_id} il=${il} ol=${ol} rate=${rate} online=${ONLINE_SCHEDULING} active_policy=${ACTIVE_SCHEDULER_POLICY}"
  "$PYTHON_BIN" "$SCHEDULER_SCRIPT" \
    --bundle "$MODEL_BUNDLE" --saturation-bundle "$SATURATION_BUNDLE" \
    --policy latency_only --placement "$SCHEDULER_PLACEMENT" \
    --overload-action "$OVERLOAD_ACTION" "${limit_args[@]}" "${kv_args[@]}" \
    --il "$il" --ol "$ol" --rate "$rate" \
    --slo-ttft "$SLO_TTFT_MS" --slo-tpot "$SLO_TPOT_MS" --tp 1 \
    --output "$latency_decision" > "${OUT_DIR}/scheduler_latency_${seq}_${workload_id}.json"
  local latency_rc=$?
  if [ "$latency_rc" -ne 0 ] && [ "$latency_rc" -ne 2 ]; then
    echo "latency_only_scheduler_failed seq=${seq} rc=${latency_rc}"
    return 20
  fi

  local gate_rc=0
  if [ "$ACTIVE_SCHEDULER_POLICY" = latency_plus_saturation ]; then
    "$PYTHON_BIN" "$SCHEDULER_SCRIPT" \
      --bundle "$MODEL_BUNDLE" --saturation-bundle "$SATURATION_BUNDLE" \
      --policy latency_plus_saturation --placement "$SCHEDULER_PLACEMENT" \
      --overload-action "$OVERLOAD_ACTION" "${limit_args[@]}" "${kv_args[@]}" \
      --il "$il" --ol "$ol" --rate "$rate" \
      --slo-ttft "$SLO_TTFT_MS" --slo-tpot "$SLO_TPOT_MS" --tp 1 \
      --output "$gate_decision" > "${OUT_DIR}/scheduler_gate_${seq}_${workload_id}.json"
    gate_rc=$?
    if [ "$gate_rc" -ne 0 ] && [ "$gate_rc" -ne 2 ]; then
      echo "saturation_gate_scheduler_failed seq=${seq} rc=${gate_rc}"
      return 21
    fi
  else
    echo "saturation_gate_skipped seq=${seq} id=${workload_id} active_policy=${ACTIVE_SCHEDULER_POLICY}"
  fi

  local active_source active_rc
  case "$ACTIVE_SCHEDULER_POLICY" in
    latency_only)
      active_source="$latency_decision"
      active_rc="$latency_rc"
      ;;
    latency_plus_saturation)
      active_source="$gate_decision"
      active_rc="$gate_rc"
      ;;
    *)
      echo "unsupported_active_scheduler_policy=${ACTIVE_SCHEDULER_POLICY}"
      return 21
      ;;
  esac
  cp "$active_source" "$active_decision"
  if [ "$active_rc" -eq 2 ]; then
    if [ "$track_counts" = true ]; then
      GATE_REJECTED=$((GATE_REJECTED + 1))
    fi
    echo "active_policy_no_admit seq=${seq} id=${workload_id} policy=${ACTIVE_SCHEDULER_POLICY}"
    return 2
  fi

  if [ "$track_counts" = true ]; then
    GATE_ADMITTED=$((GATE_ADMITTED + 1))
  fi
  local active_status overload_probability values
  read -r active_status overload_probability <<< "$(decision_status "$active_decision")"
  if [ "$active_status" = OVERLOAD_FALLBACK ]; then
    if [ "$track_counts" = true ]; then
      OVERLOAD_FALLBACKS=$((OVERLOAD_FALLBACKS + 1))
    fi
    echo "overload_fallback_selected seq=${seq} id=${workload_id} policy=${ACTIVE_SCHEDULER_POLICY} predicted_violation_probability=${overload_probability}"
  elif [ "$active_status" != OK ]; then
    echo "unexpected_active_status seq=${seq} id=${workload_id} status=${active_status}"
    return 22
  fi

  values=$(placement_values "$active_decision") || return 22
  local prefill_node prefill_freq decode_node decode_freq predicted_power
  read -r prefill_node prefill_freq decode_node decode_freq predicted_power <<< "$values"
  echo "scheduler_selected seq=${seq} policy=${ACTIVE_SCHEDULER_POLICY} status=${active_status} prefill=${prefill_node}:${prefill_freq} decode=${decode_node}:${decode_freq} predicted_cluster_power_w=${predicted_power}"
  if [ "$SCHEDULER_PLACEMENT" = l40s-prefill-l4-decode ] && \
    { [ "$prefill_node" != neptune ] || [ "$decode_node" != ganymede ]; }; then
    echo "unexpected_role_selection=true seq=${seq}"
    return 23
  fi
  if [ -z "$FIRST_PLACEMENT" ]; then
    FIRST_PLACEMENT="$active_decision"
  fi
  return 0
}

decision_scale_signal() {
  "$PYTHON_BIN" - "$1" "$ACTIVE_SCHEDULER_POLICY" "$SLO_TTFT_MS" "$SLO_TPOT_MS" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
policy = sys.argv[2]
slo_ttft_ms = float(sys.argv[3])
slo_tpot_ms = float(sys.argv[4])
recommended = result.get("recommended", {})
prefill = recommended.get("prefill", {})
decode = recommended.get("decode", {})
if policy == "latency_only":
    prefill_unsafe = (
        prefill.get("is_safe") is not True
        or float(prefill.get("p99_ttft_ms", float("inf"))) > slo_ttft_ms
    )
    decode_unsafe = (
        decode.get("is_safe") is not True
        or float(decode.get("p99_tpot_ms", float("inf"))) > slo_tpot_ms
    )
    prefill_signal = prefill.get("p_violate", "NA")
    decode_signal = decode.get("p_violate", "NA")
elif policy == "latency_plus_saturation":
    prefill_unsafe = prefill.get("saturation_safe") is False
    decode_unsafe = decode.get("saturation_safe") is False
    prefill_signal = prefill.get("p_saturated", "NA")
    decode_signal = decode.get("p_saturated", "NA")
else:
    raise SystemExit(f"unsupported scale signal policy: {policy}")
print(
    int(prefill_unsafe),
    int(decode_unsafe),
    prefill_signal,
    decode_signal,
)
PY
}

launch_scale_instance() {
  local role="$1"
  local ordinal="$2"
  local node http_port kv_port clock_key instance_id
  if [ "$role" = prefill ]; then
    node=neptune
    http_port=$((PREFILL_HTTP_PORT + ordinal - 1))
    kv_port=$((KV_PORT + ordinal - 1))
  else
    node=ganymede
    http_port=$((DECODE_HTTP_PORT + ordinal - 1))
    kv_port=$((KV_PORT + ordinal - 1))
  fi
  instance_id="${role}-${ordinal}"
  clock_key="${node}-${instance_id}"
  : > "${CLOCK_CONTROL_DIR}/${clock_key}.ack"
  echo "scale_out_launch role=${role} ordinal=${ordinal} node=${node} http_port=${http_port} kv_port=${kv_port} clock_key=${clock_key}"
  printf '\n===== scale_launch time=%s role=%s ordinal=%s =====\n' \
    "$(date --iso-8601=seconds)" "$role" "$ordinal" \
    >> "${OUT_DIR}/scale_${instance_id}_step.log"
  PD_ROLE_OVERRIDE="$role" \
  PD_INSTANCE_ID_OVERRIDE="$instance_id" \
  PD_HTTP_PORT_OVERRIDE="$http_port" \
  PD_KV_PORT_OVERRIDE="$kv_port" \
  PD_CLOCK_KEY_OVERRIDE="$clock_key" \
    srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist="$node" \
      --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
      --mem="$VLLM_STEP_MEM" \
      "$NODE_SCRIPT" serve </dev/null \
      >> "${OUT_DIR}/scale_${instance_id}_step.log" 2>&1 &
  local scale_pid=$!
  SCALE_STEP_PIDS+=("$scale_pid")
  echo "scale_out_step_pid role=${role} ordinal=${ordinal} pid=${scale_pid}"
}

wait_for_registry_counts() {
  local expected_prefill="$1"
  local expected_decode="$2"
  local seq="$3"
  local registry_file="${OUT_DIR}/registry_scale_${seq}.json"
  local elapsed=0
  while [ "$elapsed" -lt 600 ]; do
    curl -fsS "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" \
      > "$registry_file" 2>/dev/null || true
    if "$PYTHON_BIN" - "$registry_file" "$expected_prefill" "$expected_decode" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ready = (
    len(data.get("prefill", [])) >= int(sys.argv[2])
    and len(data.get("decode", [])) >= int(sys.argv[3])
)
raise SystemExit(0 if ready else 1)
PY
    then
      echo "scale_out_registry_ready seq=${seq} prefill=${expected_prefill} decode=${expected_decode} elapsed_s=${elapsed}"
      return 0
    fi
    local pid
    for pid in "${SCALE_STEP_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "scale_out_step_died pid=${pid} seq=${seq}"
        return 1
      fi
    done
    sleep 5
    elapsed=$((elapsed + 5))
    echo "waiting_scale_out_registry seq=${seq} elapsed_s=${elapsed} expected_prefill=${expected_prefill} expected_decode=${expected_decode}"
  done
  echo "scale_out_registry_timeout seq=${seq}"
  return 1
}

decision_is_safe() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
recommended = result.get("recommended", {})
print(str(
    result.get("status") == "OK" and recommended.get("is_safe") is True
).lower())
PY
}

wait_for_registry_exact() {
  local expected_prefill="$1"
  local expected_decode="$2"
  local seq="$3"
  local registry_file="${OUT_DIR}/registry_scale_in_${seq}.json"
  local elapsed=0
  while [ "$elapsed" -lt 60 ]; do
    curl -fsS "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" \
      > "$registry_file" 2>/dev/null || true
    if "$PYTHON_BIN" - "$registry_file" "$expected_prefill" "$expected_decode" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ready = (
    len(data.get("prefill", [])) == int(sys.argv[2])
    and len(data.get("decode", [])) == int(sys.argv[3])
)
raise SystemExit(0 if ready else 1)
PY
    then
      echo "scale_in_registry_ready seq=${seq} prefill=${expected_prefill} decode=${expected_decode} elapsed_s=${elapsed}"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "waiting_scale_in_registry seq=${seq} elapsed_s=${elapsed} expected_prefill=${expected_prefill} expected_decode=${expected_decode}"
  done
  echo "scale_in_registry_timeout seq=${seq}"
  return 1
}

maybe_scale_in() {
  local seq="$1"
  local workload_id="$2"
  local il="$3"
  local ol="$4"
  local rate="$5"
  if [ "$SCHEDULER_ENABLED" != true ] || \
    [ "$SATURATION_ACTION" != scale_out ] || \
    [ "$SCALE_IN_ENABLED" != true ] || \
    { [ "$CURRENT_PREFILL_INSTANCES" -eq 1 ] && \
      [ "$CURRENT_DECODE_INSTANCES" -eq 1 ]; }; then
    return 0
  fi

  local old_prefill="$CURRENT_PREFILL_INSTANCES"
  local old_decode="$CURRENT_DECODE_INSTANCES"
  local probe_decision="${OUT_DIR}/decision_scale_in_probe_${seq}_${workload_id}.json"
  echo "scale_in_probe seq=${seq} id=${workload_id} current_prefill=${old_prefill} current_decode=${old_decode} target_prefill=1 target_decode=1"
  CURRENT_PREFILL_INSTANCES=1
  CURRENT_DECODE_INSTANCES=1
  schedule_one "$seq" "$workload_id" "$il" "$ol" "$rate" false
  local probe_rc=$?
  if [ -s "${OUT_DIR}/decision_active_${seq}_${workload_id}.json" ]; then
    cp "${OUT_DIR}/decision_active_${seq}_${workload_id}.json" "$probe_decision"
  fi
  CURRENT_PREFILL_INSTANCES="$old_prefill"
  CURRENT_DECODE_INSTANCES="$old_decode"
  if [ "$probe_rc" -ne 0 ]; then
    echo "scale_in_probe_failed seq=${seq} rc=${probe_rc}"
    return "$probe_rc"
  fi

  local probe_safe
  probe_safe=$(decision_is_safe "$probe_decision")
  if [ "$probe_safe" != true ]; then
    echo "scale_in_retain seq=${seq} prefill_instances=${old_prefill} decode_instances=${old_decode} reason=single_instance_not_safe"
    return 0
  fi

  echo "scale_in_trigger seq=${seq} id=${workload_id} prefill_instances=${old_prefill}->1 decode_instances=${old_decode}->1 action=stop_extra_vllm_instances"
  stop_scale_instances
  CURRENT_PREFILL_INSTANCES=1
  CURRENT_DECODE_INSTANCES=1
  if ! wait_for_registry_exact 1 1 "$seq"; then
    return 32
  fi
  SCALE_IN_EVENTS=$((SCALE_IN_EVENTS + 1))
  echo "scale_in_complete seq=${seq} prefill_instances=1 decode_instances=1"
  return 0
}

maybe_scale_out() {
  local seq="$1"
  local workload_id="$2"
  local il="$3"
  local ol="$4"
  local rate="$5"
  local decision="$6"
  if [ "$SCHEDULER_ENABLED" != true ] || \
    [ "$SATURATION_ACTION" != scale_out ]; then
    return 0
  fi

  local scale_prefill scale_decode prefill_signal_value decode_signal_value
  read -r scale_prefill scale_decode prefill_signal_value decode_signal_value \
    <<< "$(decision_scale_signal "$decision")"
  local old_prefill="$CURRENT_PREFILL_INSTANCES"
  local old_decode="$CURRENT_DECODE_INSTANCES"
  if [ "$scale_prefill" -eq 0 ] && [ "$scale_decode" -eq 0 ]; then
    return 0
  fi

  echo "scale_out_trigger seq=${seq} id=${workload_id} trigger_policy=${ACTIVE_SCHEDULER_POLICY} prefill_signal_value=${prefill_signal_value} decode_signal_value=${decode_signal_value} prefill_instances=${old_prefill} decode_instances=${old_decode}"
  cp "$decision" "${OUT_DIR}/decision_pre_scale_${seq}_${workload_id}.json"
  local changed scaled_schedule_rc
  while [ "$scale_prefill" -eq 1 ] || [ "$scale_decode" -eq 1 ]; do
    changed=false
    if [ "$scale_prefill" -eq 1 ] && \
      [ "$CURRENT_PREFILL_INSTANCES" -lt "$MAX_PREFILL_INSTANCES" ]; then
      CURRENT_PREFILL_INSTANCES=$((CURRENT_PREFILL_INSTANCES + 1))
      changed=true
    fi
    if [ "$scale_decode" -eq 1 ] && \
      [ "$CURRENT_DECODE_INSTANCES" -lt "$MAX_DECODE_INSTANCES" ]; then
      CURRENT_DECODE_INSTANCES=$((CURRENT_DECODE_INSTANCES + 1))
      changed=true
    fi
    if [ "$changed" != true ]; then
      break
    fi
    echo "scale_out_plan seq=${seq} prefill_instances=${CURRENT_PREFILL_INSTANCES} decode_instances=${CURRENT_DECODE_INSTANCES}"
    schedule_one "$seq" "$workload_id" "$il" "$ol" "$rate" false
    scaled_schedule_rc=$?
    if [ "$scaled_schedule_rc" -ne 0 ]; then
      CURRENT_PREFILL_INSTANCES="$old_prefill"
      CURRENT_DECODE_INSTANCES="$old_decode"
      echo "scale_out_reschedule_failed seq=${seq} rc=${scaled_schedule_rc}"
      return "$scaled_schedule_rc"
    fi
    decision="${OUT_DIR}/decision_active_${seq}_${workload_id}.json"
    read -r scale_prefill scale_decode prefill_signal_value decode_signal_value \
      <<< "$(decision_scale_signal "$decision")"
  done
  if [ "$old_prefill" -eq "$CURRENT_PREFILL_INSTANCES" ] && \
    [ "$old_decode" -eq "$CURRENT_DECODE_INSTANCES" ]; then
    echo "scale_out_capacity_exhausted trigger_policy=${ACTIVE_SCHEDULER_POLICY} prefill_instances=${CURRENT_PREFILL_INSTANCES} decode_instances=${CURRENT_DECODE_INSTANCES} prefill_signal_value=${prefill_signal_value} decode_signal_value=${decode_signal_value}"
    return 0
  fi
  if [ "$scale_prefill" -eq 1 ] || [ "$scale_decode" -eq 1 ]; then
    echo "scale_out_capacity_exhausted trigger_policy=${ACTIVE_SCHEDULER_POLICY} prefill_instances=${CURRENT_PREFILL_INSTANCES} decode_instances=${CURRENT_DECODE_INSTANCES} prefill_signal_value=${prefill_signal_value} decode_signal_value=${decode_signal_value}"
  fi
  cp "${OUT_DIR}/decision_active_${seq}_${workload_id}.json" "$PLACEMENT_FILE"

  local ordinal
  for ((ordinal = old_prefill + 1; ordinal <= CURRENT_PREFILL_INSTANCES; ordinal++)); do
    launch_scale_instance prefill "$ordinal"
  done
  for ((ordinal = old_decode + 1; ordinal <= CURRENT_DECODE_INSTANCES; ordinal++)); do
    launch_scale_instance decode "$ordinal"
  done
  if ! wait_for_registry_counts \
    "$CURRENT_PREFILL_INSTANCES" "$CURRENT_DECODE_INSTANCES" "$seq"; then
    return 31
  fi
  SCALE_OUT_EVENTS=$((SCALE_OUT_EVENTS + 1))
  echo "scale_out_complete seq=${seq} prefill_instances=${CURRENT_PREFILL_INSTANCES} decode_instances=${CURRENT_DECODE_INSTANCES}"
  return 0
}

write_clock_request() {
  local seq="$1"
  local neptune_freq="$2"
  local ganymede_freq="$3"
  local ordinal key
  for ((ordinal = 1; ordinal <= CURRENT_PREFILL_INSTANCES; ordinal++)); do
    key=neptune
    if [ "$ordinal" -gt 1 ]; then
      key="neptune-prefill-${ordinal}"
    fi
    printf '%s %s\n' "$seq" "$neptune_freq" > "${CLOCK_CONTROL_DIR}/${key}.request.tmp"
    mv "${CLOCK_CONTROL_DIR}/${key}.request.tmp" "${CLOCK_CONTROL_DIR}/${key}.request"
  done
  for ((ordinal = 1; ordinal <= CURRENT_DECODE_INSTANCES; ordinal++)); do
    key=ganymede
    if [ "$ordinal" -gt 1 ]; then
      key="ganymede-decode-${ordinal}"
    fi
    printf '%s %s\n' "$seq" "$ganymede_freq" > "${CLOCK_CONTROL_DIR}/${key}.request.tmp"
    mv "${CLOCK_CONTROL_DIR}/${key}.request.tmp" "${CLOCK_CONTROL_DIR}/${key}.request"
  done
  echo "clock_request seq=${seq} neptune_mhz=${neptune_freq} neptune_instances=${CURRENT_PREFILL_INSTANCES} ganymede_mhz=${ganymede_freq} ganymede_instances=${CURRENT_DECODE_INSTANCES}"
}

wait_clock_ack() {
  local seq="$1"
  local node_group="$2"
  local target="$3"
  local ack="${CLOCK_CONTROL_DIR}/${node_group}.ack"
  local elapsed=0
  while true; do
    local ack_seq ack_target ack_rc observed ack_source ack_line
    ack_source=""
    ack_line=""
    if [ "$CLOCK_ACK_TRANSPORT" = http ] || \
      [ "$CLOCK_ACK_TRANSPORT" = http_with_file_fallback ]; then
      ack_line=$(
        curl -fsS --connect-timeout 1 --max-time 2 \
          "http://${PROXY_IP}:${PROXY_HTTP_PORT}/control/clock-ack/${node_group}/${seq}" \
          2>/dev/null || true
      )
      if [ -n "$ack_line" ]; then
        read -r ack_seq ack_target ack_rc observed <<< "$ack_line"
        ack_source=http
      fi
    fi
    if [ -z "$ack_source" ] && \
      { [ "$CLOCK_ACK_TRANSPORT" = file ] || \
        [ "$CLOCK_ACK_TRANSPORT" = http_with_file_fallback ]; } && \
      [ -s "$ack" ]; then
      read -r ack_seq ack_target ack_rc observed < "$ack" || true
      ack_source=file
    fi
    if [ -n "$ack_source" ]; then
      if [ "${ack_seq:-}" = "$seq" ] && [ -n "${ack_target:-}" ] && \
        [ -n "${ack_rc:-}" ] && [ -n "${observed:-}" ]; then
        echo "clock_ack seq=${ack_seq} node_group=${node_group} target_mhz=${ack_target} rc=${ack_rc} observed_mhz=${observed:-NA} verification_mode=${CLOCK_ACK_MODE} transport=${ack_source}"
        if [ "$ack_target" != "$target" ]; then
          return 1
        fi
        if [ "$ack_rc" != 0 ]; then
          if [ "$ack_rc" = 33 ] && [ "$CLOCK_MISMATCH_ACTION" = continue ]; then
            echo "clock_target_not_sustained seq=${ack_seq} node_group=${node_group} target_mhz=${ack_target} observed_mhz=${observed:-NA} action=continue"
            return 0
          fi
          return 1
        fi
        return 0
      fi
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge 90 ]; then
      echo "clock_ack_timeout seq=${seq} node_group=${node_group} target_mhz=${target}"
      return 1
    fi
  done
}

section "scheduler_sweep_with_saturation_gate"
echo "workload_source=Dual_Sweep_LLM_Phase2_common_measured_grid"
echo "slo_ttft_ms=${SLO_TTFT_MS} slo_tpot_ms=${SLO_TPOT_MS} saturation_threshold=0.30"
echo "saturation_model=five_seed_mean_local_retrained_75"
echo "placement=${SCHEDULER_PLACEMENT}"
echo "overload_action=${OVERLOAD_ACTION}"
echo "online_scheduling=${ONLINE_SCHEDULING} max_l4_freq=${MAX_L4_FREQ:-none} max_l40s_freq=${MAX_L40S_FREQ:-none}"
echo "saturation_action=${SATURATION_ACTION} scale_in_enabled=${SCALE_IN_ENABLED} initial_prefill_instances=${INITIAL_PREFILL_INSTANCES} initial_decode_instances=${INITIAL_DECODE_INSTANCES} prefill_tp=${PREFILL_TP_SIZE} decode_tp=${DECODE_TP_SIZE} max_prefill_instances=${MAX_PREFILL_INSTANCES} max_decode_instances=${MAX_DECODE_INSTANCES} window_warmup_seconds=${WINDOW_WARMUP_SECONDS}"
echo "power_monitor_ganymede_gpus=${POWER_MONITOR_GANYMEDE_GPUS} power_monitor_neptune_gpus=${POWER_MONITOR_NEPTUNE_GPUS} power_monitor_total_gpus=${POWER_MONITOR_TOTAL_GPUS}"
echo "kv_effective_bandwidth_gbps=${KV_EFFECTIVE_BANDWIDTH_GBPS:-disabled} kv_num_layers=${KV_NUM_LAYERS} kv_num_heads=${KV_NUM_HEADS} kv_head_dim=${KV_HEAD_DIM} kv_bytes_per_element=${KV_BYTES_PER_ELEMENT} dispatch_ms=${DISPATCH_MS}"
echo "clock_mismatch_action=${CLOCK_MISMATCH_ACTION} clock_ack_mode=${CLOCK_ACK_MODE} clock_ack_transport=${CLOCK_ACK_TRANSPORT}"
echo "scheduler_enabled=${SCHEDULER_ENABLED} active_scheduler_policy=${ACTIVE_SCHEDULER_POLICY} gpu_clock_control_mode=${GPU_CLOCK_CONTROL_MODE} policy_label=${POLICY_LABEL}"
FIRST_PLACEMENT=""
WORKLOAD_COUNT=0
while IFS=, read -r workload_id il ol rate field5 field6 field7; do
  if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
    continue
  fi
  WORKLOAD_COUNT=$((WORKLOAD_COUNT + 1))
  if [ "$SCHEDULER_ENABLED" != true ]; then
    continue
  fi
  if [ "$ONLINE_SCHEDULING" = true ] && [ "$WORKLOAD_COUNT" -gt 1 ]; then
    continue
  fi
  schedule_one "$WORKLOAD_COUNT" "$workload_id" "$il" "$ol" "$rate"
  schedule_rc=$?
  if [ "$schedule_rc" -eq 2 ]; then
    continue
  fi
  if [ "$schedule_rc" -ne 0 ]; then
    exit "$schedule_rc"
  fi
done < "$WORKLOAD_FILE"

if [ "$SCHEDULER_ENABLED" != true ]; then
  if [ -z "$FIXED_PLACEMENT_FILE" ] || [ ! -s "$FIXED_PLACEMENT_FILE" ]; then
    echo "fixed_placement_missing=${FIXED_PLACEMENT_FILE:-unset}"
    exit 24
  fi
  FIRST_PLACEMENT="$FIXED_PLACEMENT_FILE"
fi

if [ "$WORKLOAD_COUNT" -ne "$EXPECTED_WORKLOAD_COUNT" ]; then
  echo "unexpected_scheduler_counts workloads=${WORKLOAD_COUNT} admitted=${GATE_ADMITTED} rejected=${GATE_REJECTED}"
  exit 24
fi

if [ "$SCHEDULER_ENABLED" = true ] && [ "$GATE_ADMITTED" -eq 0 ]; then
  if [ "$ALLOW_ZERO_ADMIT" != true ]; then
    echo "unexpected_scheduler_counts workloads=${WORKLOAD_COUNT} admitted=${GATE_ADMITTED} rejected=${GATE_REJECTED}"
    exit 24
  fi
  section "result"
  echo "workload_count=${WORKLOAD_COUNT}"
  echo "gate_admitted=${GATE_ADMITTED} gate_rejected=${GATE_REJECTED}"
  echo "overload_fallbacks=${OVERLOAD_FALLBACKS}"
  echo "decision_only=true"
  echo "gpu_allocated=false"
  echo "vllm_started=false"
  echo "clock_control_requested=false"
  echo "parent_reset_verified=not_required"
  echo "job_exit_rc=0"
  exit 0
fi

if [ -z "$FIRST_PLACEMENT" ]; then
  echo "missing_first_placement=true"
  exit 24
fi
cp "$FIRST_PLACEMENT" "$PLACEMENT_FILE"

section "allocation"
echo "time=$(date --iso-8601=seconds)"
echo "batch_host=$(hostname -s)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "model=${MODEL} workload_count=${WORKLOAD_COUNT} slurm_time_limit=${JOB_TIME_LIMIT_LABEL}"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>&1 || true
scontrol show job "${SLURM_JOB_ID:-}" 2>&1 || true

if [ "$(hostname -s)" != ganymede ]; then
  echo "expected_batch_host_ganymede=true actual=$(hostname -s)"
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true"
  exit 3
fi
for required in "$MODEL_BUNDLE" "$SATURATION_BUNDLE" "$ENERGY_SUMMARY_SCRIPT" \
  "$POWER_MONITOR_SCRIPT" "$LIVE_SUMMARY_SCRIPT"; do
  if [ ! -s "$required" ]; then
    echo "required_artifact_missing=${required}"
    exit 3
  fi
done
if [ "$RUN_REFERENCE_COMPARISON" = true ]; then
  for required in "$COMPARE_SCRIPT" "$REFERENCE_COMPARISON"; do
    if [ ! -s "$required" ]; then
      echo "required_artifact_missing=${required}"
      exit 3
    fi
  done
fi

section "100gbe_and_environment_preflight"
srun --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1 \
  --cpus-per-task=8 --mem=4G "$NODE_SCRIPT" preflight 2>&1 | tee "${OUT_DIR}/preflight.log"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "preflight_rc=${PREFLIGHT_RC}"
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  exit "$PREFLIGHT_RC"
fi

section "start_allocation_wide_gpu_power_monitor"
GANYMEDE_POWER_LOG="${OUT_DIR}/power_monitor_ganymede.log"
NEPTUNE_POWER_LOG="${OUT_DIR}/power_monitor_neptune.log"
srun --overlap --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=ganymede \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task="$POWER_MONITOR_GANYMEDE_GPUS" \
  --cpus-per-task=1 --mem=1G \
  "$PYTHON_BIN" -u "$POWER_MONITOR_SCRIPT" --output-dir "$OUT_DIR" \
  --expected-gpus "$POWER_MONITOR_GANYMEDE_GPUS" \
  > "$GANYMEDE_POWER_LOG" 2>&1 &
POWER_MONITOR_PIDS+=("$!")
srun --overlap --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=neptune \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task="$POWER_MONITOR_NEPTUNE_GPUS" \
  --cpus-per-task=1 --mem=1G \
  "$PYTHON_BIN" -u "$POWER_MONITOR_SCRIPT" --output-dir "$OUT_DIR" \
  --expected-gpus "$POWER_MONITOR_NEPTUNE_GPUS" \
  > "$NEPTUNE_POWER_LOG" 2>&1 &
POWER_MONITOR_PIDS+=("$!")
POWER_MONITOR_LOGS+=("$GANYMEDE_POWER_LOG" "$NEPTUNE_POWER_LOG")
echo "power_monitor_pids=${POWER_MONITOR_PIDS[*]} ganymede_gpus=${POWER_MONITOR_GANYMEDE_GPUS} neptune_gpus=${POWER_MONITOR_NEPTUNE_GPUS} total_gpus=${POWER_MONITOR_TOTAL_GPUS}"
POWER_MONITOR_READY=false
for attempt in $(seq 1 30); do
  if "$PYTHON_BIN" - "$OUT_DIR" "$POWER_MONITOR_GANYMEDE_GPUS" "$POWER_MONITOR_NEPTUNE_GPUS" <<'PY'
import csv
import glob
import os
import sys

out_dir, ganymede_text, neptune_text = sys.argv[1:]
expected_by_host = {
    "ganymede": int(ganymede_text),
    "neptune": int(neptune_text),
}
paths = glob.glob(os.path.join(out_dir, "allocation_*_power.csv"))
if len(paths) != 2:
    raise SystemExit(1)
seen_hosts = set()
for path in paths:
    counts = {}
    host = ""
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            host = row.get("host", "").split(".", 1)[0]
            uuid = row.get("gpu_uuid", "")
            if uuid:
                counts[uuid] = counts.get(uuid, 0) + 1
    expected = expected_by_host.get(host)
    if expected is None or len(counts) != expected or min(counts.values(), default=0) < 2:
        raise SystemExit(1)
    seen_hosts.add(host)
if seen_hosts != set(expected_by_host):
    raise SystemExit(1)
PY
  then
    POWER_MONITOR_READY=true
    break
  fi
  if ! power_monitors_alive; then
    break
  fi
  sleep 1
done
echo "power_monitor_ready=${POWER_MONITOR_READY}"
if [ "$POWER_MONITOR_READY" != true ]; then
  dump_power_monitor_logs
  exit 28
fi

section "start_proxy"
export PROXY_REGISTER_HOST=0.0.0.0
export PROXY_HTTP_HOST=0.0.0.0
"$PYTHON_BIN" -u "$PROXY_SCRIPT" > "${OUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
echo "proxy_pid=${PROXY_PID}"
if ! wait_for_url proxy "http://127.0.0.1:${PROXY_HTTP_PORT}/health" 60; then
  tail -n 200 "${OUT_DIR}/proxy.log" || true
  exit 4
fi

section "start_prefill_and_decode_once"
if [ "$GPU_CLOCK_CONTROL_MODE" = manual ]; then
  RESET_REQUIRED=true
else
  RESET_REQUIRED=false
fi
: > "${OUT_DIR}/server_step.log"
: > "${OUT_DIR}/prefill_primary_step.log"
: > "${OUT_DIR}/decode_primary_step.log"
PD_TP_SIZE_OVERRIDE="$PREFILL_TP_SIZE" \
  srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=neptune \
    --ntasks=1 --ntasks-per-node=1 --gpus-per-task="$PREFILL_TP_SIZE" \
    --cpus-per-task="$((8 * PREFILL_TP_SIZE))" --mem="$VLLM_STEP_MEM" \
    "$NODE_SCRIPT" serve >> "${OUT_DIR}/prefill_primary_step.log" 2>&1 &
SERVER_STEP_PIDS+=("$!")
PD_TP_SIZE_OVERRIDE="$DECODE_TP_SIZE" \
  srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=ganymede \
    --ntasks=1 --ntasks-per-node=1 --gpus-per-task="$DECODE_TP_SIZE" \
    --cpus-per-task="$((8 * DECODE_TP_SIZE))" --mem="$VLLM_STEP_MEM" \
    "$NODE_SCRIPT" serve >> "${OUT_DIR}/decode_primary_step.log" 2>&1 &
SERVER_STEP_PIDS+=("$!")
echo "server_step_pids=${SERVER_STEP_PIDS[*]} prefill_tp=${PREFILL_TP_SIZE} decode_tp=${DECODE_TP_SIZE}"
echo "server_step_pids=${SERVER_STEP_PIDS[*]} prefill_tp=${PREFILL_TP_SIZE} decode_tp=${DECODE_TP_SIZE}" > "${OUT_DIR}/server_step.log"

if ! wait_for_url prefill "http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/v1/models" 600; then
  FINAL_RC=5
fi
if [ "$FINAL_RC" -eq 0 ] && ! wait_for_url decode "http://${DECODE_IP}:${DECODE_HTTP_PORT}/v1/models" 600; then
  FINAL_RC=6
fi
if [ "$FINAL_RC" -ne 0 ]; then
  tail -n 240 "${OUT_DIR}/prefill_server.log" 2>/dev/null || true
  tail -n 240 "${OUT_DIR}/decode_server.log" 2>/dev/null || true
  exit "$FINAL_RC"
fi

section "start_initial_scale_instances"
for ((ordinal = 2; ordinal <= INITIAL_PREFILL_INSTANCES; ordinal++)); do
  launch_scale_instance prefill "$ordinal"
done
for ((ordinal = 2; ordinal <= INITIAL_DECODE_INSTANCES; ordinal++)); do
  launch_scale_instance decode "$ordinal"
done
if ! wait_for_registry_counts \
  "$INITIAL_PREFILL_INSTANCES" "$INITIAL_DECODE_INSTANCES" initial; then
  echo "initial_instances_not_ready=true prefill=${INITIAL_PREFILL_INSTANCES} decode=${INITIAL_DECODE_INSTANCES}"
  exit 7
fi

section "wait_for_pd_registry"
REGISTRY_FILE="${OUT_DIR}/registry.json"
REGISTRY_READY=false
for attempt in $(seq 1 60); do
  curl -fsS "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" 2>/dev/null || true
  if "$PYTHON_BIN" - "$REGISTRY_FILE" \
    "$INITIAL_PREFILL_INSTANCES" "$INITIAL_DECODE_INSTANCES" \
    "$PREFILL_TP_SIZE" "$DECODE_TP_SIZE" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ready = (
    len(data.get("prefill", [])) == int(sys.argv[2])
    and len(data.get("decode", [])) == int(sys.argv[3])
    and data.get("prefill_tp_sizes") == [int(sys.argv[4])] * int(sys.argv[2])
    and data.get("decode_tp_sizes") == [int(sys.argv[5])] * int(sys.argv[3])
)
raise SystemExit(0 if ready else 1)
PY
  then
    REGISTRY_READY=true
    break
  fi
  sleep 2
done
cat "$REGISTRY_FILE" 2>/dev/null || true
echo
echo "registry_ready=${REGISTRY_READY}"
if [ "$REGISTRY_READY" != true ]; then
  exit 7
fi

section "pd_smoke_request"
SMOKE_FILE="${OUT_DIR}/smoke_response.json"
curl -fsS --max-time 180 -X POST "http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  --data-binary "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of Sweden is\",\"max_tokens\":8,\"temperature\":0,\"stream\":false}" \
  > "$SMOKE_FILE"
SMOKE_RC=$?
echo "smoke_curl_rc=${SMOKE_RC}"
if [ "$SMOKE_RC" -ne 0 ]; then
  exit 8
fi

VLLM_BURSTINESS_SUPPORTED="${VLLM_BURSTINESS_SUPPORTED_OVERRIDE:-true}"
case "$VLLM_BURSTINESS_SUPPORTED" in
  true|false) ;;
  *)
    echo "invalid_vllm_burstiness_supported=${VLLM_BURSTINESS_SUPPORTED}"
    exit 8
    ;;
esac
echo "vllm_burstiness_supported=${VLLM_BURSTINESS_SUPPORTED}"

section "multi_workload_bench"
SEQ=0
while IFS=, read -r workload_id il ol rate field5 field6 field7; do
  if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
    continue
  fi
  SEQ=$((SEQ + 1))
  if [ "$ONLINE_SCHEDULING" = true ]; then
    window_seconds="$field5"
    burstiness="$field6"
    description="$field7"
  else
    window_seconds="$BENCH_WINDOW_SECONDS"
    burstiness=1
    description="$field5"
  fi

  if [ "$SCHEDULER_ENABLED" = true ] && [ "$ONLINE_SCHEDULING" = true ]; then
    maybe_scale_in "$SEQ" "$workload_id" "$il" "$ol" "$rate"
    scale_in_rc=$?
    if [ "$scale_in_rc" -ne 0 ]; then
      FINAL_RC="$scale_in_rc"
      echo "scale_in_failed seq=${SEQ} id=${workload_id} rc=${scale_in_rc}"
      break
    fi
  fi

  if [ "$SCHEDULER_ENABLED" = true ] && [ "$ONLINE_SCHEDULING" = true ] && [ "$SEQ" -gt 1 ]; then
    section "online_scheduler_window_${SEQ}_${workload_id}"
    schedule_one "$SEQ" "$workload_id" "$il" "$ol" "$rate"
    schedule_rc=$?
    if [ "$schedule_rc" -eq 2 ]; then
      echo "online_scheduler_action=NO_ADMIT seq=${SEQ} id=${workload_id}"
      continue
    fi
    if [ "$schedule_rc" -ne 0 ]; then
      FINAL_RC="$schedule_rc"
      break
    fi
  fi

  if [ "$SCHEDULER_ENABLED" = true ]; then
    placement="${OUT_DIR}/decision_active_${SEQ}_${workload_id}.json"
  else
    placement="$FIXED_PLACEMENT_FILE"
  fi
  maybe_scale_out "$SEQ" "$workload_id" "$il" "$ol" "$rate" "$placement"
  scale_rc=$?
  if [ "$scale_rc" -ne 0 ]; then
    FINAL_RC="$scale_rc"
    echo "scale_out_failed seq=${SEQ} id=${workload_id} rc=${scale_rc}"
    break
  fi
  if ! values=$(placement_values "$placement" 2>/dev/null); then
    section "workload_${SEQ}_${workload_id}_not_admitted"
    echo "saturation_gate_action=NO_ADMIT seq=${SEQ} id=${workload_id}"
    continue
  fi
  cp "$placement" "$PLACEMENT_FILE"
  read -r prefill_node neptune_freq decode_node ganymede_freq predicted_power <<< "$values"

  section "workload_${SEQ}_${workload_id}"
  echo "workload_prepare time=$(date --iso-8601=seconds) seq=${SEQ} id=${workload_id} il=${il} ol=${ol} request_rate=${rate} window_seconds=${window_seconds} burstiness=${burstiness} description=${description}"
  echo "fixed_placement prefill_node=${prefill_node} decode_node=${decode_node} gpu_clock_control_mode=${GPU_CLOCK_CONTROL_MODE}"
  if [ "$GPU_CLOCK_CONTROL_MODE" = manual ]; then
    echo "recommendation neptune_prefill_mhz=${neptune_freq} ganymede_decode_mhz=${ganymede_freq} predicted_cluster_power_w=${predicted_power}"
    write_clock_request "$SEQ" "$neptune_freq" "$ganymede_freq"
    for ((ordinal = 1; ordinal <= CURRENT_PREFILL_INSTANCES; ordinal++)); do
      clock_key=neptune
      if [ "$ordinal" -gt 1 ]; then
        clock_key="neptune-prefill-${ordinal}"
      fi
      if ! wait_clock_ack "$SEQ" "$clock_key" "$neptune_freq"; then
        FINAL_RC=24
        break
      fi
    done
    if [ "$FINAL_RC" -ne 0 ]; then
      break
    fi
    for ((ordinal = 1; ordinal <= CURRENT_DECODE_INSTANCES; ordinal++)); do
      clock_key=ganymede
      if [ "$ordinal" -gt 1 ]; then
        clock_key="ganymede-decode-${ordinal}"
      fi
      if ! wait_clock_ack "$SEQ" "$clock_key" "$ganymede_freq"; then
        FINAL_RC=25
        break
      fi
    done
    if [ "$FINAL_RC" -ne 0 ]; then
      break
    fi
  else
    write_clock_request "$SEQ" 0 0
    echo "automatic_dvfs_window seq=${SEQ} manual_clock_commands=false"
  fi

  if [ "$WINDOW_WARMUP_SECONDS" -gt 0 ]; then
    WARMUP_START_TS=$(date +%s.%N)
    echo "${WARMUP_START_TS},warmup_start,${SEQ},${workload_id},${POLICY_LABEL}" >> "${OUT_DIR}/events.csv"
    echo "window_warmup_start time=$(date --iso-8601=seconds) seq=${SEQ} id=${workload_id} seconds=${WINDOW_WARMUP_SECONDS}"
    sleep "$WINDOW_WARMUP_SECONDS"
    WARMUP_END_TS=$(date +%s.%N)
    echo "${WARMUP_END_TS},warmup_end,${SEQ},${workload_id},${POLICY_LABEL}" >> "${OUT_DIR}/events.csv"
    echo "window_warmup_end time=$(date --iso-8601=seconds) seq=${SEQ} id=${workload_id}"
  fi

  NUM_PROMPTS=$((rate * window_seconds))
  if [ "$NUM_PROMPTS" -lt "$MIN_PROMPTS" ]; then
    NUM_PROMPTS="$MIN_PROMPTS"
  fi
  if [ "$NUM_PROMPTS" -gt "$MAX_PROMPTS" ]; then
    NUM_PROMPTS="$MAX_PROMPTS"
  fi
  BENCH_FILE="${OUT_DIR}/bench_${SEQ}_${workload_id}.txt"
  BURSTINESS_ARGS=()
  DETAIL_ARGS=()
  if [ "$VLLM_BURSTINESS_SUPPORTED" = true ] && [ -n "$burstiness" ]; then
    BURSTINESS_ARGS+=(--burstiness "$burstiness")
  fi
  if [ "$BENCH_SAVE_DETAILED" = true ]; then
    DETAIL_ARGS+=(
      --save-result
      --save-detailed
      --result-dir "$OUT_DIR"
      --result-filename "detailed_${SEQ}_${workload_id}.json"
    )
  fi
  echo "bench_config num_prompts=${NUM_PROMPTS} nominal_window_s=${window_seconds} timeout_s=${BENCH_TIMEOUT_SECONDS} num_warmups=${BENCH_NUM_WARMUPS} max_concurrency=${MAX_CONCURRENCY} finite_rate_random_arrivals=true burstiness_applied=${VLLM_BURSTINESS_SUPPORTED} burstiness_value=${burstiness:-none} save_detailed=${BENCH_SAVE_DETAILED}"
  START_TS=$(date +%s.%N)
  echo "${START_TS},workload_start,${SEQ},${workload_id},${POLICY_LABEL}" >> "${OUT_DIR}/events.csv"
  echo "energy_event unix_ts=${START_TS} event=workload_start seq=${SEQ} workload_id=${workload_id}"
  timeout --signal=INT --kill-after=20s "${BENCH_TIMEOUT_SECONDS}s" "$VLLM_BIN" bench serve \
    --backend openai --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --dataset-name random \
    "${BURSTINESS_ARGS[@]}" \
    "${DETAIL_ARGS[@]}" \
    --random-input-len "$il" --random-output-len "$ol" --max-concurrency "$MAX_CONCURRENCY" \
    --num-warmups "$BENCH_NUM_WARMUPS" --temperature 0 </dev/null > "$BENCH_FILE" 2>&1
  BENCH_RC=$?
  END_TS=$(date +%s.%N)
  echo "${END_TS},workload_end,${SEQ},${workload_id},${POLICY_LABEL}" >> "${OUT_DIR}/events.csv"
  BENCHMARKS_COMPLETED=$((BENCHMARKS_COMPLETED + 1))
  echo "energy_event unix_ts=${END_TS} event=workload_end seq=${SEQ} workload_id=${workload_id}"
  echo "bench_rc=${BENCH_RC} seq=${SEQ} id=${workload_id}"
  tail -n 80 "$BENCH_FILE" || true
  read -r BENCH_SUCCESSFUL BENCH_FAILED <<< "$(
    "$PYTHON_BIN" - "$BENCH_FILE" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
successful = re.search(r"Successful requests:\s+(\d+)", text)
failed = re.search(r"Failed requests:\s+(\d+)", text)
print(
    int(successful.group(1)) if successful else -1,
    int(failed.group(1)) if failed else -1,
)
PY
  )"
  echo "bench_request_counts seq=${SEQ} id=${workload_id} successful=${BENCH_SUCCESSFUL} failed=${BENCH_FAILED}"
  if [ "$BENCH_RC" -ne 0 ] || [ "$BENCH_SUCCESSFUL" -le 0 ] || [ "$BENCH_FAILED" -ne 0 ]; then
    BENCH_FAILURES=$((BENCH_FAILURES + 1))
    FINAL_RC=9
  fi
  if ! server_steps_alive; then
    echo "server_step_died_during_sweep=true seq=${SEQ}"
    FINAL_RC=26
    break
  fi
  if ! scale_instances_alive; then
    echo "scale_instance_died_during_sweep=true seq=${SEQ} pids=${SCALE_STEP_PIDS[*]}"
    FINAL_RC=26
    break
  fi
done < "$WORKLOAD_FILE"

if [ "$REQUIRE_ALL_WORKLOADS_EXECUTED" = true ] && \
  [ "$BENCHMARKS_COMPLETED" -ne "$EXPECTED_WORKLOAD_COUNT" ]; then
  echo "incomplete_benchmark_sweep=true completed=${BENCHMARKS_COMPLETED} expected=${EXPECTED_WORKLOAD_COUNT}"
  if [ "$FINAL_RC" -eq 0 ]; then
    FINAL_RC=33
  fi
fi

if ! power_monitors_alive; then
  echo "power_monitor_died_during_sweep=true pids=${POWER_MONITOR_PIDS[*]}"
  dump_power_monitor_logs
  if [ "$FINAL_RC" -eq 0 ]; then
    FINAL_RC=28
  fi
fi

section "stop_servers_then_reset"
stop_components
if ! reset_gpu_clocks; then
  FINAL_RC=11
fi

section "observed_frequency_summary"
"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv
import glob
import json
import os
import sys

out_dir = sys.argv[1]
for path in sorted(glob.glob(os.path.join(out_dir, "clock_*_*.json"))):
    data = json.load(open(path, encoding="utf-8"))
    print(
        f"clock_probe={os.path.basename(path)} target_mhz={data.get('target_freq_mhz')} "
        f"active_min_mhz={data.get('active_clock_min_mhz')} "
        f"active_max_mhz={data.get('active_clock_max_mhz')} "
        f"active_mean_mhz={data.get('active_clock_mean_mhz')}"
    )
for path in sorted(glob.glob(os.path.join(out_dir, "*_telemetry.csv"))):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"telemetry_file={os.path.basename(path)} samples={len(rows)}")
    by_seq = {}
    for row in rows:
        by_seq.setdefault(row.get("workload_seq", "0"), []).append(row)
    for seq, samples in sorted(by_seq.items(), key=lambda item: int(item[0] or 0)):
        active = []
        for row in samples:
            try:
                if float(row["gpu_util_pct"].strip()) >= 10:
                    active.append(float(row["gpu_sm_mhz"].strip()))
            except (TypeError, ValueError):
                pass
        target = samples[-1].get("target_freq_mhz", "NA") if samples else "NA"
        print(
            f"telemetry_seq={seq} target_mhz={target} active_samples={len(active)} "
            f"active_min_mhz={min(active) if active else 'NA'} "
            f"active_max_mhz={max(active) if active else 'NA'}"
        )
PY

section "gpu_energy_summary"
"$PYTHON_BIN" "$ENERGY_SUMMARY_SCRIPT" \
  --out-dir "$OUT_DIR" \
  --output-json "${OUT_DIR}/energy_summary.json" \
  --output-csv "${OUT_DIR}/energy_by_workload.csv" \
  > "${OUT_DIR}/energy_summary.log" 2>&1
ENERGY_RC=$?
cat "${OUT_DIR}/energy_summary.log" 2>/dev/null || true
if [ "$ENERGY_RC" -ne 0 ]; then
  echo "energy_summary_failed_rc=${ENERGY_RC}"
  FINAL_RC=27
fi

if [ "$ENERGY_RC" -eq 0 ]; then
  section "live_performance_power_summary"
  LIVE_SUMMARY_ARGS=()
  if [ "$SCHEDULER_ENABLED" != true ]; then
    LIVE_SUMMARY_ARGS+=(--no-decisions)
  else
    LIVE_SUMMARY_ARGS+=(--decision-prefix decision_active)
  fi
  "$PYTHON_BIN" "$LIVE_SUMMARY_SCRIPT" \
    --out-dir "$OUT_DIR" \
    --workloads "$WORKLOAD_FILE" \
    --slo-ttft "$SLO_TTFT_MS" \
    --slo-tpot "$SLO_TPOT_MS" \
    --output-json "${OUT_DIR}/live_summary.json" \
    --output-csv "${OUT_DIR}/live_summary.csv" \
    "${LIVE_SUMMARY_ARGS[@]}" \
    > "${OUT_DIR}/live_summary.log" 2>&1
  LIVE_SUMMARY_RC=$?
  cat "${OUT_DIR}/live_summary.log" 2>/dev/null || true
  if [ "$LIVE_SUMMARY_RC" -ne 0 ]; then
    echo "live_summary_failed_rc=${LIVE_SUMMARY_RC}"
    FINAL_RC=29
  fi
fi

if [ "$ENERGY_RC" -eq 0 ] && [ "$RUN_REFERENCE_COMPARISON" = true ]; then
  section "saturation_gate_power_comparison"
  "$PYTHON_BIN" "$COMPARE_SCRIPT" \
    --out-dir "$OUT_DIR" \
    --workloads "$WORKLOAD_FILE" \
    --reference-comparison "$REFERENCE_COMPARISON" \
    --output-json "${OUT_DIR}/comparison.json" \
    --output-csv "${OUT_DIR}/comparison.csv" \
    > "${OUT_DIR}/comparison.log" 2>&1
  COMPARE_RC=$?
  cat "${OUT_DIR}/comparison.log" 2>/dev/null || true
  if [ "$COMPARE_RC" -ne 0 ]; then
    echo "comparison_failed_rc=${COMPARE_RC}"
    FINAL_RC=28
  fi
fi

section "result"
echo "workload_count=${WORKLOAD_COUNT}"
echo "gate_admitted=${GATE_ADMITTED} gate_rejected=${GATE_REJECTED}"
echo "overload_fallbacks=${OVERLOAD_FALLBACKS}"
echo "scale_out_events=${SCALE_OUT_EVENTS} scale_in_events=${SCALE_IN_EVENTS}"
echo "bench_failures=${BENCH_FAILURES}"
echo "benchmarks_completed=${BENCHMARKS_COMPLETED} expected_workloads=${EXPECTED_WORKLOAD_COUNT}"
echo "parent_reset_verified=$([ "$RESET_REQUIRED" = true ] && echo "$RESET_DONE" || echo not_required)"
echo "scheduler_enabled=${SCHEDULER_ENABLED}"
echo "active_scheduler_policy=${ACTIVE_SCHEDULER_POLICY}"
echo "saturation_action=${SATURATION_ACTION}"
echo "scale_in_enabled=${SCALE_IN_ENABLED}"
echo "prefill_instances=${CURRENT_PREFILL_INSTANCES} decode_instances=${CURRENT_DECODE_INSTANCES}"
echo "gpu_clock_control_mode=${GPU_CLOCK_CONTROL_MODE}"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
