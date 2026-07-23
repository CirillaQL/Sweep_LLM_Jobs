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
OUT_DIR="${JOB_DIR}/saturation_gate_results"
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
LIVE_SUMMARY_SCRIPT="${SOURCE_JOB_DIR}/summarize_live_results.py"
COMPARE_SCRIPT="${SOURCE_JOB_DIR}/compare_gate_results.py"
REFERENCE_COMPARISON="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-122040-luqia-vllm-pd-baseline-energy-r1/baseline_results/comparison.json"
PLACEMENT_FILE="${OUT_DIR}/active_placement.json"
CLOCK_CONTROL_DIR="${OUT_DIR}/clock_control"
MODEL="mistralai/Mistral-7B-v0.1"

SLO_TTFT_MS=500
SLO_TPOT_MS=200
BENCH_WINDOW_SECONDS="${BENCH_WINDOW_SECONDS_OVERRIDE:-10}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS_OVERRIDE:-180}"
MIN_PROMPTS="${MIN_PROMPTS_OVERRIDE:-12}"
MAX_PROMPTS="${MAX_PROMPTS_OVERRIDE:-120}"
EXPECTED_WORKLOAD_COUNT="${EXPECTED_WORKLOAD_COUNT_OVERRIDE:-3}"
ALLOW_ZERO_ADMIT="${ALLOW_ZERO_ADMIT_OVERRIDE:-false}"
SCHEDULER_PLACEMENT="${SCHEDULER_PLACEMENT_OVERRIDE:-l40s-prefill-l4-decode}"
OVERLOAD_ACTION="${OVERLOAD_ACTION_OVERRIDE:-min-slo-violation}"
RUN_REFERENCE_COMPARISON="${RUN_REFERENCE_COMPARISON_OVERRIDE:-true}"
ONLINE_SCHEDULING="${ONLINE_SCHEDULING_OVERRIDE:-false}"
MAX_L4_FREQ="${MAX_L4_FREQ_OVERRIDE:-}"
MAX_L40S_FREQ="${MAX_L40S_FREQ_OVERRIDE:-}"
CLOCK_MISMATCH_ACTION="${CLOCK_MISMATCH_ACTION_OVERRIDE:-abort}"

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

# Minerva may expose an SMT-adjusted value that conflicts with the requested
# Slurm TRES when this batch script launches nested srun steps.
unset SLURM_CPUS_PER_TASK

mkdir -p "$OUT_DIR" "$CLOCK_CONTROL_DIR"
: > "${CLOCK_CONTROL_DIR}/neptune.ack"
: > "${CLOCK_CONTROL_DIR}/ganymede.ack"
echo "unix_ts,event,seq,workload_id,policy" > "${OUT_DIR}/events.csv"
ulimit -n 65536

PROXY_PID=""
SERVER_STEP_PID=""
FINAL_RC=0
RESET_REQUIRED=false
RESET_DONE=false
BENCH_FAILURES=0
GATE_ADMITTED=0
GATE_REJECTED=0
OVERLOAD_FALLBACKS=0

section() {
  echo
  echo "===== $1 ====="
}

stop_components() {
  if [ -n "$SERVER_STEP_PID" ] && kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    echo "stopping_server_step_pid=${SERVER_STEP_PID}"
    kill "$SERVER_STEP_PID" 2>/dev/null || true
    wait "$SERVER_STEP_PID" 2>/dev/null || true
  fi
  SERVER_STEP_PID=""
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
  srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    "$NODE_SCRIPT" reset > "${OUT_DIR}/reset_step.log" 2>&1
  local reset_rc=$?
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
    if [ -n "$SERVER_STEP_PID" ] && ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
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
  local latency_decision="${OUT_DIR}/decision_latency_${seq}_${workload_id}.json"
  local gate_decision="${OUT_DIR}/decision_gate_${seq}_${workload_id}.json"
  local limit_args=()
  if [ -n "$MAX_L4_FREQ" ]; then
    limit_args+=(--max-l4-freq "$MAX_L4_FREQ")
  fi
  if [ -n "$MAX_L40S_FREQ" ]; then
    limit_args+=(--max-l40s-freq "$MAX_L40S_FREQ")
  fi

  echo "scheduler_workload seq=${seq} id=${workload_id} il=${il} ol=${ol} rate=${rate} online=${ONLINE_SCHEDULING}"
  "$PYTHON_BIN" "$SCHEDULER_SCRIPT" \
    --bundle "$MODEL_BUNDLE" --saturation-bundle "$SATURATION_BUNDLE" \
    --policy latency_only --placement "$SCHEDULER_PLACEMENT" \
    --overload-action "$OVERLOAD_ACTION" "${limit_args[@]}" \
    --il "$il" --ol "$ol" --rate "$rate" \
    --slo-ttft "$SLO_TTFT_MS" --slo-tpot "$SLO_TPOT_MS" --tp 1 \
    --output "$latency_decision" > "${OUT_DIR}/scheduler_latency_${seq}_${workload_id}.json"
  local latency_rc=$?
  if [ "$latency_rc" -ne 0 ]; then
    echo "latency_only_scheduler_failed seq=${seq} rc=${latency_rc}"
    return 20
  fi

  "$PYTHON_BIN" "$SCHEDULER_SCRIPT" \
    --bundle "$MODEL_BUNDLE" --saturation-bundle "$SATURATION_BUNDLE" \
    --policy latency_plus_saturation --placement "$SCHEDULER_PLACEMENT" \
    --overload-action "$OVERLOAD_ACTION" "${limit_args[@]}" \
    --il "$il" --ol "$ol" --rate "$rate" \
    --slo-ttft "$SLO_TTFT_MS" --slo-tpot "$SLO_TPOT_MS" --tp 1 \
    --output "$gate_decision" > "${OUT_DIR}/scheduler_gate_${seq}_${workload_id}.json"
  local gate_rc=$?
  if [ "$gate_rc" -ne 0 ] && [ "$gate_rc" -ne 2 ]; then
    echo "saturation_gate_scheduler_failed seq=${seq} rc=${gate_rc}"
    return 21
  fi
  if [ "$gate_rc" -eq 2 ]; then
    GATE_REJECTED=$((GATE_REJECTED + 1))
    echo "saturation_gate_no_admit seq=${seq} id=${workload_id}"
    return 2
  fi

  GATE_ADMITTED=$((GATE_ADMITTED + 1))
  local gate_status overload_probability values
  read -r gate_status overload_probability <<< "$(decision_status "$gate_decision")"
  if [ "$gate_status" = OVERLOAD_FALLBACK ]; then
    OVERLOAD_FALLBACKS=$((OVERLOAD_FALLBACKS + 1))
    echo "overload_fallback_selected seq=${seq} id=${workload_id} predicted_violation_probability=${overload_probability}"
  elif [ "$gate_status" != OK ]; then
    echo "unexpected_gate_status seq=${seq} id=${workload_id} status=${gate_status}"
    return 22
  fi

  values=$(placement_values "$gate_decision") || return 22
  local prefill_node prefill_freq decode_node decode_freq predicted_power
  read -r prefill_node prefill_freq decode_node decode_freq predicted_power <<< "$values"
  echo "scheduler_selected seq=${seq} status=${gate_status} prefill=${prefill_node}:${prefill_freq} decode=${decode_node}:${decode_freq} predicted_cluster_power_w=${predicted_power}"
  if [ "$SCHEDULER_PLACEMENT" = l40s-prefill-l4-decode ] && \
    { [ "$prefill_node" != neptune ] || [ "$decode_node" != ganymede ]; }; then
    echo "unexpected_role_selection=true seq=${seq}"
    return 23
  fi
  if [ -z "$FIRST_PLACEMENT" ]; then
    FIRST_PLACEMENT="$gate_decision"
  fi
  return 0
}

write_clock_request() {
  local seq="$1"
  local neptune_freq="$2"
  local ganymede_freq="$3"
  printf '%s %s\n' "$seq" "$neptune_freq" > "${CLOCK_CONTROL_DIR}/neptune.request.tmp"
  mv "${CLOCK_CONTROL_DIR}/neptune.request.tmp" "${CLOCK_CONTROL_DIR}/neptune.request"
  printf '%s %s\n' "$seq" "$ganymede_freq" > "${CLOCK_CONTROL_DIR}/ganymede.request.tmp"
  mv "${CLOCK_CONTROL_DIR}/ganymede.request.tmp" "${CLOCK_CONTROL_DIR}/ganymede.request"
  echo "clock_request seq=${seq} neptune_mhz=${neptune_freq} ganymede_mhz=${ganymede_freq}"
}

wait_clock_ack() {
  local seq="$1"
  local node_group="$2"
  local target="$3"
  local ack="${CLOCK_CONTROL_DIR}/${node_group}.ack"
  local elapsed=0
  while true; do
    if [ -s "$ack" ]; then
      local ack_seq ack_target ack_rc observed
      read -r ack_seq ack_target ack_rc observed < "$ack" || true
      if [ "${ack_seq:-}" = "$seq" ] && [ -n "${ack_target:-}" ] && \
        [ -n "${ack_rc:-}" ] && [ -n "${observed:-}" ]; then
        echo "clock_ack seq=${ack_seq} node_group=${node_group} target_mhz=${ack_target} rc=${ack_rc} observed_active_mean_mhz=${observed:-NA}"
        if [ "$ack_target" != "$target" ]; then
          return 1
        fi
        if [ "$ack_rc" != 0 ]; then
          if [ "$ack_rc" = 33 ] && [ "$CLOCK_MISMATCH_ACTION" = continue ]; then
            echo "clock_target_not_sustained seq=${ack_seq} node_group=${node_group} target_mhz=${ack_target} observed_active_mean_mhz=${observed:-NA} action=continue"
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
echo "clock_mismatch_action=${CLOCK_MISMATCH_ACTION}"
FIRST_PLACEMENT=""
WORKLOAD_COUNT=0
while IFS=, read -r workload_id il ol rate field5 field6 field7; do
  if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
    continue
  fi
  WORKLOAD_COUNT=$((WORKLOAD_COUNT + 1))
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

if [ "$WORKLOAD_COUNT" -ne "$EXPECTED_WORKLOAD_COUNT" ]; then
  echo "unexpected_scheduler_counts workloads=${WORKLOAD_COUNT} admitted=${GATE_ADMITTED} rejected=${GATE_REJECTED}"
  exit 24
fi

if [ "$GATE_ADMITTED" -eq 0 ]; then
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
echo "model=${MODEL} workload_count=${WORKLOAD_COUNT} slurm_time_limit=00:28:00"
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
  "$LIVE_SUMMARY_SCRIPT"; do
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
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 "$NODE_SCRIPT" preflight 2>&1 | tee "${OUT_DIR}/preflight.log"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "preflight_rc=${PREFLIGHT_RC}"
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  exit "$PREFLIGHT_RC"
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
RESET_REQUIRED=true
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" serve > "${OUT_DIR}/server_step.log" 2>&1 &
SERVER_STEP_PID=$!
echo "server_step_pid=${SERVER_STEP_PID}"

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

section "wait_for_pd_registry"
REGISTRY_FILE="${OUT_DIR}/registry.json"
REGISTRY_READY=false
for attempt in $(seq 1 60); do
  curl -fsS "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" 2>/dev/null || true
  if "$PYTHON_BIN" - "$REGISTRY_FILE" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if len(data.get("prefill", [])) == 1 and len(data.get("decode", [])) == 1 else 1)
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

VLLM_BURSTINESS_SUPPORTED=false
if "$VLLM_BIN" bench serve --help 2>&1 | grep -q -- "--burstiness"; then
  VLLM_BURSTINESS_SUPPORTED=true
fi
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

  if [ "$ONLINE_SCHEDULING" = true ] && [ "$SEQ" -gt 1 ]; then
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

  placement="${OUT_DIR}/decision_gate_${SEQ}_${workload_id}.json"
  if ! values=$(placement_values "$placement" 2>/dev/null); then
    section "workload_${SEQ}_${workload_id}_not_admitted"
    echo "saturation_gate_action=NO_ADMIT seq=${SEQ} id=${workload_id}"
    continue
  fi
  cp "$placement" "$PLACEMENT_FILE"
  read -r prefill_node neptune_freq decode_node ganymede_freq predicted_power <<< "$values"

  section "workload_${SEQ}_${workload_id}"
  echo "workload_start time=$(date --iso-8601=seconds) seq=${SEQ} id=${workload_id} il=${il} ol=${ol} request_rate=${rate} window_seconds=${window_seconds} burstiness=${burstiness} description=${description}"
  echo "recommendation neptune_prefill_mhz=${neptune_freq} ganymede_decode_mhz=${ganymede_freq} predicted_cluster_power_w=${predicted_power}"
  write_clock_request "$SEQ" "$neptune_freq" "$ganymede_freq"
  if ! wait_clock_ack "$SEQ" neptune "$neptune_freq"; then
    FINAL_RC=24
    break
  fi
  if ! wait_clock_ack "$SEQ" ganymede "$ganymede_freq"; then
    FINAL_RC=25
    break
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
  if [ "$VLLM_BURSTINESS_SUPPORTED" = true ] && [ -n "$burstiness" ]; then
    BURSTINESS_ARGS+=(--burstiness "$burstiness")
  fi
  echo "bench_config num_prompts=${NUM_PROMPTS} nominal_window_s=${window_seconds} timeout_s=${BENCH_TIMEOUT_SECONDS} finite_rate_random_arrivals=true burstiness_applied=${VLLM_BURSTINESS_SUPPORTED}"
  START_TS=$(date +%s.%N)
  echo "${START_TS},workload_start,${SEQ},${workload_id},latency_plus_saturation" >> "${OUT_DIR}/events.csv"
  echo "energy_event unix_ts=${START_TS} event=workload_start seq=${SEQ} workload_id=${workload_id}"
  timeout --signal=INT --kill-after=20s "${BENCH_TIMEOUT_SECONDS}s" "$VLLM_BIN" bench serve \
    --backend openai --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --dataset-name random \
    "${BURSTINESS_ARGS[@]}" \
    --random-input-len "$il" --random-output-len "$ol" --max-concurrency 16 \
    --num-warmups 1 --temperature 0 > "$BENCH_FILE" 2>&1
  BENCH_RC=$?
  END_TS=$(date +%s.%N)
  echo "${END_TS},workload_end,${SEQ},${workload_id},latency_plus_saturation" >> "${OUT_DIR}/events.csv"
  echo "energy_event unix_ts=${END_TS} event=workload_end seq=${SEQ} workload_id=${workload_id}"
  echo "bench_rc=${BENCH_RC} seq=${SEQ} id=${workload_id}"
  tail -n 80 "$BENCH_FILE" || true
  if [ "$BENCH_RC" -ne 0 ]; then
    BENCH_FAILURES=$((BENCH_FAILURES + 1))
    FINAL_RC=9
  fi
  if [ -n "$SERVER_STEP_PID" ] && ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    echo "server_step_died_during_sweep=true seq=${SEQ}"
    FINAL_RC=26
    break
  fi
done < "$WORKLOAD_FILE"

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
  "$PYTHON_BIN" "$LIVE_SUMMARY_SCRIPT" \
    --out-dir "$OUT_DIR" \
    --workloads "$WORKLOAD_FILE" \
    --slo-ttft "$SLO_TTFT_MS" \
    --slo-tpot "$SLO_TPOT_MS" \
    --output-json "${OUT_DIR}/live_summary.json" \
    --output-csv "${OUT_DIR}/live_summary.csv" \
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
echo "bench_failures=${BENCH_FAILURES}"
echo "parent_reset_verified=${RESET_DONE}"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
