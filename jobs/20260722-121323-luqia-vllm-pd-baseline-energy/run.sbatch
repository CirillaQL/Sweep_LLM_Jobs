#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-baseline-energy
#SBATCH --partition=long
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-121323-luqia-vllm-pd-baseline-energy/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-121323-luqia-vllm-pd-baseline-energy/slurm-%j.err

set -uo pipefail

SOURCE_JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-121323-luqia-vllm-pd-baseline-energy"
JOB_DIR="${JOB_DIR_OVERRIDE:-${SOURCE_JOB_DIR}}"
OUT_DIR="${JOB_DIR}/baseline_results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
NODE_SCRIPT="${SOURCE_JOB_DIR}/start_pd_node.sh"
PROXY_SCRIPT="${SOURCE_JOB_DIR}/pd_proxy.py"
CLOCK_PROBE="${SOURCE_JOB_DIR}/clock_probe.py"
ENERGY_SCRIPT="${SOURCE_JOB_DIR}/energy_summary.py"
COMPARE_SCRIPT="${SOURCE_JOB_DIR}/compare_results.py"
WORKLOAD_FILE="${SOURCE_JOB_DIR}/workloads.csv"
PHASE_CONTROL_FILE="${OUT_DIR}/phase_control.txt"
EVENTS_FILE="${OUT_DIR}/events.csv"
SCHEDULER_RESULTS_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-114259-luqia-vllm-scheduler-multi-rgc-safe-r2/scheduler_multi_results"
MODEL="mistralai/Mistral-7B-v0.1"

BENCH_WINDOW_SECONDS=10
BENCH_TIMEOUT_SECONDS=180

PROXY_IP=10.1.0.3
PREFILL_IP=10.1.0.6
DECODE_IP=10.1.0.3
PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
PREFILL_HTTP_PORT=$((32000 + PORT_OFFSET))
DECODE_HTTP_PORT=$((33000 + PORT_OFFSET))
KV_PORT=$((34000 + PORT_OFFSET))

export HOME="/data/users/chjing"
export USER=chjing
export LOGNAME=chjing
export HF_HOME="${HOME}/.cache/huggingface"
export XDG_CACHE_HOME="${HOME}/.cache"
export PATH="${ENV_DIR}/bin:${PATH}"
export JOB_DIR OUT_DIR ENV_DIR PYTHON_BIN VLLM_BIN MODEL CLOCK_PROBE
export PHASE_CONTROL_FILE PROXY_IP PREFILL_IP DECODE_IP
export PROXY_HTTP_PORT PROXY_REGISTER_PORT PREFILL_HTTP_PORT DECODE_HTTP_PORT KV_PORT

unset SLURM_CPUS_PER_TASK
mkdir -p "$OUT_DIR"
printf '0 server_start none\n' > "$PHASE_CONTROL_FILE"
echo "unix_ts,event,seq,workload_id" > "$EVENTS_FILE"
ulimit -n 65536

PROXY_PID=""
SERVER_STEP_PID=""
RESET_REQUIRED=false
RESET_DONE=false
FINAL_RC=0
BENCH_FAILURES=0

section() {
  echo
  echo "===== $1 ====="
}

mark_event() {
  "$PYTHON_BIN" - "$EVENTS_FILE" "$1" "$2" "$3" <<'PY'
import sys
import time
path, event, seq, workload_id = sys.argv[1:]
timestamp = time.time()
with open(path, "a", encoding="utf-8") as handle:
    handle.write(f"{timestamp:.9f},{event},{seq},{workload_id}\n")
print(f"energy_event unix_ts={timestamp:.9f} event={event} seq={seq} workload_id={workload_id}")
PY
}

set_phase() {
  printf '%s %s %s\n' "$1" "$2" "$3" > "$PHASE_CONTROL_FILE"
  echo "phase_control seq=$1 phase=$2 workload_id=$3"
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
  local name="$1" url="$2" max_seconds="$3" elapsed=0
  while ! curl -fsS "$url" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo "waiting name=${name} elapsed_s=${elapsed} url=${url}"
    if [ -n "$SERVER_STEP_PID" ] && ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
      echo "server_step_died_while_waiting=true"
      return 1
    fi
    if [ "$elapsed" -ge "$max_seconds" ]; then
      return 1
    fi
  done
  echo "ready name=${name} elapsed_s=${elapsed} url=${url}"
}

section "fixed_baseline_configuration"
echo "placement=neptune:prefill,ganymede:decode"
echo "frequency_policy=default_dvfs_no_scheduler_no_lgc"
echo "workload_file=${WORKLOAD_FILE}"
cat "$WORKLOAD_FILE"

section "allocation"
echo "time=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset} nodes=${SLURM_JOB_NODELIST:-unset}"
echo "slurm_time_limit=00:20:00"
scontrol show job "${SLURM_JOB_ID:-}" 2>&1 || true
if [ "$(hostname -s)" != ganymede ]; then
  echo "expected_batch_host_ganymede=true actual=$(hostname -s)"
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  exit 3
fi

section "preflight"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 "$NODE_SCRIPT" preflight 2>&1 | tee "${OUT_DIR}/preflight.log"
PREFLIGHT_RC=${PIPESTATUS[0]}
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  exit "$PREFLIGHT_RC"
fi

section "prestart_default_dvfs"
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" default > "${OUT_DIR}/prestart_rgc.log" 2>&1
DEFAULT_RC=$?
cat "${OUT_DIR}/prestart_rgc.log" || true
echo "prestart_default_dvfs_rc=${DEFAULT_RC}"
if [ "$DEFAULT_RC" -ne 0 ]; then
  exit 4
fi
RESET_REQUIRED=true

section "start_proxy"
export PROXY_REGISTER_HOST=0.0.0.0
export PROXY_HTTP_HOST=0.0.0.0
"$PYTHON_BIN" -u "$PROXY_SCRIPT" > "${OUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
if ! wait_for_url proxy "http://127.0.0.1:${PROXY_HTTP_PORT}/health" 60; then
  exit 5
fi

section "start_fixed_pd_servers"
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" serve > "${OUT_DIR}/server_step.log" 2>&1 &
SERVER_STEP_PID=$!
if ! wait_for_url prefill "http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/v1/models" 600; then
  exit 6
fi
if ! wait_for_url decode "http://${DECODE_IP}:${DECODE_HTTP_PORT}/v1/models" 600; then
  exit 7
fi

section "wait_for_registry"
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
echo "registry_ready=${REGISTRY_READY}"
if [ "$REGISTRY_READY" != true ]; then
  exit 8
fi

section "pd_smoke_request"
curl -fsS --max-time 180 -X POST "http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  --data-binary "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of Sweden is\",\"max_tokens\":8,\"temperature\":0,\"stream\":false}" \
  > "${OUT_DIR}/smoke_response.json"
SMOKE_RC=$?
echo "smoke_curl_rc=${SMOKE_RC}"
if [ "$SMOKE_RC" -ne 0 ]; then
  exit 9
fi

section "baseline_multi_workload_bench"
SEQ=0
while IFS=, read -r workload_id il ol rate description; do
  if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
    continue
  fi
  SEQ=$((SEQ + 1))
  NUM_PROMPTS=$((rate * BENCH_WINDOW_SECONDS))
  if [ "$NUM_PROMPTS" -lt 12 ]; then NUM_PROMPTS=12; fi
  if [ "$NUM_PROMPTS" -gt 120 ]; then NUM_PROMPTS=120; fi
  BENCH_FILE="${OUT_DIR}/bench_${SEQ}_${workload_id}.txt"

  section "workload_${SEQ}_${workload_id}"
  echo "baseline_workload seq=${SEQ} il=${il} ol=${ol} rate=${rate} num_prompts=${NUM_PROMPTS}"
  set_phase "$SEQ" benchmark "$workload_id"
  mark_event workload_start "$SEQ" "$workload_id"
  timeout --signal=INT --kill-after=20s "${BENCH_TIMEOUT_SECONDS}s" "$VLLM_BIN" bench serve \
    --backend openai --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --dataset-name random \
    --random-input-len "$il" --random-output-len "$ol" --max-concurrency 16 \
    --num-warmups 1 --temperature 0 > "$BENCH_FILE" 2>&1
  BENCH_RC=$?
  mark_event workload_end "$SEQ" "$workload_id"
  set_phase "$SEQ" between "$workload_id"
  echo "bench_rc=${BENCH_RC} seq=${SEQ} id=${workload_id}"
  tail -n 80 "$BENCH_FILE" || true
  if [ "$BENCH_RC" -ne 0 ]; then
    BENCH_FAILURES=$((BENCH_FAILURES + 1))
    FINAL_RC=10
  fi
  if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    FINAL_RC=11
    break
  fi
done < "$WORKLOAD_FILE"

section "stop_servers_then_final_rgc"
set_phase "$SEQ" server_stop none
stop_components
if ! reset_gpu_clocks; then
  FINAL_RC=12
fi

section "energy_summary"
"$PYTHON_BIN" "$ENERGY_SCRIPT" --out-dir "$OUT_DIR" \
  --output-json "${OUT_DIR}/energy_summary.json" \
  --output-csv "${OUT_DIR}/energy_by_workload.csv" \
  > "${OUT_DIR}/energy_summary.log" 2>&1
ENERGY_RC=$?
cat "${OUT_DIR}/energy_summary.log" || true
if [ "$ENERGY_RC" -ne 0 ]; then
  FINAL_RC=13
fi

section "scheduler_vs_baseline_comparison"
if [ -d "$SCHEDULER_RESULTS_DIR" ] && [ "$ENERGY_RC" -eq 0 ]; then
  "$PYTHON_BIN" "$COMPARE_SCRIPT" --baseline-dir "$OUT_DIR" \
    --scheduler-dir "$SCHEDULER_RESULTS_DIR" \
    --output-json "${OUT_DIR}/comparison.json" \
    --output-csv "${OUT_DIR}/comparison.csv" \
    > "${OUT_DIR}/comparison.log" 2>&1
  COMPARE_RC=$?
  cat "${OUT_DIR}/comparison.log" || true
  if [ "$COMPARE_RC" -ne 0 ]; then
    FINAL_RC=14
  fi
else
  echo "scheduler_results_missing=true path=${SCHEDULER_RESULTS_DIR}"
  FINAL_RC=15
fi

section "result"
echo "frequency_policy=default_dvfs_no_scheduler_no_lgc"
echo "workload_count=${SEQ} bench_failures=${BENCH_FAILURES}"
echo "parent_reset_verified=${RESET_DONE}"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
