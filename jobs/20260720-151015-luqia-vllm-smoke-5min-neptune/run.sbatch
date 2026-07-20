#!/usr/bin/env bash
#SBATCH --job-name=vllm-smoke5-ne
#SBATCH --partition=long
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --nodelist=neptune
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

ROOT="/data/users/chjing/vLLM_test"
MODEL="mistralai/Mistral-7B-v0.1"
BENCH_SECONDS=300
MONITOR_INTERVAL=5
PORT=$((24000 + (${SLURM_JOB_ID:-0} % 1000)))

JOB_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT_DIR="${JOB_DIR}/vllm_smoke_results"
mkdir -p "$OUT_DIR"

GPU_ID=0
SERVER_PID=""
MONITOR_PID=""
BENCH_RC=999
BENCH_ELAPSED=0

section() {
  echo
  echo "===== $1 ====="
}

stop_pid() {
  local pid="$1"
  local label="$2"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "stopping_${label}_pid=${pid}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  section "cleanup"
  stop_pid "$SERVER_PID" "vllm_server"
  stop_pid "$MONITOR_PID" "gpu_monitor"
  echo "job_exit_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_monitor() {
  local output="$1"
  {
    echo "unix_ts,iso_ts,index,name,clocks_sm_mhz,clocks_mem_mhz,power_w,util_gpu_pct,memory_used_mib,temperature_c"
    while true; do
      local unix_ts iso_ts row
      unix_ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso_ts=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,name,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${unix_ts},${iso_ts},${row}"
      else
        echo "${unix_ts},${iso_ts},NA,NA,NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$output" &
  MONITOR_PID=$!
  echo "monitor_pid=${MONITOR_PID}"
}

start_server() {
  section "start_vllm_server"
  cd "$ROOT" || exit 2
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "model=${MODEL}"
  echo "port=${PORT}"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host localhost \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --disable-log-requests \
    > "${OUT_DIR}/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  echo "server_pid=${SERVER_PID}"

  local elapsed=0
  while ! curl -fsS "http://localhost:${PORT}/v1/models" > "${OUT_DIR}/models_response.json" 2>/dev/null; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo "waiting_for_server_s=${elapsed}"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server_died=true"
      tail -n 200 "${OUT_DIR}/vllm_server.log" || true
      exit 3
    fi
    if [ "$elapsed" -ge 420 ]; then
      echo "server_start_timeout=true"
      tail -n 200 "${OUT_DIR}/vllm_server.log" || true
      exit 4
    fi
  done

  echo "server_ready=true"
  echo "server_ready_after_s=${elapsed}"
  cat "${OUT_DIR}/models_response.json" || true
}

run_bench_for_5min() {
  section "vllm_bench_5min"
  cd "$ROOT" || return 2

  local started=$SECONDS
  local deadline=$((started + BENCH_SECONDS))
  local iteration=1
  local rc=0

  echo "bench_start=$(date --iso-8601=seconds)"
  while [ "$SECONDS" -lt "$deadline" ]; do
    local remaining=$((deadline - SECONDS))
    local output="${OUT_DIR}/bench_iter${iteration}.txt"
    echo "bench_iter=${iteration} remaining_s=${remaining} start=$(date --iso-8601=seconds)"

    timeout --signal=INT --kill-after=20s "${remaining}s" vllm bench serve \
      --backend openai \
      --base-url "http://localhost:${PORT}" \
      --model "$MODEL" \
      --num-prompts 120 \
      --request-rate 2 \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 128 \
      --max-concurrency 64 \
      --num-warmups 5 \
      > "$output" 2>&1
    rc=$?

    echo "bench_iter=${iteration} rc=${rc} end=$(date --iso-8601=seconds)"
    tail -n 50 "$output" || true
    if [ "$rc" -eq 124 ]; then
      echo "bench_reached_300s_deadline=true"
      rc=0
      break
    fi
    if [ "$rc" -ne 0 ]; then
      echo "bench_failed=true"
      break
    fi
    iteration=$((iteration + 1))
  done

  BENCH_ELAPSED=$((SECONDS - started))
  BENCH_RC=$rc
  echo "bench_end=$(date --iso-8601=seconds)"
  echo "bench_elapsed_s=${BENCH_ELAPSED}"
  echo "bench_rc=${BENCH_RC}"
  return "$rc"
}

section "job_info"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES_initial=${CUDA_VISIBLE_DEVICES:-unset}"
echo "bench_duration_s=${BENCH_SECONDS}"
echo "output_dir=${OUT_DIR}"
echo "python=$(command -v python || true)"
echo "vllm=$(command -v vllm || true)"
python -c 'import vllm; print("vllm_version=" + vllm.__version__)'
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,memory.total,driver_version --format=csv

start_monitor "${OUT_DIR}/nvidia_smi_monitor.csv"
start_server
run_bench_for_5min
BENCH_RC=$?

section "post_bench_validation"
if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "server_alive_after_bench=true"
else
  echo "server_alive_after_bench=false"
  BENCH_RC=6
fi
curl -fsS "http://localhost:${PORT}/v1/models" || BENCH_RC=7
echo
echo "final_bench_elapsed_s=${BENCH_ELAPSED}"
echo "final_bench_rc=${BENCH_RC}"

exit "$BENCH_RC"
