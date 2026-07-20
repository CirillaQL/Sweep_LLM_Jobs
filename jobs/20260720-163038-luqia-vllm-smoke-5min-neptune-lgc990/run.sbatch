#!/usr/bin/env bash
#SBATCH --job-name=vllm-ne-lgc990
#SBATCH --partition=long
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --nodelist=neptune
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-163038-luqia-vllm-smoke-5min-neptune-lgc990/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-163038-luqia-vllm-smoke-5min-neptune-lgc990/slurm-%j.err

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-163038-luqia-vllm-smoke-5min-neptune-lgc990"
OUT_DIR="${JOB_DIR}/vllm_smoke_results"
ROOT="/data/users/chjing/vLLM_test"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
MODEL="mistralai/Mistral-7B-v0.1"
TARGET_SM_FREQ=990
BENCH_SECONDS=300
MONITOR_INTERVAL=5
PORT=$((28000 + (${SLURM_JOB_ID:-0} % 1000)))

export HOME="/data/users/chjing"
export USER="chjing"
export LOGNAME="chjing"
export HF_HOME="${HOME}/.cache/huggingface"
export XDG_CACHE_HOME="${HOME}/.cache"
export PATH="${ENV_DIR}/bin:${PATH}"

mkdir -p "$OUT_DIR"
ulimit -n 65536

GPU_ID="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_ID="${GPU_ID%%,*}"
GPU_ID="${GPU_ID:-0}"

SERVER_PID=""
MONITOR_PID=""
BENCH_RC=999
BENCH_ELAPSED=0
CLOCK_SET_RC=999

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

query_clock() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,clocks.gr,clocks.sm,clocks.mem,clocks.max.graphics,power.draw,utilization.gpu,memory.used,temperature.gpu \
    --format=csv 2>&1 || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  section "cleanup_and_restore"
  stop_pid "$SERVER_PID" "vllm_server"
  stop_pid "$MONITOR_PID" "gpu_monitor"
  echo "reset_clock_command=sudo nvidia-smi -i ${GPU_ID} -rgc"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
  query_clock "after_locked_clock_reset"
  echo "job_exit_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_monitor() {
  {
    echo "unix_ts,iso_ts,index,name,clocks_sm_mhz,clocks_mem_mhz,power_w,util_gpu_pct,memory_used_mib,temperature_c"
    while true; do
      local unix_ts iso_ts row
      unix_ts=$($PYTHON_BIN -c 'import time; print(f"{time.time():.6f}")')
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
  } > "${OUT_DIR}/nvidia_smi_monitor.csv" &
  MONITOR_PID=$!
  echo "monitor_pid=${MONITOR_PID}"
}

set_locked_clock() {
  section "set_locked_gpu_clock"
  echo "allocated_gpu_id=${GPU_ID}"
  echo "pre_reset_command=sudo nvidia-smi -i ${GPU_ID} -rgc"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
  echo "set_command=sudo nvidia-smi -i ${GPU_ID} -lgc ${TARGET_SM_FREQ},${TARGET_SM_FREQ}"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -lgc "${TARGET_SM_FREQ},${TARGET_SM_FREQ}" 2>&1
  CLOCK_SET_RC=$?
  echo "set_frequency_rc=${CLOCK_SET_RC}"
  if [ "$CLOCK_SET_RC" -ne 0 ]; then
    echo "locked_clock_setup_failed=true"
    exit 5
  fi
  sleep 5
  query_clock "after_locked_clock_change"
}

start_server() {
  section "start_vllm_server"
  cd "$ROOT" || exit 2
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "model=${MODEL}"
  echo "port=${PORT}"

  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host localhost \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
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
      tail -n 240 "${OUT_DIR}/vllm_server.log" || true
      exit 3
    fi
    if [ "$elapsed" -ge 480 ]; then
      echo "server_start_timeout=true"
      tail -n 240 "${OUT_DIR}/vllm_server.log" || true
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

    timeout --signal=INT --kill-after=20s "${remaining}s" "$VLLM_BIN" bench serve \
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
    tail -n 60 "$output" || true
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

summarize_monitor() {
  section "gpu_frequency_summary"
  "$PYTHON_BIN" - "${OUT_DIR}/nvidia_smi_monitor.csv" "$TARGET_SM_FREQ" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
target = float(sys.argv[2])
samples = []
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            clock = float(row["clocks_sm_mhz"].strip())
            util = float(row["util_gpu_pct"].strip())
            power = float(row["power_w"].strip())
        except (KeyError, TypeError, ValueError):
            continue
        if util >= 10.0:
            samples.append((clock, util, power))

clocks = [row[0] for row in samples]
print(f"monitor_file={path}")
print(f"target_sm_mhz={target:.0f}")
print(f"active_samples={len(samples)}")
if clocks:
    print(f"active_sm_min_mhz={min(clocks):.0f}")
    print(f"active_sm_max_mhz={max(clocks):.0f}")
    print(f"active_sm_mean_mhz={statistics.fmean(clocks):.2f}")
    outside = sum(abs(value - target) > 15.0 for value in clocks)
    print(f"active_samples_outside_target_15mhz={outside}")
    print(f"cap_effective={outside == 0}")
else:
    print("cap_effective=False")
PY
}

section "job_info"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "HOME=${HOME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES_initial=${CUDA_VISIBLE_DEVICES:-unset}"
echo "allocated_gpu_id=${GPU_ID}"
echo "python_bin=${PYTHON_BIN}"
echo "vllm_bin=${VLLM_BIN}"
echo "target_sm_mhz=${TARGET_SM_FREQ}"
echo "bench_duration_s=${BENCH_SECONDS}"
echo "output_dir=${OUT_DIR}"

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true"
  exit 2
fi
"$PYTHON_BIN" -c 'import torch, vllm; print(f"vllm_version={vllm.__version__}"); print(f"torch_version={torch.__version__}"); print(f"cuda_available={torch.cuda.is_available()}")'
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,memory.total,driver_version --format=csv
query_clock "baseline_before_locked_clock"

start_monitor
set_locked_clock
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
query_clock "after_benchmark_before_reset"

stop_pid "$MONITOR_PID" "gpu_monitor"
MONITOR_PID=""
summarize_monitor

echo "final_bench_elapsed_s=${BENCH_ELAPSED}"
echo "final_bench_rc=${BENCH_RC}"
exit "$BENCH_RC"
