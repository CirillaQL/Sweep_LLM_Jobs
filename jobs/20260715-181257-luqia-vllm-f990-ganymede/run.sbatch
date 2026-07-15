#!/usr/bin/env bash
#SBATCH --job-name=vllm-f990-ga
#SBATCH --partition=long
#SBATCH --time=00:28:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --gpu-freq=900,verbose
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

ROOT="/data/users/chjing/vLLM_test"
MODEL="mistralai/Mistral-7B-v0.1"
REQUESTED_SLURM_FREQ=900
TARGET_SM_FREQ=990
TARGET_MEM_FREQ=6251
MONITOR_INTERVAL=5
RUN_SECONDS=600
PORT=$((21000 + (${SLURM_JOB_ID:-0} % 1000)))

JOB_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT_DIR="${JOB_DIR}/vllm_f990_results"
mkdir -p "$OUT_DIR"

GPU_ID="${SLURM_JOB_GPUS:-0}"
GPU_ID="${GPU_ID%%,*}"
GPU_ID="${GPU_ID:-0}"

SERVER_PID=""
SMI_MONITOR_PID=""
JING_MONITOR_PID=""
SET_FREQ_RC=999

section() {
  echo
  echo "===== $1 ====="
}

query_gpu() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,clocks.max.graphics,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,power.draw,power.limit,utilization.gpu,memory.used \
    --format=csv || true
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
  section "cleanup_and_restore"
  stop_pid "$JING_MONITOR_PID" "jing_monitor"
  stop_pid "$SMI_MONITOR_PID" "smi_monitor"
  stop_pid "$SERVER_PID" "vllm_server"
  echo "reset_command=sudo nvidia-smi -i ${GPU_ID} -rac"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rac 2>&1 || true
  query_gpu "after_clock_reset"
  echo "job_exit_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT INT TERM

start_smi_monitor() {
  local output="$1"
  {
    echo "unix_ts,iso_ts,index,clocks_gr_mhz,clocks_sm_mhz,clocks_mem_mhz,app_gr_mhz,app_mem_mhz,power_w,util_gpu_pct,memory_used_mib"
    while true; do
      local unix_ts iso_ts row
      unix_ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso_ts=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,clocks.gr,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,power.draw,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${unix_ts},${iso_ts},${row}"
      else
        echo "${unix_ts},${iso_ts},NA,NA,NA,NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$output" &
  SMI_MONITOR_PID=$!
  echo "smi_monitor_pid=${SMI_MONITOR_PID}"
}

start_jing_monitor() {
  local output="$1"
  if [ -r "${ROOT}/gpu_monitor.py" ]; then
    python3 "${ROOT}/gpu_monitor.py" \
      --monitor --interval "$MONITOR_INTERVAL" --output "$output" --gpu-id "$GPU_ID" &
    JING_MONITOR_PID=$!
    echo "jing_monitor_pid=${JING_MONITOR_PID}"
  else
    echo "jing_monitor_missing=${ROOT}/gpu_monitor.py"
  fi
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
  while ! curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 10
    elapsed=$((elapsed + 10))
    echo "waiting_for_server_s=${elapsed}"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server_died=true"
      tail -n 160 "${OUT_DIR}/vllm_server.log" || true
      exit 3
    fi
    if [ "$elapsed" -ge 600 ]; then
      echo "server_start_timeout=true"
      tail -n 160 "${OUT_DIR}/vllm_server.log" || true
      exit 4
    fi
  done
  echo "server_ready=true"
}

run_benchmark() {
  section "vllm_benchmark_10min"
  cd "$ROOT" || exit 2
  local deadline=$((SECONDS + RUN_SECONDS))
  local iteration=1
  while [ "$SECONDS" -lt "$deadline" ]; do
    local remaining=$((deadline - SECONDS))
    local bench_file="${OUT_DIR}/bench_iter${iteration}.txt"
    echo "bench_iter=${iteration} remaining_s=${remaining} start=$(date --iso-8601=seconds)"
    timeout "${remaining}s" vllm bench serve \
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
      > "$bench_file" 2>&1
    local bench_rc=$?
    echo "bench_iter=${iteration} rc=${bench_rc} end=$(date --iso-8601=seconds)"
    tail -n 35 "$bench_file" || true
    if [ "$bench_rc" -ne 0 ] && [ "$bench_rc" -ne 124 ]; then
      echo "benchmark_failed=true"
      break
    fi
    if [ "$bench_rc" -eq 124 ]; then
      echo "benchmark_reached_test_deadline=true"
      break
    fi
    iteration=$((iteration + 1))
  done
}

summarize_frequency() {
  section "measured_frequency_summary"
  python3 - "${OUT_DIR}/nvidia_smi_monitor.csv" "$TARGET_SM_FREQ" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
target = float(sys.argv[2])
rows = []
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            rows.append((float(row["clocks_sm_mhz"]), float(row["util_gpu_pct"])))
        except (KeyError, TypeError, ValueError):
            pass

active10 = [clock for clock, util in rows if util >= 10]
active50 = [clock for clock, util in rows if util >= 50]

def report(label, values):
    print(f"{label}_samples={len(values)}")
    if values:
        print(f"{label}_min_mhz={min(values):.1f}")
        print(f"{label}_max_mhz={max(values):.1f}")
        print(f"{label}_mean_mhz={statistics.fmean(values):.1f}")
        print(f"{label}_above_target_plus_45={sum(v > target + 45 for v in values)}")

print(f"monitor_file={path}")
print(f"target_sm_mhz={target:.1f}")
report("all", [clock for clock, _ in rows])
report("active10", active10)
report("active50", active50)
print(f"cap_effective={bool(active10) and max(active10) <= target + 45}")
PY
}

section "job_info"
echo "host=$(hostname)"
echo "time=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "requested_slurm_freq_mhz=${REQUESTED_SLURM_FREQ}"
echo "target_application_clock_mhz=${TARGET_SM_FREQ}"
echo "monitor_interval_s=${MONITOR_INTERVAL}"
echo "benchmark_duration_s=${RUN_SECONDS}"
echo "output_dir=${OUT_DIR}"
python -c 'import vllm; print("vllm_version=" + vllm.__version__)' || true
scontrol show job "$SLURM_JOB_ID" || true

section "supported_clocks"
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS | sed -n '1,240p' || true
query_gpu "baseline_before_set"

section "set_frequency_chjing_style"
timeout 30 sudo nvidia-smi -pm 1 2>&1 || true
echo "set_command=sudo nvidia-smi -i ${GPU_ID} -ac ${TARGET_MEM_FREQ},${TARGET_SM_FREQ}"
timeout 30 sudo nvidia-smi -i "$GPU_ID" -ac "${TARGET_MEM_FREQ},${TARGET_SM_FREQ}" 2>&1
SET_FREQ_RC=$?
echo "set_frequency_rc=${SET_FREQ_RC}"
query_gpu "after_set_before_server"

start_smi_monitor "${OUT_DIR}/nvidia_smi_monitor.csv"
start_jing_monitor "${OUT_DIR}/jing_gpu_monitor.csv"
start_server
query_gpu "after_server_ready"
run_benchmark
query_gpu "after_benchmark"

stop_pid "$JING_MONITOR_PID" "jing_monitor"
JING_MONITOR_PID=""
stop_pid "$SMI_MONITOR_PID" "smi_monitor"
SMI_MONITOR_PID=""
summarize_frequency

