#!/usr/bin/env bash
#SBATCH --job-name=vllm-lgc990-ga
#SBATCH --partition=long
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

ROOT="/data/users/chjing/vLLM_test"
MODEL="mistralai/Mistral-7B-v0.1"
TARGET_SM_FREQ=990
TEST_SECONDS=600
MONITOR_INTERVAL=5
PORT=$((22000 + (${SLURM_JOB_ID:-0} % 1000)))

JOB_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT_DIR="${JOB_DIR}/vllm_bench_lgc990_results"
mkdir -p "$OUT_DIR"

GPU_ID=0

SERVER_PID=""
SMI_MONITOR_PID=""
CHJING_MONITOR_PID=""
SET_FREQ_RC=999
BENCHMARK_RC=999
BENCHMARK_ELAPSED=0

section() {
  echo
  echo "===== $1 ====="
}

query_gpu() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu \
    --format=csv 2>&1 || true
  nvidia-smi -i "$GPU_ID" -q -d CLOCK 2>&1 || true
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
  stop_pid "$SERVER_PID" "vllm_server"
  stop_pid "$CHJING_MONITOR_PID" "chjing_monitor"
  stop_pid "$SMI_MONITOR_PID" "smi_monitor"
  echo "reset_command=sudo nvidia-smi -i ${GPU_ID} -rgc"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
  query_gpu "after_locked_clock_reset"
  echo "job_exit_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_smi_monitor() {
  local output="$1"
  {
    echo "unix_ts,iso_ts,index,clocks_gr_mhz,clocks_sm_mhz,clocks_mem_mhz,max_gr_mhz,power_w,power_limit_w,util_gpu_pct,memory_used_mib,temperature_c"
    while true; do
      local unix_ts iso_ts row
      unix_ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso_ts=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,clocks.gr,clocks.sm,clocks.mem,clocks.max.graphics,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${unix_ts},${iso_ts},${row}"
      else
        echo "${unix_ts},${iso_ts},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$output" &
  SMI_MONITOR_PID=$!
  echo "smi_monitor_pid=${SMI_MONITOR_PID}"
}

start_chjing_monitor() {
  local output="$1"
  if [ -r "${ROOT}/gpu_monitor.py" ]; then
    python3 -u "${ROOT}/gpu_monitor.py" \
      --monitor --interval "$MONITOR_INTERVAL" --output "$output" --gpu-id "$GPU_ID" &
    CHJING_MONITOR_PID=$!
    echo "chjing_monitor_pid=${CHJING_MONITOR_PID}"
  else
    echo "chjing_monitor_missing=${ROOT}/gpu_monitor.py"
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
  section "vllm_bench_10min"
  cd "$ROOT" || return 2

  local benchmark_start=$SECONDS
  local deadline=$((benchmark_start + TEST_SECONDS))
  local iteration=1
  local bench_rc=0

  echo "benchmark_start=$(date --iso-8601=seconds)"
  while [ "$SECONDS" -lt "$deadline" ]; do
    local remaining=$((deadline - SECONDS))
    local bench_file="${OUT_DIR}/bench_iter${iteration}.txt"
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
      > "$bench_file" 2>&1
    bench_rc=$?

    echo "bench_iter=${iteration} rc=${bench_rc} end=$(date --iso-8601=seconds)"
    tail -n 40 "$bench_file" || true
    if [ "$bench_rc" -eq 124 ]; then
      echo "benchmark_reached_600s_deadline=true"
      bench_rc=0
      break
    fi
    if [ "$bench_rc" -ne 0 ]; then
      echo "benchmark_failed=true"
      break
    fi
    iteration=$((iteration + 1))
  done

  BENCHMARK_ELAPSED=$((SECONDS - benchmark_start))
  BENCHMARK_RC=$bench_rc
  echo "benchmark_end=$(date --iso-8601=seconds)"
  echo "benchmark_elapsed_s=${BENCHMARK_ELAPSED}"
  echo "benchmark_rc=${BENCHMARK_RC}"
  return "$bench_rc"
}

summarize_monitor() {
  section "gpu_monitor_summary"
  python3 - "${OUT_DIR}/nvidia_smi_monitor.csv" "$TARGET_SM_FREQ" "$SET_FREQ_RC" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
target = float(sys.argv[2])
set_rc = int(sys.argv[3])
rows = []
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            rows.append(
                (
                    float(row["clocks_sm_mhz"].strip()),
                    float(row["clocks_gr_mhz"].strip()),
                    float(row["util_gpu_pct"].strip()),
                    float(row["power_w"].strip()),
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

active10 = [row for row in rows if row[2] >= 10.0]
active50 = [row for row in rows if row[2] >= 50.0]

def report(label, values):
    print(f"{label}_samples={len(values)}")
    if values:
        print(f"{label}_min={min(values):.1f}")
        print(f"{label}_max={max(values):.1f}")
        print(f"{label}_mean={statistics.fmean(values):.1f}")

print(f"monitor_file={path}")
print(f"set_frequency_rc={set_rc}")
print(f"target_sm_mhz={target:.1f}")
report("all_sm_mhz", [row[0] for row in rows])
report("active10_sm_mhz", [row[0] for row in active10])
report("active50_sm_mhz", [row[0] for row in active50])
report("active50_power_w", [row[3] for row in active50])
above = sum(row[0] > target + 45.0 for row in active10)
print(f"active10_above_target_plus_45={above}")
print(f"cap_effective={set_rc == 0 and bool(active10) and above == 0}")
PY
}

section "job_info"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES_initial=${CUDA_VISIBLE_DEVICES:-unset}"
echo "physical_gpu_id=${GPU_ID}"
echo "target_sm_mhz=${TARGET_SM_FREQ}"
echo "benchmark_duration_s=${TEST_SECONDS}"
echo "monitor_interval_s=${MONITOR_INTERVAL}"
echo "job_dir=${JOB_DIR}"
echo "output_dir=${OUT_DIR}"
python -c 'import vllm; print("vllm_version=" + vllm.__version__)' || true
scontrol show job "${SLURM_JOB_ID:-}" || true

section "supported_clocks"
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS | sed -n '1,240p' || true
query_gpu "baseline_before_locked_clock"

start_smi_monitor "${OUT_DIR}/nvidia_smi_monitor.csv"
start_chjing_monitor "${OUT_DIR}/chjing_gpu_monitor.csv"
sleep 5

section "set_locked_gpu_clock"
echo "pre_reset_command=sudo nvidia-smi -i ${GPU_ID} -rgc"
timeout 30 sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
echo "set_command=sudo nvidia-smi -i 0 -lgc 990,990"
timeout 30 sudo nvidia-smi -i 0 -lgc 990,990 2>&1
SET_FREQ_RC=$?
echo "set_frequency_rc=${SET_FREQ_RC}"
if [ "$SET_FREQ_RC" -ne 0 ]; then
  echo "locked_clock_setup_failed=true"
  exit 5
fi
sleep 5
query_gpu "after_locked_clock_change"

start_server
query_gpu "after_server_ready"

run_benchmark
BENCHMARK_RC=$?
query_gpu "after_benchmark"

stop_pid "$CHJING_MONITOR_PID" "chjing_monitor"
CHJING_MONITOR_PID=""
stop_pid "$SMI_MONITOR_PID" "smi_monitor"
SMI_MONITOR_PID=""
summarize_monitor

echo "final_benchmark_elapsed_s=${BENCHMARK_ELAPSED}"
echo "final_benchmark_rc=${BENCHMARK_RC}"
exit "$BENCHMARK_RC"
