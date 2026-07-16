#!/usr/bin/env bash
#SBATCH --job-name=cuda-f990-ga-r1
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260716-154233-luqia-cuda-f990-ganymede-retry1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260716-154233-luqia-cuda-f990-ganymede-retry1/slurm-%j.err

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260716-154233-luqia-cuda-f990-ganymede-retry1"
SOURCE="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260716-153851-luqia-cuda-f990-ganymede/compute_load.cu"
BINARY="${JOB_DIR}/compute_load"
MONITOR_CSV="${JOB_DIR}/gpu_frequency_monitor.csv"
NVCC_LOG="${JOB_DIR}/nvcc.log"
LOAD_LOG="${JOB_DIR}/cuda_load.log"

CUDA_HOME="/data/users/chjing/miniforge3/envs/cuda-env"
NVCC="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

TARGET_SM_FREQ=990
TARGET_MEM_FREQ=6251
LOAD_SECONDS=300
MONITOR_INTERVAL=5

GPU_ID="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_ID="${GPU_ID%%,*}"
GPU_ID="${GPU_ID:-0}"

MONITOR_PID=""
SET_FREQ_RC=999

section() {
  echo
  echo "===== $1 ====="
}

query_gpu() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,clocks.max.graphics,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu \
    --format=csv 2>&1 || true
}

stop_monitor() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "stopping_monitor_pid=${MONITOR_PID}"
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  section "cleanup_and_restore"
  stop_monitor
  echo "reset_command=sudo nvidia-smi -i ${GPU_ID} -rac"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rac 2>&1 || true
  query_gpu "after_application_clock_reset"
  echo "job_exit_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT INT TERM

start_monitor() {
  {
    echo "unix_ts,iso_ts,index,clocks_gr_mhz,clocks_sm_mhz,clocks_mem_mhz,app_gr_mhz,app_mem_mhz,max_gr_mhz,power_w,util_gpu_pct,memory_used_mib,temperature_c"
    while true; do
      local unix_ts iso_ts row
      unix_ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso_ts=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,clocks.gr,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,clocks.max.graphics,power.draw,utilization.gpu,memory.used,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${unix_ts},${iso_ts},${row}"
      else
        echo "${unix_ts},${iso_ts},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$MONITOR_CSV" &
  MONITOR_PID=$!
  echo "monitor_pid=${MONITOR_PID}"
}

summarize_monitor() {
  section "frequency_summary"
  python3 - "$MONITOR_CSV" "$TARGET_SM_FREQ" "$SET_FREQ_RC" <<'PY'
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
                    float(row["clocks_sm_mhz"]),
                    float(row["clocks_gr_mhz"]),
                    float(row["app_gr_mhz"]),
                    float(row["util_gpu_pct"]),
                    float(row["power_w"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            pass

active = [row for row in rows if row[3] >= 50.0]

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
report("active_sm_mhz", [row[0] for row in active])
report("active_graphics_mhz", [row[1] for row in active])
report("active_application_mhz", [row[2] for row in active])
report("active_power_w", [row[4] for row in active])
above = sum(row[0] > target + 45.0 for row in active)
print(f"active_above_target_plus_45={above}")
print(f"cap_effective={bool(active) and above == 0}")
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
echo "target_mem_mhz=${TARGET_MEM_FREQ}"
echo "load_seconds=${LOAD_SECONDS}"
echo "monitor_interval_seconds=${MONITOR_INTERVAL}"
echo "source=${SOURCE}"
echo "job_dir=${JOB_DIR}"

section "compile_with_chjing_cuda_env"
echo "nvcc_path=${NVCC}"
if [ ! -x "$NVCC" ]; then
  echo "nvcc_executable=false"
  exit 2
fi
if [ ! -r "$SOURCE" ]; then
  echo "source_readable=false"
  exit 2
fi
"$NVCC" --version 2>&1 || true
"$NVCC" -O3 -std=c++17 "$SOURCE" -o "$BINARY" > "$NVCC_LOG" 2>&1
NVCC_RC=$?
echo "nvcc_rc=${NVCC_RC}"
cat "$NVCC_LOG" || true
if [ "$NVCC_RC" -ne 0 ]; then
  exit 2
fi
ls -lh "$BINARY"

query_gpu "baseline_before_frequency_change"
start_monitor
sleep 5

section "set_frequency_chjing_style"
echo "persistence_command=sudo nvidia-smi -pm 1"
timeout 30 sudo nvidia-smi -pm 1 2>&1 || true
echo "set_command=sudo nvidia-smi -i ${GPU_ID} -ac ${TARGET_MEM_FREQ},${TARGET_SM_FREQ}"
timeout 30 sudo nvidia-smi -i "$GPU_ID" -ac "${TARGET_MEM_FREQ},${TARGET_SM_FREQ}" 2>&1
SET_FREQ_RC=$?
echo "set_frequency_rc=${SET_FREQ_RC}"
sleep 5
query_gpu "after_frequency_change"

section "run_cuda_load"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "command=timeout 420 ${BINARY} ${LOAD_SECONDS} 0"
timeout 420 "$BINARY" "$LOAD_SECONDS" 0 > "$LOAD_LOG" 2>&1
LOAD_RC=$?
echo "cuda_load_rc=${LOAD_RC}"
cat "$LOAD_LOG" || true
query_gpu "after_cuda_load"

stop_monitor
summarize_monitor
exit "$LOAD_RC"

