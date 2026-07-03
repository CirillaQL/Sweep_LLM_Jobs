#!/usr/bin/env bash
#SBATCH --job-name=vllm-f1000-mon
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --nodelist=europa
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --gpu-freq=1000,verbose
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

ROOT="/data/users/chjing/vLLM_test"
MODEL="mistralai/Mistral-7B-v0.1"
TARGET_SM_FREQ="${TARGET_SM_FREQ:-1000}"
FALLBACK_SM_FREQ="${FALLBACK_SM_FREQ:-990}"
TARGET_MEM_FREQ="${TARGET_MEM_FREQ:-6251}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"
RUN_SECONDS="${RUN_SECONDS:-720}"
TP="${TP:-1}"
PORT=$((18000 + (${SLURM_JOB_ID:-0} % 1000)))

JOB_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT_DIR="${JOB_DIR}/vllm_f1000_results"
mkdir -p "$OUT_DIR"

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"
GPU_ID="${GPU_ID:-0}"

SERVER_PID=""
MONITOR_PID=""
SMI_MONITOR_PID=""
ACTIVE_SM_FREQ="$TARGET_SM_FREQ"

section() {
  echo
  echo "===== $1 ====="
}

query_gpu() {
  local label="$1"
  section "$label"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu,memory.used \
    --format=csv || true
}

stop_pid() {
  local pid="$1"
  local label="$2"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping ${label} pid=${pid}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  section "cleanup"
  stop_pid "$MONITOR_PID" "gpu_monitor"
  stop_pid "$SMI_MONITOR_PID" "nvidia_smi_monitor"
  stop_pid "$SERVER_PID" "vllm_server"
  echo "Resetting GPU application clocks..."
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -rac 2>&1 || true
  timeout 30 sudo nvidia-smi -rac 2>&1 || true
  query_gpu "after_cleanup_reset"
  exit "$rc"
}

trap cleanup EXIT INT TERM

start_smi_monitor() {
  local out="$1"
  {
    echo "unix_ts,iso_ts,index,clocks_gr_mhz,clocks_mem_mhz,power_w,util_gpu_pct,memory_used_mib"
    while true; do
      local ts iso row
      ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,clocks.gr,clocks.mem,power.draw,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${ts},${iso},${row}"
      else
        echo "${ts},${iso},NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$out" &
  SMI_MONITOR_PID=$!
  echo "nvidia_smi_monitor_pid=${SMI_MONITOR_PID}"
}

start_chjing_monitor() {
  local out="$1"
  if [ -r "${ROOT}/gpu_monitor.py" ]; then
    python3 "${ROOT}/gpu_monitor.py" --monitor --interval "$MONITOR_INTERVAL" --output "$out" --gpu-id "$GPU_ID" &
    MONITOR_PID=$!
    echo "gpu_monitor_pid=${MONITOR_PID}"
  else
    echo "gpu_monitor_missing=${ROOT}/gpu_monitor.py"
  fi
}

try_set_freq() {
  local freq="$1"
  section "set_frequency_${freq}"
  echo "command=sudo nvidia-smi -i ${GPU_ID} -ac ${TARGET_MEM_FREQ},${freq}"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -ac "${TARGET_MEM_FREQ},${freq}" 2>&1
  local rc=$?
  echo "set_frequency_${freq}_rc=${rc}"
  return "$rc"
}

start_server() {
  section "start_vllm_server"
  cd "$ROOT" || exit 2
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PORT=${PORT}"
  echo "MODEL=${MODEL}"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host localhost \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
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
      tail -n 80 "${OUT_DIR}/vllm_server.log" || true
      exit 3
    fi
    if [ "$elapsed" -ge 600 ]; then
      echo "server_start_timeout=true"
      tail -n 120 "${OUT_DIR}/vllm_server.log" || true
      exit 4
    fi
  done
  echo "server_ready=true"
}

run_bench_loop() {
  section "bench_loop"
  cd "$ROOT" || exit 2
  local deadline=$((SECONDS + RUN_SECONDS))
  local iter=1
  while [ "$SECONDS" -lt "$deadline" ]; do
    local bench_file="${OUT_DIR}/bench_iter${iter}.txt"
    echo "bench_iter=${iter} start=$(date --iso-8601=seconds) seconds=${SECONDS}"
    vllm bench serve \
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
    local rc=$?
    echo "bench_iter=${iter} rc=${rc} end=$(date --iso-8601=seconds) seconds=${SECONDS}"
    tail -n 35 "$bench_file" || true
    if [ "$rc" -ne 0 ]; then
      echo "bench_failed_iter=${iter}"
      break
    fi
    iter=$((iter + 1))
  done
}

summarize_results() {
  section "summary"
  python3 - "$OUT_DIR" "$ACTIVE_SM_FREQ" <<'PY'
import csv
import os
import statistics
import sys

out_dir = sys.argv[1]
target = float(sys.argv[2])
threshold = target + 50

def fnum(value):
    try:
        return float(str(value).strip().split()[0])
    except Exception:
        return None

def summarize(path, freq_names, util_names):
    freqs = []
    utils = []
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for row in reader:
            freq = next((fnum(row.get(name)) for name in freq_names if row.get(name) not in (None, "")), None)
            util = next((fnum(row.get(name)) for name in util_names if row.get(name) not in (None, "")), None)
            if freq is not None:
                freqs.append(freq)
            if util is not None:
                utils.append(util)
    active = [freq for freq, util in zip(freqs, utils) if util >= 50] if len(freqs) == len(utils) else freqs
    return {
        "path": path,
        "exists": True,
        "samples": len(freqs),
        "active_samples": len(active),
        "min": min(freqs) if freqs else None,
        "max": max(freqs) if freqs else None,
        "mean": statistics.fmean(freqs) if freqs else None,
        "active_min": min(active) if active else None,
        "active_max": max(active) if active else None,
        "active_mean": statistics.fmean(active) if active else None,
        "looks_capped": bool(active) and max(active) <= threshold,
        "fields": fields,
    }

items = [
    summarize(os.path.join(out_dir, "monitor_gpu.csv"), ["gpu_freq_mhz", "sm_clock_mhz", "clock_mhz"], ["gpu_util_pct", "gpu_util"]),
    summarize(os.path.join(out_dir, "nvidia_smi_monitor.csv"), ["clocks_gr_mhz", "clocks_sm_mhz"], ["util_gpu_pct"]),
]
for item in items:
    print(f"summary_file={item['path']}")
    if not item["exists"]:
        print("exists=false")
        continue
    print("exists=true")
    print("fields=" + ",".join(item["fields"]))
    print(
        f"samples={item['samples']} active_samples={item['active_samples']} "
        f"min={item['min']} max={item['max']} mean={item['mean']} "
        f"active_min={item['active_min']} active_max={item['active_max']} "
        f"active_mean={item['active_mean']} looks_capped={item['looks_capped']} "
        f"target={target} threshold={threshold}"
    )

bench_files = sorted(p for p in os.listdir(out_dir) if p.startswith("bench_iter") and p.endswith(".txt"))
print(f"bench_files={len(bench_files)}")
for name in bench_files[-3:]:
    path = os.path.join(out_dir, name)
    print(f"bench_tail={path}")
    with open(path, errors="replace") as fh:
        lines = fh.readlines()[-20:]
    for line in lines:
        print(line.rstrip())
PY
}

section "job_info"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "CUDA_VISIBLE_DEVICES_initial=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "RUN_SECONDS=${RUN_SECONDS}"
echo "MONITOR_INTERVAL=${MONITOR_INTERVAL}"

section "slurm_job_record"
scontrol show job "$SLURM_JOB_ID" || true

section "supported_clocks"
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS | sed -n '1,180p' || true

query_gpu "baseline_before_set"

section "persistence_mode"
timeout 30 sudo nvidia-smi -pm 1 2>&1 || true
query_gpu "after_persistence_mode"

if try_set_freq "$TARGET_SM_FREQ"; then
  ACTIVE_SM_FREQ="$TARGET_SM_FREQ"
else
  echo "target_${TARGET_SM_FREQ}_failed=true"
  echo "Trying fallback supported L4 clock ${FALLBACK_SM_FREQ} MHz"
  if try_set_freq "$FALLBACK_SM_FREQ"; then
    ACTIVE_SM_FREQ="$FALLBACK_SM_FREQ"
  else
    echo "fallback_${FALLBACK_SM_FREQ}_failed=true"
    ACTIVE_SM_FREQ="$TARGET_SM_FREQ"
  fi
fi
echo "ACTIVE_SM_FREQ=${ACTIVE_SM_FREQ}"
query_gpu "after_set_before_server"

start_chjing_monitor "${OUT_DIR}/monitor_gpu.csv"
start_smi_monitor "${OUT_DIR}/nvidia_smi_monitor.csv"
start_server
query_gpu "after_server_ready"
run_bench_loop
query_gpu "after_bench_before_cleanup"
summarize_results
