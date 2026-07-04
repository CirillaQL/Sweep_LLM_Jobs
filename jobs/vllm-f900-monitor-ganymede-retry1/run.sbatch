#!/usr/bin/env bash
#SBATCH --job-name=vllm-f900-ga-r1
#SBATCH --partition=long
#SBATCH --time=00:30:00
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

set -u

ROOT="/data/users/chjing/vLLM_test"
MODEL="mistralai/Mistral-7B-v0.1"
TARGET_SM_FREQ="${TARGET_SM_FREQ:-900}"
TARGET_MEM_FREQ="${TARGET_MEM_FREQ:-6251}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"
RUN_SECONDS="${RUN_SECONDS:-600}"
TP="${TP:-1}"
PORT=$((19000 + (${SLURM_JOB_ID:-0} % 1000)))

JOB_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT_DIR="${JOB_DIR}/vllm_f900_results"
mkdir -p "$OUT_DIR"

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"
GPU_ID="${GPU_ID:-0}"

SERVER_PID=""
GPU_MONITOR_PID=""
SMI_MONITOR_PID=""
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
    echo "Stopping ${label} pid=${pid}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  section "cleanup"
  stop_pid "$GPU_MONITOR_PID" "gpu_monitor.py"
  stop_pid "$SMI_MONITOR_PID" "nvidia-smi monitor"
  stop_pid "$SERVER_PID" "vLLM server"
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
    echo "unix_ts,iso_ts,index,clocks_gr_mhz,clocks_sm_mhz,clocks_mem_mhz,app_gr_mhz,app_mem_mhz,power_w,util_gpu_pct,memory_used_mib"
    while true; do
      local ts iso row
      ts=$(python3 -c 'import time; print(f"{time.time():.6f}")')
      iso=$(date --iso-8601=seconds)
      row=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,clocks.gr,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,power.draw,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if [ -n "$row" ]; then
        echo "${ts},${iso},${row}"
      else
        echo "${ts},${iso},NA,NA,NA,NA,NA,NA,NA,NA,NA"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  } > "$out" &
  SMI_MONITOR_PID=$!
  echo "smi_monitor_pid=${SMI_MONITOR_PID}"
}

start_gpu_monitor() {
  local out="$1"
  if [ -r "${ROOT}/gpu_monitor.py" ]; then
    python3 "${ROOT}/gpu_monitor.py" --monitor --interval "$MONITOR_INTERVAL" --output "$out" --gpu-id "$GPU_ID" &
    GPU_MONITOR_PID=$!
    echo "gpu_monitor_pid=${GPU_MONITOR_PID}"
  else
    echo "gpu_monitor_missing=${ROOT}/gpu_monitor.py"
  fi
}

set_frequency_chjing_style() {
  section "set_frequency_chjing_style"
  echo "command=sudo nvidia-smi -i ${GPU_ID} -ac ${TARGET_MEM_FREQ},${TARGET_SM_FREQ}"
  timeout 30 sudo nvidia-smi -i "$GPU_ID" -ac "${TARGET_MEM_FREQ},${TARGET_SM_FREQ}" 2>&1
  SET_FREQ_RC=$?
  echo "set_frequency_rc=${SET_FREQ_RC}"
  return 0
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
      tail -n 120 "${OUT_DIR}/vllm_server.log" || true
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

run_bench_for_10min() {
  section "bench_loop_10min"
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
    tail -n 40 "$bench_file" || true
    if [ "$rc" -ne 0 ]; then
      echo "bench_failed_iter=${iter}"
      break
    fi
    iter=$((iter + 1))
  done
}

summarize_csvs() {
  section "summary"
  python3 - "$OUT_DIR" "$TARGET_SM_FREQ" "$SET_FREQ_RC" <<'PY'
import csv
import os
import statistics
import sys

out_dir = sys.argv[1]
target = float(sys.argv[2])
set_freq_rc = int(sys.argv[3])
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
            freq = None
            for name in freq_names:
                if row.get(name) not in (None, ""):
                    freq = fnum(row.get(name))
                    if freq is not None:
                        break
            util = None
            for name in util_names:
                if row.get(name) not in (None, ""):
                    util = fnum(row.get(name))
                    if util is not None:
                        break
            if freq is not None:
                freqs.append(freq)
            if util is not None:
                utils.append(util)
    active = [freq for freq, util in zip(freqs, utils) if util >= 50] if len(freqs) == len(utils) else freqs
    return {
        "path": path,
        "exists": True,
        "fields": fields,
        "samples": len(freqs),
        "active_samples": len(active),
        "min": min(freqs) if freqs else None,
        "max": max(freqs) if freqs else None,
        "mean": statistics.fmean(freqs) if freqs else None,
        "active_min": min(active) if active else None,
        "active_max": max(active) if active else None,
        "active_mean": statistics.fmean(active) if active else None,
        "looks_capped": bool(active) and max(active) <= threshold,
    }

items = [
    summarize(os.path.join(out_dir, "monitor_gpu.csv"), ["gpu_freq_mhz", "sm_clock_mhz", "clock_mhz"], ["gpu_util_pct", "gpu_util"]),
    summarize(os.path.join(out_dir, "nvidia_smi_monitor.csv"), ["clocks_gr_mhz", "clocks_sm_mhz"], ["util_gpu_pct"]),
]

print(f"target_sm_mhz={target}")
print(f"set_frequency_rc={set_freq_rc}")
print(f"cap_threshold_mhz={threshold}")
for item in items:
    print()
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
        f"active_mean={item['active_mean']} looks_capped={item['looks_capped']}"
    )

bench_files = sorted(p for p in os.listdir(out_dir) if p.startswith("bench_iter") and p.endswith(".txt"))
print()
print(f"bench_files={len(bench_files)}")
for name in bench_files[-3:]:
    path = os.path.join(out_dir, name)
    print(f"bench_tail={path}")
    with open(path, errors="replace") as fh:
        for line in fh.readlines()[-25:]:
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
echo "TARGET_SM_FREQ=${TARGET_SM_FREQ}"
echo "TARGET_MEM_FREQ=${TARGET_MEM_FREQ}"
echo "MONITOR_INTERVAL=${MONITOR_INTERVAL}"
echo "RUN_SECONDS=${RUN_SECONDS}"
echo "OUT_DIR=${OUT_DIR}"

section "slurm_job_record"
scontrol show job "$SLURM_JOB_ID" || true

section "nvgpufreq_status_files"
for file in /run/nvgpufreq.run /var/run/nvgpufreq.run; do
  echo "--- ${file} ---"
  if [ -e "$file" ]; then
    ls -l "$file" || true
    cat "$file" || true
    echo
  else
    echo "missing"
  fi
done

section "supported_clocks"
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS | sed -n '1,220p' || true

query_gpu "baseline_before_set"

section "persistence_mode"
timeout 30 sudo nvidia-smi -pm 1 2>&1 || true
query_gpu "after_persistence_mode"

set_frequency_chjing_style
query_gpu "after_set_before_server"

start_gpu_monitor "${OUT_DIR}/monitor_gpu.csv"
start_smi_monitor "${OUT_DIR}/nvidia_smi_monitor.csv"
start_server
query_gpu "after_server_ready"
run_bench_for_10min
query_gpu "after_bench_before_cleanup"
summarize_csvs
