#!/usr/bin/env bash
#SBATCH --job-name=gpufreq-gres-eu
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --nodelist=europa
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --gpu-freq=1200,verbose
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"

query_clocks() {
  local label="$1"
  echo
  echo "===== ${label} ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu --format=csv || true
}

run_workload() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import os
import time

print("python_gpu_workload=starting", flush=True)
try:
    import torch
    print(f"torch_version={torch.__version__}", flush=True)
    print(f"torch_cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("torch_cuda_unavailable_sleeping=20s", flush=True)
        time.sleep(20)
    else:
        torch.cuda.set_device(0)
        print(f"torch_device={torch.cuda.get_device_name(0)}", flush=True)
        a = torch.randn((4096, 4096), device="cuda")
        b = torch.randn((4096, 4096), device="cuda")
        torch.cuda.synchronize()
        end = time.time() + 20
        iterations = 0
        while time.time() < end:
            c = a @ b
            torch.cuda.synchronize()
            iterations += 1
        print(f"torch_matmul_iterations={iterations}", flush=True)
except Exception as exc:
    print(f"python_gpu_workload_error={type(exc).__name__}: {exc}", flush=True)
    print("fallback_sleeping=20s", flush=True)
    time.sleep(20)
PY
  else
    echo "python3_not_found_fallback_sleeping=20s"
    sleep 20
  fi
}

echo "target_node=europa"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "whoami=$(whoami)"
echo "id=$(id)"

echo
echo "===== slurm_job_record ====="
scontrol show job "$SLURM_JOB_ID" || true

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found."
  exit 127
fi

query_clocks "baseline_before_workload"

echo
echo "===== clock_sampling_during_workload ====="
run_workload &
workload_pid=$!
for sample in $(seq 1 25); do
  echo "--- sample=${sample} time=$(date --iso-8601=seconds) ---"
  nvidia-smi -i "$GPU_ID" --query-gpu=index,clocks.gr,clocks.mem,power.draw,utilization.gpu --format=csv,noheader || true
  if ! kill -0 "$workload_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
wait "$workload_pid"
workload_rc=$?
echo "workload_rc=${workload_rc}"

query_clocks "after_workload_before_job_end"
