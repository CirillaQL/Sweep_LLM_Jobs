#!/usr/bin/env bash
#SBATCH --job-name=gpufreq-cuda-ga
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --gres=nvgpufreq:1
#SBATCH --gpu-freq=1200,verbose
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"

query_clocks() {
  local label="$1"
  echo
  echo "===== ${label} ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,utilization.gpu --format=csv || true
}

write_cuda_burn() {
  cat > "${JOB_DIR}/cuda_burn.cu" <<'CU'
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void burn(float *data, unsigned long iters) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x = data[idx];
  for (unsigned long i = 0; i < iters; ++i) {
    x = fmaf(x, 1.000001f, 0.000001f);
    x = fmaf(x, 0.999999f, 0.000003f);
  }
  data[idx] = x;
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
  int device = 0;
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
    return 2;
  }

  const int blocks = 4096;
  const int threads = 256;
  const size_t count = (size_t)blocks * (size_t)threads;
  float *data = NULL;
  err = cudaMalloc(&data, count * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return 3;
  }
  cudaMemset(data, 1, count * sizeof(float));

  double end = now_sec() + 20.0;
  unsigned long launches = 0;
  while (now_sec() < end) {
    burn<<<blocks, threads>>>(data, 20000UL);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
      cudaFree(data);
      return 4;
    }
    launches++;
  }

  printf("cuda_burn_launches=%lu\n", launches);
  cudaFree(data);
  return 0;
}
CU
}

run_workload() {
  echo "workload_strategy=discover"
  echo "PATH=${PATH}"
  command -v nvcc || true
  command -v python3 || true
  command -v bandwidthTest || true
  command -v nbody || true
  command -v dcgmi || true

  if command -v nvcc >/dev/null 2>&1; then
    echo "workload_strategy=nvcc_cuda_burn"
    write_cuda_burn
    nvcc -O3 "${JOB_DIR}/cuda_burn.cu" -o "${JOB_DIR}/cuda_burn"
    "${JOB_DIR}/cuda_burn"
    return $?
  fi

  for candidate in \
    /usr/local/cuda/extras/demo_suite/bandwidthTest \
    /usr/local/cuda/samples/bin/x86_64/linux/release/bandwidthTest \
    "$(command -v bandwidthTest 2>/dev/null || true)"
  do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      echo "workload_strategy=bandwidthTest candidate=${candidate}"
      end=$((SECONDS + 20))
      while [ "$SECONDS" -lt "$end" ]; do
        "$candidate" --device=0 >/dev/null 2>&1 || true
      done
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    echo "workload_strategy=python_gpu_libraries"
    python3 - <<'PY'
import time

for name in ("torch", "cupy", "numba", "tensorflow", "jax"):
    try:
        __import__(name)
        print(f"python_module_available={name}", flush=True)
    except Exception as exc:
        print(f"python_module_unavailable={name}: {type(exc).__name__}", flush=True)

try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
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
    else:
        print("torch_cuda_available=False", flush=True)
        time.sleep(20)
except Exception as exc:
    print(f"python_gpu_workload_error={type(exc).__name__}: {exc}", flush=True)
    time.sleep(20)
PY
    return 0
  fi

  echo "workload_strategy=none_available_sleep_20s"
  sleep 20
}

echo "target_node=ganymede"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_dir=${JOB_DIR}"
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
for sample in $(seq 1 35); do
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
