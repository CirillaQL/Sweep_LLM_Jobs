#!/usr/bin/env bash
#SBATCH --job-name=dvfs-ganymede
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --gpu-freq=1200,verbose
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "target_node=ganymede"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Allocated GPUs:"
  nvidia-smi -L || true

  echo "GPU clocks while Slurm gpu-freq request is active:"
  nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true

  echo "Test workload starts"
  sleep 10
  echo "Test workload finished"

  echo "Final in-job GPU clocks before Slurm releases the allocation:"
  nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true
else
  echo "nvidia-smi not found; this verifies Slurm submission only."
  sleep 10
fi

echo "Slurm should reset GPU frequencies after this job completes."
