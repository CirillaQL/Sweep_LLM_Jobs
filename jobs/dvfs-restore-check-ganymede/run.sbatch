#!/usr/bin/env bash
#SBATCH --job-name=restore-ganymede
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "restore_check_node=ganymede"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU list:"
  nvidia-smi -L || true

  echo "Current clocks:"
  nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,clocks.applications.graphics,clocks.applications.memory,power.draw --format=csv || true

  echo "Clock details:"
  nvidia-smi -q -d CLOCK || true
else
  echo "nvidia-smi not found."
fi
