#!/usr/bin/env bash
#SBATCH --job-name=nvsmidvfs-gany
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"
GRAPHICS_CLOCK="${GRAPHICS_CLOCK:-1200}"

reset_gpu_clock() {
  echo "Restoring GPU ${GPU_ID} graphics clock with nvidia-smi -rgc"
  nvidia-smi -i "$GPU_ID" -rgc || true
  echo "Post-restore clocks:"
  nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true
}

trap reset_gpu_clock EXIT

echo "target_node=ganymede"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "GRAPHICS_CLOCK=${GRAPHICS_CLOCK}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found."
  exit 127
fi

echo "Baseline clocks:"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true

echo "Supported clocks:"
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS || true

echo "Setting GPU ${GPU_ID} graphics clock to ${GRAPHICS_CLOCK} MHz via nvidia-smi -lgc"
nvidia-smi -i "$GPU_ID" -lgc "$GRAPHICS_CLOCK,$GRAPHICS_CLOCK"
set_rc=$?
if [ "$set_rc" -ne 0 ]; then
  echo "Failed to set GPU graphics clock, rc=${set_rc}"
  exit "$set_rc"
fi

echo "Clocks after setting:"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true

echo "Test workload starts"
sleep 10
echo "Test workload finished"

echo "Clocks before restore:"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.draw --format=csv || true
