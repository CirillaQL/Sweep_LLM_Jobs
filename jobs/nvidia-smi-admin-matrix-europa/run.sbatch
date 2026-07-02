#!/usr/bin/env bash
#SBATCH --job-name=nvadmin-europa
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=europa
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"
GRAPHICS_CLOCK="${GRAPHICS_CLOCK:-1200}"
MEMORY_CLOCK="${MEMORY_CLOCK:-405}"
POWER_LIMIT_INT=""

attempt() {
  local name="$1"
  shift
  echo
  echo "=== BEGIN ${name} ==="
  echo "command=$*"
  "$@"
  local rc=$?
  echo "=== RESULT ${name}: rc=${rc} ==="
}

cleanup() {
  echo
  echo "=== CLEANUP: reset graphics clock ==="
  nvidia-smi -i "$GPU_ID" -rgc || true
  echo "=== CLEANUP: reset memory clock ==="
  nvidia-smi -i "$GPU_ID" -rmc || true
  if [ -n "$POWER_LIMIT_INT" ]; then
    echo "=== CLEANUP: restore power limit to ${POWER_LIMIT_INT} W ==="
    nvidia-smi -i "$GPU_ID" -pl "$POWER_LIMIT_INT" || true
  fi
  echo "=== CLEANUP: final clocks ==="
  nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.limit,power.draw --format=csv || true
}

trap cleanup EXIT

echo "target_node=europa"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "USER=${USER:-unset}"
echo "LOGNAME=${LOGNAME:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "GRAPHICS_CLOCK=${GRAPHICS_CLOCK}"
echo "MEMORY_CLOCK=${MEMORY_CLOCK}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found."
  exit 127
fi

POWER_LIMIT_RAW="$(nvidia-smi -i "$GPU_ID" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
POWER_LIMIT_INT="${POWER_LIMIT_RAW%%.*}"
echo "POWER_LIMIT_RAW=${POWER_LIMIT_RAW:-unset}"
echo "POWER_LIMIT_INT=${POWER_LIMIT_INT:-unset}"

echo
echo "=== BASELINE ==="
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.limit,power.draw --format=csv || true

echo
echo "=== SUPPORTED CLOCKS ==="
nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS || true

attempt "lock_graphics_clock" nvidia-smi -i "$GPU_ID" -lgc "$GRAPHICS_CLOCK,$GRAPHICS_CLOCK"
attempt "reset_graphics_clock" nvidia-smi -i "$GPU_ID" -rgc
attempt "lock_memory_clock" nvidia-smi -i "$GPU_ID" -lmc "$MEMORY_CLOCK,$MEMORY_CLOCK"
attempt "reset_memory_clock" nvidia-smi -i "$GPU_ID" -rmc

if [ -n "$POWER_LIMIT_INT" ]; then
  attempt "set_power_limit_to_current_value" nvidia-smi -i "$GPU_ID" -pl "$POWER_LIMIT_INT"
else
  echo "Skipping power limit test because current power limit could not be read."
fi

echo
echo "=== FINAL BEFORE CLEANUP ==="
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.gr,clocks.mem,power.limit,power.draw --format=csv || true
