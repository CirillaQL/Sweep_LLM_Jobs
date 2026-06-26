#!/usr/bin/env bash
#SBATCH --job-name=dvfs-smoke
#SBATCH --time=00:02:00

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
GRAPHICS_CLOCK="${GRAPHICS_CLOCK:-1200}"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "GPU_ID=$GPU_ID"
echo "GRAPHICS_CLOCK=$GRAPHICS_CLOCK"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Supported clocks:"
  nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS || true

  echo "Setting GPU graphics clock to ${GRAPHICS_CLOCK} MHz"
  nvidia-smi -i "$GPU_ID" -lgc "$GRAPHICS_CLOCK,$GRAPHICS_CLOCK"

  echo "Test workload starts"
  nvidia-smi -i "$GPU_ID" --query-gpu=name,clocks.gr,clocks.mem,power.draw --format=csv || true
  sleep 10
  echo "Test workload finished"

  echo "Resetting GPU graphics clock"
  nvidia-smi -i "$GPU_ID" -rgc
else
  echo "nvidia-smi not found; this verifies Slurm submission only."
  sleep 10
fi
