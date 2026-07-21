#!/usr/bin/env bash
#SBATCH --job-name=rgc-check-ganymede
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260721-153101-luqia-rgc-check-ganymede/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260721-153101-luqia-rgc-check-ganymede/slurm-%j.err

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260721-153101-luqia-rgc-check-ganymede"
OUT_DIR="${JOB_DIR}/rgc_results"
PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
PROBE="${JOB_DIR}/clock_probe.py"
TARGET_GPU=2
HISTORICAL_CAP_MHZ=780
RECOVERY_THRESHOLD_MHZ=900
SAFETY_RESET_NEEDED=true

mkdir -p "$OUT_DIR"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$SAFETY_RESET_NEEDED" = true ]; then
    echo "safety_reset_on_error=true gpu=${TARGET_GPU}"
    sudo nvidia-smi -i "$TARGET_GPU" -rgc 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "time=$(date --iso-8601=seconds) host=$(hostname -s) job=${SLURM_JOB_ID:-unset}"
echo "allocated_gpus=${CUDA_VISIBLE_DEVICES:-unset} target_gpu=${TARGET_GPU} historical_cap_mhz=${HISTORICAL_CAP_MHZ}"
nvidia-smi --query-gpu=index,name,clocks.current.graphics,clocks.max.graphics --format=csv
nvidia-smi -i "$TARGET_GPU" -q -d CLOCK > "${OUT_DIR}/clock_state_before.txt" 2>&1 || true

CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$PYTHON_BIN" "$PROBE" \
  --smi-index "$TARGET_GPU" --seconds 8 --output "${OUT_DIR}/probe_before.json"
BEFORE_MAX=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["active_clock_max_mhz"]))' "${OUT_DIR}/probe_before.json")
echo "before_active_clock_max_mhz=${BEFORE_MAX}"

RESET_PERFORMED=false
if [ "$BEFORE_MAX" -le "$RECOVERY_THRESHOLD_MHZ" ]; then
  echo "clock_not_recovered=true action=manual_rgc"
  if ! sudo nvidia-smi -i "$TARGET_GPU" -rgc; then
    echo "manual_rgc_failed=true"
    exit 20
  fi
  RESET_PERFORMED=true
else
  echo "clock_already_recovered=true action=none"
fi
SAFETY_RESET_NEEDED=false

nvidia-smi -i "$TARGET_GPU" -q -d CLOCK > "${OUT_DIR}/clock_state_after.txt" 2>&1 || true
CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$PYTHON_BIN" "$PROBE" \
  --smi-index "$TARGET_GPU" --seconds 8 --output "${OUT_DIR}/probe_after.json"
AFTER_MAX=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["active_clock_max_mhz"]))' "${OUT_DIR}/probe_after.json")
echo "after_active_clock_max_mhz=${AFTER_MAX} reset_performed=${RESET_PERFORMED}"

if [ "$AFTER_MAX" -le "$RECOVERY_THRESHOLD_MHZ" ]; then
  echo "rgc_recovery_verified=false"
  exit 21
fi
echo "rgc_recovery_verified=true"
echo "job_exit_rc=0"
