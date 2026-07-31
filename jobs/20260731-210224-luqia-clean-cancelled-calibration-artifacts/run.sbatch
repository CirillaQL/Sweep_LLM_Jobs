#!/usr/bin/env bash
# Remove only raw logs/caches belonging to cancelled calibration children.
#SBATCH --job-name=clean-cancelled-calibration
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
WORK_ROOT="/data/users/chjing/vllm_job_work"
JOB_DIR="${BROKER_ROOT}/jobs/20260731-210224-luqia-clean-cancelled-calibration-artifacts"
REPORT="${JOB_DIR}/cleanup_report.txt"

L4_JOB_DIR="${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l4-ganymede"
L40S_JOB_DIR="${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l40s-neptune"
L4_WORK_DIR="${WORK_ROOT}/calibration_250696_0"
L40S_WORK_DIR="${WORK_ROOT}/calibration_250697_0"

LOG_FILES=(
  "${L4_JOB_DIR}/slurm-250694.out"
  "${L4_JOB_DIR}/slurm-250694.err"
  "${L4_JOB_DIR}/slurm-250696.out"
  "${L4_JOB_DIR}/slurm-250696.err"
  "${L40S_JOB_DIR}/slurm-250695.out"
  "${L40S_JOB_DIR}/slurm-250695.err"
  "${L40S_JOB_DIR}/slurm-250697.out"
  "${L40S_JOB_DIR}/slurm-250697.err"
  "${WORK_ROOT}/calibration-slurm-l4-250694_0.out"
  "${WORK_ROOT}/calibration-slurm-l4-250694_0.err"
  "${WORK_ROOT}/calibration-slurm-l40s-250695_0.out"
  "${WORK_ROOT}/calibration-slurm-l40s-250695_0.err"
)

describe_path() {
  local path="$1"
  if [ -d "$path" ]; then
    du -sb -- "$path"
  elif [ -f "$path" ]; then
    stat -c '%s %n' -- "$path"
  else
    echo "missing ${path}"
  fi
}

{
  echo "cleanup_time=$(date --iso-8601=seconds)"
  echo "=== before ==="
  describe_path "$L4_WORK_DIR"
  describe_path "$L40S_WORK_DIR"
  for path in "${LOG_FILES[@]}"; do
    describe_path "$path"
  done

  # These paths are fixed constants for the two cancelled array children.
  if [ "$L4_WORK_DIR" = "${WORK_ROOT}/calibration_250696_0" ]; then
    rm -rf -- "$L4_WORK_DIR"
  else
    echo "unsafe_l4_work_path=${L4_WORK_DIR}"
    exit 3
  fi
  if [ "$L40S_WORK_DIR" = "${WORK_ROOT}/calibration_250697_0" ]; then
    rm -rf -- "$L40S_WORK_DIR"
  else
    echo "unsafe_l40s_work_path=${L40S_WORK_DIR}"
    exit 3
  fi
  for path in "${LOG_FILES[@]}"; do
    rm -f -- "$path"
  done

  echo "=== after ==="
  describe_path "$L4_WORK_DIR"
  describe_path "$L40S_WORK_DIR"
  remaining=0
  for path in "${LOG_FILES[@]}"; do
    describe_path "$path"
    if [ -e "$path" ]; then
      remaining=$((remaining + 1))
    fi
  done
  echo "remaining_targets=${remaining}"
  test "$remaining" -eq 0
  test ! -e "$L4_WORK_DIR"
  test ! -e "$L40S_WORK_DIR"
} >"$REPORT" 2>&1
