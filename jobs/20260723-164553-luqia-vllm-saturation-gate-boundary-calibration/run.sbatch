#!/usr/bin/env bash
#SBATCH --job-name=vllm-saturation-gate-boundary
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-164553-luqia-vllm-saturation-gate-boundary-calibration/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-164553-luqia-vllm-saturation-gate-boundary-calibration/slurm-%j.err

set -uo pipefail

JOB_NAME="20260723-164553-luqia-vllm-saturation-gate-boundary-calibration"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
WORKLOAD_FILE="${JOB_DIR}/workloads.csv"

FINAL_RC=0
for placement in l40s-prefill-l4-decode auto; do
  echo
  echo "===== saturation_gate_placement_${placement} ====="
  env \
    JOB_DIR_OVERRIDE="${JOB_DIR}/${placement}" \
    WORKLOAD_FILE_OVERRIDE="${WORKLOAD_FILE}" \
    EXPECTED_WORKLOAD_COUNT_OVERRIDE=4 \
    ALLOW_ZERO_ADMIT_OVERRIDE=true \
    OVERLOAD_ACTION_OVERRIDE=reject \
    SCHEDULER_PLACEMENT_OVERRIDE="${placement}" \
    bash "${SOURCE_JOB}/run.sbatch"
  rc=$?
  echo "placement=${placement} rc=${rc}"
  if [ "$rc" -ne 0 ]; then
    FINAL_RC="$rc"
  fi
done

echo
echo "===== wrapper_result ====="
echo "gpu_tres_requested=false"
echo "placements_tested=2"
echo "wrapper_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
