#!/usr/bin/env bash
#SBATCH --job-name=xpyd-255335-logdiag
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/data/users/chjing/vllm_job_work/logdiag-%j.out
#SBATCH --error=/data/users/chjing/vllm_job_work/logdiag-%j.err

set -euo pipefail

JOB_NAME="20260831-090103-luqia-xpyd-job255335-log-diagnostic"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
TARGET_CACHE="/data/users/chjing/vllm_job_work/255335"
DIAGNOSTIC_JOB_ID="${SLURM_JOB_ID:-manual}"
RESULT_DIR="${JOB_DIR}/results/${DIAGNOSTIC_JOB_ID}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env/bin/python}"

mkdir -p "${RESULT_DIR}"

if [ ! -d "${TARGET_CACHE}" ]; then
  printf '{"valid":false,"target_cache":"%s","error":"target_cache_missing"}\n' \
    "${TARGET_CACHE}" > "${RESULT_DIR}/diagnostic_summary.json"
  exit 2
fi

"${PYTHON_BIN}" "${JOB_DIR}/extract_failure_logs.py" \
  --target "${TARGET_CACHE}" \
  --output "${RESULT_DIR}"

test -s "${RESULT_DIR}/diagnostic_summary.json"
test -s "${RESULT_DIR}/file_inventory.tsv"
test -s "${RESULT_DIR}/log_tails.txt"

echo "xpyd_log_diagnostic_complete=true"
echo "xpyd_target_cache=${TARGET_CACHE}"
echo "xpyd_result_dir=${RESULT_DIR}"
