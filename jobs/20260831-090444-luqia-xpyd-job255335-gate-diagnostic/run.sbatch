#!/usr/bin/env bash
#SBATCH --job-name=xpyd-255335-gatediag
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/data/users/chjing/vllm_job_work/gatediag-%j.out
#SBATCH --error=/data/users/chjing/vllm_job_work/gatediag-%j.err

set -euo pipefail

JOB_NAME="20260831-090444-luqia-xpyd-job255335-gate-diagnostic"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
RUN_DIR="/data/users/chjing/vllm_job_work/255335/binary_dvfs/xpyd_binary_dvfs_255335"
RESULT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env/bin/python}"

mkdir -p "${RESULT_DIR}"
"${PYTHON_BIN}" "${JOB_DIR}/collect_gate_diagnostics.py" \
  --run-dir "${RUN_DIR}" --output "${RESULT_DIR}"
test -s "${RESULT_DIR}/window_results.csv"
test -s "${RESULT_DIR}/window_gate_details.json"
echo "xpyd_gate_diagnostic_complete=true"
