#!/usr/bin/env bash
#SBATCH --job-name=xpyd-diagnose-255386
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260901-204125-luqia-xpyd-job255386-log-diagnosis/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260901-204125-luqia-xpyd-job255386-log-diagnosis/slurm-%j.err

set -euo pipefail

JOB_NAME="20260901-204125-luqia-xpyd-job255386-log-diagnosis"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
TARGET_JOB_NAME="20260901-190732-luqia-xpyd-feedback-frequency-table-2p2d-uranus-ganymede"
TARGET_JOB_ID="255386"
TARGET_JOB_DIR="${BROKER_ROOT}/jobs/${TARGET_JOB_NAME}"
TARGET_RAW_ROOT="/data/users/chjing/vllm_job_work/${TARGET_JOB_ID}/feedback_frequency_table_2p2d"
DIAG_JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${JOB_DIR}/results/${DIAG_JOB_ID}"

mkdir -p "${OUT_DIR}"

python3 "${JOB_DIR}/analyze_logs.py" \
  --target-job-id "${TARGET_JOB_ID}" \
  --target-job-dir "${TARGET_JOB_DIR}" \
  --target-raw-root "${TARGET_RAW_ROOT}" \
  --output-dir "${OUT_DIR}"

echo "xpyd_diagnosis_complete=true"
echo "xpyd_target_job_id=${TARGET_JOB_ID}"
echo "xpyd_diagnosis_output=${OUT_DIR}"
