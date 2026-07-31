#!/usr/bin/env bash
# Cancel only the active full-grid calibration arrays and verify queue removal.
#SBATCH --job-name=cancel-cal-arrays
#SBATCH --partition=short
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/20260731-223239-luqia-cancel-calibration-arrays"
REPORT="${JOB_DIR}/cancel_report.txt"

{
  echo "cancel_requested_at=$(date --iso-8601=seconds)"
  echo "cancel_targets=250722,250723"
  echo "=== before ==="
  squeue --array -j 250722,250723 \
    --format='%.22i|%.12A|%.8K|%.36j|%.12T|%.12M|%.24R|%.14N' || true

  scancel 250722 250723
  sleep 10

  echo "=== after ==="
  squeue --array -j 250722,250723 \
    --format='%.22i|%.12A|%.8K|%.36j|%.12T|%.12M|%.24R|%.14N' || true
  echo "cancel_verified_at=$(date --iso-8601=seconds)"
} >"${REPORT}" 2>&1

echo "cancel_report=${REPORT}"
