#!/usr/bin/env bash
# Cancel both broken calibration arrays and record the post-cancel state.
#SBATCH --job-name=cancel-broken-calibration
#SBATCH --partition=short
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-205812-luqia-cancel-broken-calibration"
REPORT="${JOB_DIR}/cancel_report.txt"

{
  echo "cancel_time=$(date --iso-8601=seconds)"
  echo "cancel_targets=250694,250695"
  scancel 250694 250695
  cancel_rc=$?
  echo "scancel_rc=${cancel_rc}"
  sleep 3
  echo "=== remaining target jobs ==="
  squeue --array -j 250694,250695 \
    --format='%.22i|%.12A|%.8a|%.42j|%.10u|%.12T|%.12M|%.24R|%.14N' || true
  echo "=== target-node allocations ==="
  squeue --array --nodelist=ganymede,neptune \
    --format='%.22i|%.12A|%.8a|%.42j|%.10u|%.12T|%.12M|%.24R|%.14N' || true
  exit "$cancel_rc"
} >"$REPORT" 2>&1
