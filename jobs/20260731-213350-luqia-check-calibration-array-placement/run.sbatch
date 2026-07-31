#!/usr/bin/env bash
# Read-only Slurm audit for active calibration array children.
#SBATCH --job-name=check-cal-placement
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-213350-luqia-check-calibration-array-placement"
REPORT="${JOB_DIR}/diagnostic.txt"

{
  echo "checked_at=$(date --iso-8601=seconds)"
  squeue --array -j 250722,250723 \
    --format='%.22i|%.12A|%.8a|%.32j|%.12T|%.12M|%.24R|%.14N'
  for target in 250722_0 250723_0; do
    echo "target=${target}"
    scontrol show job -o "${target}" || true
  done
} >"${REPORT}" 2>&1

echo "diagnostic_report=${REPORT}"
