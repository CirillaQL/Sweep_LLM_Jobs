#!/usr/bin/env bash
# Read-only Slurm audit for the newly submitted calibration arrays.
#SBATCH --job-name=check-cal-arrays
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=diagnostic.out
#SBATCH --error=diagnostic.err

set -u

echo "checked_at=$(date --iso-8601=seconds)"
squeue --array -j 250722,250723 \
  -o 'job_id=%i array_job=%F array_task=%K state=%T node=%N reason=%R elapsed=%M'
for target in 250722_0 250723_0; do
  echo "target=${target}"
  scontrol show job -o "${target}" || true
done
