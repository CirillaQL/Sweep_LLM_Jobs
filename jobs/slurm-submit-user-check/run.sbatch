#!/usr/bin/env bash
#SBATCH --job-name=submit-user-check
#SBATCH --time=00:01:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

echo "Runtime user checks:"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "USER=${USER:-unset}"
echo "LOGNAME=${LOGNAME:-unset}"
echo "HOME=${HOME:-unset}"
echo "SLURM_JOB_USER=${SLURM_JOB_USER:-unset}"

echo "Git context:"
git rev-parse HEAD || true
git log -1 --format='commit=%H%ncommit_author=%an <%ae>%ncommit_date=%ad%nsubject=%s' || true

echo "Slurm job record:"
scontrol show job "$SLURM_JOB_ID" || true
