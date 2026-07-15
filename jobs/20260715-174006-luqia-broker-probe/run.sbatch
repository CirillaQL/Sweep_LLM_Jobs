#!/usr/bin/env bash
#SBATCH --job-name=broker-probe
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "probe=broker-alive"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "uid=$(id -u)"
echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "slurm_job_user=${SLURM_JOB_USER:-unset}"
