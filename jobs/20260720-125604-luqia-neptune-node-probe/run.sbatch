#!/usr/bin/env bash
#SBATCH --job-name=neptune-probe
#SBATCH --partition=long
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=neptune
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "hostname_fqdn=$(hostname --fqdn 2>/dev/null || hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "pwd=$(pwd)"
echo "kernel=$(uname -srmo)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "===== node_info ====="
lscpu | sed -n '1,24p'

echo
echo "===== memory_info ====="
free -h

echo
echo "probe_completed_at=$(date --iso-8601=seconds)"
