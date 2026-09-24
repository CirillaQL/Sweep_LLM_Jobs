#!/usr/bin/env bash
#SBATCH --job-name=callisto-cpu-hello
#SBATCH --partition=long
#SBATCH --nodelist=callisto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-001443-luqia-callisto-cpu-hello/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-001443-luqia-callisto-cpu-hello/slurm-%j.err

# Minimal CPU-only smoke job: no GPU requested.
set -euo pipefail

echo "hello from gsync smoke job"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "user=$(whoami)"
echo "date=$(date -Iseconds)"
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
