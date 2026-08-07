#!/usr/bin/env bash
#SBATCH --job-name=vllm-ganymede-tp4-single-pool-dvfs-r1
#SBATCH --partition=long
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260807-234548-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260807-234548-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs-r1/slurm-%j.err

set -uo pipefail

export JOB_NAME_OVERRIDE="20260807-234548-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs-r1"
exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260807-224100-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs/run.sbatch
