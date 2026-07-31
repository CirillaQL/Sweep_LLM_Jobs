#!/usr/bin/env bash
#SBATCH --job-name=vllm-cal-l4-ganymede-r2
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --array=0-11%1
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@1200
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153334-luqia-vllm-calibration-l4-ganymede-r2/slurm-%A_%a.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153334-luqia-vllm-calibration-l4-ganymede-r2/slurm-%A_%a.err

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
export JOB_NAME_OVERRIDE="20260731-153334-luqia-vllm-calibration-l4-ganymede-r2"
bash "${BROKER_ROOT}/jobs/20260731-152501-luqia-vllm-calibration-l4-ganymede-r1/run.sbatch"
