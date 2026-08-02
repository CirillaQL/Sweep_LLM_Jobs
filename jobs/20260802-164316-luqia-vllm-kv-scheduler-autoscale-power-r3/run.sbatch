#!/usr/bin/env bash
#SBATCH --job-name=vllm-kv-autoscale-power-r3
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260802-164316-luqia-vllm-kv-scheduler-autoscale-power-r3/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260802-164316-luqia-vllm-kv-scheduler-autoscale-power-r3/slurm-%j.err

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
export JOB_NAME_OVERRIDE="20260802-164316-luqia-vllm-kv-scheduler-autoscale-power-r3"
exec bash "${BROKER_ROOT}/jobs/20260802-151213-luqia-vllm-kv-scheduler-autoscale-power/run.sbatch"
