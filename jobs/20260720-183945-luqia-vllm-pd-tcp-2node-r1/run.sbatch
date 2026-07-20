#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-tcp-r1
#SBATCH --partition=long
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183945-luqia-vllm-pd-tcp-2node-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183945-luqia-vllm-pd-tcp-2node-r1/slurm-%j.err

set -uo pipefail

export JOB_DIR_OVERRIDE="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183945-luqia-vllm-pd-tcp-2node-r1"
exec bash "/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183247-luqia-vllm-pd-tcp-2node/run.sbatch"
