#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-2p2d-random-r3
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-122711-luqia-vllm-pd-2p2d-random-bench-r3/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-122711-luqia-vllm-pd-2p2d-random-bench-r3/slurm-%j.err

set -uo pipefail

export JOB_NAME_OVERRIDE="20260808-122711-luqia-vllm-pd-2p2d-random-bench-r3"
exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
