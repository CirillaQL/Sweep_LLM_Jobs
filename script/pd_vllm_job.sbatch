#!/usr/bin/env bash
# Fixed allocation for Prefill_0/Prefill_1 on two Uranus L40S GPUs and
# Decode_0/Decode_1 on two Ganymede L4 GPUs. Every vLLM instance is TP=1.
#SBATCH --job-name=vllm-pd-custom-route
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/pd-vllm-%j.out
#SBATCH --error=/data/users/chjing/pd-vllm-%j.err

set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/data/users/chjing/Sweep_LLM_Jobs_broker/script}"
export PD_CONFIG_FILE="${PD_CONFIG_FILE:-${SCRIPT_ROOT}/pd_vllm.env.example}"
export PD_OUT_DIR="${PD_OUT_DIR:-/data/users/chjing/pd_vllm_results/${SLURM_JOB_ID}}"

"${SCRIPT_ROOT}/start_pd_vllm.sh"
