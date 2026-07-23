#!/usr/bin/env bash
#SBATCH --job-name=vllm-nogate-boundary-r1
#SBATCH --partition=long
#SBATCH --time=00:35:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-162136-luqia-vllm-nogate-boundary-calibration-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-162136-luqia-vllm-nogate-boundary-calibration-r1/slurm-%j.err

set -uo pipefail

JOB_NAME="20260723-162136-luqia-vllm-nogate-boundary-calibration-r1"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-114259-luqia-vllm-scheduler-multi-rgc-safe-r2"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/workloads.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=4
export BENCH_WINDOW_SECONDS_OVERRIDE=10
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export MIN_PROMPTS_OVERRIDE=12
export MAX_PROMPTS_OVERRIDE=500
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=900

exec bash "${SOURCE_JOB}/run.sbatch"
