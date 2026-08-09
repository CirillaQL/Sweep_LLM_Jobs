#!/usr/bin/env bash
#SBATCH --job-name=vllm-nixl-1p1d-tp1
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-104340-luqia-vllm-nixl-1p1d-tp1-connectivity/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-104340-luqia-vllm-nixl-1p1d-tp1-connectivity/slurm-%j.err

set -uo pipefail

export JOB_NAME_OVERRIDE="20260809-104340-luqia-vllm-nixl-1p1d-tp1-connectivity"
export PREFILL_REPLICAS_OVERRIDE=1
export PREFILL_TP_SIZES_OVERRIDE=1
export DECODE_REPLICAS_OVERRIDE=1
export DECODE_TP_SIZES_OVERRIDE=1
export NUM_PROMPTS_OVERRIDE=8
export REQUEST_RATE_OVERRIDE=2
export INPUT_LEN_OVERRIDE=512
export OUTPUT_LEN_OVERRIDE=64
export MAX_CONCURRENCY_OVERRIDE=4
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export STARTUP_TIMEOUT_SECONDS_OVERRIDE=900
export REQUIRE_ALL_RANDOM_ROUTES_OVERRIDE=true

exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
