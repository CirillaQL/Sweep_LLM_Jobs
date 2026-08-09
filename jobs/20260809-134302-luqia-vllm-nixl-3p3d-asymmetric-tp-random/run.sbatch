#!/usr/bin/env bash
#SBATCH --job-name=vllm-nixl-3p3d-asym-tp
#SBATCH --partition=long
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-134302-luqia-vllm-nixl-3p3d-asymmetric-tp-random/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-134302-luqia-vllm-nixl-3p3d-asymmetric-tp-random/slurm-%j.err

set -uo pipefail

export JOB_NAME_OVERRIDE="20260809-134302-luqia-vllm-nixl-3p3d-asymmetric-tp-random"
export PREFILL_REPLICAS_OVERRIDE=3
export PREFILL_TP_SIZES_OVERRIDE=1,1,2
export DECODE_REPLICAS_OVERRIDE=3
export DECODE_TP_SIZES_OVERRIDE=1,1,2
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE=true
export CUSTOM_PD_RANDOM_SEED=252594
export NUM_PROMPTS_OVERRIDE=96
export REQUEST_RATE_OVERRIDE=6
export INPUT_LEN_OVERRIDE=256
export OUTPUT_LEN_OVERRIDE=32
export MAX_CONCURRENCY_OVERRIDE=24
export BENCH_TIMEOUT_SECONDS_OVERRIDE=600
export STARTUP_TIMEOUT_SECONDS_OVERRIDE=1500
export REQUIRE_ALL_RANDOM_ROUTES_OVERRIDE=true

exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
