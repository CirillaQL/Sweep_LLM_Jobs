#!/usr/bin/env bash
#SBATCH --job-name=nixl-forwarding-ttft
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-223919-luqia-vllm-nixl-3p3d-forwarding-ttft/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-223919-luqia-vllm-nixl-3p3d-forwarding-ttft/slurm-%j.err

set -uo pipefail

export JOB_NAME_OVERRIDE="20260809-223919-luqia-vllm-nixl-3p3d-forwarding-ttft"
export PREFILL_REPLICAS_OVERRIDE=3
export PREFILL_TP_SIZES_OVERRIDE=1,1,2
export DECODE_REPLICAS_OVERRIDE=3
export DECODE_TP_SIZES_OVERRIDE=1,1,2
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE=true
export CUSTOM_PD_RANDOM_SEED=260809223919

export PD_ENABLE_PREDICTIVE_DVFS_OVERRIDE=true
export PD_DVFS_SLO_TTFT_MS_OVERRIDE=500
export PD_DVFS_SLO_TPOT_MS_OVERRIDE=200
export PD_DVFS_OVERLOAD_ACTION_OVERRIDE=reject
export PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export PD_DVFS_DISPATCH_MS_OVERRIDE=0
export PD_DVFS_CLOCK_TOLERANCE_MHZ_OVERRIDE=30
export PD_DVFS_SETTLE_SECONDS_OVERRIDE=20
export PD_DVFS_CLOCK_TIMEOUT_SECONDS_OVERRIDE=60

export NUM_PROMPTS_OVERRIDE=36
export REQUEST_RATE_OVERRIDE=2
export INPUT_LEN_OVERRIDE=512
export OUTPUT_LEN_OVERRIDE=128
export MAX_CONCURRENCY_OVERRIDE=1
export BENCH_TIMEOUT_SECONDS_OVERRIDE=1800
export STARTUP_TIMEOUT_SECONDS_OVERRIDE=1500
export REQUIRE_ALL_RANDOM_ROUTES_OVERRIDE=false

exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
