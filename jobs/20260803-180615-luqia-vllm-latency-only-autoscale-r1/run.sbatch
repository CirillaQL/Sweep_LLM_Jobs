#!/usr/bin/env bash
#SBATCH --job-name=vllm-latency-autoscale
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --gpus=12
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260803-180615-luqia-vllm-latency-only-autoscale-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260803-180615-luqia-vllm-latency-only-autoscale-r1/slurm-%j.err

set -uo pipefail

JOB_NAME="20260803-180615-luqia-vllm-latency-only-autoscale-r1"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE="latency_only_autoscale_results"
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=3
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:30:00

# The active scheduler and autoscale signal are latency/SLO only. The
# saturation-gated scheduler is not run and p_saturated never triggers scale.
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_only
export POLICY_LABEL_OVERRIDE=latency_only_autoscale
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export AUTOSCALE_ACTION_OVERRIDE=scale_out
export SCALE_IN_ENABLED_OVERRIDE=true
export MAX_PREFILL_INSTANCES_OVERRIDE=4
export MAX_DECODE_INSTANCES_OVERRIDE=8
export POWER_MONITOR_GANYMEDE_GPUS_OVERRIDE=8
export POWER_MONITOR_NEPTUNE_GPUS_OVERRIDE=4
export VLLM_STEP_MEM_OVERRIDE=24G
export MAX_L4_FREQ_OVERRIDE=2040
export MAX_L40S_FREQ_OVERRIDE=2520

export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0

export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=180
export WINDOW_WARMUP_SECONDS_OVERRIDE=30
# The explicit 30-second settling period replaces vLLM bench warmup requests.
export BENCH_NUM_WARMUPS_OVERRIDE=0
export MIN_PROMPTS_OVERRIDE=4
export MAX_PROMPTS_OVERRIDE=64
export MAX_CONCURRENCY_OVERRIDE=64
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=continue

exec bash "${SOURCE_JOB}/run.sbatch"
