#!/usr/bin/env bash
#SBATCH --job-name=vllm-kv-scheduler-smoke
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260725-173829-luqia-vllm-kv-scheduler-smoke/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260725-173829-luqia-vllm-kv-scheduler-smoke/slurm-%j.err

set -uo pipefail

JOB_NAME="20260725-173829-luqia-vllm-kv-scheduler-smoke"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE="kv_scheduler_smoke_results"
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=14
export REQUIRE_ALL_WORKLOADS_EXECUTED_OVERRIDE=true
export JOB_TIME_LIMIT_LABEL_OVERRIDE=00:30:00

# Re-run the scheduler for every test window. The active policy checks both
# latency and saturation, then ranks safe candidates by predicted board power.
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_plus_saturation
export POLICY_LABEL_OVERRIDE=kv_aware_scheduler_smoke
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export SATURATION_ACTION_OVERRIDE=scale_out
export SCALE_IN_ENABLED_OVERRIDE=true
export MAX_PREFILL_INSTANCES_OVERRIDE=4
export MAX_DECODE_INSTANCES_OVERRIDE=4
export POWER_MONITOR_GPUS_PER_NODE_OVERRIDE=4
export VLLM_STEP_MEM_OVERRIDE=24G
export MAX_L4_FREQ_OVERRIDE=2040
export MAX_L40S_FREQ_OVERRIDE=2520

# Mistral-7B-v0.1 KV transfer model. 24.076 Gbit/s is the measured effective
# Neptune-Ganymede NCCL Socket/TCP bandwidth used by the existing KV experiment.
export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0

# Keep the smoke test bounded while retaining per-window scheduler decisions,
# 0.5-second GPU telemetry, event brackets, and integrated energy summaries.
export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export MIN_PROMPTS_OVERRIDE=8
export MAX_PROMPTS_OVERRIDE=500
export MAX_CONCURRENCY_OVERRIDE=128
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=continue

exec bash "${SOURCE_JOB}/run.sbatch"
