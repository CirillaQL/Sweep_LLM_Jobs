#!/usr/bin/env bash
#SBATCH --job-name=vllm-context-boundary-fix
#SBATCH --partition=long
#SBATCH --time=00:28:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260726-101748-luqia-vllm-context-boundary-fix/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260726-101748-luqia-vllm-context-boundary-fix/slurm-%j.err

set -uo pipefail

JOB_NAME="20260726-101748-luqia-vllm-context-boundary-fix"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE="context_boundary_results"
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=3

export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_plus_saturation
export POLICY_LABEL_OVERRIDE=kv_aware_context_boundary
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export MAX_L4_FREQ_OVERRIDE=2040
export MAX_L40S_FREQ_OVERRIDE=2520

export KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=24.076
export KV_NUM_LAYERS_OVERRIDE=32
export KV_NUM_HEADS_OVERRIDE=8
export KV_HEAD_DIM_OVERRIDE=128
export KV_BYTES_PER_ELEMENT_OVERRIDE=2
export DISPATCH_MS_OVERRIDE=0

export CLOCK_ACK_MODE_OVERRIDE=monitor
export CLOCK_MISMATCH_ACTION_OVERRIDE=continue
export VLLM_BURSTINESS_SUPPORTED_OVERRIDE=true

export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=180
export MIN_PROMPTS_OVERRIDE=32
export MAX_PROMPTS_OVERRIDE=32

exec bash "${SOURCE_JOB}/run.sbatch"
