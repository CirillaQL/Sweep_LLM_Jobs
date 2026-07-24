#!/usr/bin/env bash
#SBATCH --job-name=vllm-gate-ablation-off
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260724-100922-luqia-vllm-gate-ablation-off/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260724-100922-luqia-vllm-gate-ablation-off/slurm-%j.err

set -uo pipefail

JOB_NAME="20260724-100922-luqia-vllm-gate-ablation-off"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=12
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=true
export ACTIVE_SCHEDULER_POLICY_OVERRIDE=latency_only
export POLICY_LABEL_OVERRIDE=gate_off_latency_only
export SCHEDULER_PLACEMENT_OVERRIDE=l40s-prefill-l4-decode
export OVERLOAD_ACTION_OVERRIDE=min-slo-violation
export MAX_L4_FREQ_OVERRIDE=780
export MAX_L40S_FREQ_OVERRIDE=2520
export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export MIN_PROMPTS_OVERRIDE=8
export MAX_PROMPTS_OVERRIDE=100
export CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE=90
export CLOCK_MISMATCH_ACTION_OVERRIDE=continue

exec bash "${SOURCE_JOB}/run.sbatch"
