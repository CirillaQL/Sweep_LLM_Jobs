#!/usr/bin/env bash
#SBATCH --job-name=vllm-fixed-auto-dvfs
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-234005-luqia-vllm-fixed-auto-dvfs/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-234005-luqia-vllm-fixed-auto-dvfs/slurm-%j.err

set -uo pipefail

JOB_NAME="20260723-234005-luqia-vllm-fixed-auto-dvfs"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
REFERENCE_JOB="${BROKER_ROOT}/jobs/20260723-231809-luqia-vllm-online-nonuniform-dvfs-r2"

export JOB_DIR_OVERRIDE="${BROKER_ROOT}/jobs/${JOB_NAME}"
export OUT_DIR_NAME_OVERRIDE=auto_dvfs_results
export WORKLOAD_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/request_trace.csv"
export EXPECTED_WORKLOAD_COUNT_OVERRIDE=12
export ONLINE_SCHEDULING_OVERRIDE=true
export SCHEDULER_ENABLED_OVERRIDE=false
export FIXED_PLACEMENT_FILE_OVERRIDE="${JOB_DIR_OVERRIDE}/fixed_placement.json"
export GPU_CLOCK_CONTROL_MODE_OVERRIDE=auto
export POLICY_LABEL_OVERRIDE=fixed_l40s_prefill_l4_decode_auto_dvfs
export RUN_REFERENCE_COMPARISON_OVERRIDE=false
export BENCH_TIMEOUT_SECONDS_OVERRIDE=300
export MIN_PROMPTS_OVERRIDE=8
export MAX_PROMPTS_OVERRIDE=100

bash "${SOURCE_JOB}/run.sbatch"
RUN_RC=$?
if [ "$RUN_RC" -eq 0 ]; then
  /data/users/chjing/miniforge3/envs/cuda-env/bin/python \
    "${SOURCE_JOB}/compare_policy_energy.py" \
    --scheduled "${REFERENCE_JOB}/saturation_gate_results/live_summary.csv" \
    --auto "${JOB_DIR_OVERRIDE}/auto_dvfs_results/live_summary.csv" \
    --output-csv "${JOB_DIR_OVERRIDE}/auto_dvfs_results/energy_comparison.csv" \
    --output-json "${JOB_DIR_OVERRIDE}/auto_dvfs_results/energy_comparison.json"
  COMPARE_RC=$?
  if [ "$COMPARE_RC" -ne 0 ]; then
    RUN_RC="$COMPARE_RC"
  fi
fi
echo "wrapper_exit_rc=${RUN_RC}"
exit "$RUN_RC"
