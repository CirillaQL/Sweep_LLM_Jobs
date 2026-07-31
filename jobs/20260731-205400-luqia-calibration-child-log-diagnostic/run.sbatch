#!/usr/bin/env bash
# Read-only inspection of active calibration array-child logs and work dirs.
#SBATCH --job-name=calibration-child-log-check
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
WORK_ROOT="/data/users/chjing/vllm_job_work"
JOB_DIR="${BROKER_ROOT}/jobs/20260731-205400-luqia-calibration-child-log-diagnostic"
REPORT="${JOB_DIR}/diagnostic.txt"

{
  echo "diagnostic_time=$(date --iso-8601=seconds)"

  echo "=== active array children ==="
  squeue --array -j 250694,250695 \
    --format='%.22i|%.12A|%.8a|%.42j|%.10u|%.12T|%.12M|%.12l|%.24R|%.14N'

  echo "=== active child work directories ==="
  for path in "$WORK_ROOT/calibration_250696_0" "$WORK_ROOT/calibration_250697_0"; do
    echo "--- ${path} ---"
    if [ -d "$path" ]; then
      du -sh "$path"
      find "$path" -mindepth 1 -maxdepth 3 -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' \
        | sort | tail -n 160
    else
      echo "missing"
    fi
  done

  echo "=== broker-overridden child logs ==="
  for path in \
    "${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l4-ganymede/slurm-250696.out" \
    "${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l4-ganymede/slurm-250696.err" \
    "${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l40s-neptune/slurm-250697.out" \
    "${BROKER_ROOT}/jobs/20260731-174346-luqia-vllm-calibration-l40s-neptune/slurm-250697.err"; do
    echo "--- ${path} ---"
    if [ -f "$path" ]; then
      stat -c 'size=%s modified=%y' "$path"
      tail -n 240 "$path"
    else
      echo "missing"
    fi
  done

  echo "=== upload and failure markers across active child work dirs ==="
  grep -R -H -E 'segment_start|observability_integrity|clickhouse_upload|artifact_cleanup|runner_rc=|wrapper_exit_rc=|Traceback|ERROR|error=' \
    "$WORK_ROOT/calibration_250696_0" "$WORK_ROOT/calibration_250697_0" 2>/dev/null \
    | tail -n 500 || true
} >"$REPORT" 2>&1

echo "diagnostic_report=${REPORT}"
