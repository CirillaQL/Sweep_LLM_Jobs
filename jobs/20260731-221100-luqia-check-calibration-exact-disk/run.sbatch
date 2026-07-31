#!/usr/bin/env bash
# Read-only exact disk audit for active calibration array task work directories.
#SBATCH --job-name=check-cal-exact-disk
#SBATCH --partition=short
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -uo pipefail

BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
WORK_ROOT="/data/users/chjing/vllm_job_work"
JOB_DIR="${BROKER_ROOT}/jobs/20260731-221100-luqia-check-calibration-exact-disk"
REPORT="${JOB_DIR}/diagnostic.txt"

{
  echo "checked_at=$(date --iso-8601=seconds)"
  echo "=== current active tasks ==="
  squeue --array -j 250722,250723 \
    --format='%.22i|%.12A|%.8K|%.36j|%.12T|%.12M|%.24R|%.14N'

  echo "=== filesystem capacity ==="
  df -h "${WORK_ROOT}"
  df -i "${WORK_ROOT}"

  for work_dir in \
    "${WORK_ROOT}/calibration_250724_0" \
    "${WORK_ROOT}/calibration_250725_0"; do
    echo "=== ${work_dir} ==="
    if [ ! -d "${work_dir}" ]; then
      echo "missing"
      continue
    fi
    du -sh "${work_dir}"
    du -sh "${work_dir}"/* 2>/dev/null | sort -h || true
    echo "segment_dir_count=$(find "${work_dir}/results/full_observability" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
    if [ -f "${work_dir}/results/run_results.jsonl" ]; then
      echo "run_result_rows=$(wc -l < "${work_dir}/results/run_results.jsonl")"
      echo "run_success_rows=$(grep -c '\"status\": \"success\"' "${work_dir}/results/run_results.jsonl" || true)"
      echo "run_failed_rows=$(grep -c '\"status\": \"failed\"' "${work_dir}/results/run_results.jsonl" || true)"
      echo "upload_succeeded_rows=$(grep -c '\"clickhouse_upload_status\": \"succeeded\"' "${work_dir}/results/run_results.jsonl" || true)"
      echo "frequency_unverified_rows=$(grep -c '\"frequency_verified\": false' "${work_dir}/results/run_results.jsonl" || true)"
      tail -n 5 "${work_dir}/results/run_results.jsonl"
    fi
    echo "=== retained segment sizes ==="
    find "${work_dir}/results/full_observability" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null \
      | xargs -0 -r du -sh | sort -h | tail -n 20 || true
  done
} >"${REPORT}" 2>&1

echo "diagnostic_report=${REPORT}"
