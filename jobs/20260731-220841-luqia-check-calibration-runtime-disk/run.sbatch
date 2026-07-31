#!/usr/bin/env bash
# Read-only runtime, log, and disk audit for calibration arrays 250722/250723.
#SBATCH --job-name=check-cal-runtime-disk
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
JOB_DIR="${BROKER_ROOT}/jobs/20260731-220841-luqia-check-calibration-runtime-disk"
REPORT="${JOB_DIR}/diagnostic.txt"

{
  echo "checked_at=$(date --iso-8601=seconds)"

  echo "=== active array tasks ==="
  squeue --array -j 250722,250723 \
    --format='%.22i|%.12A|%.8a|%.36j|%.12T|%.12M|%.24R|%.14N'

  echo "=== accounting ==="
  sacct -j 250722,250723 --array \
    --format='JobIDRaw,JobID,JobName%36,State,ExitCode,Elapsed,NodeList,AllocTRES%80' \
    --starttime 2026-07-31T21:25:00 || true

  echo "=== work filesystem ==="
  df -h "${WORK_ROOT}"
  df -i "${WORK_ROOT}"

  echo "=== active work directories and logs ==="
  while IFS='|' read -r logical_id actual_id array_task state node elapsed; do
    [ "${state}" = "RUNNING" ] || continue
    work_dir="${WORK_ROOT}/calibration_${actual_id}_${array_task}"
    echo "--- logical=${logical_id} actual=${actual_id} task=${array_task} node=${node} elapsed=${elapsed} ---"
    if [ -d "${work_dir}" ]; then
      du -sh "${work_dir}"
      du -sh "${work_dir}"/* 2>/dev/null | sort -h || true
      find "${work_dir}/results" -maxdepth 2 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' \
        2>/dev/null | sort -k2 | tail -n 30 || true
    else
      echo "work_dir_missing=${work_dir}"
    fi

    case "${logical_id}" in
      250722_*)
        source_job_dir="${BROKER_ROOT}/jobs/20260731-212935-luqia-vllm-calibration-l4-ganymede-r1"
        ;;
      *)
        source_job_dir="${BROKER_ROOT}/jobs/20260731-212935-luqia-vllm-calibration-l40s-neptune-r1"
        ;;
    esac
    for log_path in "${source_job_dir}/slurm-${actual_id}.out" "${source_job_dir}/slurm-${actual_id}.err"; do
      echo "log=${log_path}"
      if [ -f "${log_path}" ]; then
        stat -c 'size=%s modified=%y' "${log_path}"
        grep -E 'wrapper_start|runner_start|segment_start|gpu_monitor_ready|observability_integrity|clickhouse_upload|segment_finish|artifact_cleanup|shard_fatal|runner_rc=|wrapper_exit_rc=|Traceback|ERROR|error=' \
          "${log_path}" | tail -n 80 || true
      else
        echo "missing"
      fi
    done
  done < <(
    squeue --array -h -j 250722,250723 -t RUNNING -o '%i|%A|%a|%T|%N|%M'
  )
} >"${REPORT}" 2>&1

echo "diagnostic_report=${REPORT}"
