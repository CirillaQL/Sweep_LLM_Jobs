#!/usr/bin/env bash
# Read-only Slurm and calibration-log diagnostic; intentionally avoids GPU nodes.
#SBATCH --job-name=calibration-state-check
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-205206-luqia-slurm-calibration-diagnostic"
WORK_ROOT="/data/users/chjing/vllm_job_work"
REPORT="${JOB_DIR}/diagnostic.txt"
TARGETS="250689,250690,250691,250692,250693,250694,250695"

{
  echo "diagnostic_time=$(date --iso-8601=seconds)"
  echo "diagnostic_host=$(hostname -s)"

  echo "=== squeue target jobs and array children ==="
  squeue --array -j "$TARGETS" \
    --format='%.22i|%.12A|%.8a|%.42j|%.10u|%.12T|%.12M|%.12l|%.24R|%.14N'

  echo "=== sacct target jobs and steps ==="
  sacct -X -S 2026-07-31 -j "$TARGETS" \
    --format=JobIDRaw,JobName%42,User,State%24,ExitCode,Elapsed,Start,End,NodeList%20,Reason%32 \
    --parsable2

  echo "=== allocations on ganymede and neptune ==="
  squeue --array --nodelist=ganymede,neptune \
    --format='%.22i|%.12A|%.8a|%.42j|%.10u|%.12T|%.12M|%.12l|%.24R|%.14N'

  echo "=== node state ==="
  sinfo -N -l -n ganymede,neptune

  echo "=== detailed current job records ==="
  for job_id in 250689 250690 250694 250695; do
    echo "--- scontrol job ${job_id} ---"
    scontrol show job -dd "$job_id" || true
  done

  echo "=== current calibration work directories ==="
  find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'calibration_250694_*' -o -name 'calibration_250695_*' \
       -o -name 'calibration_250689_*' -o -name 'calibration_250690_*' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort || true

  echo "=== relevant external Slurm logs ==="
  find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type f \
    \( -name 'calibration-slurm-l4-250694_*.out' \
       -o -name 'calibration-slurm-l4-250694_*.err' \
       -o -name 'calibration-slurm-l40s-250695_*.out' \
       -o -name 'calibration-slurm-l40s-250695_*.err' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort || true

  echo "=== upload/error markers from current external logs ==="
  grep -H -E 'campaign_node=|segment_start|observability_integrity|clickhouse_upload|artifact_cleanup|runner_rc=|wrapper_exit_rc=|Traceback|ERROR|error=' \
    "$WORK_ROOT"/calibration-slurm-l4-250694_*.out \
    "$WORK_ROOT"/calibration-slurm-l4-250694_*.err \
    "$WORK_ROOT"/calibration-slurm-l40s-250695_*.out \
    "$WORK_ROOT"/calibration-slurm-l40s-250695_*.err 2>/dev/null || true
} >"$REPORT" 2>&1

echo "diagnostic_report=${REPORT}"
