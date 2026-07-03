#!/usr/bin/env bash
#SBATCH --job-name=cancel-245458
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

TARGET_JOB_ID="245458"

section() {
  echo
  echo "===== $1 ====="
}

section "job_info"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "TARGET_JOB_ID=${TARGET_JOB_ID}"

section "target_before_cancel"
squeue -j "$TARGET_JOB_ID" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.9l %.20R" || true
scontrol show job "$TARGET_JOB_ID" || true

section "scancel"
scancel "$TARGET_JOB_ID"
scancel_rc=$?
echo "scancel_rc=${scancel_rc}"

section "target_after_cancel"
sleep 3
squeue -j "$TARGET_JOB_ID" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.9l %.20R" || true
sacct -j "$TARGET_JOB_ID" --format=JobID,JobName,State,ExitCode,ElapsedRaw,NodeList -P || true

if [ "$scancel_rc" -eq 0 ]; then
  echo "cancel_request=sent"
else
  echo "cancel_request=failed_or_already_gone"
fi
