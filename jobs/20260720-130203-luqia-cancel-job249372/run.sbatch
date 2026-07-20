#!/usr/bin/env bash
#SBATCH --job-name=cancel-249372
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

TARGET_JOB_ID=249372

section() {
  echo
  echo "===== $1 ====="
}

section "cancel_job_info"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "target_job_id=${TARGET_JOB_ID}"

section "target_before_cancel"
squeue -j "$TARGET_JOB_ID" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.9l %.30R" || true
scontrol show job "$TARGET_JOB_ID" || true

section "scancel"
scancel "$TARGET_JOB_ID"
SCANCEL_RC=$?
echo "scancel_rc=${SCANCEL_RC}"

section "target_after_cancel"
sleep 3
squeue -j "$TARGET_JOB_ID" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.9l %.30R" || true
sacct -j "$TARGET_JOB_ID" --format=JobID,JobName,State,ExitCode,ElapsedRaw,NodeList -P || true

if [ "$SCANCEL_RC" -eq 0 ]; then
  echo "cancel_request=sent"
  exit 0
fi

echo "cancel_request=failed_or_target_already_gone"
exit "$SCANCEL_RC"
