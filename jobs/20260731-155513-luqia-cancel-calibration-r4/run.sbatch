#!/usr/bin/env bash
#SBATCH --job-name=cancel-cal-r4
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-155513-luqia-cancel-calibration-r4/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-155513-luqia-cancel-calibration-r4/slurm-%j.err

set -uo pipefail
TARGETS=(250643 250644)
scancel "${TARGETS[@]}"
RC=$?
echo "cancel_targets=${TARGETS[*]} scancel_rc=${RC}"
sleep 3
squeue -j 250643,250644 -o '%i|%j|%T|%N|%M|%l' || true
exit "$RC"
