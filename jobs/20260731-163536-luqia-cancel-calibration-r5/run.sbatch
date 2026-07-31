#!/usr/bin/env bash
#SBATCH --job-name=cancel-cal-r5
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-163536-luqia-cancel-calibration-r5/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-163536-luqia-cancel-calibration-r5/slurm-%j.err

set -uo pipefail
TARGETS=(250672 250673)
scancel "${TARGETS[@]}"
RC=$?
echo "cancel_targets=${TARGETS[*]} scancel_rc=${RC}"
sleep 3
squeue -j 250672,250673 -o '%i|%j|%T|%N|%M|%l' || true
exit "$RC"
