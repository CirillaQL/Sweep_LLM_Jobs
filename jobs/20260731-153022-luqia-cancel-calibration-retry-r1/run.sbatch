#!/usr/bin/env bash
#SBATCH --job-name=cancel-cal-retry-r1
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153022-luqia-cancel-calibration-retry-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153022-luqia-cancel-calibration-retry-r1/slurm-%j.err

set -uo pipefail

TARGETS=(250598 250599)
echo "cancel_targets=${TARGETS[*]}"
squeue -j 250598,250599 -o '%i|%j|%T|%N|%M|%l' || true
scancel "${TARGETS[@]}"
RC=$?
echo "scancel_rc=${RC}"
sleep 3
squeue -j 250598,250599 -o '%i|%j|%T|%N|%M|%l' || true
exit "$RC"
