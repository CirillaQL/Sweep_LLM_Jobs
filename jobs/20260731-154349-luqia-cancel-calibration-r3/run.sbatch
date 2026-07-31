#!/usr/bin/env bash
#SBATCH --job-name=cancel-cal-r3
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-154349-luqia-cancel-calibration-r3/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-154349-luqia-cancel-calibration-r3/slurm-%j.err

set -uo pipefail
TARGETS=(250632 250633)
scancel "${TARGETS[@]}"
RC=$?
echo "cancel_targets=${TARGETS[*]} scancel_rc=${RC}"
sleep 3
squeue -j 250632,250633 -o '%i|%j|%T|%N|%M|%l' || true
exit "$RC"
