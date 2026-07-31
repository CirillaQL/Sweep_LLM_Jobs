#!/usr/bin/env bash
#SBATCH --job-name=cancel-cal-r2
#SBATCH --partition=short
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153756-luqia-cancel-calibration-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-153756-luqia-cancel-calibration-r2/slurm-%j.err

set -uo pipefail
TARGETS=(250615 250616)
scancel "${TARGETS[@]}"
RC=$?
echo "cancel_targets=${TARGETS[*]} scancel_rc=${RC}"
sleep 3
squeue -j 250615,250616 -o '%i|%j|%T|%N|%M|%l' || true
exit "$RC"
