#!/usr/bin/env bash
# Copy this file to a new broker job directory as run.sbatch and add READY.
#SBATCH --job-name=vllm-cal-l40s-neptune
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --array=0-6%1
#SBATCH --nodes=1
#SBATCH --nodelist=neptune
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@1200
#SBATCH --output=/data/users/chjing/vllm_job_work/calibration-slurm-l40s-%A_%a.out
#SBATCH --error=/data/users/chjing/vllm_job_work/calibration-slurm-l40s-%A_%a.err

exec /usr/bin/env bash \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-144337-luqia-vllm-calibration-common/run_calibration_task.sh l40s
