#!/usr/bin/env bash
#SBATCH --job-name=nccl-tcp-ga-ne-r1
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-173150-luqia-nccl-tcp-2node-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-173150-luqia-nccl-tcp-2node-r1/slurm-%j.err

export JOB_DIR_OVERRIDE="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-173150-luqia-nccl-tcp-2node-r1"
export TEST_SCRIPT_OVERRIDE="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-172741-luqia-nccl-tcp-2node/nccl_test.py"

source /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-172741-luqia-nccl-tcp-2node/run.sbatch
