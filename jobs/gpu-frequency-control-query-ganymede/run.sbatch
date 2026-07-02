#!/usr/bin/env bash
#SBATCH --job-name=gpu-freq-query-ga
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

GPU_ID="${GPU_ID:-${SLURM_JOB_GPUS:-0}}"
GPU_ID="${GPU_ID%%,*}"

run_section() {
  local title="$1"
  shift
  echo
  echo "===== ${title} ====="
  "$@"
  local rc=$?
  echo "===== ${title} rc=${rc} ====="
}

run_shell_section() {
  local title="$1"
  local command="$2"
  echo
  echo "===== ${title} ====="
  echo "command=${command}"
  bash -lc "$command"
  local rc=$?
  echo "===== ${title} rc=${rc} ====="
}

echo "target_node=ganymede"
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_GPU_FREQ=${SLURM_GPU_FREQ:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_ID=${GPU_ID}"
echo "whoami=$(whoami)"
echo "id=$(id)"

run_shell_section "slurm_gpu_related_config" \
  "scontrol show config | grep -Ei 'gpu|gres|cgroup|taskplugin|selecttype|prolog|epilog|launchparameters|slurmdparameters|debugflags|acctgather' || true"

run_section "slurm_job_record" scontrol show job "$SLURM_JOB_ID"
run_section "slurm_node_record" scontrol show node "$(hostname -s)"

run_shell_section "sbatch_gpu_freq_help" \
  "sbatch --help 2>&1 | grep -A4 -B2 -- '--gpu-freq' || true"

run_shell_section "srun_gpu_freq_help" \
  "srun --help 2>&1 | grep -A4 -B2 -- '--gpu-freq' || true"

if command -v nvidia-smi >/dev/null 2>&1; then
  run_section "nvidia_smi_list" nvidia-smi -L
  run_section "nvidia_smi_clock_query" nvidia-smi -i "$GPU_ID" --query-gpu=index,name,persistence_mode,clocks.gr,clocks.mem,clocks.max.graphics,clocks.max.memory,power.management,power.limit,power.draw --format=csv
  run_section "nvidia_smi_supported_clocks" nvidia-smi -i "$GPU_ID" -q -d SUPPORTED_CLOCKS
  run_section "nvidia_smi_clock_details" nvidia-smi -i "$GPU_ID" -q -d CLOCK
  run_section "nvidia_smi_power_details" nvidia-smi -i "$GPU_ID" -q -d POWER
else
  echo "nvidia-smi not found."
fi
