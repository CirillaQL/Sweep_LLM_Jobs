#!/usr/bin/env bash
#SBATCH --job-name=inspect-vllm249380
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

TARGET="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-151015-luqia-vllm-smoke-5min-neptune"

echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "target=${TARGET}"

echo
echo "===== target_files ====="
find "$TARGET" -maxdepth 2 -type f -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort || true

echo
echo "===== slurm_stdout ====="
tail -n 260 "${TARGET}/slurm-249380.out" 2>&1 || true

echo
echo "===== slurm_stderr ====="
tail -n 260 "${TARGET}/slurm-249380.err" 2>&1 || true

echo
echo "===== vllm_server_log ====="
tail -n 320 "${TARGET}/vllm_smoke_results/vllm_server.log" 2>&1 || true

echo
echo "===== models_response ====="
cat "${TARGET}/vllm_smoke_results/models_response.json" 2>&1 || true

echo
echo "===== monitor_head_tail ====="
head -n 12 "${TARGET}/vllm_smoke_results/nvidia_smi_monitor.csv" 2>&1 || true
tail -n 12 "${TARGET}/vllm_smoke_results/nvidia_smi_monitor.csv" 2>&1 || true
