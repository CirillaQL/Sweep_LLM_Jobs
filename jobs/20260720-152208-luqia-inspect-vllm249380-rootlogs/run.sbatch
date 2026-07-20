#!/usr/bin/env bash
#SBATCH --job-name=inspect-vllm-root
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"

echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "root=${ROOT}"

echo
echo "===== root_matching_files ====="
find "$ROOT" -maxdepth 1 -type f -name '*249380*' -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort || true

echo
echo "===== misplaced_result_files ====="
find "${ROOT}/vllm_smoke_results" -maxdepth 1 -type f -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort || true

echo
echo "===== slurm_stdout ====="
tail -n 320 "${ROOT}/slurm-249380.out" 2>&1 || true

echo
echo "===== slurm_stderr ====="
tail -n 320 "${ROOT}/slurm-249380.err" 2>&1 || true

echo
echo "===== vllm_server_log ====="
tail -n 420 "${ROOT}/vllm_smoke_results/vllm_server.log" 2>&1 || true

echo
echo "===== models_response ====="
cat "${ROOT}/vllm_smoke_results/models_response.json" 2>&1 || true

echo
echo "===== monitor ====="
head -n 12 "${ROOT}/vllm_smoke_results/nvidia_smi_monitor.csv" 2>&1 || true
tail -n 12 "${ROOT}/vllm_smoke_results/nvidia_smi_monitor.csv" 2>&1 || true
