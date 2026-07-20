#!/usr/bin/env bash
#SBATCH --job-name=probe-vllm-env
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-152430-luqia-probe-chjing-vllm-env/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-152430-luqia-probe-chjing-vllm-env/slurm-%j.err

set -u

echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "HOME=${HOME:-unset}"
echo "USER=${USER:-unset}"
echo "LOGNAME=${LOGNAME:-unset}"
echo "PATH=${PATH:-unset}"
echo "python=$(command -v python 2>/dev/null || true)"
echo "python3=$(command -v python3 2>/dev/null || true)"
echo "vllm=$(command -v vllm 2>/dev/null || true)"

echo
echo "===== relevant_directories ====="
ls -ld /root /data/users/chjing /data/users/chjing/.cache /data/users/chjing/.cache/huggingface /data/users/chjing/miniforge3 2>&1 || true

echo
echo "===== python_candidates ====="
{
  command -v python 2>/dev/null || true
  command -v python3 2>/dev/null || true
  find /data/users/chjing/miniforge3 -maxdepth 3 -type f -path '*/bin/python' -print 2>/dev/null || true
} | awk '!seen[$0]++' | while IFS= read -r python_bin; do
  [ -n "$python_bin" ] || continue
  echo
  echo "----- ${python_bin} -----"
  "$python_bin" --version 2>&1 || true
  "$python_bin" - <<'PY' 2>&1 || true
import os
import sys

print(f"executable={sys.executable}")
print(f"home={os.environ.get('HOME')}")
try:
    import vllm
    print(f"vllm_import=ok version={vllm.__version__}")
except Exception as exc:
    print(f"vllm_import=failed type={type(exc).__name__} error={exc}")
try:
    import torch
    print(f"torch_import=ok version={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch_import=failed type={type(exc).__name__} error={exc}")
PY
done
