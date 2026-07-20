#!/usr/bin/env bash
#SBATCH --job-name=find-vllm-env
#SBATCH --time=00:04:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-152614-luqia-find-vllm-conda-env/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-152614-luqia-find-vllm-conda-env/slurm-%j.err

set -u

echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"

echo
echo "===== conda_env_directories ====="
find /data/users/chjing/miniforge3/envs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true

echo
echo "===== python_candidates ====="
for python_bin in \
  /data/users/chjing/miniforge3/bin/python \
  /data/users/chjing/miniforge3/envs/*/bin/python; do
  [ -x "$python_bin" ] || continue
  echo
  echo "----- ${python_bin} -----"
  "$python_bin" --version 2>&1 || true
  HOME=/data/users/chjing USER=chjing LOGNAME=chjing "$python_bin" - <<'PY' 2>&1 || true
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
