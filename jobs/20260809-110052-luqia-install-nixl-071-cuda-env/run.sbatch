#!/usr/bin/env bash
#SBATCH --job-name=install-nixl-071
#SBATCH --partition=long
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-110052-luqia-install-nixl-071-cuda-env/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-110052-luqia-install-nixl-071-cuda-env/slurm-%j.err

set -euo pipefail

JOB_NAME="20260809-110052-luqia-install-nixl-071-cuda-env"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID}"
WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
INSTALL_LOG="${OUT_DIR}/install_nixl.log"

export PIP_CACHE_DIR="${WORK_DIR}/pip-cache"
export XDG_CACHE_HOME="${WORK_DIR}/xdg-cache"
export TMPDIR="${WORK_DIR}/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

cleanup() {
  local cleanup_rc=$?
  trap - EXIT INT TERM
  if [ "$WORK_DIR" = "/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}" ]; then
    if rm -rf -- "$WORK_DIR"; then
      echo "cache_cleanup_complete=true path=${WORK_DIR}"
    else
      echo "cache_cleanup_complete=false removal_failed path=${WORK_DIR}" >&2
      cleanup_rc=3
    fi
  else
    echo "cache_cleanup_complete=false unsafe_path=${WORK_DIR}" >&2
    cleanup_rc=3
  fi
  exit "$cleanup_rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
exec > >(tee -a "$INSTALL_LOG") 2>&1

echo "phase=preflight hostname=$(hostname -s) job_id=${SLURM_JOB_ID}"
echo "environment=${ENV_DIR} python=${PYTHON_BIN} package=nixl[cu12]==0.7.1"
[ -x "$PYTHON_BIN" ]
[ -w "$ENV_DIR" ]
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version
"$PYTHON_BIN" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
"$PYTHON_BIN" -m pip show nixl nixl-cu12 || true

echo "phase=install"
"$PYTHON_BIN" -m pip install --no-input --only-binary=:all: \
  --upgrade-strategy only-if-needed 'nixl[cu12]==0.7.1'

echo "phase=local_import_check"
"$PYTHON_BIN" -c '
from importlib.metadata import version
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlXferTelemetry

print("nixl=" + version("nixl"))
print("nixl-cu12=" + version("nixl-cu12"))
print(f"nixl_agent={nixl_agent.__name__}")
print(f"nixl_agent_config={nixl_agent_config.__name__}")
print(f"nixl_telemetry={nixlXferTelemetry.__name__}")
'

echo "phase=two_node_import_check"
srun --overlap --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=2 --mem=4G --gres=none \
  "$PYTHON_BIN" -c '
import socket
from importlib.metadata import version
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlXferTelemetry

print(
    f"nixl_import_ok=true host={socket.gethostname()} "
    "nixl=" + version("nixl") + " "
    "nixl_cu12=" + version("nixl-cu12"),
    flush=True,
)
'

"$PYTHON_BIN" -m pip check || true
echo "install_nixl_complete=true"
