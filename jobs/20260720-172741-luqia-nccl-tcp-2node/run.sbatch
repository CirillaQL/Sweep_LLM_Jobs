#!/usr/bin/env bash
#SBATCH --job-name=nccl-tcp-ga-ne
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-172741-luqia-nccl-tcp-2node/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-172741-luqia-nccl-tcp-2node/slurm-%j.err

set -uo pipefail

JOB_DIR="${JOB_DIR_OVERRIDE:-/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-172741-luqia-nccl-tcp-2node}"
OUT_DIR="${JOB_DIR}/nccl_results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
TEST_SCRIPT="${TEST_SCRIPT_OVERRIDE:-${JOB_DIR}/nccl_test.py}"
MASTER_PORT=$((29000 + (${SLURM_JOB_ID:-0} % 1000)))

export HOME="/data/users/chjing"
export USER="chjing"
export LOGNAME="chjing"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN TEST_SCRIPT OUT_DIR MASTER_PORT

mkdir -p "$OUT_DIR"

section() {
  echo
  echo "===== $1 ====="
}

section "allocation"
echo "time=$(date --iso-8601=seconds)"
echo "batch_host=$(hostname -s)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>&1 || true
scontrol show job "${SLURM_JOB_ID:-}" 2>&1 || true

if [ ! -x "$PYTHON_BIN" ]; then
  echo "python_missing=${PYTHON_BIN}"
  exit 2
fi

MASTER_ADDR=$(ip -4 -o address show scope global 2>/dev/null | awk '$4 ~ /^10\.1\./ {sub(/\/.*/, "", $4); print $4; exit}')
if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "master_address_not_found=true"
  exit 3
fi
export MASTER_ADDR

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "test_script=${TEST_SCRIPT}"

section "per_node_preflight"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
  host=$(hostname -s)
  case "$host" in
    ganymede) iface=eno12399np0 ;;
    neptune) iface=enp160s0f0np0 ;;
    *) iface=$(ip route get "$MASTER_ADDR" 2>/dev/null | awk "NR==1 {for(i=1;i<=NF;i++) if(\$i==\"dev\") print \$(i+1)}") ;;
  esac
  echo "host=${host} rank=${SLURM_PROCID:-unset} localid=${SLURM_LOCALID:-unset} interface=${iface:-unset} addresses=$(hostname -I 2>/dev/null || true) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  ip -brief address show dev "$iface" 2>&1 || true
  ip route get "$MASTER_ADDR" 2>&1 || true
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || true
  "$PYTHON_BIN" -c "import socket, torch; print(f\"host={socket.gethostname()} torch={torch.__version__} cuda={torch.cuda.is_available()} nccl={torch.cuda.nccl.version() if torch.cuda.is_available() else None}\")"
' 2>&1
PREFLIGHT_RC=$?
echo "preflight_rc=${PREFLIGHT_RC}"
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  exit "$PREFLIGHT_RC"
fi

section "nccl_socket_all_reduce"
timeout --signal=TERM --kill-after=30s 360s \
  srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc '
    host=$(hostname -s)
    case "$host" in
      ganymede) iface=eno12399np0 ;;
      neptune) iface=enp160s0f0np0 ;;
      *) echo "unsupported_host=${host}"; exit 10 ;;
    esac

    export RANK="${SLURM_PROCID}"
    export LOCAL_RANK=0
    export WORLD_SIZE="${SLURM_NTASKS}"
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
    export NCCL_DEBUG_FILE="${OUT_DIR}/nccl-${host}-%p.log"
    export NCCL_SOCKET_IFNAME="${iface}"
    export GLOO_SOCKET_IFNAME="${iface}"
    export NCCL_SOCKET_FAMILY=AF_INET
    export NCCL_IB_DISABLE=1
    export NCCL_NET=Socket
    export TORCH_DISTRIBUTED_DEBUG=DETAIL

    echo "launch host=${host} rank=${RANK}/${WORLD_SIZE} iface=${iface} master=${MASTER_ADDR}:${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    exec "$PYTHON_BIN" -u "$TEST_SCRIPT"
  ' 2>&1 | tee "${OUT_DIR}/torchrun.txt"
NCCL_TEST_RC=${PIPESTATUS[0]}
echo "nccl_test_rc=${NCCL_TEST_RC}"

section "nccl_network_log_summary"
for file in "${OUT_DIR}"/nccl-*.log; do
  [ -f "$file" ] || continue
  echo "--- ${file} ---"
  grep -E 'Bootstrap|NET/Socket|Using network|Init COMPLETE|Connected all rings|Connected all trees|comm .* rank' "$file" 2>&1 || true
done

section "result"
if [ "$NCCL_TEST_RC" -eq 0 ]; then
  echo "nccl_cross_node_socket_ok=true"
else
  echo "nccl_cross_node_socket_ok=false"
fi
echo "job_exit_rc=${NCCL_TEST_RC}"
exit "$NCCL_TEST_RC"
