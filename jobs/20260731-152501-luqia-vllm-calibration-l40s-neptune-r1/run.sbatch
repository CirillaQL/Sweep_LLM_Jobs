#!/usr/bin/env bash
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-152501-luqia-vllm-calibration-l40s-neptune-r1/slurm-%A_%a.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260731-152501-luqia-vllm-calibration-l40s-neptune-r1/slurm-%A_%a.err

set -uo pipefail

JOB_NAME="${JOB_NAME_OVERRIDE:-20260731-152501-luqia-vllm-calibration-l40s-neptune-r1}"
COMMON_NAME="20260731-144337-luqia-vllm-calibration-common"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
COMMON_DIR="${BROKER_ROOT}/jobs/${COMMON_NAME}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
OUT_DIR="${JOB_DIR}/calibration_results/shard_${TASK_ID}"
WORK_ROOT="/data/users/chjing/vllm_job_work"
WORK_DIR="${WORK_ROOT}/${JOB_ID}_${TASK_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"

reset_all_clocks() {
  local index
  for index in 0 1 2 3; do
    sudo -n nvidia-smi -i "$index" -rgc >/dev/null 2>&1 || true
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  reset_all_clocks
  cd "$JOB_DIR" 2>/dev/null || cd /tmp || true
  if [[ "$WORK_DIR" == "${WORK_ROOT}/${JOB_ID}_${TASK_ID}" ]] \
    && [[ "$JOB_ID" =~ ^[0-9]+$ ]] && [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    rm -rf -- "$WORK_DIR"
  else
    echo "unsafe_work_cleanup_skipped=${WORK_DIR}"
    rc=3
  fi
  echo "wrapper_exit_rc=${rc}"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p \
  "$OUT_DIR" \
  "$WORK_DIR/xdg-cache" \
  "$WORK_DIR/xdg-config" \
  "$WORK_DIR/flashinfer" \
  "$WORK_DIR/torch" \
  "$WORK_DIR/tmp"
cd "$WORK_DIR" || exit 3
export HF_HOME="/data/users/chjing/.cache/huggingface"
export XDG_CACHE_HOME="$WORK_DIR/xdg-cache"
export XDG_CONFIG_HOME="$WORK_DIR/xdg-config"
export FLASHINFER_WORKSPACE_BASE="$WORK_DIR/flashinfer"
export TORCH_HOME="$WORK_DIR/torch"
export TMPDIR="$WORK_DIR/tmp"
export PATH="${ENV_DIR}/bin:${PATH}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
set -a
source "${COMMON_DIR}/clickhouse_credentials.env"
set +a

ACTUAL_HOST="$(hostname -s)"
if [ "$ACTUAL_HOST" != "neptune" ]; then
  echo "wrong_node expected=neptune actual=${ACTUAL_HOST}" >&2
  exit 4
fi
echo "campaign_node=${ACTUAL_HOST} gpu_type=l40s shard=${TASK_ID} slurm_job_id=${JOB_ID}"
nvidia-smi -L

"${ENV_DIR}/bin/python" "${COMMON_DIR}/calibration_runner.py" \
  --manifest "${COMMON_DIR}/l40s_manifest.json" \
  --schema "${COMMON_DIR}/calibration_schema.sql" \
  --shard-id "$TASK_ID" \
  --output-dir "$OUT_DIR" \
  --vllm-bin "${ENV_DIR}/bin/vllm"
RUN_RC=$?
echo "runner_rc=${RUN_RC}"
exit "$RUN_RC"
