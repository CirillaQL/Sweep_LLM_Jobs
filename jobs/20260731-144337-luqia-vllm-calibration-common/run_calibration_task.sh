#!/usr/bin/env bash
# Run one Slurm array element with all generated artifacts outside the Git tree.

set -uo pipefail

GPU_TYPE="${1:?usage: run_calibration_task.sh l4|l40s}"
case "$GPU_TYPE" in
  l4)
    EXPECTED_HOST="ganymede"
    GPU_COUNT=8
    MANIFEST_NAME="l4_manifest.json"
    ;;
  l40s)
    EXPECTED_HOST="neptune"
    GPU_COUNT=4
    MANIFEST_NAME="l40s_manifest.json"
    ;;
  *)
    echo "unsupported_gpu_type=${GPU_TYPE}" >&2
    exit 2
    ;;
esac

COMMON_NAME="20260731-144337-luqia-vllm-calibration-common"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
COMMON_DIR="${BROKER_ROOT}/jobs/${COMMON_NAME}"
OBSERVABILITY_DIR="${BROKER_ROOT}/jobs/20260731-105244-luqia-vllm-pd-observability-fixes-r1"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
WORK_ROOT="/data/users/chjing/vllm_job_work"
WORK_DIR="${WORK_ROOT}/calibration_${JOB_ID}_${TASK_ID}"
OUT_DIR="${WORK_DIR}/results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"

reset_all_clocks() {
  local index
  for ((index = 0; index < GPU_COUNT; index++)); do
    sudo -n nvidia-smi -i "$index" -rgc >/dev/null 2>&1 || true
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  reset_all_clocks
  cd /data/users/chjing 2>/dev/null || cd /tmp || true
  if [[ "$WORK_DIR" == "${WORK_ROOT}/calibration_${JOB_ID}_${TASK_ID}" ]] \
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

mkdir -p "$WORK_ROOT"

# Every new array task removes only stale calibration-owned work directories
# and Slurm logs. A cluster job is limited to one day, so two days cannot match
# a live task. This is the periodic safety net; successful segment data is
# deleted immediately by calibration_runner.py.
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type d \
  -name 'calibration_[0-9]*_[0-9]*' -mtime +2 -exec rm -rf -- {} +
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type f \
  -name 'calibration-slurm-*.out' -mtime +2 -delete
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type f \
  -name 'calibration-slurm-*.err' -mtime +2 -delete

mkdir -p \
  "$OUT_DIR" \
  "$WORK_DIR/xdg-cache" \
  "$WORK_DIR/xdg-config" \
  "$WORK_DIR/flashinfer" \
  "$WORK_DIR/cuda-cache" \
  "$WORK_DIR/triton-cache" \
  "$WORK_DIR/torchinductor-cache" \
  "$WORK_DIR/torch" \
  "$WORK_DIR/vllm-cache" \
  "$WORK_DIR/numba-cache" \
  "$WORK_DIR/tmp"
cd "$WORK_DIR" || exit 3

export HF_HOME="/data/users/chjing/.cache/huggingface"
export XDG_CACHE_HOME="$WORK_DIR/xdg-cache"
export XDG_CONFIG_HOME="$WORK_DIR/xdg-config"
export FLASHINFER_WORKSPACE_BASE="$WORK_DIR/flashinfer"
export CUDA_CACHE_PATH="$WORK_DIR/cuda-cache"
export TRITON_CACHE_DIR="$WORK_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torchinductor-cache"
export TORCH_HOME="$WORK_DIR/torch"
export VLLM_CACHE_ROOT="$WORK_DIR/vllm-cache"
export NUMBA_CACHE_DIR="$WORK_DIR/numba-cache"
export TMPDIR="$WORK_DIR/tmp"
export PATH="${ENV_DIR}/bin:${PATH}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
set -a
source "${COMMON_DIR}/clickhouse_credentials.env"
set +a

ACTUAL_HOST="$(hostname -s)"
if [ "$ACTUAL_HOST" != "$EXPECTED_HOST" ]; then
  echo "wrong_node expected=${EXPECTED_HOST} actual=${ACTUAL_HOST}" >&2
  exit 4
fi
echo "campaign_node=${ACTUAL_HOST} gpu_type=${GPU_TYPE} shard=${TASK_ID} slurm_job_id=${JOB_ID}"
echo "artifact_root=${OUT_DIR} git_tree=false cleanup=after_verified_upload"
nvidia-smi -L

"${ENV_DIR}/bin/python" "${COMMON_DIR}/calibration_runner.py" \
  --manifest "${COMMON_DIR}/${MANIFEST_NAME}" \
  --shard-id "$TASK_ID" \
  --output-dir "$OUT_DIR" \
  --vllm-bin "${ENV_DIR}/bin/vllm" \
  --python-bin "${ENV_DIR}/bin/python" \
  --observability-tools-dir "$OBSERVABILITY_DIR" \
  --otel-bundle "${OBSERVABILITY_DIR}/vendor/otel_bundle.zip"
RUN_RC=$?
echo "runner_rc=${RUN_RC}"
exit "$RUN_RC"
