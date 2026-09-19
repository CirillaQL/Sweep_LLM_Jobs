#!/usr/bin/env bash
#SBATCH --job-name=xpyd-exp-v1-p0d0-joint-grid-r1
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260919-140000-luqia-xpyd-exp-v1-p0d0-joint-grid-validation-retry-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260919-140000-luqia-xpyd-exp-v1-p0d0-joint-grid-validation-retry-r1/slurm-%j.err

set -euo pipefail

JOB_NAME="20260919-140000-luqia-xpyd-exp-v1-p0d0-joint-grid-validation-retry-r1"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_p0d0_joint_grid_validation.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/p0d0_joint_grid_validation"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=neptune L4_NODE=ganymede
export L40S_GPU_IDS=0 L4_GPU_IDS=0
export XPYD_ENDPOINTS_PER_ROLE=1 XPYD_ENERGY_GPU_COUNT=1 XPYD_ENERGY_CUDA_VISIBLE_DEVICES=0
export XPYD_PHASE3C_CONFIG="${CONFIG}"
export XPYD_JOINT_GRID_OUTPUT="${RAW_ROOT}/joint_grid_validation"
export XPYD_ENERGY_VALIDATION_OUTPUT="${XPYD_JOINT_GRID_OUTPUT}"
export XPYD_EXPERIMENT_DRIVER_MODULE="xpyd.joint_grid_validation"
export XPYD_FEEDBACK_EVENT_LOG="${RAW_ROOT}/feedback_events.jsonl"
export XPYD_FREQUENCY_TABLE_PATH="${RAW_ROOT}/frequency_table.json"
export RESULT_DIR="${RAW_ROOT}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1500}"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then XPYD_LOCAL_MODEL_PATH="${MODEL_OVERRIDE}"; else
  XPYD_LOCAL_MODEL_PATH=""
  while IFS= read -r model_snapshot; do
    if [ -s "${model_snapshot}/config.json" ] && [ -s "${model_snapshot}/tokenizer_config.json" ] && find "${model_snapshot}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then XPYD_LOCAL_MODEL_PATH="${model_snapshot}"; break; fi
  done < <(find "${LOCAL_MODEL_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
fi
[ -n "${XPYD_LOCAL_MODEL_PATH}" ] && [ -d "${XPYD_LOCAL_MODEL_PATH}" ] || { echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2; exit 2; }
export XPYD_LOCAL_MODEL_PATH MODEL="${XPYD_LOCAL_MODEL_PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export HF_HOME="${WORK_DIR}/huggingface" HF_HUB_CACHE="${WORK_DIR}/huggingface/hub" XDG_CACHE_HOME="${WORK_DIR}/xdg-cache" XDG_CONFIG_HOME="${WORK_DIR}/xdg-config" XDG_RUNTIME_DIR="${WORK_DIR}/xdg-runtime" TORCH_HOME="${WORK_DIR}/torch-cache" TORCH_EXTENSIONS_DIR="${WORK_DIR}/torch-extensions" TRITON_CACHE_DIR="${WORK_DIR}/triton-cache" CUDA_CACHE_PATH="${WORK_DIR}/cuda-cache" TMPDIR="${WORK_DIR}/tmp"
mkdir -p "${RAW_ROOT}" "${RESULT_DIR}" "${GIT_RESULT_DIR}" "${XPYD_JOINT_GRID_OUTPUT}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

for required in "${PYTHON_BIN}" "${CONFIG}" "${SOURCE_ROOT}/run_disagg_benchmark.sh" "${SOURCE_ROOT}/gpu_monitor.py" "${SOURCE_ROOT}/paper/scripts/xpyd/joint_grid_validation.py" "${JOB_DIR}/validate_job.py"; do [ -e "${required}" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }; done
"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}" --config "${CONFIG}" --output "${GIT_RESULT_DIR}/preflight.json"

collect_results() {
  set +e
  for artifact in audit.json joint_grid_strategy_validation.json joint_grid_request_measurements.csv joint_grid_candidate_measurements.csv; do [ ! -s "${XPYD_JOINT_GRID_OUTPUT}/${artifact}" ] || cp "${XPYD_JOINT_GRID_OUTPUT}/${artifact}" "${GIT_RESULT_DIR}/"; done
  cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json" 2>/dev/null || true
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
}
trap collect_results EXIT

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_endpoint_pair=P0_to_D0_only"
echo "xpyd_joint_grid=17x15"
echo "xpyd_samples_per_pair=3"
echo "xpyd_workloads=small_light,prefill_medium,decode_medium,decode_heavy,balanced_medium"
echo "xpyd_total_measurements=3825"

cd "${SOURCE_ROOT}"
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh"
"${PYTHON_BIN}" - "${XPYD_JOINT_GRID_OUTPUT}/audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
print("xpyd_joint_grid_valid=" + str(audit.get("valid")).lower())
print("xpyd_joint_grid_failed_measurements=" + str(audit.get("failed_measurements")))
PY
