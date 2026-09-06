#!/usr/bin/env bash
#SBATCH --job-name=xpyd-windowed-feedback-energy-r11
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260906-123000-luqia-xpyd-windowed-canary-production-energy-r11/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260906-123000-luqia-xpyd-windowed-canary-production-energy-r11/slurm-%j.err

set -euo pipefail

JOB_NAME="20260906-123000-luqia-xpyd-windowed-canary-production-energy-r11"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_feedback_table_2p2d_7workloads.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/feedback_frequency_table_2p2d"
RUN_ID="${XPYD_PHASE3C_RUN_ID_OVERRIDE:-xpyd_windowed_canary_production_energy_r11_${JOB_ID}}"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"
FREQUENCY_TABLE_PATH="${RAW_ROOT}/frequency_table.json"
FEEDBACK_EVENT_LOG="${RAW_ROOT}/feedback_events.jsonl"
SERVICE_REQUEST_LOG="${RAW_ROOT}/service_request_dispatch.jsonl"
RUN_DIR="${RAW_ROOT}/xpyd_phase3c_substrate/${RUN_ID}"
STEADY_DIR="${RAW_ROOT}/steady_state"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export L40S_GPU_IDS=0,1 L4_GPU_IDS=0,1
export XPYD_ENDPOINTS_PER_ROLE=2
export XPYD_PHASE3C_CONFIG="${CONFIG}" XPYD_PHASE3C_RUN_ID="${RUN_ID}"
export XPYD_FEEDBACK_TABLE_OUTPUT_ROOT="${RAW_ROOT}/xpyd_phase3c_substrate"
export XPYD_FREQUENCY_TABLE_PATH="${FREQUENCY_TABLE_PATH}"
export XPYD_FEEDBACK_EVENT_LOG="${FEEDBACK_EVENT_LOG}"
export XPYD_SERVICE_REQUEST_LOG="${SERVICE_REQUEST_LOG}"
export RESULT_DIR="${RAW_ROOT}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1500}"
export XPYD_PROXY_STARTUP_TIMEOUT_S="${XPYD_PROXY_STARTUP_TIMEOUT_S_OVERRIDE:-600}"
export XPYD_FIXED_CLOCK_EVIDENCE="${RESULT_DIR}/fixed_clock_evidence.log"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then
  XPYD_LOCAL_MODEL_PATH="${MODEL_OVERRIDE}"
else
  XPYD_LOCAL_MODEL_PATH=""
  while IFS= read -r model_snapshot; do
    if [ -s "${model_snapshot}/config.json" ] && \
       [ -s "${model_snapshot}/tokenizer_config.json" ] && \
       find "${model_snapshot}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then
      XPYD_LOCAL_MODEL_PATH="${model_snapshot}"
      break
    fi
  done < <(find "${LOCAL_MODEL_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
fi
[ -n "${XPYD_LOCAL_MODEL_PATH}" ] && [ -d "${XPYD_LOCAL_MODEL_PATH}" ] || {
  echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2
  exit 2
}
export XPYD_LOCAL_MODEL_PATH MODEL="${XPYD_LOCAL_MODEL_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export HF_HOME="${WORK_DIR}/huggingface" HF_TOKEN_PATH="${WORK_DIR}/huggingface/token"
export HF_HUB_CACHE="${WORK_DIR}/huggingface/hub"
export HF_ASSETS_CACHE="${WORK_DIR}/huggingface/assets"
export HF_XET_CACHE="${WORK_DIR}/huggingface/xet"
export HF_DATASETS_CACHE="${WORK_DIR}/huggingface/datasets"
export HF_MODULES_CACHE="${WORK_DIR}/huggingface/modules"
export XDG_CACHE_HOME="${WORK_DIR}/xdg-cache" XDG_CONFIG_HOME="${WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${WORK_DIR}/xdg-data" XDG_STATE_HOME="${WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${WORK_DIR}/xdg-runtime"
export FLASHINFER_WORKSPACE_BASE="${WORK_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${WORK_DIR}/vllm-cache" VLLM_CONFIG_ROOT="${WORK_DIR}/vllm-config"
export TORCH_HOME="${WORK_DIR}/torch-cache" TORCH_EXTENSIONS_DIR="${WORK_DIR}/torch-extensions"
export TRITON_CACHE_DIR="${WORK_DIR}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK_DIR}/torchinductor-cache"
export CUDA_CACHE_PATH="${WORK_DIR}/cuda-cache" NUMBA_CACHE_DIR="${WORK_DIR}/numba-cache"
export RAY_TMPDIR="${WORK_DIR}/ray-tmp" PIP_CACHE_DIR="${WORK_DIR}/pip-cache"
export UV_CACHE_DIR="${WORK_DIR}/uv-cache" PYTHONPYCACHEPREFIX="${WORK_DIR}/python-pycache"
export TIKTOKEN_CACHE_DIR="${WORK_DIR}/tiktoken-cache" MPLCONFIGDIR="${WORK_DIR}/matplotlib"
export CUPY_CACHE_DIR="${WORK_DIR}/cupy-cache"
export TMPDIR="${WORK_DIR}/tmp" TMP="${WORK_DIR}/tmp" TEMP="${WORK_DIR}/tmp"

mkdir -p "${RAW_ROOT}" "${RESULT_DIR}" "${XPYD_FEEDBACK_TABLE_OUTPUT_ROOT}" "${GIT_RESULT_DIR}" \
  "${HF_HUB_CACHE}" "${HF_ASSETS_CACHE}" "${HF_XET_CACHE}" \
  "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" \
  "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" "${RAY_TMPDIR}" "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${TIKTOKEN_CACHE_DIR}" \
  "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"


export XPYD_ENERGY_VALIDATION_OUTPUT="${RAW_ROOT}/energy_validation"
mkdir -p "${XPYD_ENERGY_VALIDATION_OUTPUT}"
export PYTHONPATH="${SOURCE_ROOT}/paper/scripts${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}"
collect_results() {
  local run_status=$?
  set +e
  for artifact in summary.json summary.md audit.json canary_overhead.json hardware_grids.json \
    requests_energy.jsonl canary_attempts.jsonl experiment_config.json \
    production_requests.jsonl production_requests.csv power_trace_1s.csv \
    canary_probes.jsonl canary_probes.csv canary_probe_events.jsonl canary_candidates.jsonl canary_candidates.csv \
    feedback_events.jsonl service_dispatch.jsonl workload_windows.jsonl cleanup_errors.jsonl; do
    if [ -s "${XPYD_ENERGY_VALIDATION_OUTPUT}/${artifact}" ]; then
      cp "${XPYD_ENERGY_VALIDATION_OUTPUT}/${artifact}" "${GIT_RESULT_DIR}/"
    fi
  done
  [ ! -s "${FREQUENCY_TABLE_PATH}" ] || cp "${FREQUENCY_TABLE_PATH}" "${GIT_RESULT_DIR}/"
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
  return "${run_status}"
}
trap collect_results EXIT
cd "${SOURCE_ROOT}"
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh"
"${PYTHON_BIN}" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["valid"] is True, a' "${XPYD_ENERGY_VALIDATION_OUTPUT}/audit.json"
