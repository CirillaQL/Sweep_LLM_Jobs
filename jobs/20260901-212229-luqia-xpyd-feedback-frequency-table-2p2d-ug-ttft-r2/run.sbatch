#!/usr/bin/env bash
#SBATCH --job-name=xpyd-feedback-ttft-r2-ug
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260901-212229-luqia-xpyd-feedback-frequency-table-2p2d-ug-ttft-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260901-212229-luqia-xpyd-feedback-frequency-table-2p2d-ug-ttft-r2/slurm-%j.err

set -euo pipefail

JOB_NAME="20260901-212229-luqia-xpyd-feedback-frequency-table-2p2d-ug-ttft-r2"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_feedback_table_2p2d_7workloads.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/feedback_frequency_table_2p2d"
RUN_ID="${XPYD_PHASE3C_RUN_ID_OVERRIDE:-xpyd_feedback_table_2p2d_ttft_r2_${JOB_ID}}"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"
FREQUENCY_TABLE_PATH="${RAW_ROOT}/frequency_table.json"
FEEDBACK_EVENT_LOG="${RAW_ROOT}/feedback_events.jsonl"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export L40S_GPU_IDS=0,1 L4_GPU_IDS=0,1
export XPYD_ENDPOINTS_PER_ROLE=2
export XPYD_PHASE3C_CONFIG="${CONFIG}" XPYD_PHASE3C_RUN_ID="${RUN_ID}"
export XPYD_FEEDBACK_TABLE_OUTPUT_ROOT="${RAW_ROOT}/xpyd_phase3c_substrate"
export XPYD_FREQUENCY_TABLE_PATH="${FREQUENCY_TABLE_PATH}"
export XPYD_FEEDBACK_EVENT_LOG="${FEEDBACK_EVENT_LOG}"
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

for required in "${PYTHON_BIN}" "${CONFIG}" "${SOURCE_ROOT}/run_disagg_benchmark.sh" \
  "${SOURCE_ROOT}/gpu_monitor.py" "${SOURCE_ROOT}/paper/scripts/replay_synthetic_trace.py" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/workload_frequency_table.py" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/online_feedback_controller.py" \
  "${JOB_DIR}/validate_job.py"; do
  [ -e "${required}" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }
done

PYTHONPATH="${SOURCE_ROOT}/paper/scripts" "${PYTHON_BIN}" - \
  "${GIT_RESULT_DIR}/frequency_table_initial.json" <<'PY'
import json
from pathlib import Path
import sys

from xpyd.workload_frequency_table import WorkloadFrequencyTable

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "actuation_enabled": True,
    "search_enabled": True,
    "entries": WorkloadFrequencyTable().snapshot(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2
    exit 2
  }
done

"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}" \
  --config "${CONFIG}" --output "${GIT_RESULT_DIR}/preflight.json"

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_stage=online_binary_feedback_2P2D"
echo "xpyd_source_root=${SOURCE_ROOT}"
echo "xpyd_run_id=${RUN_ID}"
echo "xpyd_raw_root=${RAW_ROOT}"
echo "xpyd_git_result_dir=${GIT_RESULT_DIR}"
echo "xpyd_nodes=${SLURM_JOB_NODELIST:-unset}"
echo "xpyd_experiment_route=P0_to_D0"
echo "xpyd_service_route=P1_to_D1"
echo "xpyd_workloads=7_interleaved_classes_112_requests"
echo "xpyd_frequency_search=P_then_D_lower_bound_binary"
echo "xpyd_request_interval_s=25"
echo "xpyd_probe_interval_s=20"
echo "xpyd_max_concurrency=1"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"

cd "${SOURCE_ROOT}"
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh"

RUN_DIR="${XPYD_FEEDBACK_TABLE_OUTPUT_ROOT}/${RUN_ID}"
AUDIT="${RUN_DIR}/audit.json"
SUMMARY="${RUN_DIR}/summary.json"
for required in "${AUDIT}" "${SUMMARY}" "${RUN_DIR}/routes.csv" \
  "${RUN_DIR}/requests.csv" "${RUN_DIR}/workload_summary.csv" \
  "${RUN_DIR}/endpoint_summary.csv" "${RUN_DIR}/energy_summary.csv"; do
  [ -s "${required}" ] || { echo "xpyd_expected_result_missing=${required}" >&2; exit 20; }
done

for required in "${FREQUENCY_TABLE_PATH}" "${FEEDBACK_EVENT_LOG}"; do
  [ -s "${required}" ] || { echo "xpyd_feedback_result_missing=${required}" >&2; exit 22; }
done

"${PYTHON_BIN}" - "${AUDIT}" <<'PY'
import json
import sys

path = sys.argv[1]
audit = json.load(open(path, encoding="utf-8"))
accepted = audit.get("valid") is True
print("xpyd_phase3c_2p2d_service_audit_accepted=" + str(accepted).lower())
if not accepted:
    print(json.dumps(audit, indent=2, sort_keys=True))
    raise SystemExit(21)
PY

"${PYTHON_BIN}" - "${FREQUENCY_TABLE_PATH}" <<'PY'
import json
import sys

table = json.load(open(sys.argv[1], encoding="utf-8"))["entries"]
missing = sorted(key for key, entry in table.items() if entry.get("value") is None)
print("xpyd_frequency_table_missing=" + ",".join(missing))
if missing:
    raise SystemExit(23)
PY

cp "${AUDIT}" "${SUMMARY}" "${RUN_DIR}/summary.md" "${RUN_DIR}/routes.csv" \
  "${RUN_DIR}/requests.csv" "${RUN_DIR}/workload_summary.csv" \
  "${RUN_DIR}/endpoint_summary.csv" "${RUN_DIR}/energy_summary.csv" "${GIT_RESULT_DIR}/"
cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json"
cp "${FREQUENCY_TABLE_PATH}" "${FEEDBACK_EVENT_LOG}" "${GIT_RESULT_DIR}/"
printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
echo "xpyd_online_binary_feedback_2p2d_complete=true"
