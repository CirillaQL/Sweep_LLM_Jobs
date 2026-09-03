#!/usr/bin/env bash
#SBATCH --job-name=xpyd-p0d0-axis-power-sweep
#SBATCH --partition=long
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260903-203634-luqia-xpyd-p0d0-exhaustive-axis-power/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260903-203634-luqia-xpyd-p0d0-exhaustive-axis-power/slurm-%j.err

set -euo pipefail

JOB_NAME="20260903-203634-luqia-xpyd-p0d0-exhaustive-axis-power"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_feedback_table_2p2d_7workloads.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/p0d0_exhaustive_axis_power"
RUN_ID="${XPYD_PHASE3C_RUN_ID_OVERRIDE:-xpyd_p0d0_exhaustive_axis_power_${JOB_ID}}"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"
FREQUENCY_TABLE_PATH="${RAW_ROOT}/frequency_table.json"
FEEDBACK_EVENT_LOG="${RAW_ROOT}/feedback_events.jsonl"
SERVICE_REQUEST_LOG="${RAW_ROOT}/service_request_dispatch.jsonl"
RUN_DIR="${RAW_ROOT}/xpyd_phase3c_substrate/${RUN_ID}"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export L40S_GPU_IDS=0,1 L4_GPU_IDS=0,1
export XPYD_ENDPOINTS_PER_ROLE=2
export XPYD_PHASE3C_CONFIG="${CONFIG}" XPYD_PHASE3C_RUN_ID="${RUN_ID}"
export XPYD_EXPLORATION_ONLY=1
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

for required in "${PYTHON_BIN}" "${CONFIG}" "${SOURCE_ROOT}/run_disagg_benchmark.sh" \
  "${SOURCE_ROOT}/gpu_monitor.py" "${SOURCE_ROOT}/paper/scripts/replay_synthetic_trace.py" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/workload_frequency_table.py" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/online_feedback_controller.py" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/exploration_only.py" \
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
echo "xpyd_stage=P0D0_exploration_only_exhaustive_axis_power"
echo "xpyd_source_root=${SOURCE_ROOT}"
echo "xpyd_run_id=${RUN_ID}"
echo "xpyd_raw_root=${RAW_ROOT}"
echo "xpyd_git_result_dir=${GIT_RESULT_DIR}"
echo "xpyd_nodes=${SLURM_JOB_NODELIST:-unset}"
echo "xpyd_experiment_route=P0_to_D0"
echo "xpyd_formal_service_requests=0"
echo "xpyd_workloads=7"
echo "xpyd_frequency_search=P17_then_D15_minimum_mean_endpoint_power"
echo "xpyd_candidate_samples=exactly_3"
echo "xpyd_measured_probes=672_plus_one_unmeasured_warmup"
echo "xpyd_exploration_ttft_p95_slo_ms=500"
echo "xpyd_exploration_tpot_p95_slo_ms=200"
echo "xpyd_probe_interval_s=0"
echo "xpyd_post_readback_settle_s=0.5_changed_frequencies_only"
echo "xpyd_exploration_timeout_s=25200"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"

collect_compact_results() {
  set +e
  mkdir -p "${GIT_RESULT_DIR}"
  for artifact in \
    "${RUN_DIR}/audit.json" "${RUN_DIR}/summary.json" "${RUN_DIR}/summary.md" \
    "${RUN_DIR}/candidate_measurements.csv" \
    "${RUN_DIR}/best_power_configurations.csv" \
    "${RUN_DIR}/best_power_configurations.json" \
    "${FREQUENCY_TABLE_PATH}" "${FEEDBACK_EVENT_LOG}"; do
    if [ -s "${artifact}" ]; then
      cp "${artifact}" "${GIT_RESULT_DIR}/"
    fi
  done
  if [ -s "${CONFIG}" ]; then
    cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json"
  fi
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
  return 0
}
trap collect_compact_results EXIT

cd "${SOURCE_ROOT}"
benchmark_status=0
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash \
  "${SOURCE_ROOT}/run_disagg_benchmark.sh" || benchmark_status=$?
echo "xpyd_benchmark_wrapper_status=${benchmark_status}"
[ "${benchmark_status}" -eq 0 ] || exit "${benchmark_status}"

AUDIT="${RUN_DIR}/audit.json"
SUMMARY="${RUN_DIR}/summary.json"
for required in "${AUDIT}" "${SUMMARY}" "${RUN_DIR}/summary.md" \
  "${RUN_DIR}/candidate_measurements.csv" \
  "${RUN_DIR}/best_power_configurations.csv" \
  "${RUN_DIR}/best_power_configurations.json" \
  "${FREQUENCY_TABLE_PATH}" "${FEEDBACK_EVENT_LOG}"; do
  [ -s "${required}" ] || { echo "xpyd_expected_result_missing=${required}" >&2; exit 20; }
done

"${PYTHON_BIN}" - "${AUDIT}" <<'PY'
import json
import sys

path = sys.argv[1]
audit = json.load(open(path, encoding="utf-8"))
accepted = (
    audit.get("valid") is True
    and audit.get("formal_service_requests") == 0
    and audit.get("candidate_rows") == 224
    and audit.get("measured_probes") == 672
)
print("xpyd_p0d0_exploration_audit_accepted=" + str(accepted).lower())
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

cp "${AUDIT}" "${SUMMARY}" "${RUN_DIR}/summary.md" \
  "${RUN_DIR}/candidate_measurements.csv" \
  "${RUN_DIR}/best_power_configurations.csv" \
  "${RUN_DIR}/best_power_configurations.json" "${GIT_RESULT_DIR}/"
cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json"
cp "${FREQUENCY_TABLE_PATH}" "${FEEDBACK_EVENT_LOG}" "${GIT_RESULT_DIR}/"
printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
echo "xpyd_p0d0_exploration_only_axis_power_complete=true"
