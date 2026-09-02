#!/usr/bin/env bash
#SBATCH --job-name=xpyd-fixed-table-power-compare
#SBATCH --partition=long
#SBATCH --time=07:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260902-141257-luqia-xpyd-fixed-table-power-compare/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260902-141257-luqia-xpyd-fixed-table-power-compare/slurm-%j.err

set -euo pipefail

JOB_NAME="20260902-141257-luqia-xpyd-fixed-table-power-compare"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/fixed_table_power_compare"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"
BASELINE_CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_power_compare_full_frequency.json"
OPTIMIZED_CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_power_compare_fixed_table.json"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export L40S_GPU_IDS=0,1 L4_GPU_IDS=0,1 XPYD_ENDPOINTS_PER_ROLE=2
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1500}"
export XPYD_PROXY_STARTUP_TIMEOUT_S="${XPYD_PROXY_STARTUP_TIMEOUT_S_OVERRIDE:-600}"
export MODEL XPYD_LOCAL_MODEL_PATH

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
export HF_HUB_CACHE="${WORK_DIR}/huggingface/hub" HF_ASSETS_CACHE="${WORK_DIR}/huggingface/assets"
export HF_XET_CACHE="${WORK_DIR}/huggingface/xet" HF_DATASETS_CACHE="${WORK_DIR}/huggingface/datasets"
export HF_MODULES_CACHE="${WORK_DIR}/huggingface/modules"
export XDG_CACHE_HOME="${WORK_DIR}/xdg-cache" XDG_CONFIG_HOME="${WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${WORK_DIR}/xdg-data" XDG_STATE_HOME="${WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${WORK_DIR}/xdg-runtime"
export FLASHINFER_WORKSPACE_BASE="${WORK_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${WORK_DIR}/vllm-cache" VLLM_CONFIG_ROOT="${WORK_DIR}/vllm-config"
export TORCH_HOME="${WORK_DIR}/torch-cache" TORCH_EXTENSIONS_DIR="${WORK_DIR}/torch-extensions"
export TRITON_CACHE_DIR="${WORK_DIR}/triton-cache" TORCHINDUCTOR_CACHE_DIR="${WORK_DIR}/torchinductor-cache"
export CUDA_CACHE_PATH="${WORK_DIR}/cuda-cache" NUMBA_CACHE_DIR="${WORK_DIR}/numba-cache"
export RAY_TMPDIR="${WORK_DIR}/ray-tmp" PIP_CACHE_DIR="${WORK_DIR}/pip-cache"
export UV_CACHE_DIR="${WORK_DIR}/uv-cache" PYTHONPYCACHEPREFIX="${WORK_DIR}/python-pycache"
export TIKTOKEN_CACHE_DIR="${WORK_DIR}/tiktoken-cache" MPLCONFIGDIR="${WORK_DIR}/matplotlib"
export CUPY_CACHE_DIR="${WORK_DIR}/cupy-cache"
export TMPDIR="${WORK_DIR}/tmp" TMP="${WORK_DIR}/tmp" TEMP="${WORK_DIR}/tmp"

mkdir -p "${RAW_ROOT}" "${GIT_RESULT_DIR}" "${HF_HUB_CACHE}" "${HF_ASSETS_CACHE}" \
  "${HF_XET_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" \
  "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" "${RAY_TMPDIR}" "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${TIKTOKEN_CACHE_DIR}" \
  "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

for required in "${PYTHON_BIN}" "${BASELINE_CONFIG}" "${OPTIMIZED_CONFIG}" \
  "${SOURCE_ROOT}/run_disagg_benchmark.sh" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/compare_fixed_table_power.py" \
  "${JOB_DIR}/validate_job.py"; do
  [ -e "${required}" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }
done
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2
    exit 2
  }
done

"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}" \
  --baseline-config "${BASELINE_CONFIG}" --optimized-config "${OPTIMIZED_CONFIG}" \
  --output "${GIT_RESULT_DIR}/preflight.json"

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_stage=fixed_table_power_comparison"
echo "xpyd_nodes=${SLURM_JOB_NODELIST:-unset}"
echo "xpyd_measurement_route=P1_to_D1"
echo "xpyd_workloads=6_serial_windows_48_each_288_requests_per_mode"
echo "xpyd_request_interval_s=25"
echo "xpyd_baseline_frequencies=P2520_D1500"
echo "xpyd_optimized_table=job255414_without_both_heavy"
echo "xpyd_measurement_window=route_selected_to_response_completed"
echo "xpyd_raw_root=${RAW_ROOT}"

collect_compact_results() {
  set +e
  mkdir -p "${GIT_RESULT_DIR}/baseline" "${GIT_RESULT_DIR}/optimized" "${GIT_RESULT_DIR}/comparison"
  local variant run_id run_dir artifact
  for variant in baseline optimized; do
    run_id="xpyd_power_compare_${variant}_${JOB_ID}"
    run_dir="${RAW_ROOT}/${variant}/xpyd_phase3c_substrate/${run_id}"
    for artifact in audit.json summary.json summary.md routes.csv requests.csv \
      workload_summary.csv endpoint_summary.csv energy_summary.csv workload_energy_summary.csv; do
      [ -s "${run_dir}/${artifact}" ] && cp "${run_dir}/${artifact}" "${GIT_RESULT_DIR}/${variant}/"
    done
    [ -s "${RAW_ROOT}/${variant}/frequency_table.json" ] && \
      cp "${RAW_ROOT}/${variant}/frequency_table.json" "${GIT_RESULT_DIR}/${variant}/"
    [ -s "${RAW_ROOT}/${variant}/service_request_dispatch.jsonl" ] && \
      cp "${RAW_ROOT}/${variant}/service_request_dispatch.jsonl" "${GIT_RESULT_DIR}/${variant}/"
    [ -s "${RAW_ROOT}/${variant}/feedback_events.jsonl" ] && \
      cp "${RAW_ROOT}/${variant}/feedback_events.jsonl" "${GIT_RESULT_DIR}/${variant}/"
  done
  for artifact in power_comparison.json power_comparison.csv power_comparison.md; do
    [ -s "${RAW_ROOT}/comparison/${artifact}" ] && cp "${RAW_ROOT}/comparison/${artifact}" "${GIT_RESULT_DIR}/comparison/"
  done
  cp "${BASELINE_CONFIG}" "${GIT_RESULT_DIR}/baseline/experiment_config.json" 2>/dev/null || true
  cp "${OPTIMIZED_CONFIG}" "${GIT_RESULT_DIR}/optimized/experiment_config.json" 2>/dev/null || true
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
  return 0
}
trap collect_compact_results EXIT

run_variant() {
  local variant="$1"
  local config="$2"
  local variant_root="${RAW_ROOT}/${variant}"
  local run_id="xpyd_power_compare_${variant}_${JOB_ID}"
  mkdir -p "${variant_root}/launcher" "${variant_root}/xpyd_phase3c_substrate"
  export XPYD_PHASE3C_CONFIG="${config}" XPYD_PHASE3C_RUN_ID="${run_id}"
  export XPYD_FEEDBACK_TABLE_OUTPUT_ROOT="${variant_root}/xpyd_phase3c_substrate"
  export XPYD_FREQUENCY_TABLE_PATH="${variant_root}/frequency_table.json"
  export XPYD_FEEDBACK_EVENT_LOG="${variant_root}/feedback_events.jsonl"
  export XPYD_SERVICE_REQUEST_LOG="${variant_root}/service_request_dispatch.jsonl"
  export RESULT_DIR="${variant_root}/launcher"
  export XPYD_FIXED_CLOCK_EVIDENCE="${variant_root}/launcher/fixed_clock_evidence.log"
  echo "xpyd_variant_start=${variant}"
  (
    cd "${SOURCE_ROOT}"
    SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh"
  )
  local run_dir="${variant_root}/xpyd_phase3c_substrate/${run_id}"
  for required in audit.json summary.json routes.csv requests.csv workload_summary.csv \
    energy_summary.csv workload_energy_summary.csv; do
    [ -s "${run_dir}/${required}" ] || {
      echo "xpyd_${variant}_result_missing=${run_dir}/${required}" >&2
      return 20
    }
  done
  "${PYTHON_BIN}" - "${run_dir}/audit.json" "${variant_root}/service_request_dispatch.jsonl" \
    "${variant_root}/feedback_events.jsonl" <<'PY'
import json
import pathlib
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("valid") is not True:
    raise SystemExit("phase3c audit failed: " + json.dumps(audit, sort_keys=True))
dispatch = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
if len(dispatch) != 288 or any(row.get("table_hit") is not True for row in dispatch):
    raise SystemExit("expected exactly 288 fixed-table service dispatches")
events_path = pathlib.Path(sys.argv[3])
events = [json.loads(line) for line in events_path.open(encoding="utf-8") if line.strip()] if events_path.is_file() else []
if any(str(row.get("event", "")).startswith("exploration_") or str(row.get("event", "")).startswith("probe_") for row in events):
    raise SystemExit("fixed-table comparison unexpectedly performed exploration")
PY
  echo "xpyd_variant_complete=${variant}"
}

run_variant baseline "${BASELINE_CONFIG}"
sleep 30
run_variant optimized "${OPTIMIZED_CONFIG}"

BASELINE_RUN_DIR="${RAW_ROOT}/baseline/xpyd_phase3c_substrate/xpyd_power_compare_baseline_${JOB_ID}"
OPTIMIZED_RUN_DIR="${RAW_ROOT}/optimized/xpyd_phase3c_substrate/xpyd_power_compare_optimized_${JOB_ID}"
"${PYTHON_BIN}" "${SOURCE_ROOT}/paper/scripts/xpyd/compare_fixed_table_power.py" \
  --baseline-dir "${BASELINE_RUN_DIR}" --optimized-dir "${OPTIMIZED_RUN_DIR}" \
  --output-dir "${RAW_ROOT}/comparison"
collect_compact_results
echo "xpyd_fixed_table_power_comparison_complete=true"
