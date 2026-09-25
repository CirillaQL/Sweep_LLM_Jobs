#!/usr/bin/env bash
#SBATCH --job-name=k4a4-pd-tier-neptune
# Third K4a retry, P moved to neptune. 267652/267664: the uranus vLLM steps hung on NFS
# (D state, nfs_wait_bit_killable); job 267695 then measured `import torch` at 57 s there.
# 267662: ganymede GPU 0 was busy, Slurm gave GPU 1 and the GPU-0-only guard exited.
# Now the job runs on whichever GPU Slurm allocates on each node (index + UUID
# detected here, UUID re-checked before every clock change and after vLLM starts).
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=48G
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-152921-luqia-k4a4-pd-tier-poisson-neptune/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-152921-luqia-k4a4-pd-tier-poisson-neptune/slurm-%j.err

set -euo pipefail
# Mirror this script's stdout/stderr into results/ (slurm-*.out is gitignored).
mkdir -p "/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-152921-luqia-k4a4-pd-tier-poisson-neptune/results/${SLURM_JOB_ID:-manual}"
exec > >(tee -a "/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-152921-luqia-k4a4-pd-tier-poisson-neptune/results/${SLURM_JOB_ID:-manual}/job_console.log") 2>&1

JOB_NAME="20260925-152921-luqia-k4a4-pd-tier-poisson-neptune"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_p0d0_k4a_tier_poisson.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/k4a_pd_tier_poisson"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=neptune L4_NODE=ganymede
export XPYD_ENDPOINTS_PER_ROLE=1 XPYD_ENERGY_GPU_COUNT=1 XPYD_ENERGY_CUDA_VISIBLE_DEVICES=0
# XPYD_PHASE3C_CONFIG is set after GPU detection to a copy with the allocated GPU ids.
export XPYD_K1_OUTPUT="${RAW_ROOT}/k1_burst_overhead"
export XPYD_ENERGY_VALIDATION_OUTPUT="${XPYD_K1_OUTPUT}"
export XPYD_EXPERIMENT_DRIVER_MODULE="xpyd.pd_burst_overhead"
# K1b: producer sends KV asynchronously instead of blocking the forward pass.
export XPYD_P_SEND_TYPE=PUT_ASYNC
export XPYD_FEEDBACK_EVENT_LOG="${RAW_ROOT}/feedback_events.jsonl"
export XPYD_FREQUENCY_TABLE_PATH="${RAW_ROOT}/frequency_table.json"
export RESULT_DIR="${RAW_ROOT}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-900}"
# Abort (with diagnostics) if a vLLM log shows no vLLM line at all after this long.
export VLLM_NO_OUTPUT_TIMEOUT_S="${VLLM_NO_OUTPUT_TIMEOUT_S_OVERRIDE:-300}"
# short partition: 30 min wall. Leave 150 s for server shutdown, clock reset and result copy.
export XPYD_DRIVER_DEADLINE_EPOCH="$(( $(date +%s) + 30 * 60 - 150 ))"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then XPYD_LOCAL_MODEL_PATH="${MODEL_OVERRIDE}"; else
  XPYD_LOCAL_MODEL_PATH=""
  while IFS= read -r model_snapshot; do
    if [ -s "${model_snapshot}/config.json" ] && [ -s "${model_snapshot}/tokenizer_config.json" ] && find "${model_snapshot}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then XPYD_LOCAL_MODEL_PATH="${model_snapshot}"; break; fi
  done < <(find "${LOCAL_MODEL_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
fi
[ -n "${XPYD_LOCAL_MODEL_PATH}" ] && [ -d "${XPYD_LOCAL_MODEL_PATH}" ] || { echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2; exit 2; }
export XPYD_LOCAL_MODEL_PATH MODEL="${XPYD_LOCAL_MODEL_PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

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
# Keep every tool that falls back to ~ (the default would be /root) inside WORK_DIR.
export HOME="${WORK_DIR}/home"

mkdir -p "${RAW_ROOT}" "${RESULT_DIR}" "${GIT_RESULT_DIR}" "${XPYD_K1_OUTPUT}" \
  "${HF_HUB_CACHE}" "${HF_ASSETS_CACHE}" "${HF_XET_CACHE}" \
  "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" \
  "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" "${RAY_TMPDIR}" "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${TIKTOKEN_CACHE_DIR}" \
  "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${TMPDIR}" "${HOME}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

for required in "${PYTHON_BIN}" "${CONFIG}" "${SOURCE_ROOT}/run_disagg_benchmark.sh" "${SOURCE_ROOT}/gpu_monitor.py" "${SOURCE_ROOT}/paper/scripts/xpyd/pd_burst_overhead.py" "${JOB_DIR}/validate_job.py"; do [ -e "${required}" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }; done
"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}" --config "${CONFIG}" --output "${GIT_RESULT_DIR}/preflight.json"

collect_results() {
  set +e
  [ ! -s "${RESULT_DIR}/startup_diagnostics.txt" ] || cp "${RESULT_DIR}/startup_diagnostics.txt" "${GIT_RESULT_DIR}/"
  for artifact in audit.json summary.json pd_requests.csv smoke_success.json smoke_failure.json; do [ ! -s "${XPYD_K1_OUTPUT}/${artifact}" ] || cp "${XPYD_K1_OUTPUT}/${artifact}" "${GIT_RESULT_DIR}/"; done
  [ ! -s "${XPYD_K1_OUTPUT}/proxy_diagnostics.jsonl" ] || gzip -c "${XPYD_K1_OUTPUT}/proxy_diagnostics.jsonl" > "${GIT_RESULT_DIR}/proxy_diagnostics.jsonl.gz"
  [ ! -s "${XPYD_FEEDBACK_EVENT_LOG}" ] || cp "${XPYD_FEEDBACK_EVENT_LOG}" "${GIT_RESULT_DIR}/frequency_actuations.jsonl"
  [ ! -s "${RESULT_DIR}/vllm_${L40S_NODE}_8100.log" ] || tail -n 400 "${RESULT_DIR}/vllm_${L40S_NODE}_8100.log" > "${GIT_RESULT_DIR}/prefill_server_tail.log"
  [ ! -s "${RESULT_DIR}/vllm_${L4_NODE}_8200.log" ] || tail -n 400 "${RESULT_DIR}/vllm_${L4_NODE}_8200.log" > "${GIT_RESULT_DIR}/decode_server_tail.log"
  cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json" 2>/dev/null || true
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
}
trap collect_results EXIT

# Devices are not cgroup-constrained on this cluster, so every clock command and
# NVML read must use the physical GPU Slurm allocated to this job. Detect index and
# UUID per node from a job step (same allocation), then pass them on:
#   L40S_GPU_IDS / L4_GPU_IDS          -> vLLM CUDA_VISIBLE_DEVICES, clock lock/reset
#   XPYD_EXPECTED_GPU_UUID_<node>      -> launcher aborts on any UUID mismatch
#   runtime config gpu_ids             -> driver clock actuation and energy reads
detect_gpu() {
  local node="$1" line
  line=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=1 --mem=256M \
    bash -c 'i="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"; u=$(nvidia-smi -i "${i}" --query-gpu=uuid,name --format=csv,noheader 2>/dev/null); echo "gpu=${i}|${u}"' 2>&1 | grep '^gpu=' | tail -n 1)
  echo "detected_${node}=${line}" | tee -a "${GIT_RESULT_DIR}/gpu_allocation.txt" >&2
  printf '%s\n' "${line#gpu=}"
}
parse_gpu() {  # "<index>|<uuid>, <name>" -> sets GPU_INDEX GPU_UUID GPU_NAME
  GPU_INDEX="${1%%|*}"; local rest="${1#*|}"
  GPU_UUID="$(printf '%s' "${rest%%,*}" | tr -d ' ')"; GPU_NAME="${rest#*, }"
  [[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] && [[ "${GPU_UUID}" == GPU-* ]]
}
P_LINE=$(detect_gpu neptune); D_LINE=$(detect_gpu ganymede)
parse_gpu "${P_LINE}" && [[ "${GPU_NAME}" == *L40S* ]] || { echo "gpu_detect_failed neptune: ${P_LINE}" | tee -a "${GIT_RESULT_DIR}/gpu_allocation.txt" >&2; exit 3; }
export L40S_GPU_IDS="${GPU_INDEX}" XPYD_EXPECTED_GPU_UUID_neptune="${GPU_UUID}"
parse_gpu "${D_LINE}" && [[ "${GPU_NAME}" == *L4* ]] && [[ "${GPU_NAME}" != *L40S* ]] || { echo "gpu_detect_failed ganymede: ${D_LINE}" | tee -a "${GIT_RESULT_DIR}/gpu_allocation.txt" >&2; exit 3; }
export L4_GPU_IDS="${GPU_INDEX}" XPYD_EXPECTED_GPU_UUID_ganymede="${GPU_UUID}"
echo "allocated neptune=${L40S_GPU_IDS} ganymede=${L4_GPU_IDS}" | tee -a "${GIT_RESULT_DIR}/gpu_allocation.txt"

RUNTIME_CONFIG="${RAW_ROOT}/xpyd_p0d0_k4a_tier_poisson.runtime.json"
"${PYTHON_BIN}" - "${CONFIG}" "${RUNTIME_CONFIG}" "${L40S_GPU_IDS}" "${L4_GPU_IDS}" <<'PY'
import json, sys
src, dst, p_gpu, d_gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
config = json.load(open(src, encoding="utf-8"))
for endpoint in config["endpoints"]:
    gpu = p_gpu if endpoint["role"] == "prefill" else d_gpu
    endpoint["gpu_ids"] = [gpu]
    endpoint["cuda_visible_device"] = str(gpu)
json.dump(config, open(dst, "w", encoding="utf-8"), indent=2)
print("runtime_config_gpu_ids=" + ",".join("%s:%s" % (e["endpoint_id"], e["gpu_ids"][0]) for e in config["endpoints"]))
PY
cp "${RUNTIME_CONFIG}" "${GIT_RESULT_DIR}/runtime_config.json"
export XPYD_PHASE3C_CONFIG="${RUNTIME_CONFIG}"

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_endpoint_pair=P0_to_D0_only"
echo "xpyd_k1b_send_type=${XPYD_P_SEND_TYPE}"
echo "xpyd_k4a_phases=tier_poisson(2/4/6 rps x P 1305/1815/2520, D 1050),park_idle"
echo "xpyd_driver_deadline_epoch=${XPYD_DRIVER_DEADLINE_EPOCH}"

cd "${SOURCE_ROOT}"
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh"
"${PYTHON_BIN}" - "${XPYD_K1_OUTPUT}/audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
print("xpyd_k1_valid=" + str(audit.get("valid")).lower())
print("xpyd_k1_failed_measurements=" + str(audit.get("failed_measurements")))
PY
