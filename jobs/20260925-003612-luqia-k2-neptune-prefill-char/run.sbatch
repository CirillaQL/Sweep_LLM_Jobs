#!/usr/bin/env bash
#SBATCH --job-name=k2-neptune-prefill
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=neptune
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@90
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-003612-luqia-k2-neptune-prefill-char/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-003612-luqia-k2-neptune-prefill-char/slurm-%j.err

# K2 + K1 plain-vLLM control: prefill TTFT vs burst size/input length/P clock, Poisson arrivals, idle power, clock switch latency on one L40S.
set -euo pipefail
START_EPOCH=$(date +%s)

JOB_NAME="20260925-003612-luqia-k2-neptune-prefill-char"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_ROOT="${WORK_DIR}/k2_prefill_char"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"
PORT=8100
export PATH="${ENV_DIR}/bin:${PATH}"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
MODEL=""
while IFS= read -r model_snapshot; do
  if [ -s "${model_snapshot}/config.json" ] && [ -s "${model_snapshot}/tokenizer_config.json" ] && find "${model_snapshot}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then MODEL="${model_snapshot}"; break; fi
done < <(find "${LOCAL_MODEL_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ -n "${MODEL}" ] || { echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Every cache and HOME inside WORK_DIR (same layout as the r3 jobs) so nothing
# falls back to the default /root locations.
export HOME="${WORK_DIR}/home"
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
mkdir -p "${RAW_ROOT}" "${GIT_RESULT_DIR}" "${HOME}" "${HF_HUB_CACHE}" "${HF_ASSETS_CACHE}" \
  "${HF_XET_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}" "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}" \
  "${NUMBA_CACHE_DIR}" "${RAY_TMPDIR}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
  "${TIKTOKEN_CACHE_DIR}" "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

# Physical GPU index actually allocated by Slurm (devices are not cgroup
# constrained here, so this is the index nvidia-smi and NVML must use).
GPU_INDEX="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [ "$(nvidia-smi -L | wc -l)" -eq 1 ]; then GPU_INDEX=0; fi   # constrained view: only ours is visible
[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || { echo "unexpected_gpu_allocation=${GPU_INDEX}" >&2; exit 3; }
GPU_NAME=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)
{ echo "gpu_index=${GPU_INDEX}"; echo "gpu_name=${GPU_NAME}"; echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"; echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi -i "${GPU_INDEX}" --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.mem,power.draw --format=csv; } | tee "${GIT_RESULT_DIR}/gpu_allocation.txt"
case "${GPU_NAME}" in *L40S*) ;; *) echo "wrong_gpu_type=${GPU_NAME}" >&2; exit 3;; esac

SERVER_PID=""
cleanup() {
  set +e
  [ -z "${SERVER_PID}" ] || { kill "${SERVER_PID}" 2>/dev/null; sleep 5; kill -9 "${SERVER_PID}" 2>/dev/null; }
  sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc > "${GIT_RESULT_DIR}/clock_reset.txt" 2>&1
  sudo -n nvidia-smi -i "${GPU_INDEX}" -rmc >> "${GIT_RESULT_DIR}/clock_reset.txt" 2>&1
  for artifact in summary.json bursts.csv poisson.csv decode.csv gpu_events.csv power_trace.csv; do
    [ ! -s "${RAW_ROOT}/${artifact}" ] || cp "${RAW_ROOT}/${artifact}" "${GIT_RESULT_DIR}/"
  done
  [ ! -s "${RAW_ROOT}/vllm_server.log" ] || tail -n 300 "${RAW_ROOT}/vllm_server.log" > "${GIT_RESULT_DIR}/vllm_server_tail.log"
  cp "${JOB_DIR}/config.json" "${GIT_RESULT_DIR}/experiment_config.json" 2>/dev/null
  printf '%s\n' "${RAW_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
}
trap cleanup EXIT

# Fail fast if clock locking is not permitted on this node.
sudo -n nvidia-smi -i "${GPU_INDEX}" -lgc 2520,2520 | tee "${GIT_RESULT_DIR}/clock_permission.txt"

"${PYTHON_BIN}" -c "import vllm; print('vllm', vllm.__version__)"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" --host 127.0.0.1 --port "${PORT}" --tensor-parallel-size 1 \
  --max-model-len 4096 --no-enable-prefix-caching --no-enable-chunked-prefill  \
  > "${RAW_ROOT}/vllm_server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "vllm_server_exited_during_startup" >&2; exit 4; }
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || { echo "vllm_server_not_ready_in_600s" >&2; exit 4; }
echo "server_ready_after_s=$(( $(date +%s) - START_EPOCH ))" | tee -a "${GIT_RESULT_DIR}/gpu_allocation.txt"

# 30-minute wall clock minus 150 s for shutdown, clock reset and result copy.
BUDGET_S=$(( 30 * 60 - 150 - ( $(date +%s) - START_EPOCH ) ))
"${PYTHON_BIN}" "${JOB_DIR}/gpu_char.py" --config "${JOB_DIR}/config.json" --output "${RAW_ROOT}" \
  --time-budget-s "${BUDGET_S}" --model "${MODEL}" --gpu-index "${GPU_INDEX}"
