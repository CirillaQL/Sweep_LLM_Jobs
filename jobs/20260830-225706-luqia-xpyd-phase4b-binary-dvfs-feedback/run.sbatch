#!/usr/bin/env bash
#SBATCH --job-name=xpyd-4b-binary-dvfs
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/vllm_job_work/slurm-%j.out
#SBATCH --error=/data/users/chjing/vllm_job_work/slurm-%j.err

set -euo pipefail

JOB_NAME="20260830-225706-luqia-xpyd-phase4b-binary-dvfs-feedback"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_phase4b_binary_dvfs_uranus_ganymede.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
CACHE_ROOT="/data/users/chjing/vllm_job_work/${JOB_ID}"
RAW_RESULT_ROOT="${CACHE_ROOT}/binary_dvfs"
RUN_ID="${XPYD_BINARY_DVFS_RUN_ID_OVERRIDE:-xpyd_binary_dvfs_${JOB_ID}}"
RUN_DIR="${RAW_RESULT_ROOT}/${RUN_ID}"
LAUNCHER_DIR="${CACHE_ROOT}/launcher"
LIVE_CLOCK_DIR="${CACHE_ROOT}/live_clock"
GIT_RESULT_DIR="${JOB_DIR}/results/${JOB_ID}"

mkdir -p "${CACHE_ROOT}" "${RAW_RESULT_ROOT}" "${LAUNCHER_DIR}" \
  "${LIVE_CLOCK_DIR}" "${GIT_RESULT_DIR}"
exec > >(tee -a "${CACHE_ROOT}/job.log") 2>&1
archive_slurm_logs() {
  local suffix source target
  for suffix in out err; do
    source="/data/users/chjing/vllm_job_work/slurm-${JOB_ID}.${suffix}"
    target="${CACHE_ROOT}/slurm-${JOB_ID}.${suffix}"
    [ ! -e "${source}" ] || mv "${source}" "${target}" || true
  done
}
trap archive_slurm_logs EXIT

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export XPYD_PHASE4B_CONFIG="${CONFIG}" XPYD_PHASE4B_RUN_ID="${RUN_ID}"
export XPYD_PHASE4B_STAGE=stationary XPYD_PHASE4B_SMOKE=0
export XPYD_BINARY_DVFS=1 XPYD_BINARY_DVFS_OUTPUT_ROOT="${RAW_RESULT_ROOT}"
export XPYD_PHASE4B_OUTPUT_ROOT="${RAW_RESULT_ROOT}/unused_phase4b"
export RESULT_DIR="${LAUNCHER_DIR}"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1200}"
export XPYD_FIXED_CLOCK_EVIDENCE="${LAUNCHER_DIR}/fixed_clock_evidence.log"

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

export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg-cache" XDG_CONFIG_HOME="${CACHE_ROOT}/xdg-config"
export XDG_DATA_HOME="${CACHE_ROOT}/xdg-data" XDG_STATE_HOME="${CACHE_ROOT}/xdg-state"
export XDG_RUNTIME_DIR="${CACHE_ROOT}/xdg-runtime"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm-cache" VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm-config"
export TORCH_HOME="${CACHE_ROOT}/torch-cache" TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch-extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-cache" TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor-cache"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda-cache" NUMBA_CACHE_DIR="${CACHE_ROOT}/numba-cache"
export RAY_TMPDIR="${CACHE_ROOT}/ray-tmp" PIP_CACHE_DIR="${CACHE_ROOT}/pip-cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache" PYTHONPYCACHEPREFIX="${CACHE_ROOT}/python-pycache"
export TMPDIR="${CACHE_ROOT}/tmp" TMP="${CACHE_ROOT}/tmp" TEMP="${CACHE_ROOT}/tmp"
mkdir -p "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}" "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" \
  "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" \
  "${RAY_TMPDIR}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

for required in "${PYTHON_BIN}" "${CONFIG}" "${SOURCE_ROOT}/run_disagg_benchmark.sh" \
  "${SOURCE_ROOT}/paper/scripts/xpyd/binary_dvfs_evaluation.py" \
  "${JOB_DIR}/validate_job.py" "${JOB_DIR}/live_clock_monitor.py" \
  "${JOB_DIR}/summarize_binary_dvfs.py"; do
  [ -e "${required}" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }
done
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2; exit 2;
  }
done

"${PYTHON_BIN}" "${JOB_DIR}/validate_job.py" --source-root "${SOURCE_ROOT}" \
  --config "${CONFIG}" --output "${CACHE_ROOT}/preflight.json"
echo "xpyd_job=${JOB_NAME}"
echo "xpyd_stage=4B_fine_grained_binary_feedback_DVFS"
echo "xpyd_raw_cache=${CACHE_ROOT}"
echo "xpyd_git_summary=${GIT_RESULT_DIR}"
echo "xpyd_slo=TTFT_500ms_TPOT_200ms"
echo "xpyd_routes=frozen_round_robin_P0_to_D0_and_P1_to_D1"
echo "xpyd_max_concurrency=4"
echo "xpyd_requests_per_window=16"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"

MONITOR_PIDS=()
stop_live_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    [ -n "${pid}" ] && kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${MONITOR_PIDS[@]:-}"; do
    [ -n "${pid}" ] && wait "${pid}" 2>/dev/null || true
  done
  MONITOR_PIDS=()
}
finalize_job() {
  local rc=$?
  trap - EXIT
  stop_live_monitors
  if [ "${rc}" -ne 0 ]; then
    printf '{"valid":false,"phase":"4B_fine_grained_binary_feedback_DVFS","exit_code":%d,"raw_cache":"%s"}\n' \
      "${rc}" "${CACHE_ROOT}" > "${GIT_RESULT_DIR}/job_failure.json"
  fi
  archive_slurm_logs
  exit "${rc}"
}
trap finalize_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_for_live_monitor() {
  local label="$1" pid="$2" log="$3" elapsed=0
  while ! grep -Fqx "XPYD_LIVE_CLOCK_READY=${label}" "${log}" 2>/dev/null; do
    kill -0 "${pid}" 2>/dev/null || { tail -40 "${log}" >&2 || true; return 1; }
    sleep 1
    elapsed=$((elapsed + 1))
    [ "${elapsed}" -lt 60 ] || { tail -40 "${log}" >&2 || true; return 1; }
  done
}

srun --overlap --nodes=1 --ntasks=1 --nodelist=uranus --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" "${JOB_DIR}/live_clock_monitor.py" \
  --node uranus --endpoint P0:0 --endpoint P1:1 --period-s 0.2 \
  --output "${LIVE_CLOCK_DIR}/uranus.jsonl" >"${LIVE_CLOCK_DIR}/uranus.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")
srun --overlap --nodes=1 --ntasks=1 --nodelist=ganymede --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" "${JOB_DIR}/live_clock_monitor.py" \
  --node ganymede --endpoint D0:0 --endpoint D1:1 --period-s 0.2 \
  --output "${LIVE_CLOCK_DIR}/ganymede.jsonl" >"${LIVE_CLOCK_DIR}/ganymede.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")
wait_for_live_monitor uranus "${MONITOR_PIDS[0]}" "${LIVE_CLOCK_DIR}/uranus.monitor.log"
wait_for_live_monitor ganymede "${MONITOR_PIDS[1]}" "${LIVE_CLOCK_DIR}/ganymede.monitor.log"

cd "${SOURCE_ROOT}"
launcher_rc=0
SLURM_SUBMIT_DIR="${SOURCE_ROOT}" /usr/bin/env bash "${SOURCE_ROOT}/run_disagg_benchmark.sh" || launcher_rc=$?
stop_live_monitors
[ "${launcher_rc}" -eq 0 ] || {
  echo "xpyd_binary_dvfs_launcher_failed_rc=${launcher_rc}" >&2
  exit "${launcher_rc}"
}

for required in "${RUN_DIR}/binary_dvfs_audit.json" \
  "${RUN_DIR}/frequency_grids.json" "${RUN_DIR}/selected_frequencies.json" \
  "${RUN_DIR}/binary_search_decisions.csv" "${RUN_DIR}/window_results.csv" \
  "${RUN_DIR}/raw/actuator_actions.csv" "${LIVE_CLOCK_DIR}/uranus.jsonl" \
  "${LIVE_CLOCK_DIR}/ganymede.jsonl"; do
  [ -s "${required}" ] || { echo "xpyd_expected_result_missing=${required}" >&2; exit 20; }
done

"${PYTHON_BIN}" "${JOB_DIR}/summarize_binary_dvfs.py" \
  --run-dir "${RUN_DIR}" --monitor "${LIVE_CLOCK_DIR}/uranus.jsonl" \
  --monitor "${LIVE_CLOCK_DIR}/ganymede.jsonl" --output-dir "${GIT_RESULT_DIR}"
cp "${CACHE_ROOT}/preflight.json" "${GIT_RESULT_DIR}/preflight.json"
cp "${CONFIG}" "${GIT_RESULT_DIR}/experiment_config.json"
echo "${CACHE_ROOT}" > "${GIT_RESULT_DIR}/RAW_CACHE_LOCATION.txt"
echo "xpyd_binary_dvfs_feedback_complete=true"
