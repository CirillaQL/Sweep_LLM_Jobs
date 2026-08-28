#!/usr/bin/env bash
#SBATCH --job-name=xpyd-phase3d-a-uranus-ganymede-r5
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260828-110749-luqia-xpyd-phase3d-a-uranus-ganymede-r5/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260828-110749-luqia-xpyd-phase3d-a-uranus-ganymede-r5/slurm-%j.err

set -euo pipefail

JOB_NAME="20260828-110749-luqia-xpyd-phase3d-a-uranus-ganymede-r5"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_phase3d_2p2d_l40s_l4.json"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
RUN_ID="${XPYD_PHASE3D_RUN_ID_OVERRIDE:-xpyd_phase3d_a_${SLURM_JOB_ID:-manual}}"
WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
ACTUATOR_ROOT="${OUT_DIR}/phase3d_actuator_validation"
WINDOW_ROOT="${OUT_DIR}/phase3d_window_scratch"
CLOSED_LOOP_ROOT="${OUT_DIR}/phase3d_closed_loop_unused"
LIVE_CLOCK_DIR="${OUT_DIR}/live_clock_monitor"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN
export EXP=E
export L40S_NODE=uranus
export L4_NODE=ganymede
export XPYD_PHASE3D_CONFIG="$CONFIG"
export XPYD_PHASE3D_STAGE=A
export XPYD_PHASE3D_RUN_ID="$RUN_ID"
export XPYD_PHASE3D_ACTUATOR_OUTPUT_ROOT="$ACTUATOR_ROOT"
export XPYD_PHASE3D_WINDOW_OUTPUT_ROOT="$WINDOW_ROOT"
export XPYD_PHASE3D_CLOSED_LOOP_OUTPUT_ROOT="$CLOSED_LOOP_ROOT"
export RESULT_DIR="${OUT_DIR}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1200}"
export XPYD_FIXED_CLOCK_EVIDENCE="${RESULT_DIR}/fixed_clock_evidence.log"

# Reuse the local Mistral snapshot proven by successful Phase 3C Job 255143.
# An explicit override remains available if the shared cache is relocated.
LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then
  XPYD_LOCAL_MODEL_PATH="$MODEL_OVERRIDE"
else
  mapfile -t model_snapshots < <(
    find "$LOCAL_MODEL_SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
  )
  XPYD_LOCAL_MODEL_PATH=""
  for model_snapshot in "${model_snapshots[@]:-}"; do
    model_weight_found=0
    for model_weight in "$model_snapshot"/*.safetensors; do
      if [ -s "$model_weight" ]; then
        model_weight_found=1
        break
      fi
    done
    if [ -s "$model_snapshot/config.json" ] && \
       [ -s "$model_snapshot/tokenizer_config.json" ] && \
       [ "$model_weight_found" -eq 1 ]; then
      XPYD_LOCAL_MODEL_PATH="$model_snapshot"
      break
    fi
  done
fi
if [ -z "$XPYD_LOCAL_MODEL_PATH" ] || [ ! -d "$XPYD_LOCAL_MODEL_PATH" ]; then
  echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2
  exit 2
fi
export XPYD_LOCAL_MODEL_PATH
export MODEL="$XPYD_LOCAL_MODEL_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The privileged broker can pass HOME=/root into the Slurm environment.  Keep
# every runtime cache in a writable, job-isolated directory owned by chjing.
export HOME="${WORK_DIR}/home"
export HF_HOME="${WORK_DIR}/huggingface"
export HF_TOKEN_PATH="${HF_HOME}/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export XDG_CACHE_HOME="${WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${WORK_DIR}/xdg-data"
export XDG_STATE_HOME="${WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${WORK_DIR}/xdg-runtime"
export FLASHINFER_WORKSPACE_BASE="${WORK_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${WORK_DIR}/vllm-cache"
export VLLM_CONFIG_ROOT="${WORK_DIR}/vllm-config"
export TORCH_HOME="${WORK_DIR}/torch-cache"
export TORCH_EXTENSIONS_DIR="${WORK_DIR}/torch-extensions"
export TRITON_CACHE_DIR="${WORK_DIR}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK_DIR}/torchinductor-cache"
export CUDA_CACHE_PATH="${WORK_DIR}/cuda-cache"
export NUMBA_CACHE_DIR="${WORK_DIR}/numba-cache"
export RAY_TMPDIR="${WORK_DIR}/ray-tmp"
export PIP_CACHE_DIR="${WORK_DIR}/pip-cache"
export UV_CACHE_DIR="${WORK_DIR}/uv-cache"
export PYTHONPYCACHEPREFIX="${WORK_DIR}/python-pycache"
export TIKTOKEN_CACHE_DIR="${WORK_DIR}/tiktoken-cache"
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
export CUPY_CACHE_DIR="${WORK_DIR}/cupy-cache"
export TMPDIR="${WORK_DIR}/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p \
  "$OUT_DIR" "$RESULT_DIR" "$ACTUATOR_ROOT" "$WINDOW_ROOT" "$LIVE_CLOCK_DIR" \
  "$HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE" \
  "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$RAY_TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" \
  "$CUPY_CACHE_DIR" "$TMPDIR"
chmod 700 "$HOME" "$XDG_RUNTIME_DIR" "$TMPDIR"

for required in "$PYTHON_BIN" "$CONFIG" "$SOURCE_ROOT/run_disagg_benchmark.sh" \
  "$SOURCE_ROOT/gpu_monitor.py" "$JOB_DIR/validate_job.py" \
  "$JOB_DIR/live_clock_monitor.py" "$JOB_DIR/audit_live_clocks.py"; do
  [ -e "$required" ] || {
    echo "required_artifact_missing=${required}" >&2
    exit 2
  }
done
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2
    exit 2
  }
done

"$PYTHON_BIN" "$JOB_DIR/validate_job.py" \
  --source-root "$SOURCE_ROOT" \
  --config "$CONFIG" \
  --output "${OUT_DIR}/preflight.json"

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_source_root=${SOURCE_ROOT}"
echo "xpyd_run_id=${RUN_ID}"
echo "xpyd_stage=3D-A"
echo "xpyd_actuator_output_root=${ACTUATOR_ROOT}"
echo "xpyd_launcher_result_dir=${RESULT_DIR}"
echo "xpyd_nodes=${SLURM_JOB_NODELIST:-unset}"
echo "xpyd_work_dir=${WORK_DIR}"
echo "xpyd_hf_cache=${HF_HUB_CACHE}"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"

MONITOR_PIDS=()
stop_live_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${MONITOR_PIDS[@]:-}"; do
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
  done
  MONITOR_PIDS=()
}
trap stop_live_monitors EXIT
trap 'stop_live_monitors; exit 130' INT
trap 'stop_live_monitors; exit 143' TERM

wait_for_live_monitor() {
  local label="$1"
  local pid="$2"
  local log="$3"
  local elapsed=0
  local ready_marker="XPYD_LIVE_CLOCK_READY=${label}"
  while ! grep -Fqx "$ready_marker" "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "live_clock_monitor_exited_before_ready=${label} pid=${pid}" >&2
      tail -40 "$log" >&2 || true
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge 60 ]; then
      echo "live_clock_monitor_readiness_timeout=${label} pid=${pid}" >&2
      tail -40 "$log" >&2 || true
      return 1
    fi
  done
  echo "live_clock_monitor_ready=${label} marker=${ready_marker}"
}

srun --overlap --nodes=1 --ntasks=1 --nodelist=uranus \
  --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" \
    --node uranus --endpoint P0:0 --endpoint P1:1 \
    --period-s 0.2 --output "$LIVE_CLOCK_DIR/uranus.jsonl" \
    >"$LIVE_CLOCK_DIR/uranus.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")
srun --overlap --nodes=1 --ntasks=1 --nodelist=ganymede \
  --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" \
    --node ganymede --endpoint D0:0 --endpoint D1:1 \
    --period-s 0.2 --output "$LIVE_CLOCK_DIR/ganymede.jsonl" \
    >"$LIVE_CLOCK_DIR/ganymede.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")

wait_for_live_monitor uranus "${MONITOR_PIDS[0]}" \
  "$LIVE_CLOCK_DIR/uranus.monitor.log" || exit 3
wait_for_live_monitor ganymede "${MONITOR_PIDS[1]}" \
  "$LIVE_CLOCK_DIR/ganymede.monitor.log" || exit 3

cd "$SOURCE_ROOT"
launcher_rc=0
SLURM_SUBMIT_DIR="$SOURCE_ROOT" \
  /usr/bin/env bash "$SOURCE_ROOT/run_disagg_benchmark.sh" || launcher_rc=$?
stop_live_monitors
trap - EXIT INT TERM

if [ "$launcher_rc" -ne 0 ]; then
  echo "xpyd_phase3d_launcher_failed_rc=${launcher_rc}" >&2
  exit "$launcher_rc"
fi

RUN_DIR="${ACTUATOR_ROOT}/${RUN_ID}"
AUDIT="${RUN_DIR}/actuator_audit.json"
SUMMARY="${RUN_DIR}/actuator_summary.md"
for required in "$AUDIT" "$SUMMARY" "$LIVE_CLOCK_DIR/uranus.jsonl" \
  "$LIVE_CLOCK_DIR/ganymede.jsonl" "$XPYD_FIXED_CLOCK_EVIDENCE"; do
  [ -s "$required" ] || {
    echo "xpyd_expected_result_missing=${required}" >&2
    exit 20
  }
done

"$PYTHON_BIN" "$JOB_DIR/audit_live_clocks.py" \
  --run-dir "$RUN_DIR" \
  --config "$CONFIG" \
  --monitor "$LIVE_CLOCK_DIR/uranus.jsonl" \
  --monitor "$LIVE_CLOCK_DIR/ganymede.jsonl" \
  --fixed-clock-evidence "$XPYD_FIXED_CLOCK_EVIDENCE" \
  --output "${OUT_DIR}/phase3d_a_live_clock_audit.json"

echo "xpyd_phase3d_a_live_clock_validation_complete=true"
