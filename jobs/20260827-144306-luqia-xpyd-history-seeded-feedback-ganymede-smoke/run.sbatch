#!/usr/bin/env bash
#SBATCH --job-name=xpyd-history-ganymede
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260827-144306-luqia-xpyd-history-seeded-feedback-ganymede-smoke/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260827-144306-luqia-xpyd-history-seeded-feedback-ganymede-smoke/slurm-%j.err

set -euo pipefail

JOB_NAME="20260827-144306-luqia-xpyd-history-seeded-feedback-ganymede-smoke"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_history_seeded_control_neptune_ganymede.json"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
RUN_ID="${XPYD_HISTORY_RUN_ID_OVERRIDE:-history_seeded_${SLURM_JOB_ID:-manual}}"
WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
HISTORY_ROOT="${OUT_DIR}/history_seeded_closed_loop"
WINDOW_ROOT="${OUT_DIR}/history_internal_windows"
RESULT_DIR="${OUT_DIR}/launcher"
LIVE_CLOCK_DIR="${OUT_DIR}/live_clock_monitor"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN
export EXP=E
export L40S_NODE=neptune
export L4_NODE=ganymede
export XPYD_PHASE3D_CONFIG="$CONFIG"
export XPYD_PHASE3D_STAGE=B
export XPYD_PHASE3D_RUN_ID="$RUN_ID"
export XPYD_PHASE3D_RUN_ID_OVERRIDE="$RUN_ID"
export XPYD_HISTORY_OUTPUT_ROOT="$HISTORY_ROOT"
export XPYD_PHASE3D_WINDOW_OUTPUT_ROOT="$WINDOW_ROOT"
export XPYD_PHASE3D_CLOSED_LOOP_OUTPUT_ROOT="$HISTORY_ROOT"
export RESULT_DIR
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1200}"
export XPYD_FIXED_CLOCK_EVIDENCE="${RESULT_DIR}/fixed_clock_evidence.log"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then
  XPYD_LOCAL_MODEL_PATH="$MODEL_OVERRIDE"
else
  XPYD_LOCAL_MODEL_PATH=""
  while IFS= read -r model_snapshot; do
    if [ -s "$model_snapshot/config.json" ] && \
       [ -s "$model_snapshot/tokenizer_config.json" ] && \
       find "$model_snapshot" -maxdepth 1 -name '*.safetensors' -type f -size +0c -print -quit | grep -q .; then
      XPYD_LOCAL_MODEL_PATH="$model_snapshot"
      break
    fi
  done < <(find "$LOCAL_MODEL_SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
fi
if [ -z "$XPYD_LOCAL_MODEL_PATH" ] || [ ! -d "$XPYD_LOCAL_MODEL_PATH" ]; then
  echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2
  exit 2
fi
export XPYD_LOCAL_MODEL_PATH MODEL="$XPYD_LOCAL_MODEL_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

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
export TMP="$TMPDIR" TEMP="$TMPDIR"

mkdir -p "$OUT_DIR" "$HISTORY_ROOT" "$WINDOW_ROOT" "$RESULT_DIR" \
  "$LIVE_CLOCK_DIR" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE" \
  "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$XDG_RUNTIME_DIR" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" \
  "$VLLM_CONFIG_ROOT" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$NUMBA_CACHE_DIR" "$RAY_TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" \
  "$CUPY_CACHE_DIR" "$TMPDIR"
chmod 700 "$XDG_RUNTIME_DIR" "$TMPDIR"

for required in "$PYTHON_BIN" "$CONFIG" "$SOURCE_ROOT/run_disagg_benchmark.sh" \
  "$JOB_DIR/resolve_evidence.py" "$JOB_DIR/validate_job.py" \
  "$JOB_DIR/live_clock_monitor.py" "$JOB_DIR/audit_history_run.py"; do
  [ -e "$required" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }
done
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2; exit 2;
  }
done

actuator_args=(--kind actuator --root "$BROKER_ROOT/jobs")
[ -z "${XPYD_ACTUATOR_AUDIT_OVERRIDE:-}" ] || actuator_args+=(--override "$XPYD_ACTUATOR_AUDIT_OVERRIDE")
history_args=(--kind history --root "$BROKER_ROOT/jobs")
DEFAULT_HISTORY_SUMMARY="/data/users/chjing/vLLM_test/results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json"
if [ -n "${XPYD_HISTORY_SUMMARY_OVERRIDE:-}" ]; then
  history_args+=(--override "$XPYD_HISTORY_SUMMARY_OVERRIDE")
elif [ -s "$DEFAULT_HISTORY_SUMMARY" ]; then
  history_args+=(--override "$DEFAULT_HISTORY_SUMMARY")
fi
XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT="$($PYTHON_BIN "$JOB_DIR/resolve_evidence.py" "${actuator_args[@]}")"
XPYD_HISTORY_SUMMARY="$($PYTHON_BIN "$JOB_DIR/resolve_evidence.py" "${history_args[@]}")"
export XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT XPYD_HISTORY_SUMMARY

"$PYTHON_BIN" "$JOB_DIR/validate_job.py" --source-root "$SOURCE_ROOT" \
  --config "$CONFIG" --actuator-audit "$XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT" \
  --history-summary "$XPYD_HISTORY_SUMMARY" --output "$OUT_DIR/preflight.json"

echo "xpyd_job=${JOB_NAME}"
echo "xpyd_run_id=${RUN_ID}"
echo "xpyd_controller=history_seeded_plus_online_feedback"
echo "xpyd_history_summary=${XPYD_HISTORY_SUMMARY}"
echo "xpyd_actuator_audit=${XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT}"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"
echo "xpyd_cache_root=${WORK_DIR}"

MONITOR_PIDS=()
stop_live_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "${MONITOR_PIDS[@]:-}"; do [ -n "$pid" ] && wait "$pid" 2>/dev/null || true; done
  MONITOR_PIDS=()
}
trap stop_live_monitors EXIT
trap 'stop_live_monitors; exit 130' INT
trap 'stop_live_monitors; exit 143' TERM

wait_for_monitor() {
  local label="$1" pid="$2" output="$3" log="$4" elapsed=0
  while [ ! -s "$output" ]; do
    kill -0 "$pid" 2>/dev/null || { tail -40 "$log" >&2 || true; return 1; }
    sleep 1
    elapsed=$((elapsed + 1))
    [ "$elapsed" -lt 60 ] || { tail -40 "$log" >&2 || true; return 1; }
  done
  "$PYTHON_BIN" - "$label" "$output" <<'PY'
import json, sys
label, path = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    row = json.loads(stream.readline())
if row.get("status") != "success" or row.get("node") != label:
    raise SystemExit(f"invalid monitor sample: {row}")
PY
}

srun --overlap --nodes=1 --ntasks=1 --nodelist=neptune \
  --gpus-per-node=2 --gpu-bind=none env CUDA_VISIBLE_DEVICES=0,1 \
  "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" --node neptune \
  --endpoint P0:0 --endpoint P1:1 --period-s 0.2 \
  --output "$LIVE_CLOCK_DIR/neptune.jsonl" >"$LIVE_CLOCK_DIR/neptune.log" 2>&1 &
MONITOR_PIDS+=("$!")
srun --overlap --nodes=1 --ntasks=1 --nodelist=ganymede \
  --gpus-per-node=2 --gpu-bind=none env CUDA_VISIBLE_DEVICES=0,1 \
  "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" --node ganymede \
  --endpoint D0:0 --endpoint D1:1 --period-s 0.2 \
  --output "$LIVE_CLOCK_DIR/ganymede.jsonl" >"$LIVE_CLOCK_DIR/ganymede.log" 2>&1 &
MONITOR_PIDS+=("$!")
wait_for_monitor neptune "${MONITOR_PIDS[0]}" "$LIVE_CLOCK_DIR/neptune.jsonl" "$LIVE_CLOCK_DIR/neptune.log"
wait_for_monitor ganymede "${MONITOR_PIDS[1]}" "$LIVE_CLOCK_DIR/ganymede.jsonl" "$LIVE_CLOCK_DIR/ganymede.log"

cd "$SOURCE_ROOT"
launcher_rc=0
SLURM_SUBMIT_DIR="$SOURCE_ROOT" /usr/bin/env bash "$SOURCE_ROOT/run_disagg_benchmark.sh" || launcher_rc=$?
stop_live_monitors
trap - EXIT INT TERM
[ "$launcher_rc" -eq 0 ] || { echo "history_controller_launcher_failed_rc=${launcher_rc}" >&2; exit "$launcher_rc"; }

RUN_DIR="${HISTORY_ROOT}/${RUN_ID}"
for required in "$RUN_DIR/history_control_audit.json" \
  "$RUN_DIR/history_decisions.csv" "$RUN_DIR/history_actual_outcomes.csv" \
  "$RUN_DIR/raw/actuator_actions.csv" "$LIVE_CLOCK_DIR/neptune.jsonl" \
  "$LIVE_CLOCK_DIR/ganymede.jsonl"; do
  [ -s "$required" ] || { echo "expected_result_missing=${required}" >&2; exit 20; }
done
"$PYTHON_BIN" "$JOB_DIR/audit_history_run.py" --run-dir "$RUN_DIR" \
  --monitor "$LIVE_CLOCK_DIR/neptune.jsonl" --monitor "$LIVE_CLOCK_DIR/ganymede.jsonl" \
  --output "$OUT_DIR/history_live_validation.json"
echo "xpyd_history_seeded_feedback_validation_complete=true"
