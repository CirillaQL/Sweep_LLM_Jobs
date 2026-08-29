#!/usr/bin/env bash
#SBATCH --job-name=xpyd-phase4b-r2-energy
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260829-113708-luqia-xpyd-phase4b-r2-actual-energy/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260829-113708-luqia-xpyd-phase4b-r2-actual-energy/slurm-%j.err

set -euo pipefail

JOB_NAME="20260829-113708-luqia-xpyd-phase4b-r2-actual-energy"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_phase4b_evaluation_uranus_ganymede_r2.json"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
RUN_ID="${XPYD_PHASE4B_RUN_ID_OVERRIDE:-xpyd_phase4b_r2_${SLURM_JOB_ID:-manual}}"
WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
PHASE4B_ROOT="${OUT_DIR}/phase4b_stationary"
LIVE_CLOCK_DIR="${OUT_DIR}/live_clock_monitor"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN EXP=E L40S_NODE=uranus L4_NODE=ganymede
export XPYD_PHASE4B_CONFIG="$CONFIG" XPYD_PHASE4B_RUN_ID="$RUN_ID"
export XPYD_PHASE4B_STAGE=stationary XPYD_PHASE4B_SMOKE=0
export XPYD_PHASE4B_OUTPUT_ROOT="$PHASE4B_ROOT"
export RESULT_DIR="${OUT_DIR}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1200}"
export XPYD_FIXED_CLOCK_EVIDENCE="${RESULT_DIR}/fixed_clock_evidence.log"

LOCAL_MODEL_SNAPSHOT_ROOT="${XPYD_LOCAL_MODEL_ROOT_OVERRIDE:-/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots}"
if [ -n "${MODEL_OVERRIDE:-}" ]; then
  XPYD_LOCAL_MODEL_PATH="$MODEL_OVERRIDE"
else
  XPYD_LOCAL_MODEL_PATH=""
  while IFS= read -r model_snapshot; do
    if [ -s "$model_snapshot/config.json" ] && [ -s "$model_snapshot/tokenizer_config.json" ] && \
       find "$model_snapshot" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then
      XPYD_LOCAL_MODEL_PATH="$model_snapshot"
      break
    fi
  done < <(find "$LOCAL_MODEL_SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
fi
[ -n "$XPYD_LOCAL_MODEL_PATH" ] && [ -d "$XPYD_LOCAL_MODEL_PATH" ] || {
  echo "local_mistral_snapshot_not_found=${LOCAL_MODEL_SNAPSHOT_ROOT}" >&2; exit 2;
}
export XPYD_LOCAL_MODEL_PATH MODEL="$XPYD_LOCAL_MODEL_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export HOME="${WORK_DIR}/home"
export HF_HOME="${WORK_DIR}/huggingface" HF_HUB_CACHE="${WORK_DIR}/huggingface/hub"
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
export TMPDIR="${WORK_DIR}/tmp" TMP="${WORK_DIR}/tmp" TEMP="${WORK_DIR}/tmp"

mkdir -p "$OUT_DIR" "$RESULT_DIR" "$PHASE4B_ROOT" "$LIVE_CLOCK_DIR" \
  "$HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$RAY_TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
chmod 700 "$HOME" "$XDG_RUNTIME_DIR" "$TMPDIR"

for required in "$PYTHON_BIN" "$CONFIG" "$SOURCE_ROOT/run_disagg_benchmark.sh" \
  "$JOB_DIR/validate_job.py" "$JOB_DIR/live_clock_monitor.py" "$JOB_DIR/audit_phase4b_results.py"; do
  [ -e "$required" ] || { echo "required_artifact_missing=${required}" >&2; exit 2; }
done
for command_name in srun scontrol nvidia-smi curl sudo; do
  command -v "$command_name" >/dev/null 2>&1 || { echo "required_command_missing=${command_name}" >&2; exit 2; }
done

"$PYTHON_BIN" "$JOB_DIR/validate_job.py" --source-root "$SOURCE_ROOT" \
  --config "$CONFIG" --output "${OUT_DIR}/preflight.json"
echo "xpyd_job=${JOB_NAME}"
echo "xpyd_stage=4B_stationary_actual_energy"
echo "xpyd_slo=TTFT_500ms_TPOT_200ms"
echo "xpyd_energy_baseline=same_job_STATIC"
echo "xpyd_local_model=${XPYD_LOCAL_MODEL_PATH}"

MONITOR_PIDS=()
stop_live_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${MONITOR_PIDS[@]:-}"; do [ -n "$pid" ] && wait "$pid" 2>/dev/null || true; done
  MONITOR_PIDS=()
}
trap stop_live_monitors EXIT
trap 'stop_live_monitors; exit 130' INT
trap 'stop_live_monitors; exit 143' TERM
wait_for_live_monitor() {
  local label="$1" pid="$2" log="$3" elapsed=0
  while ! grep -Fqx "XPYD_LIVE_CLOCK_READY=${label}" "$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { tail -40 "$log" >&2 || true; return 1; }
    sleep 1; elapsed=$((elapsed + 1))
    [ "$elapsed" -lt 60 ] || { tail -40 "$log" >&2 || true; return 1; }
  done
}

srun --overlap --nodes=1 --ntasks=1 --nodelist=uranus --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" \
  --node uranus --endpoint P0:0 --endpoint P1:1 --period-s 0.2 \
  --output "$LIVE_CLOCK_DIR/uranus.jsonl" >"$LIVE_CLOCK_DIR/uranus.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")
srun --overlap --nodes=1 --ntasks=1 --nodelist=ganymede --gpus-per-node=2 --gpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" "$JOB_DIR/live_clock_monitor.py" \
  --node ganymede --endpoint D0:0 --endpoint D1:1 --period-s 0.2 \
  --output "$LIVE_CLOCK_DIR/ganymede.jsonl" >"$LIVE_CLOCK_DIR/ganymede.monitor.log" 2>&1 &
MONITOR_PIDS+=("$!")
wait_for_live_monitor uranus "${MONITOR_PIDS[0]}" "$LIVE_CLOCK_DIR/uranus.monitor.log"
wait_for_live_monitor ganymede "${MONITOR_PIDS[1]}" "$LIVE_CLOCK_DIR/ganymede.monitor.log"

cd "$SOURCE_ROOT"
launcher_rc=0
SLURM_SUBMIT_DIR="$SOURCE_ROOT" /usr/bin/env bash "$SOURCE_ROOT/run_disagg_benchmark.sh" || launcher_rc=$?
stop_live_monitors
trap - EXIT INT TERM
[ "$launcher_rc" -eq 0 ] || { echo "xpyd_phase4b_launcher_failed_rc=${launcher_rc}" >&2; exit "$launcher_rc"; }

RUN_DIR="${PHASE4B_ROOT}/${RUN_ID}"
for required in "$RUN_DIR/phase4b_audit.json" "$RUN_DIR/phase4b_summary.json" \
  "$RUN_DIR/phase4b_policy_summary.csv" "$RUN_DIR/phase4b_stationary_results.csv" \
  "$RUN_DIR/phase4b_control_actions.csv" "$RUN_DIR/phase4b_feedback_decision_trace.csv" \
  "$RUN_DIR/raw/actuator_actions.csv" "$LIVE_CLOCK_DIR/uranus.jsonl" "$LIVE_CLOCK_DIR/ganymede.jsonl"; do
  [ -s "$required" ] || { echo "xpyd_expected_result_missing=${required}" >&2; exit 20; }
done
"$PYTHON_BIN" "$JOB_DIR/audit_phase4b_results.py" --run-dir "$RUN_DIR" \
  --monitor "$LIVE_CLOCK_DIR/uranus.jsonl" --monitor "$LIVE_CLOCK_DIR/ganymede.jsonl" \
  --output "${OUT_DIR}/phase4b_r2_energy_audit.json"
echo "xpyd_phase4b_r2_stationary_actual_energy_complete=true"
