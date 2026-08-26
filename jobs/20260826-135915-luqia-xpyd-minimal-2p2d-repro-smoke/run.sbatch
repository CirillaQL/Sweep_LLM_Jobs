#!/usr/bin/env bash
#SBATCH --job-name=xpyd-phase3c-2p2d-repro
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,io
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260826-135915-luqia-xpyd-minimal-2p2d-repro-smoke/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260826-135915-luqia-xpyd-minimal-2p2d-repro-smoke/slurm-%j.err

set -euo pipefail

JOB_NAME="20260826-135915-luqia-xpyd-minimal-2p2d-repro-smoke"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_ROOT="${JOB_DIR}/source"
ENV_DIR="${XPYD_ENV_DIR_OVERRIDE:-/data/users/chjing/miniforge3/envs/cuda-env}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${ENV_DIR}/bin/python}"
CONFIG="${SOURCE_ROOT}/paper/configs/xpyd_phase3c_2p2d_l40s_l4.json"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
RUN_ID="${XPYD_PHASE3C_RUN_ID_OVERRIDE:-xpyd_phase3c_repro_${SLURM_JOB_ID:-manual}}"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN
export EXP=E
export MODEL="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
export L40S_NODE=neptune
export L4_NODE=io
export XPYD_PHASE3C_CONFIG="$CONFIG"
export XPYD_PHASE3C_RUN_ID="$RUN_ID"
export XPYD_PHASE3C_OUTPUT_ROOT="${OUT_DIR}/xpyd_phase3c_substrate"
export RESULT_DIR="${OUT_DIR}/launcher"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S_OVERRIDE:-1200}"

mkdir -p "$OUT_DIR" "$RESULT_DIR" "$XPYD_PHASE3C_OUTPUT_ROOT"

for required in "$PYTHON_BIN" "$CONFIG" "$SOURCE_ROOT/run_disagg_benchmark.sh" \
  "$SOURCE_ROOT/gpu_monitor.py" "$JOB_DIR/validate_job.py"; do
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
echo "xpyd_output_root=${XPYD_PHASE3C_OUTPUT_ROOT}"
echo "xpyd_launcher_result_dir=${RESULT_DIR}"
echo "xpyd_nodes=${SLURM_JOB_NODELIST:-unset}"

cd "$SOURCE_ROOT"
SLURM_SUBMIT_DIR="$SOURCE_ROOT" \
  /usr/bin/env bash "$SOURCE_ROOT/run_disagg_benchmark.sh"

AUDIT="${XPYD_PHASE3C_OUTPUT_ROOT}/${RUN_ID}/audit.json"
SUMMARY="${XPYD_PHASE3C_OUTPUT_ROOT}/${RUN_ID}/summary.json"
for required in "$AUDIT" "$SUMMARY"; do
  [ -s "$required" ] || {
    echo "xpyd_expected_result_missing=${required}" >&2
    exit 20
  }
done

"$PYTHON_BIN" - "$AUDIT" <<'PY'
import json
import sys

path = sys.argv[1]
audit = json.load(open(path, encoding="utf-8"))
accepted = audit.get("valid")
print("xpyd_phase3c_audit_accepted=" + str(bool(accepted)).lower())
if not accepted:
    print(json.dumps(audit, indent=2, sort_keys=True))
    raise SystemExit(21)
PY

echo "xpyd_phase3c_reproduction_complete=true"
