#!/usr/bin/env bash
#SBATCH --job-name=uranus-l40s-lgc-probe
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@60
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-103824-luqia-uranus-l40s-lgc-probe/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-103824-luqia-uranus-l40s-lgc-probe/slurm-%j.err

set -euo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-103824-luqia-uranus-l40s-lgc-probe"
PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
PROBE_SCRIPT="${JOB_DIR}/gpu_clock_probe.py"
TARGET_MHZ=990
TOLERANCE_MHZ=15
RESTORE_DEPARTURE_MHZ=90

LOCK_COMMAND_RC=not_run
LOCK_EFFECTIVE=false
INITIAL_RGC_RC=not_run
FIRST_RESTORE_RGC_RC=not_run
POST_RESET_PROBE_RC=not_run
FINAL_RGC_RC=not_run
BASELINE_MIN_MHZ=NA
BASELINE_MAX_MHZ=NA
BASELINE_MEAN_MHZ=NA
LOCKED_MIN_MHZ=NA
LOCKED_MAX_MHZ=NA
LOCKED_MEAN_MHZ=NA
POST_RESET_MIN_MHZ=NA
POST_RESET_MAX_MHZ=NA
POST_RESET_MEAN_MHZ=NA
RESTORE_VERIFIED=false

section() {
  echo
  echo "===== $1 ====="
}

probe_values() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    int(round(data["active_clock_min_mhz"])),
    int(round(data["active_clock_max_mhz"])),
    round(data["active_clock_mean_mhz"], 3),
)
PY
}

cleanup() {
  local original_rc=$?
  trap - EXIT INT TERM
  section restore_default_clock

  FIRST_RESTORE_RGC_RC=0
  sudo -n nvidia-smi -i "$GPU_SELECTOR" -rgc || FIRST_RESTORE_RGC_RC=$?
  echo "first_restore_rgc_rc=${FIRST_RESTORE_RGC_RC}"

  POST_RESET_PROBE_RC=0
  "$PYTHON_BIN" "$PROBE_SCRIPT" \
    --smi-index "$GPU_INDEX" --seconds 6 \
    --output "${JOB_DIR}/post_reset_probe.json" || POST_RESET_PROBE_RC=$?
  if [ "$POST_RESET_PROBE_RC" -eq 0 ]; then
    read -r POST_RESET_MIN_MHZ POST_RESET_MAX_MHZ POST_RESET_MEAN_MHZ \
      <<< "$(probe_values "${JOB_DIR}/post_reset_probe.json")"
  fi
  echo "post_reset_probe_rc=${POST_RESET_PROBE_RC} active_min_mhz=${POST_RESET_MIN_MHZ} active_max_mhz=${POST_RESET_MAX_MHZ} active_mean_mhz=${POST_RESET_MEAN_MHZ}"

  FINAL_RGC_RC=0
  sudo -n nvidia-smi -i "$GPU_SELECTOR" -rgc || FINAL_RGC_RC=$?
  echo "final_rgc_rc=${FINAL_RGC_RC} final_gpu_control_operation=rgc"

  if [ "$FIRST_RESTORE_RGC_RC" -eq 0 ] && \
    [ "$POST_RESET_PROBE_RC" -eq 0 ] && \
    [ "$FINAL_RGC_RC" -eq 0 ] && \
    [ "$POST_RESET_MAX_MHZ" != NA ] && \
    [ "$POST_RESET_MAX_MHZ" -gt "$((TARGET_MHZ + RESTORE_DEPARTURE_MHZ))" ]; then
    RESTORE_VERIFIED=true
  fi

  {
    echo "host=$(hostname -s)"
    echo "gpu_selector=${GPU_SELECTOR}"
    echo "gpu_index=${GPU_INDEX}"
    echo "target_mhz=${TARGET_MHZ}"
    echo "initial_rgc_rc=${INITIAL_RGC_RC}"
    echo "baseline_active_min_mhz=${BASELINE_MIN_MHZ}"
    echo "baseline_active_max_mhz=${BASELINE_MAX_MHZ}"
    echo "baseline_active_mean_mhz=${BASELINE_MEAN_MHZ}"
    echo "lock_command_rc=${LOCK_COMMAND_RC}"
    echo "lock_effective=${LOCK_EFFECTIVE}"
    echo "locked_active_min_mhz=${LOCKED_MIN_MHZ}"
    echo "locked_active_max_mhz=${LOCKED_MAX_MHZ}"
    echo "locked_active_mean_mhz=${LOCKED_MEAN_MHZ}"
    echo "first_restore_rgc_rc=${FIRST_RESTORE_RGC_RC}"
    echo "post_reset_probe_rc=${POST_RESET_PROBE_RC}"
    echo "post_reset_active_min_mhz=${POST_RESET_MIN_MHZ}"
    echo "post_reset_active_max_mhz=${POST_RESET_MAX_MHZ}"
    echo "post_reset_active_mean_mhz=${POST_RESET_MEAN_MHZ}"
    echo "final_rgc_rc=${FINAL_RGC_RC}"
    echo "restore_verified=${RESTORE_VERIFIED}"
  } | tee "${JOB_DIR}/clock_test_summary.txt"

  if [ "$RESTORE_VERIFIED" != true ] && [ "$original_rc" -eq 0 ]; then
    original_rc=30
  fi
  echo "job_exit_rc=${original_rc}"
  exit "$original_rc"
}

section allocation
echo "timestamp=$(date --iso-8601=seconds)"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} host=$(hostname -s)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [ "$(hostname -s)" != uranus ]; then
  echo "expected_host=uranus actual=$(hostname -s)"
  exit 10
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -s "$PROBE_SCRIPT" ]; then
  echo "probe_environment_missing=true"
  exit 11
fi

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$VISIBLE_GPUS"
if [ -z "$VISIBLE_GPUS" ] || [ "${#VISIBLE_GPU_ARRAY[@]}" -ne 1 ]; then
  echo "expected_exactly_one_visible_gpu=true visible=${VISIBLE_GPUS:-unset}"
  exit 12
fi
GPU_SELECTOR="${VISIBLE_GPU_ARRAY[0]}"
if [[ "$GPU_SELECTOR" =~ ^[0-9]+$ ]]; then
  GPU_INDEX="$GPU_SELECTOR"
else
  GPU_INDEX="$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=index --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
fi
echo "gpu_selector=${GPU_SELECTOR} gpu_index=${GPU_INDEX}"

# Install cleanup immediately before the first clock-control command. At this
# point the exact allocated GPU has been resolved, so every later exit can
# safely restore that GPU without touching any other device.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi -i "$GPU_SELECTOR" \
  --query-gpu=index,name,uuid,power.draw,clocks.current.graphics,clocks.max.graphics,pstate \
  --format=csv

section establish_default_baseline
INITIAL_RGC_RC=0
sudo -n nvidia-smi -i "$GPU_SELECTOR" -rgc || INITIAL_RGC_RC=$?
echo "initial_rgc_rc=${INITIAL_RGC_RC}"
if [ "$INITIAL_RGC_RC" -ne 0 ]; then
  exit 13
fi
"$PYTHON_BIN" "$PROBE_SCRIPT" \
  --smi-index "$GPU_INDEX" --seconds 6 \
  --output "${JOB_DIR}/baseline_probe.json"
read -r BASELINE_MIN_MHZ BASELINE_MAX_MHZ BASELINE_MEAN_MHZ \
  <<< "$(probe_values "${JOB_DIR}/baseline_probe.json")"
echo "baseline_active_min_mhz=${BASELINE_MIN_MHZ} baseline_active_max_mhz=${BASELINE_MAX_MHZ} baseline_active_mean_mhz=${BASELINE_MEAN_MHZ}"

section apply_990mhz_graphics_clock_lock
LOCK_COMMAND_RC=0
sudo -n nvidia-smi -i "$GPU_SELECTOR" -lgc "${TARGET_MHZ},${TARGET_MHZ}" || LOCK_COMMAND_RC=$?
echo "lock_command_rc=${LOCK_COMMAND_RC} target_mhz=${TARGET_MHZ}"
if [ "$LOCK_COMMAND_RC" -ne 0 ]; then
  exit 20
fi
nvidia-smi -i "$GPU_SELECTOR" \
  --query-gpu=index,name,clocks.current.graphics,clocks.max.graphics,pstate \
  --format=csv

section verify_locked_clock_under_cuda_load
"$PYTHON_BIN" "$PROBE_SCRIPT" \
  --smi-index "$GPU_INDEX" --seconds 8 \
  --output "${JOB_DIR}/locked_probe.json"
read -r LOCKED_MIN_MHZ LOCKED_MAX_MHZ LOCKED_MEAN_MHZ \
  <<< "$(probe_values "${JOB_DIR}/locked_probe.json")"
echo "locked_active_min_mhz=${LOCKED_MIN_MHZ} locked_active_max_mhz=${LOCKED_MAX_MHZ} locked_active_mean_mhz=${LOCKED_MEAN_MHZ}"
if [ "$LOCKED_MIN_MHZ" -ge "$((TARGET_MHZ - TOLERANCE_MHZ))" ] && \
  [ "$LOCKED_MAX_MHZ" -le "$((TARGET_MHZ + TOLERANCE_MHZ))" ]; then
  LOCK_EFFECTIVE=true
fi
echo "lock_effective=${LOCK_EFFECTIVE} tolerance_mhz=${TOLERANCE_MHZ}"
if [ "$LOCK_EFFECTIVE" != true ]; then
  exit 21
fi

exit 0
