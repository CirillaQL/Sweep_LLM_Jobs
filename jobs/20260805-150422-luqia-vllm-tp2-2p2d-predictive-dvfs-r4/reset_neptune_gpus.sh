#!/usr/bin/env bash

set -uo pipefail

PHASE="${1:-unknown}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"

echo "gpu_full_reset_start phase=${PHASE} host=$(hostname -s) visible_gpus=${VISIBLE_GPUS} time=$(date --iso-8601=seconds)"
echo "gpu_process_snapshot_before=true"
timeout 20 nvidia-smi -i "$VISIBLE_GPUS" \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv 2>&1 || true

RESET_RC=0
timeout 120 sudo nvidia-smi --gpu-reset -i "$VISIBLE_GPUS" || RESET_RC=$?
echo "gpu_full_reset_command_rc=${RESET_RC} phase=${PHASE}"
if [ "$RESET_RC" -ne 0 ]; then
  exit "$RESET_RC"
fi

sleep 5
nvidia-smi -i "$VISIBLE_GPUS" \
  --query-gpu=index,pci.bus_id,name,uuid,clocks.current.graphics,clocks.max.graphics,memory.used \
  --format=csv

CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" "$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"expected 4 CUDA devices after reset, found {count}")
for index in range(count):
    with torch.cuda.device(index):
        left = torch.ones((256, 256), device=f"cuda:{index}")
        right = left @ left
        torch.cuda.synchronize(index)
        value = float(right[0, 0].item())
        if value != 256.0:
            raise SystemExit(f"unexpected CUDA result gpu={index} value={value}")
        print(f"cuda_reset_validation_ok gpu={index} value={value}")
PY

echo "gpu_full_reset_verified=true phase=${PHASE} host=$(hostname -s) time=$(date --iso-8601=seconds)"
