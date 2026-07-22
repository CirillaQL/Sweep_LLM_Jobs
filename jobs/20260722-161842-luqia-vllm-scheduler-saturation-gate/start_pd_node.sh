#!/usr/bin/env bash

set -uo pipefail

MODE="${1:-serve}"
HOST=$(hostname -s)
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE_GPUS%%,*}"

case "$HOST" in
  neptune)
    NODE_GROUP=neptune
    NODE_IP=10.1.0.6
    IFACE=enp160s0f0np0
    ;;
  ganymede)
    NODE_GROUP=ganymede
    NODE_IP=10.1.0.3
    IFACE=eno12399np0
    ;;
  *)
    echo "unsupported_host=${HOST}"
    exit 10
    ;;
esac

PLACEMENT_VALUES=$("$PYTHON_BIN" - "$PLACEMENT_FILE" "$NODE_GROUP" <<'PY'
import json
import sys

placement = json.load(open(sys.argv[1], encoding="utf-8"))
node_group = sys.argv[2]
recommended = placement["recommended"]
for role in ("prefill", "decode"):
    spec = recommended[role]
    if spec["node_group"] == node_group:
        print(role, int(spec.get("rec_freq_mhz", spec["freq_mhz"])), spec["gpu_type"])
        break
else:
    raise SystemExit(f"no scheduled role for node_group={node_group}")
PY
)
read -r ROLE TARGET_FREQ EXPECTED_GPU <<< "$PLACEMENT_VALUES"

case "$ROLE" in
  prefill)
    PEER_IP="$DECODE_IP"
    HTTP_PORT="$PREFILL_HTTP_PORT"
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE=1e1
    ;;
  decode)
    PEER_IP="$PREFILL_IP"
    HTTP_PORT="$DECODE_HTTP_PORT"
    KV_ROLE=kv_consumer
    KV_BUFFER_SIZE=8e9
    ;;
  *)
    echo "unsupported_role=${ROLE:-unset}"
    exit 12
    ;;
esac

echo "host=${HOST} node_group=${NODE_GROUP} role=${ROLE} mode=${MODE}"
echo "scheduled_gpu=${EXPECTED_GPU} scheduled_freq_mhz=${TARGET_FREQ} gpu_id=${GPU_ID}"
echo "node_ip=${NODE_IP} peer_ip=${PEER_IP} interface=${IFACE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ip -brief address show dev "$IFACE" 2>&1 || true
ip route get "$PEER_IP" 2>&1 || true
echo "link_speed_mbps=$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo unknown)"
echo "link_mtu=$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || true

GPU_NAME=$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader 2>/dev/null || true)
case "$EXPECTED_GPU:$GPU_NAME" in
  l4:*L4*|l40s:*L40S*) ;;
  *)
    echo "scheduled_gpu_mismatch=true expected=${EXPECTED_GPU} actual=${GPU_NAME}"
    exit 13
    ;;
esac

if [ "$MODE" = reset ]; then
  echo "parent_reset_step=true host=${HOST} gpu_id=${GPU_ID} rec_freq_mhz=${TARGET_FREQ}"
  echo "reset_command_initial=sudo_nvidia_smi_rgc"
  if ! sudo nvidia-smi -i "$GPU_ID" -rgc; then
    echo "reset_gpu_clock_failed=true host=${HOST} gpu_id=${GPU_ID}"
    exit 15
  fi
  MAX_FREQ=$(nvidia-smi -i "$GPU_ID" --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  RESET_PROBE_FILE="${OUT_DIR}/reset_${HOST}_probe.json"
  RESET_PROBE_RC=0
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$CLOCK_PROBE" \
    --smi-index "$GPU_ID" --seconds 5 --output "$RESET_PROBE_FILE" || RESET_PROBE_RC=$?
  if [ "$RESET_PROBE_RC" -eq 0 ]; then
    AFTER_MIN=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["active_clock_min_mhz"]))' "$RESET_PROBE_FILE")
    AFTER_MAX=$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["active_clock_max_mhz"]))' "$RESET_PROBE_FILE")
    echo "reset_probe_active_min_mhz=${AFTER_MIN} reset_probe_active_max_mhz=${AFTER_MAX} gpu_max_mhz=${MAX_FREQ} previous_rec_freq_mhz=${TARGET_FREQ}"
    if [ "$AFTER_MIN" -le "$TARGET_FREQ" ] && [ "$AFTER_MAX" -ge "$TARGET_FREQ" ]; then
      echo "reset_probe_departed_previous_target=indeterminate reason=default_dvfs_range_includes_previous_target"
    else
      echo "reset_probe_departed_previous_target=true"
    fi
  else
    echo "reset_probe_rc=${RESET_PROBE_RC}"
  fi
  # Active DVFS is not required to reach the advertised hardware maximum after
  # reset; power and thermal limits may keep L4 near 1.2 GHz. Make a second
  # successful -rgc the final GPU control operation before this node exits.
  echo "reset_command_final=sudo_nvidia_smi_rgc"
  if ! sudo nvidia-smi -i "$GPU_ID" -rgc; then
    echo "final_reset_gpu_clock_failed=true host=${HOST} gpu_id=${GPU_ID}"
    exit 18
  fi
  if [ "$RESET_PROBE_RC" -ne 0 ]; then
    echo "reset_gpu_clock_verified=false reason=post_reset_probe_failed"
    exit 19
  fi
  echo "reset_gpu_clock_verified=true verification=double_rgc_final_operation host=${HOST} gpu_id=${GPU_ID}"
  exit 0
fi

if [ "$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || true)" != "100000" ]; then
  echo "required_100gbe_link_missing=true"
  exit 11
fi

if [ "$MODE" = preflight ]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
import socket

for name in ("torch", "vllm", "aiohttp", "msgpack", "zmq"):
    module = importlib.import_module(name)
    print(f"host={socket.gethostname()} import={name} version={getattr(module, '__version__', 'unknown')}")
PY
  exit 0
fi

export VLLM_HOST_IP="$NODE_IP"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="${OUT_DIR}/nccl-${ROLE}-${HOST}-%p.log"
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_DISTRIBUTED_DEBUG=DETAIL

KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_buffer_size\":\"${KV_BUFFER_SIZE}\",\"kv_port\":\"${KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_port\":\"${PROXY_REGISTER_PORT}\",\"http_port\":\"${HTTP_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"
SERVER_LOG="${OUT_DIR}/${ROLE}_server.log"
TELEMETRY_FILE="${OUT_DIR}/${ROLE}_${HOST}_telemetry.csv"
SERVER_PID=""
MONITOR_PID=""
CLOCK_CONTROLLER_PID=""
CLOCK_LOCKED=false

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$CLOCK_CONTROLLER_PID" ] && kill -0 "$CLOCK_CONTROLLER_PID" 2>/dev/null; then
    kill "$CLOCK_CONTROLLER_PID" 2>/dev/null || true
    wait "$CLOCK_CONTROLLER_PID" 2>/dev/null || true
  fi
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [ "$CLOCK_LOCKED" = true ]; then
    echo "reset_gpu_clock gpu_id=${GPU_ID}"
    sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
  fi
  echo "node_server_exit host=${HOST} role=${ROLE} rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor() {
  echo "unix_ts,workload_seq,target_freq_mhz,rx_bytes,tx_bytes,gpu_util_pct,gpu_power_w,gpu_sm_mhz,gpu_memory_used_mib" > "$TELEMETRY_FILE"
  while true; do
    unix_ts=$(date +%s.%N)
    workload_seq=0
    target_freq="$TARGET_FREQ"
    if [ -s "${CLOCK_CONTROL_DIR}/${NODE_GROUP}.request" ]; then
      read -r workload_seq target_freq < "${CLOCK_CONTROL_DIR}/${NODE_GROUP}.request" || true
    fi
    rx=$(cat "/sys/class/net/${IFACE}/statistics/rx_bytes" 2>/dev/null || echo NA)
    tx=$(cat "/sys/class/net/${IFACE}/statistics/tx_bytes" 2>/dev/null || echo NA)
    gpu=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    echo "${unix_ts},${workload_seq},${target_freq},${rx},${tx},${gpu:-NA,NA,NA,NA}" >> "$TELEMETRY_FILE"
    sleep 0.5
  done
}

clock_controller() {
  local request_file="${CLOCK_CONTROL_DIR}/${NODE_GROUP}.request"
  local ack_file="${CLOCK_CONTROL_DIR}/${NODE_GROUP}.ack"
  local last_seq=0
  while true; do
    if [ -s "$request_file" ]; then
      local seq target rc observed probe_file
      read -r seq target < "$request_file" || true
      if [ -n "${seq:-}" ] && [ "$seq" != "$last_seq" ]; then
        rc=0
        observed=NA
        probe_file="${OUT_DIR}/clock_${seq}_${NODE_GROUP}.json"
        echo "dynamic_clock_apply host=${HOST} seq=${seq} target_mhz=${target}"
        if ! sudo nvidia-smi -i "$GPU_ID" -lgc "${target},${target}"; then
          rc=31
        elif ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$CLOCK_PROBE" \
          --smi-index "$GPU_ID" --seconds 2 --output "$probe_file"; then
          rc=32
        else
          observed=$(
            "$PYTHON_BIN" - "$probe_file" "$target" <<'PY'
import json
import sys

path, target_text = sys.argv[1:]
target = int(target_text)
data = json.load(open(path, encoding="utf-8"))
data["target_freq_mhz"] = target
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
mean = float(data["active_clock_mean_mhz"])
print(round(mean))
raise SystemExit(0 if abs(mean - target) <= 90 else 1)
PY
          ) || rc=33
        fi
        # The parent pre-creates this file. Writing it in place avoids a stale
        # negative lookup or directory-entry cache on the shared filesystem.
        # A partial read is harmless because the parent only accepts a complete
        # matching sequence and retries once per second.
        printf '%s %s %s %s\n' "$seq" "$target" "$rc" "$observed" > "$ack_file"
        echo "dynamic_clock_ack host=${HOST} seq=${seq} target_mhz=${target} rc=${rc} observed_active_mean_mhz=${observed}"
        last_seq="$seq"
      fi
    fi
    sleep 1
  done
}

echo "lock_gpu_clock gpu_id=${GPU_ID} target_mhz=${TARGET_FREQ}"
if ! sudo nvidia-smi -i "$GPU_ID" -lgc "${TARGET_FREQ},${TARGET_FREQ}"; then
  echo "lock_gpu_clock_failed=true"
  exit 14
fi
CLOCK_LOCKED=true
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.current.graphics,clocks.max.graphics --format=csv 2>&1 || true

echo "launch_vllm host=${HOST} role=${ROLE} ip=${NODE_IP} http_port=${HTTP_PORT} kv_port=${KV_PORT}"
echo "NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "kv_transfer_config=${KV_CONFIG}"

"$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$HTTP_PORT" \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.82 \
  --kv-transfer-config "$KV_CONFIG" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
monitor &
MONITOR_PID=$!
clock_controller &
CLOCK_CONTROLLER_PID=$!

wait "$SERVER_PID"
exit $?
