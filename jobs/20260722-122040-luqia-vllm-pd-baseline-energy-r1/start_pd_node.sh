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
    ROLE=prefill
    PEER_IP="$DECODE_IP"
    HTTP_PORT="$PREFILL_HTTP_PORT"
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE=1e1
    ;;
  ganymede)
    NODE_GROUP=ganymede
    NODE_IP=10.1.0.3
    IFACE=eno12399np0
    ROLE=decode
    PEER_IP="$PREFILL_IP"
    HTTP_PORT="$DECODE_HTTP_PORT"
    KV_ROLE=kv_consumer
    KV_BUFFER_SIZE=8e9
    ;;
  *)
    echo "unsupported_host=${HOST}"
    exit 10
    ;;
esac

echo "host=${HOST} node_group=${NODE_GROUP} role=${ROLE} mode=${MODE} gpu_id=${GPU_ID}"
echo "frequency_policy=default_dvfs_no_lgc"
echo "node_ip=${NODE_IP} peer_ip=${PEER_IP} interface=${IFACE}"
ip -brief address show dev "$IFACE" 2>&1 || true
ip route get "$PEER_IP" 2>&1 || true
echo "link_speed_mbps=$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo unknown)"
echo "link_mtu=$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo unknown)"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || true

GPU_NAME=$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader 2>/dev/null || true)
case "$NODE_GROUP:$GPU_NAME" in
  ganymede:*L4*|neptune:*L40S*) ;;
  *)
    echo "gpu_mismatch=true node_group=${NODE_GROUP} actual=${GPU_NAME}"
    exit 13
    ;;
esac

if [ "$MODE" = default ]; then
  echo "prestart_default_clock_command=sudo_nvidia_smi_rgc"
  sudo nvidia-smi -i "$GPU_ID" -rgc
  rc=$?
  echo "prestart_default_clock_rc=${rc} host=${HOST}"
  exit "$rc"
fi

if [ "$MODE" = reset ]; then
  echo "parent_reset_step=true host=${HOST} gpu_id=${GPU_ID}"
  echo "reset_command_initial=sudo_nvidia_smi_rgc"
  if ! sudo nvidia-smi -i "$GPU_ID" -rgc; then
    echo "reset_gpu_clock_failed=true host=${HOST}"
    exit 15
  fi
  RESET_PROBE_FILE="${OUT_DIR}/reset_${HOST}_probe.json"
  RESET_PROBE_RC=0
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$CLOCK_PROBE" \
    --smi-index "$GPU_ID" --seconds 5 --output "$RESET_PROBE_FILE" || RESET_PROBE_RC=$?
  if [ "$RESET_PROBE_RC" -eq 0 ]; then
    "$PYTHON_BIN" - "$RESET_PROBE_FILE" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    f"reset_probe_active_min_mhz={d['active_clock_min_mhz']} "
    f"reset_probe_active_max_mhz={d['active_clock_max_mhz']} "
    f"reset_probe_active_mean_mhz={d['active_clock_mean_mhz']}"
)
PY
  else
    echo "reset_probe_rc=${RESET_PROBE_RC}"
  fi
  echo "reset_command_final=sudo_nvidia_smi_rgc"
  if ! sudo nvidia-smi -i "$GPU_ID" -rgc; then
    echo "final_reset_gpu_clock_failed=true host=${HOST}"
    exit 18
  fi
  if [ "$RESET_PROBE_RC" -ne 0 ]; then
    echo "reset_gpu_clock_verified=false reason=post_reset_probe_failed"
    exit 19
  fi
  echo "reset_gpu_clock_verified=true verification=double_rgc_final_operation host=${HOST}"
  exit 0
fi

if [ "$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || true)" != 100000 ]; then
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

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  echo "node_cleanup_fallback_rgc=true host=${HOST}"
  sudo nvidia-smi -i "$GPU_ID" -rgc 2>&1 || true
  echo "node_server_exit host=${HOST} role=${ROLE} rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor() {
  echo "unix_ts,seq,phase,workload_id,rx_bytes,tx_bytes,gpu_util_pct,gpu_power_w,gpu_sm_mhz,gpu_memory_used_mib" > "$TELEMETRY_FILE"
  while true; do
    unix_ts=$(date +%s.%N)
    seq=0
    phase=server_start
    workload_id=none
    if [ -s "$PHASE_CONTROL_FILE" ]; then
      read -r seq phase workload_id < "$PHASE_CONTROL_FILE" || true
    fi
    rx=$(cat "/sys/class/net/${IFACE}/statistics/rx_bytes" 2>/dev/null || echo NA)
    tx=$(cat "/sys/class/net/${IFACE}/statistics/tx_bytes" 2>/dev/null || echo NA)
    gpu=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    echo "${unix_ts},${seq},${phase},${workload_id},${rx},${tx},${gpu:-NA,NA,NA,NA}" >> "$TELEMETRY_FILE"
    sleep 0.5
  done
}

echo "launch_vllm host=${HOST} role=${ROLE} default_dvfs=true"
echo "kv_transfer_config=${KV_CONFIG}"
"$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 --port "$HTTP_PORT" --tensor-parallel-size 1 \
  --dtype float16 --enforce-eager --max-model-len 4096 \
  --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.82 --kv-transfer-config "$KV_CONFIG" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
monitor &
MONITOR_PID=$!

wait "$SERVER_PID"
exit $?
