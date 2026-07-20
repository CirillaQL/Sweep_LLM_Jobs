#!/usr/bin/env bash

set -uo pipefail

MODE="${1:-serve}"
HOST=$(hostname -s)

case "$HOST" in
  neptune)
    ROLE=prefill
    NODE_IP="$PREFILL_IP"
    PEER_IP="$DECODE_IP"
    IFACE=enp160s0f0np0
    HTTP_PORT="$PREFILL_HTTP_PORT"
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE=1e1
    ;;
  ganymede)
    ROLE=decode
    NODE_IP="$DECODE_IP"
    PEER_IP="$PREFILL_IP"
    IFACE=eno12399np0
    HTTP_PORT="$DECODE_HTTP_PORT"
    KV_ROLE=kv_consumer
    KV_BUFFER_SIZE=8e9
    ;;
  *)
    echo "unsupported_host=${HOST}"
    exit 10
    ;;
esac

echo "host=${HOST} role=${ROLE} mode=${MODE}"
echo "node_ip=${NODE_IP} peer_ip=${PEER_IP} interface=${IFACE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ip -brief address show dev "$IFACE" 2>&1 || true
ip route get "$PEER_IP" 2>&1 || true
echo "link_speed_mbps=$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo unknown)"
echo "link_mtu=$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || true

if [ "$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || true)" != "100000" ]; then
  echo "required_100gbe_link_missing=true"
  exit 11
fi

if [ "$MODE" = preflight ]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
import socket

for name in ("torch", "vllm", "aiohttp", "msgpack", "zmq", "quart"):
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
  echo "node_server_exit host=${HOST} role=${ROLE} rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor() {
  echo "unix_ts,rx_bytes,tx_bytes,gpu_util_pct,gpu_power_w,gpu_sm_mhz,gpu_memory_used_mib" > "$TELEMETRY_FILE"
  while true; do
    unix_ts=$(date +%s)
    rx=$(cat "/sys/class/net/${IFACE}/statistics/rx_bytes" 2>/dev/null || echo NA)
    tx=$(cat "/sys/class/net/${IFACE}/statistics/tx_bytes" 2>/dev/null || echo NA)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    echo "${unix_ts},${rx},${tx},${gpu:-NA,NA,NA,NA}" >> "$TELEMETRY_FILE"
    sleep 2
  done
}

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

wait "$SERVER_PID"
exit $?
