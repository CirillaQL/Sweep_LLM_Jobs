#!/usr/bin/env bash

set -uo pipefail

MODE="${1:-serve}"
HOST=$(hostname -s)
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE_GPUS%%,*}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-64}"
OTLP_TRACES_ENDPOINT="${OTLP_TRACES_ENDPOINT:?OTLP_TRACES_ENDPOINT is required}"

case "$MODE" in
  preflight|serve) ;;
  *)
    echo "unsupported_mode=${MODE}"
    exit 20
    ;;
esac

case "$MAX_NUM_SEQS" in
  ''|*[!0-9]*|0)
    echo "invalid_max_num_seqs=${MAX_NUM_SEQS:-unset}"
    exit 22
    ;;
esac

case "$HOST" in
  neptune)
    NODE_GROUP=neptune
    ROLE=prefill
    EXPECTED_GPU=l40s
    NODE_IP=10.1.0.6
    PEER_IP="$DECODE_IP"
    IFACE=enp160s0f0np0
    HTTP_PORT="$PREFILL_HTTP_PORT"
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE=1e1
    ;;
  ganymede)
    NODE_GROUP=ganymede
    ROLE=decode
    EXPECTED_GPU=l4
    NODE_IP=10.1.0.3
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

NODE_WORK_DIR="${WORK_ROOT_BASE:?WORK_ROOT_BASE is required}/${SLURM_JOB_ID:-manual}/${HOST}"
export NODE_WORK_DIR
export XDG_CACHE_HOME="${NODE_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${NODE_WORK_DIR}/xdg-config"
export FLASHINFER_WORKSPACE_BASE="${NODE_WORK_DIR}/flashinfer"
export CUDA_CACHE_PATH="${NODE_WORK_DIR}/cuda-cache"
export TRITON_CACHE_DIR="${NODE_WORK_DIR}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${NODE_WORK_DIR}/torchinductor-cache"
export TORCH_HOME="${NODE_WORK_DIR}/torch"
export VLLM_CACHE_ROOT="${NODE_WORK_DIR}/vllm-cache"
export NUMBA_CACHE_DIR="${NODE_WORK_DIR}/numba-cache"
export TMPDIR="${NODE_WORK_DIR}/tmp"

for work_dir in \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$FLASHINFER_WORKSPACE_BASE" \
  "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$NUMBA_CACHE_DIR" "$TMPDIR"; do
  if ! mkdir -p "$work_dir" || [ ! -w "$work_dir" ]; then
    echo "work_directory_not_writable=true path=${work_dir}"
    exit 23
  fi
done
cd "$NODE_WORK_DIR" || exit 23

echo "host=${HOST} node_group=${NODE_GROUP} role=${ROLE} mode=${MODE}"
echo "expected_gpu=${EXPECTED_GPU} gpu_id=${GPU_ID}"
echo "gpu_dvfs=automatic_hardware manual_frequency_control=false scheduler_prediction=false"
echo "node_work_dir=${NODE_WORK_DIR}"
echo "node_cwd=${PWD}"
echo "flashinfer_workspace_base=${FLASHINFER_WORKSPACE_BASE}"
echo "xdg_cache_home=${XDG_CACHE_HOME} xdg_config_home=${XDG_CONFIG_HOME}"
echo "max_num_seqs=${MAX_NUM_SEQS}"
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

if [ "$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || true)" != "100000" ]; then
  echo "required_100gbe_link_missing=true"
  exit 11
fi

if [ "$MODE" = preflight ]; then
  if ! "$PYTHON_BIN" - \
    "${OUT_DIR}/environment_${HOST}.csv" "$HOST" "$ROLE" "$NODE_GROUP" \
    "$NODE_IP" "$PEER_IP" "$IFACE" "$EXPECTED_GPU" "$MAX_NUM_SEQS" \
    "$MODEL" "$OTLP_TRACES_ENDPOINT" <<'PY'
import csv
import importlib
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

(
    output,
    host,
    role,
    node_group,
    node_ip,
    peer_ip,
    interface,
    expected_gpu,
    max_num_seqs,
    model,
    otlp_endpoint,
) = sys.argv[1:]

for name in (
    "torch",
    "vllm",
    "aiohttp",
    "msgpack",
    "zmq",
    "opentelemetry.sdk.trace",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.proto.collector.trace.v1.trace_service_pb2",
    "vllm.v1.attention.backends.flash_attn",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
):
    module = importlib.import_module(name)
    print(
        f"host={socket.gethostname()} import={name} "
        f"version={getattr(module, '__version__', 'unknown')}"
    )

from flashinfer.jit import env as flashinfer_jit_env

node_work_dir = Path(os.environ["NODE_WORK_DIR"]).resolve()
flashinfer_workspace = Path(
    flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
).resolve()
if (
    flashinfer_workspace != node_work_dir
    and node_work_dir not in flashinfer_workspace.parents
):
    raise RuntimeError(
        f"flashinfer workspace escaped chjing work directory: "
        f"{flashinfer_workspace} not under {node_work_dir}"
    )
if not flashinfer_workspace.is_dir() or not os.access(
    flashinfer_workspace, os.W_OK
):
    raise RuntimeError(
        f"flashinfer workspace is not writable: {flashinfer_workspace}"
    )
print(
    f"host={socket.gethostname()} "
    f"flashinfer_workspace={flashinfer_workspace} writable=true"
)

def read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""

def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

row = {
    "unix_ns": __import__("time").time_ns(),
    "hostname": host,
    "role": role,
    "node_group": node_group,
    "node_ip": node_ip,
    "peer_ip": peer_ip,
    "interface": interface,
    "link_speed_mbps": read(f"/sys/class/net/{interface}/speed"),
    "link_mtu": read(f"/sys/class/net/{interface}/mtu"),
    "kernel": platform.release(),
    "cpu_count": os.cpu_count(),
    "gpu": command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ),
    "expected_gpu": expected_gpu,
    "model": model,
    "max_num_seqs": max_num_seqs,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.82,
    "kv_connector": "P2pNcclConnector",
    "kv_cache_metrics": True,
    "kv_cache_metrics_sample": 1.0,
    "enable_logging_iteration_details": True,
    "collect_detailed_traces": "all",
    "enable_prefix_caching": True,
    "enable_request_id_headers": True,
    "otlp_endpoint": otlp_endpoint,
    "dvfs_mode": "automatic_hardware",
    "manual_frequency_control": False,
    "scheduler_prediction": False,
    "attention_backend": "FLASH_ATTN",
    "node_work_dir": os.environ["NODE_WORK_DIR"],
    "runtime_cwd": os.getcwd(),
    "flashinfer_workspace_base": os.environ["FLASHINFER_WORKSPACE_BASE"],
    "flashinfer_workspace": str(flashinfer_workspace),
    "xdg_cache_home": os.environ["XDG_CACHE_HOME"],
    "xdg_config_home": os.environ["XDG_CONFIG_HOME"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
}
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
PY
  then
    echo "node_preflight_python_failed=true host=${HOST}"
    exit 14
  fi
  echo "node_preflight_ok=true host=${HOST}"
  exit 0
fi

export VLLM_HOST_IP="$NODE_IP"
export OTEL_SERVICE_NAME="vllm-${ROLE}-${HOST}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$OTLP_TRACES_ENDPOINT"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
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
  echo "unix_ts,rx_bytes,tx_bytes,gpu_util_pct,gpu_power_w,gpu_sm_mhz,gpu_memory_used_mib,gpu_mem_util_pct,gpu_power_limit_w,gpu_mem_clock_mhz,gpu_temperature_c,gpu_memory_total_mib,gpu_pstate,dvfs_mode" > "$TELEMETRY_FILE"
  while true; do
    unix_ts=$(date +%s.%N)
    rx=$(cat "/sys/class/net/${IFACE}/statistics/rx_bytes" 2>/dev/null || echo NA)
    tx=$(cat "/sys/class/net/${IFACE}/statistics/tx_bytes" 2>/dev/null || echo NA)
    gpu=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used,utilization.memory,power.limit,clocks.mem,temperature.gpu,memory.total,pstate --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    echo "${unix_ts},${rx},${tx},${gpu:-NA,NA,NA,NA,NA,NA,NA,NA,NA,NA},automatic_hardware" >> "$TELEMETRY_FILE"
    sleep 0.5
  done
}

echo "automatic_dvfs_enabled=true manual_frequency_control=false"
echo "attention_backend=FLASH_ATTN backend_selection=explicit"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,clocks.current.graphics,clocks.max.graphics,pstate --format=csv 2>&1 || true

echo "launch_vllm host=${HOST} role=${ROLE} ip=${NODE_IP} http_port=${HTTP_PORT} kv_port=${KV_PORT}"
echo "NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "kv_transfer_config=${KV_CONFIG}"
echo "observability kv_cache_metrics=true kv_cache_metrics_sample=1.0 enable_logging_iteration_details=true collect_detailed_traces=all attention_backend=FLASH_ATTN otlp=${OTLP_TRACES_ENDPOINT}"

"$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$HTTP_PORT" \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.82 \
  --attention-config.backend FLASH_ATTN \
  --enable-prefix-caching \
  --enable-request-id-headers \
  --enable-log-requests \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 1.0 \
  --enable-logging-iteration-details \
  --enable-mfu-metrics \
  --otlp-traces-endpoint "$OTLP_TRACES_ENDPOINT" \
  --collect-detailed-traces all \
  --kv-transfer-config "$KV_CONFIG" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
monitor &
MONITOR_PID=$!

wait "$SERVER_PID"
exit $?
