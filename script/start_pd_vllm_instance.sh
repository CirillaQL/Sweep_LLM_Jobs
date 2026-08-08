#!/usr/bin/env bash
# Start one vLLM Prefill or Decode instance inside an allocated Slurm GPU step.

set -euo pipefail

ROLE="${1:-}"
ORDINAL="${2:-0}"

die() {
  echo "pd_instance_error=$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  [ -n "${!name:-}" ] || die "missing_environment_variable name=${name}"
}

case "$ROLE" in
  prefill)
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE="${PREFILL_KV_BUFFER_SIZE:-1e1}"
    ;;
  decode)
    KV_ROLE=kv_consumer
    KV_BUFFER_SIZE="${DECODE_KV_BUFFER_SIZE:-8e9}"
    ;;
  *)
    die "role_must_be_prefill_or_decode value=${ROLE:-unset}"
    ;;
esac

for name in MODEL VLLM_BIN PYTHON_BIN PD_INSTANCE_NAME PD_EXPECTED_GPU_MODEL \
  PD_NODE_IP PD_NET_IFACE PD_HTTP_PORT PD_KV_PORT PD_TP_SIZE PROXY_IP \
  PROXY_REGISTER_PORT PD_OUT_DIR; do
  require_env "$name"
done

for value_name in PD_HTTP_PORT PD_KV_PORT PD_TP_SIZE PROXY_REGISTER_PORT; do
  value="${!value_name}"
  case "$value" in
    ''|*[!0-9]*|0) die "invalid_positive_integer name=${value_name} value=${value:-unset}" ;;
  esac
done

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
[ -n "$VISIBLE_GPUS" ] || die "CUDA_VISIBLE_DEVICES_is_empty"
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$VISIBLE_GPUS"
[ "${#VISIBLE_GPU_ARRAY[@]}" -eq "$PD_TP_SIZE" ] || \
  die "visible_gpu_count_mismatch expected=${PD_TP_SIZE} actual=${#VISIBLE_GPU_ARRAY[@]} visible=${VISIBLE_GPUS}"
[ "$PD_TP_SIZE" -eq 1 ] || die "instance_tp_size_must_be_one value=${PD_TP_SIZE}"

[ -x "$VLLM_BIN" ] || die "vllm_binary_not_executable path=${VLLM_BIN}"
[ -x "$PYTHON_BIN" ] || die "python_binary_not_executable path=${PYTHON_BIN}"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia_smi_not_found"
ip link show dev "$PD_NET_IFACE" >/dev/null 2>&1 || \
  die "network_interface_not_found interface=${PD_NET_IFACE}"

GPU_NAME=$(nvidia-smi --id="${VISIBLE_GPU_ARRAY[0]}" \
  --query-gpu=name --format=csv,noheader | head -n 1)
[ -n "$GPU_NAME" ] || die "gpu_name_query_failed visible=${VISIBLE_GPUS}"
case "${GPU_NAME,,}" in
  *"${PD_EXPECTED_GPU_MODEL,,}"*) ;;
  *)
    die "gpu_model_mismatch instance=${PD_INSTANCE_NAME} expected=${PD_EXPECTED_GPU_MODEL} actual=${GPU_NAME}"
    ;;
esac

if [ -n "${REQUIRE_LINK_SPEED_MBPS:-}" ]; then
  actual_link_speed=$(<"/sys/class/net/${PD_NET_IFACE}/speed")
  [ "$actual_link_speed" = "$REQUIRE_LINK_SPEED_MBPS" ] || \
    die "link_speed_mismatch expected=${REQUIRE_LINK_SPEED_MBPS} actual=${actual_link_speed}"
fi

mkdir -p "$PD_OUT_DIR"
export VLLM_HOST_IP="$PD_NODE_IP"
export NCCL_SOCKET_IFNAME="$PD_NET_IFACE"
export GLOO_SOCKET_IFNAME="$PD_NET_IFACE"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

KV_CONFIG=$(
  KV_ROLE="$KV_ROLE" KV_BUFFER_SIZE="$KV_BUFFER_SIZE" \
  "$PYTHON_BIN" -c '
import json
import os

print(json.dumps({
    "kv_connector": "P2pNcclConnector",
    "kv_role": os.environ["KV_ROLE"],
    "kv_buffer_size": os.environ["KV_BUFFER_SIZE"],
    "kv_port": os.environ["PD_KV_PORT"],
    "kv_connector_extra_config": {
        "proxy_ip": os.environ["PROXY_IP"],
        "proxy_port": os.environ["PROXY_REGISTER_PORT"],
        "http_port": os.environ["PD_HTTP_PORT"],
        "send_type": os.environ.get("PD_SEND_TYPE", "PUT_ASYNC"),
        # vLLM 0.15.1 writes this value directly to os.environ while creating
        # the P2P NCCL context, so it must remain a string.
        "nccl_num_channels": os.environ.get("PD_NCCL_NUM_CHANNELS", "16"),
    },
}))
'
)

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
DTYPE="${DTYPE:-float16}"

VLLM_ARGS=(
  serve "$MODEL"
  --host 0.0.0.0
  --port "$PD_HTTP_PORT"
  --tensor-parallel-size "$PD_TP_SIZE"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-transfer-config "$KV_CONFIG"
)

if [ "${ENFORCE_EAGER:-true}" = true ]; then
  VLLM_ARGS+=(--enforce-eager)
fi
if [ -n "${ATTENTION_BACKEND:-}" ]; then
  VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

echo "pd_instance_start instance=${PD_INSTANCE_NAME} role=${ROLE} ordinal=${ORDINAL} host=$(hostname -s) gpu_model=${GPU_NAME} node_ip=${PD_NODE_IP} http_port=${PD_HTTP_PORT} kv_port=${PD_KV_PORT} tp=${PD_TP_SIZE} visible_gpus=${VISIBLE_GPUS}"
echo "pd_instance_network interface=${PD_NET_IFACE} proxy=${PROXY_IP}:${PROXY_REGISTER_PORT} nccl_net=${NCCL_NET}"
echo "pd_instance_model model=${MODEL} max_model_len=${MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_num_seqs=${MAX_NUM_SEQS}"

exec "$VLLM_BIN" "${VLLM_ARGS[@]}"
