#!/usr/bin/env bash

set -uo pipefail

MODE="${1:-serve}"
HOST=$(hostname -s)
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$VISIBLE_GPUS"
GPU_ID="${VISIBLE_GPUS%%,*}"
CLOCK_ACK_TOLERANCE_MHZ="${CLOCK_ACK_TOLERANCE_MHZ_OVERRIDE:-90}"
GPU_CLOCK_CONTROL_MODE="${GPU_CLOCK_CONTROL_MODE_OVERRIDE:-manual}"
GPU_STARTUP_CLOCK_MODE="${GPU_STARTUP_CLOCK_MODE_OVERRIDE:-scheduled}"
DEFER_INSTANCE_TELEMETRY_UNTIL_READY="${DEFER_INSTANCE_TELEMETRY_UNTIL_READY_OVERRIDE:-false}"
CLOCK_ACK_MODE="${CLOCK_ACK_MODE_OVERRIDE:-monitor}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS_OVERRIDE:-$MAX_MODEL_LEN}"
INSTANCE_ID="${PD_INSTANCE_ID_OVERRIDE:-primary}"

for value_name in MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS; do
  value="${!value_name}"
  case "$value" in
    ''|*[!0-9]*|0)
      echo "invalid_${value_name,,}=${value:-unset}"
      exit 22
      ;;
  esac
done
if [ "$MAX_MODEL_LEN" -lt 2 ]; then
  echo "max_model_len_too_small=${MAX_MODEL_LEN}"
  exit 22
fi

case "$CLOCK_ACK_MODE" in
  monitor|active_probe) ;;
  *)
    echo "unsupported_clock_ack_mode=${CLOCK_ACK_MODE}"
    exit 21
    ;;
esac
case "$GPU_STARTUP_CLOCK_MODE" in
  scheduled|max) ;;
  *)
    echo "unsupported_gpu_startup_clock_mode=${GPU_STARTUP_CLOCK_MODE}"
    exit 21
    ;;
esac
case "$DEFER_INSTANCE_TELEMETRY_UNTIL_READY" in
  true|false) ;;
  *)
    echo "invalid_defer_instance_telemetry_until_ready=${DEFER_INSTANCE_TELEMETRY_UNTIL_READY}"
    exit 21
    ;;
esac

case "$HOST" in
  neptune)
    NODE_GROUP=neptune
    NODE_IP=10.1.0.6
    IFACE=enp160s0f0np0
    ;;
  uranus)
    NODE_GROUP=uranus
    NODE_IP=10.1.0.5
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

PLACEMENT_VALUES=$("$PYTHON_BIN" - "$PLACEMENT_FILE" "$NODE_GROUP" "${PD_ROLE_OVERRIDE:-}" <<'PY'
import json
import sys

placement = json.load(open(sys.argv[1], encoding="utf-8"))
node_group = sys.argv[2]
role_override = sys.argv[3]
recommended = placement["recommended"]
if role_override:
    if role_override not in {"prefill", "decode"}:
        raise SystemExit(f"unsupported role override: {role_override}")
    spec = recommended[role_override]
    print(
        role_override,
        int(spec.get("rec_freq_mhz", spec["freq_mhz"])),
        spec["gpu_type"],
    )
    raise SystemExit(0)
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
    TP_SIZE="${PD_TP_SIZE_OVERRIDE:-${PREFILL_TP_SIZE_OVERRIDE:-1}}"
    ;;
  decode)
    TP_SIZE="${PD_TP_SIZE_OVERRIDE:-${DECODE_TP_SIZE_OVERRIDE:-1}}"
    ;;
esac
case "$TP_SIZE" in
  ''|*[!0-9]*|0)
    echo "invalid_tensor_parallel_size=${TP_SIZE:-unset} role=${ROLE:-unset}"
    exit 22
    ;;
esac
EXPECTED_VISIBLE_GPU_COUNT="$TP_SIZE"
if [ "$MODE" = preflight ]; then
  EXPECTED_VISIBLE_GPU_COUNT=1
fi
if [ "${#VISIBLE_GPU_ARRAY[@]}" -ne "$EXPECTED_VISIBLE_GPU_COUNT" ]; then
  echo "visible_gpu_count_mismatch=true mode=${MODE} role=${ROLE} tp_size=${TP_SIZE} expected=${EXPECTED_VISIBLE_GPU_COUNT} actual=${#VISIBLE_GPU_ARRAY[@]} visible_gpus=${VISIBLE_GPUS}"
  exit 14
fi

case "$ROLE" in
  prefill)
    PEER_IP="$DECODE_IP"
    HTTP_PORT="${PD_HTTP_PORT_OVERRIDE:-$PREFILL_HTTP_PORT}"
    KV_ROLE=kv_producer
    KV_BUFFER_SIZE=1e1
    ;;
  decode)
    PEER_IP="$PREFILL_IP"
    HTTP_PORT="${PD_HTTP_PORT_OVERRIDE:-$DECODE_HTTP_PORT}"
    KV_ROLE=kv_consumer
    KV_BUFFER_SIZE=8e9
    ;;
  *)
    echo "unsupported_role=${ROLE:-unset}"
    exit 12
    ;;
esac
INSTANCE_SUFFIX=""
if [ "$INSTANCE_ID" != primary ]; then
  INSTANCE_SUFFIX="_${INSTANCE_ID}"
fi
CLOCK_KEY="${PD_CLOCK_KEY_OVERRIDE:-$NODE_GROUP}"
INSTANCE_KV_PORT="${PD_KV_PORT_OVERRIDE:-$KV_PORT}"

echo "host=${HOST} node_group=${NODE_GROUP} role=${ROLE} instance_id=${INSTANCE_ID} mode=${MODE}"
echo "scheduled_gpu=${EXPECTED_GPU} scheduled_freq_mhz=${TARGET_FREQ} gpu_id=${GPU_ID}"
echo "gpu_clock_control_mode=${GPU_CLOCK_CONTROL_MODE}"
echo "gpu_startup_clock_mode=${GPU_STARTUP_CLOCK_MODE}"
echo "defer_instance_telemetry_until_ready=${DEFER_INSTANCE_TELEMETRY_UNTIL_READY}"
echo "clock_ack_mode=${CLOCK_ACK_MODE}"
echo "tensor_parallel_size=${TP_SIZE}"
echo "max_num_seqs=${MAX_NUM_SEQS}"
echo "max_model_len=${MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
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
  echo "parent_reset_step=true host=${HOST} gpu_ids=${VISIBLE_GPUS} rec_freq_mhz=${TARGET_FREQ}"
  echo "reset_command_initial=sudo_nvidia_smi_rgc"
  for reset_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
    if ! sudo nvidia-smi -i "$reset_gpu_id" -rgc; then
      echo "reset_gpu_clock_failed=true host=${HOST} gpu_id=${reset_gpu_id}"
      exit 15
    fi
  done
  MAX_FREQ=$(nvidia-smi -i "$GPU_ID" --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  RESET_PROBE_FILE="${OUT_DIR}/reset_${HOST}_gpu_${GPU_ID}_probe.json"
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
  for reset_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
    if ! sudo nvidia-smi -i "$reset_gpu_id" -rgc; then
      echo "final_reset_gpu_clock_failed=true host=${HOST} gpu_id=${reset_gpu_id}"
      exit 18
    fi
  done
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
export NCCL_DEBUG_FILE="${OUT_DIR}/nccl-${ROLE}-${HOST}${INSTANCE_SUFFIX}-%p.log"
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_DISTRIBUTED_DEBUG=DETAIL

KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_buffer_size\":\"${KV_BUFFER_SIZE}\",\"kv_port\":\"${INSTANCE_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_port\":\"${PROXY_REGISTER_PORT}\",\"http_port\":\"${HTTP_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"
SERVER_LOG="${OUT_DIR}/${ROLE}${INSTANCE_SUFFIX}_server.log"
TELEMETRY_FILE="${OUT_DIR}/${ROLE}_${HOST}${INSTANCE_SUFFIX}_telemetry.csv"
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
    echo "reset_gpu_clocks gpu_ids=${VISIBLE_GPUS}"
    for cleanup_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
      sudo nvidia-smi -i "$cleanup_gpu_id" -rgc 2>&1 || true
    done
  fi
  echo "node_server_exit host=${HOST} role=${ROLE} rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor() {
  echo "unix_ts,host,role,instance_id,gpu_id,gpu_uuid,workload_seq,target_freq_mhz,rx_bytes,tx_bytes,gpu_util_pct,gpu_power_w,gpu_sm_mhz,gpu_memory_used_mib,gpu_mem_util_pct,gpu_power_limit_w,gpu_mem_clock_mhz,gpu_temperature_c,gpu_memory_total_mib,gpu_pstate" >> "$TELEMETRY_FILE"
  while true; do
    unix_ts=$(date +%s.%N)
    workload_seq=0
    target_freq="$TARGET_FREQ"
    if [ -s "${CLOCK_CONTROL_DIR}/${CLOCK_KEY}.request" ]; then
      read -r workload_seq target_freq < "${CLOCK_CONTROL_DIR}/${CLOCK_KEY}.request" || true
    fi
    rx=$(cat "/sys/class/net/${IFACE}/statistics/rx_bytes" 2>/dev/null || echo NA)
    tx=$(cat "/sys/class/net/${IFACE}/statistics/tx_bytes" 2>/dev/null || echo NA)
    for monitor_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
      monitor_gpu_uuid=$(nvidia-smi -i "$monitor_gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)
      gpu=$(nvidia-smi -i "$monitor_gpu_id" --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used,utilization.memory,power.limit,clocks.mem,temperature.gpu,memory.total,pstate --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      echo "${unix_ts},${HOST},${ROLE},${INSTANCE_ID},${monitor_gpu_id},${monitor_gpu_uuid:-NA},${workload_seq},${target_freq},${rx},${tx},${gpu:-NA,NA,NA,NA,NA,NA,NA,NA,NA,NA}" >> "$TELEMETRY_FILE"
    done
    sleep 0.5
  done
}

clock_controller() {
  local request_file="${CLOCK_CONTROL_DIR}/${CLOCK_KEY}.request"
  local ack_file="${CLOCK_CONTROL_DIR}/${CLOCK_KEY}.ack"
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
        for clock_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
          if ! sudo nvidia-smi -i "$clock_gpu_id" -lgc "${target},${target}"; then
            rc=31
          fi
        done
        if [ "$rc" -eq 0 ] && [ "$CLOCK_ACK_MODE" = active_probe ]; then
          if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$CLOCK_PROBE" \
            --smi-index "$GPU_ID" --seconds 2 --output "$probe_file"; then
            rc=32
          else
            observed=$(
              "$PYTHON_BIN" - "$probe_file" "$target" "$CLOCK_ACK_TOLERANCE_MHZ" <<'PY'
import json
import sys

path, target_text, tolerance_text = sys.argv[1:]
target = int(target_text)
tolerance = int(tolerance_text)
data = json.load(open(path, encoding="utf-8"))
data["target_freq_mhz"] = target
data["ack_tolerance_mhz"] = tolerance
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
mean = float(data["active_clock_mean_mhz"])
print(round(mean))
raise SystemExit(0 if abs(mean - target) <= tolerance else 1)
PY
            ) || rc=33
          fi
        elif [ "$rc" -eq 0 ]; then
          # Starting a new CUDA process while vLLM owns almost all L4 memory can
          # fail during context creation. Treat successful -lgc as the control
          # acknowledgement and verify sustained clocks from the existing
          # 0.5-second workload telemetry instead.
          observed=$(
            nvidia-smi -i "$GPU_ID" --query-gpu=clocks.current.graphics \
              --format=csv,noheader,nounits 2>/dev/null \
              | head -n 1 | tr -d ' '
          )
          observed="${observed:-NA}"
          echo "clock_runtime_verification host=${HOST} seq=${seq} source=${TELEMETRY_FILE}"
        fi
        # The parent pre-creates this file. Writing it in place avoids a stale
        # negative lookup or directory-entry cache on the shared filesystem.
        # A partial read is harmless because the parent only accepts a complete
        # matching sequence and retries once per second.
        printf '%s %s %s %s\n' "$seq" "$target" "$rc" "$observed" > "$ack_file"
        ack_publish_rc=1
        ack_payload=$(
          printf '{"node_group":"%s","seq":%s,"target_mhz":%s,"rc":%s,"observed_mhz":"%s"}' \
            "$CLOCK_KEY" "$seq" "$target" "$rc" "$observed"
        )
        for ack_publish_attempt in 1 2 3; do
          if curl -fsS --connect-timeout 2 --max-time 5 \
            -H "Content-Type: application/json" \
            -d "$ack_payload" \
            "http://${PROXY_IP}:${PROXY_HTTP_PORT}/control/clock-ack" \
            >/dev/null 2>&1; then
            ack_publish_rc=0
            break
          fi
          sleep 1
        done
        echo "clock_ack_transport host=${HOST} seq=${seq} transport=http rc=${ack_publish_rc} attempts=${ack_publish_attempt}"
        echo "dynamic_clock_ack host=${HOST} seq=${seq} target_mhz=${target} rc=${rc} observed_mhz=${observed} verification_mode=${CLOCK_ACK_MODE}"
        last_seq="$seq"
      fi
    fi
    sleep 1
  done
}

if [ "$GPU_CLOCK_CONTROL_MODE" = manual ]; then
  STARTUP_FREQ="$TARGET_FREQ"
  if [ "$GPU_STARTUP_CLOCK_MODE" = max ]; then
    STARTUP_FREQ=$(nvidia-smi -i "$GPU_ID" \
      --query-gpu=clocks.max.graphics --format=csv,noheader,nounits \
      | head -n 1 | tr -d ' ')
    case "$STARTUP_FREQ" in
      ''|*[!0-9]*)
        echo "startup_max_clock_query_failed=true gpu_id=${GPU_ID} value=${STARTUP_FREQ:-unset}"
        exit 14
        ;;
    esac
  fi
  echo "lock_gpu_clocks gpu_ids=${VISIBLE_GPUS} target_mhz=${STARTUP_FREQ} startup_mode=${GPU_STARTUP_CLOCK_MODE} scheduled_mhz=${TARGET_FREQ}"
  for lock_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
    if ! sudo nvidia-smi -i "$lock_gpu_id" -lgc "${STARTUP_FREQ},${STARTUP_FREQ}"; then
      echo "lock_gpu_clock_failed=true gpu_id=${lock_gpu_id}"
      exit 14
    fi
  done
  CLOCK_LOCKED=true
elif [ "$GPU_CLOCK_CONTROL_MODE" = auto ]; then
  echo "automatic_dvfs_enabled=true manual_clock_commands=false"
else
  echo "unsupported_gpu_clock_control_mode=${GPU_CLOCK_CONTROL_MODE}"
  exit 20
fi
for query_gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
  nvidia-smi -i "$query_gpu_id" --query-gpu=index,name,clocks.current.graphics,clocks.max.graphics --format=csv 2>&1 || true
done

echo "launch_vllm host=${HOST} role=${ROLE} instance_id=${INSTANCE_ID} ip=${NODE_IP} http_port=${HTTP_PORT} kv_port=${INSTANCE_KV_PORT}"
echo "NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "kv_transfer_config=${KV_CONFIG}"

"$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$HTTP_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype float16 \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.82 \
  --kv-transfer-config "$KV_CONFIG" \
  >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
if [ "$DEFER_INSTANCE_TELEMETRY_UNTIL_READY" = true ]; then
  (
    echo "instance_telemetry_waiting_for_http=true url=http://127.0.0.1:${HTTP_PORT}/v1/models"
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if curl -fsS --connect-timeout 2 --max-time 5 \
        "http://127.0.0.1:${HTTP_PORT}/v1/models" >/dev/null 2>&1; then
        echo "instance_telemetry_http_ready=true"
        monitor
        exit $?
      fi
      sleep 2
    done
    echo "instance_telemetry_not_started=true reason=server_exited_before_http_ready"
  ) &
else
  monitor &
fi
MONITOR_PID=$!
if [ "$GPU_CLOCK_CONTROL_MODE" = manual ]; then
  clock_controller &
  CLOCK_CONTROLLER_PID=$!
fi

wait "$SERVER_PID"
exit $?
