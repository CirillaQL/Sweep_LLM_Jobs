#!/usr/bin/env bash
# Launch independent mixed-TP Prefill and Decode vLLM instances in an existing
# Slurm allocation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -n "${PD_CONFIG_FILE:-}" ]; then
  # The config is a trusted shell environment file; see pd_vllm.env.example.
  # shellcheck source=/dev/null
  source "$PD_CONFIG_FILE"
fi

die() {
  echo "pd_launcher_error=$*" >&2
  exit 2
}

positive_integer() {
  case "$2" in
    ''|*[!0-9]*|0) die "invalid_positive_integer name=$1 value=${2:-unset}" ;;
  esac
}

command -v srun >/dev/null 2>&1 || die "srun_not_found"
command -v curl >/dev/null 2>&1 || die "curl_not_found"
[ -n "${SLURM_JOB_ID:-}" ] || die "must_run_inside_a_slurm_allocation"

PYTHON_BIN="${PYTHON_BIN:-/data/users/chjing/miniforge3/envs/cuda-env/bin/python}"
VLLM_BIN="${VLLM_BIN:-/data/users/chjing/miniforge3/envs/cuda-env/bin/vllm}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/scheduler_custom_policy.py}"
INSTANCE_SCRIPT="${INSTANCE_SCRIPT:-${SCRIPT_DIR}/start_pd_vllm_instance.sh}"
CLOCK_AGENT_SCRIPT="${CLOCK_AGENT_SCRIPT:-${SCRIPT_DIR}/pd_clock_agent.py}"
DVFS_PREDICTOR_SCRIPT="${DVFS_PREDICTOR_SCRIPT:-${SCRIPT_DIR}/request_dvfs_predictor.py}"
PD_ENABLE_PREDICTIVE_DVFS="${PD_ENABLE_PREDICTIVE_DVFS:-false}"
PD_DVFS_SOURCE_JOB_DIR="${PD_DVFS_SOURCE_JOB_DIR:-/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate}"
PD_DVFS_SCHEDULER_SCRIPT="${PD_DVFS_SCHEDULER_SCRIPT:-${PD_DVFS_SOURCE_JOB_DIR}/scheduler.py}"
PD_DVFS_MODEL_BUNDLE="${PD_DVFS_MODEL_BUNDLE:-${PD_DVFS_SOURCE_JOB_DIR}/model_bundle.json}"
PD_DVFS_SATURATION_BUNDLE="${PD_DVFS_SATURATION_BUNDLE:-${PD_DVFS_SOURCE_JOB_DIR}/saturation_bundle.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
REQUIRE_LINK_SPEED_MBPS="${REQUIRE_LINK_SPEED_MBPS:-}"

PREFILL_NODE="${PREFILL_NODE:-uranus}"
PREFILL_IP="${PREFILL_IP:-10.1.0.5}"
PREFILL_NET_IFACE="${PREFILL_NET_IFACE:-enp160s0f0np0}"
DECODE_NODE="${DECODE_NODE:-ganymede}"
DECODE_IP="${DECODE_IP:-10.1.0.3}"
DECODE_NET_IFACE="${DECODE_NET_IFACE:-eno12399np0}"
PROXY_NODE="${PROXY_NODE:-$DECODE_NODE}"
PROXY_IP="${PROXY_IP:-$DECODE_IP}"

PREFILL_REPLICAS="${PREFILL_REPLICAS:-2}"
DECODE_REPLICAS="${DECODE_REPLICAS:-2}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
PREFILL_TP_SIZES="${PREFILL_TP_SIZES:-}"
DECODE_TP_SIZES="${DECODE_TP_SIZES:-}"
PREFILL_GPU_MODEL="${PREFILL_GPU_MODEL:-L40S}"
DECODE_GPU_MODEL="${DECODE_GPU_MODEL:-L4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
VLLM_STEP_MEM="${VLLM_STEP_MEM:-32G}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"

for pair in \
  PREFILL_REPLICAS:$PREFILL_REPLICAS DECODE_REPLICAS:$DECODE_REPLICAS \
  PREFILL_TP_SIZE:$PREFILL_TP_SIZE DECODE_TP_SIZE:$DECODE_TP_SIZE \
  CPUS_PER_GPU:$CPUS_PER_GPU STARTUP_TIMEOUT_SECONDS:$STARTUP_TIMEOUT_SECONDS; do
  positive_integer "${pair%%:*}" "${pair#*:}"
done

PREFILL_TP_SIZE_ARRAY=()
DECODE_TP_SIZE_ARRAY=()
if [ -n "$PREFILL_TP_SIZES" ]; then
  IFS=',' read -r -a PREFILL_TP_SIZE_ARRAY <<< "$PREFILL_TP_SIZES"
else
  for ((ordinal = 0; ordinal < PREFILL_REPLICAS; ordinal++)); do
    PREFILL_TP_SIZE_ARRAY+=("$PREFILL_TP_SIZE")
  done
fi
if [ -n "$DECODE_TP_SIZES" ]; then
  IFS=',' read -r -a DECODE_TP_SIZE_ARRAY <<< "$DECODE_TP_SIZES"
else
  for ((ordinal = 0; ordinal < DECODE_REPLICAS; ordinal++)); do
    DECODE_TP_SIZE_ARRAY+=("$DECODE_TP_SIZE")
  done
fi
[ "${#PREFILL_TP_SIZE_ARRAY[@]}" -eq "$PREFILL_REPLICAS" ] || \
  die "prefill_tp_size_count_mismatch replicas=${PREFILL_REPLICAS} sizes=${PREFILL_TP_SIZES}"
[ "${#DECODE_TP_SIZE_ARRAY[@]}" -eq "$DECODE_REPLICAS" ] || \
  die "decode_tp_size_count_mismatch replicas=${DECODE_REPLICAS} sizes=${DECODE_TP_SIZES}"
for ordinal in "${!PREFILL_TP_SIZE_ARRAY[@]}"; do
  positive_integer "PREFILL_TP_SIZES[$ordinal]" "${PREFILL_TP_SIZE_ARRAY[$ordinal]}"
done
for ordinal in "${!DECODE_TP_SIZE_ARRAY[@]}"; do
  positive_integer "DECODE_TP_SIZES[$ordinal]" "${DECODE_TP_SIZE_ARRAY[$ordinal]}"
done
PREFILL_TP_SIZES=$(IFS=,; echo "${PREFILL_TP_SIZE_ARRAY[*]}")
DECODE_TP_SIZES=$(IFS=,; echo "${DECODE_TP_SIZE_ARRAY[*]}")
export PREFILL_TP_SIZES DECODE_TP_SIZES
PREFILL_GPU_COUNT=0
DECODE_GPU_COUNT=0
for tp_size in "${PREFILL_TP_SIZE_ARRAY[@]}"; do
  PREFILL_GPU_COUNT=$((PREFILL_GPU_COUNT + tp_size))
done
for tp_size in "${DECODE_TP_SIZE_ARRAY[@]}"; do
  DECODE_GPU_COUNT=$((DECODE_GPU_COUNT + tp_size))
done

[ -x "$PYTHON_BIN" ] || die "python_binary_not_executable path=${PYTHON_BIN}"
[ -x "$VLLM_BIN" ] || die "vllm_binary_not_executable path=${VLLM_BIN}"
[ -r "$PROXY_SCRIPT" ] || die "proxy_script_not_readable path=${PROXY_SCRIPT}"
[ -x "$INSTANCE_SCRIPT" ] || die "instance_script_not_executable path=${INSTANCE_SCRIPT}"
if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
  for artifact in "$CLOCK_AGENT_SCRIPT" "$DVFS_PREDICTOR_SCRIPT" \
    "$PD_DVFS_SCHEDULER_SCRIPT" "$PD_DVFS_MODEL_BUNDLE" \
    "$PD_DVFS_SATURATION_BUNDLE"; do
    [ -r "$artifact" ] || die "dvfs_artifact_not_readable path=${artifact}"
  done
fi

# Batch-level resource variables must not leak into nested steps. Every GPU
# step below declares its own exact CPU, memory, and GPU requirements.
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK
unset SLURM_GPUS
unset SLURM_GPUS_PER_NODE
unset SLURM_GPUS_PER_SOCKET
unset SLURM_GPUS_PER_TASK
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

PORT_SLOT_COUNT="${PD_PORT_SLOT_COUNT:-60}"
PORT_SLOT_STRIDE="${PD_PORT_SLOT_STRIDE:-16}"
PORT_OFFSET="${PD_PORT_OFFSET:-$(((SLURM_JOB_ID % PORT_SLOT_COUNT) * PORT_SLOT_STRIDE))}"
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-$((30000 + PORT_OFFSET))}"
PROXY_REGISTER_PORT="${PROXY_REGISTER_PORT:-$((31000 + PORT_OFFSET))}"
PREFILL_HTTP_PORT_BASE="${PREFILL_HTTP_PORT_BASE:-$((32000 + PORT_OFFSET))}"
DECODE_HTTP_PORT_BASE="${DECODE_HTTP_PORT_BASE:-$((33000 + PORT_OFFSET))}"
PREFILL_KV_PORT_BASE="${PREFILL_KV_PORT_BASE:-$((34000 + PORT_OFFSET))}"
DECODE_KV_PORT_BASE="${DECODE_KV_PORT_BASE:-$((34000 + PORT_OFFSET))}"

PD_OUT_DIR="${PD_OUT_DIR:-$PWD/pd_vllm_${SLURM_JOB_ID}}"
PD_WORK_DIR="${PD_WORK_DIR:-/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}}"
case "$PD_WORK_DIR" in
  /data/users/chjing/vllm_job_work/"${SLURM_JOB_ID}"|/data/users/chjing/vllm_job_work/"${SLURM_JOB_ID}"/*) ;;
  *) die "unsafe_pd_work_dir path=${PD_WORK_DIR}" ;;
esac

# Slurm steps can inherit HOME=/root from the launcher. Force every writable
# user/cache/config location into this job's disposable work tree. In
# particular, NIXL otherwise probes $HOME/.nixl.cfg during agent startup.
umask 077
export HOME="${PD_WORK_DIR}/home"
export XDG_CACHE_HOME="${PD_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${PD_WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${PD_WORK_DIR}/xdg-data"
export XDG_STATE_HOME="${PD_WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${PD_WORK_DIR}/xdg-runtime"
export NIXL_CONFIG_FILE="${PD_WORK_DIR}/nixl/nixl.cfg"
export HF_HOME="${PD_WORK_DIR}/huggingface"
export HF_TOKEN_PATH="${HF_HOME}/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export FLASHINFER_WORKSPACE_BASE="${PD_WORK_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${PD_WORK_DIR}/vllm-cache"
export VLLM_CONFIG_ROOT="${PD_WORK_DIR}/vllm-config"
export TORCH_HOME="${PD_WORK_DIR}/torch-cache"
export TORCH_EXTENSIONS_DIR="${PD_WORK_DIR}/torch-extensions"
export TRITON_CACHE_DIR="${PD_WORK_DIR}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${PD_WORK_DIR}/torchinductor-cache"
export CUDA_CACHE_PATH="${PD_WORK_DIR}/cuda-cache"
export NUMBA_CACHE_DIR="${PD_WORK_DIR}/numba-cache"
export RAY_TMPDIR="${PD_WORK_DIR}/ray-tmp"
export PIP_CACHE_DIR="${PD_WORK_DIR}/pip-cache"
export UV_CACHE_DIR="${PD_WORK_DIR}/uv-cache"
export PYTHONPYCACHEPREFIX="${PD_WORK_DIR}/python-pycache"
export TIKTOKEN_CACHE_DIR="${PD_WORK_DIR}/tiktoken-cache"
export MPLCONFIGDIR="${PD_WORK_DIR}/matplotlib"
export CUPY_CACHE_DIR="${PD_WORK_DIR}/cupy-cache"
export TMPDIR="${PD_WORK_DIR}/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p \
  "$PD_OUT_DIR" "$PD_WORK_DIR" "$HOME" "$HF_HOME" "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" "$HF_XET_CACHE" "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" \
  "$(dirname -- "$NIXL_CONFIG_FILE")" "$FLASHINFER_WORKSPACE_BASE" \
  "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$RAY_TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" \
  "$CUPY_CACHE_DIR" "$TMPDIR"
touch "$NIXL_CONFIG_FILE"
chmod 600 "$NIXL_CONFIG_FILE"
chmod 700 "$HOME" "$XDG_RUNTIME_DIR"
RUNTIME_PROXY_SCRIPT="${PD_WORK_DIR}/scheduler_custom_policy.py"
RUNTIME_INSTANCE_SCRIPT="${PD_WORK_DIR}/start_pd_vllm_instance.sh"
cp -- "$PROXY_SCRIPT" "$RUNTIME_PROXY_SCRIPT"
cp -- "$INSTANCE_SCRIPT" "$RUNTIME_INSTANCE_SCRIPT"
chmod +x "$RUNTIME_PROXY_SCRIPT" "$RUNTIME_INSTANCE_SCRIPT"
RUNTIME_CLOCK_AGENT_SCRIPT="${PD_WORK_DIR}/pd_clock_agent.py"
RUNTIME_DVFS_PREDICTOR_SCRIPT="${PD_WORK_DIR}/request_dvfs_predictor.py"
RUNTIME_DVFS_SCHEDULER_SCRIPT="${PD_WORK_DIR}/portable_sweep_scheduler.py"
RUNTIME_DVFS_MODEL_BUNDLE="${PD_WORK_DIR}/model_bundle.json"
RUNTIME_DVFS_SATURATION_BUNDLE="${PD_WORK_DIR}/saturation_bundle.json"
if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
  cp -- "$CLOCK_AGENT_SCRIPT" "$RUNTIME_CLOCK_AGENT_SCRIPT"
  cp -- "$DVFS_PREDICTOR_SCRIPT" "$RUNTIME_DVFS_PREDICTOR_SCRIPT"
  cp -- "$PD_DVFS_SCHEDULER_SCRIPT" "$RUNTIME_DVFS_SCHEDULER_SCRIPT"
  cp -- "$PD_DVFS_MODEL_BUNDLE" "$RUNTIME_DVFS_MODEL_BUNDLE"
  cp -- "$PD_DVFS_SATURATION_BUNDLE" "$RUNTIME_DVFS_SATURATION_BUNDLE"
  chmod +x "$RUNTIME_CLOCK_AGENT_SCRIPT" "$RUNTIME_DVFS_PREDICTOR_SCRIPT" \
    "$RUNTIME_DVFS_SCHEDULER_SCRIPT"
fi

STEP_PIDS=()
cleanup_done=false
reset_node_clocks() {
  local node="$1"
  local gpu_count="$2"
  [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ] || return 0
  echo "pd_clock_reset_start node=${node} gpu_count=${gpu_count}"
  srun --overlap --exact --nodes=1 --nodelist="$node" --ntasks=1 \
    --cpus-per-task=1 --mem=1G --gres=none \
    env PD_RESET_GPU_COUNT="$gpu_count" bash -c '
      rc=0
      for ((gpu_id = 0; gpu_id < PD_RESET_GPU_COUNT; gpu_id++)); do
        sudo -n nvidia-smi -i "$gpu_id" -rgc || rc=1
      done
      exit "$rc"
    ' </dev/null || echo "pd_clock_reset_failed node=${node}" >&2
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$cleanup_done" = false ]; then
    cleanup_done=true
    for pid in "${STEP_PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    for pid in "${STEP_PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    reset_node_clocks "$PREFILL_NODE" "$PREFILL_GPU_COUNT"
    reset_node_clocks "$DECODE_NODE" "$DECODE_GPU_COUNT"
    case "$PD_WORK_DIR" in
      /data/users/chjing/vllm_job_work/"${SLURM_JOB_ID}"|/data/users/chjing/vllm_job_work/"${SLURM_JOB_ID}"/*)
        rm -rf -- "$PD_WORK_DIR"
        echo "pd_work_dir_removed=${PD_WORK_DIR}"
        ;;
      *)
        echo "pd_work_dir_preserved_untrusted_path=${PD_WORK_DIR}" >&2
        ;;
    esac
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_url() {
  local label="$1"
  local url="$2"
  local elapsed=0
  while [ "$elapsed" -lt "$STARTUP_TIMEOUT_SECONDS" ]; do
    if curl -fsS --connect-timeout 2 --max-time 5 "$url" >/dev/null 2>&1; then
      echo "pd_ready label=${label} elapsed_s=${elapsed} url=${url}"
      return 0
    fi
    for pid in "${STEP_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || {
        echo "pd_step_died_while_waiting label=${label} pid=${pid}" >&2
        return 1
      }
    done
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "pd_readiness_timeout label=${label} url=${url}" >&2
  return 1
}

wait_for_registry() {
  local expected_prefill="$1"
  local expected_decode="$2"
  local registry_file="$3"
  local elapsed=0
  while [ "$elapsed" -lt "$STARTUP_TIMEOUT_SECONDS" ]; do
    if curl -fsS --connect-timeout 2 --max-time 5 \
      "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$registry_file" && \
      "$PYTHON_BIN" -c '
import json
import sys

path, expected_p, expected_d = sys.argv[1:]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ready = (
    len(data.get("prefill", [])) == int(expected_p)
    and len(data.get("decode", [])) == int(expected_d)
)
raise SystemExit(0 if ready else 1)
' "$registry_file" "$expected_prefill" "$expected_decode"; then
      echo "pd_registry_ready prefill=${expected_prefill} decode=${expected_decode} elapsed_s=${elapsed}"
      return 0
    fi
    for pid in "${STEP_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || {
        echo "pd_step_died_while_waiting_for_registry pid=${pid}" >&2
        return 1
      }
    done
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "pd_registry_timeout prefill=${expected_prefill} decode=${expected_decode}" >&2
  return 1
}

launch_instance() {
  local role="$1"
  local ordinal="$2"
  local node node_ip net_iface tp_size http_port kv_port instance_name gpu_model
  local kv_offset=0
  local index
  if [ "$role" = prefill ]; then
    node="$PREFILL_NODE"
    node_ip="$PREFILL_IP"
    net_iface="$PREFILL_NET_IFACE"
    tp_size="${PREFILL_TP_SIZE_ARRAY[$ordinal]}"
    instance_name="Prefill_${ordinal}"
    gpu_model="$PREFILL_GPU_MODEL"
    http_port=$((PREFILL_HTTP_PORT_BASE + ordinal))
    for ((index = 0; index < ordinal; index++)); do
      kv_offset=$((kv_offset + PREFILL_TP_SIZE_ARRAY[index]))
    done
    kv_port=$((PREFILL_KV_PORT_BASE + kv_offset))
  else
    node="$DECODE_NODE"
    node_ip="$DECODE_IP"
    net_iface="$DECODE_NET_IFACE"
    tp_size="${DECODE_TP_SIZE_ARRAY[$ordinal]}"
    instance_name="Decode_${ordinal}"
    gpu_model="$DECODE_GPU_MODEL"
    http_port=$((DECODE_HTTP_PORT_BASE + ordinal))
    for ((index = 0; index < ordinal; index++)); do
      kv_offset=$((kv_offset + DECODE_TP_SIZE_ARRAY[index]))
    done
    kv_port=$((DECODE_KV_PORT_BASE + kv_offset))
  fi
  local log_file="${PD_OUT_DIR}/${instance_name}.log"
  echo "pd_launch instance=${instance_name} role=${role} node=${node} gpu_model=${gpu_model} http_port=${http_port} kv_port=${kv_port} tp=${tp_size}"
  # Do not use --overlap for GPU steps: each step must own a different GPU.
  srun --exclusive --kill-on-bad-exit=1 --exact --nodes=1 --nodelist="$node" \
    --ntasks=1 --ntasks-per-node=1 --gpus-per-task="$tp_size" \
    --cpus-per-task="$((CPUS_PER_GPU * tp_size))" --mem="$VLLM_STEP_MEM" \
    env \
      MODEL="$MODEL" VLLM_BIN="$VLLM_BIN" PYTHON_BIN="$PYTHON_BIN" \
      PD_INSTANCE_NAME="$instance_name" PD_EXPECTED_GPU_MODEL="$gpu_model" \
      PD_NODE_IP="$node_ip" PD_NET_IFACE="$net_iface" \
      PD_NODE_NAME="$node" \
      PD_HTTP_PORT="$http_port" PD_KV_PORT="$kv_port" PD_TP_SIZE="$tp_size" \
      PROXY_IP="$PROXY_IP" PROXY_HTTP_PORT="$PROXY_HTTP_PORT" \
      PROXY_REGISTER_PORT="$PROXY_REGISTER_PORT" \
      PD_KV_CONNECTOR="${PD_KV_CONNECTOR:-NixlConnector}" \
      PD_KV_LOAD_FAILURE_POLICY="${PD_KV_LOAD_FAILURE_POLICY:-fail}" \
      PD_OUT_DIR="$PD_OUT_DIR" PD_WORK_DIR="$PD_WORK_DIR" \
      PD_ENABLE_PREDICTIVE_DVFS="$PD_ENABLE_PREDICTIVE_DVFS" \
      PD_CLOCK_AGENT_SCRIPT="$RUNTIME_CLOCK_AGENT_SCRIPT" \
      PD_DVFS_TELEMETRY_INTERVAL_SECONDS="${PD_DVFS_TELEMETRY_INTERVAL_SECONDS:-0.5}" \
      PD_DVFS_SETTLE_SECONDS="${PD_DVFS_SETTLE_SECONDS:-0}" \
      CUSTOM_POLICY_ADMIN_TOKEN="${CUSTOM_POLICY_ADMIN_TOKEN:-}" \
      HOME="$HOME" NIXL_CONFIG_FILE="$NIXL_CONFIG_FILE" \
      XDG_CACHE_HOME="$XDG_CACHE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" \
      XDG_DATA_HOME="$XDG_DATA_HOME" XDG_STATE_HOME="$XDG_STATE_HOME" \
      XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
      MAX_NUM_SEQS="$MAX_NUM_SEQS" \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" DTYPE="$DTYPE" \
      ENFORCE_EAGER="$ENFORCE_EAGER" ATTENTION_BACKEND="$ATTENTION_BACKEND" \
      REQUIRE_LINK_SPEED_MBPS="$REQUIRE_LINK_SPEED_MBPS" \
      "$RUNTIME_INSTANCE_SCRIPT" "$role" "$ordinal" \
      >"$log_file" 2>&1 &
  STEP_PIDS+=("$!")
  wait_for_url "$instance_name" "http://${node_ip}:${http_port}/v1/models"
  local instance_type
  if [ "$role" = prefill ]; then
    instance_type=P
  else
    instance_type=D
  fi
  local registration
  registration=$("$PYTHON_BIN" -c '
import json
import sys

instance_type, http_address, kv_address, tp_size, instance_name, node_name, gpu_type = sys.argv[1:]
print(json.dumps({
    "type": instance_type,
    "http_address": http_address,
    "kv_address": kv_address,
    "tp_size": int(tp_size),
    "instance_name": instance_name,
    "node_name": node_name,
    "gpu_type": gpu_type.lower(),
}))
' "$instance_type" "${node_ip}:${http_port}" "${node_ip}:${kv_port}" "$tp_size" \
    "$instance_name" "$node" "$gpu_model")
  local register_auth_args=()
  if [ -n "${CUSTOM_POLICY_ADMIN_TOKEN:-}" ]; then
    register_auth_args=(-H "X-Admin-Token: ${CUSTOM_POLICY_ADMIN_TOKEN}")
  fi
  curl -fsS --connect-timeout 2 --max-time 5 \
    "${register_auth_args[@]}" \
    -H 'Content-Type: application/json' \
    --data "$registration" \
    "http://${PROXY_IP}:${PROXY_HTTP_PORT}/control/register-instance" \
    >/dev/null
  echo "pd_instance_registered instance=${instance_name} alias=${instance_type}${ordinal} connector=${PD_KV_CONNECTOR:-NixlConnector}"
}

print_instance_mapping() {
  local ordinal index kv_offset tp_size
  for ((ordinal = 0; ordinal < PREFILL_REPLICAS; ordinal++)); do
    kv_offset=0
    for ((index = 0; index < ordinal; index++)); do
      kv_offset=$((kv_offset + PREFILL_TP_SIZE_ARRAY[index]))
    done
    tp_size="${PREFILL_TP_SIZE_ARRAY[$ordinal]}"
    echo "pd_instance_map instance=Prefill_${ordinal} alias=P${ordinal} role=prefill node=${PREFILL_NODE} node_ip=${PREFILL_IP} http_endpoint=${PREFILL_IP}:$((PREFILL_HTTP_PORT_BASE + ordinal)) kv_endpoint=${PREFILL_IP}:$((PREFILL_KV_PORT_BASE + kv_offset)) gpu=${PREFILL_GPU_MODEL} tp=${tp_size}"
  done
  for ((ordinal = 0; ordinal < DECODE_REPLICAS; ordinal++)); do
    kv_offset=0
    for ((index = 0; index < ordinal; index++)); do
      kv_offset=$((kv_offset + DECODE_TP_SIZE_ARRAY[index]))
    done
    tp_size="${DECODE_TP_SIZE_ARRAY[$ordinal]}"
    echo "pd_instance_map instance=Decode_${ordinal} alias=D${ordinal} role=decode node=${DECODE_NODE} node_ip=${DECODE_IP} http_endpoint=${DECODE_IP}:$((DECODE_HTTP_PORT_BASE + ordinal)) kv_endpoint=${DECODE_IP}:$((DECODE_KV_PORT_BASE + kv_offset)) gpu=${DECODE_GPU_MODEL} tp=${tp_size}"
  done
}

echo "pd_topology connector=${PD_KV_CONNECTOR:-NixlConnector} proxy=${PROXY_NODE}/${PROXY_IP}:${PROXY_HTTP_PORT} prefill=${PREFILL_NODE}/${PREFILL_IP} replicas=${PREFILL_REPLICAS} gpu=${PREFILL_GPU_MODEL} tp_sizes=${PREFILL_TP_SIZES} decode=${DECODE_NODE}/${DECODE_IP} replicas=${DECODE_REPLICAS} gpu=${DECODE_GPU_MODEL} tp_sizes=${DECODE_TP_SIZES}"
echo "pd_paths output=${PD_OUT_DIR} work=${PD_WORK_DIR} proxy_script=${RUNTIME_PROXY_SCRIPT}"
echo "pd_cache_paths home=${HOME} nixl_config=${NIXL_CONFIG_FILE} hf=${HF_HOME} hf_hub=${HF_HUB_CACHE} hf_assets=${HF_ASSETS_CACHE} hf_xet=${HF_XET_CACHE} xdg=${XDG_CACHE_HOME} xdg_config=${XDG_CONFIG_HOME} xdg_data=${XDG_DATA_HOME} xdg_state=${XDG_STATE_HOME} xdg_runtime=${XDG_RUNTIME_DIR} flashinfer=${FLASHINFER_WORKSPACE_BASE} vllm=${VLLM_CACHE_ROOT} vllm_config=${VLLM_CONFIG_ROOT} torch=${TORCH_HOME} torch_extensions=${TORCH_EXTENSIONS_DIR} triton=${TRITON_CACHE_DIR} torchinductor=${TORCHINDUCTOR_CACHE_DIR} cuda=${CUDA_CACHE_PATH} numba=${NUMBA_CACHE_DIR} ray_tmp=${RAY_TMPDIR} pip=${PIP_CACHE_DIR} uv=${UV_CACHE_DIR} pycache=${PYTHONPYCACHEPREFIX} tiktoken=${TIKTOKEN_CACHE_DIR} matplotlib=${MPLCONFIGDIR} cupy=${CUPY_CACHE_DIR} tmp=${TMPDIR}"
print_instance_mapping

# Any relative files created by the proxy or its child processes also land in
# the disposable per-job tree.
cd "$PD_WORK_DIR"

srun --overlap --kill-on-bad-exit=1 --exact --nodes=1 --nodelist="$PROXY_NODE" \
  --ntasks=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --gres=none \
  env PROXY_REGISTER_HOST=0.0.0.0 PROXY_REGISTER_PORT="$PROXY_REGISTER_PORT" \
    PROXY_HTTP_HOST=0.0.0.0 PROXY_HTTP_PORT="$PROXY_HTTP_PORT" \
    PREFILL_HTTP_PORT_BASE="$PREFILL_HTTP_PORT_BASE" \
    DECODE_HTTP_PORT_BASE="$DECODE_HTTP_PORT_BASE" \
    PREFILL_TP_SIZES="$PREFILL_TP_SIZES" \
    DECODE_TP_SIZES="$DECODE_TP_SIZES" \
    PD_KV_CONNECTOR="${PD_KV_CONNECTOR:-NixlConnector}" \
    CUSTOM_PD_DEFAULT_ROUTE="${CUSTOM_PD_DEFAULT_ROUTE:-random}" \
    CUSTOM_PD_RANDOM_SEED="${CUSTOM_PD_RANDOM_SEED:-}" \
    CUSTOM_PD_ALLOW_ASYMMETRIC_TP="${CUSTOM_PD_ALLOW_ASYMMETRIC_TP:-false}" \
    CUSTOM_POLICY_ADMIN_TOKEN="${CUSTOM_POLICY_ADMIN_TOKEN:-}" \
    PD_ENABLE_PREDICTIVE_DVFS="$PD_ENABLE_PREDICTIVE_DVFS" \
    PD_DVFS_SCHEDULER_SCRIPT="$RUNTIME_DVFS_SCHEDULER_SCRIPT" \
    PD_DVFS_MODEL_BUNDLE="$RUNTIME_DVFS_MODEL_BUNDLE" \
    PD_DVFS_SATURATION_BUNDLE="$RUNTIME_DVFS_SATURATION_BUNDLE" \
    PD_DVFS_SLO_TTFT_MS="${PD_DVFS_SLO_TTFT_MS:-500}" \
    PD_DVFS_SLO_TPOT_MS="${PD_DVFS_SLO_TPOT_MS:-200}" \
    PD_DVFS_EXPECTED_REQUEST_RATE="${PD_DVFS_EXPECTED_REQUEST_RATE:-0}" \
    PD_DVFS_MIN_REQUEST_RATE="${PD_DVFS_MIN_REQUEST_RATE:-0.25}" \
    PD_DVFS_RATE_WINDOW_SECONDS="${PD_DVFS_RATE_WINDOW_SECONDS:-10}" \
    PD_DVFS_CLOCK_TIMEOUT_SECONDS="${PD_DVFS_CLOCK_TIMEOUT_SECONDS:-30}" \
    PD_DVFS_CLOCK_TOLERANCE_MHZ="${PD_DVFS_CLOCK_TOLERANCE_MHZ:-30}" \
    PD_DVFS_SETTLE_SECONDS="${PD_DVFS_SETTLE_SECONDS:-0}" \
    PD_DVFS_OVERLOAD_ACTION="${PD_DVFS_OVERLOAD_ACTION:-reject}" \
    PD_DVFS_INPUT_TOKENS_OVERRIDE="${PD_DVFS_INPUT_TOKENS_OVERRIDE:-}" \
    PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS="${PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS:-}" \
    PD_DVFS_DISPATCH_MS="${PD_DVFS_DISPATCH_MS:-0}" \
    PD_DVFS_DECISIONS_FILE="${PD_OUT_DIR}/request_dvfs_decisions.jsonl" \
    PD_REQUEST_TRACE_FILE="${PD_REQUEST_TRACE_FILE:-}" \
    PD_OUT_DIR="$PD_OUT_DIR" \
    "$PYTHON_BIN" -u "$RUNTIME_PROXY_SCRIPT" >"${PD_OUT_DIR}/proxy.log" 2>&1 &
STEP_PIDS+=("$!")
wait_for_url proxy "http://${PROXY_IP}:${PROXY_HTTP_PORT}/health"

# Consumers are made ready first. This avoids exposing a registered producer
# to a Decode peer whose HTTP service is still loading the model.
for ((ordinal = 0; ordinal < DECODE_REPLICAS; ordinal++)); do
  launch_instance decode "$ordinal"
done
for ((ordinal = 0; ordinal < PREFILL_REPLICAS; ordinal++)); do
  launch_instance prefill "$ordinal"
done

REGISTRY_FILE="${PD_OUT_DIR}/registry.json"
wait_for_registry "$PREFILL_REPLICAS" "$DECODE_REPLICAS" "$REGISTRY_FILE"
"$PYTHON_BIN" -c '
import json

import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(data, indent=2))
' "$REGISTRY_FILE"

echo "pd_cluster_ready endpoint=http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1 prefill_replicas=${PREFILL_REPLICAS} decode_replicas=${DECODE_REPLICAS} connector=${PD_KV_CONNECTOR:-NixlConnector} default_route=${CUSTOM_PD_DEFAULT_ROUTE:-random} allow_asymmetric_tp=${CUSTOM_PD_ALLOW_ASYMMETRIC_TP:-false}"
echo "pd_cluster_waiting_for_steps=true"
wait -n "${STEP_PIDS[@]}"
echo "pd_step_exited=true"
exit 1
