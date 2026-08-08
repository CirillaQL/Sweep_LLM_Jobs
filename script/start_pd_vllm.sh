#!/usr/bin/env bash
# Launch four independent TP=1 vLLM instances in an existing Slurm allocation:
# Prefill_0/Prefill_1 on L40S and Decode_0/Decode_1 on L4.

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
[ "$PREFILL_REPLICAS" -eq 2 ] && [ "$DECODE_REPLICAS" -eq 2 ] || \
  die "custom_scheduler_requires_exactly_two_prefill_and_two_decode_instances"
[ "$PREFILL_TP_SIZE" -eq 1 ] && [ "$DECODE_TP_SIZE" -eq 1 ] || \
  die "four_instance_topology_requires_tp_size_one"

[ -x "$PYTHON_BIN" ] || die "python_binary_not_executable path=${PYTHON_BIN}"
[ -x "$VLLM_BIN" ] || die "vllm_binary_not_executable path=${VLLM_BIN}"
[ -r "$PROXY_SCRIPT" ] || die "proxy_script_not_readable path=${PROXY_SCRIPT}"
[ -x "$INSTANCE_SCRIPT" ] || die "instance_script_not_executable path=${INSTANCE_SCRIPT}"

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

PORT_OFFSET="${PD_PORT_OFFSET:-$((SLURM_JOB_ID % 500))}"
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-$((30000 + PORT_OFFSET))}"
PROXY_REGISTER_PORT="${PROXY_REGISTER_PORT:-$((31000 + PORT_OFFSET))}"
PREFILL_HTTP_PORT_BASE="${PREFILL_HTTP_PORT_BASE:-$((32000 + PORT_OFFSET))}"
DECODE_HTTP_PORT_BASE="${DECODE_HTTP_PORT_BASE:-$((33000 + PORT_OFFSET))}"
PREFILL_KV_PORT_BASE="${PREFILL_KV_PORT_BASE:-$((34000 + PORT_OFFSET))}"
DECODE_KV_PORT_BASE="${DECODE_KV_PORT_BASE:-$((34000 + PORT_OFFSET))}"

PD_OUT_DIR="${PD_OUT_DIR:-$PWD/pd_vllm_${SLURM_JOB_ID}}"
PD_WORK_DIR="${PD_WORK_DIR:-/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-${PD_WORK_DIR}/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME}/token}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PD_WORK_DIR}/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${PD_WORK_DIR}/xdg-config}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PD_WORK_DIR}/flashinfer}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PD_WORK_DIR}/vllm-cache}"
export TORCH_HOME="${TORCH_HOME:-${PD_WORK_DIR}/torch-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PD_WORK_DIR}/torch-extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PD_WORK_DIR}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PD_WORK_DIR}/torchinductor-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PD_WORK_DIR}/cuda-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PD_WORK_DIR}/numba-cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${PD_WORK_DIR}/ray-tmp}"
export TMPDIR="${TMPDIR:-${PD_WORK_DIR}/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p \
  "$PD_OUT_DIR" "$PD_WORK_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$RAY_TMPDIR" "$TMPDIR"
RUNTIME_PROXY_SCRIPT="${PD_WORK_DIR}/scheduler_custom_policy.py"
RUNTIME_INSTANCE_SCRIPT="${PD_WORK_DIR}/start_pd_vllm_instance.sh"
cp -- "$PROXY_SCRIPT" "$RUNTIME_PROXY_SCRIPT"
cp -- "$INSTANCE_SCRIPT" "$RUNTIME_INSTANCE_SCRIPT"
chmod +x "$RUNTIME_PROXY_SCRIPT" "$RUNTIME_INSTANCE_SCRIPT"

STEP_PIDS=()
cleanup_done=false
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
  if [ "$role" = prefill ]; then
    node="$PREFILL_NODE"
    node_ip="$PREFILL_IP"
    net_iface="$PREFILL_NET_IFACE"
    tp_size="$PREFILL_TP_SIZE"
    instance_name="Prefill_${ordinal}"
    gpu_model="$PREFILL_GPU_MODEL"
    http_port=$((PREFILL_HTTP_PORT_BASE + ordinal))
    kv_port=$((PREFILL_KV_PORT_BASE + ordinal * tp_size))
  else
    node="$DECODE_NODE"
    node_ip="$DECODE_IP"
    net_iface="$DECODE_NET_IFACE"
    tp_size="$DECODE_TP_SIZE"
    instance_name="Decode_${ordinal}"
    gpu_model="$DECODE_GPU_MODEL"
    http_port=$((DECODE_HTTP_PORT_BASE + ordinal))
    kv_port=$((DECODE_KV_PORT_BASE + ordinal * tp_size))
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
      PD_HTTP_PORT="$http_port" PD_KV_PORT="$kv_port" PD_TP_SIZE="$tp_size" \
      PROXY_IP="$PROXY_IP" PROXY_REGISTER_PORT="$PROXY_REGISTER_PORT" \
      PD_OUT_DIR="$PD_OUT_DIR" PD_WORK_DIR="$PD_WORK_DIR" \
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
}

print_instance_mapping() {
  local ordinal
  for ((ordinal = 0; ordinal < PREFILL_REPLICAS; ordinal++)); do
    echo "pd_instance_map instance=Prefill_${ordinal} alias=P${ordinal} role=prefill node=${PREFILL_NODE} node_ip=${PREFILL_IP} http_endpoint=${PREFILL_IP}:$((PREFILL_HTTP_PORT_BASE + ordinal)) kv_endpoint=${PREFILL_IP}:$((PREFILL_KV_PORT_BASE + ordinal * PREFILL_TP_SIZE)) gpu=${PREFILL_GPU_MODEL} tp=${PREFILL_TP_SIZE}"
  done
  for ((ordinal = 0; ordinal < DECODE_REPLICAS; ordinal++)); do
    echo "pd_instance_map instance=Decode_${ordinal} alias=D${ordinal} role=decode node=${DECODE_NODE} node_ip=${DECODE_IP} http_endpoint=${DECODE_IP}:$((DECODE_HTTP_PORT_BASE + ordinal)) kv_endpoint=${DECODE_IP}:$((DECODE_KV_PORT_BASE + ordinal * DECODE_TP_SIZE)) gpu=${DECODE_GPU_MODEL} tp=${DECODE_TP_SIZE}"
  done
}

echo "pd_topology proxy=${PROXY_NODE}/${PROXY_IP}:${PROXY_HTTP_PORT} prefill=${PREFILL_NODE}/${PREFILL_IP} replicas=${PREFILL_REPLICAS} gpu=${PREFILL_GPU_MODEL} tp=${PREFILL_TP_SIZE} decode=${DECODE_NODE}/${DECODE_IP} replicas=${DECODE_REPLICAS} gpu=${DECODE_GPU_MODEL} tp=${DECODE_TP_SIZE}"
echo "pd_paths output=${PD_OUT_DIR} work=${PD_WORK_DIR} proxy_script=${RUNTIME_PROXY_SCRIPT}"
echo "pd_cache_paths hf=${HF_HOME} hf_hub=${HF_HUB_CACHE} hf_assets=${HF_ASSETS_CACHE} hf_xet=${HF_XET_CACHE} xdg=${XDG_CACHE_HOME} xdg_config=${XDG_CONFIG_HOME} flashinfer=${FLASHINFER_WORKSPACE_BASE} vllm=${VLLM_CACHE_ROOT} torch=${TORCH_HOME} torch_extensions=${TORCH_EXTENSIONS_DIR} triton=${TRITON_CACHE_DIR} torchinductor=${TORCHINDUCTOR_CACHE_DIR} cuda=${CUDA_CACHE_PATH} numba=${NUMBA_CACHE_DIR} ray_tmp=${RAY_TMPDIR} tmp=${TMPDIR}"
print_instance_mapping

srun --overlap --kill-on-bad-exit=1 --exact --nodes=1 --nodelist="$PROXY_NODE" \
  --ntasks=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --gres=none \
  env PROXY_REGISTER_HOST=0.0.0.0 PROXY_REGISTER_PORT="$PROXY_REGISTER_PORT" \
    PROXY_HTTP_HOST=0.0.0.0 PROXY_HTTP_PORT="$PROXY_HTTP_PORT" \
    CUSTOM_PD_DEFAULT_ROUTE="${CUSTOM_PD_DEFAULT_ROUTE:-random}" \
    CUSTOM_PD_RANDOM_SEED="${CUSTOM_PD_RANDOM_SEED:-}" \
    CUSTOM_POLICY_ADMIN_TOKEN="${CUSTOM_POLICY_ADMIN_TOKEN:-}" \
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

echo "pd_cluster_ready endpoint=http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1 instances=Prefill_0,Prefill_1,Decode_0,Decode_1 default_route=${CUSTOM_PD_DEFAULT_ROUTE:-random} random_selection=P[bit0]-D[bit1]"
echo "pd_cluster_waiting_for_steps=true"
wait -n "${STEP_PIDS[@]}"
echo "pd_step_exited=true"
exit 1
