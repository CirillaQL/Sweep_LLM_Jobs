#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-2p2d-rr-bench
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/slurm-%j.err

set -uo pipefail

SOURCE_JOB_NAME="20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench"
JOB_NAME="${JOB_NAME_OVERRIDE:-$SOURCE_JOB_NAME}"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SOURCE_JOB_DIR="${BROKER_ROOT}/jobs/${SOURCE_JOB_NAME}"
SCRIPT_ROOT="${BROKER_ROOT}/script"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
MODEL="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"

NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE:-32}"
REQUEST_RATE="${REQUEST_RATE_OVERRIDE:-4}"
INPUT_LEN="${INPUT_LEN_OVERRIDE:-512}"
OUTPUT_LEN="${OUTPUT_LEN_OVERRIDE:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY_OVERRIDE:-16}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS_OVERRIDE:-300}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS_OVERRIDE:-900}"

PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
export PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
export PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
export PREFILL_HTTP_PORT_BASE=$((32000 + PORT_OFFSET))
export DECODE_HTTP_PORT_BASE=$((33000 + PORT_OFFSET))
export PREFILL_KV_PORT_BASE=$((34000 + PORT_OFFSET))
export DECODE_KV_PORT_BASE=$((34000 + PORT_OFFSET))

OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
export PD_OUT_DIR="${OUT_DIR}/pd_runtime"
export PD_WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
export HF_HOME="${PD_WORK_DIR}/huggingface"
export HF_TOKEN_PATH="${HF_HOME}/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
export XDG_CACHE_HOME="${PD_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${PD_WORK_DIR}/xdg-config"
export FLASHINFER_WORKSPACE_BASE="${PD_WORK_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${PD_WORK_DIR}/vllm-cache"
export TORCH_HOME="${PD_WORK_DIR}/torch-cache"
export TORCH_EXTENSIONS_DIR="${PD_WORK_DIR}/torch-extensions"
export TRITON_CACHE_DIR="${PD_WORK_DIR}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${PD_WORK_DIR}/torchinductor-cache"
export CUDA_CACHE_PATH="${PD_WORK_DIR}/cuda-cache"
export NUMBA_CACHE_DIR="${PD_WORK_DIR}/numba-cache"
export RAY_TMPDIR="${PD_WORK_DIR}/ray-tmp"
export TMPDIR="${PD_WORK_DIR}/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
BENCH_LOG="${OUT_DIR}/vllm_bench.txt"
BENCH_RESULT="vllm_bench_detailed.json"
REGISTRY_FILE="${OUT_DIR}/registry.json"
ROUTE_CHECK_FILE="${OUT_DIR}/round_robin_check.json"
CHECKER="${SOURCE_JOB_DIR}/check_round_robin.py"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN VLLM_BIN MODEL STARTUP_TIMEOUT_SECONDS
export PROXY_SCRIPT="${SCRIPT_ROOT}/scheduler_custom_policy.py"
export INSTANCE_SCRIPT="${SCRIPT_ROOT}/start_pd_vllm_instance.sh"
export PREFILL_NODE=uranus
export PREFILL_IP=10.1.0.5
export PREFILL_NET_IFACE=enp160s0f0np0
export PREFILL_REPLICAS=2
export PREFILL_TP_SIZE=1
export PREFILL_GPU_MODEL=L40S
export DECODE_NODE=ganymede
export DECODE_IP=10.1.0.3
export DECODE_NET_IFACE=eno12399np0
export DECODE_REPLICAS=2
export DECODE_TP_SIZE=1
export DECODE_GPU_MODEL=L4
export PROXY_NODE=ganymede
export PROXY_IP=10.1.0.3
export CUSTOM_PD_DEFAULT_ROUTE=round_robin
export MAX_MODEL_LEN=8192
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=32
export GPU_MEMORY_UTILIZATION=0.82
export DTYPE=float16
export ENFORCE_EAGER=true
export CPUS_PER_GPU=8
export VLLM_STEP_MEM=32G

PD_LAUNCHER_PID=""

stop_launcher() {
  if [ -n "$PD_LAUNCHER_PID" ] && kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
    kill "$PD_LAUNCHER_PID" 2>/dev/null || true
    wait "$PD_LAUNCHER_PID" 2>/dev/null || true
  fi
  PD_LAUNCHER_PID=""
}

cleanup() {
  local cleanup_rc=$?
  trap - EXIT INT TERM
  stop_launcher
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] && \
    [ "$PD_WORK_DIR" = "/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}" ]; then
    if rm -rf -- "$PD_WORK_DIR"; then
      echo "cache_cleanup_complete=true path=${PD_WORK_DIR}"
    else
      echo "cache_cleanup_complete=false removal_failed path=${PD_WORK_DIR}" >&2
      cleanup_rc=3
    fi
  else
    echo "cache_cleanup_complete=false unsafe_path=${PD_WORK_DIR}" >&2
    cleanup_rc=3
  fi
  echo "job_cleanup_rc=${cleanup_rc}"
  exit "$cleanup_rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_registry() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
      echo "pd_launcher_exited_before_ready=true"
      return 1
    fi
    if curl -fsS --connect-timeout 2 --max-time 5 \
      "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" && \
      "$PYTHON_BIN" -c '
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
ready = (
    data.get("prefill_count") == 2
    and data.get("decode_count") == 2
    and data.get("default_route") == "round_robin"
)
raise SystemExit(0 if ready else 1)
' "$REGISTRY_FILE"; then
      echo "pd_registry_ready=true registry=${REGISTRY_FILE}"
      return 0
    fi
    sleep 2
  done
  echo "pd_registry_ready=false timeout_s=${STARTUP_TIMEOUT_SECONDS}"
  return 1
}

mkdir -p "$OUT_DIR" "$PD_OUT_DIR"
ulimit -n 65536

echo "job_name=${JOB_NAME}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "model=${MODEL}"
echo "work_dir=${PD_WORK_DIR}"
echo "cache_paths hf=${HF_HOME} hf_hub=${HF_HUB_CACHE} hf_assets=${HF_ASSETS_CACHE} hf_xet=${HF_XET_CACHE} xdg=${XDG_CACHE_HOME} xdg_config=${XDG_CONFIG_HOME} flashinfer=${FLASHINFER_WORKSPACE_BASE} vllm=${VLLM_CACHE_ROOT} torch=${TORCH_HOME} torch_extensions=${TORCH_EXTENSIONS_DIR} triton=${TRITON_CACHE_DIR} torchinductor=${TORCHINDUCTOR_CACHE_DIR} cuda=${CUDA_CACHE_PATH} numba=${NUMBA_CACHE_DIR} ray_tmp=${RAY_TMPDIR} tmp=${TMPDIR}"
echo "round_robin_order=P0-D0,P0-D1,P1-D0,P1-D1"
echo "bench num_prompts=${NUM_PROMPTS} request_rate=${REQUEST_RATE} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN} max_concurrency=${MAX_CONCURRENCY}"
echo "output_dir=${OUT_DIR}"

for required in \
  "${SCRIPT_ROOT}/start_pd_vllm.sh" \
  "$INSTANCE_SCRIPT" \
  "$PROXY_SCRIPT" \
  "$CHECKER"; do
  if [ ! -s "$required" ]; then
    echo "required_artifact_missing=${required}"
    exit 2
  fi
done
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true python=${PYTHON_BIN} vllm=${VLLM_BIN}"
  exit 3
fi
if ! command -v curl >/dev/null 2>&1 || ! command -v timeout >/dev/null 2>&1; then
  echo "required_command_missing=true"
  exit 4
fi

echo "phase=start_pd_cluster"
"${SCRIPT_ROOT}/start_pd_vllm.sh" &
PD_LAUNCHER_PID=$!
wait_for_registry || exit 5
"$PYTHON_BIN" -m json.tool "$REGISTRY_FILE"

echo "phase=vllm_bench"
timeout --signal=INT --kill-after=20s "${BENCH_TIMEOUT_SECONDS}s" \
  "$VLLM_BIN" bench serve \
  --backend openai \
  --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --num-warmups 0 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --result-dir "$OUT_DIR" \
  --result-filename "$BENCH_RESULT" \
  </dev/null > "$BENCH_LOG" 2>&1
BENCH_RC=$?
cat "$BENCH_LOG"

read -r SUCCESSFUL_REQUESTS FAILED_REQUESTS <<< "$("$PYTHON_BIN" -c '
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
successful = re.search(r"Successful requests:\s+(\d+)", text)
failed = re.search(r"Failed requests:\s+(\d+)", text)
print(successful.group(1) if successful else -1, failed.group(1) if failed else -1)
' "$BENCH_LOG")"
echo "bench_rc=${BENCH_RC} successful_requests=${SUCCESSFUL_REQUESTS} failed_requests=${FAILED_REQUESTS}"

echo "phase=verify_round_robin"
ROUTE_CHECK_RC=0
"$PYTHON_BIN" "$CHECKER" \
  --proxy-log "${PD_OUT_DIR}/proxy.log" \
  --expected-requests "$NUM_PROMPTS" \
  --output "$ROUTE_CHECK_FILE" || ROUTE_CHECK_RC=$?
cat "$ROUTE_CHECK_FILE" 2>/dev/null || true
echo "round_robin_check_rc=${ROUTE_CHECK_RC}"

FINAL_RC=0
if [ "$BENCH_RC" -ne 0 ] || \
  [ "$SUCCESSFUL_REQUESTS" -ne "$NUM_PROMPTS" ] || \
  [ "$FAILED_REQUESTS" -ne 0 ]; then
  FINAL_RC=10
fi
if [ "$ROUTE_CHECK_RC" -ne 0 ]; then
  FINAL_RC=11
fi
if ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
  echo "pd_launcher_alive_after_bench=false"
  FINAL_RC=12
else
  echo "pd_launcher_alive_after_bench=true"
fi

echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
