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
REQUIRE_ALL_RANDOM_ROUTES="${REQUIRE_ALL_RANDOM_ROUTES_OVERRIDE:-false}"
WARMUP_ALL_ROUTES="${WARMUP_ALL_ROUTES_OVERRIDE:-false}"
# Route warmup must use the measured output shape.  The predictive DVFS model
# is not calibrated for OL=1 and can otherwise reject every clock candidate.
WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS_OVERRIDE:-$OUTPUT_LEN}"
WARMUP_TIMEOUT_SECONDS="${WARMUP_TIMEOUT_SECONDS_OVERRIDE:-300}"
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP="${CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE:-false}"
export PD_ENABLE_PREDICTIVE_DVFS="${PD_ENABLE_PREDICTIVE_DVFS_OVERRIDE:-false}"

PORT_SLOT_COUNT=60
PORT_SLOT_STRIDE=16
PORT_OFFSET=$(((${SLURM_JOB_ID:-0} % PORT_SLOT_COUNT) * PORT_SLOT_STRIDE))
export PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
export PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
export PREFILL_HTTP_PORT_BASE=$((32000 + PORT_OFFSET))
export DECODE_HTTP_PORT_BASE=$((33000 + PORT_OFFSET))
export PREFILL_KV_PORT_BASE=$((34000 + PORT_OFFSET))
export DECODE_KV_PORT_BASE=$((34000 + PORT_OFFSET))

OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
export PD_OUT_DIR="${OUT_DIR}/pd_runtime"
export PD_WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
export HOME="${PD_WORK_DIR}/home"
export HF_HOME="${PD_WORK_DIR}/huggingface"
export HF_TOKEN_PATH="${HF_HOME}/token"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export XDG_CACHE_HOME="${PD_WORK_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${PD_WORK_DIR}/xdg-config"
export XDG_DATA_HOME="${PD_WORK_DIR}/xdg-data"
export XDG_STATE_HOME="${PD_WORK_DIR}/xdg-state"
export XDG_RUNTIME_DIR="${PD_WORK_DIR}/xdg-runtime"
export NIXL_CONFIG_FILE="${PD_WORK_DIR}/nixl/nixl.cfg"
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
BENCH_LOG="${OUT_DIR}/vllm_bench.txt"
BENCH_RESULT="vllm_bench_detailed.json"
REGISTRY_FILE="${OUT_DIR}/registry.json"
ROUTE_CHECK_FILE="${OUT_DIR}/random_route_check.json"
CHECKER="${SOURCE_JOB_DIR}/check_random_routes.py"
DVFS_CHECKER="${SCRIPT_ROOT}/check_request_dvfs.py"
DVFS_CHECK_FILE="${OUT_DIR}/request_dvfs_check.json"
DVFS_TTFT_CORRECTOR="${SCRIPT_ROOT}/correct_vllm_bench_dvfs_ttft.py"
FORWARDING_TTFT_SUMMARIZER="${SCRIPT_ROOT}/summarize_forwarding_ttft.py"
DVFS_CORRECTED_BENCH_RESULT="${OUT_DIR}/vllm_bench_detailed_excluding_dvfs_wait.json"
DVFS_CORRECTED_BENCH_LOG="${OUT_DIR}/vllm_bench_ttft_excluding_dvfs_wait.txt"
FORWARDING_TTFT_SUMMARY="${OUT_DIR}/forwarding_ttft_summary.json"
FORWARDING_TTFT_REQUESTS="${OUT_DIR}/forwarding_ttft_requests.csv"
WARMUP_SCRIPT="${SCRIPT_ROOT}/warmup_pd_routes.py"
WARMUP_RESULT="${OUT_DIR}/route_warmup.json"
WARMUP_DECISIONS="${OUT_DIR}/route_warmup_dvfs_decisions.jsonl"
MEASUREMENT_PROXY_LOG="${OUT_DIR}/proxy_measurement.log"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHON_BIN VLLM_BIN MODEL STARTUP_TIMEOUT_SECONDS
export PROXY_SCRIPT="${SCRIPT_ROOT}/scheduler_custom_policy.py"
export INSTANCE_SCRIPT="${SCRIPT_ROOT}/start_pd_vllm_instance.sh"
export PREFILL_NODE=uranus
export PREFILL_IP=10.1.0.5
export PREFILL_NET_IFACE=enp160s0f0np0
export PREFILL_REPLICAS="${PREFILL_REPLICAS_OVERRIDE:-2}"
export PREFILL_TP_SIZE="${PREFILL_TP_SIZE_OVERRIDE:-1}"
export PREFILL_TP_SIZES="${PREFILL_TP_SIZES_OVERRIDE:-1,1}"
export PREFILL_GPU_MODEL=L40S
export DECODE_NODE=ganymede
export DECODE_IP=10.1.0.3
export DECODE_NET_IFACE=eno12399np0
export DECODE_REPLICAS="${DECODE_REPLICAS_OVERRIDE:-2}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE_OVERRIDE:-1}"
export DECODE_TP_SIZES="${DECODE_TP_SIZES_OVERRIDE:-1,1}"
export DECODE_GPU_MODEL=L4
export PROXY_NODE=ganymede
export PROXY_IP=10.1.0.3
export CUSTOM_PD_DEFAULT_ROUTE=random
export PD_KV_CONNECTOR=NixlConnector
export PD_KV_LOAD_FAILURE_POLICY=fail
export PD_DVFS_SLO_TTFT_MS="${PD_DVFS_SLO_TTFT_MS_OVERRIDE:-500}"
export PD_DVFS_SLO_TPOT_MS="${PD_DVFS_SLO_TPOT_MS_OVERRIDE:-200}"
export PD_DVFS_EXPECTED_REQUEST_RATE="${PD_DVFS_EXPECTED_REQUEST_RATE_OVERRIDE:-$REQUEST_RATE}"
export PD_DVFS_INPUT_TOKENS_OVERRIDE="${PD_DVFS_INPUT_TOKENS_OVERRIDE:-$INPUT_LEN}"
export PD_DVFS_OVERLOAD_ACTION="${PD_DVFS_OVERLOAD_ACTION_OVERRIDE:-reject}"
export PD_DVFS_CLOCK_TOLERANCE_MHZ="${PD_DVFS_CLOCK_TOLERANCE_MHZ_OVERRIDE:-30}"
export PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS="${PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE:-}"
export PD_DVFS_DISPATCH_MS="${PD_DVFS_DISPATCH_MS_OVERRIDE:-0}"
export PD_DVFS_SETTLE_SECONDS="${PD_DVFS_SETTLE_SECONDS_OVERRIDE:-0}"
export PD_DVFS_CLOCK_TIMEOUT_SECONDS="${PD_DVFS_CLOCK_TIMEOUT_SECONDS_OVERRIDE:-30}"
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

path, expected_p, expected_d, expected_p_tp, expected_d_tp, allow_raw = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
prefill_tp = [int(value) for value in expected_p_tp.split(",")]
decode_tp = [int(value) for value in expected_d_tp.split(",")]
allow_asymmetric = allow_raw.lower() == "true"
expected_routes = [
    f"P{p_index}-D{d_index}"
    for p_index, p_tp in enumerate(prefill_tp)
    for d_index, d_tp in enumerate(decode_tp)
    if allow_asymmetric or p_tp == d_tp
]
ready = (
    data.get("prefill_count") == int(expected_p)
    and data.get("decode_count") == int(expected_d)
    and data.get("prefill_tp_sizes") == prefill_tp
    and data.get("decode_tp_sizes") == decode_tp
    and data.get("supported_routes") == expected_routes
    and data.get("allow_asymmetric_tp") == allow_asymmetric
    and data.get("default_route") == "random"
    and data.get("kv_connector") == "NixlConnector"
)
raise SystemExit(0 if ready else 1)
' "$REGISTRY_FILE" "$PREFILL_REPLICAS" "$DECODE_REPLICAS" \
      "$PREFILL_TP_SIZES" "$DECODE_TP_SIZES" \
      "$CUSTOM_PD_ALLOW_ASYMMETRIC_TP"; then
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
echo "cache_paths home=${HOME} nixl_config=${NIXL_CONFIG_FILE} hf=${HF_HOME} hf_hub=${HF_HUB_CACHE} hf_assets=${HF_ASSETS_CACHE} hf_xet=${HF_XET_CACHE} xdg=${XDG_CACHE_HOME} xdg_config=${XDG_CONFIG_HOME} xdg_data=${XDG_DATA_HOME} xdg_state=${XDG_STATE_HOME} xdg_runtime=${XDG_RUNTIME_DIR} flashinfer=${FLASHINFER_WORKSPACE_BASE} vllm=${VLLM_CACHE_ROOT} vllm_config=${VLLM_CONFIG_ROOT} torch=${TORCH_HOME} torch_extensions=${TORCH_EXTENSIONS_DIR} triton=${TRITON_CACHE_DIR} torchinductor=${TORCHINDUCTOR_CACHE_DIR} cuda=${CUDA_CACHE_PATH} numba=${NUMBA_CACHE_DIR} ray_tmp=${RAY_TMPDIR} pip=${PIP_CACHE_DIR} uv=${UV_CACHE_DIR} pycache=${PYTHONPYCACHEPREFIX} tiktoken=${TIKTOKEN_CACHE_DIR} matplotlib=${MPLCONFIGDIR} cupy=${CUPY_CACHE_DIR} tmp=${TMPDIR}"
echo "random_route_policy=prefill_first,allow_asymmetric_tp=${CUSTOM_PD_ALLOW_ASYMMETRIC_TP}"
echo "mixed_tp prefill=${PREFILL_TP_SIZES} decode=${DECODE_TP_SIZES} require_all_routes=${REQUIRE_ALL_RANDOM_ROUTES}"
echo "bench num_prompts=${NUM_PROMPTS} request_rate=${REQUEST_RATE} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN} max_concurrency=${MAX_CONCURRENCY}"
echo "warmup all_routes=${WARMUP_ALL_ROUTES} input_tokens=${INPUT_LEN} output_tokens=${WARMUP_OUTPUT_TOKENS} timeout_seconds=${WARMUP_TIMEOUT_SECONDS}"
echo "predictive_dvfs enabled=${PD_ENABLE_PREDICTIVE_DVFS} slo_ttft_ms=${PD_DVFS_SLO_TTFT_MS} slo_tpot_ms=${PD_DVFS_SLO_TPOT_MS} expected_rate=${PD_DVFS_EXPECTED_REQUEST_RATE} overload_action=${PD_DVFS_OVERLOAD_ACTION} settle_seconds=${PD_DVFS_SETTLE_SECONDS}"
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
if [ "$WARMUP_ALL_ROUTES" = true ] && [ ! -s "$WARMUP_SCRIPT" ]; then
  echo "required_artifact_missing=${WARMUP_SCRIPT}"
  exit 2
fi
if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
  for required in \
    "$DVFS_CHECKER" \
    "$DVFS_TTFT_CORRECTOR" \
    "$FORWARDING_TTFT_SUMMARIZER"; do
    if [ ! -s "$required" ]; then
      echo "required_artifact_missing=${required}"
      exit 2
    fi
  done
fi
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

MEASUREMENT_REQUEST_START=0
PROXY_MEASUREMENT_START_LINE=1
if [ "$WARMUP_ALL_ROUTES" = true ]; then
  echo "phase=warmup_all_pd_routes"
  WARMUP_ARGS=(
    --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}"
    --model "$MODEL"
    --prefill-tp-sizes "$PREFILL_TP_SIZES"
    --decode-tp-sizes "$DECODE_TP_SIZES"
    --input-tokens "$INPUT_LEN"
    --output-tokens "$WARMUP_OUTPUT_TOKENS"
    --timeout-seconds "$WARMUP_TIMEOUT_SECONDS"
    --output "$WARMUP_RESULT"
  )
  if [ "$CUSTOM_PD_ALLOW_ASYMMETRIC_TP" = true ]; then
    WARMUP_ARGS+=(--allow-asymmetric-tp)
  fi
  "$PYTHON_BIN" "$WARMUP_SCRIPT" "${WARMUP_ARGS[@]}" || exit 6
  "$PYTHON_BIN" -m json.tool "$WARMUP_RESULT"
  MEASUREMENT_REQUEST_START="$("$PYTHON_BIN" -c '
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["request_count"])
' "$WARMUP_RESULT")"
  if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
    cp "${PD_OUT_DIR}/request_dvfs_decisions.jsonl" "$WARMUP_DECISIONS"
    : > "${PD_OUT_DIR}/request_dvfs_decisions.jsonl"
  fi
  # The proxy is unbuffered, but allow the final warmup log line to reach disk.
  sleep 1
  PROXY_MEASUREMENT_START_LINE=$(( $(wc -l < "${PD_OUT_DIR}/proxy.log") + 1 ))
  echo "route_warmup_complete=true routes=${MEASUREMENT_REQUEST_START} measurement_proxy_start_line=${PROXY_MEASUREMENT_START_LINE}"
fi

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

tail -n "+${PROXY_MEASUREMENT_START_LINE}" \
  "${PD_OUT_DIR}/proxy.log" > "$MEASUREMENT_PROXY_LOG"

TTFT_CORRECTION_RC=0
if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
  echo "phase=correct_ttft_excluding_dvfs_wait"
  "$PYTHON_BIN" "$DVFS_TTFT_CORRECTOR" \
    --input "${OUT_DIR}/${BENCH_RESULT}" \
    --wait-seconds "$PD_DVFS_SETTLE_SECONDS" \
    --output "$DVFS_CORRECTED_BENCH_RESULT" \
    > "$DVFS_CORRECTED_BENCH_LOG" 2>&1 || TTFT_CORRECTION_RC=$?
  cat "$DVFS_CORRECTED_BENCH_LOG" 2>/dev/null || true
  echo "ttft_correction_rc=${TTFT_CORRECTION_RC}"
fi

echo "phase=verify_random_routes"
ROUTE_CHECK_RC=0
ROUTE_CHECK_ARGS=(
  --proxy-log "$MEASUREMENT_PROXY_LOG" \
  --expected-requests "$NUM_PROMPTS" \
  --expected-count-start "$MEASUREMENT_REQUEST_START" \
  --expected-prefill-count "$PREFILL_REPLICAS" \
  --expected-decode-count "$DECODE_REPLICAS" \
  --expected-prefill-tp-sizes "$PREFILL_TP_SIZES" \
  --expected-decode-tp-sizes "$DECODE_TP_SIZES" \
  --output "$ROUTE_CHECK_FILE"
)
if [ "$REQUIRE_ALL_RANDOM_ROUTES" = true ]; then
  ROUTE_CHECK_ARGS+=(--require-all-routes)
fi
if [ "$CUSTOM_PD_ALLOW_ASYMMETRIC_TP" = true ]; then
  ROUTE_CHECK_ARGS+=(--allow-asymmetric-tp)
fi
"$PYTHON_BIN" "$CHECKER" "${ROUTE_CHECK_ARGS[@]}" || ROUTE_CHECK_RC=$?
cat "$ROUTE_CHECK_FILE" 2>/dev/null || true
echo "random_route_check_rc=${ROUTE_CHECK_RC}"

DVFS_CHECK_RC=0
FORWARDING_TTFT_RC=0
if [ "$PD_ENABLE_PREDICTIVE_DVFS" = true ]; then
  echo "phase=verify_request_dvfs"
  "$PYTHON_BIN" "$DVFS_CHECKER" \
    --decisions "${PD_OUT_DIR}/request_dvfs_decisions.jsonl" \
    --telemetry-dir "$PD_OUT_DIR" \
    --expected-requests "$NUM_PROMPTS" \
    --clock-tolerance-mhz "$PD_DVFS_CLOCK_TOLERANCE_MHZ" \
    --settle-seconds "$PD_DVFS_SETTLE_SECONDS" \
    --output "$DVFS_CHECK_FILE" || DVFS_CHECK_RC=$?
  echo "request_dvfs_check_rc=${DVFS_CHECK_RC}"

  echo "phase=summarize_exact_forwarding_ttft"
  "$PYTHON_BIN" "$FORWARDING_TTFT_SUMMARIZER" \
    --decisions "${PD_OUT_DIR}/request_dvfs_decisions.jsonl" \
    --expected-requests "$NUM_PROMPTS" \
    --slo-ttft-ms "$PD_DVFS_SLO_TTFT_MS" \
    --output "$FORWARDING_TTFT_SUMMARY" \
    --csv-output "$FORWARDING_TTFT_REQUESTS" || FORWARDING_TTFT_RC=$?
  echo "forwarding_ttft_summary_rc=${FORWARDING_TTFT_RC}"
fi

FINAL_RC=0
if [ "$BENCH_RC" -ne 0 ] || \
  [ "$SUCCESSFUL_REQUESTS" -ne "$NUM_PROMPTS" ] || \
  [ "$FAILED_REQUESTS" -ne 0 ]; then
  FINAL_RC=10
fi
if [ "$ROUTE_CHECK_RC" -ne 0 ]; then
  FINAL_RC=11
fi
if [ "$DVFS_CHECK_RC" -ne 0 ]; then
  FINAL_RC=13
fi
if [ "$TTFT_CORRECTION_RC" -ne 0 ]; then
  FINAL_RC=14
fi
if [ "$FORWARDING_TTFT_RC" -ne 0 ]; then
  FINAL_RC=15
fi
if ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
  echo "pd_launcher_alive_after_bench=false"
  FINAL_RC=12
else
  echo "pd_launcher_alive_after_bench=true"
fi

echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
