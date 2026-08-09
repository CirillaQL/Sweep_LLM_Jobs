#!/usr/bin/env bash
#SBATCH --job-name=nixl-kv-length-profile
#SBATCH --partition=long
#SBATCH --time=00:45:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-212136-luqia-vllm-nixl-kv-length-profile/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260809-212136-luqia-vllm-nixl-kv-length-profile/slurm-%j.err

set -uo pipefail

JOB_NAME="20260809-212136-luqia-vllm-nixl-kv-length-profile"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
SCRIPT_ROOT="${BROKER_ROOT}/script"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
MODEL="${MODEL_OVERRIDE:-mistralai/Mistral-7B-v0.1}"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
export PD_OUT_DIR="${OUT_DIR}/pd_runtime"
export PD_WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
export PD_REQUEST_TRACE_FILE="${PD_OUT_DIR}/request_trace.jsonl"
export PYTHON_BIN MODEL
export VLLM_BIN="${ENV_DIR}/bin/vllm"
export PATH="${ENV_DIR}/bin:${PATH}"

export PREFILL_NODE=uranus
export PREFILL_IP=10.1.0.5
export PREFILL_NET_IFACE=enp160s0f0np0
export PREFILL_REPLICAS=1
export PREFILL_TP_SIZE=1
export PREFILL_TP_SIZES=1
export PREFILL_GPU_MODEL=L40S
export DECODE_NODE=ganymede
export DECODE_IP=10.1.0.3
export DECODE_NET_IFACE=eno12399np0
export DECODE_REPLICAS=1
export DECODE_TP_SIZE=1
export DECODE_TP_SIZES=1
export DECODE_GPU_MODEL=L4
export PROXY_NODE=ganymede
export PROXY_IP=10.1.0.3
export PROXY_SCRIPT="${SCRIPT_ROOT}/scheduler_custom_policy.py"
export INSTANCE_SCRIPT="${SCRIPT_ROOT}/start_pd_vllm_instance.sh"
export PD_KV_CONNECTOR=NixlConnector
export PD_KV_LOAD_FAILURE_POLICY=fail
export CUSTOM_PD_DEFAULT_ROUTE=P0-D0
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP=false
export PD_ENABLE_PREDICTIVE_DVFS=false
export STARTUP_TIMEOUT_SECONDS=1500
export MAX_MODEL_LEN=8192
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=8
export GPU_MEMORY_UTILIZATION=0.82
export DTYPE=float16
export ENFORCE_EAGER=true
export CPUS_PER_GPU=8
export VLLM_STEP_MEM=32G

PORT_SLOT_COUNT=60
PORT_SLOT_STRIDE=16
PORT_OFFSET=$(((${SLURM_JOB_ID:-0} % PORT_SLOT_COUNT) * PORT_SLOT_STRIDE))
export PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
export PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
export PREFILL_HTTP_PORT_BASE=$((32000 + PORT_OFFSET))
export DECODE_HTTP_PORT_BASE=$((33000 + PORT_OFFSET))
export PREFILL_KV_PORT_BASE=$((34000 + PORT_OFFSET))
export DECODE_KV_PORT_BASE=$((34000 + PORT_OFFSET))

PD_LAUNCHER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$PD_LAUNCHER_PID" ] && kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
    kill "$PD_LAUNCHER_PID" 2>/dev/null || true
    wait "$PD_LAUNCHER_PID" 2>/dev/null || true
  fi
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] && \
    [ "$PD_WORK_DIR" = "/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}" ]; then
    if [ -d "$PD_WORK_DIR" ]; then
      rm -rf -- "$PD_WORK_DIR"
    fi
    echo "cache_cleanup_complete=true path=${PD_WORK_DIR}"
  else
    echo "cache_cleanup_complete=false unsafe_path=${PD_WORK_DIR}" >&2
    rc=3
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT_DIR" "$PD_OUT_DIR"
ulimit -n 65536

for required in \
  "${SCRIPT_ROOT}/start_pd_vllm.sh" \
  "$PROXY_SCRIPT" \
  "$INSTANCE_SCRIPT" \
  "${JOB_DIR}/run_tagged_length_bench.py"; do
  if [ ! -s "$required" ]; then
    echo "required_artifact_missing=${required}"
    exit 2
  fi
done
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true python=${PYTHON_BIN} vllm=${VLLM_BIN}"
  exit 3
fi

echo "job_name=${JOB_NAME} slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "topology=P0:uranus:L40S:TP1,D0:ganymede:L4:TP1 connector=NixlConnector route=P0-D0"
echo "lengths=128,256,512,1024,2048,4096 repeats=3 warmups=2 output_tokens=1"
echo "work_dir=${PD_WORK_DIR} output_dir=${OUT_DIR} predictive_dvfs=false"

"${SCRIPT_ROOT}/start_pd_vllm.sh" >"${OUT_DIR}/pd_launcher.log" 2>&1 &
PD_LAUNCHER_PID=$!

REGISTRY_FILE="${OUT_DIR}/registry.json"
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
    echo "pd_launcher_exited_before_ready=true"
    exit 5
  fi
  if curl -fsS --connect-timeout 2 --max-time 5 \
    "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" && \
    "$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = (
    d.get("prefill_count") == 1
    and d.get("decode_count") == 1
    and d.get("prefill_tp_sizes") == [1]
    and d.get("decode_tp_sizes") == [1]
    and d.get("supported_routes") == ["P0-D0"]
    and d.get("kv_connector") == "NixlConnector"
)
raise SystemExit(0 if ok else 1)
' "$REGISTRY_FILE"; then
    echo "pd_registry_ready=true"
    break
  fi
  sleep 2
done
if [ "$SECONDS" -ge "$deadline" ]; then
  echo "pd_registry_ready=false timeout_s=${STARTUP_TIMEOUT_SECONDS}"
  exit 6
fi
"$PYTHON_BIN" -m json.tool "$REGISTRY_FILE"

echo "phase=tagged_nixl_length_profile"
"$PYTHON_BIN" "${JOB_DIR}/run_tagged_length_bench.py" \
  --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" \
  --model "$MODEL" \
  --decode-log "${PD_OUT_DIR}/Decode_0.log" \
  --proxy-trace "$PD_REQUEST_TRACE_FILE" \
  --output-dir "$OUT_DIR" \
  --lengths 128,256,512,1024,2048,4096 \
  --repeats 3 \
  --warmups 2 \
  --warmup-length 512 \
  --output-tokens 1 \
  --metrics-wait-seconds 15 \
  --timeout-seconds 180 \
  >"${OUT_DIR}/tagged_length_bench.log" 2>&1
BENCH_RC=$?
cat "${OUT_DIR}/tagged_length_bench.log"

TRACE_COUNT=0
if [ -s "$PD_REQUEST_TRACE_FILE" ]; then
  TRACE_COUNT=$(wc -l < "$PD_REQUEST_TRACE_FILE")
fi
echo "tagged_trace_count=${TRACE_COUNT} expected=20 bench_rc=${BENCH_RC}"
cat "${OUT_DIR}/nixl_length_profile.csv" 2>/dev/null || true

FINAL_RC=0
if [ "$BENCH_RC" -ne 0 ]; then
  FINAL_RC=10
fi
if [ "$TRACE_COUNT" -ne 20 ]; then
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
