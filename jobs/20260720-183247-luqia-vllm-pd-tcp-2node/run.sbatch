#!/usr/bin/env bash
#SBATCH --job-name=vllm-pd-tcp
#SBATCH --partition=long
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,neptune
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183247-luqia-vllm-pd-tcp-2node/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183247-luqia-vllm-pd-tcp-2node/slurm-%j.err

set -uo pipefail

SOURCE_JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-183247-luqia-vllm-pd-tcp-2node"
JOB_DIR="${JOB_DIR_OVERRIDE:-${SOURCE_JOB_DIR}}"
OUT_DIR="${JOB_DIR}/pd_tcp_results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
NODE_SCRIPT="${SOURCE_JOB_DIR}/start_pd_node.sh"
PROXY_SCRIPT="${SOURCE_JOB_DIR}/pd_proxy.py"
MODEL="mistralai/Mistral-7B-v0.1"

PREFILL_IP=10.1.0.6
DECODE_IP=10.1.0.3
PROXY_IP="$DECODE_IP"
PORT_OFFSET=$((${SLURM_JOB_ID:-0} % 500))
PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
PREFILL_HTTP_PORT=$((32000 + PORT_OFFSET))
DECODE_HTTP_PORT=$((33000 + PORT_OFFSET))
KV_PORT=$((34000 + PORT_OFFSET))

export HOME="/data/users/chjing"
export USER=chjing
export LOGNAME=chjing
export HF_HOME="${HOME}/.cache/huggingface"
export XDG_CACHE_HOME="${HOME}/.cache"
export PATH="${ENV_DIR}/bin:${PATH}"
export JOB_DIR OUT_DIR ENV_DIR PYTHON_BIN VLLM_BIN MODEL
export PREFILL_IP DECODE_IP PROXY_IP PROXY_HTTP_PORT PROXY_REGISTER_PORT
export PREFILL_HTTP_PORT DECODE_HTTP_PORT KV_PORT

# Minerva may expose an SMT-adjusted SLURM_CPUS_PER_TASK that disagrees with
# SLURM_TRES_PER_TASK=cpu:<requested>.  Nested srun rejects that combination.
unset SLURM_CPUS_PER_TASK

mkdir -p "$OUT_DIR"
ulimit -n 65536

PROXY_PID=""
SERVER_STEP_PID=""
FINAL_RC=0

section() {
  echo
  echo "===== $1 ====="
}

stop_components() {
  if [ -n "$SERVER_STEP_PID" ] && kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    echo "stopping_server_step_pid=${SERVER_STEP_PID}"
    kill "$SERVER_STEP_PID" 2>/dev/null || true
    wait "$SERVER_STEP_PID" 2>/dev/null || true
  fi
  SERVER_STEP_PID=""
  if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "stopping_proxy_pid=${PROXY_PID}"
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  PROXY_PID=""
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_components
  echo "job_cleanup_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_url() {
  local name="$1"
  local url="$2"
  local max_seconds="$3"
  local elapsed=0
  while ! curl -fsS "$url" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo "waiting name=${name} elapsed_s=${elapsed} url=${url}"
    if [ -n "$SERVER_STEP_PID" ] && ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
      echo "server_step_died_while_waiting=true"
      return 1
    fi
    if [ "$elapsed" -ge "$max_seconds" ]; then
      echo "wait_timeout name=${name} url=${url}"
      return 1
    fi
  done
  echo "ready name=${name} elapsed_s=${elapsed} url=${url}"
}

section "allocation"
echo "time=$(date --iso-8601=seconds)"
echo "batch_host=$(hostname -s)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "model=${MODEL}"
echo "prefill=neptune:${PREFILL_HTTP_PORT} ip=${PREFILL_IP} iface=enp160s0f0np0"
echo "decode=ganymede:${DECODE_HTTP_PORT} ip=${DECODE_IP} iface=eno12399np0"
echo "proxy=${PROXY_IP}:${PROXY_HTTP_PORT} registry_port=${PROXY_REGISTER_PORT}"
echo "kv_port=${KV_PORT}"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>&1 || true
scontrol show job "${SLURM_JOB_ID:-}" 2>&1 || true

if [ "$(hostname -s)" != ganymede ]; then
  echo "expected_batch_host_ganymede=true actual=$(hostname -s)"
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "vllm_environment_missing=true"
  exit 3
fi

section "100gbe_and_environment_preflight"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 "$NODE_SCRIPT" preflight 2>&1 | tee "${OUT_DIR}/preflight.log"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "preflight_rc=${PREFLIGHT_RC}"
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  exit "$PREFLIGHT_RC"
fi

section "start_proxy"
export PROXY_REGISTER_HOST=0.0.0.0
export PROXY_HTTP_HOST=0.0.0.0
"$PYTHON_BIN" -u "$PROXY_SCRIPT" > "${OUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
echo "proxy_pid=${PROXY_PID}"
if ! wait_for_url proxy "http://127.0.0.1:${PROXY_HTTP_PORT}/health" 60; then
  tail -n 200 "${OUT_DIR}/proxy.log" || true
  exit 4
fi

section "start_prefill_and_decode"
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  "$NODE_SCRIPT" serve > "${OUT_DIR}/server_step.log" 2>&1 &
SERVER_STEP_PID=$!
echo "server_step_pid=${SERVER_STEP_PID}"

if ! wait_for_url prefill "http://${PREFILL_IP}:${PREFILL_HTTP_PORT}/v1/models" 600; then
  FINAL_RC=5
fi
if [ "$FINAL_RC" -eq 0 ] && ! wait_for_url decode "http://${DECODE_IP}:${DECODE_HTTP_PORT}/v1/models" 600; then
  FINAL_RC=6
fi
if [ "$FINAL_RC" -ne 0 ]; then
  tail -n 240 "${OUT_DIR}/prefill_server.log" 2>/dev/null || true
  tail -n 240 "${OUT_DIR}/decode_server.log" 2>/dev/null || true
  exit "$FINAL_RC"
fi

section "wait_for_pd_registry"
REGISTRY_FILE="${OUT_DIR}/registry.json"
REGISTRY_READY=false
for attempt in $(seq 1 60); do
  curl -fsS "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" 2>/dev/null || true
  if "$PYTHON_BIN" - "$REGISTRY_FILE" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if len(data.get("prefill", [])) == 1 and len(data.get("decode", [])) == 1 else 1)
PY
  then
    REGISTRY_READY=true
    break
  fi
  sleep 2
done
cat "$REGISTRY_FILE" 2>/dev/null || true
echo
echo "registry_ready=${REGISTRY_READY}"
if [ "$REGISTRY_READY" != true ]; then
  tail -n 240 "${OUT_DIR}/proxy.log" || true
  exit 7
fi

section "pd_smoke_request"
SMOKE_FILE="${OUT_DIR}/smoke_response.json"
curl -fsS --max-time 300 \
  -X POST "http://${PROXY_IP}:${PROXY_HTTP_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  --data-binary "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of Sweden is\",\"max_tokens\":32,\"temperature\":0,\"stream\":false}" \
  > "$SMOKE_FILE"
SMOKE_CURL_RC=$?
cat "$SMOKE_FILE" 2>/dev/null || true
echo
echo "smoke_curl_rc=${SMOKE_CURL_RC}"
if [ "$SMOKE_CURL_RC" -ne 0 ]; then
  exit 8
fi

"$PYTHON_BIN" - "$SMOKE_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
choices = data.get("choices") or []
if not choices or not isinstance(choices[0].get("text"), str):
    raise SystemExit(f"invalid completion response: {data}")
usage = data.get("usage") or {}
print(f"pd_smoke_valid=true completion_tokens={usage.get('completion_tokens')} text={choices[0]['text']!r}")
PY

section "pd_vllm_bench"
timeout --signal=INT --kill-after=30s 300s "$VLLM_BIN" bench serve \
  --backend openai \
  --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}" \
  --model "$MODEL" \
  --num-prompts 16 \
  --request-rate 1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 64 \
  --max-concurrency 4 \
  --num-warmups 2 \
  --temperature 0 \
  > "${OUT_DIR}/bench.txt" 2>&1
BENCH_RC=$?
echo "bench_rc=${BENCH_RC}"
tail -n 100 "${OUT_DIR}/bench.txt" || true
if [ "$BENCH_RC" -ne 0 ]; then
  FINAL_RC=9
fi

section "stop_servers_and_collect_logs"
stop_components

echo "--- proxy routes ---"
grep -E 'registry_add|route count=|proxy_error' "${OUT_DIR}/proxy.log" 2>&1 || true
echo "--- P2P connector evidence ---"
grep -hE 'P2pNcclConnector|KV transfer|KVTransfer|kv_transfer|PUT_ASYNC' \
  "${OUT_DIR}/prefill_server.log" "${OUT_DIR}/decode_server.log" 2>&1 | tail -n 160 || true
echo "--- NCCL network evidence ---"
grep -hE 'NET/Socket|Using network Socket|Connected all rings|GDR 0|GPU Direct RDMA Disabled' \
  "${OUT_DIR}"/nccl-*.log 2>&1 | tail -n 200 || true

SOCKET_HITS=$(grep -hE 'NET/Socket.*(eno12399np0|enp160s0f0np0)|Using network Socket' \
  "${OUT_DIR}"/nccl-*.log 2>/dev/null | wc -l)
echo "nccl_socket_evidence_lines=${SOCKET_HITS}"
if [ "$SOCKET_HITS" -eq 0 ]; then
  echo "nccl_socket_path_verified=false"
  FINAL_RC=10
else
  echo "nccl_socket_path_verified=true"
fi

section "network_and_gpu_telemetry_summary"
"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv
import glob
import os
import sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_telemetry.csv"))):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"telemetry_file={path} samples={len(rows)}")
    if len(rows) < 2:
        continue
    first, last = rows[0], rows[-1]
    rx_delta = int(last["rx_bytes"]) - int(first["rx_bytes"])
    tx_delta = int(last["tx_bytes"]) - int(first["tx_bytes"])
    utils = []
    for row in rows:
        try:
            utils.append(float(row["gpu_util_pct"].strip()))
        except (TypeError, ValueError):
            pass
    print(f"network_rx_delta_bytes={rx_delta}")
    print(f"network_tx_delta_bytes={tx_delta}")
    print(f"gpu_util_max_pct={max(utils) if utils else 'NA'}")
PY

section "result"
echo "pd_smoke_request_ok=true"
echo "pd_bench_rc=${BENCH_RC}"
echo "pd_over_100gbe_tcp_socket_ok=$([ "$FINAL_RC" -eq 0 ] && echo true || echo false)"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
