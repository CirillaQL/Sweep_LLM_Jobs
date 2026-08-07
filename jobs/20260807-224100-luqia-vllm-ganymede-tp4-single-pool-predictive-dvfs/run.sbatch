#!/usr/bin/env bash
#SBATCH --job-name=vllm-ganymede-tp4-single-pool-predictive-dvfs
#SBATCH --partition=long
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260807-224100-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260807-224100-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs/slurm-%j.err

set -uo pipefail

JOB_NAME="20260807-224100-luqia-vllm-ganymede-tp4-single-pool-predictive-dvfs"
COMMON_NAME="20260807-172233-luqia-vllm-uranus-tp4-single-pool-predictive-dvfs"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
COMMON_DIR="${BROKER_ROOT}/jobs/${COMMON_NAME}"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
OUT_DIR="${JOB_DIR}/tp4_single_pool_results"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PYTHON_BIN="${ENV_DIR}/bin/python"
VLLM_BIN="${ENV_DIR}/bin/vllm"
MODEL="mistralai/Mistral-7B-v0.1"
TP_SIZE=4
SERVER_PID=""
MONITOR_PID=""
FINAL_RC=0
PORT=$((22000 + SLURM_JOB_ID % 20000))
WORK_ROOT="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}"
ALLOCATED_GPU_CSV="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a GPU_IDS <<< "$ALLOCATED_GPU_CSV"
if [ "${#GPU_IDS[@]}" -ne "$TP_SIZE" ]; then
  echo "allocated_gpu_count_mismatch expected=${TP_SIZE} actual=${#GPU_IDS[@]} devices=${ALLOCATED_GPU_CSV}" >&2
  exit 4
fi

reset_clocks() {
  local gpu_id
  for gpu_id in "${GPU_IDS[@]}"; do
    sudo -n nvidia-smi -i "$gpu_id" -rgc
  done
}

stop_processes() {
  if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _attempt in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

cleanup() {
  local cleanup_rc=$?
  trap - EXIT INT TERM
  stop_processes
  {
    echo "reset_start=$(date --iso-8601=seconds)"
    reset_clocks
    echo "reset_query"
    nvidia-smi -i "$ALLOCATED_GPU_CSV" --query-gpu=index,clocks.sm,clocks.max.sm --format=csv,noheader,nounits
    echo "reset_complete=true"
    if [[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] \
      && [ "$WORK_ROOT" = "/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}" ]; then
      if rm -rf -- "$WORK_ROOT"; then
        echo "cache_cleanup_complete=true path=${WORK_ROOT}"
      else
        echo "cache_cleanup_complete=false removal_failed path=${WORK_ROOT}"
        cleanup_rc=3
      fi
    else
      echo "cache_cleanup_complete=false unsafe_path=${WORK_ROOT}"
      cleanup_rc=3
    fi
  } >> "${OUT_DIR}/reset_step.log" 2>&1
  exit "$cleanup_rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT_DIR" "$WORK_ROOT"/{xdg-cache,xdg-config,flashinfer,cuda-cache,triton-cache,torchinductor-cache,torch,vllm-cache,numba-cache,tmp}
export HF_HOME="/data/users/chjing/.cache/huggingface"
export XDG_CACHE_HOME="$WORK_ROOT/xdg-cache"
export XDG_CONFIG_HOME="$WORK_ROOT/xdg-config"
export FLASHINFER_WORKSPACE_BASE="$WORK_ROOT/flashinfer"
export CUDA_CACHE_PATH="$WORK_ROOT/cuda-cache"
export TRITON_CACHE_DIR="$WORK_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/torchinductor-cache"
export TORCH_HOME="$WORK_ROOT/torch"
export VLLM_CACHE_ROOT="$WORK_ROOT/vllm-cache"
export NUMBA_CACHE_DIR="$WORK_ROOT/numba-cache"
export TMPDIR="$WORK_ROOT/tmp"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU_CSV"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

{
  echo "host=$(hostname -s) job_id=${SLURM_JOB_ID} topology=single_pool gpu_type=l4 tp=${TP_SIZE}"
  echo "model=${MODEL} port=${PORT} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "${OUT_DIR}/preflight.log" 2>&1

reset_clocks >> "${OUT_DIR}/reset_step.log" 2>&1

"$PYTHON_BIN" "${COMMON_DIR}/predict_single_pool.py" \
  --scheduler "${SOURCE_JOB}/scheduler.py" \
  --bundle "${SOURCE_JOB}/model_bundle.json" \
  --workloads "${JOB_DIR}/request_trace.csv" \
  --out-dir "$OUT_DIR" --gpu-type l4 --node ganymede \
  --tp "$TP_SIZE" --slo-ttft-ms 500 --slo-tpot-ms 200

"$PYTHON_BIN" -u "${COMMON_DIR}/monitor_single_pool.py" \
  --output "${OUT_DIR}/telemetry.csv" --gpu-ids "$ALLOCATED_GPU_CSV" --interval 0.5 \
  > "${OUT_DIR}/telemetry.log" 2>&1 &
MONITOR_PID=$!

setsid "$VLLM_BIN" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --dtype float16 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  > "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

SERVER_READY=false
for _attempt in $(seq 1 200); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server_exited_before_ready=true" >&2
    tail -n 200 "${OUT_DIR}/server.log" >&2 || true
    exit 5
  fi
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" > "${OUT_DIR}/models.json" 2>/dev/null; then
    SERVER_READY=true
    break
  fi
  sleep 3
done
if [ "$SERVER_READY" != true ]; then
  echo "server_ready_timeout=true" >&2
  exit 6
fi

curl -fsS --max-time 180 -X POST "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of Sweden is\",\"max_tokens\":8,\"temperature\":0}" \
  > "${OUT_DIR}/smoke_response.json"

echo "unix_ts,event,seq,workload_id,policy" > "${OUT_DIR}/events.csv"
SEQ=0
while IFS=, read -r workload_id il ol rate window_seconds burstiness description; do
  if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
    continue
  fi
  SEQ=$((SEQ + 1))
  DECISION_FILE="${OUT_DIR}/decision_${SEQ}_${workload_id}.json"
  frequency="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_frequency_mhz"])' "$DECISION_FILE")"
  predicted_ttft="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["predicted_p99_ttft_ms"])' "$DECISION_FILE")"
  predicted_tpot="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["predicted_p99_tpot_ms"])' "$DECISION_FILE")"
  echo "window seq=${SEQ} id=${workload_id} il=${il} ol=${ol} rate=${rate} target_mhz=${frequency} predicted_ttft_ms=${predicted_ttft} predicted_tpot_ms=${predicted_tpot}"
  for gpu_id in "${GPU_IDS[@]}"; do
    sudo -n nvidia-smi -i "$gpu_id" -lgc "$frequency"
  done
  nvidia-smi -i "$ALLOCATED_GPU_CSV" --query-gpu=index,clocks.sm --format=csv,noheader,nounits \
    > "${OUT_DIR}/clock_apply_${SEQ}_${workload_id}.log"
  sleep 5
  num_prompts="$($PYTHON_BIN -c 'import sys; print(max(12, min(96, int(float(sys.argv[1]) * float(sys.argv[2])))))' "$rate" "$window_seconds")"
  start_ts="$(date +%s.%N)"
  echo "${start_ts},workload_start,${SEQ},${workload_id},single_pool_l4_tp4_predictive_dvfs" >> "${OUT_DIR}/events.csv"
  timeout --signal=INT --kill-after=20s 300s "$VLLM_BIN" bench serve \
    --backend openai --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" \
    --num-prompts "$num_prompts" --request-rate "$rate" --dataset-name random \
    --burstiness "$burstiness" --save-result --save-detailed --result-dir "$OUT_DIR" \
    --result-filename "detailed_${SEQ}_${workload_id}.json" \
    --random-input-len "$il" --random-output-len "$ol" --max-concurrency 32 \
    --num-warmups 0 --temperature 0 </dev/null \
    > "${OUT_DIR}/bench_${SEQ}_${workload_id}.txt" 2>&1
  bench_rc=$?
  end_ts="$(date +%s.%N)"
  echo "${end_ts},workload_end,${SEQ},${workload_id},single_pool_l4_tp4_predictive_dvfs" >> "${OUT_DIR}/events.csv"
  if [ "$bench_rc" -ne 0 ]; then
    echo "benchmark_failed seq=${SEQ} id=${workload_id} rc=${bench_rc}" >&2
    FINAL_RC="$bench_rc"
    break
  fi
done < "${JOB_DIR}/request_trace.csv"

if [ "$FINAL_RC" -eq 0 ]; then
  "$PYTHON_BIN" "${COMMON_DIR}/check_single_pool.py" \
    --out-dir "$OUT_DIR" --expected-gpus 4 --clock-tolerance-mhz 90 \
    --slo-ttft-ms 500 --slo-tpot-ms 200 --gpu-type l4 --node ganymede --tp 4 \
    || FINAL_RC=$?
fi

echo "final_rc=${FINAL_RC}"
exit "$FINAL_RC"
