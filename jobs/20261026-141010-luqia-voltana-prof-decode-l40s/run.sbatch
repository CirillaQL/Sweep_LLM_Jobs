#!/usr/bin/env bash
#SBATCH --job-name=voltana-decode-l40s
#SBATCH --partition=long
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=neptune
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261026-141010-luqia-voltana-prof-decode-l40s/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261026-141010-luqia-voltana-prof-decode-l40s/slurm-%j.err

# VoltanaLLM EcoPred decode latency profile on one L40S (neptune), Mistral-7B-v0.1, TP 1.
# For each batch size in 1, 8k, 8k+1, 128k, 128k+1 (<= max running) it sends that many 1-token prompts at once with outputs sized to fill the KV cache; per decode step it records batch size, KV tokens and ITL and fits ITL = a*bs + b*tokens + c per clock and batch-size bucket (RANSAC, ITL in (3, 300) ms).
# Clocks (MHz, snapped to the GPU's supported clocks before use): 2520,1800,1440,1080,720
# Each clock is locked by bench_serving_voltanallm.py (sudo nvidia-smi -lgc) around every run;
# the actual SM clock / power / throttle reasons are sampled to clocks.csv.gz so power-capped
# points can be dropped. At the end fit_latency.py fits latency-decode.csv (the lookup table
# EcoFreq reads via --freq-manager-latency-lookup-table-path).
# Needs 20261029-105703-luqia-voltana-env-setup-v6 (env, patched checkout, model link). Raw per-iteration data stays in
# /data/users/chjing/voltana_work/runs/<jobid>/ (see RAW_CACHE_LOCATION.txt); only the table,
# plot, clock trace and log tails come back through git. Expected run time about 6-10 h (KV ~200k tokens, outputs up to 8176 tokens, 45 batch sizes per clock).
set -uo pipefail
START_EPOCH=$(date +%s)
JOB_NAME="20261026-141010-luqia-voltana-prof-decode-l40s"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
VOLTANA_ROOT="/data/users/chjing/voltana_work"
WORK="${VOLTANA_ROOT}/runs/${JOB_ID}"
RAW="${WORK}/decode"
PRED_DIR="${VOLTANA_ROOT}/VoltanaLLM/voltana-llm-scripts/predictions"
ENV_DIR="/data/users/chjing/miniforge3/envs/voltanallm"
PY="${ENV_DIR}/bin/python"
PHASE="decode"
GPU_TYPE="L40S"
REQUESTED_FREQS="2520,1800,1440,1080,720"
PORT=$(( 30000 + ${JOB_ID//[^0-9]/0} % 10000 ))

mkdir -p "${OUT}" "${RAW}"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "job_start=$(date -Iseconds) host=$(hostname) job=${JOB_ID}"
fail() { echo "RESULT=failed stage=$1"; exit 1; }

grep -q "env-setup-v6" "${VOLTANA_ROOT}/ENV_READY" 2>/dev/null || fail "env_not_ready (run 20261029-105703-luqia-voltana-env-setup-v6 first; v4/v5 lack the decode extra_batch_info and fit fixes)"
echo "env_ready=$(cat "${VOLTANA_ROOT}/ENV_READY")"
export PATH="${ENV_DIR}/bin:${PATH}" CUDA_HOME="${ENV_DIR}"

# ---- caches and HOME inside WORK; JIT caches shared with the setup job ---------------------
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" TMP="${WORK}/tmp" TEMP="${WORK}/tmp"
export XDG_CACHE_HOME="${WORK}/xdg-cache" XDG_CONFIG_HOME="${WORK}/xdg-config"
export HF_HOME="${WORK}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${WORK}/torch-cache" TORCH_EXTENSIONS_DIR="${WORK}/torch-extensions"
export CUDA_CACHE_PATH="${WORK}/cuda-cache" MPLCONFIGDIR="${WORK}/matplotlib"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache"
export FLASHINFER_WORKSPACE_BASE="${VOLTANA_ROOT}/cache/flashinfer"
export TRITON_CACHE_DIR="${VOLTANA_ROOT}/cache/triton"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

# ---- GPU: physical index (devices are not cgroup constrained here) ---------------------------
GPU_INDEX="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [ "$(nvidia-smi -L | wc -l)" -eq 1 ]; then GPU_INDEX=0; fi   # constrained view: only ours is visible
[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || fail "unexpected_gpu_allocation=${GPU_INDEX}"
GPU_NAME=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)
{ echo "gpu_index=${GPU_INDEX}"; echo "gpu_name=${GPU_NAME}"; echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"; echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi -i "${GPU_INDEX}" --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.mem,power.draw,power.limit --format=csv; } | tee "${OUT}/gpu_allocation.txt"
case "${GPU_NAME}" in *"${GPU_TYPE}"*) ;; *) fail "wrong_gpu_type=${GPU_NAME}";; esac
case "${GPU_TYPE}:${GPU_NAME}" in L4:*L40S*) fail "wrong_gpu_type=${GPU_NAME}";; esac
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}" NVML_GPU_IDX="${GPU_INDEX}"

# ---- snap the requested clocks to supported ones (labels must match what -lgc applies) -------
SUPPORTED=$(nvidia-smi -i "${GPU_INDEX}" --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -un | tr '\n' ' ')
FREQ_LIST=$("${PY}" -c '
import sys
supported = sorted({int(c) for c in sys.argv[2].split()})
out = []
for f in (int(x) for x in sys.argv[1].split(",")):
    s = min(supported, key=lambda c: abs(c - f))
    if s not in out:
        out.append(s)
print(",".join(map(str, out)))
' "${REQUESTED_FREQS}" "${SUPPORTED}") || fail freq_snap
{ echo "requested=${REQUESTED_FREQS}"; echo "snapped=${FREQ_LIST}"; echo "supported=${SUPPORTED}"; } | tee "${OUT}/freqs.txt"

SERVER_PID=""
MONITOR_PID=""
PROFILE_PID=""
cleanup() {
  set +e
  [ -z "${PROFILE_PID}" ] || kill "${PROFILE_PID}" 2>/dev/null
  [ -z "${MONITOR_PID}" ] || kill "${MONITOR_PID}" 2>/dev/null
  [ -z "${SERVER_PID}" ] || { kill "${SERVER_PID}" 2>/dev/null; sleep 5; kill -9 "${SERVER_PID}" 2>/dev/null; }
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null
  sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc > "${OUT}/clock_reset.txt" 2>&1
  [ ! -s "${RAW}/clocks.csv" ] || gzip -c "${RAW}/clocks.csv" > "${OUT}/clocks.csv.gz"
  [ ! -s "${RAW}/server.log" ] || tail -n 300 "${RAW}/server.log" > "${OUT}/server_tail.log"
  [ ! -s "${RAW}/profile.log" ] || tail -n 200 "${RAW}/profile.log" | tr '\r' '\n' | tail -n 200 > "${OUT}/profile_tail.log"
  [ ! -s "${RAW}/fit.log" ] || cp "${RAW}/fit.log" "${OUT}/"
  find "${RAW}" -maxdepth 2 -name '*-extra.json' -printf '%P %s\n' 2>/dev/null | sort > "${OUT}/raw_files.txt"
  printf '%s\n' "${RAW}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - START_EPOCH ))"
}
trap cleanup EXIT
trap 'exit 143' TERM

# Fail fast if clock locking is not permitted; judge by nvidia-smi's "All done" (job 267572
# exited 9 right after a successful lock, so the exit status alone is not trusted).
sudo -n nvidia-smi -i "${GPU_INDEX}" -lgc 1500,1500 > "${OUT}/clock_permission.txt" 2>&1
echo "lock_rc=$?" >> "${OUT}/clock_permission.txt"
grep -q "All done" "${OUT}/clock_permission.txt" || fail clock_lock_not_permitted
sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc >> "${OUT}/clock_permission.txt" 2>&1

# ---- server (p_server.sh / d_server.sh exec sglang, so SERVER_PID is the server itself) ------
export MODEL_PATH="${VOLTANA_ROOT}/models/Mistral-7B-v0.1"
export DATASET_PATH="/data/users/chjing/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
export HOST=127.0.0.1 PORT GPU_TAG="L40S-48G" FREQ_LIST
export CONTEXT_LEN=16384 MAX_RUNNING=1024 CUDA_GRAPH_MAX_BS=1024 MEM_FRACTION=0.88
[ -e "${MODEL_PATH}/config.json" ] || fail "model_link_missing ${MODEL_PATH}"
cd "${RAW}" || fail cd_raw
bash "${PRED_DIR}/decode/d_server.sh" 0 > "${RAW}/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || fail server_exited_during_startup
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || fail server_not_ready_in_1800s
echo "server_ready_after_s=$(( $(date +%s) - START_EPOCH ))" | tee -a "${OUT}/gpu_allocation.txt"
curl -sf "http://127.0.0.1:${PORT}/get_server_info" > "${OUT}/server_info.json"
# Evidence: our own processes must not match the shared launcher's pkill patterns.
{ echo "pkill_pattern_matches_for_this_job:"; pgrep -a -f 'python.*vllm|vllm\.entrypoints' | grep -v pgrep || echo none; } >> "${OUT}/gpu_allocation.txt"

nvidia-smi -i "${GPU_INDEX}" \
  --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 1000 > "${RAW}/clocks.csv" &
MONITOR_PID=$!

# ---- profile (background + wait, so the TERM before the time limit is handled at once) -----
"${PY}" "${PRED_DIR}/decode/run_profile.py" > "${RAW}/profile.log" 2>&1 &
PROFILE_PID=$!
wait "${PROFILE_PID}"
PROFILE_RC=$?
PROFILE_PID=""
echo "profile_rc=${PROFILE_RC} profile_s=$(( $(date +%s) - START_EPOCH ))"
kill "${MONITOR_PID}" 2>/dev/null; MONITOR_PID=""
kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null; SERVER_PID=""

# ---- fit (also on a partial run: every clock that finished is usable) -----------------------
"${PY}" "${PRED_DIR}/fit_latency.py" --phase decode --data-dir "${RAW}" --out-dir "${RAW}/fit" \
  > "${RAW}/fit.log" 2>&1
FIT_RC=$?
for f in latency-decode.csv latency-decode.png; do
  [ ! -s "${RAW}/fit/${f}" ] || cp "${RAW}/fit/${f}" "${OUT}/"
done
echo "fit_rc=${FIT_RC}"; [ ! -s "${OUT}/latency-decode.csv" ] || cat "${OUT}/latency-decode.csv"
if [ "${PROFILE_RC}" -eq 0 ] && [ "${FIT_RC}" -eq 0 ]; then echo "RESULT=ok"; exit 0; fi
echo "RESULT=failed profile_rc=${PROFILE_RC} fit_rc=${FIT_RC}"; exit 1
