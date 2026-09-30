#!/usr/bin/env bash
#SBATCH --job-name=canatune-r6b-measure
# CanaTune r6: model measurements E1 + E2 (CanaTune fe98353), outside the control loop.
# Canary pair (P0, D0) only; the proxy runs round robin but receives no traffic, the
# other pair idles at driver clocks. Stages (canatune.measure defaults):
#   lut     P 2520/1545/1080: lengths 16..3072 x 7, sequential, P counter per request
#   power   idle power at each P and D clock
#   decode  D 2040/1170/735 x concurrency 1/4/8/16/24/32, 60 s closed loop, output 256
#   load    P 2520/1545/1080: Poisson 0.5..7 req/s, 90 s each, default mix; stops a scan
#           once TTFT p95 > 3 s
#   mix     P 2520: short and long mixes, same rates
# Every request logs P round trip / gap / decode first token; every window logs vLLM
# counter deltas, energy per GPU and sampled queue state. Resumable: a crash restarts
# the service and continues after the last finished window.
# r6b: resubmission of r6 (job 270364), which ran next to the baseline rerun because its
# squeue guard failed open, and died because both jobs' GPU agents used port 9300. The
# baseline is done; this job drops the guard and gives its agents a job-specific port.
#   P: 2 x L40S on neptune, D: 2 x L4 on ganymede.
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260930-213856-luqia-canatune-e1e2-model-measure-r6b/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260930-213856-luqia-canatune-e1e2-model-measure-r6b/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20260930-213856-luqia-canatune-e1e2-model-measure-r6b"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="fe983538c16fb6817e69ff0472275054256b0614"
P_NODE=neptune
D_NODE=ganymede
WALL_S=10800
RESERVE_S=480   # teardown, result copy and clock reset before the wall limit
mkdir -p "${OUT}"
# HTTP access lines only bloat the console (and the broker history).
exec > >(grep --line-buffered -v '"GET /' | tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} commit=${COMMIT}"

# ---- caches and HOME inside WORK (nothing lands in /root or ~) ----------------------
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" TMP="${WORK}/tmp" TEMP="${WORK}/tmp"
export XDG_CACHE_HOME="${WORK}/xdg-cache" XDG_CONFIG_HOME="${WORK}/xdg-config"
export XDG_DATA_HOME="${WORK}/xdg-data" XDG_STATE_HOME="${WORK}/xdg-state"
export XDG_RUNTIME_DIR="${WORK}/xdg-runtime" PIP_CACHE_DIR="${WORK}/pip-cache"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache" PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0
export PATH="${ENV_DIR}/bin:${PATH}"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}" "${XDG_RUNTIME_DIR}" "${PIP_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
chmod 700 "${XDG_RUNTIME_DIR}" "${TMPDIR}"

P_IDS=""; P_UUIDS=""; D_IDS=""; D_UUIDS=""
reset_clocks() {  # UUID-checked reset of the GPUs this job was allocated
  local node="$1" ids="$2" uuids="$3"
  [ -n "${ids}" ] || return 0
  timeout 120 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c '
    IFS=, read -r -a ids <<< "$1"; IFS=, read -r -a uuids <<< "$2"
    for i in "${!ids[@]}"; do
      u=$(nvidia-smi -i "${ids[$i]}" --query-gpu=uuid --format=csv,noheader | tr -d " ")
      if [ "$u" = "${uuids[$i]}" ]; then
        sudo -n nvidia-smi -i "${ids[$i]}" -rgc >/dev/null 2>&1 && echo "reset ${ids[$i]} ok" || echo "reset ${ids[$i]} FAILED"
      else echo "skip ${ids[$i]}: uuid $u != ${uuids[$i]}"; fi
      sleep 3  # let the clock fall after the reset before reading it
      nvidia-smi -i "${ids[$i]}" --query-gpu=index,uuid,clocks.sm,clocks_event_reasons.active --format=csv,noheader
    done' _ "${ids}" "${uuids}" 2>&1 | sed "s/^/[${node}] /"
}
finish() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset.txt"
  reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset.txt"
  for node in "${P_NODE}" "${D_NODE}"; do
    timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
      'ps -o pid,etime,args -u "$(id -un)" | grep -E "vllm.entrypoints|canatune" | grep -v grep || echo none' \
      2>&1 | sed "s/^/[${node} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  done
  for run in measure; do
    [ -d "${WORK}/${run}/records" ] && cp "${WORK}/${run}/records/"*.jsonl "${OUT}/${run}/" 2>/dev/null
    # Telemetry polls /metrics every 0.25 s: drop those access lines, keep requests and errors.
    for f in "${WORK}/${run}/pd"/*/logs/*.log; do
      [ -f "$f" ] && grep -vE 'GET /(metrics|health)' "$f" | tail -n 400 > "${OUT}/${run}/$(basename "$f" .log)_tail.log"
    done
  done
  [ -f "${WORK}/job_config.yaml" ] && cp "${WORK}/job_config.yaml" "${OUT}/"
  [ -f "${WORK}/baseline.yaml" ] && cp "${WORK}/baseline.yaml" "${OUT}/"
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }


# ---- 1. code at the pinned commit + CPU tests -----------------------------------------
timeout 300 git clone --quiet "${REPO}" "${WORK}/CanaTune" || fail clone
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" || fail checkout
git log -1 --format='%H %s' | tee "${OUT}/commit.txt"
timeout 300 "${PY}" -m pip install --quiet --target "${WORK}/pytest-lib" "pytest>=8.3,<9" || fail pip_pytest
PYTHONPATH="${PWD}/src:${WORK}/pytest-lib" timeout 600 "${PY}" -m pytest -p no:cacheprovider \
  > "${OUT}/pytest.txt" 2>&1 || { tail -n 30 "${OUT}/pytest.txt"; fail pytest; }
tail -n 1 "${OUT}/pytest.txt"
export PROJECT_ROOT="${WORK}/CanaTune" PYTHONPATH="${WORK}/CanaTune/src"

# ---- 2. model snapshot (offline) ----------------------------------------------------------
SNAP_ROOT="/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots"
MODEL_PATH=""
while IFS= read -r snap; do
  if [ -s "${snap}/config.json" ] && [ -s "${snap}/tokenizer_config.json" ] && \
     find "${snap}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then MODEL_PATH="${snap}"; break; fi
done < <(find "${SNAP_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ -n "${MODEL_PATH}" ] || fail model_snapshot
export MODEL_PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "model=${MODEL_PATH}"

# ---- 3. allocated GPUs (index + UUID) and node IPs -------------------------------------
detect() {  # prints "<ids>;<uuids>;<names>"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --gpus-per-node=2 --mem=256M bash -c '
    list="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"; ids=""; uuids=""; names=""
    IFS=, read -r -a arr <<< "$list"
    for i in "${arr[@]}"; do
      line=$(nvidia-smi -i "$i" --query-gpu=uuid,name --format=csv,noheader)
      ids="${ids:+$ids,}$i"; uuids="${uuids:+$uuids,}$(echo "${line%%,*}" | tr -d " ")"; names="${names:+$names|}${line#*, }"
    done
    echo "gpus=${ids};${uuids};${names}"' 2>&1 | grep '^gpus=' | tail -n 1 | sed 's/^gpus=//'
}
P_LINE=$(detect "${P_NODE}"); D_LINE=$(detect "${D_NODE}")
echo "detected ${P_NODE}=${P_LINE}" | tee "${OUT}/gpu_allocation.txt"
echo "detected ${D_NODE}=${D_LINE}" | tee -a "${OUT}/gpu_allocation.txt"
IFS=';' read -r P_IDS P_UUIDS P_NAMES <<< "${P_LINE}"
IFS=';' read -r D_IDS D_UUIDS D_NAMES <<< "${D_LINE}"
[[ "${P_IDS}" =~ ^[0-9]+,[0-9]+$ && "${P_NAMES}" == *L40S*"|"*L40S* ]] || fail gpu_detect_prefill 3
[[ "${D_IDS}" =~ ^[0-9]+,[0-9]+$ && "${D_NAMES}" == *L4* && "${D_NAMES}" != *L40S* ]] || fail gpu_detect_decode 3
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP=$(ip_of "${D_NODE}")
[[ "${P_IP}" =~ ^[0-9.]+$ && "${D_IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
echo "ips ${P_NODE}=${P_IP} ${D_NODE}=${D_IP}" | tee -a "${OUT}/gpu_allocation.txt"

# ---- 4. job configs -----------------------------------------------------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
AGENT_PORT=$(( 9300 + ${JOB_ID//[^0-9]/} % 500 ))   # r6: a fixed 9300 collided with another job
common=(--base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=true
  --set "clock_control.agent_port=${AGENT_PORT}"
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}")
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/measure.yaml" \
  --set routing.policy=round_robin || fail job_config_measure
cp "${WORK}/measure.yaml" "${WORK}/job_config.yaml"

# ---- 5. cantune run, then the baseline on the same trace -----------------------------------
export PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
export DECODE_GPU_MEMORY_UTILIZATION=0.82 PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=1
export PYTHONFAULTHANDLER=1   # native stack on a segfault (r5 run_a lost D0 to one)
mkdir -p "${OUT}/measure" "${WORK}/measure"
export CANATUNE_CONFIG="${WORK}/measure.yaml" PD_WORK_DIR="${WORK}/measure/pd"
export CANATUNE_RECORD_DIR="${WORK}/measure/records" MEASURE_OUT="${OUT}/measure/data"  # records/ copies land in measure/, never over data
echo "--- clocks before the measurements (reset)"
reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
ATTEMPT=0
while :; do
  ATTEMPT=$(( ATTEMPT + 1 ))
  LEFT=$(( T0 + WALL_S - RESERVE_S - $(date +%s) ))
  [ "${LEFT}" -gt 1200 ] || { echo "RESULT=partial (no time for attempt ${ATTEMPT})"; exit 1; }
  export MEASURE_DEADLINE_S=$(( LEFT - 300 ))   # service start + shutdown
  echo "=== measure attempt ${ATTEMPT} start=$(date -Iseconds) deadline_s=${MEASURE_DEADLINE_S}"
  "${PY}" scripts/measure_model.py; RC=$?
  case "${RC}" in
    0) echo "RESULT=ok (all windows done)"; exit 0;;
    7) echo "RESULT=partial (stopped for time)"; exit 1;;
    *) echo "attempt ${ATTEMPT} failed rc=${RC}; resuming after the last finished window"
       [ "${ATTEMPT}" -lt 3 ] || { echo "RESULT=failed (3 attempts)"; exit 1; }
       sleep 30;;
  esac
done
