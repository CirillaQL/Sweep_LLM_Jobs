#!/usr/bin/env bash
#SBATCH --job-name=canatune-smoke2b
# CanaTune smoke test (2), baseline rerun. Job 270039's baseline inherited the clocks
# CanaTune left locked (G0 Park 600/300, G1 H 1545/1170): the agents' reset on exit
# did not happen, so pair 0 served half the traffic at Park clocks and collapsed.
# This job replays the same trace (270039 cantune/trace.json, 3933 requests, 970 s)
# with round robin over both pairs and driver-managed clocks. CanaTune 479b73c resets
# every GPU through the agents at the start of the baseline and before the agents
# stop; the job also resets and prints the clocks before starting.
#   P: 2 x L40S on neptune, D: 2 x L4 on ganymede; pairs (P0, D0), (P1, D1).
#SBATCH --partition=long
#SBATCH --time=01:15:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260930-093452-luqia-canatune-smoke2-baseline-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260930-093452-luqia-canatune-smoke2-baseline-r2/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20260930-093452-luqia-canatune-smoke2-baseline-r2"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="479b73ca0016870c0fd335783bddb8c935d5ec68"
P_NODE=neptune
D_NODE=ganymede
WALL_S=4500
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
  for run in baseline; do
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
common=(--base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=true
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}")
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/cantune.yaml" \
  --set routing.policy=cantune --set canary.max_duty=1.0 --set canary.min_interval_s=300 \
  || fail job_config_cantune
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/baseline.yaml" \
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1]]' \
  || fail job_config_baseline
cp "${WORK}/baseline.yaml" "${WORK}/job_config.yaml"

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
run_mode() {  # $1 = run name, $2 = cantune | baseline, $3 = publish deadline in s
  local run="$1" mode="$2" deadline="$3"
  mkdir -p "${OUT}/${run}" "${WORK}/${run}"
  export CANATUNE_CONFIG="${WORK}/${mode}.yaml" SMOKE_MODE="${mode}"
  export PD_WORK_DIR="${WORK}/${run}/pd" CANATUNE_RECORD_DIR="${WORK}/${run}/records"
  export CANATUNE_TIER_TABLE="${WORK}/${run}/tiers.json" CANATUNE_RISK_TABLE="${WORK}/${run}/risk.json"
  export SMOKE_OUT="${OUT}/${run}" SMOKE_DEADLINE_S="${deadline}"
  echo "=== ${run} (${mode}) start=$(date -Iseconds) deadline_s=${deadline} proxy_port=${PROXY_PORT}"
  "${PY}" scripts/smoke_production.py
  local rc=$?
  case "${rc}" in
    0) echo "RESULT_${run}=ok";;
    2) echo "RESULT_${run}=failed stage=deadline (no table before the deadline)";;
    3) echo "RESULT_${run}=failed stage=service_stopped_early";;
    5) echo "RESULT_${run}=failed stage=calibration_failed (3 attempts)";;
    6) echo "RESULT_${run}=failed stage=load_run";;
    *) echo "RESULT_${run}=failed stage=smoke rc=${rc}";;
  esac
  return "${rc}"
}
TRACE="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260929-235619-luqia-canatune-smoke2-production-r1/results/270039/cantune/trace.json"
[ -s "${TRACE}" ] || fail trace_missing
cp "${TRACE}" "${OUT}/trace.json"
echo "--- clocks before the baseline (reset)"
reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
export SMOKE_TRACE="${TRACE}"
run_mode baseline baseline 0; RC_B=$?
if [ "${RC_B}" -eq 0 ]; then echo "RESULT=ok (baseline done)"; exit 0; fi
echo "RESULT=failed baseline=${RC_B}"
exit 1
