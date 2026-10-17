#!/usr/bin/env bash
#SBATCH --job-name=canatune-levers
# Lever isolation for the paper's motivation (section 3.1, "Prefill and Decode Save Energy
# Differently"): the clock and batching levers, each alone, over the full clock range, under
# load, with one measurement method. One P/D pair (P0: L40S on neptune, D0: L4 on ganymede,
# Mistral-7B, P2pNccl PUT_ASYNC); the second pair only idles. canatune.measure via
# scripts/measure_model.py (CanaTune de75904, branch tp), outside the control loop: clocks are
# locked through the agents (snapped to supported clocks), requests go straight to vLLM, and
# every window records the energy counter of every GPU, vLLM counter deltas and per-request
# stage times. Windows are >= 60 s, so the ~100 ms resolution of the energy counter does not
# matter. Six runs, each a fresh service and a resumable plan (plans/*.json):
#   pclk  clock only, prefill: P 210..2520 (16 clocks) x open loop 0.5 and 2.0 req/s, 90 s,
#         default mix (128/512/1024/2048 prompts, output 64), D at 2040; idle power per P clock
#   dclk  clock only, decode: D 210..2040 (13 clocks) x closed loop 8 and 32 concurrent,
#         60 s, 128-token prompts, output 256, P at 2520; idle power per D clock
#   dbat  batching only, decode: D 2040 and 1170 x concurrency 1,2,4,8,16,32,48 (as dclk)
#   pb25  batching only, prefill: P 2520, closed loop 1,2,4,8,16 concurrent, 1024-token
#         prompts, output 1 (P batches up to the concurrency), D at 2040
#   pb5   as pb25 with 512-token prompts and up to 32 concurrent
#   pb10  as pb25 at P 1080
# (Prefill batching stops at 16 x 1024 / 32 x 512 tokens in flight, below D0's KV capacity of
# about 26k tokens, so that P2pNccl never spills or drops a KV cache.)
#SBATCH --partition=long
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261017-212918-luqia-lever-isolation-pd/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261017-212918-luqia-lever-isolation-pd/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261017-212918-luqia-lever-isolation-pd"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="de75904"   # branch tp (the clock-grid job's version)
P_NODE=neptune
D_NODE=ganymede
WALL_S=21600
RESERVE_S=480   # teardown, result copy and clock reset before the wall limit
RUNS="pclk dclk dbat pb25 pb5 pb10"
mkdir -p "${OUT}"
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
      sleep 3
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
  for run in ${RUNS}; do
    mkdir -p "${OUT}/${run}"
    [ -d "${WORK}/${run}/records" ] && cp "${WORK}/${run}/records/"*.jsonl "${OUT}/${run}/" 2>/dev/null
    for f in "${WORK}/${run}/pd"/*/logs/*.log; do
      [ -f "$f" ] && grep -vE 'GET /(metrics|health)' "$f" | tail -n 300 > "${OUT}/${run}/$(basename "$f" .log)_tail.log"
    done
  done
  [ -f "${WORK}/measure.yaml" ] && cp "${WORK}/measure.yaml" "${OUT}/job_config.yaml"
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }

# ---- 1. code at the pinned commit + CPU tests -----------------------------------------
timeout 300 git clone --quiet --branch tp "${REPO}" "${WORK}/CanaTune" || fail clone
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" || fail checkout
git log -1 --format='%H %s' | tee "${OUT}/commit.txt"
timeout 300 "${PY}" -m pip install --quiet --target "${WORK}/pytest-lib" "pytest>=8.3,<9" || fail pip_pytest
PYTHONPATH="${PWD}/src:${WORK}/pytest-lib" timeout 600 "${PY}" -m pytest -p no:cacheprovider \
  --deselect tests/test_measure.py::test_stalled_window_is_repeated_and_does_not_stop_the_scan \
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

# ---- 4. job config (round robin: the Router, Controller and Canary never run) ---------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
AGENT_PORT=$(( 9300 + ${JOB_ID//[^0-9]/} % 500 ))
"${PY}" scripts/make_job_config.py --base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}" \
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}" \
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}" \
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=true \
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}" \
  --set "clock_control.agent_port=${AGENT_PORT}" \
  --set kv_transfer.connector=P2pNcclConnector \
  --set routing.policy=round_robin --out "${WORK}/measure.yaml" || fail job_config

# ---- 5. measurement plans ------------------------------------------------------------------
mkdir -p "${OUT}/plans"
"${PY}" - "${OUT}/plans" <<'EOF' || fail plans
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
P = [210, 360, 510, 660, 810, 960, 1110, 1260, 1410, 1560, 1710, 1860, 2010, 2160, 2310, 2520]
D = [210, 360, 510, 660, 810, 960, 1110, 1260, 1410, 1560, 1710, 1860, 2040]
base = {"warmup_s": 90.0, "warmup_rate": 1.0, "power_s": 15.0, "settle_s": 2.0,
        "decode_window_s": 60.0, "load_window_s": 90.0, "seed": 20261017}
decode_mix = {"default": [[128, 256]]}
plans = {
    "pclk": {"tag": "pclk", "stages": ["warmup", "power", "scans"],
             "prefill_clocks": P, "decode_clocks": [2040],
             "base_prefill_clock": 2520, "base_decode_clock": 2040, "mix_clock": 2520,
             "scans": [{"name": "pclk", "p": f, "d": 2040, "mix": "default",
                        "rates": [0.5, 2.0]} for f in P]},
    "dclk": {"tag": "dclk", "stages": ["warmup", "power", "decode"],
             "prefill_clocks": [2520], "decode_clocks": D,
             "base_prefill_clock": 2520, "base_decode_clock": 2040, "mix_clock": 2520,
             "mixes": decode_mix, "decode_concurrency": [8, 32], "decode_output": 256},
    "dbat": {"tag": "dbat", "stages": ["warmup", "decode"],
             "prefill_clocks": [2520], "decode_clocks": [2040, 1170],
             "base_prefill_clock": 2520, "base_decode_clock": 2040, "mix_clock": 2520,
             "mixes": decode_mix, "decode_concurrency": [1, 2, 4, 8, 16, 32, 48],
             "decode_output": 256},
}
for name, p_mhz, length, conc in (("pb25", 2520, 1024, [1, 2, 4, 8, 16]),
                                  ("pb5", 2520, 512, [1, 2, 4, 8, 16, 32]),
                                  ("pb10", 1080, 1024, [1, 2, 4, 8, 16])):
    plans[name] = {"tag": name, "stages": ["warmup", "decode"],
                   "prefill_clocks": [p_mhz], "decode_clocks": [2040],
                   "base_prefill_clock": p_mhz, "base_decode_clock": 2040, "mix_clock": p_mhz,
                   "mixes": {"default": [[length, 1]]}, "decode_concurrency": conc,
                   "decode_output": 1}
for name, plan in plans.items():
    (out / f"{name}.json").write_text(json.dumps({**base, **plan}, indent=1) + "\n")
print("plans", sorted(plans))
EOF

# ---- 6. runs: one fresh service per plan, resumable --------------------------------------------
export PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
unset DECODE_GPU_MEMORY_UTILIZATION   # auto: the configured reserve (memory_plan, P2pNccl terms)
export PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=1
export PYTHONFAULTHANDLER=1
export CANATUNE_CONFIG="${WORK}/measure.yaml"
echo "--- clocks before the measurements (reset)"
reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
declare -A BUDGET=([pclk]=6600 [dclk]=4200 [dbat]=2400 [pb25]=1200 [pb5]=1320 [pb10]=1200)
STATUS=0
for run in ${RUNS}; do
  mkdir -p "${OUT}/${run}" "${WORK}/${run}"
  export MEASURE_PLAN="${OUT}/plans/${run}.json" MEASURE_OUT="${OUT}/${run}/data"
  export PD_WORK_DIR="${WORK}/${run}/pd" CANATUNE_RECORD_DIR="${WORK}/${run}/records"
  RC=1
  for attempt in 1 2; do
    LEFT=$(( T0 + WALL_S - RESERVE_S - $(date +%s) ))
    [ "${LEFT}" -gt 900 ] || { echo "RESULT_${run}=skipped (no time)"; STATUS=1; break 2; }
    DL=$(( LEFT - 300 )); [ "${DL}" -lt "${BUDGET[$run]}" ] || DL="${BUDGET[$run]}"
    export MEASURE_DEADLINE_S="${DL}"
    echo "=== ${run} attempt ${attempt} start=$(date -Iseconds) deadline_s=${DL}"
    "${PY}" scripts/measure_model.py; RC=$?
    [ "${RC}" -eq 0 ] && break
    [ "${RC}" -eq 7 ] && break   # stopped for time: keep what was measured, move on
    echo "${run} attempt ${attempt} failed rc=${RC}; resuming after the last finished window"
    sleep 30
  done
  case "${RC}" in
    0) echo "RESULT_${run}=ok";;
    7) echo "RESULT_${run}=partial (time)"; STATUS=1;;
    *) echo "RESULT_${run}=failed rc=${RC}"; STATUS=1;;
  esac
done
echo "RESULT=$([ "${STATUS}" -eq 0 ] && echo ok || echo partial)"
exit "${STATUS}"
