#!/usr/bin/env bash
#SBATCH --job-name=canatune-smoke1
# CanaTune smoke test (1): Canary cold-start calibration on 2 P/D pairs.
#   P: 2 x L40S on neptune, D: 2 x L4 on ganymede; G0 = (P0, D0) Canary, G1 = (P1, D1).
# The job clones CanaTune at a pinned commit, runs its CPU tests, starts GPU agents
# (clock locks via sudo nvidia-smi, UUID-checked), 2+2 vLLM servers and the CanaTune
# proxy. With no tier table, production G1 runs at MAX and the Canary calibrates
# at once (no external traffic). The job ends when the table is published (or at
# the deadline), collects the results and resets every clock it may have locked.
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-234719-luqia-canatune-smoke1-calibration/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-234719-luqia-canatune-smoke1-calibration/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20260925-234719-luqia-canatune-smoke1-calibration"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="c19b77a457dd5745eddcfd5307cd707f355aadc3"
P_NODE=neptune
D_NODE=ganymede
WALL_S=3600
RESERVE_S=480   # teardown, result copy and clock reset before the wall limit
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
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
  [ -d "${WORK}/records" ] && cp "${WORK}/records/"*.jsonl "${OUT}/" 2>/dev/null
  for f in "${WORK}/pd"/*/logs/*.log; do [ -f "$f" ] && tail -n 300 "$f" > "${OUT}/$(basename "$f" .log)_tail.log"; done
  [ -f "${WORK}/job_config.yaml" ] && cp "${WORK}/job_config.yaml" "${OUT}/"
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

# ---- 4. job config ------------------------------------------------------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
"${PY}" scripts/make_job_config.py --base config.yaml --out "${WORK}/job_config.yaml" \
  --prefill-node "${P_NODE}" --decode-node "${D_NODE}" \
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}" \
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}" \
  --python "${PY}" --work-dir "${WORK}" \
  --set routing.policy=cantune --set clock_control.enabled=true \
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}" || fail job_config

# ---- 5. run until the Canary publishes ---------------------------------------------------
export CANATUNE_CONFIG="${WORK}/job_config.yaml" PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" PD_WORK_DIR="${WORK}/pd" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
export DECODE_GPU_MEMORY_UTILIZATION=0.82 PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=1
export CANATUNE_RECORD_DIR="${WORK}/records" CANATUNE_TIER_TABLE="${WORK}/tiers.json"
export CANATUNE_RISK_TABLE="${WORK}/risk.json" SMOKE_OUT="${OUT}"
export SMOKE_DEADLINE_S=$(( T0 + WALL_S - RESERVE_S - $(date +%s) ))
echo "smoke_deadline_s=${SMOKE_DEADLINE_S} proxy_port=${PROXY_PORT}"
"${PY}" scripts/smoke_calibrate.py
RC=$?
case "${RC}" in
  0) echo "RESULT=ok (tier table published)";;
  2) echo "RESULT=failed stage=deadline (no table before the deadline)";;
  3) echo "RESULT=failed stage=service_stopped_early";;
  *) echo "RESULT=failed stage=smoke rc=${RC}";;
esac
exit "${RC}"
