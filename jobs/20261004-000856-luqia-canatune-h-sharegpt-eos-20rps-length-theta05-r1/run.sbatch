#!/usr/bin/env bash
#SBATCH --job-name=canatune-h
# CanaTune job H: job G's setup (ShareGPT, EOS honoured, max_tokens 256, length-aware D
# model) into overload: 1, 2, 4, 8, 12, 16, 20 req/s. Changes since job G: the router's
# KV-wall check counts the growth of the sequences already on D (half a mean output each,
# a whole one per request on its way; the simulated job F replay had preemptions and full
# effort at 10 req/s without it), router.theta 0.10 -> 0.05, the SLO is part of the table
# identity. Simulated beforehand (overload_sim, job E/F hardware): overload from ~12-16
# req/s in both modes; CanaTune rejects after hold_max_ms (30 s), the baseline queues.
# (CanaTune 3763da3; controller.solver off: every group at the Canary's H).
#   G0 = (P0, D0) Canary, G1 / G2 production; P: 3 x L40S on neptune, D: 3 x L4 on ganymede.
#   Deployment inputs only: model config.json, kv_buffer 1 GB, the output cap
#   canary.max_output_tokens=256 (cold-start probes generate it); the rest from the Canary.
#   0. ShareGPT_V3_unfiltered_cleaned_split.json (public HF dataset, no token) into a shared
#      cache /data/users/chjing/datasets, downloaded once (size checked); disk and HF checked first.
#   1. cantune  cold start: the Canary calibrates and publishes, production at H; then
#               scripts/bench_production.py runs `vllm bench serve` per request rate:
#               1, 2, 4, 8, 12, 16, 20 req/s (Poisson, 180 s of arrivals each, 30 s gap),
#               completions with text prompts (the proxy counts them with the tokenizer),
#               --sharegpt-output-len 256 without --ignore-eos: one cap, the model stops at
#               EOS (real output lengths; greedy, so nearly the same outputs in both modes);
#               (the Canary does not run again after the cold start: its duty budget
#               counts the cold start, a known issue; the D model is length-independent).
#               Overload best_effort; production feedback -> MAX.
#   2. baseline the same benchmark stages, round robin over the three pairs, driver clocks.
# Energy: NVML counters of all 6 GPUs (1 s), per stage over the benchmark's own span;
# network: NIC byte counters of both nodes every 2 s. Canary scheduling at config defaults.
#SBATCH --partition=long
#SBATCH --time=05:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261004-000856-luqia-canatune-h-sharegpt-eos-20rps-length-theta05-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261004-000856-luqia-canatune-h-sharegpt-eos-20rps-length-theta05-r1/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20261004-000856-luqia-canatune-h-sharegpt-eos-20rps-length-theta05-r1"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="3763da35f7f05929a9a1772537d2bb215f9aebd7"
P_NODE=neptune
D_NODE=ganymede
WALL_S=18000
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
  [ -n "${NET_PIDS:-}" ] && kill ${NET_PIDS} 2>/dev/null
  for run in cantune cantune_retry baseline; do
    [ -d "${WORK}/${run}" ] || continue
    [ -d "${WORK}/${run}/records" ] && cp "${WORK}/${run}/records/"*.jsonl "${OUT}/${run}/" 2>/dev/null
    for f in tiers.json risk.json risk.json.slack.json; do
      [ -f "${WORK}/${run}/$f" ] && cp "${WORK}/${run}/$f" "${OUT}/${run}/"
    done
    [ -f "${OUT}/${run}/energy.jsonl" ] && gzip -f "${OUT}/${run}/energy.jsonl"
    # Full vLLM logs (gzip); telemetry /metrics and /health polling lines dropped.
    mkdir -p "${OUT}/${run}/logs"
    for f in "${WORK}/${run}/pd"/*/logs/*.log; do
      [ -f "$f" ] && grep -vE 'GET /(metrics|health)' "$f" | gzip -c > "${OUT}/${run}/logs/$(basename "$f" .log).log.gz"
    done
  done
  for f in job_config.yaml cantune.yaml baseline.yaml; do
    [ -f "${WORK}/$f" ] && cp "${WORK}/$f" "${OUT}/"
  done
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }

# ---- 0. ShareGPT (public HF dataset, no token): shared cache, downloaded once ------------
DATA_DIR="/data/users/chjing/datasets"
SHAREGPT="${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_BYTES=672837942
mkdir -p "${DATA_DIR}" || fail dataset_dir
df -h /data | tee "${OUT}/data_disk.txt"
if [ "$(stat -c %s "${SHAREGPT}" 2>/dev/null || echo 0)" != "${SHAREGPT_BYTES}" ]; then
  FREE_KB=$(df -Pk "${DATA_DIR}" | awk 'NR==2 {print $4}')
  [ "${FREE_KB:-0}" -gt $(( 3 * 1024 * 1024 )) ] || fail data_disk_full   # 0.7 GB + margin
  timeout 60 curl -fsSIL -o /dev/null "${SHAREGPT_URL}" || fail huggingface_unreachable
  PART="${SHAREGPT}.part.${JOB_ID}"
  timeout 1800 curl -fsSL --retry 3 -o "${PART}" "${SHAREGPT_URL}" || { rm -f "${PART}"; fail sharegpt_download; }
  [ "$(stat -c %s "${PART}")" = "${SHAREGPT_BYTES}" ] || { rm -f "${PART}"; fail sharegpt_size; }
  mv -f "${PART}" "${SHAREGPT}" || fail sharegpt_move
  echo "sharegpt downloaded"
fi
echo "sharegpt=${SHAREGPT} bytes=$(stat -c %s "${SHAREGPT}")"

# ---- 1. code at the pinned commit + CPU tests -----------------------------------------
timeout 300 git clone --quiet "${REPO}" "${WORK}/CanaTune" || fail clone
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" || fail checkout
git log -1 --format='%H %s' | tee "${OUT}/commit.txt"
timeout 300 "${PY}" -m pip install --quiet --target "${WORK}/pytest-lib" "pytest>=8.3,<9" || fail pip_pytest
# The stall test of canatune.measure is timing-sensitive under load (unrelated to this run).
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
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --gpus-per-node=3 --mem=256M bash -c '
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
[[ "${P_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+$ && "${P_NAMES}" == *L40S*"|"*L40S*"|"*L40S* ]] || fail gpu_detect_prefill 3
[[ "${D_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+$ && "${D_NAMES}" == *L4* && "${D_NAMES}" != *L40S* ]] || fail gpu_detect_decode 3
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
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}"
  --set "clock_control.agent_port=${AGENT_PORT}")
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/cantune.yaml" \
  --set routing.policy=cantune --set canary.max_output_tokens=256 \
  --set controller.solver=false || fail job_config_cantune
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/baseline.yaml" \
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1], [P2, D2]]' \
  || fail job_config_baseline
cp "${WORK}/cantune.yaml" "${WORK}/job_config.yaml"

# ---- 5. cantune run, then baseline: the same benchmark stages ------------------------------
export PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
export DECODE_GPU_MEMORY_UTILIZATION=0.82 PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export BENCH_DATASET="${SHAREGPT}" BENCH_RATES="1,2,4,8,12,16,20" BENCH_STAGE_S=180 BENCH_GAP_S=30
export BENCH_OUTPUT_LEN=256 BENCH_IGNORE_EOS=0 BENCH_SEED=20261002 BENCH_BURSTINESS=1
export BENCH_ENDPOINT=/v1/completions BENCH_CMD="${ENV_DIR}/bin/vllm bench serve"
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=1
export PYTHONFAULTHANDLER=1   # native stack on a segfault (r5 run_a lost D0 to one)
run_mode() {  # $1 = run name, $2 = cantune | baseline, $3 = publish deadline in s
  local run="$1" mode="$2" deadline="$3"
  mkdir -p "${OUT}/${run}" "${WORK}/${run}"
  export CANATUNE_CONFIG="${WORK}/${mode}.yaml" BENCH_MODE="${mode}"
  export PD_WORK_DIR="${WORK}/${run}/pd" CANATUNE_RECORD_DIR="${WORK}/${run}/records"
  export CANATUNE_TIER_TABLE="${WORK}/${run}/tiers.json" CANATUNE_RISK_TABLE="${WORK}/${run}/risk.json"
  export BENCH_OUT="${OUT}/${run}" BENCH_DEADLINE_S="${deadline}"
  echo "=== ${run} (${mode}) start=$(date -Iseconds) deadline_s=${deadline} proxy_port=${PROXY_PORT}"
  "${PY}" scripts/bench_production.py
  local rc=$?
  case "${rc}" in
    0) echo "RESULT_${run}=ok";;
    2) echo "RESULT_${run}=failed stage=deadline (no table before the deadline)";;
    3) echo "RESULT_${run}=failed stage=service_stopped_early";;
    4) echo "RESULT_${run}=failed stage=never_ready";;
    5) echo "RESULT_${run}=failed stage=calibration_failed (3 attempts)";;
    6) echo "RESULT_${run}=failed stage=benchmark";;
    *) echo "RESULT_${run}=failed stage=bench rc=${rc}";;
  esac
  return "${rc}"
}
# NIC bytes of both nodes every 2 s (three P share neptune's NIC)
NET_PIDS=""
for node in "${P_NODE}" "${D_NODE}"; do
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=128M bash -c \
    'while true; do echo "t=$(date +%s.%N)"; cat /proc/net/dev; sleep 2; done' \
    > "${OUT}/net_${node}.txt" 2>&1 &
  NET_PIDS="${NET_PIDS} $!"
done
LOAD_S=4200   # one benchmark run: start-up (~5 min) + 7 stages (~4-5 min each, overload drains longer) + teardown
LEFT=$(( T0 + WALL_S - RESERVE_S - $(date +%s) ))
run_mode cantune cantune $(( LEFT - 2 * LOAD_S )); RC=$?
if [ "${RC}" -eq 3 ]; then  # a service crashed (r5: vLLM segfault): one retry if time allows
  sleep 30
  LEFT=$(( T0 + WALL_S - RESERVE_S - $(date +%s) ))
  if [ "${LEFT}" -gt $(( 2 * LOAD_S + 2400 )) ]; then
    run_mode cantune_retry cantune $(( LEFT - 2 * LOAD_S )); RC=$?
  fi
fi
[ "${RC}" -eq 0 ] || { echo "RESULT=failed cantune=${RC}"; exit 1; }

RC_B=skipped
if [ $(( T0 + WALL_S - RESERVE_S - $(date +%s) )) -gt "${LOAD_S}" ]; then
  echo "--- clocks before the baseline (reset)"
  reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
  reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
  run_mode baseline baseline 0; RC_B=$?
fi
echo "RESULT=cantune:ok baseline:${RC_B}"
[ "${RC_B}" = 0 ] && exit 0
exit 1
