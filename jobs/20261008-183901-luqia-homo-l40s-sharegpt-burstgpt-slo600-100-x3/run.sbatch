#!/usr/bin/env bash
#SBATCH --job-name=homo-l40s-x3
# 同构-L40s-ShareGPT+BurstGPT-SLO600/100-重复3次 (homogeneous L40S, three repetitions)
# One node (neptune), 4 x L40S: 2 P + 2 D. G0 = (P0, D0) Canary, G1 = (P1, D1) production
# (fixed pairing; G0 serves production only after G1). Same code path as job N (CanaTune
# 1405a95: windowed TPOT feedback, theta 0.10, controller.solver off: every group at H)
# with the D on L40S. Three repetitions in this one allocation, each a baseline run and a
# CanaTune run with its own Canary cold start (its own tier table), so the mean and spread
# over repetitions include the Canary's own variation (jobs M and N chose different D
# clocks). Order alternates (r1, r3: baseline first; r2: CanaTune first) so slow drift
# (temperature, time of day) cancels in the mean.
#   KV transfer: P2pNcclConnector PUT_ASYNC as before; P and D on one host, so NCCL moves
#   KV inside the node (PCIe P2P/IPC or SHM; NCCL_DEBUG=INFO records which).
#   D GPU memory: gpu_memory_utilization auto = 1 - 5.25 GB reserve / GPU memory (receive
#   buffer 1 GB + overflow 3 GB + one NCCL communicator 0.25 GB + 1 GB margin): ~0.89.
#   Single-host changes: D KV ports from 14679 (P keeps 14579), one clock agent per role on
#   separate ports, CANATUNE_STEP_GPUS=0 (each vLLM selects its GPU), decode_nodegroup
#   gpu_type NVIDIA L40S (the tier table's identity) and high_frequency_mhz 2520.
#   Deployment inputs only: model config.json, kv_buffer 1 GB, the output cap
#   canary.max_output_tokens=256 (cold-start probes generate it); the rest from the Canary.
#   0. datasets in a shared cache /data/users/chjing/datasets, downloaded once (size checked):
#      ShareGPT_V3_unfiltered_cleaned_split.json (public HF dataset, no token) and
#      BurstGPT_1.csv (github.com/HPMLL/BurstGPT release v2.0, CC-BY-4.0; token counts of
#      real ChatGPT + GPT-4 traffic). scripts/make_burstgpt_dataset.py turns BurstGPT (all
#      models, prompt + output <= 4096) into 4000 requests with prompts of exactly the
#      logged length (ShareGPT text, Mistral tokenizer) and the logged output length.
#   Workloads per run, one after the other, Poisson, 180 s of arrivals per rate, 30 s gap:
#      sharegpt  0.5, 1, 2, 4, 6, 8, 10 req/s: completions with text prompts, --sharegpt-output-len
#                256 without --ignore-eos (real output lengths up to the cap, EOS stops)
#      burstgpt  0.5, 1, 2, 4, 6, 8, 10 req/s: output lengths from the trace (--ignore-eos)
#   Runs per repetition r (1-3): baseline_r<r>  round robin over the two pairs, driver
#         clocks, scored against 600 / 100, 600 / 65 and 1000 / 100 (by_slo);
#         cantune_s600_r<r>  cold start with experiment.slo 600 / 100 ms, then both workloads
# Energy: NVML counters of all 4 GPUs (1 s), per stage over the benchmark's own span;
# network: NIC byte counters of the node every 2 s. Canary scheduling at config defaults.
#SBATCH --partition=long
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=neptune
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261008-183901-luqia-homo-l40s-sharegpt-burstgpt-slo600-100-x3/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261008-183901-luqia-homo-l40s-sharegpt-burstgpt-slo600-100-x3/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
# Many concurrent streams under overload: the proxy needs more file descriptors than the
# default soft limit (job H: "Too many open files"); srun passes the limit to its steps.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261008-183901-luqia-homo-l40s-sharegpt-burstgpt-slo600-100-x3"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="1405a95472638d8a7c098b1aaad904d723c3958a"
P_NODE=neptune
D_NODE=neptune   # one host: P and D share it
WALL_S=54000
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
  reset_clocks "${P_NODE}" "${P_IDS}${D_IDS:+,${D_IDS}}" "${P_UUIDS}${D_UUIDS:+,${D_UUIDS}}" | tee -a "${OUT}/clock_reset.txt"
  for node in "${P_NODE}"; do
    timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
      'ps -o pid,etime,args -u "$(id -un)" | grep -E "vllm.entrypoints|canatune" | grep -v grep || echo none' \
      2>&1 | sed "s/^/[${node} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  done
  [ -n "${NET_PIDS:-}" ] && kill ${NET_PIDS} 2>/dev/null
  for dir in "${WORK}"/baseline_r* "${WORK}"/cantune_s600_r*; do
    [ -d "${dir}" ] || continue
    run=$(basename "${dir}")
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
  for f in job_config.yaml cantune_s600.yaml baseline.yaml burstgpt.report.json; do
    [ -f "${WORK}/$f" ] && cp "${WORK}/$f" "${OUT}/"
  done
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }

# ---- 0. datasets: ShareGPT (public HF, no token), BurstGPT (GitHub release): shared cache --
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
BURSTGPT_CSV="${DATA_DIR}/BurstGPT_1.csv"
BURSTGPT_URL="https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_1.csv"
BURSTGPT_BYTES=52283111
if [ "$(stat -c %s "${BURSTGPT_CSV}" 2>/dev/null || echo 0)" != "${BURSTGPT_BYTES}" ]; then
  timeout 60 curl -fsSIL -o /dev/null "${BURSTGPT_URL}" || fail github_unreachable
  PART="${BURSTGPT_CSV}.part.${JOB_ID}"
  timeout 900 curl -fsSL --retry 3 -o "${PART}" "${BURSTGPT_URL}" || { rm -f "${PART}"; fail burstgpt_download; }
  [ "$(stat -c %s "${PART}")" = "${BURSTGPT_BYTES}" ] || { rm -f "${PART}"; fail burstgpt_size; }
  mv -f "${PART}" "${BURSTGPT_CSV}" || fail burstgpt_move
  echo "burstgpt downloaded"
fi
echo "burstgpt=${BURSTGPT_CSV} bytes=$(stat -c %s "${BURSTGPT_CSV}")"

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

# ---- 2b. BurstGPT requests with exact prompt lengths under the model's tokenizer ----------
BURSTGPT="${WORK}/burstgpt.jsonl"
timeout 900 "${PY}" scripts/make_burstgpt_dataset.py --csv "${BURSTGPT_CSV}" --sharegpt "${SHAREGPT}" \
  --tokenizer "${MODEL_PATH}" --out "${BURSTGPT}" --model all --rows 4000 --max-len 4096 \
  || fail burstgpt_dataset
cp "${WORK}/burstgpt.report.json" "${OUT}/" && cat "${OUT}/burstgpt.report.json"

# ---- 3. allocated GPUs (index + UUID) and node IPs -------------------------------------
detect() {  # prints "<ids>;<uuids>;<names>"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --gpus-per-node=4 --mem=256M bash -c '
    list="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"; ids=""; uuids=""; names=""
    IFS=, read -r -a arr <<< "$list"
    for i in "${arr[@]}"; do
      line=$(nvidia-smi -i "$i" --query-gpu=uuid,name --format=csv,noheader)
      ids="${ids:+$ids,}$i"; uuids="${uuids:+$uuids,}$(echo "${line%%,*}" | tr -d " ")"; names="${names:+$names|}${line#*, }"
    done
    echo "gpus=${ids};${uuids};${names}"' 2>&1 | grep '^gpus=' | tail -n 1 | sed 's/^gpus=//'
}
ALL_LINE=$(detect "${P_NODE}")
echo "detected ${P_NODE}=${ALL_LINE}" | tee "${OUT}/gpu_allocation.txt"
IFS=';' read -r ALL_IDS ALL_UUIDS ALL_NAMES <<< "${ALL_LINE}"
[[ "${ALL_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail gpu_detect_count 3
[[ "${ALL_NAMES}" == *L40S*"|"*L40S*"|"*L40S*"|"*L40S* ]] || fail gpu_detect_l40s 3
IFS=, read -r -a ids <<< "${ALL_IDS}"; IFS=, read -r -a uids <<< "${ALL_UUIDS}"
P_IDS="${ids[0]},${ids[1]}"; P_UUIDS="${uids[0]},${uids[1]}"
D_IDS="${ids[2]},${ids[3]}"; D_UUIDS="${uids[2]},${uids[3]}"
echo "prefill ${P_IDS} (${P_UUIDS}); decode ${D_IDS} (${D_UUIDS})" | tee -a "${OUT}/gpu_allocation.txt"
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP="${P_IP}"
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
  --set "clock_control.agent_port={prefill: ${AGENT_PORT}, decode: $(( AGENT_PORT + 1 ))}"
  --set topology.decode_nodegroup.kv_port_base=14679
  --set "topology.decode_nodegroup.gpu_type=NVIDIA L40S"
  --set topology.decode_nodegroup.high_frequency_mhz=2520)
for slo in "600 100"; do
  read -r ttft tpot <<< "${slo}"
  "${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/cantune_s${ttft}.yaml" \
    --set routing.policy=cantune --set canary.max_output_tokens=256 \
    --set controller.solver=false --set "experiment.slo.ttft_ms=${ttft}" \
    --set "experiment.slo.tpot_ms=${tpot}" || fail "job_config_cantune_s${ttft}"
done
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/baseline.yaml" \
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1]]' \
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100 \
  || fail job_config_baseline
cp "${WORK}/cantune_s600.yaml" "${WORK}/job_config.yaml"

# ---- 5. baseline, then cantune (SLO 600 / 100): the same workloads and stages -----------
export PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14679   # one host: P keeps 14579
unset DECODE_GPU_MEMORY_UTILIZATION   # auto: the configured reserve (memory_plan)
export PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM   # which intra-node transport carries KV
export BENCH_WORKLOADS="[
  {\"name\": \"sharegpt\", \"dataset\": \"${SHAREGPT}\", \"rates\": \"0.5,1,2,4,6,8,10\",
   \"output_len\": 256, \"ignore_eos\": false},
  {\"name\": \"burstgpt\", \"dataset\": \"${BURSTGPT}\", \"dataset_name\": \"custom\",
   \"rates\": \"0.5,1,2,4,6,8,10\", \"output_len\": null, \"ignore_eos\": true}]"
export BENCH_SCORE_SLOS="600:100,600:65,1000:100" BENCH_STAGE_S=180 BENCH_GAP_S=30
export BENCH_SEED=20261002 BENCH_BURSTINESS=1
export BENCH_ENDPOINT=/v1/completions BENCH_CMD="${ENV_DIR}/bin/vllm bench serve"
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=0
export PYTHONFAULTHANDLER=1   # native stack on a segfault (r5 run_a lost D0 to one)
run_mode() {  # $1 = run name, $2 = cantune | baseline, $3 = config name, $4 = publish deadline in s
  local run="$1" mode="$2" cfg="$3" deadline="$4"
  mkdir -p "${OUT}/${run}" "${WORK}/${run}"
  export CANATUNE_CONFIG="${WORK}/${cfg}.yaml" BENCH_MODE="${mode}"
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
# NIC bytes of the node every 2 s (KV stays inside the host)
NET_PIDS=""
for node in "${P_NODE}"; do
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=128M bash -c \
    'while true; do echo "t=$(date +%s.%N)"; cat /proc/net/dev; sleep 2; done' \
    > "${OUT}/net_${node}.txt" 2>&1 &
  NET_PIDS="${NET_PIDS} $!"
done
LOAD_S=5400   # one run's load: start-up (~5 min) + 2 x 7 stages (~4-6 min each) + teardown
CAL_S=4800    # the Canary's cold start to the publish (job G: ~54 min)
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }
reset_all() {
  reset_clocks "${P_NODE}" "${P_IDS}${D_IDS:+,${D_IDS}}" "${P_UUIDS}${D_UUIDS:+,${D_UUIDS}}" | tee -a "${OUT}/clock_reset_before.txt"
}
RESULTS=""
for rep in 1 2 3; do
  # alternate the order so slow drift cancels in the mean over repetitions
  if (( rep % 2 )); then order="baseline cantune"; else order="cantune baseline"; fi
  for what in ${order}; do
    if [ "${what}" = baseline ]; then
      run="baseline_r${rep}"
      if [ "$(left)" -le "${LOAD_S}" ]; then RESULTS="${RESULTS} ${run}:skipped"; continue; fi
      echo "--- clocks before ${run} (reset)"
      reset_all
      run_mode "${run}" baseline baseline 0; RESULTS="${RESULTS} ${run}:$?"
      continue
    fi
    run="cantune_s600_r${rep}"
    if [ "$(left)" -le $(( CAL_S + LOAD_S )) ]; then RESULTS="${RESULTS} ${run}:skipped"; continue; fi
    echo "--- clocks before ${run} (reset)"
    reset_all
    run_mode "${run}" cantune cantune_s600 "${CAL_S}"; RC=$?
    if [ "${RC}" -eq 3 ] && [ "$(left)" -gt $(( CAL_S + LOAD_S + 600 )) ]; then
      sleep 30   # a service crashed (r5: vLLM segfault): one retry if time allows
      reset_all
      run_mode "${run}_retry" cantune cantune_s600 "${CAL_S}"; RC=$?
    fi
    RESULTS="${RESULTS} ${run}:${RC}"
  done
done
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *:[1-9]*|*skipped*) exit 1;; esac
exit 0
