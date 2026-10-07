#!/usr/bin/env bash
#SBATCH --job-name=canatune-n
# CanaTune job N: job M again (fixed P/D pairing, SLO TTFT 600 / TPOT 100 ms, ShareGPT and
# BurstGPT at 0.5-6 req/s, the same stages and seeds) with the windowed production TPOT
# feedback (router.tpot_feedback=window, the new default): per D the 0.95 quantile of the
# token gaps of the last 5 s against the TPOT SLO; one live gap counts only from 5 x the
# SLO and never as severe. Job M ran the previous rule (every request's latest gap
# against 0.9 x SLO, a live gap of 2 x SLO severe) and spent 125 of 181 s in full effort
# at ShareGPT 4 req/s with TPOT p95 81 ms (-7.6 %); this run measures what the window
# rule recovers. Confirming / full-effort events now record the feedback that caused them.
# (CanaTune 2b3f002; theta 0.10; controller.solver off: every group at H; D GPU
# memory utilization 0.82 as in M.)
#   G0 = (P0, D0) Canary, G1 / G2 production; P: 3 x L40S on neptune, D: 3 x L4 on ganymede.
#   Deployment inputs only: model config.json, kv_buffer 1 GB, the output cap
#   canary.max_output_tokens=256 (cold-start probes generate it); the rest from the Canary.
#   0. datasets in a shared cache /data/users/chjing/datasets, downloaded once (size checked):
#      ShareGPT_V3_unfiltered_cleaned_split.json (public HF dataset, no token) and
#      BurstGPT_1.csv (github.com/HPMLL/BurstGPT release v2.0, CC-BY-4.0; token counts of
#      real ChatGPT + GPT-4 traffic). scripts/make_burstgpt_dataset.py turns BurstGPT (all
#      models, prompt + output <= 4096) into 4000 requests with prompts of exactly the
#      logged length (ShareGPT text, Mistral tokenizer) and the logged output length.
#   Workloads per run, one after the other, Poisson, 180 s of arrivals per rate, 30 s gap:
#      sharegpt  0.5, 1, 2, 4, 6 req/s: completions with text prompts, --sharegpt-output-len
#                256 without --ignore-eos (real output lengths up to the cap, EOS stops)
#      burstgpt  0.5, 1, 2, 4, 6 req/s: output lengths from the trace (--ignore-eos)
#   Runs: 1. baseline  round robin over the three pairs, driver clocks; scored against
#                      600 / 100 and, for comparison with K, 600 / 65 and 1000 / 100 (by_slo)
#         2. cantune_s600  cold start with experiment.slo 600 / 100 ms: the Canary calibrates
#                      for it and publishes, production at H, then both workloads
#   Simulated beforehand (overload_sim, job G hardware, D step noise calibrated to job M's
#      token gaps): ShareGPT 6 req/s -5.6 -> -10.8 %, BurstGPT 6 req/s +0.2 -> -2.3 %, time
#      in full effort -27 to -51 %, goodput as before up to 6 req/s; the simulator only
#      partly reproduces M's false triggers, so the real gain may be larger.
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
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261007-160023-luqia-canatune-n-fixed-window-p95-slo600-100-r1/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261007-160023-luqia-canatune-n-fixed-window-p95-slo600-100-r1/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
# Many concurrent streams under overload: the proxy needs more file descriptors than the
# default soft limit (job H: "Too many open files"); srun passes the limit to its steps.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261007-160023-luqia-canatune-n-fixed-window-p95-slo600-100-r1"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="2b3f002579a1c5510e090be0d9c96cebc54e442a"
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
  for run in baseline cantune_s600 cantune_s600_retry; do
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
for slo in "600 100"; do
  read -r ttft tpot <<< "${slo}"
  "${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/cantune_s${ttft}.yaml" \
    --set routing.policy=cantune --set canary.max_output_tokens=256 \
    --set controller.solver=false --set "experiment.slo.ttft_ms=${ttft}" \
    --set "experiment.slo.tpot_ms=${tpot}" || fail "job_config_cantune_s${ttft}"
done
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/baseline.yaml" \
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1], [P2, D2]]' \
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
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
export DECODE_GPU_MEMORY_UTILIZATION=0.82 PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'
export BENCH_WORKLOADS="[
  {\"name\": \"sharegpt\", \"dataset\": \"${SHAREGPT}\", \"rates\": \"0.5,1,2,4,6\",
   \"output_len\": 256, \"ignore_eos\": false},
  {\"name\": \"burstgpt\", \"dataset\": \"${BURSTGPT}\", \"dataset_name\": \"custom\",
   \"rates\": \"0.5,1,2,4,6\", \"output_len\": null, \"ignore_eos\": true}]"
export BENCH_SCORE_SLOS="600:100,600:65,1000:100" BENCH_STAGE_S=180 BENCH_GAP_S=30
export BENCH_SEED=20261002 BENCH_BURSTINESS=1
export BENCH_ENDPOINT=/v1/completions BENCH_CMD="${ENV_DIR}/bin/vllm bench serve"
export STARTUP_TIMEOUT_S=900 ENDPOINT_TIMEOUT_S=1200 SHUTDOWN_GRACE_S=60 CANATUNE_STEP_GPUS=1
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
# NIC bytes of both nodes every 2 s (three P share neptune's NIC)
NET_PIDS=""
for node in "${P_NODE}" "${D_NODE}"; do
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=128M bash -c \
    'while true; do echo "t=$(date +%s.%N)"; cat /proc/net/dev; sleep 2; done' \
    > "${OUT}/net_${node}.txt" 2>&1 &
  NET_PIDS="${NET_PIDS} $!"
done
LOAD_S=3600   # one run's load: start-up (~5 min) + 2 x 5 stages (~4-5 min each) + teardown
CAL_S=4800    # the Canary's cold start to the publish (job G: ~54 min)
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }
reset_all() {
  reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
  reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
}
run_mode baseline baseline baseline 0; RC_B=$?
RESULTS="baseline:${RC_B}"
for ttft in 600; do
  run="cantune_s${ttft}"
  echo "--- clocks before ${run} (reset)"
  reset_all
  if [ "$(left)" -le $(( CAL_S + LOAD_S )) ]; then RESULTS="${RESULTS} ${run}:skipped"; continue; fi
  run_mode "${run}" cantune "${run}" "${CAL_S}"; RC=$?
  if [ "${RC}" -eq 3 ] && [ "$(left)" -gt $(( CAL_S + LOAD_S + 600 )) ]; then
    sleep 30   # a service crashed (r5: vLLM segfault): one retry if time allows
    reset_all
    run_mode "${run}_retry" cantune "${run}" "${CAL_S}"; RC=$?
  fi
  RESULTS="${RESULTS} ${run}:${RC}"
done
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *:[1-9]*|*skipped*) exit 1;; esac
exit 0
