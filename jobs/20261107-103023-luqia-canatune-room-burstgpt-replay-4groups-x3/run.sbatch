#!/usr/bin/env bash
#SBATCH --job-name=canatune-room-replay
# 真实负载回放-异构-固定配对-P2pNccl-4组-SLO600/100, room 路由 (3 CanaTune runs, no baseline)
# P: 4 x L40S on neptune, D: 4 x L4 on ganymede, both nodes exclusive. G0 = (P0, D0) Canary,
# G1 / G2 / G3 production, fixed pairing, P2pNcclConnector PUT_ASYNC.
# Code: CanaTuneLLM branch spill-before-max (HEAD b0b2088 plus its uncommitted changes),
# packed next to this script as canatune_src.tar.gz (SOURCE.txt: provenance and sha256).
# Router: group_choice=room (the setting of the A100 runs 10651458 / 10654271), i.e. the first
# group in a fixed order whose P busy share is below p_rho and whose D holds fewer sequences
# than d_sequences (both measured by the Canary at the working clocks), with redirect.
# Only the router, the code version and its new defaults differ from the fill runs of job
# 20261014-185821 (Slurm 275892): same nodes, connector, SLO 600 / 100, Canary output 256.
# Workload: the same real-time BurstGPT replay as 275892 and the A100 replay 10654271
# (BurstGPT_1.csv seconds 1263605-1272005, 26282 requests, 3.13 req/s, no compression).
# No baseline in this job: compare with baseline_r1 of 275892 (same nodes, same replay).
# Every CanaTune run is a cold start (empty tier and risk tables; the Canary calibrates on
# G0, then the replay starts).
#   Runs: room_r1..r3 (cold start, experiment.slo 600 / 100 ms, then the replay)
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261107-103023-luqia-canatune-room-burstgpt-replay-4groups-x3/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261107-103023-luqia-canatune-room-burstgpt-replay-4groups-x3/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
# Many concurrent streams under overload: the proxy needs more file descriptors than the
# default soft limit (job H: "Too many open files"); srun passes the limit to its steps.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261107-103023-luqia-canatune-room-burstgpt-replay-4groups-x3"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
CODE_TAR="${JOB_DIR}/canatune_src.tar.gz"
CODE_SHA256="93435c2af7868c2973b4f9050657287769e43eb0f6c9a6af619ef651429b0623"
P_NODE=neptune
D_NODE=ganymede
WALL_S=57600
RESERVE_S=480   # teardown, result copy and clock reset before the wall limit
mkdir -p "${OUT}"
# HTTP access lines only bloat the console (and the broker history).
exec > >(grep --line-buffered -v '"GET /' | tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} code_sha256=${CODE_SHA256}"

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
  for dir in "${WORK}"/room_r*; do
    [ -d "${dir}" ] || continue
    run=$(basename "${dir}")
    mkdir -p "${OUT}/${run}"
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
  for f in job_config.yaml cantune_s600.yaml burstgpt_replay.report.json; do
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

# ---- 1. code from the packed working tree (checksum-verified) + CPU tests ----------------
echo "${CODE_SHA256}  ${CODE_TAR}" | sha256sum -c - || fail code_checksum
mkdir -p "${WORK}/CanaTune" && tar -xzf "${CODE_TAR}" -C "${WORK}/CanaTune" || fail code_extract
cd "${WORK}/CanaTune" || fail code_dir
cp "${JOB_DIR}/SOURCE.txt" "${OUT}/commit.txt" && cat "${OUT}/commit.txt"
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

# ---- 2b. the BurstGPT replay window, prompts of the logged lengths from ShareGPT ----------
REPLAY="${WORK}/burstgpt_replay.jsonl"
timeout 1800 "${PY}" scripts/make_burstgpt_dataset.py --csv "${BURSTGPT_CSV}" --sharegpt "${SHAREGPT}" \
  --tokenizer "${MODEL_PATH}" --out "${REPLAY}" --model all --max-len 4096 \
  --window-start 1263605 --window-s 8400 || fail burstgpt_replay_dataset
cp "${WORK}/burstgpt_replay.report.json" "${OUT}/" && cat "${OUT}/burstgpt_replay.report.json"
# The same requests as 275892 (whose report had 26282 rows written).
grep -q '"rows_written": 26282' "${OUT}/burstgpt_replay.report.json" || fail burstgpt_replay_rows

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
P_LINE=$(detect "${P_NODE}"); D_LINE=$(detect "${D_NODE}")
echo "detected ${P_NODE}=${P_LINE}" | tee "${OUT}/gpu_allocation.txt"
echo "detected ${D_NODE}=${D_LINE}" | tee -a "${OUT}/gpu_allocation.txt"
IFS=';' read -r P_IDS P_UUIDS P_NAMES <<< "${P_LINE}"
IFS=';' read -r D_IDS D_UUIDS D_NAMES <<< "${D_LINE}"
[[ "${P_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ && "${P_NAMES}" == *L40S*"|"*L40S*"|"*L40S*"|"*L40S* ]] || fail gpu_detect_prefill 3
[[ "${D_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ && "${D_NAMES}" == *L4* && "${D_NAMES}" != *L40S* ]] || fail gpu_detect_decode 3
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP=$(ip_of "${D_NODE}")
[[ "${P_IP}" =~ ^[0-9.]+$ && "${D_IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
echo "ips ${P_NODE}=${P_IP} ${D_NODE}=${D_IP}" | tee -a "${OUT}/gpu_allocation.txt"

# ---- 4. job config: as 275892's fill config, but group_choice=room ------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
AGENT_PORT=$(( 9300 + ${JOB_ID//[^0-9]/} % 500 ))   # r6: a fixed 9300 collided with another job
common=(--base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=true
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}"
  --set "clock_control.agent_port=${AGENT_PORT}"
  --set kv_transfer.connector=P2pNcclConnector)
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/cantune_s600.yaml" \
  --set routing.policy=cantune --set canary.max_output_tokens=256 \
  --set controller.solver=false --set router.group_choice=room \
  --set controller.redirect=true \
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100 || fail job_config_cantune_s600
cp "${WORK}/cantune_s600.yaml" "${WORK}/job_config.yaml"
grep -nE "group_choice|connector:|ttft_ms|tpot_ms|solver|redirect|workers" "${WORK}/job_config.yaml"

# ---- 5. cantune room (SLO 600 / 100), three cold starts, each followed by the replay -------
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
export CANATUNE_PROXY_WORKERS=0       # one service process, as in 275892 (3.13 req/s)
export BENCH_WORKLOADS="[
  {\"name\": \"replay\", \"dataset\": \"${REPLAY}\", \"replay\": true, \"segment_s\": 600,
   \"ignore_eos\": true}]"
export BENCH_SCORE_SLOS="600:100,1000:100" BENCH_STAGE_S=180 BENCH_GAP_S=30
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
LOAD_S=10200  # one run: start-up (~6 min) + the replay (2 h 20 min) + teardown
CAL_S=5400    # the Canary's cold start to the publish: fill ~57 min (275725), room adds room_p/room_d
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }
reset_all() {
  reset_clocks "${P_NODE}" "${P_IDS}" "${P_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
  reset_clocks "${D_NODE}" "${D_IDS}" "${D_UUIDS}" | tee -a "${OUT}/clock_reset_before.txt"
}
preflight() {  # NVML works and the loaded driver module is the one on disk, on both nodes
  local node out ok=0
  for node in "${P_NODE}" "${D_NODE}"; do
    out=$(timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c '
      k=$(awk "/NVRM version/{for(i=1;i<=NF;i++) if(\$i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/){print \$i; exit}}" /proc/driver/nvidia/version)
      d=$(modinfo -F version nvidia); nvidia-smi -L >/dev/null 2>&1 && n=ok || n=FAIL
      echo "loaded=$k disk=$d nvml=$n"' 2>&1 | grep '^loaded=')
    echo "[${node}] ${out}" | tee -a "${OUT}/preflight.txt"
    [[ "${out}" =~ ^loaded=([0-9.]+)\ disk=([0-9.]+)\ nvml=ok$ && "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]] || ok=1
  done
  return "${ok}"
}
clock_errors() {  # $1 = run: clock_error events in its records
  local n
  n=$(grep -c '"event": "clock_error"' "${WORK}/$1/records/events.jsonl" 2>/dev/null || true)
  echo "$1 clock_errors=${n:-0}" | tee -a "${OUT}/clock_errors.txt"
  # room: the measured limits, online knee updates, redirects, MAX entries; receive-buffer
  # overflow / dropped KV in the vLLM logs
  for e in locator_room redirect p_slot_update kv_gate_update admission_calibration capacity_expand; do
    n=$(cat "${WORK}/$1/records/"*.jsonl 2>/dev/null | grep -c "\"event\": \"$e\"" || true)
    echo "$1 ${e}=${n:-0}" | tee -a "${OUT}/clock_errors.txt"
  done
  n=$(cat "${WORK}/$1/pd"/*/logs/*.log 2>/dev/null | grep -cE "Out Of (Threshold|Memory)|kv_cache is None" || true)
  echo "$1 kv_overflow_or_drop=${n:-0}" | tee -a "${OUT}/clock_errors.txt"
}
RESULTS=""
for rep in 1 2 3; do
  run="room_r${rep}"
  if [ "$(left)" -le $(( CAL_S + LOAD_S )) ]; then
    RESULTS="${RESULTS} ${run}:skipped"; continue
  fi
  echo "--- preflight and clock reset before ${run}"
  preflight || { RESULTS="${RESULTS} ${run}:preflight_failed"; echo "RESULT=${RESULTS}"; fail "preflight_${run}" 4; }
  reset_all
  # cold start: no tier or risk table from an earlier run or job
  rm -f "${WORK}/${run}/tiers.json" "${WORK}/${run}/risk.json" "${WORK}/${run}/risk.json.slack.json"
  run_mode "${run}" cantune cantune_s600 "${CAL_S}"; RC=$?
  clock_errors "${run}"
  if [ "${RC}" -eq 3 ] && [ "$(left)" -gt $(( CAL_S + LOAD_S + 600 )) ]; then
    sleep 30   # a service crashed: one retry if time allows (as in 275892)
    preflight || { RESULTS="${RESULTS} ${run}b:preflight_failed"; echo "RESULT=${RESULTS}"; fail "preflight_${run}b" 4; }
    reset_all
    run_mode "${run}b" cantune cantune_s600 "${CAL_S}"; RC=$?
    clock_errors "${run}b"
  fi
  RESULTS="${RESULTS} ${run}:${RC}"
done
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *:[1-9]*|*skipped*|*failed*) exit 1;; esac
exit 0
