#!/usr/bin/env bash
#SBATCH --job-name=canatune-forced-recal
# 强制重新标定-默认合成长度冷启动-30分钟后全部变短-异构-固定配对-P2pNccl-4组-SLO600/100
# Does a forced full re-calibration after a length shift move the clocks, and what does it
# do to the energy? P: 4 x L40S on neptune, D: 4 x L4 on ganymede, both nodes exclusive,
# Mistral-7B. G0 = (P0, D0) Canary, G1 / G2 / G3 production, fixed pairing, P2pNcclConnector
# PUT_ASYNC, solver off, router.group_choice=fill, controller.redirect=true.
# Code: CanaTuneLLM branch multi-rho (HEAD 9f3016d plus its uncommitted changes), packed next
# to this script as canatune_src.tar.gz (SOURCE.txt: provenance and sha256).
# Workload: one real-time replay, Poisson arrivals (make_shift_replay.py, seed 20261113):
#   phase A  30 min, 1.2 req/s: the mix the cold start assumes, prompts drawn uniformly from
#            canary.default_prompts (128 / 512 / 1024 / 2048), every request generating
#            canary.max_output_tokens (256)
#   phase S  2.5 h, 2 req/s: short requests only, prompt mean 64 and output mean 32 tokens
#            (clipped log-normal), nothing above 128 tokens
# Runs, each replaying the same requests, reported per 10-minute segment:
#   adapt_r1    cold start with the default probe lengths, then the replay. force_recal.py
#               asks for one full calibration (POST /canatune/canary "full") 1 h into the
#               replay, 30 min after the shift. By then the Canary's length window (last 2000
#               finished requests) has held phase S alone for ~13 min (checked before the
#               POST), and frozen-table operation on the short mix has two clean segments
#               (40-60 min) to compare with. No other experiment can start: periodic
#               recheck, length-shift verification and drift relocation are switched off, so
#               the forced calibration is the only difference from frozen_r1.
#   frozen_r1   the cold-start calibration of adapt_r1 (tier and risk tables exported at the
#               publish), canary.recalibrate=false, no forced calibration
#   baseline_r1 round robin over the four pairs, driver clocks reset with -rgc
# Settings that differ from the defaults in adapt_r1, and why:
#   canary.max_duty=1.0             the cold start counts toward the duty budget (~55 min of
#                                   experiments), which at 0.1 refuses any start for ~9 h
#   canary.locator.cache_ttl_s=2400 a manual "full" keeps the window cache (keys carry no
#                                   lengths); 2400 s < the age of the last cold-start window
#                                   when the calibration is requested (checked before the POST)
#   canary.periodic_s / length_shift / drift_min_samples = 1e9: no other experiment
# Proxy: decision process + 4 relay processes (CANATUNE_PROXY_WORKERS=4) in all three runs.
# Lessons of 280147 applied: the load stays far below what three groups carry (no pressure
# that aborts the experiment, no full effort, whose exit compares req/s); the longest prompt
# is 2048 tokens (KV well below the 1 GB P2pNccl receive buffer).
# r2 of 20261113-184612 (Slurm 283686), which stopped in the CPU tests after 62 s on one
# timing-sensitive test and ran nothing. Here the CPU tests only report; an import and
# syntax check of what the runs use is the gate. Nothing else differs.
# Read: recal_summary.txt (H before -> after, power / J per token / goodput of A, S_before,
# S_during, S_after for each run), adapt_r1/forced_recal.jsonl, tiers_before/after_forced.json.
#SBATCH --partition=long
#SBATCH --time=12:30:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261114-082951-luqia-canatune-forced-recal-default-to-short-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261114-082951-luqia-canatune-forced-recal-default-to-short-r2/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
# Many concurrent streams under overload: the proxy needs more file descriptors than the
# default soft limit (job H: "Too many open files"); srun passes the limit to its steps.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261114-082951-luqia-canatune-forced-recal-default-to-short-r2"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
CODE_TAR="${JOB_DIR}/canatune_src.tar.gz"
CODE_SHA256="07d80b74fa4ba6a80e8983f6cd89046eb45237820ba411c679523d30afc32c27"
P_NODE=neptune
D_NODE=ganymede
WALL_S=45000
RESERVE_S=480   # teardown, result copy and clock reset before the wall limit
# ---- the experiment's own parameters -----------------------------------------------------
PHASE_A_S=1800; PHASE_A_RATE=1.2     # default-length mix (lengths read from the job config)
PHASE_S_S=9000; PHASE_S_RATE=2.0     # short mix
S_PROMPT_MEAN=64; S_OUTPUT_MEAN=32; S_MAX=128
SEGMENT_S=600
FORCE_AT_S=3600                      # replay second at which the full calibration is requested
CACHE_TTL_S=2400                     # locator window cache in adapt_r1 (default 3600)
GROUP_CHOICE=fill
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
  [ -n "${FORCE_PID:-}" ] && kill "${FORCE_PID}" 2>/dev/null
  for dir in "${WORK}"/adapt_r* "${WORK}"/frozen_r* "${WORK}"/baseline_r*; do
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
  for f in job_config.yaml adaptive.yaml frozen.yaml baseline.yaml shift_replay.report.json; do
    [ -f "${WORK}/$f" ] && cp "${WORK}/$f" "${OUT}/"
  done
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }

# ---- 0. dataset: ShareGPT (public HF, no token), prompt text only: shared cache ----------
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

# ---- 1. code from the packed working tree (checksum-verified) + CPU tests ----------------
echo "${CODE_SHA256}  ${CODE_TAR}" | sha256sum -c - || fail code_checksum
mkdir -p "${WORK}/CanaTune" && tar -xzf "${CODE_TAR}" -C "${WORK}/CanaTune" || fail code_extract
cd "${WORK}/CanaTune" || fail code_dir
cp "${JOB_DIR}/SOURCE.txt" "${OUT}/commit.txt" && cat "${OUT}/commit.txt"
# The proxy processes talk over Unix sockets in a temporary directory; TMPDIR of this job is
# on NFS (/data), so the sockets go to CANATUNE_IPC_DIR (node-local /dev/shm) instead.
sed -i 's|tempfile.mkdtemp(prefix="canatune-")|tempfile.mkdtemp(prefix="canatune-", dir=os.environ.get("CANATUNE_IPC_DIR"))|' src/canatune/proxy/workers.py
grep -q 'dir=os.environ.get("CANATUNE_IPC_DIR")' src/canatune/proxy/workers.py || fail ipc_dir_patch
echo "patched: proxy IPC sockets in CANATUNE_IPC_DIR" | tee -a "${OUT}/commit.txt"
# The CPU tests are a report only (pytest.txt): 283686 lost a 5 h queue wait to one
# timing-sensitive test (40 ms window, 46 of 80 workers started in time on the node; it
# passes locally). What must hold is that the packed code loads on this Python: checked below.
if timeout 300 "${PY}" -m pip install --quiet --target "${WORK}/pytest-lib" "pytest>=8.3,<9"; then
  PYTHONPATH="${PWD}/src:${WORK}/pytest-lib" timeout 600 "${PY}" -m pytest -p no:cacheprovider \
    --deselect tests/test_measure.py::test_stalled_window_is_repeated_and_does_not_stop_the_scan \
    > "${OUT}/pytest.txt" 2>&1 || echo "WARNING: CPU tests not clean (see pytest.txt), continuing"
  tail -n 1 "${OUT}/pytest.txt"; grep -E "^FAILED|^ERROR" "${OUT}/pytest.txt"
else
  echo "WARNING: pytest not installed, CPU tests skipped"
fi
PYTHONPATH="${PWD}/src" "${PY}" -c 'import canatune.service, canatune.controller.process_controller, canatune.controller.canary, canatune.controller.locator, canatune.proxy.proxy, canatune.infrastructure.gpu_agent, canatune.api.control' || fail import_check
for f in "${JOB_DIR}/make_shift_replay.py" "${JOB_DIR}/force_recal.py" "${JOB_DIR}/summarize_recal.py" \
         scripts/bench_production.py scripts/make_job_config.py; do
  "${PY}" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$f" || fail "syntax_$(basename "$f")"
done
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

# ---- 4. job configs -----------------------------------------------------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
AGENT_PORT=$(( 9300 + ${JOB_ID//[^0-9]/} % 500 ))   # r6: a fixed 9300 collided with another job
common=(--base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=true
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}"
  --set "clock_control.agent_port=${AGENT_PORT}"
  --set kv_transfer.connector=P2pNcclConnector)
cantune=(--set routing.policy=cantune --set canary.max_output_tokens=256
  --set controller.solver=false --set "router.group_choice=${GROUP_CHOICE}"
  --set controller.redirect=true
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100)
# adaptive: only the forced calibration may run after the cold start (see the header)
"${PY}" scripts/make_job_config.py "${common[@]}" "${cantune[@]}" --out "${WORK}/adaptive.yaml" \
  --set canary.max_duty=1.0 --set "canary.locator.cache_ttl_s=${CACHE_TTL_S}" \
  --set canary.periodic_s=1000000000 --set canary.length_shift=1000000000 \
  --set canary.drift_min_samples=1000000000 || fail job_config_adaptive
"${PY}" scripts/make_job_config.py "${common[@]}" "${cantune[@]}" --out "${WORK}/frozen.yaml" \
  --set canary.recalibrate=false || fail job_config_frozen
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/baseline.yaml" \
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1], [P2, D2], [P3, D3]]' \
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100 \
  || fail job_config_baseline
cp "${WORK}/adaptive.yaml" "${WORK}/job_config.yaml"
grep -nE "group_choice|connector:|ttft_ms|tpot_ms|solver|redirect|workers|max_duty|cache_ttl_s|periodic_s|length_shift|drift_min_samples|recalibrate" "${WORK}/job_config.yaml"

# ---- 4b. the two-phase replay; phase A lengths are the Canary's cold-start lengths --------
REPLAY="${WORK}/shift_replay.jsonl"
read -r A_PROMPTS A_OUTPUT < <("${PY}" - "${WORK}/adaptive.yaml" <<'EOF'
import sys, yaml
canary = yaml.safe_load(open(sys.argv[1]))["canary"]
print(",".join(str(int(p)) for p in canary["default_prompts"]), int(canary["max_output_tokens"]))
EOF
)
[[ "${A_PROMPTS}" =~ ^[0-9,]+$ && "${A_OUTPUT}" =~ ^[0-9]+$ ]] || fail cold_start_lengths
echo "phase A lengths: prompts ${A_PROMPTS}, output ${A_OUTPUT}"
timeout 2400 "${PY}" "${JOB_DIR}/make_shift_replay.py" --scripts "${WORK}/CanaTune/scripts" \
  --sharegpt "${SHAREGPT}" --tokenizer "${MODEL_PATH}" --out "${REPLAY}" \
  --a-s "${PHASE_A_S}" --a-rate "${PHASE_A_RATE}" --a-prompts "${A_PROMPTS}" --a-output "${A_OUTPUT}" \
  --s-s "${PHASE_S_S}" --s-rate "${PHASE_S_RATE}" --s-prompt-mean "${S_PROMPT_MEAN}" \
  --s-output-mean "${S_OUTPUT_MEAN}" --s-max "${S_MAX}" || fail replay_dataset
cp "${WORK}/shift_replay.report.json" "${OUT}/" || fail replay_report

# ---- 5. adapt_r1, frozen_r1, baseline_r1: the same replay ---------------------------------
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
# Decision process + 4 relay processes (as the A100 runs; first use on this cluster): the
# forced calibration ramps short probes to tens of req/s in the decision process, which must
# not also relay production streams. The same setting in all three runs.
export CANATUNE_PROXY_WORKERS=4 CANATUNE_IPC_DIR=/dev/shm
export BENCH_WORKLOADS="[
  {\"name\": \"replay\", \"dataset\": \"${REPLAY}\", \"replay\": true, \"segment_s\": ${SEGMENT_S},
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
LOAD_S=11700  # one run: start-up (~6 min) + the replay (3 h) + teardown
CAL_S=5400    # the Canary's cold start to the publish (280588: 3950-4330 s after the start)
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
run_guarded() {  # $1 = run, $2 = mode, $3 = config, $4 = publish deadline, $5 = time needed
  local run="$1" RC
  if [ "$(left)" -le "$5" ]; then RESULTS="${RESULTS} ${run}:skipped"; return 1; fi
  echo "--- preflight and clock reset before ${run}"
  preflight || { RESULTS="${RESULTS} ${run}:preflight_failed"; echo "RESULT=${RESULTS}"; fail "preflight_${run}" 4; }
  reset_all
  run_mode "$1" "$2" "$3" "$4"; RC=$?
  [ "$2" != baseline ] && clock_errors "${run}"
  RESULTS="${RESULTS} ${run}:${RC}"
  return "${RC}"
}
# adapt_r1: cold start (no tier or risk table), the replay, one forced full calibration.
# The calibration published by the cold start is exported for frozen_r1 before any request.
mkdir -p "${WORK}/adapt_r1" "${OUT}/adapt_r1"
rm -f "${WORK}/adapt_r1/tiers.json" "${WORK}/adapt_r1/risk.json" "${WORK}/adapt_r1/risk.json.slack.json"
export BENCH_CALIBRATION_OUT="${WORK}/cold_calibration"
sync; sleep 1
"${PY}" "${JOB_DIR}/force_recal.py" --base "http://127.0.0.1:${PROXY_PORT}" \
  --console "${OUT}/job_console.log" --console-offset "$(stat -c %s "${OUT}/job_console.log")" \
  --report "${WORK}/shift_replay.report.json" --out "${OUT}/adapt_r1" \
  --cache-ttl-s "${CACHE_TTL_S}" --not-before-s "${FORCE_AT_S}" &
FORCE_PID=$!
run_guarded adapt_r1 cantune adaptive "${CAL_S}" $(( CAL_S + LOAD_S ))
if kill -0 "${FORCE_PID}" 2>/dev/null; then
  kill "${FORCE_PID}" 2>/dev/null; RESULTS="${RESULTS} forced_recal:unfinished"
else
  wait "${FORCE_PID}"; RESULTS="${RESULTS} forced_recal:$?"
fi
FORCE_PID=""
unset BENCH_CALIBRATION_OUT
if [ -s "${WORK}/cold_calibration/tier_table.json" ]; then
  mkdir -p "${WORK}/frozen_r1" "${OUT}/cold_calibration"
  cp "${WORK}/cold_calibration/"*.json "${OUT}/cold_calibration/"
  cp "${WORK}/cold_calibration/tier_table.json" "${WORK}/frozen_r1/tiers.json"
  cp "${WORK}/cold_calibration/risk_table.json" "${WORK}/frozen_r1/risk.json"
  cp "${WORK}/cold_calibration/risk_table.json.slack.json" "${WORK}/frozen_r1/risk.json.slack.json"
  run_guarded frozen_r1 static frozen 0 "${LOAD_S}"
else
  RESULTS="${RESULTS} frozen_r1:no_cold_table"
fi
run_guarded baseline_r1 baseline baseline 0 "${LOAD_S}"
"${PY}" "${JOB_DIR}/summarize_recal.py" "${OUT}" "${SEGMENT_S}" || echo "summary failed"
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *:[1-9]*|*skipped*|*failed*|*no_cold*|*unfinished*) exit 1;; esac
exit 0
