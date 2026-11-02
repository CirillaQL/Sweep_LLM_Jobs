#!/usr/bin/env bash
#SBATCH --job-name=voltana-e2e
#SBATCH --partition=long
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261102-215321-luqia-voltana-e2e-1p1d-l40s-l4/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261102-215321-luqia-voltana-e2e-1p1d-l40s-l4/slurm-%j.err

# VoltanaLLM v0.1 (SGLang 0.4.7.post1) end to end, 1P1D: prefill on one L40S (neptune), decode on
# one L4 (ganymede), Mistral-7B-v0.1, real KV transfer over NIXL (UCX over TCP, 100 GbE, no RDMA).
# EcoFreq reads the EcoPred tables of 280101 (prefill-L40S) and 280104 (decode-L4). EcoRoute has
# nothing to route with one decode instance, so the router is SGLang's mini_lb (round robin).
# One server launch; the driver switches each configuration at run time over HTTP:
#   static_max      /set_freq P 1800, D 2040 (EcoFreq in dummy mode)   paper: SGLang max clock
#   static_low      /set_freq P 1080, D 1020                           paper: SGLang "sweet" clock
#   voltana_itl100  EcoFreq on, P TTFT budget 350 ms, D ITL 100 ms, f lists {1080,1800} / {1020,2040}
#   voltana_itl150  the same with D ITL 150 ms
# Workload: ShareGPT (context <= 3000 tokens, dataset output lengths, ignore EOS: the client's
# default, as in the paper's scripts), Poisson, 180 s of arrivals per rate, rates 1-6 req/s, same
# seed in every configuration. A warm-up smoke stage (20 requests) runs first and must succeed.
# SLOs (see README): end-to-end TTFT 600 ms (paper, LLaMA-3.1-8B); ITL 60 ms is below the L4's
# floor (~57 ms at bs 1), so ITL 100 / 150 ms. EcoFreq's prefill budget is 600 ms minus ~250 ms
# reserved for the TCP KV transfer and D-side scheduling (its own offset only covers P queueing).
# Margin 0 (the code multiplies the SLO by 1 + margin, i.e. a margin loosens it).
# voltana-e2e.patch is applied to a job-local copy of the sglang package (the shared checkout is
# untouched): EcoFreq's setter locks clocks with `sudo -n nvidia-smi` (the cluster does not allow
# the original root python) and NVML uses the physical GPU index (no device cgroups here).
# Energy: our own NVML sampler per GPU (0.2 s, cumulative counter) over each stage's benchmark span.
set -uo pipefail
T0=$(date +%s)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261102-215321-luqia-voltana-e2e-1p1d-l40s-l4"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
VOLTANA_ROOT="/data/users/chjing/voltana_work"
WORK="${VOLTANA_ROOT}/runs/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/voltanallm"
PY="${ENV_DIR}/bin/python"
P_NODE=neptune
D_NODE=ganymede
WALL_S=21600
RESERVE_S=600

# ---- experiment ---------------------------------------------------------------------------
P_FLIST="1080 1800"; D_FLIST="1020 2040"          # {low, max}: two levels as in the paper's main runs
P_SLO_MS=350                                      # EcoFreq prefill budget = 600 - 250 (KV over TCP)
D_SLO_ITL100=100; D_SLO_ITL150=150
CONFIGS="static_max static_low voltana_itl100 voltana_itl150"
RATES="1 2 3 4 5 6"
STAGE_S=180; GAP_S=30; STAGE_TIMEOUT_S=900; BENCH_SEED=20261002
P_TABLE_SRC="${BROKER_ROOT}/jobs/20261026-141000-luqia-voltana-prof-prefill-l40s/results/280101/latency-prefill.csv"
D_TABLE_SRC="${BROKER_ROOT}/jobs/20261026-141030-luqia-voltana-prof-decode-l4/results/280104/latency-decode.csv"
MODEL_PATH="${VOLTANA_ROOT}/models/Mistral-7B-v0.1"
SHAREGPT="/data/users/chjing/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

BASE=$(( 30000 + ${JOB_ID//[^0-9]/0} % 10000 ))
P_PORT=${BASE}; D_PORT=$(( BASE + 1 )); BOOT_PORT=$(( BASE + 2 )); LB_PORT=$(( BASE + 3 ))

mkdir -p "${OUT}" "${WORK}/bench" "${WORK}/tables"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "job_start=$(date -Iseconds) host=$(hostname) job=${JOB_ID}"
fail() { echo "RESULT=failed stage=$1"; exit 1; }

grep -q "env-setup-v6" "${VOLTANA_ROOT}/ENV_READY" 2>/dev/null || fail "env_not_ready (needs 20261029-105703-luqia-voltana-env-setup-v6)"
export PATH="${ENV_DIR}/bin:${PATH}" CUDA_HOME="${ENV_DIR}"
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
[ -e "${MODEL_PATH}/config.json" ] || fail "model_link_missing ${MODEL_PATH}"
[ -s "${SHAREGPT}" ] || fail "sharegpt_missing ${SHAREGPT}"

# ---- 1. job-local sglang with voltana-e2e.patch -------------------------------------------
mkdir -p "${WORK}/overlay/python"
cp -a "${VOLTANA_ROOT}/VoltanaLLM/python/sglang" "${WORK}/overlay/python/" || fail overlay_copy
patch -p1 -d "${WORK}/overlay" < "${JOB_DIR}/voltana-e2e.patch" > "${OUT}/patch.txt" 2>&1 || { cat "${OUT}/patch.txt"; fail overlay_patch; }
export PYTHONPATH="${WORK}/overlay/python"
IMPORTED=$("${PY}" -c 'import sglang.srt.managers.freq_manager as m; print(m.__file__)' 2>&1 | tail -n 1)
echo "freq_manager=${IMPORTED}" | tee "${OUT}/overlay.txt"
[[ "${IMPORTED}" == "${WORK}/overlay/python/"* ]] || fail "overlay_not_imported (${IMPORTED})"

# ---- 2. EcoPred tables ---------------------------------------------------------------------
cp "${P_TABLE_SRC}" "${WORK}/tables/latency-prefill-l40s.csv" || fail prefill_table_missing
cp "${D_TABLE_SRC}" "${WORK}/tables/latency-decode-l4.csv" || fail decode_table_missing
P_TABLE="${WORK}/tables/latency-prefill-l40s.csv"; D_TABLE="${WORK}/tables/latency-decode-l4.csv"
mkdir -p "${OUT}/tables" && cp "${WORK}/tables/"*.csv "${OUT}/tables/"
for pair in "${P_TABLE}:${P_FLIST}" "${D_TABLE}:${D_FLIST}"; do
  table=${pair%%:*}
  for f in ${pair#*:}; do
    cut -d, -f1 "${table}" | grep -qx "${f}" || fail "freq_${f}_not_in_$(basename "${table}")"
  done
done

# ---- 3. node IPs, servers ------------------------------------------------------------------
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP=$(ip_of "${D_NODE}")
[[ "${P_IP}" =~ ^[0-9.]+$ && "${D_IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
echo "ips ${P_NODE}=${P_IP} ${D_NODE}=${D_IP} ports p=${P_PORT} d=${D_PORT} boot=${BOOT_PORT} lb=${LB_PORT}" | tee "${OUT}/topology.txt"
export WORK JOB_DIR PY MODEL_PATH P_PORT D_PORT BOOT_PORT P_TABLE D_TABLE P_SLO_MS P_FLIST D_FLIST
export D_SLO_MS="${D_SLO_ITL100}"

P_PID=""; D_PID=""; LB_PID=""; BENCH_PID=""
post() { curl -sf -m 10 -X POST -H 'Content-Type: application/json' -d "$2" "$1" > /dev/null; }
reset_clock() {  # $1 node, $2 role
  local idx; idx=$(cat "${WORK}/$2_gpu_index" 2>/dev/null) || return 0
  timeout 120 srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M bash -c \
    "sudo -n nvidia-smi -i ${idx} -rgc; sleep 3; nvidia-smi -i ${idx} --query-gpu=index,uuid,clocks.sm --format=csv,noheader" \
    2>&1 | sed "s/^/[$1] /"
}
cleanup() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  [ -z "${BENCH_PID}" ] || kill "${BENCH_PID}" 2>/dev/null
  post"http://${P_IP}:${P_PORT}/unset_freq" '{}'; post "http://${D_IP}:${D_PORT}/unset_freq" '{}'
  sleep 3
  for pid in ${LB_PID} ${P_PID} ${D_PID}; do kill "${pid}" 2>/dev/null; done
  sleep 15
  for pid in ${LB_PID} ${P_PID} ${D_PID}; do kill -9 "${pid}" 2>/dev/null; done
  { reset_clock "${P_NODE}" prefill; reset_clock "${D_NODE}" decode; } | tee "${OUT}/clock_reset.txt"
  for node in "${P_NODE}" "${D_NODE}"; do
    timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
      'ps -o pid,etime,args -u "$(id -un)" | grep -E "sglang|gpu_sampler" | grep -v grep || echo none' \
      2>&1 | sed "s/^/[${node} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  done
  if [ -s "${WORK}/stages.csv" ]; then
    "${PY}" "${JOB_DIR}/analyze_e2e.py" --work "${WORK}" --out "${OUT}" > "${OUT}/analyze.log" 2>&1
    echo "analyze_rc=$?"; cat "${OUT}/summary.txt" 2>/dev/null; cat "${OUT}/transfer_fit.txt" 2>/dev/null
    cp "${WORK}/stages.csv" "${OUT}/"
  fi
  for role in prefill decode; do
    [ ! -s "${WORK}/${role}_gpu.txt" ] || cp "${WORK}/${role}_gpu.txt" "${OUT}/"
    [ ! -s "${WORK}/${role}_nvml.csv" ] || gzip -c "${WORK}/${role}_nvml.csv" > "${OUT}/${role}_nvml.csv.gz"
    [ ! -s "${WORK}/${role}.log" ] || gzip -c "${WORK}/${role}.log" > "${OUT}/${role}.log.gz"
  done
  [ ! -s "${WORK}/router.log" ] || tail -n 300 "${WORK}/router.log" > "${OUT}/router_tail.log"
  mkdir -p "${OUT}/bench"; cp "${WORK}/bench/"*.jsonl "${OUT}/bench/" 2>/dev/null
  echo "${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap cleanup EXIT
trap 'exit 143' TERM

srun --overlap --nodes=1 --ntasks=1 --nodelist="${P_NODE}" --gpus-per-node=1 --cpus-per-task=8 --mem=48G \
  bash "${JOB_DIR}/node_server.sh" prefill > "${WORK}/prefill.log" 2>&1 &
P_PID=$!
srun --overlap --nodes=1 --ntasks=1 --nodelist="${D_NODE}" --gpus-per-node=1 --cpus-per-task=8 --mem=48G \
  bash "${JOB_DIR}/node_server.sh" decode > "${WORK}/decode.log" 2>&1 &
D_PID=$!
wait_health() {  # $1 url, $2 pid, $3 name
  for _ in $(seq 1 240); do
    curl -sf -m 5 "$1/health" > /dev/null && { echo "$3_ready_after_s=$(( $(date +%s) - T0 ))"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "$3 exited:"; tail -n 40 "${WORK}/$3.log"; return 1; }
    sleep 5
  done
  tail -n 40 "${WORK}/$3.log"; return 1
}
wait_health "http://${P_IP}:${P_PORT}" "${P_PID}" prefill || fail prefill_not_ready
wait_health "http://${D_IP}:${D_PORT}" "${D_PID}" decode || fail decode_not_ready
cat "${WORK}/prefill_gpu.txt" "${WORK}/decode_gpu.txt" | tee "${OUT}/gpu_allocation.txt"
curl -sf "http://${P_IP}:${P_PORT}/get_server_info" > "${OUT}/server_info_prefill.json"
curl -sf "http://${D_IP}:${D_PORT}/get_server_info" > "${OUT}/server_info_decode.json"

"${PY}" -m sglang.srt.disaggregation.launch_lb --prefill "http://${P_IP}:${P_PORT}" \
  --prefill-bootstrap-ports "${BOOT_PORT}" --decode "http://${D_IP}:${D_PORT}" \
  --host 127.0.0.1 --port "${LB_PORT}" > "${WORK}/router.log" 2>&1 &
LB_PID=$!
wait_health "http://127.0.0.1:${LB_PORT}" "${LB_PID}" router || fail router_not_ready

# ---- 4. one request through P -> NIXL -> D --------------------------------------------------
curl -sf -m 300 -X POST -H 'Content-Type: application/json' "http://127.0.0.1:${LB_PORT}/generate" \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}' \
  > "${OUT}/smoke_generate.json" 2>&1
cat "${OUT}/smoke_generate.json"; echo
grep -q '"text"' "${OUT}/smoke_generate.json" || fail "pd_generate (P->NIXL->D request failed; see prefill/decode logs)"

# ---- 5. stages -----------------------------------------------------------------------------
echo "stage,config,rate,num_prompts,t_start,t_end,rc" > "${WORK}/stages.csv"
apply_config() {
  case "$1" in
    static_max) post "http://${P_IP}:${P_PORT}/set_freq" "{\"freq\": ${P_FLIST##* }}" &&
                post "http://${D_IP}:${D_PORT}/set_freq" "{\"freq\": ${D_FLIST##* }}" ;;
    static_low) post "http://${P_IP}:${P_PORT}/set_freq" "{\"freq\": ${P_FLIST%% *}}" &&
                post "http://${D_IP}:${D_PORT}/set_freq" "{\"freq\": ${D_FLIST%% *}}" ;;
    voltana_itl*) local itl="${1#voltana_itl}"
                post "http://${P_IP}:${P_PORT}/set_freq_manager_state" "{\"dummy_run\": false, \"slo\": ${P_SLO_MS}}" &&
                post "http://${D_IP}:${D_PORT}/set_freq_manager_state" "{\"dummy_run\": false, \"slo\": ${itl}}" ;;
    *) return 1 ;;
  esac
}
run_stage() {  # $1 stage, $2 config, $3 rate, $4 prompts
  local t_start t_end rc
  rm -f "${WORK}/bench/$1.jsonl" "${WORK}/bench/$1.jsonl-detail"
  t_start=$(date +%s.%N)
  # background + wait, so the TERM before the time limit reaches the trap at once
  timeout "${STAGE_TIMEOUT_S}" "${PY}" -m sglang.bench_serving_voltanallm --backend sglang \
    --host 127.0.0.1 --port "${LB_PORT}" --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" \
    --dataset-name sharegpt --dataset-path "${SHAREGPT}" --sharegpt-context-len 3000 \
    --num-prompts "$4" --request-rate "$3" --seed "${BENCH_SEED}" --warmup-requests 0 \
    --pd-separated --disable-tqdm --output-file "${WORK}/bench/$1.jsonl" --output-details \
    > "${WORK}/bench/$1.log" 2>&1 &
  BENCH_PID=$!
  wait "${BENCH_PID}"
  rc=$?
  BENCH_PID=""
  t_end=$(date +%s.%N)
  echo "$1,$2,$3,$4,${t_start},${t_end},${rc}" >> "${WORK}/stages.csv"
  echo "stage $1 rc=${rc} s=$(awk -v a="${t_start}" -v b="${t_end}" 'BEGIN {printf "%.0f", b - a}') $(grep -E 'Successful requests|Mean TTFT|Mean ITL' "${WORK}/bench/$1.log" | tr -s ' ' | tr '\n' ' ')"
  return "${rc}"
}
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }

apply_config static_max || fail apply_static_max
sleep 5
run_stage smoke smoke 1 20 || fail "smoke_bench rc=$?"
"${PY}" -c '
import json, sys
errors = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["errors"]
bad = sum(1 for e in errors if e)
print("smoke: %d/%d ok" % (len(errors) - bad, len(errors)))
sys.exit(1 if bad > len(errors) // 2 else 0)' "${WORK}/bench/smoke.jsonl-detail" || fail smoke_requests_failed
sleep "${GAP_S}"

STATUS=ok
for cfg in ${CONFIGS}; do
  apply_config "${cfg}" || { echo "apply ${cfg} failed"; STATUS=partial; continue; }
  echo "=== ${cfg} $(date -Iseconds)"
  sleep 5
  for rate in ${RATES}; do
    if [ "$(left)" -le $(( STAGE_TIMEOUT_S + GAP_S )) ]; then echo "skip ${cfg}_r${rate}: wall time"; STATUS=partial; continue; fi
    run_stage "${cfg}_r${rate}" "${cfg}" "${rate}" $(( rate * STAGE_S )) || STATUS=partial
    kill -0 "${P_PID}" 2>/dev/null && kill -0 "${D_PID}" 2>/dev/null || fail "server_died_in_${cfg}_r${rate}"
    sleep "${GAP_S}"
  done
done
echo "RESULT=${STATUS}"
[ "${STATUS}" = ok ]
