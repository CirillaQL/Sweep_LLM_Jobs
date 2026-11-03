#!/usr/bin/env bash
#SBATCH --job-name=voltana-4p4d
#SBATCH --partition=long
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=uranus,europa
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --signal=B:TERM@300
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261103-154838-luqia-voltana-e2e-4p4d-uranus-europa-r2/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261103-154838-luqia-voltana-e2e-4p4d-uranus-europa-r2/slurm-%j.err

# VoltanaLLM v0.1 (SGLang 0.4.7.post1) end to end, 4P4D, as the baseline to CanaTune's main run
# (20261013-210335-luqia-canatune-fill-redirect-4groups-x3-slo600-100, Slurm 275725, neptune /
# ganymede) on uranus / europa: the same GPUs (4 x L40S prefill on uranus, 4 x L4 decode on europa,
# both exclusive; the nodes of CanaTune's 275830, whose baseline_r1 is the same-node reference), model
# (Mistral-7B-v0.1), client (`vllm bench serve` of the cuda-env, the same arguments), workloads
# (ShareGPT capped at 256 output tokens with EOS stops, BurstGPT with the trace's output lengths;
# 0.5-8 req/s, 180 s per rate, 30 s gap, seed 20261002), scoring (TTFT and vLLM's TPOT per
# request, goodput = good / offered) and energy (all 8 GPUs over each benchmark's own span).
# KV moves over NIXL (UCX over TCP, 100 GbE, no RDMA); CanaTune uses vLLM's P2pNccl.
# One server launch (8 servers + VoltanaLLM's router); configurations switch at run time:
#   rr_default       round robin, clocks left to the driver           CanaTune's baseline
#   rr_max           round robin, P 1800 / D 2040 locked                paper: SGLang at max clock
#   voltana_s600     EcoFreq + EcoRoute (tile_aware), TTFT SLO 600      VoltanaLLM
#   voltana_s1000    the same with TTFT SLO 1000                        TCP-adjusted SLO
#   voltana_s600_rr  EcoFreq with round robin (EcoRoute ablation; only if time is left)
# EcoFreq: f lists {1080, 1800} (L40S) / {1020, 2040} (L4), two levels as in the paper's main runs;
# prefill budget = TTFT SLO - 250 ms for the TCP KV transfer (its own offset covers P queueing
# only); decode ITL SLO 100 ms (= CanaTune's TPOT SLO); margin 0. Every stage is scored at
# 600/100, 1000/100, 600/150 and 1000/150 (TTFT/TPOT ms).
# voltana-4p4d.patch is applied to a job-local copy of the sglang package (the shared checkout is
# untouched): the setter locks clocks with `sudo -n nvidia-smi`, NVML uses the physical GPU index,
# the router's server-state loop actually starts (it never did with run_echo_lb), the router's
# tokenizer is the local model, and prefill servers are chosen round robin (the paper's baselines;
# the code picks at random).
# r2 of 20261103-145017-luqia-voltana-e2e-4p4d-uranus-europa (280148: every streamed completion
# failed): the patch also fixes VoltanaLLM's streamed /v1/completions (adapter.py reads
# stream_options.include_prompt, a field StreamOptions does not have; vllm bench always sends
# stream_options) and the router's state refresh (the ITL table lookup asserted for clocks outside
# the table's range, which dropped that round's update of all servers).
set -uo pipefail
T0=$(date +%s)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261103-154838-luqia-voltana-e2e-4p4d-uranus-europa-r2"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
VOLTANA_ROOT="/data/users/chjing/voltana_work"
WORK="${VOLTANA_ROOT}/runs/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/voltanallm"
PY="${ENV_DIR}/bin/python"
CT_ENV="/data/users/chjing/miniforge3/envs/cuda-env"   # CanaTune's env: its `vllm bench serve`
P_NODE=uranus
D_NODE=europa
WALL_S=36000
RESERVE_S=900   # teardown, analysis, result copy and clock reset before the wall limit

# ---- experiment ---------------------------------------------------------------------------
P_FLIST="1080 1800"; D_FLIST="1020 2040"   # {low, high}: two levels as in the paper's main runs
TRANSFER_MS=250                            # KV over TCP + D-side scheduling (r2: ~138 ms + 0.65 ms/token)
D_SLO_MS=100                               # decode ITL SLO = CanaTune's TPOT SLO
CONFIGS="rr_default rr_max voltana_s600 voltana_s1000 voltana_s600_rr"
WORKLOADS="sharegpt burstgpt"
RATES="0.5 1 2 4 6 8"
STAGE_S=180; GAP_S=30; BENCH_SEED=20261002
STAGE_TIMEOUT_S=$(( 3 * STAGE_S + 600 ))   # as bench_production.py
IDLE_TIMEOUT_S=900
SCORE_SLOS="600:100,1000:100,600:150,1000:150"
P_TABLE_SRC="${BROKER_ROOT}/jobs/20261026-141000-luqia-voltana-prof-prefill-l40s/results/280101/latency-prefill.csv"
D_TABLE_SRC="${BROKER_ROOT}/jobs/20261026-141030-luqia-voltana-prof-decode-l4/results/280104/latency-decode.csv"
MODEL_PATH="${VOLTANA_ROOT}/models/Mistral-7B-v0.1"
DATA_DIR="/data/users/chjing/datasets"
SHAREGPT="${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
BURSTGPT_CSV="${DATA_DIR}/BurstGPT_1.csv"
CT_REF_JOB="${BROKER_ROOT}/jobs/20261013-210335-luqia-canatune-fill-redirect-4groups-x3-slo600-100/results/275725"
# same nodes as this job: baseline_r1 (round robin, driver clocks) and one CanaTune run at SLO 1000 / 150
CT_NODE_REF_JOB="${BROKER_ROOT}/jobs/20261014-131315-luqia-canatune-4groups-slo1000-150-uranus-europa/results/275830"
CT_BURSTGPT="/data/users/chjing/vllm_job_work/275725/burstgpt.jsonl"   # the exact file 275725 sent
CT_NODE_BURSTGPT="/data/users/chjing/vllm_job_work/275830/burstgpt.jsonl"
CT_REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
CT_COMMIT="13b605249c0929c8fb9444489074c905f1360c5d"                   # 275725's commit
CT_SNAP_ROOT="/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots"

BASE=$(( 30000 + ${JOB_ID//[^0-9]/0} % 10000 ))
P_PORTS="$(( BASE )) $(( BASE + 1 )) $(( BASE + 2 )) $(( BASE + 3 ))"
D_PORTS="$(( BASE + 10 )) $(( BASE + 11 )) $(( BASE + 12 )) $(( BASE + 13 ))"
BOOT_PORTS="$(( BASE + 20 )) $(( BASE + 21 )) $(( BASE + 22 )) $(( BASE + 23 ))"
LB_PORT=$(( BASE + 30 ))

mkdir -p "${OUT}" "${WORK}/logs" "${WORK}/tables" "${WORK}/data"
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
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache" PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0
export FLASHINFER_WORKSPACE_BASE="${VOLTANA_ROOT}/cache/flashinfer"
export TRITON_CACHE_DIR="${VOLTANA_ROOT}/cache/triton"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"
[ -e "${MODEL_PATH}/config.json" ] || fail "model_link_missing ${MODEL_PATH}"
[ -s "${SHAREGPT}" ] || fail "sharegpt_missing ${SHAREGPT}"

# ---- 1. job-local sglang with voltana-4p4d.patch -------------------------------------------
mkdir -p "${WORK}/overlay/python"
cp -a "${VOLTANA_ROOT}/VoltanaLLM/python/sglang" "${WORK}/overlay/python/" || fail overlay_copy
patch -p1 -d "${WORK}/overlay" < "${JOB_DIR}/voltana-4p4d.patch" > "${OUT}/patch.txt" 2>&1 || { cat "${OUT}/patch.txt"; fail overlay_patch; }
# mini_lb / echo_lb_new import aiofiles (not in the env; 280125 failed on it): job-local, env untouched
timeout 300 "${PY}" -m pip install --quiet --no-deps --target "${WORK}/pylibs" "aiofiles==24.1.0" || fail pip_aiofiles
export PYTHONPATH="${WORK}/overlay/python:${WORK}/pylibs"
IMPORTED=$("${PY}" -c 'import sglang.srt.managers.freq_manager as m; print(m.__file__)' 2>&1 | tail -n 1)
echo "freq_manager=${IMPORTED}" | tee "${OUT}/overlay.txt"
[[ "${IMPORTED}" == "${WORK}/overlay/python/"* ]] || fail "overlay_not_imported (${IMPORTED})"
# everything the router imports must load before the servers take minutes to start
"${PY}" -c 'import sglang.srt.disaggregation.launch_lb, sglang.srt.disaggregation.mini_lb, sglang.srt.disaggregation.echo_lb_new, uvloop' \
  >> "${OUT}/overlay.txt" 2>&1 || { tail -n 5 "${OUT}/overlay.txt"; fail router_imports; }

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

# ---- 3. client and workloads exactly as CanaTune's 275725 -----------------------------------
VLLM_BENCH=(env -u PYTHONPATH -u PYTHONNOUSERSITE "${CT_ENV}/bin/vllm" bench serve)
env -u PYTHONPATH "${CT_ENV}/bin/python" -c 'import vllm; print("client vllm", vllm.__version__)' \
  | tee "${OUT}/client.txt" || fail client_vllm_missing
# CanaTune's tokenizer is its HF snapshot; the servers load MODEL_PATH: they must be the same files
CT_SNAP=""
while IFS= read -r snap; do
  if [ -s "${snap}/config.json" ] && [ -s "${snap}/tokenizer_config.json" ]; then CT_SNAP="${snap}"; break; fi
done < <(find "${CT_SNAP_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ -n "${CT_SNAP}" ] || fail canatune_snapshot_missing
for f in config.json tokenizer.json tokenizer_config.json tokenizer.model; do
  [ -e "${CT_SNAP}/${f}" ] || continue
  a=$(md5sum < "${CT_SNAP}/${f}"); b=$(md5sum < "${MODEL_PATH}/${f}" 2>/dev/null)
  echo "model_file ${f} canatune=${a%% *} voltana=${b%% *}" | tee -a "${OUT}/client.txt"
  [ "${a}" = "${b}" ] || fail "model_files_differ ${f}"
done
BENCH_TOKENIZER="${CT_SNAP}"
# BurstGPT: the very file 275725 sent; regenerated at its commit only if it is gone
BURSTGPT="${WORK}/data/burstgpt.jsonl"
if [ -s "${CT_BURSTGPT}" ]; then
  cp "${CT_BURSTGPT}" "${BURSTGPT}" || fail burstgpt_copy
  echo "burstgpt=copied from ${CT_BURSTGPT}" | tee -a "${OUT}/client.txt"
else
  [ -s "${BURSTGPT_CSV}" ] || fail "burstgpt_csv_missing ${BURSTGPT_CSV}"
  timeout 300 git clone --quiet "${CT_REPO}" "${WORK}/CanaTune" || fail canatune_clone
  git -C "${WORK}/CanaTune" checkout --quiet "${CT_COMMIT}" || fail canatune_checkout
  ( cd "${WORK}/CanaTune" && timeout 900 env -u PYTHONPATH "${CT_ENV}/bin/python" scripts/make_burstgpt_dataset.py \
      --csv "${BURSTGPT_CSV}" --sharegpt "${SHAREGPT}" --tokenizer "${CT_SNAP}" --out "${BURSTGPT}" \
      --model all --rows 4000 --max-len 4096 ) > "${OUT}/burstgpt_make.log" 2>&1 || fail burstgpt_dataset
  echo "burstgpt=regenerated at ${CT_COMMIT} (275725's file is gone)" | tee -a "${OUT}/client.txt"
fi
echo "burstgpt_sha256=$(sha256sum < "${BURSTGPT}" | cut -d' ' -f1) lines=$(wc -l < "${BURSTGPT}")" | tee -a "${OUT}/client.txt"
# 275830 regenerated its own file with the same script and arguments: it should be identical
[ -s "${CT_NODE_BURSTGPT}" ] && echo "burstgpt_275830_sha256=$(sha256sum < "${CT_NODE_BURSTGPT}" | cut -d' ' -f1)" | tee -a "${OUT}/client.txt"
[ -s "${CT_REF_JOB}/baseline_r1/stages/sharegpt_r0_0.5rps/command.txt" ] && \
  cp "${CT_REF_JOB}/baseline_r1/stages/sharegpt_r0_0.5rps/command.txt" "${OUT}/canatune_command_example.txt"

# ---- 4. node IPs, servers ------------------------------------------------------------------
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP=$(ip_of "${D_NODE}")
[[ "${P_IP}" =~ ^[0-9.]+$ && "${D_IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
read -r -a P_PORT_ARR <<< "${P_PORTS}"; read -r -a D_PORT_ARR <<< "${D_PORTS}"
P_URLS=(); D_URLS=()
for i in 0 1 2 3; do P_URLS+=("http://${P_IP}:${P_PORT_ARR[$i]}"); D_URLS+=("http://${D_IP}:${D_PORT_ARR[$i]}"); done
ALL_URLS=("${P_URLS[@]}" "${D_URLS[@]}")
{ echo "ips ${P_NODE}=${P_IP} ${D_NODE}=${D_IP} lb=${LB_PORT}"
  echo "prefill ${P_URLS[*]} bootstrap ${BOOT_PORTS}"; echo "decode ${D_URLS[*]}"; } | tee "${OUT}/topology.txt"
export WORK JOB_DIR PY MODEL_PATH P_PORTS D_PORTS BOOT_PORTS P_TABLE D_TABLE P_FLIST D_FLIST D_SLO_MS
export P_SLO_MS=$(( 600 - TRANSFER_MS ))

P_STEP=""; D_STEP=""; LB_PID=""; BENCH_PID=""; NET_PIDS=""
post() { curl -sf -m 10 -X POST -H 'Content-Type: application/json' -d "$2" "$1" > /dev/null; }
reset_clocks() {  # $1 node, $2 role
  local ids; ids=$(cat "${WORK}/$2_gpu_indices" 2>/dev/null) || return 0
  timeout 120 srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M bash -c '
    IFS=, read -r -a ids <<< "$1"
    for i in "${ids[@]}"; do
      sudo -n nvidia-smi -i "$i" -rgc > /dev/null 2>&1 && echo "reset $i ok" || echo "reset $i FAILED"
    done
    sleep 3
    nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks_event_reasons.active --format=csv,noheader' _ "${ids}" \
    2>&1 | sed "s/^/[$1] /"
}
cleanup() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  [ -z "${BENCH_PID}" ] || kill "${BENCH_PID}" 2>/dev/null
  post "http://127.0.0.1:${LB_PORT}/end_benchmark" '{}'
  for url in "${ALL_URLS[@]}"; do post "${url}/unset_freq" '{}'; done
  sleep 3
  for pid in ${LB_PID} ${P_STEP} ${D_STEP}; do kill "${pid}" 2>/dev/null; done
  sleep 20
  for pid in ${LB_PID} ${P_STEP} ${D_STEP} ${NET_PIDS}; do kill -9 "${pid}" 2>/dev/null; done
  { reset_clocks "${P_NODE}" prefill; reset_clocks "${D_NODE}" decode; } | tee "${OUT}/clock_reset.txt"
  for node in "${P_NODE}" "${D_NODE}"; do
    # only this job's processes (the shared account also runs other jobs)
    timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
      "ps -o pid,etime,args -u \"\$(id -un)\" | grep -E 'runs/${JOB_ID}/' | grep -v grep || echo none" \
      2>&1 | sed "s/^/[${node} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  done
  if [ -s "${WORK}/stages.csv" ]; then
    cp "${WORK}/stages.csv" "${OUT}/"
    "${PY}" "${JOB_DIR}/analyze_4p4d.py" --work "${WORK}" --out "${OUT}" --score-slos "${SCORE_SLOS}" \
      --canatune "ct275830=${CT_NODE_REF_JOB}" "ct275725=${CT_REF_JOB}" > "${OUT}/analyze.log" 2>&1
    echo "analyze_rc=$?"; cat "${OUT}/summary.txt" 2>/dev/null
  fi
  # per stage: the client's result and log, the router's server-state log
  for dir in "${WORK}"/*/stages/*/; do
    [ -d "${dir}" ] || continue
    rel=${dir#"${WORK}/"}; mkdir -p "${OUT}/${rel}"
    for f in bench.json router_state.jsonl; do [ -s "${dir}/${f}" ] && gzip -c "${dir}/${f}" > "${OUT}/${rel}/${f}.gz"; done
    for f in command.txt bench.log; do [ -s "${dir}/${f}" ] && cp "${dir}/${f}" "${OUT}/${rel}/"; done
  done
  mkdir -p "${OUT}/logs"
  for f in "${WORK}"/logs/*.log; do
    [ -s "${f}" ] && grep -vE 'GET /(get_server_info|health)' "${f}" | gzip -c > "${OUT}/logs/$(basename "${f}").gz"
  done
  for role in prefill decode; do
    [ ! -s "${WORK}/${role}_gpus.txt" ] || cp "${WORK}/${role}_gpus.txt" "${OUT}/"
    [ ! -s "${WORK}/${role}_nvml.csv" ] || gzip -c "${WORK}/${role}_nvml.csv" > "${OUT}/${role}_nvml.csv.gz"
  done
  for node in "${P_NODE}" "${D_NODE}"; do
    [ ! -s "${WORK}/net_${node}.txt" ] || gzip -c "${WORK}/net_${node}.txt" > "${OUT}/net_${node}.txt.gz"
  done
  echo "${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap cleanup EXIT
trap 'exit 143' TERM

srun --overlap --nodes=1 --ntasks=1 --nodelist="${P_NODE}" --gpus-per-node=4 --cpus-per-task=32 --mem=0 \
  bash "${JOB_DIR}/node_servers.sh" prefill > "${WORK}/logs/node_prefill.log" 2>&1 &
P_STEP=$!
srun --overlap --nodes=1 --ntasks=1 --nodelist="${D_NODE}" --gpus-per-node=4 --cpus-per-task=32 --mem=0 \
  bash "${JOB_DIR}/node_servers.sh" decode > "${WORK}/logs/node_decode.log" 2>&1 &
D_STEP=$!
# NIC bytes of both nodes every 2 s (four P share uranus's NIC), as in 275725
for node in "${P_NODE}" "${D_NODE}"; do
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=128M bash -c \
    'while true; do echo "t=$(date +%s.%N)"; cat /proc/net/dev; sleep 2; done' > "${WORK}/net_${node}.txt" 2>&1 &
  NET_PIDS="${NET_PIDS} $!"
done
wait_health() {  # $1 url, $2 pid of the step or process, $3 log
  for _ in $(seq 1 240); do
    curl -sf -m 5 "$1/health" > /dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "$1: exited"; tail -n 40 "$3"; return 1; }
    sleep 5
  done
  echo "$1: not ready"; tail -n 40 "$3"; return 1
}
for i in 0 1 2 3; do
  wait_health "${P_URLS[$i]}" "${P_STEP}" "${WORK}/logs/prefill${i}.log" || { tail -n 20 "${WORK}/logs/node_prefill.log"; fail "prefill${i}_not_ready"; }
  wait_health "${D_URLS[$i]}" "${D_STEP}" "${WORK}/logs/decode${i}.log" || { tail -n 20 "${WORK}/logs/node_decode.log"; fail "decode${i}_not_ready"; }
done
echo "servers_ready_after_s=$(( $(date +%s) - T0 ))"
cat "${WORK}/prefill_gpus.txt" "${WORK}/decode_gpus.txt" | tee "${OUT}/gpu_allocation.txt"
curl -sf "${P_URLS[0]}/get_server_info" > "${OUT}/server_info_prefill0.json"
curl -sf "${D_URLS[0]}/get_server_info" > "${OUT}/server_info_decode0.json"

read -r -a boot_ports <<< "${BOOT_PORTS}"
VOLTANA_ROUTER_TOKENIZER="${MODEL_PATH}" VOLTANA_PREFILL_POLICY=rr TILE_SIZE=128 \
  "${PY}" -m sglang.srt.disaggregation.launch_lb --echo-lb --policy rr \
  --prefill "${P_URLS[@]}" --prefill-bootstrap-ports "${boot_ports[@]}" --decode "${D_URLS[@]}" \
  --host 127.0.0.1 --port "${LB_PORT}" --latency-lookuptable-path "${D_TABLE}" \
  --echo-refresh-server-info-interval 0.05 --echo-log-path "${WORK}/logs/router_echo.log" \
  --echo-log-server-info-path "${WORK}/router_state_init.jsonl" > "${WORK}/logs/router.log" 2>&1 &
LB_PID=$!
wait_health "http://127.0.0.1:${LB_PORT}" "${LB_PID}" "${WORK}/logs/router.log" || fail router_not_ready

# ---- 5. stages -----------------------------------------------------------------------------
echo "config,workload,stage,rate,num_prompts,policy,t_launch,t_end,rc,idle_s" > "${WORK}/stages.csv"
policy_of() { case "$1" in voltana_s600|voltana_s1000) echo tile_aware ;; *) echo rr ;; esac; }
apply_config() {  # one control message per server (EcoFreq keeps only the latest one queued)
  local url ok=0 p_slo
  case "$1" in
    rr_default) for url in "${ALL_URLS[@]}"; do post "${url}/unset_freq" '{}' || ok=1; done ;;
    rr_max) for url in "${P_URLS[@]}"; do post "${url}/set_freq" "{\"freq\": ${P_FLIST##* }}" || ok=1; done
            for url in "${D_URLS[@]}"; do post "${url}/set_freq" "{\"freq\": ${D_FLIST##* }}" || ok=1; done ;;
    voltana_s600|voltana_s600_rr|voltana_s1000)
      p_slo=600; [ "$1" = voltana_s1000 ] && p_slo=1000
      for url in "${P_URLS[@]}"; do
        post "${url}/set_freq_manager_state" "{\"dummy_run\": false, \"slo\": $(( p_slo - TRANSFER_MS ))}" || ok=1
      done
      for url in "${D_URLS[@]}"; do post "${url}/set_freq_manager_state" "{\"dummy_run\": false, \"slo\": ${D_SLO_MS}}" || ok=1; done ;;
    *) return 1 ;;
  esac
  return "${ok}"
}
busy() {  # requests still on the servers (running, queued, in KV transfer)
  "${PY}" - "${ALL_URLS[@]}" <<'EOF'
import json, sys, urllib.request
keys = ("bs", "len_waiting_queue", "len_prefill_bootstrap_queue", "len_prefill_inflight_queue",
        "len_decode_prealloc_queue", "len_decode_transfer_queue", "len_decode_retracted_queue")
total = 0
for url in sys.argv[1:]:
    try:
        state = json.load(urllib.request.urlopen(url + "/get_server_info", timeout=10))["internal_states"][0]
        total += sum(int(state.get(k) or 0) for k in keys)
    except Exception:
        total += 1
print(total)
EOF
}
wait_idle() {  # -> seconds waited; 1 if the servers did not empty within IDLE_TIMEOUT_S
  local t=0
  while [ "$(busy)" != 0 ]; do
    [ "${t}" -ge "${IDLE_TIMEOUT_S}" ] && { echo "${t}"; return 1; }
    sleep 5; t=$(( t + 5 ))
  done
  echo "${t}"
}
bench_args() {  # $1 workload, $2 rate, $3 prompts, $4 result dir: bench_production.py's command
  local slo_ttft=600 slo_tpot=100
  echo --backend openai --base-url "http://127.0.0.1:${LB_PORT}" --endpoint /v1/completions \
    --model "${MODEL_PATH}" --tokenizer "${BENCH_TOKENIZER}" \
    --num-prompts "$3" --request-rate "$2" --burstiness 1 --seed "${BENCH_SEED}" --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
    --goodput "ttft:${slo_ttft}" "tpot:${slo_tpot}" --save-result --save-detailed \
    --result-dir "$4" --result-filename bench.json --disable-tqdm
  case "$1" in
    sharegpt|smoke) echo --dataset-name sharegpt --dataset-path "${SHAREGPT}" --sharegpt-output-len 256 ;;
    burstgpt) echo --dataset-name custom --dataset-path "${BURSTGPT}" --custom-output-len -1 \
                --skip-chat-template --disable-shuffle --ignore-eos ;;
  esac
}
run_stage() {  # $1 config, $2 workload, $3 stage, $4 rate, $5 prompts -> client exit code
  local dir="${WORK}/$1/stages/$3" policy t_launch t_end rc idle args
  policy=$(policy_of "$1")
  mkdir -p "${dir}"; rm -f "${dir}/bench.json"
  # paths without spaces: word splitting of bench_args is safe
  read -r -a args <<< "$(bench_args "$2" "$4" "$5" "${dir}" | tr '\n' ' ')"
  printf '%q ' "${VLLM_BENCH[@]}" "${args[@]}" > "${dir}/command.txt"; echo >> "${dir}/command.txt"
  post "http://127.0.0.1:${LB_PORT}/start_benchmark" \
    "{\"policy\": \"${policy}\", \"server_info_log_file\": \"${dir}/router_state.jsonl\"}" || echo "start_benchmark failed"
  sleep 2   # the router's server-state loop fills before the first request (tile_aware needs it)
  t_launch=$(date +%s.%N)
  # background + wait, so the TERM before the time limit reaches the trap at once
  timeout "${STAGE_TIMEOUT_S}" "${VLLM_BENCH[@]}" "${args[@]}" > "${dir}/bench.log" 2>&1 &
  BENCH_PID=$!
  wait "${BENCH_PID}"
  rc=$?
  BENCH_PID=""
  t_end=$(date +%s.%N)
  post "http://127.0.0.1:${LB_PORT}/end_benchmark" '{}'
  idle=$(wait_idle) || idle="${idle}_timeout"
  echo "$1,$2,$3,$4,$5,${policy},${t_launch},${t_end},${rc},${idle}" >> "${WORK}/stages.csv"
  echo "stage $1/$3 rc=${rc} s=$(awk -v a="${t_launch}" -v b="${t_end}" 'BEGIN {printf "%.0f", b - a}') idle_s=${idle}" \
    "$(grep -E 'Successful requests|Mean TTFT|Mean TPOT|Request goodput' "${dir}/bench.log" | tr -s ' ' | tr '\n' ' ')"
  return "${rc}"
}
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }
alive() { kill -0 "${P_STEP}" 2>/dev/null && kill -0 "${D_STEP}" 2>/dev/null && kill -0 "${LB_PID}" 2>/dev/null; }

# one completion through router -> P -> NIXL -> D, then a 20-request warm-up with the real client
apply_config rr_default || fail apply_rr_default
sleep 5
curl -sf -m 300 -X POST -H 'Content-Type: application/json' "http://127.0.0.1:${LB_PORT}/v1/completions" \
  -d "{\"model\": \"${MODEL_PATH}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" \
  > "${OUT}/smoke_completion.json" 2>&1
cat "${OUT}/smoke_completion.json"; echo
grep -q '"text"' "${OUT}/smoke_completion.json" || fail "pd_completion (router -> P -> NIXL -> D failed; see logs)"
run_stage smoke smoke smoke 1 20 || fail "smoke_bench rc=$?"
"${PY}" -c '
import json, sys
r = json.load(open(sys.argv[1]))
print("smoke: completed %s/20, output tokens %s, itls %d" % (r.get("completed"), r.get("total_output_tokens"),
      sum(len(x) for x in r.get("itls") or [])))
sys.exit(0 if (r.get("completed") or 0) >= 18 and (r.get("total_output_tokens") or 0) > 20 and any(r.get("itls") or []) else 1)' \
  "${WORK}/smoke/stages/smoke/bench.json" || fail smoke_requests_failed
sleep "${GAP_S}"

STATUS=ok
for cfg in ${CONFIGS}; do
  if [ "$(left)" -le $(( 2 * (STAGE_S + GAP_S) * 6 )) ]; then echo "skip ${cfg}: wall time"; STATUS=partial; continue; fi
  apply_config "${cfg}" || { echo "apply ${cfg} failed"; STATUS=partial; continue; }
  echo "=== ${cfg} policy=$(policy_of "${cfg}") $(date -Iseconds)"
  sleep 10
  for wl in ${WORKLOADS}; do
    i=0
    for rate in ${RATES}; do
      stage="${wl}_r${i}_${rate}rps"; i=$(( i + 1 ))
      if [ "$(left)" -le $(( STAGE_TIMEOUT_S + IDLE_TIMEOUT_S )) ]; then echo "skip ${cfg}/${stage}: wall time"; STATUS=partial; continue; fi
      prompts=$(awk -v r="${rate}" -v s="${STAGE_S}" 'BEGIN {n = int(r * s + 0.5); print (n < 1 ? 1 : n)}')
      run_stage "${cfg}" "${wl}" "${stage}" "${rate}" "${prompts}"
      rc=$?
      alive || fail "server_or_router_died_in_${cfg}_${stage}"
      sleep "${GAP_S}"
      # as bench_production.py: a failed stage skips the rest of its workload
      [ "${rc}" -eq 0 ] || { echo "skip rest of ${cfg}/${wl}: ${stage} rc=${rc}"; STATUS=partial; break; }
    done
  done
done
echo "RESULT=${STATUS}"
[ "${STATUS}" = ok ]
