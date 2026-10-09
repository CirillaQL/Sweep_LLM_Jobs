#!/usr/bin/env bash
#SBATCH --job-name=nixl-smoke
# NIXL 冒烟测试 (NixlConnector smoke test, CanaTune 48e1476), two nodes, short.
# P = 2 x L40S on uranus, D = 2 x L4 on ganymede (cross-node L40S -> L4 KV, as in the
# neptune -> ganymede jobs; neptune's driver is broken until the admins reload it).
#   0. NIXL: nixl 0.9.0 + nixl-cu12 0.9.0 (vLLM 0.15.1's time; the wheel bundles UCX)
#      installed once with --no-deps into a shared directory, then an import check and
#      one UCX agent; torch's CUDA version and the NICs of both nodes are recorded.
#   1. run nixl:  baseline (round robin over (P0, D0), (P1, D1)), NixlConnector,
#      kv_load_failure_policy fail, D gpu_memory_utilization auto (no receive buffer:
#      1 - 1.25 GB / GPU memory). ShareGPT completions, output cap 256 (EOS stops),
#      Poisson 1, 4, 8 req/s, 60 s of arrivals each.
#   2. run p2p:   the same with P2pNcclConnector PUT_ASYNC (the connector of all earlier
#      jobs) on the same GPUs, if at least 11 min are left: TTFT / TPOT side by side.
# Recorded per run: bench summaries, per-request records (status: a failed NIXL read is a
# decode_error), vLLM logs (gz) and counts of NIXL / KV warnings in them.
# Clock control off (driver clocks, no energy): this checks function and latency only.
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus,ganymede
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-144820-luqia-nixl-smoke-l4-2node/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-144820-luqia-nixl-smoke-l4-2node/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261009-144820-luqia-nixl-smoke-l4-2node"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
NIXL_DIR="/data/users/chjing/pylibs/nixl-0.9.0"   # shared: installed once, reused
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="48e1476892bf88b294e1e74a68d48f8ff3465e58"
P_NODE=uranus
D_NODE=ganymede
WALL_S=1800
RESERVE_S=150   # teardown and result copy before the wall limit
mkdir -p "${OUT}"
exec > >(grep --line-buffered -v '"GET /' | tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} commit=${COMMIT}"

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

finish() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  for node in "${P_NODE}" "${D_NODE}"; do
    timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
      'ps -o pid,etime,args -u "$(id -un)" | grep -E "vllm.entrypoints|canatune" | grep -v grep || echo none' \
      2>&1 | sed "s/^/[${node} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  done
  for dir in "${WORK}"/run_*; do
    [ -d "${dir}" ] || continue
    run=$(basename "${dir}")
    mkdir -p "${OUT}/${run}/logs"
    [ -d "${dir}/records" ] && cp "${dir}/records/"*.jsonl "${OUT}/${run}/" 2>/dev/null
    for f in "${dir}/pd"/*/logs/*.log; do
      [ -f "$f" ] || continue
      name=$(basename "$f" .log)
      grep -vE 'GET /(metrics|health)' "$f" | gzip -c > "${OUT}/${run}/logs/${name}.log.gz"
      {
        printf '%s nixl_lines=%s nixl_errors=%s kv_cache_none=%s p2p_overflow=%s errors=%s\n' "${name}" \
          "$(grep -c -i 'nixl' "$f")" \
          "$(grep -i 'nixl' "$f" | grep -c -i -E 'error|fail|timeout|abort')" \
          "$(grep -c 'kv_cache is None' "$f")" \
          "$(grep -c -E 'Out Of (Memory|Threshold)' "$f")" \
          "$(grep -c -E '\bERROR\b' "$f")"
      } >> "${OUT}/${run}/log_counts.txt"
      grep -i -E 'nixl|ucx' "$f" | grep -v -E 'GET /' | head -n 60 > "${OUT}/${run}/logs/${name}.nixl_head.txt"
    done
    [ -f "${dir}/records/requests.jsonl" ] && "${PY}" - "${dir}/records/requests.jsonl" \
      > "${OUT}/${run}/request_status.txt" 2>&1 <<'PY'
import collections, json, sys
counts = collections.Counter(json.loads(line).get("status") for line in open(sys.argv[1]))
print(dict(counts))
PY
  done
  [ -f "${WORK}/job_config_nixl.yaml" ] && cp "${WORK}"/job_config_*.yaml "${OUT}/"
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }

SHAREGPT="/data/users/chjing/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
[ "$(stat -c %s "${SHAREGPT}" 2>/dev/null || echo 0)" = 672837942 ] || fail sharegpt_cache

# ---- 0. NVML on both nodes (a driver/library mismatch fails here, not in vLLM) ----------
for node in "${P_NODE}" "${D_NODE}"; do
  timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c \
    'cat /proc/driver/nvidia/version | head -1; modinfo -F version nvidia; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader; ip -br -4 addr' \
    2>&1 | sed "s/^/[${node}] /" | tee -a "${OUT}/nodes.txt"
  timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M nvidia-smi -L >/dev/null 2>&1 \
    || fail "nvml_${node}" 3
done

# ---- 1. code at the pinned commit + CPU tests -----------------------------------------
timeout 300 git clone --quiet "${REPO}" "${WORK}/CanaTune" || fail clone
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" || fail checkout
git log -1 --format='%H %s' | tee "${OUT}/commit.txt"
timeout 300 "${PY}" -m pip install --quiet --target "${WORK}/pytest-lib" "pytest>=8.3,<9" || fail pip_pytest
PYTHONPATH="${PWD}/src:${WORK}/pytest-lib" timeout 300 "${PY}" -m pytest -q -p no:cacheprovider \
  --deselect tests/test_measure.py::test_stalled_window_is_repeated_and_does_not_stop_the_scan \
  > "${OUT}/pytest.txt" 2>&1 || { tail -n 30 "${OUT}/pytest.txt"; fail pytest; }
tail -n 1 "${OUT}/pytest.txt"

# ---- 2. NIXL (shared directory; --no-deps: torch and numpy come from the env) -----------
if [ ! -f "${NIXL_DIR}/.complete" ]; then
  PART="${NIXL_DIR}.part.${JOB_ID}"
  rm -rf "${PART}"
  timeout 600 "${PY}" -m pip install --quiet --no-deps --target "${PART}" \
    "nixl==0.9.0" "nixl-cu12==0.9.0" || { rm -rf "${PART}"; fail pip_nixl; }
  touch "${PART}/.complete"
  if [ ! -f "${NIXL_DIR}/.complete" ]; then rm -rf "${NIXL_DIR}"; mv "${PART}" "${NIXL_DIR}"; else rm -rf "${PART}"; fi
fi
export PROJECT_ROOT="${WORK}/CanaTune" PYTHONPATH="${WORK}/CanaTune/src:${NIXL_DIR}"
for node in "${P_NODE}" "${D_NODE}"; do
  timeout 300 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=1 --mem=4G \
    "${PY}" -c '
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
import nixl
from nixl._api import nixl_agent, nixl_agent_config
agent = nixl_agent("smoke", nixl_agent_config(backends=["UCX"]))
print("nixl", nixl.__file__, "plugins", agent.get_plugin_list())
' 2>&1 | grep -v -E '^\s*$' | tail -n 8 | sed "s/^/[${node}] /" | tee -a "${OUT}/nixl_check.txt"
  grep -q "^\[${node}\] nixl " "${OUT}/nixl_check.txt" || fail "nixl_import_${node}"
done

# ---- 3. model snapshot (offline) ----------------------------------------------------------
SNAP_ROOT="/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots"
MODEL_PATH=""
while IFS= read -r snap; do
  if [ -s "${snap}/config.json" ] && [ -s "${snap}/tokenizer_config.json" ] && \
     find "${snap}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then MODEL_PATH="${snap}"; break; fi
done < <(find "${SNAP_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ -n "${MODEL_PATH}" ] || fail model_snapshot
export MODEL_PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "model=${MODEL_PATH}"

# ---- 4. allocated GPUs (index + UUID) and node IPs -------------------------------------
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
echo "detected ${P_NODE}=${P_LINE} ${D_NODE}=${D_LINE}" | tee "${OUT}/gpu_allocation.txt"
IFS=';' read -r P_IDS P_UUIDS P_NAMES <<< "${P_LINE}"
IFS=';' read -r D_IDS D_UUIDS D_NAMES <<< "${D_LINE}"
[[ "${P_IDS}" =~ ^[0-9]+,[0-9]+$ && "${D_IDS}" =~ ^[0-9]+,[0-9]+$ ]] || fail gpu_detect_count 3
ip_of() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" --mem=256M hostname -I | awk '{print $1}'; }
P_IP=$(ip_of "${P_NODE}"); D_IP=$(ip_of "${D_NODE}")
[[ "${P_IP}" =~ ^[0-9.]+$ && "${D_IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
echo "ips ${P_NODE}=${P_IP} ${D_NODE}=${D_IP}" | tee -a "${OUT}/gpu_allocation.txt"

# ---- 5. job configs: baseline round robin, NIXL and P2pNccl -------------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
common=(--base config.yaml --prefill-node "${P_NODE}" --decode-node "${D_NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=false
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}"
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1]]'
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100)
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/job_config_nixl.yaml" \
  --set kv_transfer.connector=NixlConnector || fail job_config_nixl
"${PY}" scripts/make_job_config.py "${common[@]}" --out "${WORK}/job_config_p2p.yaml" \
  --set kv_transfer.connector=P2pNcclConnector || fail job_config_p2p

export PREFILL_NODE="${P_NODE}" DECODE_NODE="${D_NODE}"
export CANATUNE_PREFILL_HOST="${P_IP}" CANATUNE_DECODE_HOST="${D_IP}"
export CANATUNE_PREFILL_KV_HOST="${P_IP}" CANATUNE_DECODE_KV_HOST="${D_IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14579
unset DECODE_GPU_MEMORY_UTILIZATION   # auto (memory_plan, per connector)
export PREFILL_KV_SEND_TYPE=PUT_ASYNC NCCL_SOCKET_IFNAME='^lo'   # P2pNccl run only
export BENCH_WORKLOADS="[
  {\"name\": \"sharegpt\", \"dataset\": \"${SHAREGPT}\", \"rates\": \"1,4,8\",
   \"output_len\": 256, \"ignore_eos\": false}]"
export BENCH_SCORE_SLOS="600:100" BENCH_STAGE_S=60 BENCH_GAP_S=10
export BENCH_SEED=20261002 BENCH_BURSTINESS=1
export BENCH_ENDPOINT=/v1/completions BENCH_CMD="${ENV_DIR}/bin/vllm bench serve"
export STARTUP_TIMEOUT_S=600 ENDPOINT_TIMEOUT_S=780 SHUTDOWN_GRACE_S=30 CANATUNE_STEP_GPUS=0
export PYTHONFAULTHANDLER=1
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }
run_mode() {  # $1 = run name, $2 = config file
  local run="$1" cfg="$2"
  mkdir -p "${OUT}/${run}" "${WORK}/${run}"
  export CANATUNE_CONFIG="${cfg}" BENCH_MODE=baseline
  export PD_WORK_DIR="${WORK}/${run}/pd" CANATUNE_RECORD_DIR="${WORK}/${run}/records"
  export CANATUNE_TIER_TABLE="${WORK}/${run}/tiers.json" CANATUNE_RISK_TABLE="${WORK}/${run}/risk.json"
  export BENCH_OUT="${OUT}/${run}" BENCH_DEADLINE_S=0
  echo "=== ${run} start=$(date -Iseconds) left_s=$(left)"
  timeout $(( $(left) )) "${PY}" scripts/bench_production.py
  local rc=$?
  echo "RESULT_${run}=${rc}"
  return "${rc}"
}
RESULTS=""
run_mode run_nixl "${WORK}/job_config_nixl.yaml"; RESULTS="${RESULTS} run_nixl:$?"
if [ "$(left)" -ge 660 ]; then
  sleep 10
  run_mode run_p2p "${WORK}/job_config_p2p.yaml"; RESULTS="${RESULTS} run_p2p:$?"
else
  RESULTS="${RESULTS} run_p2p:skipped"
fi
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *run_nixl:0*) exit 0;; esac
exit 1
