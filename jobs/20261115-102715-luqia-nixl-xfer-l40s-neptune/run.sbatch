#!/usr/bin/env bash
#SBATCH --job-name=nixl-xfer-l40s
# 同构L40S-单节点-NIXL传输验证 (NixlConnector KV transfer check, L40S/L40S on neptune, short)
# Question: which NIXL setting moves KV between two L40S of one node fast enough for the
# planned L40S/L40S runs (room routing needs D-pull, i.e. vLLM 0.15.1's NIXL READ)?
# No P2pNccl run. Evidence so far:
#   275371 (raw NIXL, no vLLM, neptune GPU0 <- GPU1): VRAM READ 0.62 GB/s (UCX software
#     emulation over TCP), VRAM WRITE 10-16 GB/s (cuda_ipc), DRAM READ 6.5-10 GB/s.
#   274777 (vLLM, ganymede L4, VRAM): 0.22 GB/s effective, TTFT p95 0.69-1.26 s.
#   275522 (vLLM, neptune -> ganymede, DRAM): 61 ms per transfer at low load, but the D
#     call took 566 ms to its first token (median), and transfers collapsed under load.
# One node, neptune, 4 x L40S, 2P2D: each pair (P_i, D_i) on GPUs of one NUMA node
# (sysfs numa_node of the GPU's PCI device; 275371 saw GPU0/1 on NUMA 0, GPU2/3 on NUMA 1).
# Baseline routing (round robin over both pairs), driver clocks (no clock control, no
# energy), NixlConnector, kv_load_failure_policy fail, D gpu_memory_utilization auto.
# Variants, in this order (a variant starts only with >= 8 min left):
#   cpu       kv_buffer_device=cpu  (host buffer; the recommended setting)
#   cuda      kv_buffer_device=cuda (VRAM; what the Vera A100 runs use over InfiniBand)
#   cpu_b128  kv_buffer_device=cpu, vLLM --block-size 128 on P and D (8x fewer, 8x larger
#             transfer descriptors than the default 16-token blocks)
# UCX_PROTO_INFO=y in every variant: the logs name the transport UCX picks for each lane.
# Workload (make_xfer_replay.py), real-time replay, 3 segments x 50 s, output 16 tokens:
#   w00 big 1 req/s x 2048-token prompts | w01 conc 4 req/s x 512/1024/2048 |
#   w02 short 8 req/s x log-normal mean 64, cap 128 (the 283796 short mix)
# Code: CanaTune multi-rho 9f3016d + working tree (the tarball of job 283796, unchanged).
# In-job patch (the unpacked copy only): both vLLM launchers append ${VLLM_EXTRA_ARGS}.
# Read: xfer_summary.txt (per variant and segment: TTFT p50/p95, prefill and decode_first
# medians, vLLM's KV transfer time / MB / GB/s, UCX transports), host_facts.txt.
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=neptune
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261115-102715-luqia-nixl-xfer-l40s-neptune/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261115-102715-luqia-nixl-xfer-l40s-neptune/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
JOB_NAME="20261115-102715-luqia-nixl-xfer-l40s-neptune"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
NIXL_DIR="/data/users/chjing/pylibs/nixl-0.9.0"   # shared, installed by job 274777
CODE_TAR="${JOB_DIR}/canatune_src.tar.gz"
CODE_SHA256="07d80b74fa4ba6a80e8983f6cd89046eb45237820ba411c679523d30afc32c27"
NODE=neptune
WALL_S=1800
RESERVE_S=120          # teardown, log copy and summary before the wall limit
SEGMENT_S=50
VARIANT_MIN_S=480      # start-up (~4 min) + 150 s replay + drain and teardown
VARIANTS="cpu cuda cpu_b128"
mkdir -p "${OUT}"
exec > >(grep --line-buffered -v '"GET /' | tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} code_sha256=${CODE_SHA256}"

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

SUMMARIZED=0
summarize() {
  [ "${SUMMARIZED}" = 1 ] && return 0
  SUMMARIZED=1
  "${PY}" "${JOB_DIR}/summarize_xfer.py" "${OUT}" "${SEGMENT_S}" ${VARIANTS} || echo "WARNING: summary failed"
}
finish() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --mem=256M bash -c \
    'ps -o pid,etime,args -u "$(id -un)" | grep -E "vllm.entrypoints|canatune" | grep -v grep || echo none' \
    2>&1 | sed "s/^/[${NODE} leftover] /" | tee -a "${OUT}/leftover_processes.txt"
  for v in ${VARIANTS}; do
    dir="${WORK}/${v}"
    [ -d "${dir}" ] || continue
    mkdir -p "${OUT}/${v}/logs"
    [ -d "${dir}/records" ] && cp "${dir}/records/"*.jsonl "${OUT}/${v}/" 2>/dev/null
    for f in "${dir}/pd"/*/logs/*.log; do
      [ -f "$f" ] && grep -vE 'GET /(metrics|health)' "$f" | gzip -c > "${OUT}/${v}/logs/$(basename "$f" .log).log.gz"
    done
    [ -f "${WORK}/${v}.yaml" ] && cp "${WORK}/${v}.yaml" "${OUT}/${v}/config.yaml"
  done
  [ -f "${WORK}/xfer_replay.report.json" ] && cp "${WORK}/xfer_replay.report.json" "${OUT}/"
  summarize
  echo "RAW=${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
fail() { echo "RESULT=failed stage=$1"; exit "${2:-1}"; }
left() { echo $(( T0 + WALL_S - RESERVE_S - $(date +%s) )); }

SHAREGPT="/data/users/chjing/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
[ "$(stat -c %s "${SHAREGPT}" 2>/dev/null || echo 0)" = 672837942 ] || fail sharegpt_cache

# ---- 0. NVML and driver (neptune had a driver/library mismatch on 2026-10-09) -----------
out=$(timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --mem=256M bash -c '
  k=$(awk "/NVRM version/{for(i=1;i<=NF;i++) if(\$i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/){print \$i; exit}}" /proc/driver/nvidia/version)
  d=$(modinfo -F version nvidia); nvidia-smi -L >/dev/null 2>&1 && n=ok || n=FAIL
  echo "loaded=$k disk=$d nvml=$n"' 2>&1 | grep '^loaded=')
echo "[${NODE}] ${out}" | tee "${OUT}/preflight.txt"
[[ "${out}" =~ ^loaded=([0-9.]+)\ disk=([0-9.]+)\ nvml=ok$ && "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]] || fail nvml 3

# ---- 1. code from the packed working tree (checksum-verified), launcher patch, checks ----
echo "${CODE_SHA256}  ${CODE_TAR}" | sha256sum -c - || fail code_checksum
mkdir -p "${WORK}/CanaTune" && tar -xzf "${CODE_TAR}" -C "${WORK}/CanaTune" || fail code_extract
cd "${WORK}/CanaTune" || fail code_dir
cp "${JOB_DIR}/SOURCE.txt" "${OUT}/commit.txt" && cat "${OUT}/commit.txt"
for s in scripts/start_4p_prefill_node.sh scripts/start_4d_decode_node.sh; do
  sed -i 's|--no-enable-prefix-caching --no-enable-chunked-prefill \\|--no-enable-prefix-caching --no-enable-chunked-prefill ${VLLM_EXTRA_ARGS:-} \\|' "$s"
  grep -q 'no-enable-chunked-prefill ${VLLM_EXTRA_ARGS:-}' "$s" || fail "launcher_patch_$(basename "$s")"
  bash -n "$s" || fail "launcher_syntax_$(basename "$s")"
done
echo "patched: vLLM launchers append \${VLLM_EXTRA_ARGS}" | tee -a "${OUT}/commit.txt"
PYTHONPATH="${PWD}/src" "${PY}" -c 'import canatune.service, canatune.controller.process_controller, canatune.proxy.proxy, canatune.infrastructure.memory_plan' || fail import_check
for f in "${JOB_DIR}/make_xfer_replay.py" "${JOB_DIR}/summarize_xfer.py" scripts/bench_production.py scripts/make_job_config.py; do
  "${PY}" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$f" || fail "syntax_$(basename "$f")"
done
[ -f "${NIXL_DIR}/.complete" ] || fail nixl_not_installed
export PROJECT_ROOT="${WORK}/CanaTune" PYTHONPATH="${WORK}/CanaTune/src:${NIXL_DIR}"

# ---- 2. NIXL import, host facts (ptrace scope decides UCX's cma shared-memory RMA) --------
timeout 300 srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --gpus-per-node=1 --mem=4G \
  "${PY}" -c '
import torch, vllm, nixl
from nixl._api import nixl_agent, nixl_agent_config
agent = nixl_agent("check", nixl_agent_config(backends=["UCX"]))
print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
print("nixl", nixl.__file__, "plugins", agent.get_plugin_list())
' 2>&1 | grep -E '^(torch|nixl) ' | tee "${OUT}/nixl_check.txt"
grep -q '^nixl ' "${OUT}/nixl_check.txt" || fail nixl_import
timeout 120 srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --mem=256M bash -c '
  echo "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"
  echo "dev_shm=$(df -h /dev/shm | awk "NR==2{print \$2\" total, \"\$4\" free\"}")"
  echo "mem_total=$(awk "/MemTotal/{print \$2\" kB\"}" /proc/meminfo)"
  echo "memlock=$(ulimit -l)"
  nvidia-smi topo -m; nvidia-smi topo -p2p r' 2>&1 | tee "${OUT}/host_facts.txt"

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

# ---- 4. allocated GPUs with NUMA node; pairs inside one NUMA node --------------------------
LINE=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --gpus-per-node=4 --mem=256M bash -c '
  list="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"; IFS=, read -r -a arr <<< "$list"
  for i in "${arr[@]}"; do
    q=$(nvidia-smi -i "$i" --query-gpu=uuid,name,pci.bus_id --format=csv,noheader)
    IFS=, read -r uuid name bus <<< "$q"
    bus=$(echo "$bus" | tr -d " " | tr "A-F" "a-f"); bus="${bus: -12}"
    numa=$(cat "/sys/bus/pci/devices/${bus}/numa_node" 2>/dev/null || echo -1)
    echo "gpu $i $(echo "$uuid" | tr -d " ") ${numa} ${name# }"
  done' 2>&1 | grep '^gpu ')
echo "${LINE}" | tee "${OUT}/gpu_allocation.txt"
[ "$(echo "${LINE}" | grep -c 'L40S')" = 4 ] || fail gpu_detect 3
ORDER=(); while IFS= read -r row; do ORDER+=("${row}"); done < <(echo "${LINE}" | sort -k4,4n -k2,2n)   # by NUMA node, then index
read -r _ g0 u0 n0 _ <<< "${ORDER[0]}"; read -r _ g1 u1 n1 _ <<< "${ORDER[1]}"
read -r _ g2 u2 n2 _ <<< "${ORDER[2]}"; read -r _ g3 u3 n3 _ <<< "${ORDER[3]}"
PAIRING=numa; [[ "${n0}" == "${n1}" && "${n2}" == "${n3}" && "${n0}" != -1 ]] || PAIRING=fallback
P_IDS="${g0},${g2}"; P_UUIDS="${u0},${u2}"; D_IDS="${g1},${g3}"; D_UUIDS="${u1},${u3}"
echo "pairing=${PAIRING}: (P0 ${g0}, D0 ${g1}) numa ${n0}/${n1}; (P1 ${g2}, D1 ${g3}) numa ${n2}/${n3}" \
  | tee -a "${OUT}/gpu_allocation.txt"
IP=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" --mem=256M hostname -I | awk '{print $1}')
[[ "${IP}" =~ ^[0-9.]+$ ]] || fail resolve_ip
echo "ip ${NODE}=${IP}" | tee -a "${OUT}/gpu_allocation.txt"

# ---- 5. configs (one per variant) and the replay ----------------------------------------
PROXY_PORT=$(( 18000 + ${JOB_ID//[^0-9]/} % 1000 ))
common=(--base config.yaml --prefill-node "${NODE}" --decode-node "${NODE}"
  --prefill-gpus "${P_IDS}" --prefill-uuids "${P_UUIDS}"
  --decode-gpus "${D_IDS}" --decode-uuids "${D_UUIDS}"
  --python "${PY}" --work-dir "${WORK}" --set clock_control.enabled=false
  --set proxy.host=127.0.0.1 --set "proxy.port=${PROXY_PORT}"
  --set topology.decode_nodegroup.kv_port_base=14679
  --set "topology.decode_nodegroup.gpu_type=NVIDIA L40S"
  --set topology.decode_nodegroup.high_frequency_mhz=2520
  --set kv_transfer.connector=NixlConnector
  --set routing.policy=round_robin --set 'routing.production_pairs=[[P0, D0], [P1, D1]]'
  --set experiment.slo.ttft_ms=600 --set experiment.slo.tpot_ms=100)
for v in ${VARIANTS}; do
  case "${v}" in cuda) dev=cuda;; *) dev=cpu;; esac
  "${PY}" scripts/make_job_config.py "${common[@]}" --set "kv_transfer.kv_buffer_device=${dev}" \
    --out "${WORK}/${v}.yaml" || fail "job_config_${v}"
done
grep -nE "connector:|kv_buffer_device|policy:|enabled:" "${WORK}/cpu.yaml" | head
REPLAY="${WORK}/xfer_replay.jsonl"
timeout 900 "${PY}" "${JOB_DIR}/make_xfer_replay.py" --scripts "${WORK}/CanaTune/scripts" \
  --sharegpt "${SHAREGPT}" --tokenizer "${MODEL_PATH}" --out "${REPLAY}" \
  --segment-s "${SEGMENT_S}" || fail replay_dataset
cp "${WORK}/xfer_replay.report.json" "${OUT}/"

# ---- 6. the variants ------------------------------------------------------------------------
export PREFILL_NODE="${NODE}" DECODE_NODE="${NODE}"
export CANATUNE_PREFILL_HOST="${IP}" CANATUNE_DECODE_HOST="${IP}"
export CANATUNE_PREFILL_KV_HOST="${IP}" CANATUNE_DECODE_KV_HOST="${IP}"
export PYTHON_BIN="${PY}" MAX_MODEL_LEN=4096
export PREFILL_GPU_IDS="${P_IDS}" PREFILL_GPU_UUIDS="${P_UUIDS}"
export DECODE_GPU_IDS="${D_IDS}" DECODE_GPU_UUIDS="${D_UUIDS}"
export PREFILL_HTTP_PORT_BASE=8100 PREFILL_KV_PORT_BASE=14579
export DECODE_HTTP_PORT_BASE=8200 DECODE_KV_PORT_BASE=14679   # one host: P keeps 14579
unset DECODE_GPU_MEMORY_UTILIZATION   # auto (memory_plan, NIXL terms)
export UCX_PROTO_INFO=y
export BENCH_WORKLOADS="[
  {\"name\": \"replay\", \"dataset\": \"${REPLAY}\", \"replay\": true, \"segment_s\": ${SEGMENT_S},
   \"ignore_eos\": true}]"
export BENCH_SCORE_SLOS="600:100" BENCH_STAGE_S=60 BENCH_GAP_S=10
export BENCH_SEED=20261115 BENCH_BURSTINESS=1
export BENCH_ENDPOINT=/v1/completions BENCH_CMD="${ENV_DIR}/bin/vllm bench serve"
# STARTUP: per vLLM instance (P0 then P1 in one launcher); ENDPOINT: all four healthy.
export STARTUP_TIMEOUT_S=420 ENDPOINT_TIMEOUT_S=600 SHUTDOWN_GRACE_S=20 CANATUNE_STEP_GPUS=0
export PYTHONFAULTHANDLER=1
RESULTS=""
for v in ${VARIANTS}; do
  if [ "$(left)" -lt "${VARIANT_MIN_S}" ]; then
    echo "=== ${v} skipped: $(left) s left"; RESULTS="${RESULTS} ${v}:skipped"; continue
  fi
  case "${v}" in cpu_b128) export VLLM_EXTRA_ARGS="--block-size 128";; *) unset VLLM_EXTRA_ARGS;; esac
  mkdir -p "${OUT}/${v}" "${WORK}/${v}"
  export CANATUNE_CONFIG="${WORK}/${v}.yaml" BENCH_MODE=baseline BENCH_DEADLINE_S=0
  export PD_WORK_DIR="${WORK}/${v}/pd" CANATUNE_RECORD_DIR="${WORK}/${v}/records"
  export CANATUNE_TIER_TABLE="${WORK}/${v}/tiers.json" CANATUNE_RISK_TABLE="${WORK}/${v}/risk.json"
  export BENCH_OUT="${OUT}/${v}"
  echo "=== ${v} start=$(date -Iseconds) left_s=$(left) extra='${VLLM_EXTRA_ARGS:-}'"
  timeout "$(left)" "${PY}" scripts/bench_production.py
  rc=$?
  echo "RESULT_${v}=${rc}"; RESULTS="${RESULTS} ${v}:${rc}"
  sleep 5
done
echo "RESULT=${RESULTS}"
case " ${RESULTS} " in *" cpu:0 "*) exit 0;; esac
exit 1
