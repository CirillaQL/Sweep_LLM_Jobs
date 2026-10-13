#!/usr/bin/env bash
#SBATCH --job-name=europa-check
# europa 节点检查 (one L4, no vLLM): driver / NVML, network (addresses, NICs, speeds,
# RDMA devices and modules, latency to the other GPU nodes) and whether the CanaTune GPU
# agent (13b6052, scripts as in the jobs) can lock clocks: start it on the allocated GPU,
# read its allowed clocks, lock two of them with a short matmul load at each (SM clock and
# power read back over the agent's API), reset, read the clocks again.
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=europa
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@60
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-000543-luqia-europa-net-agent-clock-check/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-000543-luqia-europa-net-agent-clock-check/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20261014-000543-luqia-europa-net-agent-clock-check"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO="https://github.com/CirillaQL/CanaTuneLLM.git"
COMMIT="13b605249c0929c8fb9444489074c905f1360c5d"
AGENT_PORT=$(( 9300 + ${JOB_ID//[^0-9]/} % 500 ))
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} commit=${COMMIT}"
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" XDG_CACHE_HOME="${WORK}/xdg-cache"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache" PYTHONNOUSERSITE=1 GIT_TERMINAL_PROMPT=0
export PATH="${ENV_DIR}/bin:${PATH}"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}"

GPU="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"; GPU="${GPU%%,*}"
AGENT_PID=""
cleanup() {
  set +e
  [ -n "${AGENT_PID}" ] && kill "${AGENT_PID}" 2>/dev/null && sleep 2
  [ -n "${GPU}" ] && sudo -n nvidia-smi -i "${GPU}" -rgc >/dev/null 2>&1 \
    && echo "final reset GPU ${GPU}: ok" || echo "final reset GPU ${GPU}: FAILED"
  nvidia-smi -i "${GPU}" --query-gpu=index,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap cleanup EXIT

echo "== 1. driver / NVML / GPU"
k=$(awk '/NVRM version/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/){print $i; exit}}' /proc/driver/nvidia/version)
echo "loaded=${k} disk=$(modinfo -F version nvidia)"
nvidia-smi -L
echo "allocated GPU index=${GPU}"
[ -n "${GPU}" ] || { echo "RESULT=failed stage=no_gpu"; exit 1; }
UUID=$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
nvidia-smi -i "${GPU}" --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit --format=csv
nvidia-smi -i "${GPU}" -q -d SUPPORTED_CLOCKS | grep -m1 -A3 "Memory"
echo "sudo rules:"; sudo -n -l 2>&1 | tail -n 5

echo "== 2. network"
echo "hostname -I: $(hostname -I)"
ip -br addr
ip route
for n in /sys/class/net/*; do
  i=$(basename "$n"); [ "$i" = lo ] && continue
  drv=$(basename "$(readlink -f "$n/device/driver" 2>/dev/null)" 2>/dev/null)
  echo "nic ${i}: state=$(cat "$n/operstate") speed=$(cat "$n/speed" 2>/dev/null)Mb/s mtu=$(cat "$n/mtu") driver=${drv:-none}"
done
echo "rdma devices: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"
echo "rdma modules: $(lsmod | awk '{print $1}' | grep -E '^(bnxt_re|mlx5_ib|irdma|ib_core|rdma_ucm|nvidia_peermem)$' | tr '\n' ' ')"
for host in neptune ganymede uranus callisto; do
  ip=$(getent hosts "${host}" | awk '{print $1}' | head -1)
  rtt=$(ping -c 5 -q "${host}" 2>/dev/null | awk -F/ '/rtt|round-trip/ {print $5}')
  echo "to ${host}: ip=${ip:-?} rtt_avg_ms=${rtt:-unreachable}"
done

echo "== 3. CanaTune GPU agent"
timeout 300 git clone --quiet "${REPO}" "${WORK}/CanaTune" || { echo "RESULT=failed stage=clone"; exit 1; }
cd "${WORK}/CanaTune" && git checkout --quiet "${COMMIT}" && git log -1 --format='%H %s'
export PYTHONPATH="${WORK}/CanaTune/src"
CANATUNE_AGENT_GPUS="${GPU}" CANATUNE_AGENT_GPU_UUIDS="${UUID}" CANATUNE_AGENT_PORT="${AGENT_PORT}" \
  "${PY}" -m canatune.infrastructure.gpu_agent > "${OUT}/agent.log" 2>&1 &
AGENT_PID=$!
URL="http://127.0.0.1:${AGENT_PORT}"
for _ in $(seq 60); do curl -fs "${URL}/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fs "${URL}/health" || { echo "RESULT=failed stage=agent_start"; tail -n 30 "${OUT}/agent.log"; exit 1; }
echo
curl -fs "${URL}/net" | "${PY}" -c 'import json,sys; d=json.load(sys.stdin); print("agent /net interfaces:", sorted(d["interfaces"]))'
ALLOWED=$(curl -fs "${URL}/gpus/${GPU}/clocks" | "${PY}" -c 'import json,sys; v=json.load(sys.stdin)["sm_mhz"]; print(" ".join(map(str,v)))')
echo "allowed SM clocks (${GPU}): ${ALLOWED}"
read -r -a A <<< "${ALLOWED}"
N=${#A[@]}
[ "${N}" -ge 2 ] || { echo "RESULT=failed stage=no_allowed_clocks"; exit 1; }
# a high-middle and a low-middle clock of the GPU's own list (sorted either way)
TARGETS=$(printf '%s\n' "${A[@]}" | sort -n | awk -v n="${N}" '{c[NR]=$1} END {print c[int(n*0.75)+0>n?n:int(n*0.75)], c[int(n*0.35)>0?int(n*0.35):1]}')
load() {  # $1 seconds of matmul on the allocated GPU
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c '
import sys, time, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16); b = torch.randn_like(a)
end = time.time() + float(sys.argv[1])
while time.time() < end:
    (a @ b); torch.cuda.synchronize()' "$1" &
  LOAD_PID=$!
}
for mhz in ${TARGETS}; do
  echo "-- lock ${mhz} MHz"
  curl -fs -X POST "${URL}/gpus/${GPU}/lock" -H 'Content-Type: application/json' -d "{\"mhz\": ${mhz}}"; echo
  load 12; sleep 7
  echo "under load: $(curl -fs "${URL}/gpus")"
  wait "${LOAD_PID}"
  sleep 2; echo "idle after: $(curl -fs "${URL}/gpus")"
done
echo "-- reset"
curl -fs -X POST "${URL}/gpus/${GPU}/reset"; echo
load 10; sleep 6
echo "default clocks under load: $(curl -fs "${URL}/gpus")"
wait "${LOAD_PID}"
kill "${AGENT_PID}" 2>/dev/null; wait "${AGENT_PID}" 2>/dev/null; AGENT_PID=""
echo "agent log tail:"; tail -n 5 "${OUT}/agent.log"
echo "RESULT=ok"
