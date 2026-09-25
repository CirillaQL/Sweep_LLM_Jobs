#!/usr/bin/env bash
#SBATCH --job-name=uranus-vllm-cleanup
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/tmp/uranus-cleanup-%j.out
#SBATCH --error=/tmp/uranus-cleanup-%j.err

# Kill vLLM servers left behind on uranus by the failed K4a jobs 267652 and 267664
# (stuck in D state, wchan nfs_wait_bit_killable), then check whether uranus can
# read /data again. Only processes of this account whose cgroup belongs to one of
# those two jobs are touched. No sudo, no GPU, no clock changes.
set -uo pipefail
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-135753-luqia-uranus-stuck-vllm-cleanup"
OUT="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
# uranus may still hang on /data: log to local disk, copy to /data (bounded) at the end.
LOCAL_LOG="/tmp/uranus-cleanup-${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "${LOCAL_LOG}") 2>&1
publish() {
  timeout -s KILL 60 mkdir -p "${OUT}" && timeout -s KILL 60 cp "${LOCAL_LOG}" "${OUT}/cleanup.log" \
    || echo "could not write ${OUT} (NFS still unavailable?); local copy: uranus:${LOCAL_LOG}" >&2
}
trap publish EXIT
TARGET_JOBS="267652 267664"
ME="$(id -un)"
echo "start=$(date -Iseconds) host=$(hostname) user=${ME} targets=${TARGET_JOBS}"

snapshot() {  # all processes of this account that look like vLLM servers, with job cgroup
  echo "--- $1"
  local pid cg
  for pid in $(timeout -s KILL 20 pgrep -u "${ME}" -f 'vllm.entrypoints.openai.api_server'); do
    cg=$(timeout -s KILL 5 grep -o 'job_[0-9]*' "/proc/${pid}/cgroup" 2>/dev/null | head -n 1)
    printf '%s %s ' "${pid}" "${cg:-job_unknown}"
    timeout -s KILL 10 ps -o stat=,etime=,wchan:24= -p "${pid}" 2>/dev/null || echo "(gone)"
  done
}

targets() {  # PIDs whose cgroup names one of the target jobs
  local pid cg job
  for pid in $(timeout -s KILL 20 pgrep -u "${ME}" -f 'vllm.entrypoints.openai.api_server'); do
    cg=$(timeout -s KILL 5 grep -o 'job_[0-9]*' "/proc/${pid}/cgroup" 2>/dev/null | head -n 1)
    job="${cg#job_}"
    for t in ${TARGET_JOBS}; do [ "${job}" = "${t}" ] && echo "${pid}"; done
  done
}

snapshot "before"
PIDS="$(targets | tr '\n' ' ')"
echo "target_pids=${PIDS:-none}"
if [ -n "${PIDS// /}" ]; then
  kill -TERM ${PIDS} 2>/dev/null; sleep 10
  snapshot "after SIGTERM"
  PIDS="$(targets | tr '\n' ' ')"
  [ -z "${PIDS// /}" ] || { kill -KILL ${PIDS} 2>/dev/null; sleep 20; }
  snapshot "after SIGKILL"
fi
LEFT="$(targets | tr '\n' ' ')"
echo "remaining_target_pids=${LEFT:-none}"

echo "--- /data read check on uranus (each step killed after its timeout)"
PY=/data/users/chjing/miniforge3/envs/cuda-env/bin/python
MODEL_DIR=/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots
check() {
  local name="$1"; shift
  local t0 rc
  t0=$(date +%s%N)
  if timeout -s KILL "$@" >/dev/null 2>&1; then rc=0; else rc=$?; fi
  echo "${name} rc=${rc} ms=$(( ($(date +%s%N) - t0) / 1000000 )) (rc 137 = killed at timeout)"
}
check stat_python 20 stat "${PY}"
check python_start 60 "${PY}" -c 'import sys'
check import_torch 120 "${PY}" -c 'import torch'
SAFE=$(timeout -s KILL 20 find "${MODEL_DIR}" -maxdepth 2 -name '*.safetensors' -print -quit 2>/dev/null)
echo "safetensors=${SAFE:-not_found}"
[ -z "${SAFE}" ] || check read_64MiB_weights 120 dd if="${SAFE}" of=/dev/null bs=1M count=64
timeout -s KILL 10 df -h /data | tail -n 1
echo "end=$(date -Iseconds)"
