#!/usr/bin/env bash
#SBATCH --job-name=nvml-clock-probe
# Can chjing change GPU clocks through NVML without sudo? One L40S (neptune) and one L4
# (ganymede), allocated GPUs only. nvml_probe.py tries locked clocks and application clocks
# as the plain user, reads the SM clock back idle and under a short load, and resets. The
# teardown then resets both the established way (sudo -n nvidia-smi -rgc / -rac), UUID-checked.
# No vLLM. Results: results/${SLURM_JOB_ID}/ (summary.txt first).
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=neptune,ganymede
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=8G
#SBATCH --signal=B:TERM@60
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261105-154809-luqia-nvml-clock-probe/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261105-154809-luqia-nvml-clock-probe/slurm-%j.err

set -uo pipefail
T0=$(date +%s)
JOB_NAME="20261105-154809-luqia-nvml-clock-probe"
JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
NODES="neptune ganymede"
mkdir -p "${OUT}" "${WORK}/home" "${WORK}/tmp" "${WORK}/pycache"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID}"

export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" XDG_CACHE_HOME="${WORK}/tmp"
export PYTHONPYCACHEPREFIX="${WORK}/pycache" PYTHONNOUSERSITE=1

declare -A GPU_ID GPU_UUID
query='index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,utilization.gpu,clocks_event_reasons.active'
on_node() { local node="$1" mem="$2"; shift 2
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem="${mem}" "$@"; }
reset_clocks() {  # the allocated GPU only, by UUID
  local node="$1" id="${GPU_ID[$1]:-}" uuid="${GPU_UUID[$1]:-}"
  [ -n "${id}" ] || return 0
  timeout 60 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=256M bash -c '
    u=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | tr -d " ")
    [ "$u" = "$2" ] || { echo "skip $1: uuid $u != $2"; exit 0; }
    sudo -n nvidia-smi -i "$1" -rgc >/dev/null 2>&1 && echo "rgc $1 ok" || echo "rgc $1 FAILED"
    sudo -n nvidia-smi -i "$1" -rac >/dev/null 2>&1 && echo "rac $1 ok" || echo "rac $1 FAILED"
    sleep 3
    nvidia-smi -i "$1" --query-gpu="$3" --format=csv,noheader' _ "${id}" "${uuid}" "${query}" 2>&1 | sed "s/^/[${node}] /"
}
finish() {
  set +e
  echo "--- teardown $(date -Iseconds)"
  for node in ${NODES}; do reset_clocks "${node}" | tee -a "${OUT}/clock_reset.txt"; done
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - T0 ))"
}
trap finish EXIT
trap 'exit 143' TERM

"${PY}" -c 'import pynvml; print("pynvml", pynvml.__file__)' || { echo "RESULT=failed stage=import_pynvml"; exit 1; }

for node in ${NODES}; do
  line=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=1 --mem=256M bash -c '
    i="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
    echo "gpu=$i;$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader | tr -d " ")"' 2>&1 | grep '^gpu=' | tail -n 1 | sed 's/^gpu=//')
  IFS=';' read -r id uuid <<< "${line}"
  [[ "${id}" =~ ^[0-9]+$ && "${uuid}" == GPU-* ]] || { echo "RESULT=failed stage=gpu_detect node=${node} line=${line}"; exit 3; }
  GPU_ID[${node}]="${id}"; GPU_UUID[${node}]="${uuid}"
  echo "${node}: gpu ${id} ${uuid}" | tee -a "${OUT}/gpu_allocation.txt"
done

for node in ${NODES}; do
  echo "=== ${node} before"
  on_node "${node}" 256M nvidia-smi -i "${GPU_ID[${node}]}" --query-gpu="${query}" --format=csv
  on_node "${node}" 256M bash -c 'id; cat /run/nvgpufreq.run 2>/dev/null || echo "no /run/nvgpufreq.run"'
  timeout -s TERM 240 srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --mem=6G \
    "${PY}" "${JOB_DIR}/nvml_probe.py" "${GPU_UUID[${node}]}" "${OUT}/probe_${node}.json" \
    2>&1 | tee -a "${OUT}/probe_console.txt"
  echo "${node} probe rc=${PIPESTATUS[0]}"
  echo "=== ${node} after"
  on_node "${node}" 256M nvidia-smi -i "${GPU_ID[${node}]}" --query-gpu="${query}" --format=csv
done
grep -h '^NVML_PROBE' "${OUT}/probe_console.txt" | tee "${OUT}/summary.txt"
echo "RESULT=done"
