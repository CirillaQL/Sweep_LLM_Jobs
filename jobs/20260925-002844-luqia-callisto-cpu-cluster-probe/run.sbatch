#!/usr/bin/env bash
#SBATCH --job-name=callisto-cpu-probe
#SBATCH --partition=long
#SBATCH --nodelist=callisto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-002844-luqia-callisto-cpu-cluster-probe/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-002844-luqia-callisto-cpu-cluster-probe/slurm-%j.err

# Read-only CPU probe: no GPU requested, nothing on the cluster is modified.
# Output is written under results/ so the broker commits it back to Git
# (slurm-*.out is gitignored).
set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260925-002844-luqia-callisto-cpu-cluster-probe"
OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
MODEL_ROOT="/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots"
mkdir -p "${OUT_DIR}"
exec > >(tee "${OUT_DIR}/probe.txt") 2>&1

section() { printf '\n===== %s =====\n' "$1"; }
run() { printf '$ %s\n' "$*"; timeout 60 "$@" 2>&1; printf '[exit=%s]\n' "$?"; }

section "job"
echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) user=$(whoami) date=$(date -Iseconds)"
echo "pwd=$(pwd)"
run id

section "partitions (name, time limit, max nodes, default time, state)"
run sinfo -o "%P %l %L %D %t %N"
run scontrol show partition

section "nodes: GPUs configured vs allocated"
for n in callisto europa ganymede neptune uranus sensai; do
  run bash -c "scontrol show node $n | grep -E 'NodeName|State|CfgTRES|AllocTRES|Gres|RealMemory|AllocMem'"
done

section "queue on GPU nodes"
run squeue -w callisto,europa,ganymede,neptune,uranus -o "%.10i %.9P %.10u %.2t %.10M %.10l %.5D %.4C %.8m %b %R"
run squeue -u "$(whoami)" -o "%.10i %.9P %.20j %.2t %.10M %.10l %R"

section "my QoS / association limits"
run sacctmgr -n show assoc user="$(whoami)" format=Cluster,Account,Partition,QOS,MaxJobs,MaxSubmit,GrpTRES
run sacctmgr -n show qos format=Name,MaxWall,MaxTRESPU,MaxJobsPU,Priority

section "software environment"
run ls -ld "${ENV_DIR}" "${ENV_DIR}/bin/python"
run "${ENV_DIR}/bin/python" -c "import sys; print(sys.version)"
run "${ENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as m
for pkg in ("vllm", "torch", "nvidia-ml-py", "pynvml", "transformers"):
    try:
        print(pkg, m.version(pkg))
    except m.PackageNotFoundError:
        print(pkg, "not installed")
PY

section "model snapshot"
run bash -c "ls -la '${MODEL_ROOT}'/*/ | head -30"
run bash -c "du -sh '${MODEL_ROOT}' 2>/dev/null"

section "GPU tooling and clock-lock permission (read-only)"
run which nvidia-smi
run nvidia-smi -L
run sudo -n -l /usr/bin/nvidia-smi

section "filesystem"
run df -h /data /tmp

echo
echo "probe_done=1"
