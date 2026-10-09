#!/usr/bin/env bash
#SBATCH --job-name=nv-check
# Read-only check of neptune's NVIDIA driver after job 274323 (homo-l40s x3) failed with
# "Failed to initialize NVML: Driver/library version mismatch" (NVML library 595.91) from
# 2026-10-09 ~06:00-06:20: kernel module version vs user-space library version, whether the
# node rebooted since, recent NVIDIA package changes, and whether the GPU SM clocks are still
# locked (job 274323 left P at 1545 MHz / D at 1965 MHz when its clock reset failed).
# Changes nothing: no clock lock/reset, no GPU allocation (devices are visible without one).
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=neptune
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-094544-luqia-neptune-nvidia-driver-check/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-094544-luqia-neptune-nvidia-driver-check/slurm-%j.err

set -u
OUT="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261009-094544-luqia-neptune-nvidia-driver-check/results/${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/check.txt") 2>&1
sec() { echo; echo "==== $*"; }

sec "host / time / uptime"
hostname; date -Iseconds; uptime; uname -r
who -b 2>/dev/null || true
cat /etc/os-release 2>/dev/null | head -3

sec "kernel module loaded (/proc/driver/nvidia/version)"
cat /proc/driver/nvidia/version 2>&1
sec "kernel module on disk (modinfo)"
/sbin/modinfo nvidia 2>&1 | grep -E '^(filename|version|vermagic)' || modinfo nvidia 2>&1 | grep -E '^(filename|version|vermagic)'
sec "loaded module params / refcount"
cat /sys/module/nvidia/version 2>&1
lsmod 2>/dev/null | grep -E '^nvidia' || true

sec "user-space NVML library"
ldconfig -p 2>/dev/null | grep -E 'libnvidia-ml|libcuda\.so' || true
ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* /usr/lib64/libnvidia-ml.so* 2>/dev/null || true

sec "nvidia-smi"
nvidia-smi 2>&1 | head -40
sec "per-GPU clocks (idle unlocked GPUs drop to a low SM clock; a GPU held at 1545 / 1965 with no work is still locked)"
nvidia-smi --query-gpu=index,uuid,name,driver_version,pstate,utilization.gpu,memory.used,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks_event_reasons.active --format=csv 2>&1
sec "clock details"
nvidia-smi -q -d CLOCK 2>&1 | grep -E 'GPU 0000|Graphics|SM|Locked|Applications|Default' | head -60
sec "processes on the GPUs"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1

sec "recent NVIDIA package changes"
( grep -ih nvidia /var/log/dpkg.log /var/log/dpkg.log.1 2>/dev/null | grep -E ' (install|upgrade|remove) ' | tail -20 ) || true
( grep -iA4 -B1 nvidia /var/log/apt/history.log 2>/dev/null | tail -30 ) || true
( rpm -qa --last 2>/dev/null | grep -i nvidia | head -20 ) || true
( grep -i nvidia /var/log/dnf.rpm.log /var/log/yum.log 2>/dev/null | tail -20 ) || true
sec "kernel messages (if readable)"
( dmesg -T 2>/dev/null | grep -iE 'NVRM|nvidia' | tail -15 ) || echo "dmesg not readable"
sec "done"; date -Iseconds
