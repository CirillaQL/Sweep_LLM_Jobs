#!/usr/bin/env bash
#SBATCH --job-name=gpufreq-src-inspect
#SBATCH --time=00:04:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

ROOTS=(
  "/data/users/chjing/slurm-nvgpufreq"
  "/data/users/chjing/build-nvgpufreq"
  "/data/users/chjing/vLLM_test"
  "/data/users/chjing/EnergyAwareCUDASTF"
)

TERMS='nvgpufreq|gpu[-_ ]?freq|GPU_FREQ|GpuFreq|frequency|clock|nvidia-smi|nvml|SetApplicationsClocks|ResetApplicationsClocks|cuda|power|prolog|epilog'

section() {
  echo
  echo "===== $1 ====="
}

run_shell() {
  local title="$1"
  local command="$2"
  section "$title"
  echo "command=${command}"
  bash --noprofile --norc -lc "$command"
  local rc=$?
  echo "===== ${title} rc=${rc} ====="
}

print_file_head() {
  local file="$1"
  local lines="${2:-220}"
  section "file_head ${file}"
  if [ -r "$file" ] && [ -f "$file" ]; then
    sed -n "1,${lines}p" "$file"
  else
    echo "not_readable_or_not_file"
  fi
}

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "USER=${USER:-unset}"
echo "LOGNAME=${LOGNAME:-unset}"
echo "HOME=${HOME:-unset}"

section "slurm_job_record"
scontrol show job "${SLURM_JOB_ID:-0}" || true

for root in "${ROOTS[@]}"; do
  section "root_check ${root}"
  if [ -e "$root" ]; then
    stat "$root" || true
    ls -ld "$root" || true
  else
    echo "missing"
  fi
done

for root in "${ROOTS[@]}"; do
  if [ -d "$root" ]; then
    run_shell "tree_like_listing ${root}" \
      "find '$root' -maxdepth 3 \\( -path '*/.git' -o -path '*/__pycache__' -o -path '*/.cache' \\) -prune -o -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM %p\n' | sort | head -n 260"
  fi
done

for root in "${ROOTS[@]}"; do
  if [ -d "$root" ]; then
    run_shell "keyword_hits ${root}" \
      "grep -RInE '$TERMS' '$root' --include='*.c' --include='*.h' --include='*.cc' --include='*.cpp' --include='*.cu' --include='*.py' --include='*.sh' --include='*.md' --include='*.txt' --include='*.conf' --include='*.service' --include='*.json' --include='CMakeLists.txt' 2>/dev/null | head -n 220"
  fi
done

run_shell "interesting_file_candidates" \
  "for root in ${ROOTS[*]}; do [ -d \"\$root\" ] && find \"\$root\" -maxdepth 4 -type f \\( -iname '*gpu*' -o -iname '*freq*' -o -iname '*clock*' -o -iname '*nvml*' -o -iname '*slurm*' -o -iname '*prolog*' -o -iname '*epilog*' -o -iname '*.conf' -o -iname '*.service' \\) -printf '%M %u %g %s %p\n'; done | sort | head -n 160"

section "selected_file_heads"
for file in \
  "/data/users/chjing/slurm-nvgpufreq/README.md" \
  "/data/users/chjing/slurm-nvgpufreq/README" \
  "/data/users/chjing/slurm-nvgpufreq/CMakeLists.txt" \
  "/data/users/chjing/slurm-nvgpufreq/nvgpufreq.c" \
  "/data/users/chjing/slurm-nvgpufreq/gpu_freq.c" \
  "/data/users/chjing/slurm-nvgpufreq/test_nvgpufreq.sh" \
  "/data/users/chjing/test_nvgpufreq.sh" \
  "/data/users/chjing/test_nvgpufreq_2.sh" \
  "/data/users/chjing/measure_dvfs_transition.sh" \
  "/data/users/chjing/measure_dvfs.cu" \
  "/data/users/chjing/vLLM_test/README.md" \
  "/data/users/chjing/EnergyAwareCUDASTF/README.md"
do
  if [ -e "$file" ]; then
    print_file_head "$file" 260
  fi
done

section "grep_context_in_slurm_nvgpufreq"
if [ -d "/data/users/chjing/slurm-nvgpufreq" ]; then
  grep -RInE "$TERMS" /data/users/chjing/slurm-nvgpufreq \
    --include='*.c' --include='*.h' --include='*.sh' --include='*.md' --include='*.txt' --include='CMakeLists.txt' 2>/dev/null \
    | head -n 260 || true
fi

section "note"
echo "This job only inspects filenames and selected text/source files in known chjing directories."
