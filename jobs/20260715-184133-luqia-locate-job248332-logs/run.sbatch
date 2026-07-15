#!/usr/bin/env bash
#SBATCH --job-name=locate-248332-logs
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

SEARCH_ROOT="/data/users/chjing"
TARGET_JOB_ID=248332

section() {
  echo
  echo "===== $1 ====="
}

section "locator_identity"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "locator_slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "search_root=${SEARCH_ROOT}"

section "matching_files"
mapfile -t FILES < <(
  find "$SEARCH_ROOT" -maxdepth 7 -type f \
    \( \
      -name "slurm-${TARGET_JOB_ID}.out" -o \
      -name "slurm-${TARGET_JOB_ID}.err" -o \
      -path '*/vllm_f990_results/vllm_server.log' -o \
      -path '*/vllm_f990_results/bench_iter*.txt' -o \
      -path '*/vllm_f990_results/nvidia_smi_monitor.csv' -o \
      -path '*/vllm_f990_results/jing_gpu_monitor.csv' \
    \) -print 2>/dev/null | sort -u
)

echo "matching_file_count=${#FILES[@]}"
printf '%s\n' "${FILES[@]}"

section "matching_directories"
find "$SEARCH_ROOT" -maxdepth 7 -type d -name 'vllm_f990_results' \
  -print -exec ls -lah {} \; 2>&1 || true

section "file_contents"
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no_matching_logs_found=true"
else
  for path in "${FILES[@]}"; do
    echo
    echo "----- ${path} -----"
    ls -lh "$path" 2>&1 || true
    case "$path" in
      *.csv)
        echo "--- first 8 lines ---"
        head -n 8 "$path" 2>&1 || true
        echo "--- last 30 lines ---"
        tail -n 30 "$path" 2>&1 || true
        ;;
      *)
        echo "--- last 400 lines ---"
        tail -n 400 "$path" 2>&1 || true
        ;;
    esac
  done
fi

section "recent_candidate_logs"
find "$SEARCH_ROOT" -maxdepth 5 -type f -mmin -120 \
  \( -name 'slurm-*.out' -o -name 'slurm-*.err' -o -name 'vllm_server.log' \) \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort | tail -n 100 || true

